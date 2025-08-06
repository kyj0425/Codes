import cv2
import numpy as np
from math import ceil

from modules.kcf import KCFTracker

class Detector:
    def __init__(self, model_path, prototxt_path):
        self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        self.tracker = KCFTracker()
        self.is_tracking = False
        self.tracking_fail_count = 0

        self.rect = None
        self.prev_bbox = [0, 0, 0, 0]

        # ultra-fast 얼굴 검출 모델용 변수
        self.priors = []
        self.iou_threshold = 0.3
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0], [64.0, 96.0], [128.0, 192.0, 256.0]]
        self.strides = [8.0, 16.0, 32.0, 64.0]

    def process(self, frame, detect_th=0.7):
        # Set frame size
        self.height, self.width = frame.shape[:2]
        if len(self.priors) == 0:
            self.priors = self._define_img_size((self.width // 2, self.height // 2))

        if self.is_tracking:
            is_valid, curr_bbox = self.tracker.update(frame)
            if not is_valid:
                self.tracking_fail_count += 1
                if self.tracking_fail_count >= 10:
                    self.is_tracking = False
                    self.tracking_fail_count = 0
        else:
            curr_bbox = self._detect_largest_face(frame, detect_th)
            if curr_bbox:
                curr_bbox = [curr_bbox[0], curr_bbox[1], curr_bbox[2] - curr_bbox[0], curr_bbox[3] - curr_bbox[1]]
                self.tracker.init(frame, curr_bbox)  # (x,y,w,h)
                self.is_tracking = True
                self.tracking_fail_count = 0

        # 얼굴 추적기 잔떨림 무시
        if self.is_tracking:
            curr_bbox = [curr if abs(curr - prev) > 1 else prev for curr, prev in zip(curr_bbox, self.prev_bbox)]
        else:
            curr_bbox = self.prev_bbox[:]
            self.tracking_fail_count += 1

        self.rect = [curr_bbox[0], curr_bbox[1], curr_bbox[0]+curr_bbox[2], curr_bbox[1]+curr_bbox[3]]
        return self.is_tracking

    def get_face_rect(self):
        return self.rect


    def _detect_largest_face(self, frame, detect_th):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (w//2, h//2)), 1 / 128.0, (w//2, h//2), 127, True)
        self.detector.setInput(blob)

        bboxes, scores = self.detector.forward(["boxes", "scores"])

        # 정제
        bboxes = np.expand_dims(np.reshape(bboxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        bboxes = self._convert_locations_to_boxes(bboxes, self.priors, self.center_variance, self.size_variance)
        bboxes = self._center_form_to_corner_form(bboxes)
        bboxes = self._predict(w, h, scores, bboxes, detect_th)
        bboxes = sorted(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)

        if len(bboxes) > 0:
            return bboxes[0].tolist()  # (xs, ys, xe, ye)
        else:
            return []

    def _convert_locations_to_boxes(self, locations, priors, center_variance,
                                   size_variance):
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)

    def _center_form_to_corner_form(self,locations):
        return np.concatenate([locations[..., :2] - locations[..., 2:] / 2,
                               locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)

    def _predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self._hard_nms(box_probs,
                                 iou_threshold=iou_threshold,
                                 top_k=top_k,
                                 )
            picked_box_probs.append(box_probs)

        if not picked_box_probs:
            return np.array([])

        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32)

    def _hard_nms(self,box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self._iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]

    def _iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self._area_of(overlap_left_top, overlap_right_bottom)
        area0 = self._area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self._area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def _area_of(self,left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def _define_img_size(self, image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in self.strides]
            feature_map_w_h_list.append(feature_map)

        for i in range(0, len(image_size)):
            shrinkage_list.append(self.strides)
        priors = self._generate_priors(feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes)
        return priors

    def _generate_priors(self,feature_map_list, shrinkage_list, image_size, min_boxes):
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h

                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h
                        ])
        return np.clip(priors, 0.0, 1.0)