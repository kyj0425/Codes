import cv2
import mediapipe as mp
import re


class Detector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.landmark = None
        self.width = None
        self.height = None

    def process(self, frame):
        # Set frame size
        self.height, self.width = frame.shape[:2]

        # Convert image (bgr to rgb)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        # Detect landmark
        results = self.pose.process(rgb)
        landmark = str(results.pose_landmarks).split('landmark')[1:]
        self.landmark = [list(map(float, re.findall('[0-9][.][0-9]+', lms))) for lms in landmark]
        return len(self.landmark) > 0

    def get_face_rect(self):
        if len(self.landmark) > 0:
            cx, cy = self.landmark[0][0], self.landmark[0][1]
            br = self.landmark[1][0]
            bl = self.landmark[12][0]
            bu = self.landmark[10][1]
            bb = self.landmark[4][1]

            w = max(br, bl) - min(br, bl)
            h = (max(bu, bb) - min(bu, bb)) * 2

            sx = int((cx - w / 2) * self.width)
            sy = int((cy - h / 2) * self.height)
            ex = sx + int(w * self.width)
            ey = sy + int(h * self.height)

            return max(sx, 0), max(sy, 0), min(ex, self.width), min(ey, self.height)
        return None

    def get_nose_rect(self):
        if self.landmark is not None:
            cx, cy = self.landmark[0][0], self.landmark[0][1]
            br = self.landmark[2][0]
            bl = self.landmark[5][0]
            bu = self.landmark[1][1]
            bb = self.landmark[10][1]

            w = max(br, bl) - min(br, bl)
            h = (max(bu, bb) - min(bu, bb)) / 2

            sx = int((cx - w / 2) * self.width)
            sy = int((cy - h / 2) * self.height)
            ex = sx + int(w * self.width)
            ey = sy + int(h * self.height)

            return max(sx, 0), max(sy, 0), min(ex, self.width), min(ey, self.height)
        return 0, 0, 0, 0
    
    def get_eyes_rect(self):
        if self.landmark is not None:
            cx, cy = self.landmark[0][0], self.landmark[5][1]   # 코의 x 값과 눈의 y값 == 미간 위치 정도
            br = self.landmark[7][0] #== 왼쪽 귀 x값
            bl = self.landmark[8][0] #== 오른쪽 귀 x값
            bu = self.landmark[1][1] #== 눈 y 값
            bb = self.landmark[10][1] #== 입 y 값

            w = max(br, bl) - min(br, bl)
            h = (max(bu, bb) - min(bu, bb)) / 2

            sx = int((cx - w / 2) * self.width)
            sy = int((cy - h / 2) * self.height)
            ex = sx + int(w * self.width)
            ey = sy + int(h * self.height)

            return max(sx, 0), max(sy, 0), min(ex, self.width), min(ey, self.height)
        return 0, 0, 0, 0
