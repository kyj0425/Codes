import cv2
import os
import numpy as np
import pickle
import onnxruntime
from face_detector import *
import glob
from tqdm import tqdm

def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    trans_dim, shape_dim, exp_dim = 12, 40, 10

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def TDDFA_detector(session, face_box, sx, sy, ex, ey):
    face_box = cv2.resize(face_box, dsize=(120, 120), interpolation=cv2.INTER_CUBIC)
    # Inference face detection using 3DDFA(by kunyoung)
    face_box = face_box.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
    face_box = (face_box - 127.5) / 128.0
    param = session.run(None, {'input': face_box})[0]
    param = param.flatten().astype(np.float32)
    param = param * param_std + param_mean  # re-scale
    head_vers = recon_vers([param], [[sx, sy, ex, ey]], u_base, w_shp_base, w_exp_base)[0]

    return head_vers.T[:, :2]


def recon_vers(param_lst, roi_box_lst, u_base, w_shp_base, w_exp_base):
    ver_lst = []
    for param, roi_box in zip(param_lst, roi_box_lst):
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        pts3d = R @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp). \
            reshape(3, -1, order='F') + offset
        pts3d = similar_transform(pts3d, roi_box, 120)

        ver_lst.append(pts3d)

    return ver_lst


def _skin_mask(face, PFLD_vers, face_detection):
    """
    face: cropped face image
    PFLD_vers: ndarray (N_landmarks, 2), cropped 기준 좌표
    """
    mask = np.zeros(face.shape[:2], dtype=np.uint8)

    # 왼쪽 볼 영역 polygon (1, 2, 3, 31)
    left_cheek_pts = np.array([
        PFLD_vers[1],
        PFLD_vers[2],
        PFLD_vers[3],
        PFLD_vers[31]
    ], dtype=np.int32)

    # 오른쪽 볼 영역 polygon (15, 14, 13, 35)
    right_cheek_pts = np.array([
        PFLD_vers[15],
        PFLD_vers[14],
        PFLD_vers[13],
        PFLD_vers[35]
    ], dtype=np.int32)

    # 각각 영역 채우기
    mask = cv2.fillPoly(mask, [left_cheek_pts], color=255)
    mask = cv2.fillPoly(mask, [right_cheek_pts], color=255)

    # 마스크 적용
    ret_face = face.copy()
    ret_face[mask == 0] = 0

    return ret_face


def process_frame(frame, frames_list):
    box = face_detector.detection(frame)
    if box[0] == -1:
        print("no box")
        return

    sx, sy, ex, ey = box
    face_crop = frame[sy:ey, sx:ex]

    PFLD_vers = TDDFA_detector(LD_session, face_crop, sx, sy, ex, ey)
    PFLD_vers_cropped = PFLD_vers.copy()
    PFLD_vers_cropped[:, 0] -= sx
    PFLD_vers_cropped[:, 1] -= sy

    # ✅ 볼 영역만 추출
    ret_face = _skin_mask(face_crop.copy(), PFLD_vers_cropped, face_detection=True)

    # ✅ 원본 frame에 볼 ROI 시각화
    left_cheek_pts = np.array([
        PFLD_vers[1],
        PFLD_vers[2],
        PFLD_vers[3],
        PFLD_vers[31]
    ], dtype=np.int32)

    right_cheek_pts = np.array([
        PFLD_vers[15],
        PFLD_vers[14],
        PFLD_vers[13],
        PFLD_vers[35]
    ], dtype=np.int32)

    # 좌표 타입 정수로 변환
    left_cheek_pts = left_cheek_pts.reshape((-1, 1, 2)).astype(np.int32)
    right_cheek_pts = right_cheek_pts.reshape((-1, 1, 2)).astype(np.int32)

    # 색칠
    frame = cv2.resize(frame, (640, 480))
    overlay = frame.copy()

    cv2.fillPoly(overlay, [left_cheek_pts], color=(255, 0, 0))   # 파랑
    cv2.fillPoly(overlay, [right_cheek_pts], color=(0, 255, 0))  # 초록

    # 투명도 적용 (선택 사항)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 결과 시각화
    cv2.imshow("ret_face", ret_face)
    cv2.imshow("face", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        raise KeyboardInterrupt

    frames_list.append(ret_face)




if __name__ == '__main__':
    bfm = _load('../model/bfm_slim.pkl')
    r = _load('../model/param_mean_std_62d_120x120.pkl')

    param_mean = r.get('mean')
    param_std = r.get('std')
    u = bfm.get('u').astype(np.float32)  # fix bug
    w_shp = bfm.get('w_shp').astype(np.float32)[..., :50]
    w_exp = bfm.get('w_exp').astype(np.float32)[..., :12]
    tri = bfm.get('tri')
    tri = _to_ctype(tri.T).astype(np.int32)
    keypoints = bfm.get('keypoints').astype(np.compat.long)  # fix bug
    w = np.concatenate((w_shp, w_exp), axis=1)
    w_norm = np.linalg.norm(w, axis=0)
    u_base = u[keypoints].reshape(-1, 1)
    w_shp_base = w_shp[keypoints]
    w_exp_base = w_exp[keypoints]

    # 모델 로드
    LD_session = onnxruntime.InferenceSession('../model/TDDFA.onnx', providers=['CPUExecutionProvider'])

    base_path = r"E:\rsp_dataset\android_dataset"
    if "sensor" in base_path:
        video_folders = [
            os.path.join(base_path, folder)
            for folder in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, folder)) and "RGB" in folder
        ]
    else:
        video_folders= [base_path]
    face_detector = FaceDetector()  # 클래스 인스턴스 생성
    for folder in tqdm(video_folders):
        frames = []

        # PNG 프레임 확인
        png_files = sorted(glob.glob(os.path.join(folder, '*.png')))
        mp4_files = sorted(glob.glob(os.path.join(folder, '*.mp4')))

        if len(png_files) > 0:
            # 🟢 PNG 이미지 처리
            for png_file in png_files:
                frame = cv2.imread(png_file)
                if frame is None:
                    continue
                # 🔽 아래는 공통 처리
                process_frame(frame, frames)

        elif len(mp4_files) > 0:
            for mp4_file in mp4_files:
                # 🔵 MP4 영상 처리 (첫 번째 mp4 파일만 처리)
                cap = cv2.VideoCapture(mp4_file)
                if not cap.isOpened():
                    print(f"{mp4_files[0]} - 비디오 열기 실패")
                    continue
                while True:
                    ret, frame = cap.read()
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    if not ret:
                        break
                    # 🔽 아래는 공통 처리
                    process_frame(frame, frames)
                cap.release()

        else:
            print(f"{folder} - PNG 또는 MP4 없음")
            continue

