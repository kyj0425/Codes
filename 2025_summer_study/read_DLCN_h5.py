# import h5py
# import numpy as np
# import cv2
# import time

# # DLCN 데이터셋의 .h5 파일 경로
# h5_file = r'D:\Download\DLCN\P1_4.h5'  # 실제 경로로 바꿔줘

# # .h5 파일 열기
# with h5py.File(h5_file, 'r') as f:
#     imgs = f['imgs'][:]   # 이미지 시퀀스 불러오기 (numpy 배열)
#     bvp = f['bvp'][:]     # 생체신호 (BVP) 불러오기 (numpy 배열)

# # 확인
# print("imgs shape:", imgs.shape)  # 예: (frames, height, width, channels)
# print("bvp shape:", bvp.shape)    # 예: (frames,)


# for i in range(0, len(imgs)):
#     frame = imgs[i].astype('uint8')
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV는 uint8 타입 선호
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(30) & 0xFF == ord('q'):  # 30ms 간격 (FPS 약 33)
#         break

# cv2.destroyAllWindows()


import os
import h5py
import numpy as np
import cv2

# ------------------------
# 1. 경로 설정
# ------------------------
h5_folder = r'D:\Download\DLCN'
output_base = r'D:\Visual Studio Python\Dataset\DLCN'

# ------------------------
# 2. 모든 h5 파일 처리
# ------------------------
for filename in os.listdir(h5_folder):
    if filename.endswith('.h5'):
        h5_path = os.path.join(h5_folder, filename)
        base_name = os.path.splitext(filename)[0]  # 파일명에서 확장자 제거 (예: P1_1)

        # 저장 경로 설정
        img_save_dir = os.path.join(output_base, base_name, 'frames')
        bvp_save_path = os.path.join(output_base, base_name, 'bvp.npy')

        # 저장 폴더 생성
        os.makedirs(img_save_dir, exist_ok=True)

        print(f"📂 처리 중: {filename}")

        # .h5 파일 열기
        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs'][:]   # (frames, H, W, 3)
            bvp = f['bvp'][:]     # (frames,)

        # 프레임 이미지 저장
        for i in range(len(imgs)):
            frame = imgs[i].astype('uint8')
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_filename = os.path.join(img_save_dir, f'frame_{i:04d}.png')
            cv2.imwrite(frame_filename, frame)
        print(f"  ✅ {len(imgs)}개 프레임 저장 완료")

        # BVP 저장
        np.save(bvp_save_path, bvp)
        print(f"  ✅ {len(bvp)}개 BVP 값 저장 완료: {bvp_save_path}\n")
