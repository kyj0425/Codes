import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

# 공통 npy 파일 이름
npy_file_name = 'subject00_simul00_mode00_RGB_pbv.npy'

# 두 개의 디렉토리 리스트
npy_base_dirs = [
    r"D:\Visual Studio Python\base_code\rPPG_extract\output\PBV",
    r"D:\Visual Studio Python\summer_study_2025\rPPG_output\PBV"
]

# 불러온 데이터들을 저장할 리스트
signals = []
labels = []

# 각 디렉토리에서 파일 불러오기
for dir_path in npy_base_dirs:
    file_path = os.path.join(dir_path, npy_file_name)
    if os.path.exists(file_path):
        signal = np.load(file_path)
        signal_z = zscore(signal)  # z-score 정규화
        signals.append(signal_z)   # 리스트에 추가해야 함!
        labels.append(os.path.basename(dir_path))  # 디렉토리 이름만 라벨로 사용
    else:
        print(f"[경고] 파일이 존재하지 않음: {file_path}")

# 시각화
plt.figure(figsize=(14, 5))
for signal, label in zip(signals, labels):
    plt.plot(signal, label=label)

plt.title(f"rPPG Signal Comparison: {npy_file_name}")
plt.xlabel("Time (frame index)")
plt.ylabel("Signal Amplitude (Z-score Normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

