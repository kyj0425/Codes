import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import pandas as pd

# 공통 파일명
npy_file_name = 'subject00_simul00_mode00_RGB.npy'
csv_base_dir = r'D:\Visual Studio Python\sample_code\rppg_data\0-0'
cppg_csv_path = r'D:\Visual Studio Python\sample_code\sample_path\cppg_data\subject00_simul00_mode00_cppg.csv'
csv_file_name = 'rppg_after.csv'

# 디렉토리 설정
npy_base_dirs = [
    #r'D:\Visual Studio Python\ArcheryGame\rPPG_output\GREEN',
    #r'D:\Visual Studio Python\ArcheryGame\rPPG_output\LGI',
    #r'D:\Visual Studio Python\ArcheryGame\rPPG_output\OMIT',
    # r'D:\Visual Studio Python\ArcheryGame\rPPG_output\PBV'
    # r'D:\Visual Studio Python\ArcheryGame\rPPG_output\POS',
    # r'D:\Visual Studio Python\base_code\rPPG_extract\output\CHROM',
    # r'D:\Visual Studio Python\ArcheryGame\rPPG_output\ICA'
    # r"D:\Visual Studio Python\base_code\rPPG_extract\output\PBV"
    # r"D:\Visual Studio Python\base_code_1\rPPG_extract\output\PBV"
    r"D:\Visual Studio Python\summer_study_2025\rPPG_output\PBV"
]

# 라벨
labels = [os.path.basename(d) for d in npy_base_dirs]

plt.figure(figsize=(12, 6))

# 1. rPPG CSV 로딩
rppg_csv_path = os.path.join(csv_base_dir, csv_file_name)
rppg_df = pd.read_csv(rppg_csv_path)
rppg_time = rppg_df['Time'].values
rppg_signal = rppg_df.iloc[:, 3].values  # 3번째 컬럼이 rPPG 값이라고 가정

# 2. CPPG CSV 로딩
cppg_df = pd.read_csv(cppg_csv_path)
cppg_time = cppg_df['PPG Signal'].values  # 시간
cppg_signal = cppg_df.iloc[:, 1].values   # 값

# 3. CPPG 시작 시간 이후의 rPPG만 선택
cppg_start_time = cppg_time[0]
valid_indices = np.where(rppg_time >= cppg_start_time)[0]
clipped_rppg_time = rppg_time[valid_indices]
clipped_rppg_signal = rppg_signal[valid_indices]

# 4. npy 파일 로딩 (rPPG 신호와 길이 맞추기, 클리핑된 길이만큼)
npy_signals = {}
for base_dir, label in zip(npy_base_dirs, labels):
    file_path = os.path.join(base_dir, npy_file_name)
    try:
        data = np.load(file_path)
        print(data.shape)  # 예: (1047,)
        if data.ndim == 1:
            norm_data = zscore(data)
            if len(norm_data) < len(rppg_signal):  # rppg_signal의 길이만큼 맞추기
                norm_data = np.pad(norm_data, (0, len(rppg_signal) - len(norm_data)), 'constant')  # 뒤에 0으로 패딩
            # rppg와 동일한 길이로 자르기 (혹은 슬라이싱)
            norm_data = norm_data[:len(rppg_signal)]  # rppg와 길이 맞추기
            clipped_data = norm_data[valid_indices]  # rPPG와 같은 인덱스만 사용
            npy_signals[label] = clipped_data
        else:
            print(f"[경고] {label}의 데이터 shape이 1D가 아님: {data.shape}")
    except Exception as e:
        print(f"[오류] {label}에서 파일 로드 실패: {e}")

# # 4. npy 파일 로딩 (rPPG 신호와 길이 맞추기, 클리핑된 길이만큼)
# npy_signals = {}
# for base_dir, label in zip(npy_base_dirs, labels):
#     file_path = os.path.join(base_dir, npy_file_name)
#     try:
#         data = np.load(file_path)
#         if data.ndim == 1:
#             norm_data = zscore(data)
#             clipped_data = norm_data[valid_indices]  # rPPG와 같은 인덱스만 사용
#             npy_signals[label] = clipped_data
#         else:
#             print(f"[경고] {label}의 데이터 shape이 1D가 아님: {data.shape}")
#     except Exception as e:
#         print(f"[오류] {label}에서 파일 로드 실패: {e}")

# 5. 그래프 시각화
# CPPG (전체 그대로)
plt.plot(cppg_time, zscore(cppg_signal), label="CPPG Signal", linestyle='--', color='red')

# # rPPG (CPPG 시작 이후만)
# plt.plot(clipped_rppg_time, zscore(clipped_rppg_signal), label="YCbCr Signal", color='black')

# npy 시각화 (CPPG 시작 이후만)
for label, signal in npy_signals.items():
    plt.plot(clipped_rppg_time, signal, label=f'{label} - Signal')

# 시각화 설정
plt.title("rPPG vs CPPG Signals (Start-Aligned Only)")
plt.xlabel("Linux Timestamp")
plt.ylabel("Normalized Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
