import pandas as pd
import numpy as np
from scipy.signal import resample
import os

csv_path = r"D:\Visual Studio Python\hi_proj\smu_data\sensor"
csv_file = "subject09_simul00_mode00_cppg.csv"
input_fp = os.path.join(csv_path, csv_file)

df = pd.read_csv(input_fp)
ts_col = 'PPG Signal'          # 시간: 마이크로초(μs) 단위 리눅스 타임스탬프
ppg_col = 'Linux Timestamp'    # 신호값

# 원본 시간/신호 추출
time_orig = df[ts_col].values          # 마이크로초(μs) 단위
ppg_orig = df[ppg_col].values

fs_orig = 255
fs_new  = 30
num_new = int(np.floor(len(df) * fs_new / fs_orig))

# PPG 신호만 리샘플링
ppg_resampled = resample(ppg_orig, num_new)

# 시간축: 등간격으로 원본 시간축에서 보간
time_resampled = np.linspace(time_orig[0], time_orig[-1], num_new)

# 마이크로초(μs) → 초(s)로 변환!
time_resampled_sec = time_resampled / 1e6

out_df = pd.DataFrame({
    'time_s': time_resampled_sec,
    'ppg'   : ppg_resampled
})

output_fp = os.path.join(csv_path, "subject09_cppg_30fps.csv")
out_df.to_csv(output_fp, index=False)

print(f"✅ {fs_orig}Hz → {fs_new}Hz 리샘플링 완료, 저장: {output_fp}")
