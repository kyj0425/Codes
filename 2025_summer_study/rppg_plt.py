import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 각 데이터셋의 CSV 파일 경로와 라벨 설정
datasets = {
    'Dataset 0': [
        r'D:/Visual Studio Python/sample_code/rppg_data/0-0/rppg_before.csv',
        r'D:/Visual Studio Python/sample_code/rppg_data/0-1/rppg_after.csv',
        r'D:/Visual Studio Python/sample_code/rppg_data/0-2/rppg_after.csv',
        r'D:\Visual Studio Python\ArcheryGame\rPPG_output\POS\subject00_simul00_mode00_RGB_pos.npy']
    # ],
    # 'Dataset 1': [
    #     r'D:/Visual Studio Python/sample_code/rppg_data/1-0/rppg_before.csv',
    #     r'D:/Visual Studio Python/sample_code/rppg_data/1-1/rppg_after.csv',
    #     r'D:/Visual Studio Python/sample_code/rppg_data/1-2/rppg_after.csv'
    # ],
    # 'Dataset 2': [
    #     r'D:/Visual Studio Python/sample_code/rppg_data/2-0/rppg_before.csv',
    #     r'D:/Visual Studio Python/sample_code/rppg_data/2-1/rppg_after.csv',
    #     r'D:/Visual Studio Python/sample_code/rppg_data/2-2/rppg_after.csv'
    # ]
}

labels = ['Before (Light0)', 'After (Light1)', 'After (Light2)', 'test_npy']

# 서브플롯 설정
# 페이지 1: 3개의 서브플롯
fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

for ax, (dataset_name, file_list) in zip(axes, datasets.items()):
    for file, label in zip(file_list, labels):
        df = pd.read_csv(file)
        time = df['Frame']  # 또는 'Time' 열
        signal = df['rPPG Signal']
        ax.plot(time, signal, label=label)

    ax.set_title(f'rPPG Signal - {dataset_name}')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel('Frame Number')
plt.tight_layout()


# 페이지 1: 3개의 서브플롯
fig2, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)

for ax, (dataset_name, file_list) in zip(axes, datasets.items()):
    for file, label in zip(file_list, labels):
        df = pd.read_csv(file)
        time = df['Frame']  # 또는 'Time' 열
        signal = df['rPPG HR (bpm)']
        ax.plot(time, signal, label=label)

    ax.set_title(f'rPPG HR (bpm) - {dataset_name}')
    ax.set_ylabel('Amplitude')
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel('Frame Number')
plt.tight_layout()
plt.show()



# 1. Load CSV
csv_df = pd.read_csv("csv_data.csv", encoding='cp949')  # 또는 utf-8 등
csv_timestamps = csv_df["timestamp"]
csv_values = csv_df["value"]

# 2. Load NPY
npy_values = np.load("npy_data.npy")

# 3. 시간축 맞추기: npy도 timestamp가 있다면 같이 그리기
# 예: timestamp가 없다면 그냥 인덱스로 간단히 표현
npy_timestamps = np.arange(len(npy_values))

# 4. Plotting
plt.figure(figsize=(12, 5))
plt.plot(csv_timestamps, csv_values, label="CSV Signal", color='blue')
plt.plot(npy_timestamps, npy_values, label="NPY Signal", color='red', linestyle='--')

plt.xlabel("Time" if isinstance(csv_timestamps[0], (int, float)) else "Index")
plt.ylabel("Signal Value")
plt.title("Comparison: CSV vs NPY Signal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
