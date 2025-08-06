import csv
import os
from collections import deque

import cv2,glob
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.cppg_manager import ContactPPG

class SignalVisualizer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.signal_pulse = np.zeros(width, dtype=np.float32)  # 신호 파형
        self.pulse_length = width
        self.COLOR_RPPG = (0, 255, 0)  # RPPG 신호 색상 (초록색)

    def draw_signal(self, frame, signal, color, position_y):
        """신호를 그래프에 그립니다."""
        x, y, w, h = 0, position_y, self.width, self.height  # 그래프를 그릴 영역 설정

        # 신호를 그래프 범위에 맞게 정규화
        if len(signal) > 0:
            signal_min = min(signal)
            signal_max = max(signal)
            normalized_signal = np.interp(signal, (signal_min, signal_max), (0, h))

            # 신호 그리기
            for i in range(1, len(normalized_signal)):
                p1 = (x + int((i - 1) * w / len(normalized_signal)), y + int(h - normalized_signal[i - 1]))
                p2 = (x + int(i * w / len(normalized_signal)), y + int(h - normalized_signal[i]))
                cv2.line(frame, p1, p2, color, 2)


def process_cppg(cppg_path, cppg_fps, viewfinder_width, viewfinder_height):
    """CPPG 신호를 기존 방식으로 처리하는 함수."""
    paths = os.listdir(cppg_path)
    sig_paths = sorted([path for path in paths if 'cppg' in path])

    for sig_path in tqdm(sig_paths, desc="Processing CPPG files"):

        cppg = ContactPPG()

        sig_file = os.path.join(cppg_path, sig_path)
        signal_df = pd.read_csv(sig_file)
        timestamps = signal_df['PPG Signal'].to_numpy()

        print(cppg_fps)
        if 'Linux Timestamp' in signal_df.columns:
            ppg_signals = signal_df['Linux Timestamp'].to_numpy()
        else:
            ppg_signals = np.zeros(len(sig_file))

        cppg_csv_path = os.path.join("./rPPG_extract/cppg_output", os.path.basename(sig_path))
        os.makedirs(os.path.dirname(cppg_csv_path), exist_ok=True)

        with open(cppg_csv_path, mode='w', newline='') as cppg_file:
            cppg_writer = csv.writer(cppg_file)
            cppg_writer.writerow(['Frame', 'CPPG HR (bpm)', "time"])

            for ni in range(len(ppg_signals)):
                cppg_ret, cppg_pulse = cppg.process(ppg_signals[ni], movements=0, is_update=True, fps=cppg_fps)
                cppg_hr, _ = cppg.get_hr(True, cppg_ret)

                # 1초에 한 번씩만 기록 (fps 간격으로)
                if ni % cppg_fps == 0:
                     cppg_writer.writerow([int(ni/cppg_fps), cppg_hr, timestamps[ni]])

if __name__ == '__main__':
    viewfinder_width = 640
    viewfinder_height = 480
    fps = 255

    process_cppg(r"C:\Users\thdgm\code\13_HAII\haii_signal_analysis\smu_data\sensor", fps, viewfinder_width, viewfinder_height)