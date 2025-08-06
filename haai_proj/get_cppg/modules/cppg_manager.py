import time
from collections import deque

import cv2
import numpy as np
from scipy import signal

from modules.filter import *


class ContactPPG:
    MIN_FPS = 10
    TARGET_BAND = (42, 180)
    WINDOW_SIZE = 150

    STATE_SUCCESS = 0
    STATE_MORE_FRAME = 1
    STATE_LOW_FPS = 2

    def __init__(self):
        self.live_fps = 30

        self.buffer = []
        self.bandpassed = []
        self.pulse = []

        self.curr_hr = 70
        self.filtered_hr = 70
        self.confidence = 0.0

        self.bpms = []
        self.fail_count = 0

    def reset(self):
        self.buffer = []
        self.bandpassed = []
        self.pulse = []

        self.curr_hr = 0
        self.confidence = 0.0

        self.fail_count = 0

    def get_hr(self, ret_detection, ret_ppg):
        self.fail_count = (self.fail_count + 1) if ret_ppg != self.STATE_SUCCESS or not ret_detection else 0
        if ret_ppg != self.STATE_SUCCESS or self.fail_count >= self.live_fps:
            return -1, len(self.buffer)
        else:
            return self.filtered_hr, len(self.buffer)

    def get_raw_hr(self):
        return self.curr_hr

    def get_confidence(self):
        return self.confidence

    def get_signal(self):
        return self.bandpassed
    def process(self, val, movements, is_update,fps):
        self.live_fps=fps
        self.WINDOW_SIZE=self.live_fps*5

        self.buffer.append(val)
        self.buffer = self.buffer[-self.WINDOW_SIZE:]
        if len(self.buffer) < self.WINDOW_SIZE:
            return self.STATE_MORE_FRAME, []

        if self.live_fps < self.MIN_FPS:
            return self.STATE_LOW_FPS, []

        # Signal processing
        self._signal_process()

        # Update hr parameters
        if is_update:
            self._estimate_hr(self.bandpassed, self.live_fps)
            self._post_processing(movements)

        return self.STATE_SUCCESS, self.pulse

    import cv2

    def _signal_process(self):
        raw_signal = np.array(self.buffer).transpose()
        detrended = detrend_signal(raw_signal, self.live_fps)

        # 노이즈 구간 제거 추가
        eliminated = eliminate_noisy_segment(detrended, self.live_fps)

        # Bandpassed 신호 생성
        self.bandpassed = cppg_filter_bandpass(-eliminated, self.live_fps, self.TARGET_BAND)

        self.pulse.append(self.bandpassed[-1])
        self.pulse = self.pulse[-self.WINDOW_SIZE:]

        # 시각화용 그래프 그리기
        # self.visualize_signal(self.bandpassed)

    def visualize_signal(self, signal):
        """
        bandpassed 신호를 빈 화면에 시각화하는 함수
        """
        # 빈 화면 생성 (크기: 640x480, 검정색)
        graph_height, graph_width = 480, 640
        blank_image = np.zeros((graph_height, graph_width, 3), np.uint8)

        if len(signal) > 1:
            min_val, max_val = min(signal), max(signal)

            # 차이가 너무 작으면 스케일을 늘려서 더 눈에 띄게 함
            if max_val - min_val < 1e-5:
                min_val -= 0.01
                max_val += 0.01

            signal_normalized = np.interp(signal, (min_val, max_val), (0, graph_height - 1))

            for i in range(1, len(signal_normalized)):
                p1 = (int((i - 1) * graph_width / len(signal_normalized)), graph_height - int(signal_normalized[i - 1]))
                p2 = (int(i * graph_width / len(signal_normalized)), graph_height - int(signal_normalized[i]))
                cv2.line(blank_image, p1, p2, (0, 255, 0), 1)

        # 화면에 그래프 표시
        cv2.imshow("Bandpassed Signal", blank_image)
        cv2.waitKey(1)  # 잠깐의 지연을 주어 화면이 업데이트되도록 함

    def _estimate_hr(self, arr, srate):
        try:
            # 주파수 스펙트럼
            power_spectrum, freq, T = fft(arr, srate, self.WINDOW_SIZE)

            # HR 계산
            fundamental_peak = np.argmax(power_spectrum)
            self.curr_hr = int(freq[fundamental_peak] * 60)
        except Exception as e:
            print("[CPPG fft]:", e)
            pass  # Do nothing

    def _post_processing(self, movement):

        filtered = self.curr_hr
        # self.bpms.append(filtered)
        # filtered = round(np.mean(self.bpms))

        # if len(self.bpms) > 150:
        #     sort_bpm = sorted(self.bpms)
        #     filtered = round(np.mean(sort_bpm[30:-30]))
        #     del self.bpms[0]
        # elif len(self.bpms) > 30:
        #     sort_bpm = sorted(self.bpms)
        #     filtered = round(np.mean(sort_bpm[10:-10]))

        self.filtered_hr = filtered