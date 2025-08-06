import cv2
import numpy as np
from scipy import signal

class rPPG:
    TARGET_FPS = 30
    WINDOW_DURATION = 600
    WINDOW_SIZE = TARGET_FPS * WINDOW_DURATION

    BAND = (42, 180)

    def __init__(self):
        # Set buffers
        self.buffer = []
        self.ppg = [0] * self.WINDOW_SIZE
        self.bpms = []

        # pre-calculated skin division
        self.skin_low = np.array([0, 133, 77], np.uint8)
        self.skin_heigh = np.array([235, 173, 127], np.uint8)

        # 두개의 1차원 시퀀스의 이산 선형 컨볼루션, mode='same'은 length의 출력값을 반환한다.
        self.detrend_norm = [np.convolve(np.ones(self.WINDOW_SIZE), np.ones(i), mode='same') for i in range(1, 31)]
        self.band_passed=0
    def process(self, frame):
        # Crop face image
        # crop = frame[rect[1]: rect[3], rect[0]: rect[2], ...]
        crop = frame

        # Calculate ppg value
        val, light_val, cr_val, cb_val = self._get_raw_value(crop)
        self.buffer.append(val)

        # Adjust buffer length
        buffer_len = len(self.buffer)
        if buffer_len > self.WINDOW_SIZE:
            del self.buffer[0]
            buffer_len -= 1

        # Process
        if buffer_len <= self.WINDOW_SIZE:
            fps=30
            # List to Numpy array
            ppg_signal = np.array(self.buffer)

            # Get pulse
            pulse_signal = self._get_pulse_signal(ppg_signal, fps)
            self.ppg.append(pulse_signal[-1])
            if len(self.ppg) > self.WINDOW_SIZE:
                del self.ppg[0]

            # Get bpm
            bpm = self._get_bpm(pulse_signal, fps)
            self.bpms.append(bpm)
            if len(self.bpms) > self.WINDOW_SIZE:
                del self.bpms[0]

            return self.ppg, np.mean(self.bpms), light_val, cr_val, cb_val
        return self.ppg, 0, light_val, cr_val, cb_val

    def _get_raw_value(self, img):
        try:

            # Convert RGB to ycrcb
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            light, cr, cb = cv2.split(ycrcb)
            # Get skin mask
            mask = cv2.inRange(ycrcb, self.skin_low, self.skin_heigh)
            mask[mask == 255] = 1

            n_pixels = np.sum(mask)
            if n_pixels == 0:
                n_pixels = img.shape[0] * img.shape[1]
            else:
                cr[mask == 0] = 0
                cb[mask == 0] = 0
                light[mask == 0] = 0
            value = (np.sum(cr) + np.sum(cb)) / n_pixels
            light_val = np.sum(light) / n_pixels
            cr_val = np.sum(cr) / n_pixels
            cb_val = np.sum(cb) / n_pixels

            return value, light_val, cr_val, cb_val
        except:
            return self.buffer[-1] if len(self.buffer) > 0 else 0.0

    def _get_pulse_signal(self, arr, fps):
        raw_signal = arr.transpose()

        detrended = self._detrend_signal(raw_signal, fps)
        self.band_passed = self._bandpass_filtering(detrended, fps)

        return self.band_passed
    def get_signal(self):
        return self.band_passed
    def _detrend_signal(self, arr, fps):
        try:
            kernel_size = round(fps)
            mean = np.convolve(arr, np.ones(kernel_size), mode='same') / self.detrend_norm[kernel_size]
            return (arr - mean) / (mean + 1e-15)
        except:
            return arr

    def _bandpass_filtering(self, arr, fps):
        try:
            nyq = 60 * fps / 2
            coef_vector = signal.butter(5, [self.BAND[0] / nyq, self.BAND[1] / nyq], 'bandpass')
            return signal.filtfilt(*coef_vector, arr)
        except ValueError:
            return arr

    def _get_bpm(self, arr, fps):
        signal_len = arr.shape[0]
        fft = np.fft.rfft(arr, n=signal_len)
        freq = np.fft.rfftfreq(signal_len, d=1 / fps)
        frequency_spectrum = np.abs(fft)
        peak = np.argmax(frequency_spectrum)
        bpm = int(freq[peak] * 60)
        return bpm