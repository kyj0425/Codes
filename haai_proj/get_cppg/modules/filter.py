import numpy as np
from scipy import signal


def detrend_signal(arr, wsize):
    try:
        if not isinstance(wsize, int):
            wsize = int(wsize)

        norm = np.convolve(np.ones(len(arr)), np.ones(wsize), mode='same')
        mean = np.convolve(arr, np.ones(wsize), mode='same') / norm

        return (arr - mean) / (mean + 1e-15)
    except ValueError:
        return arr


def filter_bandpass(arr, srate, band):
    try:
        nyq = 60 * srate / 2
        coef_vector = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
        return signal.filtfilt(*coef_vector, arr)
    except ValueError:
        return arr


def cppg_filter_bandpass(arr, srate, band):
    try:
        nyq = srate / 2
        coef_vector = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
        return signal.filtfilt(*coef_vector, arr)
    except ValueError:
        return arr


def eliminate_noisy_segment(arr, srate):
    len_segment = int(srate)
    n_segment = len(arr) // len_segment

    # segment별 표준편차 계산
    std_list = []
    for i in range(n_segment - 1):
        segment = arr[i * len_segment: (i + 1) * len_segment]
        std = np.std(segment)
        std_list.append(std)

    # 마지막 segment 표준편차 계산
    segment = arr[(n_segment - 1) * len_segment:]
    std = np.std(segment)
    std_list.append(std)

    # 표준편차가 가장 높은 segment 인덱스 검출
    max_std_segment = np.argmax(std_list)

    eliminated = np.concatenate((arr[:max_std_segment * len_segment], arr[(max_std_segment + 1) * len_segment:]))

    return eliminated


def fft(arr, srate, window_size):
    # Hamming 윈도우 적용
    windowed_arr = arr * np.hanning(len(arr))

    # 제로패딩 신호 생성
    n = len(windowed_arr)
    pad_factor = max(1.0, 60 * srate / window_size)
    n_padded = int(n * pad_factor)
    T = n_padded / srate

    # FFT 수행
    fft = np.fft.rfft(windowed_arr, n=n_padded)
    f = np.fft.rfftfreq(n_padded, d=1 / srate)

    return np.abs(fft), f, T


def get_weight_kernel(freq):
    mid_freq = 90 / 60  # 정상 심박수의 중앙값 (90)의 주파수
    mid_idx = np.fabs(freq - mid_freq).argmin()  # 정상 심박수 중앙값의 주파수 인덱스를 구함
    window = signal.gaussian(freq.shape[0], std=mid_idx / 3)  # 정상 심박 범위의 gaussian kernel 계산
    kernel = np.zeros((freq.shape[0]), np.float32)  # 주파수 범위의 빈 kernel 생성
    cropped = window[freq.shape[0] // 2 - mid_idx:]  # 사용 될 gaussian kernel crop
    kernel[: len(cropped)] = cropped  # frequency spectrum에 대한 가중치 커널 생성
    kernel[mid_idx:] = kernel[mid_idx]
    return kernel