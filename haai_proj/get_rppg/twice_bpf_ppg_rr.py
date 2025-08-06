import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import zscore
import os
from scipy import signal

# === 도미넌트 주파수대만 남기는 함수 ===
def selective_bandpass_dominant_freq(signal, fs, bandwidth_hz=0.05):
    """
    주어진 signal에서 dominant 주파수대(± bandwidth_hz/2)만 남김 (band-pass)
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(signal)
    power = np.abs(fft_vals) ** 2
    peak_idx = np.argmax(power)
    dom_freq = freqs[peak_idx]
    mask = (freqs >= dom_freq - bandwidth_hz/2) & (freqs <= dom_freq + bandwidth_hz/2)
    fft_vals[~mask] = 0
    filtered_signal = np.fft.irfft(fft_vals, n=N)
    return filtered_signal, dom_freq

# ✅ 사용자 입력
subject_id = "subject09"
rppg_alg_name = "cgcr_rppg"

# ✅ 파일 경로 설정
cppg_fp = rf"D:\Visual Studio Python\hi_proj\smu_data\sensor\{subject_id}_cppg_30fps.csv"
rppg_fp = rf"D:\Visual Studio Python\hi_proj\rppg_output\{rppg_alg_name}\filtered_rppg\{subject_id}_filtered.csv"
save_dir = rf"D:\Visual Studio Python\hi_proj\window_plots\{rppg_alg_name}_dominant_plot\{subject_id}_plot"

# ✅ 파라미터
fps = 30
window_size = fps * 30
step_size = fps * 1
lowcut = 42 / 60
highcut = 180 / 60
nyq = 0.5 * fps
b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype='band')

# ✅ 데이터 로드
cppg_data = pd.read_csv(cppg_fp)['ppg'].to_numpy()
rppg_data = pd.read_csv(rppg_fp)['ppg_signal'].to_numpy()

# === rPPG 신호에서 dominant만 남기기 (여기서 적용!) ===
rppg_data_domonly, dom_freq_total = selective_bandpass_dominant_freq(rppg_data, fs=fps, bandwidth_hz=0.05)

# === cPPG 처리(원본과 동일) ===
kernel_size = round(30)
norm_full = np.convolve(np.ones(len(cppg_data)), np.ones(kernel_size), mode='same')
mean_full = np.convolve(cppg_data, np.ones(kernel_size), mode='same') / norm_full
detrended_cppg_signal = (cppg_data - mean_full) / (mean_full + 1e-15)
filtered_cppg_signal = signal.filtfilt(b, a, detrended_cppg_signal)

# === 앞뒤 5초씩 자르기
filtered_cppg_signal = filtered_cppg_signal[150:-150]
rppg_data_domonly = rppg_data_domonly[150:-150]

# === 전체 신호의 RR interval(피크) 평균/표준편차 및 평균차이 ===
def rr_stats_from_signal(signal, label):
    peaks, _ = find_peaks(signal)
    peaks = peaks[signal[peaks] >= 0.0]
    rr_intervals = np.diff(peaks)
    rr_intervals_ms = rr_intervals / fps * 1000
    mean_rr = np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
    std_rr = np.std(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
    print(f"[전체 {label}] 피크 수: {len(peaks)} / RR interval 개수: {len(rr_intervals_ms)}")
    print(f"[전체 {label}] 평균 RR interval: {mean_rr:.2f} ms / 표준편차: {std_rr:.2f} ms")
    return mean_rr, std_rr

mean_rr_cppg, std_rr_cppg = rr_stats_from_signal(filtered_cppg_signal, "cPPG")
mean_rr_rppg, std_rr_rppg = rr_stats_from_signal(rppg_data_domonly, "rPPG (Dominant Only)")
total_rr_diff = abs(mean_rr_cppg - mean_rr_rppg)

# === 슬라이딩 윈도우 분석 ===
num_windows = (len(filtered_cppg_signal) - window_size) // step_size + 1

for i in range(num_windows):
    start = i * step_size
    end = start + window_size

    cppg_segment = filtered_cppg_signal[start:end]
    rppg_segment = rppg_data_domonly[start:end]

    # z-score 정규화 (윈도우 기준)
    cppg_segment = zscore(cppg_segment)
    rppg_segment = zscore(rppg_segment)

    # === 피크 ===
    cppg_window_peaks, _ = find_peaks(cppg_segment)
    rppg_window_peaks, _ = find_peaks(rppg_segment)
    cppg_window_peaks = cppg_window_peaks[cppg_segment[cppg_window_peaks] >= 0.0]
    rppg_window_peaks = rppg_window_peaks[rppg_segment[rppg_window_peaks] >= 0.0]

    # === 싱크(동기화): 각 윈도우별 첫 피크 기준 ===
    if len(cppg_window_peaks) > 0 and len(rppg_window_peaks) > 0:
        lag_win = cppg_window_peaks[0] - rppg_window_peaks[0]
        rppg_aligned = np.roll(rppg_segment, lag_win)
        if lag_win > 0:
            rppg_aligned[:lag_win] = 0
        elif lag_win < 0:
            rppg_aligned[lag_win:] = 0
    else:
        lag_win = None
        rppg_aligned = rppg_segment.copy()

    # === 동기화된 신호로 피크 재탐지 ===
    rppg_window_peaks_aligned, _ = find_peaks(rppg_aligned)
    rppg_window_peaks_aligned = rppg_window_peaks_aligned[rppg_aligned[rppg_window_peaks_aligned] >= 0.0]

    # === RR interval 정보 ===
    def calc_window_rr(peaks):
        rr_intervals = np.diff(peaks)
        rr_intervals_ms = rr_intervals / fps * 1000  # ms
        mean_rr_ms = np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
        std_rr_ms = np.std(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
        return mean_rr_ms, std_rr_ms

    mean_rr_cppg_win, std_rr_cppg_win = calc_window_rr(cppg_window_peaks)
    mean_rr_rppg_win, std_rr_rppg_win = calc_window_rr(rppg_window_peaks_aligned)
    rr_diff_win = abs(mean_rr_cppg_win - mean_rr_rppg_win)

    # === PSD (Welch) ===
    freqs_cppg, psd_cppg = welch(cppg_segment, fs=fps, window='hann', nperseg=len(cppg_segment)//2)
    freqs_rppg, psd_rppg = welch(rppg_aligned, fs=fps, window='hann', nperseg=len(rppg_aligned)//2)

    # === Subplot ===
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    axs[0].plot(cppg_segment, label='cPPG (Sensor)')
    axs[0].plot(rppg_aligned, label='rPPG (Dominant Only)', linestyle='--')
    axs[0].scatter(cppg_window_peaks, cppg_segment[cppg_window_peaks], color='red', marker='o', label='cPPG Peaks')
    axs[0].scatter(rppg_window_peaks_aligned, rppg_aligned[rppg_window_peaks_aligned], color='green', marker='x', label='rPPG Peaks')
    axs[0].set_title(
        f'{subject_id} - {rppg_alg_name} | Window {i+1}: Frames {start}-{end-1} | '
        f'Lag: {lag_win if lag_win is not None else "N/A"}\n'
        f'|RR diff|: {rr_diff_win:.1f} ms (cPPG: {mean_rr_cppg_win:.1f}, rPPG: {mean_rr_rppg_win:.1f})'
    )
    axs[0].set_xlabel('Frame Index (Window-relative)')
    axs[0].set_ylabel('Amplitude')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(freqs_cppg, psd_cppg, label='cPPG PSD')
    axs[1].plot(freqs_rppg, psd_rppg, label='rPPG PSD', linestyle='--')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Power')
    axs[1].set_title('Power Spectral Density (Frequency Domain)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir, f"{subject_id}_window_{i+1:03d}_frames_{start}_{end-1}_dominantonly.png"
    )
    plt.savefig(save_path)
    # plt.show()
    plt.close(fig)

# === 전체 RR interval summary ===
print("\n===== 전체 신호 RR interval summary =====")
print(f"{subject_id} - {rppg_alg_name}")
print(f"[전체 cPPG] 평균 RR: {mean_rr_cppg:.2f} ms / 표준편차: {std_rr_cppg:.2f} ms")
print(f"[전체 rPPG(Dominant Only)] 평균 RR: {mean_rr_rppg:.2f} ms / 표준편차: {std_rr_rppg:.2f} ms")
print(f"[전체 평균 RR interval 차이] |RR_diff| = {total_rr_diff:.2f} ms")
