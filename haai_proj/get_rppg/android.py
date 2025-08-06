# # import cv2
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from scipy import signal
# # import os
# # from scipy.stats import zscore

# # # ✅ CSV 파일 경로
# # csv_path = r"D:\Visual Studio Python\hi_proj\smu_data\android"
# # csv_file = r"subject09_measurement_log_1754292353440.csv"
# # save_csv_file = r"subject09_filtered.csv"

# # # ✅ 정규화된 FFT 기반 PSD 계산 함수 (Hanning window + 보정 포함)
# # def compute_psd_fft(signal, fs):
# #     N = signal.shape[0]
# #     window = np.hanning(N)
# #     sig_win = signal * window
# #     U = np.sum(window ** 2) / N
# #     X = np.fft.rfft(sig_win)
# #     freqs = np.fft.rfftfreq(N, d=1 / fs)
# #     psd = (1 / (fs * N)) * (np.abs(X) ** 2) / U
# #     return freqs, psd

# # # ✅ 출력 폴더 설정
# # output_dir_1 = r".\rppg_output\raw_rppg"
# # output_dir_2 = r".\rppg_output\detrended_rppg"
# # output_dir_3 = r".\rppg_output\filtered_rppg"
# # os.makedirs(output_dir_1, exist_ok=True)
# # os.makedirs(output_dir_2, exist_ok=True)
# # os.makedirs(output_dir_3, exist_ok=True)

# # # ✅ CSV 불러오기 및 rPPG 신호 생성
# # file_path = os.path.join(csv_path, csv_file)
# # df = pd.read_csv(file_path)
# # df = df[df["nPixels"] > 0].copy()
# # df["rppg_signal"] = (df["cb"] + df["cr"]) / df["nPixels"]
# # raw_ppg_signal = df["rppg_signal"].to_numpy()
# # detrended_ppg_signal = df["rppg_signal"].to_numpy()
# # filtered_ppg_signal = df["rppg_signal"].to_numpy()
# # arr1 = []

# # # ✅ 파라미터
# # fps = 30
# # window_size = fps * 5      # 5초 = 150프레임
# # step_size = fps            # 1초 슬라이딩 = 30프레임
# # lowcut = 42 / 60           # 0.7Hz 상황에 맞춰서 범위 변화
# # highcut = 180 / 60         # 3.0Hz
# # nyq = 0.5 * fps
# # b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype='band')

# # # # ✅ 결과 저장 리스트
# # # bpm_records = []

# # # # ✅ 슬라이딩 윈도우 분석_ raw_data만 저장
# # # for start in range(0, len(raw_ppg_signal) - window_size + 1, step_size):
# # #     end = start + window_size
# # #     segment = raw_ppg_signal[start:end]

# # #     #DC 성분 제거
# # #     filtered = segment - np.mean(segment)

# # #     # ✅ FFT 기반 PSD 계산
# # #     freq, psd = compute_psd_fft(filtered, fs=fps)

# # #     # 도미넌트 주파수 기반 BPM
# # #     i = np.argmax(psd)
# # #     dominant_freq = freq[i]
# # #     dominant_bpm = dominant_freq * 60

# # #     # 보간 기반 피크 주파수 계산
# # #     if 1 <= i <= len(psd) - 2:
# # #         alpha = psd[i - 1]
# # #         beta = psd[i]
# # #         gamma = psd[i + 1]
# # #         p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
# # #         interp_freq = freq[i] + p * (freq[1] - freq[0])
# # #     else:
# # #         interp_freq = dominant_freq

# # #     interpolated_bpm = interp_freq * 60

# # #     # 결과 저장
# # #     bpm_records.append({
# # #         "start_frame": start,
# # #         "end_frame": end,
# # #         "dominant_bpm": dominant_bpm,
# # #         "interpolated_bpm": interpolated_bpm
# # #     })

    




# # # # ✅ BPM CSV 저장
# # # bpm_df = pd.DataFrame(bpm_records)
# # # bpm_df.to_csv(os.path.join(output_dir_1, csv_file), index=False)

# # # print("✅ raw_data_도미넌트 및 보간 BPM 추정 + 시각화 완료.")

# # # # ✅ 슬라이딩 윈도우 분석_detrend까지 진행
# # # bpm_records = []

# # # for start in range(0, len(detrended_ppg_signal) - window_size + 1, step_size):
# # #     end = start + window_size
# # #     segment = detrended_ppg_signal[start:end]

# # #     # 디트렌드
# # #     kernel_size = round(fps)
# # #     norm = np.convolve(np.ones(len(segment)), np.ones(kernel_size), mode='same')
# # #     mean = np.convolve(segment, np.ones(kernel_size), mode='same') / norm
# # #     detrended = (segment - mean) / (mean + 1e-15)

# # #     #detrend만 진행
# # #     filtered = detrended

# # #     # ✅ FFT 기반 PSD 계산
# # #     freq, psd = compute_psd_fft(filtered, fs=fps)

# # #     # 도미넌트 주파수 기반 BPM
# # #     i = np.argmax(psd)
# # #     dominant_freq = freq[i]
# # #     dominant_bpm = dominant_freq * 60

# # #     # 보간 기반 피크 주파수 계산
# # #     if 1 <= i <= len(psd) - 2:
# # #         alpha = psd[i - 1]
# # #         beta = psd[i]
# # #         gamma = psd[i + 1]
# # #         p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
# # #         interp_freq = freq[i] + p * (freq[1] - freq[0])
# # #     else:
# # #         interp_freq = dominant_freq

# # #     interpolated_bpm = interp_freq * 60

# # #     # 결과 저장
# # #     bpm_records.append({
# # #         "start_frame": start,
# # #         "end_frame": end,
# # #         "dominant_bpm": dominant_bpm,
# # #         "interpolated_bpm": interpolated_bpm
# # #     })



# # # # ✅ BPM CSV 저장 
# # # bpm_df = pd.DataFrame(bpm_records)
# # # bpm_df.to_csv(os.path.join(output_dir_2, csv_file), index=False)

# # # print("✅ detrended_data_도미넌트 및 보간 BPM 추정 + 시각화 완료.")


# # # ✅ 슬라이딩 윈도우 분석_ 밴드패스 필터링 까지
# # bpm_records = []
# # for start in range(0, len(filtered_ppg_signal) - window_size + 1, step_size):
# #     end = start + window_size
# #     segment = filtered_ppg_signal[start:end]

# #     # 디트렌드
# #     kernel_size = round(fps)
# #     norm = np.convolve(np.ones(len(segment)), np.ones(kernel_size), mode='same')
# #     mean = np.convolve(segment, np.ones(kernel_size), mode='same') / norm
# #     detrended = (segment - mean) / (mean + 1e-15)

# #     # 필터링
# #     filtered = signal.filtfilt(b, a, detrended)

# #     # ✅ FFT 기반 PSD 계산
# #     freq, psd = compute_psd_fft(filtered, fs=fps)

# #     # 도미넌트 주파수 기반 BPM
# #     i = np.argmax(psd)
# #     dominant_freq = freq[i]
# #     dominant_bpm = dominant_freq * 60

# #     # 보간 기반 피크 주파수 계산
# #     if 1 <= i <= len(psd) - 2:
# #         alpha = psd[i - 1]
# #         beta = psd[i]
# #         gamma = psd[i + 1]
# #         p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
# #         interp_freq = freq[i] + p * (freq[1] - freq[0])
# #     else:
# #         interp_freq = dominant_freq

# #     interpolated_bpm = interp_freq * 60

# #     # 결과 저장
# #     bpm_records.append({
# #         "start_frame": start,
# #         "end_frame": end,
# #         "dominant_bpm": dominant_bpm,
# #         "interpolated_bpm": interpolated_bpm
# #     })

# # # ✅ BPM CSV 저장
# # bpm_df = pd.DataFrame(bpm_records)
# # bpm_df.to_csv(os.path.join(output_dir_3, csv_file), index=False)




# # # # 1. 전체 시퀀스에 대해 디트렌드

# # # kernel_size = round(fps)
# # # norm_full = np.convolve(np.ones(len(raw_ppg_signal)),
# # #                         np.ones(kernel_size),
# # #                         mode='same')
# # # mean_full = np.convolve(raw_ppg_signal,
# # #                         np.ones(kernel_size),
# # #                         mode='same') / norm_full
# # # detrended_ppg_signal = (raw_ppg_signal - mean_full) / (mean_full + 1e-15)

# # # # 2. 전체 시퀀스에 대해 필터링
# # # filtered_ppg_signal = signal.filtfilt(b, a, detrended_ppg_signal)



# # # # 1. 데이터프레임 구성
# # # filtered_df = pd.DataFrame({
# # #     "frame": np.arange(len(filtered_ppg_signal)),
# # #     "time": df["timestamp"],
# # #     "ppg_signal": filtered_ppg_signal
# # # })

# # # # 2. CSV로 저장 (경로는 원하는 대로!)
# # # filtered_df.to_csv(os.path.join(output_dir_3, save_csv_file), index=False)


# # # # 3. z-score 정규화
# # # raw_z = zscore(raw_ppg_signal)
# # # det_z = zscore(detrended_ppg_signal)
# # # fil_z = zscore(filtered_ppg_signal)


# # # fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# # # # 1번째: Raw
# # # axs[0].plot(raw_z, label='Raw (Z-score)')
# # # axs[0].set_title('Raw rPPG (Z-score)')
# # # axs[0].set_ylabel('Amplitude')
# # # axs[0].legend()
# # # axs[0].grid(True)

# # # # 2번째: Detrended
# # # axs[1].plot(det_z, label='Detrended (Z-score)', color='orange')
# # # axs[1].set_title('Detrended rPPG (Z-score)')
# # # axs[1].set_ylabel('Amplitude')
# # # axs[1].legend()
# # # axs[1].grid(True)

# # # # 3번째: Filtered
# # # axs[2].plot(fil_z, label='Filtered (Z-score)', color='green')
# # # axs[2].set_title('Filtered rPPG (Z-score)')
# # # axs[2].set_xlabel('Frame Index')
# # # axs[2].set_ylabel('Amplitude')
# # # axs[2].legend()
# # # axs[2].grid(True)

# # # plt.tight_layout()
# # # plt.show()

# # import cv2
# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # from scipy import signal
# # import os
# # from scipy.stats import zscore
# # from scipy.signal import find_peaks

# # # ✅ CSV 파일 경로
# # rppg_path = r"D:\Visual Studio Python\hi_proj\smu_data\android"
# # rppg_file = r"subject00_measurement_log_1754291321149.csv"

# # cppg_path = r"D:\Visual Studio Python\hi_proj\smu_data\sensor"
# # cppg_file = r"subject00_cppg_30fps.csv"

# # cppg_data = pd.read_csv(os.path.join(cppg_path, cppg_file))
# # cppg_data = cppg_data[:150]

# # # ✅ 정규화된 FFT 기반 PSD 계산 함수 (Hanning window + 보정 포함)
# # def compute_psd_fft(signal, fs):
# #     N = signal.shape[0]
# #     window = np.hanning(N)
# #     sig_win = signal * window
# #     U = np.sum(window ** 2) / N
# #     X = np.fft.rfft(sig_win)
# #     freqs = np.fft.rfftfreq(N, d=1 / fs)
# #     psd = (1 / (fs * N)) * (np.abs(X) ** 2) / U
# #     return freqs, psd

# # # ✅ 출력 폴더 설정
# # output_dir = r".\rppg_output"
# # os.makedirs(output_dir, exist_ok=True)

# # # ✅ CSV 불러오기 및 rPPG 신호 생성
# # df = pd.read_csv(os.path.join(rppg_path, rppg_file))
# # df = df[df["nPixels"] > 0].copy()
# # df["rppg_signal"] = (df["cb"] + df["cr"]) / df["nPixels"]
# # ppg_signal = df["rppg_signal"].to_numpy()

# # # ✅ 파라미터
# # fps = 30
# # window_size = fps * 5      # 5초 = 150프레임
# # step_size = fps            # 1초 슬라이딩 = 30프레임
# # lowcut = 42 / 60           # 0.7Hz
# # highcut = 180 / 60         # 3.0Hz
# # nyq = 0.5 * fps
# # b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype='band')

# # # ✅ 결과 저장 리스트
# # bpm_records = []

# # # ✅ 슬라이딩 윈도우 분석
# # for start in range(0, len(ppg_signal) - window_size + 1, step_size):
# #     end = start + window_size
# #     segment = ppg_signal[start:end]
# #     cppg_segment = cppg_data[start:end]

# #     # 디트렌드
# #     kernel_size = round(fps)
# #     norm = np.convolve(np.ones(len(segment)), np.ones(kernel_size), mode='same')
# #     mean = np.convolve(segment, np.ones(kernel_size), mode='same') / norm
# #     detrended = (segment - mean) / (mean + 1e-15)

# #     # 필터링
# #     filtered = signal.filtfilt(b, a, detrended)


# #     # ✅ FFT 기반 PSD 계산
# #     freq, psd = compute_psd_fft(filtered, fs=fps)

# #     # 도미넌트 주파수 기반 BPM
# #     i = np.argmax(psd)
# #     dominant_freq = freq[i]
# #     dominant_bpm = dominant_freq * 60

# #     # 보간 기반 피크 주파수 계산
# #     if 1 <= i <= len(psd) - 2:
# #         alpha = psd[i - 1]
# #         beta = psd[i]
# #         gamma = psd[i + 1]
# #         p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
# #         interp_freq = freq[i] + p * (freq[1] - freq[0])
# #     else:
# #         interp_freq = dominant_freq

# #     interpolated_bpm = interp_freq * 60

# #     # # 결과 저장
# #     # bpm_records.append({
# #     #     "start_frame": start,
# #     #     "end_frame": end,
# #     #     "dominant_bpm": dominant_bpm,
# #     #     "interpolated_bpm": interpolated_bpm
# #     # })

# #     # ✅ 시각화
# #     fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# #     #axs[0].plot(filtered, label="rPPG (filtered)")
# #     axs[0].plot(cppg_segment, label = "cppg(segmented)")
# #     axs[0].set_title(f"Frames {start}-{end}\nDominant BPM: {dominant_bpm:.2f}, Interpolated BPM: {interpolated_bpm:.2f}")
# #     axs[0].set_xlabel("Frame Index")
# #     axs[0].set_ylabel("Amplitude")
# #     axs[0].legend()
# #     axs[0].text(5, np.max(filtered) * 0.9, f"Interp BPM: {interpolated_bpm:.2f}", fontsize=11, color='red')

# #     axs[1].plot(freq, psd, color='blue', label='cbcr filtered')
# #     axs[1].set_xlim(0, 5)
# #     axs[1].set_xlabel("Frequency (Hz)")
# #     axs[1].set_ylabel("Power (PSD)")
# #     axs[1].set_title("Frequency Spectrum (FFT-based PSD)")
# #     axs[1].axvline(0.7, color='red', linestyle='--')
# #     axs[1].axvline(3.0, color='red', linestyle='--')
# #     axs[1].text(0.7, max(psd) * 0.9, "0.7 Hz", color='red', rotation=90, va='bottom')
# #     axs[1].text(3.0, max(psd) * 0.9, "3.0 Hz", color='red', rotation=90, va='bottom')
# #     axs[1].axvline(dominant_freq, color='gray', linestyle='--', label=f"Dominant: {dominant_freq:.2f}Hz")
# #     axs[1].axvline(interp_freq, color='black', linestyle=':', label=f"Interp: {interp_freq:.2f}Hz")
# #     axs[1].legend()

# #     plt.tight_layout()
# #     save_path = os.path.join(output_dir, f"rppg_{start}_{end}_bpm{int(interpolated_bpm)}.png")
# #     plt.savefig(save_path)
# #     plt.show()
# #     plt.close()

# # # # ✅ BPM CSV 저장
# # # bpm_df = pd.DataFrame(bpm_records)
# # # bpm_df.to_csv(os.path.join(output_dir, "bpm_summary.csv"), index=False)

# # print("✅ 도미넌트 및 보간 BPM 추정 + 시각화 완료.")


# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks, resample
# from scipy.stats import zscore
# from scipy import signal

# # 파라미터
# fps = 30
# window_size = fps * 5      # 5초 = 150프레임
# step_size = window_size    # 150프레임씩 슬라이딩

# # cppg 로드 및 다운샘플링 (30Hz)
# cppg_path = r"D:\Visual Studio Python\hi_proj\smu_data\sensor"
# cppg_file = r"subject00_cppg_30fps.csv"
# cppg_data = pd.read_csv(os.path.join(cppg_path, cppg_file))
# cppg_signal = cppg_data['ppg'].to_numpy()

# # rppg 로드
# rppg_path = r"D:\Visual Studio Python\hi_proj\smu_data\android"
# rppg_file = r"subject00_measurement_log_1754291321149.csv"
# df = pd.read_csv(os.path.join(rppg_path, rppg_file))
# df = df[df["nPixels"] > 0].copy()
# df["rppg_signal"] = (df["cb"] + df["cr"]) / df["nPixels"]
# rppg_signal = df["rppg_signal"].to_numpy()


# # ✅ 정규화된 FFT 기반 PSD 계산 함수 (Hanning window + 보정 포함)
# def compute_psd_fft(signal, fs):
#     N = signal.shape[0]
#     window = np.hanning(N)
#     sig_win = signal * window
#     U = np.sum(window ** 2) / N
#     X = np.fft.rfft(sig_win)
#     freqs = np.fft.rfftfreq(N, d=1 / fs)
#     psd = (1 / (fs * N)) * (np.abs(X) ** 2) / U
#     return freqs, psd

# # ✅ 출력 폴더 설정
# output_dir_1 = r".\rppg_output\raw_rppg"
# output_dir_2 = r".\rppg_output\detrended_rppg"
# output_dir_3 = r".\rppg_output\filtered_rppg"
# os.makedirs(output_dir_1, exist_ok=True)
# os.makedirs(output_dir_2, exist_ok=True)
# os.makedirs(output_dir_3, exist_ok=True)


# # ✅ 파라미터
# fps = 30
# window_size = fps * 5      # 5초 = 150프레임
# step_size = fps            # 1초 슬라이딩 = 30프레임
# lowcut = 42 / 60           # 0.7Hz 상황에 맞춰서 범위 변화
# highcut = 180 / 60         # 3.0Hz
# nyq = 0.5 * fps
# b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype='band')


# for start in range(0, min(len(rppg_signal), len(cppg_signal)) - window_size + 1, window_size):
#     end = start + window_size
#     cppg_win = cppg_signal[start:end]
#     rppg_win = rppg_signal[start:end]

#     cppg_win = zscore(cppg_win)
#     rppg_win = zscore(rppg_win)

#     # 디트렌드
#     kernel_size = round(fps)
#     norm = np.convolve(np.ones(len(rppg_win)), np.ones(kernel_size), mode='same')
#     mean = np.convolve(rppg_win, np.ones(kernel_size), mode='same') / norm
#     detrended = (rppg_win - mean) / (mean + 1e-15)

#     # 필터링
#     filtered = signal.filtfilt(b, a, detrended)

#     rppg_peaks, _ = find_peaks(filtered)
#     cppg_peaks, _ = find_peaks(cppg_win)

#     # --- 첫 피크 정렬 (없으면 그대로) ---
#     def align(signal, peaks):
#         if len(peaks) == 0 or peaks[0] == 0: return signal
#         shift = peaks[0]
#         return np.roll(signal, -shift)[:window_size]
#     cppg_al = align(cppg_win, cppg_peaks)
#     rppg_al = align(rppg_win, rppg_peaks)
#     cppg_pk_al = cppg_peaks - cppg_peaks[0] if len(cppg_peaks) > 0 else cppg_peaks
#     rppg_pk_al = rppg_peaks - rppg_peaks[0] if len(rppg_peaks) > 0 else rppg_peaks

#     # --- FFT (PSD) ---
#     def compute_psd_fft(signal, fs):
#         N = signal.shape[0]
#         window = np.hanning(N)
#         sig_win = signal * window
#         U = np.sum(window ** 2) / N
#         X = np.fft.rfft(sig_win)
#         freqs = np.fft.rfftfreq(N, d=1 / fs)
#         psd = (1 / (fs * N)) * (np.abs(X) ** 2) / U
#         return freqs, psd
#     freq, psd = compute_psd_fft(rppg_al, fs=fps)
#     i = np.argmax(psd)
#     dominant_freq = freq[i]
#     dominant_bpm = dominant_freq * 60



#     # === 여기서부터 filtered 신호 사용 ===
#     # 피크 찾기도 filtered에서!
    
#     rppg_al = align(filtered, rppg_peaks)
#     rppg_pk_al = rppg_peaks - rppg_peaks[0] if len(rppg_peaks) > 0 else rppg_peaks

#     # cppg도 그대로
    
#     cppg_al = align(cppg_win, cppg_peaks)
#     cppg_pk_al = cppg_peaks - cppg_peaks[0] if len(cppg_peaks) > 0 else cppg_peaks

#     # FFT (PSD)는 filtered 기준으로
#     freq, psd = compute_psd_fft(rppg_al, fs=fps)
#     i = np.argmax(psd)
#     dominant_freq = freq[i]
#     dominant_bpm = dominant_freq * 60

#     # --- 시각화 ---
#     fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
#     axs[0].plot(cppg_al, label="CPPG (aligned)", linewidth=2)
#     axs[0].plot(rppg_al, label="rPPG (aligned, filtered)", linewidth=2, alpha=0.7)
#     axs[0].scatter(cppg_pk_al, cppg_al[cppg_pk_al], color='red', label='CPPG Peaks', zorder=5)
#     axs[0].scatter(rppg_pk_al, rppg_al[rppg_pk_al], color='green', label='rPPG Peaks', zorder=5)
#     axs[0].set_title(f"Frames {start}-{end}\nDominant BPM: {dominant_bpm:.2f}")
#     axs[0].set_xlabel("Frame Index (0 = first peak)")
#     axs[0].set_ylabel("Amplitude")
#     axs[0].legend()
#     axs[0].grid(True)

#     axs[1].plot(freq, psd, color='blue', label='rPPG PSD')
#     axs[1].set_xlim(0, 5)
#     axs[1].set_xlabel("Frequency (Hz)")
#     axs[1].set_ylabel("Power (PSD)")
#     axs[1].set_title("Frequency Spectrum (FFT-based PSD)")
#     axs[1].axvline(0.7, color='red', linestyle='--')
#     axs[1].axvline(3.0, color='red', linestyle='--')
#     axs[1].text(0.7, max(psd) * 0.9, "0.7 Hz", color='red', rotation=90, va='bottom')
#     axs[1].text(3.0, max(psd) * 0.9, "3.0 Hz", color='red', rotation=90, va='bottom')
#     axs[1].axvline(dominant_freq, color='gray', linestyle='--', label=f"Dominant: {dominant_freq:.2f}Hz")
#     axs[1].legend()
#     axs[1].grid(True)

#     plt.tight_layout()
#     plt.show()
#     plt.close()


# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.stats import zscore
# from scipy import signal

# # 파라미터
# fps = 30
# window_size = fps * 5      # 5초 = 150프레임
# step_size = window_size    # 150프레임씩 슬라이딩

# # cppg 로드 (z-score만)
# cppg_path = r"D:\Visual Studio Python\hi_proj\smu_data\sensor"
# cppg_file = r"subject00_cppg_30fps.csv"
# cppg_data = pd.read_csv(os.path.join(cppg_path, cppg_file))
# cppg_signal = cppg_data['ppg'].to_numpy()

# # rppg 로드
# rppg_path = r"D:\Visual Studio Python\hi_proj\smu_data\android"
# rppg_file = r"subject00_measurement_log_1754291321149.csv"
# df = pd.read_csv(os.path.join(rppg_path, rppg_file))
# df = df[df["nPixels"] > 0].copy()
# df["rppg_signal"] = (df["cb"] + df["cr"]) / df["nPixels"]
# rppg_signal = df["rppg_signal"].to_numpy()

# # bandpass 필터 파라미터
# lowcut = 42 / 60           # 0.7Hz
# highcut = 180 / 60         # 3.0Hz
# nyq = 0.5 * fps
# b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype='band')

# # 피크 탐지 조건
# peak_kwargs = dict(distance=20, prominence=0.1)

# for start in range(0, min(len(rppg_signal), len(cppg_signal)) - window_size + 1, window_size):
#     end = start + window_size

#     # cppg: z-score만
#     cppg_win = cppg_signal[start:end]
#     cppg_z = zscore(cppg_win)
#     cppg_peaks, _ = find_peaks(cppg_z, **peak_kwargs)

#     # rppg: z-score → detrend → bandpass
#     rppg_win = rppg_signal[start:end]
#     rppg_z = zscore(rppg_win)
#     kernel_size = round(fps)
#     norm = np.convolve(np.ones(len(rppg_z)), np.ones(kernel_size), mode='same')
#     mean = np.convolve(rppg_z, np.ones(kernel_size), mode='same') / norm
#     rppg_detrended = (rppg_z - mean) / (mean + 1e-15)
#     rppg_filtered = signal.filtfilt(b, a, rppg_detrended)
#     rppg_peaks, _ = find_peaks(rppg_filtered, **peak_kwargs)

#     # --- 시각화 ---
#     plt.figure(figsize=(12, 5))
#     plt.plot(cppg_z, label='CPPG (z-score only)', linewidth=2)
#     plt.plot(rppg_filtered, label='rPPG (detrend+bandpass)', linewidth=2)
#     plt.scatter(cppg_peaks, cppg_z[cppg_peaks], color='red', label='CPPG Peaks', zorder=5)
#     plt.scatter(rppg_peaks, rppg_filtered[rppg_peaks], color='green', label='rPPG Peaks', zorder=5)
#     plt.title(f'CPPG vs rPPG (Frames {start}-{end})')
#     plt.xlabel('Frame Index')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     plt.close()

#===============================================================================



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.stats import zscore

# # ✅ 파일 경로 설정
# cppg_fp = r"D:\Visual Studio Python\hi_proj\smu_data\sensor\subject00_cppg_30fps.csv"
# rppg_fp = r"D:\Visual Studio Python\hi_proj\rppg_output\filtered_rppg\subject00_filtered.csv"


# # ✅ 데이터 로드
# cppg_data = pd.read_csv(cppg_fp)['ppg'].to_numpy()
# rppg_data = pd.read_csv(rppg_fp)['ppg_signal'].to_numpy()

# # ✅ 파라미터 설정
# fps = 30
# window_size = fps * 5      # 5초 = 150프레임
# step_size = fps            # 1초 슬라이딩 = 30프레임

# # ✅ 전체에서 첫 번째 Peak 찾기
# cppg_peaks, _ = find_peaks(cppg_data, distance=fps//2)
# rppg_peaks, _ = find_peaks(rppg_data, distance=fps//2)

# if len(cppg_peaks) == 0 or len(rppg_peaks) == 0:
#     raise ValueError("Peak detection failed. No peaks found in cPPG or rPPG.")

# cppg_first_peak = cppg_peaks[0]
# rppg_first_peak = rppg_peaks[0]

# # ✅ 싱크 차이 (lag)
# lag = cppg_first_peak - rppg_first_peak

# # ✅ rPPG Shift (전체 데이터 기준)
# rppg_synced = np.roll(rppg_data, lag)

# # Shift로 인해 깨진 부분 Zero로 채움
# if lag > 0:
#     rppg_synced[:lag] = 0
# elif lag < 0:
#     rppg_synced[lag:] = 0

# print(f"싱크 정렬 완료. Lag: {lag} frames")

# # ✅ 앞 5초(150프레임) 제거 후 슬라이딩 윈도우
# cppg_data = cppg_data[window_size:]
# rppg_synced = rppg_synced[window_size:]

# num_windows = (len(cppg_data) - window_size) // step_size + 1

# for i in range(num_windows):
#     start = i * step_size
#     end = start + window_size

#     cppg_segment = cppg_data[start:end]
#     rppg_segment = rppg_synced[start:end]

#     cppg_segment = zscore(cppg_segment)
#     rppg_segment = zscore(rppg_segment)

#     # 각 윈도우 내 Peak Detection
#     cppg_window_peaks, _ = find_peaks(cppg_segment, distance=fps//2)
#     rppg_window_peaks, _ = find_peaks(rppg_segment, distance=fps//2)

#     # Plot
#     plt.figure(figsize=(12, 4))
#     plt.plot(np.arange(window_size), cppg_segment, label='cPPG (Sensor)')
#     plt.plot(np.arange(window_size), rppg_segment, label='rPPG (Synced)', linestyle='--')

#     # Peak 찍기
#     plt.scatter(cppg_window_peaks, cppg_segment[cppg_window_peaks], color='red', marker='o', label='cPPG Peaks')
#     plt.scatter(rppg_window_peaks, rppg_segment[rppg_window_peaks], color='green', marker='x', label='rPPG Peaks')

#     plt.title(f'Window {i+1}: Frames {start} - {end-1}')
#     plt.xlabel('Frame Index (Window-relative)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

#     # 👉 첫 윈도우만 볼 거면 break
#     # if i == 0:
#     #     break


#=================================================================
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# from scipy.stats import zscore
# import os

# # ✅ 파일 경로 설정
# cppg_fp = r"D:\Visual Studio Python\hi_proj\smu_data\sensor\subject00_cppg_30fps.csv"
# rppg_fp = r"D:\Visual Studio Python\hi_proj\rppg_output\filtered_rppg\subject00_filtered.csv"

# # ✅ 파일 이름 추출
# cppg_fname = os.path.basename(cppg_fp)
# rppg_fname = os.path.basename(rppg_fp)

# # ✅ 데이터 로드
# cppg_data = pd.read_csv(cppg_fp)['ppg'].to_numpy()
# rppg_data = pd.read_csv(rppg_fp)['ppg_signal'].to_numpy()

# # ✅ 파라미터 설정
# fps = 30
# window_size = fps * 5      # 5초 = 150프레임
# step_size = fps            # 1초 슬라이딩 = 30프레임

# # ✅ 전체에서 첫 번째 Peak 찾기 (싱크용)
# cppg_peaks, _ = find_peaks(cppg_data, distance=fps//2)
# rppg_peaks, _ = find_peaks(rppg_data, distance=fps//2)

# if len(cppg_peaks) == 0 or len(rppg_peaks) == 0:
#     raise ValueError("Peak detection failed. No peaks found in cPPG or rPPG.")

# cppg_first_peak = cppg_peaks[0]
# rppg_first_peak = rppg_peaks[0]

# # ✅ 싱크 차이 (lag)
# lag = cppg_first_peak - rppg_first_peak

# # ✅ rPPG Shift (전체 데이터 기준)
# rppg_synced = np.roll(rppg_data, lag)
# if lag > 0:
#     rppg_synced[:lag] = 0
# elif lag < 0:
#     rppg_synced[lag:] = 0

# print(f"싱크 정렬 완료. Lag: {lag} frames")

# # ✅ 앞 5초(150프레임) 제거 후 슬라이딩 윈도우
# cppg_data = cppg_data[window_size:]
# rppg_synced = rppg_synced[window_size:]

# # === 4. 전체 RR interval 정보 출력 함수 ===
# def print_rr_summary(data, label, fname, fps):
#     peaks, _ = find_peaks(data, distance=fps//2)
#     rr_intervals = np.diff(peaks)
#     rr_intervals_sec = rr_intervals / fps*100
#     mean_rr_sec = np.mean(rr_intervals_sec) if len(rr_intervals_sec) > 0 else 0
#     std_rr_sec = np.std(rr_intervals_sec) if len(rr_intervals_sec) > 0 else 0
#     print(f"\nFile: {fname} ({label})")
#     print(f"Count of RR interval: {len(rr_intervals_sec)}")
#     print(f"Mean RR interval (ms): {mean_rr_sec:.3f}")
#     print(f"Std RR interval (ms): {std_rr_sec:.3f}")

# print_rr_summary(cppg_data, "cPPG", cppg_fname, fps)
# print_rr_summary(rppg_synced, "rPPG", rppg_fname, fps)

# # === 히스토그램 (선택) ===
# # peaks, _ = find_peaks(cppg_data, distance=fps//2)
# # rr_intervals_sec = np.diff(peaks) / fps
# # plt.hist(rr_intervals_sec, bins=30, alpha=0.5, label='cPPG RR')
# # peaks, _ = find_peaks(rppg_synced, distance=fps//2)
# # rr_intervals_sec = np.diff(peaks) / fps
# # plt.hist(rr_intervals_sec, bins=30, alpha=0.5, label='rPPG RR')
# # plt.xlabel('RR interval (sec)')
# # plt.ylabel('Count')
# # plt.legend()
# # plt.title('RR interval histogram')
# # plt.show()

# # === 슬라이딩 윈도우 구간 Plot ===
# num_windows = (len(cppg_data) - window_size) // step_size + 1

# for i in range(num_windows):
#     start = i * step_size
#     end = start + window_size

#     cppg_segment = cppg_data[start:end]
#     rppg_segment = rppg_synced[start:end]

#     cppg_segment = zscore(cppg_segment)
#     rppg_segment = zscore(rppg_segment)

#     # 각 윈도우 내 Peak Detection
#     cppg_window_peaks, _ = find_peaks(cppg_segment, distance=fps//2)
#     rppg_window_peaks, _ = find_peaks(rppg_segment, distance=fps//2)

#     # Plot
#     plt.figure(figsize=(12, 4))
#     plt.plot(np.arange(window_size), cppg_segment, label='cPPG (Sensor)')
#     plt.plot(np.arange(window_size), rppg_segment, label='rPPG (Synced)', linestyle='--')
#     plt.scatter(cppg_window_peaks, cppg_segment[cppg_window_peaks], color='red', marker='o', label='cPPG Peaks')
#     plt.scatter(rppg_window_peaks, rppg_segment[rppg_window_peaks], color='green', marker='x', label='rPPG Peaks')
#     plt.title(f'Window {i+1}: Frames {start} - {end-1}')
#     plt.xlabel('Frame Index (Window-relative)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

#     # 첫 윈도우만 볼 거면 break
#     # if i == 0:
#     #     break

#==========================cppg, rppg's peak, rr interval count, mean(cppg_detrend_bandpassed)=================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch
from scipy.stats import zscore
import os
from scipy import signal

# ✅ 사용자 입력: subject 번호와 알고리즘명
subject_id = "subject09"
rppg_alg_name = "cgcr_rppg"  # 예: "cbcr_rppg", "cgcr_rppg" 등

# ✅ 파일 경로 설정
cppg_fp = rf"D:\Visual Studio Python\hi_proj\smu_data\sensor\{subject_id}_cppg_30fps.csv"
rppg_fp = rf"D:\Visual Studio Python\hi_proj\rppg_output\{rppg_alg_name}\filtered_rppg\{subject_id}_filtered.csv"
save_dir = rf"D:\Visual Studio Python\hi_proj\window_plots\{rppg_alg_name}_plot\{subject_id}_plot"

# ✅ 파라미터
fps = 30
window_size = fps * 30      # 5초 = 150프레임
step_size = fps * 1        # 1초 슬라이딩 = 30프레임
lowcut = 42 / 60           # 0.7Hz
highcut = 180 / 60         # 3.0Hz
nyq = 0.5 * fps
b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype='band')

# ✅ 데이터 로드
cppg_data = pd.read_csv(cppg_fp)['ppg'].to_numpy()
rppg_data = pd.read_csv(rppg_fp)['ppg_signal'].to_numpy()

# 1. 전체 cppg 시퀀스에 대해 detrend & bandpass filtering
kernel_size = round(30)
norm_full = np.convolve(np.ones(len(cppg_data)), np.ones(kernel_size), mode='same')
mean_full = np.convolve(cppg_data, np.ones(kernel_size), mode='same') / norm_full
detrended_cppg_signal = (cppg_data - mean_full) / (mean_full + 1e-15)
filtered_cppg_signal = signal.filtfilt(b, a, detrended_cppg_signal)

# ✅ 1. 앞뒤 5초씩 자르기
filtered_cppg_signal = filtered_cppg_signal[150:-150]
rppg_data = rppg_data[150:-150]

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
mean_rr_rppg, std_rr_rppg = rr_stats_from_signal(rppg_data, "rPPG")
total_rr_diff = abs(mean_rr_cppg - mean_rr_rppg)

# === 슬라이딩 윈도우 구간 분석 (윈도우별 싱크 적용) ===
num_windows = (len(filtered_cppg_signal) - window_size) // step_size + 1

for i in range(num_windows):
    start = i * step_size
    end = start + window_size

    cppg_segment = filtered_cppg_signal[start:end]
    rppg_segment = rppg_data[start:end]

    # z-score 정규화 (윈도우 기준)
    cppg_segment = zscore(cppg_segment)
    rppg_segment = zscore(rppg_segment)

    # === 윈도우 내 첫 피크 각각 찾기 ===
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
        rppg_aligned = rppg_segment.copy()  # 싱크 실패시 원본 사용

    # 동기화된 rPPG에서 피크 재탐지
    rppg_window_peaks_aligned, _ = find_peaks(rppg_aligned)
    rppg_window_peaks_aligned = rppg_window_peaks_aligned[rppg_aligned[rppg_window_peaks_aligned] >= 0.0]

    # === 윈도우별 RR interval 정보 계산 및 차이 ===
    def calc_window_rr(peaks):
        rr_intervals = np.diff(peaks)
        rr_intervals_ms = rr_intervals / fps * 1000  # ms
        mean_rr_ms = np.mean(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
        std_rr_ms = np.std(rr_intervals_ms) if len(rr_intervals_ms) > 0 else 0
        return mean_rr_ms, std_rr_ms

    mean_rr_cppg_win, std_rr_cppg_win = calc_window_rr(cppg_window_peaks)
    mean_rr_rppg_win, std_rr_rppg_win = calc_window_rr(rppg_window_peaks_aligned)
    rr_diff_win = abs(mean_rr_cppg_win - mean_rr_rppg_win)  # 윈도우별 차이

    # === PSD (Welch) 계산 ===
    freqs_cppg, psd_cppg = welch(cppg_segment, fs=fps, window='hann', nperseg=len(cppg_segment)//2)
    freqs_rppg, psd_rppg = welch(rppg_aligned, fs=fps, window='hann', nperseg=len(rppg_aligned)//2)

    # === Subplot: 위 - Time, 아래 - PSD ===
    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    # [0] Time Domain Plot
    axs[0].plot(cppg_segment, label='cPPG (Sensor)')
    axs[0].plot(rppg_aligned, label='rPPG (Aligned)', linestyle='--')
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

    # [1] Frequency Domain Plot (PSD)
    axs[1].plot(freqs_cppg, psd_cppg, label='cPPG PSD')
    axs[1].plot(freqs_rppg, psd_rppg, label='rPPG PSD', linestyle='--')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Power')
    axs[1].set_title('Power Spectral Density (Frequency Domain)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    # === 이미지 저장 ===
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(
    #     save_dir, f"{subject_id}_window_{i+1:03d}_frames_{start}_{end-1}_with_psd_win_sync.png"
    # )
    # plt.savefig(save_path)
    plt.show()
    plt.close(fig)  # 메모리 절약

# === 마지막에 전체 RR interval 차이 출력 ===
print("\n===== 전체 신호 RR interval summary =====")
print(f"{subject_id} - {rppg_alg_name}")
print(f"[전체 cPPG] 평균 RR: {mean_rr_cppg:.2f} ms / 표준편차: {std_rr_cppg:.2f} ms")
print(f"[전체 rPPG] 평균 RR: {mean_rr_rppg:.2f} ms / 표준편차: {std_rr_rppg:.2f} ms")
print(f"[전체 평균 RR interval 차이] |RR_diff| = {total_rr_diff:.2f} ms")



#=====================rppg 신호, detrend 및 filtering(저장은 따로 구현 필요)===============================
# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy import signal
# import os
# from scipy.stats import zscore

# # ✅ CSV 파일 경로
# csv_path = r"D:\Visual Studio Python\hi_proj\smu_data\android"
# csv_file = r"subject09_measurement_log_1754292353440.csv"
# save_csv_file = r"subject09_filtered.csv"

# # ✅ 정규화된 FFT 기반 PSD 계산 함수 (Hanning window + 보정 포함)
# def compute_psd_fft(signal, fs):
#     N = signal.shape[0]
#     window = np.hanning(N)
#     sig_win = signal * window
#     U = np.sum(window ** 2) / N
#     X = np.fft.rfft(sig_win)
#     freqs = np.fft.rfftfreq(N, d=1 / fs)
#     psd = (1 / (fs * N)) * (np.abs(X) ** 2) / U
#     return freqs, psd

# # ✅ 출력 폴더 설정
# output_dir_1 = r".\rppg_output\crgr_rppg\raw_rppg"
# output_dir_2 = r".\rppg_output\crgr_rppg\detrended_rppg"
# output_dir_3 = r".\rppg_output\crgr_rppg\filtered_rppg"
# os.makedirs(output_dir_1, exist_ok=True)
# os.makedirs(output_dir_2, exist_ok=True)
# os.makedirs(output_dir_3, exist_ok=True)

# # ✅ CSV 불러오기 및 rPPG 신호 생성
# file_path = os.path.join(csv_path, csv_file)
# df = pd.read_csv(file_path)
# df = df[df["nPixels"] > 0].copy()
# # df["rppg_signal"] = (df["cb"] + df["cr"]) / df["nPixels"]
# df["rppg_signal"] = (df["cr"]/df["nPixels"])/(df["cg"]/df["nPixels"])
# raw_ppg_signal = df["rppg_signal"].to_numpy()
# detrended_ppg_signal = df["rppg_signal"].to_numpy()
# filtered_ppg_signal = df["rppg_signal"].to_numpy()
# arr1 = []

# # ✅ 파라미터
# fps = 30
# window_size = fps * 5      # 5초 = 150프레임
# step_size = fps            # 1초 슬라이딩 = 30프레임
# lowcut = 42 / 60           # 0.7Hz 상황에 맞춰서 범위 변화
# highcut = 180 / 60         # 3.0Hz
# nyq = 0.5 * fps
# b, a = signal.butter(5, [lowcut / nyq, highcut / nyq], btype='band')

# # # ✅ 결과 저장 리스트
# # bpm_records = []


# # 1. 전체 시퀀스에 대해 디트렌드

# kernel_size = round(fps)
# norm_full = np.convolve(np.ones(len(raw_ppg_signal)),
#                         np.ones(kernel_size),
#                         mode='same')
# mean_full = np.convolve(raw_ppg_signal,
#                         np.ones(kernel_size),
#                         mode='same') / norm_full
# detrended_ppg_signal = (raw_ppg_signal - mean_full) / (mean_full + 1e-15)

# # 2. 전체 시퀀스에 대해 필터링
# filtered_ppg_signal = signal.filtfilt(b, a, detrended_ppg_signal)



# # 1. 데이터프레임 구성
# filtered_df = pd.DataFrame({
#     "frame": np.arange(len(filtered_ppg_signal)),
#     "time": df["timestamp"],
#     "ppg_signal": filtered_ppg_signal
# })

# # 2. CSV로 저장 (경로는 원하는 대로!)
# filtered_df.to_csv(os.path.join(output_dir_3, save_csv_file), index=False)


# # 3. z-score 정규화
# raw_z = zscore(raw_ppg_signal)
# det_z = zscore(detrended_ppg_signal)
# fil_z = zscore(filtered_ppg_signal)


# fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# # 1번째: Raw
# axs[0].plot(raw_z, label='Raw (Z-score)')
# axs[0].set_title('Raw rPPG (Z-score)')
# axs[0].set_ylabel('Amplitude')
# axs[0].legend()
# axs[0].grid(True)

# # 2번째: Detrended
# axs[1].plot(det_z, label='Detrended (Z-score)', color='orange')
# axs[1].set_title('Detrended rPPG (Z-score)')
# axs[1].set_ylabel('Amplitude')
# axs[1].legend()
# axs[1].grid(True)

# # 3번째: Filtered
# axs[2].plot(fil_z, label='Filtered (Z-score)', color='green')
# axs[2].set_title('Filtered rPPG (Z-score)')
# axs[2].set_xlabel('Frame Index')
# axs[2].set_ylabel('Amplitude')
# axs[2].legend()
# axs[2].grid(True)

# plt.tight_layout()
# plt.show()

#========================================================

#==================band pass filtering 코드(rppg의 2번쨰 bandpass filtering)=============
def selective_bandpass_dominant_freq(signal, fs, bandwidth_hz=0.05):
    """
    주어진 signal에서 가장 도미넌트한 주파수만 남기고 나머지 제거하는 bandpass
    bandwidth_hz: 중심 주파수를 기준으로 남길 허용 대역폭 (예: ±0.05Hz)
    """
    N = len(signal)
    freqs = np.fft.rfftfreq(N, d=1/fs)
    fft_vals = np.fft.rfft(signal)

    # Power spectrum
    power = np.abs(fft_vals)**2
    peak_idx = np.argmax(power)
    dom_freq = freqs[peak_idx]

    # 주파수 마스크: 도미넌트 주파수 ± bandwidth_hz만 허용
    mask = (freqs >= dom_freq - bandwidth_hz/2) & (freqs <= dom_freq + bandwidth_hz/2)
    fft_vals[~mask] = 0

    # 역 FFT → 시간영역으로 복원
    filtered_signal = np.fft.irfft(fft_vals, n=N)
    return filtered_signal, dom_freq