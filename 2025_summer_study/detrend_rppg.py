import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from modules.YCbCr import rPPG

class Detrend:
    def __init__(self, f_name="subject00_simul02_mode00_RGB"):
        # 기본 파일 이름 설정
        self.f_name = f_name
        self.i_dirs_1 = [
            r"D:\Visual Studio Python\summer_study_2025\raw_rPPG_output\GREEN",
            r"D:\Visual Studio Python\summer_study_2025\raw_rPPG_output\LGI",
            r"D:\Visual Studio Python\summer_study_2025\raw_rPPG_output\OMIT",
            r"D:\Visual Studio Python\ArcheryGame\rPPG_output\PBV"
        ]
        self.i_dirs_2 = [
            r"D:\Visual Studio Python\summer_study_2025\raw_rPPG_output\CHROM",
            r"D:\Visual Studio Python\summer_study_2025\raw_rPPG_output\POS",
            r"D:\Visual Studio Python\summer_study_2025\raw_rPPG_output\ICA"
        ]
        self.o_base_dir = r"D:\Visual Studio Python\summer_study_2025\rPPG_output"

    def detrend_save(self):
        # i_dirs_1과 i_dirs_2에서 반복하여 파일을 처리
        for i_dir in self.i_dirs_1:
            i_path = os.path.join(i_dir, self.f_name)
            rppg = rPPG()

            if os.path.exists(i_path + ".npy"):
                signal_data = np.load(i_path + ".npy")
                if signal_data is None:
                    print(f"Failed to load data from {i_path}")
                    continue

                detrended = rppg._detrend_signal(signal_data, 30)
                filtered = rppg._bandpass_filtering(detrended, 30)
                normalized = zscore(filtered)

                subfolder_name = os.path.basename(i_dir)
                # save_dir = os.path.join(self.o_base_dir, subfolder_name)
                # os.makedirs(save_dir, exist_ok=True)

                # save_path = os.path.join(save_dir, self.f_name + '.npy')
                # np.save(save_path, filtered)

                plt.plot(normalized, label=subfolder_name)
            else:
                print(f"{i_path}.npy not found.")

        for i_dir in self.i_dirs_2:
            i_path = os.path.join(i_dir, self.f_name)
            rppg = rPPG()

            if os.path.exists(i_path + ".npy"):
                signal_data = np.load(i_path + ".npy")
                if signal_data is None:
                    print(f"Failed to load data from {i_path}")
                    continue
                
                normalized = zscore(signal_data)

                subfolder_name = os.path.basename(i_dir)
                # save_dir = os.path.join(self.o_base_dir, subfolder_name)
                # os.makedirs(save_dir, exist_ok=True)

                # save_path = os.path.join(save_dir, self.f_name + '.npy')
                # np.save(save_path, filtered)

                plt.plot(normalized, label=subfolder_name)
            else:
                print(f"{i_path}.npy not found.")

        # 그래프를 표시
        plt.title("Filtered rPPG Signals of\n" + self.f_name)
        plt.xlabel("Frames")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# # 인스턴스를 만들고 메서드를 호출하는 방식
# detrend_instance = Detrend()
# detrend_instance.detrend_save()
