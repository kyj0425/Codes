import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from modules.YCbCr import rPPG
from detrend_rppg import Detrend

# simul: 0~2, mode: 0~2
for simul_id in range(3):
    for mode_id in range(3):
        f_name = f"subject00_simul{simul_id:02}_mode{mode_id:02}_RGB"
        detrend_instance = Detrend(f_name=f_name)
        print(f"Processing: {f_name}")
        detrend_instance.detrend_save()

# detrend = Detrend("subject00_simul00_qmode00_RGB")
# detrend.detrend_save()

# bpms = []

# i_dir = r"D:\Visual Studio Python\summer_study_2025\rPPG_output\CHROM"
# f_name = "subject00_simul00_mode00_RGB"

# i_path = os.path.join(i_dir, f_name)

# signal_data = np.load(i_path + '.npy')
# rppg = rPPG()
# bpms.append(rppg._get_bpm(signal_data, 30))
# print(bpms)

# bpm = rppg._get_bpm(signal_data, fps)
# bpms.append(bpm)
# if len(bpms) > WINDOW_SIZE:
#     del bpms[0]