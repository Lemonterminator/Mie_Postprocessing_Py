import pandas as pd

import numpy as np

path = r"C:\Users\LJI008\Mie_Postprocessing_Py\BC20241010_HZ_Nozzle5\penetration_results\T1\condition_01_stats.npz"
testpoint = 1

# Test Point Data
from test_matrix.Nozzle5 import *

import numpy as np

def strictly_increasing_filter(seq):
    """
    Filters a sequence to be strictly monotonically increasing.
    Non-increasing values are replaced with np.nan.
    
    Parameters:
        seq (array-like): Input sequence (e.g., list or NumPy array).
    
    Returns:
        np.ndarray: Filtered sequence with np.nan for invalid values.
    """
    seq = np.array(seq, dtype=float)
    filtered = np.full_like(seq, np.nan)
    
    if len(seq) == 0:
        return filtered

    # Always keep the first valid value
    filtered[0] = seq[0]
    last_valid = seq[0]

    for i in range(1, len(seq)):
        if seq[i] > last_valid:
            filtered[i] = seq[i]
            last_valid = seq[i]
        # else: keep np.nan

    return filtered

data = np.load(path)

# Test point data
TP_data = data["condition_data_cleaned"]

TP_data = TP_data.reshape(-1, TP_data.shape[2])


penetration_series = TP_data

for i in penetration_series.shape[0]:
    penetration_series[i] = strictly_increasing_filter(penetration_series[i])

df = pd.DataFrame(penetration_series.T)

# Marking the right-censored data points (frames after the first right-censored frame set to 1)
is_right_censored = np.zeros((df.shape[0], 1))
is_right_censored[data["first_right_censored_idx"]:] = 1
df["is_right_censored"] = is_right_censored

# From the saved test matrix 
# df["FPS"]               = FPS

intv = 1.0 / FPS
time_s = np.arange(0, intv*df.shape[0], intv)
df["time_ms"]           = time_s*1000.0
df["tilt_angle_radian"] = np.deg2rad((180 - UMBRELLA_ANGLE) / 2.0)
df["plumes"]            = PLUMES
df["diameter_mm"]       = DIAMETER
df["chamber_pressure"]  = T_GROUP_TO_COND[testpoint]["chamber_pressure"]
df["injection_duration"] = T_GROUP_TO_COND[testpoint]["injection_duration"]
1
