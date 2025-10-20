import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

# Constants to be set

# Nozzle 1 
# from test_matrix.Nozzle1 import *
# path = r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241003_HZ_Nozzle1\penetration_results"

# Nozzle 2 
# from test_matrix.Nozzle2 import *
# path = r"Mie_Postprocessing_Py/BC20241017_HZ_Nozzle2/penetration_results"

# Nozzle 3
# from test_matrix.Nozzle3 import *
# path = r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241014_HZ_Nozzle3\penetration_results"

# Nozzle 4
# from test_matrix.Nozzle4 import *
# path = r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241007_HZ_Nozzle4\penetration_results"

# Nozzle 5
# from test_matrix.Nozzle5 import *
# path = r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241010_HZ_Nozzle5\penetration_results"

# Nozzle 6
# from test_matrix.Nozzle6 import *
# path = r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241011_HZ_Nozzle6\penetration_results"

# Nozzle 7
# from test_matrix.Nozzle7 import * 
# path = r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241015_HZ_Nozzle7\penetration_results"

# Nozzle 8
from test_matrix.Nozzle8 import *
path = r"C:\Users\Jiang\Documents\Mie_Py\Mie_Postprocessing_Py\BC20241016_HZ_Nozzle8\penetration_results"

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
    last_valid = 0

    for i in range(1, len(seq)):
        if not np.isnan(seq[i]):
            if seq[i] > last_valid:
                filtered[i] = seq[i]
                last_valid = seq[i]

        # else: keep np.nan

    return filtered


def testpoint_dataframe(data, testpoint):
    
    # Test point data
    TP_data = data["condition_data_cleaned"]

    TP_data = TP_data.reshape(-1, TP_data.shape[2])

    df_all = pd.DataFrame()

    penetration_series = TP_data

    for i in range(penetration_series.shape[0]):
        penetration_series[i] = strictly_increasing_filter(penetration_series[i])

        df = pd.DataFrame()
        df["penetration_pixels"] = penetration_series[i]

        # Marking the right-censored data points (frames after the first right-censored frame set to 1)
        is_right_censored = np.zeros((df.shape[0], 1))
        if ~np.isnan(data["first_right_censored_idx"]):
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

        df_clean = df.dropna()
        df_all = pd.concat([df_all, df_clean], axis=0, ignore_index=False)
    return df_all

def main():
    subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

    df = pd.DataFrame()
    for subfolder in subfolders:
        testpoint = int(subfolder.replace("T", ""))
        data_path = os.path.join(path, subfolder, "condition_01_stats.npz")
        
        data = np.load(data_path)
        df_tp  = testpoint_dataframe(data, testpoint)
        df = pd.concat([df, df_tp], axis=0, ignore_index=True)

    name = str(Path(path).parts[-2]) + "_penetration_data.csv"
    output_csv_path = os.path.join(path, name)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved DataFrame to {output_csv_path}")


if __name__ == "__main__":
    main()