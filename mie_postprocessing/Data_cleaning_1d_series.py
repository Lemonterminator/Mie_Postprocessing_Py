import numpy as np

def remove_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    return data[np.abs(z_scores) < threshold]

def remove_outliers_iqr(data, factor=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    return data[(data >= lower) & (data <= upper)]

def remove_outliers_mad(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return data[np.abs(modified_z) < threshold]


# Examples
'''
P, N = 5, 100
np.random.seed(0)
data = np.random.randn(P, N) * 10
data[0, [5, 20, 50]] = 100  # 人为加一些 outliers

cleaned = [remove_outliers_iqr(row) for row in data]
'''