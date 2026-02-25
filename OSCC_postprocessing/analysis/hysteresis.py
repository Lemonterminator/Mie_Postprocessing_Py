from OSCC_postprocessing.analysis.multihole_utils import resolve_backend 
from OSCC_postprocessing.analysis.single_plume import * 
use_gpu, triangle_backend, xp = resolve_backend(use_gpu="auto", triangle_backend="auto")

def longest_true_run(mask): # type:ignore
    """返回最长 True 连续段的 (start, end_exclusive)，若无 True 返回 None。"""
    """Retruan the longest True duration"""
    m = mask.astype(bool)
    if not m.any():
        return None
    diff = xp.diff(m.astype(xp.int8))
    starts = xp.where(diff == 1)[0] + 1
    ends   = xp.where(diff == -1)[0] + 1
    if m[0]:
        starts = xp.r_[0, starts]
    if m[-1]:
        ends = xp.r_[ends, len(m)]
    lengths = ends - starts
    k = xp.argmax(lengths)
    return int(starts[k]), int(ends[k])

def _get_cupy():
    """Return the CuPy module if available, otherwise ``None``."""
    try:
        import cupy as cp  # type: ignore

        return cp
    except Exception:
        return None
    
def _as_numpy(arr):
    if USING_CUPY and hasattr(arr, "__cuda_array_interface__"):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_py_scalar(x):
    # x 可能是 cupy scalar / cupy 0-d array / numpy scalar / python number
    if isinstance(x, cp.ndarray):
        return float(x.get().item())
    try:
        return float(x)
    except TypeError:
        # 兜底：比如 cupy scalar
        return float(cp.asarray(x).get().item())
    



def mad(x):
    med = xp.median(x)
    return xp.median(xp.abs(x - med)) + 1e-12

def hysteresis_threshold(y, th_lo, th_hi):
    """
    滞回阈值：>th_hi 触发进入 high；<th_lo 退出 high。
    
    Hysteresis threshold:  
        When the value > th_hi, triggers high state;
        when the value < th_lo, exits high state
        """
    high = y > th_hi
    low  = y < th_lo

    mask = xp.zeros_like(y, dtype=bool)
    state = False
    for i in range(len(y)):
        if not state:
            if high[i]:
                state = True
        else:
            if low[i]:
                state = False
        mask[i] = state
    return mask

def fill_short_false_runs(mask, max_len=3):
    """
    把 mask 中短的 False 段填成 True（填洞）。
    Filling short False pulses into True (hole-filling)
    """
    m = mask.copy()
    # 找 False runs
    diff = xp.diff(m.astype(xp.int8))
    starts = xp.where(diff == -1)[0] + 1  # True->False 后 False 段开始
    ends   = xp.where(diff ==  1)[0] + 1  # False->True 后 False 段结束
    if m[0] == False:
        starts = xp.r_[0, starts]
    if m[-1] == False:
        ends = xp.r_[ends, len(m)]
    for s, e in zip(starts, ends):
        if (e - s) <= max_len:
            m[s:e] = True
    return m

def remove_short_true_runs(mask, min_len=5):
    """把 mask 中短的 True 段删掉（去小岛）。"""
    """Filling up short True pulses with False (removing islands)"""
    m = mask.copy()
    diff = xp.diff(m.astype(xp.int8))
    starts = xp.where(diff == 1)[0] + 1
    ends   = xp.where(diff == -1)[0] + 1
    if m[0] == True:
        starts = xp.r_[0, starts]
    if m[-1] == True:
        ends = xp.r_[ends, len(m)]
    for s, e in zip(starts, ends):
        if (e - s) < min_len:
            m[s:e] = False
    return m

def longest_true_run(mask):
    """返回最长 True 连续段的 (start, end_exclusive)，若无 True 返回 None。"""
    """Retruan the longest True duration"""
    m = mask.astype(bool)
    if not m.any():
        return None
    diff = xp.diff(m.astype(xp.int8))
    starts = xp.where(diff == 1)[0] + 1
    ends   = xp.where(diff == -1)[0] + 1
    if m[0]:
        starts = xp.r_[0, starts]
    if m[-1]:
        ends = xp.r_[ends, len(m)]
    lengths = ends - starts
    k = xp.argmax(lengths)
    return int(starts[k]), int(ends[k])

def detect_single_high_interval(y, x=None,
                                base_quantile=0.10,
                                k_hi=0.9, 
                                k_lo=0.1,
                                th_lo=None,
                                th_hi=None,
                                fill_hole_len=3,
                                min_island_len=5):
    """
    y: 1D 信号
    x: 可选真实坐标；不传则用索引
    阈值：th_hi = base + k_hi * sigma, th_lo = base + k_lo * sigma
    sigma 用 MAD 估计（鲁棒）

    y: 1D signal,
    x: Can accecpt real coordinates, 
    """
    y = xp.asarray(y)
    if x is None:
        x = xp.arange(len(y))
    else:
        x = xp.asarray(x)


    # 1) 基线与尺度（鲁棒）
    base = xp.quantile(y, base_quantile)
    sigma = 1.4826 * mad(y - base)  # MAD->std 等效（对高斯噪声）
    if th_hi is None:
        th_hi = base + k_hi * sigma
    if th_lo is None:
        th_lo = base + k_lo * sigma

    # 2) 滞回二值化
    mask = hysteresis_threshold(y, th_lo=th_lo, th_hi=th_hi)

    # 3) 清理：填洞 + 去小岛
    mask = fill_short_false_runs(mask, max_len=fill_hole_len)
    mask = remove_short_true_runs(mask, min_len=min_island_len)

    # 4) 只保留最长段
    run = longest_true_run(mask)
    if run is None:
        return None, mask, (th_lo, th_hi)

    s, e = run
    # 输出坐标：起点 x[s]，终点 x[e-1]（最后一个 True 的点）
    return (x[s], x[e-1], s, e-1), mask, (th_lo, th_hi)
