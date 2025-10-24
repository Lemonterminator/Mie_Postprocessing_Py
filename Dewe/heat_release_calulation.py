from __future__ import annotations
import numpy as np

def hrr_calc(
    chmbP_bar,                    # 1D array-like pressure in bar
    Fs: float = 1e5,              # sampling rate [Hz]
    V_m3: float = 8.5e-3,         # chamber volume [m^3]
    gamma: float = 1.35,          # heat capacity ratio
    *,
    fc_p: float = 1000.0,         # pre-filter cutoff for P [Hz]
    order_p: int = 3,             # pre-filter order
    fc_hrr: float = 600.0,        # post-filter cutoff for HRR [Hz]
    order_hrr: int = 5,           # post-filter order
    time: np.ndarray | None = None,  # optional time vector [s]; if None uses Fs
    return_dataframe: bool = True     # if pandas is available, return a DataFrame
):
    """
    Compute heat-release rate (constant-volume model):
        HRR(t) = (V / (gamma-1)) * dP/dt
    and cumulative heat via trapezoidal integration.

    Inputs
    ------
    chmbP_bar : array-like, shape (N,)
        Chamber pressure in bar.
    Fs : float
        Sampling frequency [Hz]. Ignored if `time` is provided.
    V_m3 : float
        Chamber volume [m^3].
    gamma : float
        Heat capacity ratio.
    fc_p, order_p : float, int
        Butterworth low-pass for pressure before differentiation.
    fc_hrr, order_hrr : float, int
        Butterworth low-pass for HRR after differentiation.
    time : array-like or None
        Time vector [s]. If None, uniform sampling with dt=1/Fs is assumed.
    return_dataframe : bool
        If True and pandas is installed, returns a DataFrame with columns:
        ["time_s", "P_bar", "HRR_W", "Q_J"]. Otherwise returns dict of arrays.

    Returns
    -------
    pandas.DataFrame or dict
        time_s [s], P_bar [bar], HRR_W [W], Q_J [J].
    """
    # --- Imports that may not be present everywhere
    try:
        from scipy.signal import butter, filtfilt
        from scipy.integrate import cumulative_trapezoid
    except Exception as exc:
        raise RuntimeError("This function requires SciPy: pip install scipy") from exc

    P_bar = np.asarray(chmbP_bar, dtype=float).reshape(-1)
    if P_bar.ndim != 1:
        raise ValueError("chmbP_bar must be 1D.")

    N = P_bar.size
    if time is None:
        dt = 1.0 / float(Fs)
        t = np.arange(N, dtype=float) * dt
    else:
        t = np.asarray(time, dtype=float).reshape(-1)
        if t.size != N:
            raise ValueError("time and chmbP_bar must have the same length.")
        if not np.all(np.isfinite(np.diff(t))) or np.any(np.diff(t) <= 0):
            raise ValueError("time must be strictly increasing and finite.")
        # dt not constant; we will use t directly for gradient/integration
        dt = None

    # Units: bar -> Pa
    P = P_bar * 1e5  # [Pa]

    # --- Pre-filter pressure (zero-phase)
    if fc_p is not None and fc_p > 0:
        if time is not None:
            # Use local Nyquist from median dt
            fs_eff = 1.0 / np.median(np.diff(t))
        else:
            fs_eff = Fs
        wn_p = float(fc_p) / (fs_eff / 2.0)
        if not (0 < wn_p < 1):
            raise ValueError(f"Pressure cutoff fc_p={fc_p} invalid for Fs={fs_eff}")
        bP, aP = butter(order_p, wn_p, btype="low")
        P = filtfilt(bP, aP, P, axis=0)

    # --- Differentiate pressure
    if time is None:
        dPdt = np.gradient(P, 1.0 / Fs)  # Pa/s
    else:
        dPdt = np.gradient(P, t)         # Pa/s for nonuniform t

    # --- Heat-release rate (Watts)
    HRR = (V_m3 / (gamma - 1.0)) * dPdt  # [W] = J/s

    # --- Post-filter HRR (optional)
    if fc_hrr is not None and fc_hrr > 0:
        if time is not None:
            fs_eff = 1.0 / np.median(np.diff(t))
        else:
            fs_eff = Fs
        wn_q = float(fc_hrr) / (fs_eff / 2.0)
        if not (0 < wn_q < 1):
            raise ValueError(f"HRR cutoff fc_hrr={fc_hrr} invalid for Fs={fs_eff}")
        bQ, aQ = butter(order_hrr, wn_q, btype="low")
        HRR = filtfilt(bQ, aQ, HRR, axis=0)

    # --- Integrate HRR to cumulative heat [J]
    # cumulative_trapezoid returns length N-1; prepend 0 to align
    Q = cumulative_trapezoid(HRR, t, initial=0.0)

    result = {
        "time_s": t,
        "P_bar": P / 1e5,
        "HRR_W": HRR,
        "Q_J": Q,
    }

    if return_dataframe:
        try:
            import pandas as pd  # type: ignore
            return pd.DataFrame(result)
        except Exception:
            # Fall back to dict if pandas not available
            return result
    else:
        return result
