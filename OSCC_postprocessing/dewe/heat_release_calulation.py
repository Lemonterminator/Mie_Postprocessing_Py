from __future__ import annotations

import numpy as np


def _effective_sample_rate(Fs: float, time: np.ndarray | None, t: np.ndarray) -> float:
    """Return the effective sample rate used for digital filter design."""
    if time is None:
        return float(Fs)
    return 1.0 / np.median(np.diff(t))


def hrr_calc(
    chmbP_bar,
    Fs: float = 1e5,
    V_m3: float = 8.5e-3,
    gamma: float = 1.35,
    *,
    fc_p: float = 1000.0,
    order_p: int = 3,
    fc_hrr: float = 600.0,
    order_hrr: int = 5,
    fir_taps_p: int = 101,
    fir_taps_hrr: int = 101,
    fir_window: str = "hamming",
    time: np.ndarray | None = None,
    return_dataframe: bool = True,
    filter: str | None = "butterworth",
):
    """
    Compute heat-release rate from chamber pressure using a constant-volume model.

    Parameters
    ----------
    chmbP_bar : array-like, shape (N,)
        Chamber pressure in bar.
    Fs : float
        Sampling frequency [Hz]. Ignored if ``time`` is provided.
    V_m3 : float
        Chamber volume [m^3].
    gamma : float
        Heat capacity ratio.
    fc_p, order_p : float, int
        Pressure low-pass settings before differentiation.
    fc_hrr, order_hrr : float, int
        HRR low-pass settings after differentiation.
    fir_taps_p, fir_taps_hrr : int
        FIR tap counts used when ``filter="fir"``.
    fir_window : str
        Window passed to ``scipy.signal.firwin`` when ``filter="fir"``.
    time : array-like or None
        Time vector [s]. If None, uniform sampling with dt = 1 / Fs is assumed.
    return_dataframe : bool
        If True and pandas is available, returns a DataFrame. Otherwise returns
        a dict of arrays.
    filter : {"butterworth", "fir", "none", None}
        Filter family used for both pressure and HRR smoothing. ``"butterworth"``
        preserves the historical behavior.

    Returns
    -------
    pandas.DataFrame or dict
        Keys/columns: ``time_s``, ``P_bar``, ``HRR_W``, ``Q_J``.
    """
    try:
        from scipy.integrate import cumulative_trapezoid
        from scipy.signal import butter, filtfilt, firwin
    except Exception as exc:
        raise RuntimeError("This function requires SciPy: pip install scipy") from exc

    P_bar = np.asarray(chmbP_bar, dtype=float).reshape(-1)
    if P_bar.ndim != 1:
        raise ValueError("chmbP_bar must be 1D.")

    N = P_bar.size
    if time is None:
        t = np.arange(N, dtype=float) / float(Fs)
    else:
        t = np.asarray(time, dtype=float).reshape(-1)
        if t.size != N:
            raise ValueError("time and chmbP_bar must have the same length.")
        if not np.all(np.isfinite(np.diff(t))) or np.any(np.diff(t) <= 0):
            raise ValueError("time must be strictly increasing and finite.")

    filter_name = "none" if filter is None else str(filter).strip().lower()

    def apply_lowpass(
        signal: np.ndarray,
        *,
        cutoff_hz: float | None,
        order: int,
        fir_taps: int,
        stage: str,
    ) -> np.ndarray:
        if cutoff_hz is None or cutoff_hz <= 0:
            return signal

        fs_eff = _effective_sample_rate(Fs, time, t)
        nyquist_hz = fs_eff / 2.0
        if not np.isfinite(fs_eff) or fs_eff <= 0:
            raise ValueError(f"{stage} effective sample rate is invalid: Fs={fs_eff}")

        cutoff = float(cutoff_hz)
        if cutoff >= nyquist_hz:
            # Clamp slightly below Nyquist so borderline user config does not
            # fail on floating-point roundoff or low-rate data.
            cutoff = max(np.nextafter(nyquist_hz, 0.0), nyquist_hz * 0.999)

        wn = cutoff / nyquist_hz
        if not (0 < wn < 1):
            raise ValueError(
                f"{stage} cutoff {cutoff_hz} invalid for Fs={fs_eff} (Nyquist={nyquist_hz})"
            )

        if filter_name in {"none", "off"}:
            return signal
        if filter_name == "butterworth":
            b, a = butter(order, wn, btype="low")
            return filtfilt(b, a, signal, axis=0)
        if filter_name == "fir":
            if fir_taps < 3:
                raise ValueError(f"{stage} FIR requires fir_taps >= 3, got {fir_taps}")
            b = firwin(fir_taps, cutoff, window=fir_window, fs=fs_eff)
            return filtfilt(b, [1.0], signal, axis=0)
        raise ValueError(
            f"Unsupported filter='{filter}'. Use 'butterworth', 'fir', or None."
        )

    P = apply_lowpass(
        P_bar * 1e5,
        cutoff_hz=fc_p,
        order=order_p,
        fir_taps=fir_taps_p,
        stage="Pressure",
    )

    if time is None:
        dPdt = np.gradient(P, 1.0 / Fs)
    else:
        dPdt = np.gradient(P, t)

    HRR = (V_m3 / (gamma - 1.0)) * dPdt
    HRR = apply_lowpass(
        HRR,
        cutoff_hz=fc_hrr,
        order=order_hrr,
        fir_taps=fir_taps_hrr,
        stage="HRR",
    )

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
            return result
    return result
