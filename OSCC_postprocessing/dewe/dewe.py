"""
Dewesoft Data Reader and Analysis Module
=========================================

This module provides tools for reading, processing, and visualizing Dewesoft (.dxd)
measurement data, with a focus on combustion chamber analysis.

Overview
--------
The module replicates the functionality of the MATLAB Slidemaker toolchain
(F:\\G\\Slidemaker), providing Python equivalents for:

- **DeweDataHandling.m** → `load_dataframe()`, `align_dewe_dataframe_to_soe()`
- **HrrCalc.m** → `hrr_calc()` (in heat_release_calulation.py)
- **ReactivePlottingFunc.m** → `plot_reactive()`

Key Features
------------
1. **DXD File Reading**: Direct reading of Dewesoft .dxd files via DWDataReaderLib DLL
2. **Data Alignment**: Align data to Start of Energization (SoE) based on injection current
3. **Heat Release Calculation**: Compute HRR using constant-volume formula
4. **Reactive Plotting**: Dual Y-axis plots matching MATLAB visualization style

Main Functions
--------------
- `load_dataframe(path)`: Load .dxd or .csv into a pandas DataFrame
- `align_dewe_dataframe_to_soe(df)`: Align/truncate data to injection start
- `plot_reactive(df)`: Create dual Y-axis plot with pressure, temperature, and HRR
- `plot_dataframe(df)`: General-purpose DataFrame plotting utility

MATLAB Equivalence
------------------
The data processing pipeline mirrors the MATLAB implementation:

1. **Start of Energization Detection** (DeweDataHandling.m lines 82-84):
   ```matlab
   idx_Zero{n} = find(gradient(dataCells{n}{1}.injectionCurrent) > 0.05, 1) - 3;
   ```
   Python equivalent: `align_dewe_dataframe_to_soe(df, grad_threshold=0.05, pre_samples=3)`

2. **Heat Release Rate Calculation** (HrrCalc.m line 15):
   ```matlab
   HRR = dChmbP * (0.0085 / (gamma - 1));  % V=8.5L, gamma=1.35
   ```
   Python equivalent: `hrr_calc(pressure, V_m3=8.5e-3, gamma=1.35)`

3. **Reactive Plot** (ReactivePlottingFunc.m):
   - Left Y-axis: Chamber Pressure (red), Temperature/10 (green)
   - Right Y-axis: Heat Release Rate (blue)
   Python equivalent: `plot_reactive(df)`

Dependencies
------------
- numpy: Array operations and numerical computation
- pandas: DataFrame handling for tabular data
- matplotlib: Visualization (optional, required for plotting)
- scipy: Signal filtering for HRR calculation (optional)

Windows-Specific
----------------
The DWDataReaderLib DLL is Windows-only. On other platforms, use pre-exported
CSV files via `load_dataframe(path.csv)`.

Example Usage
-------------
```python
from OSCC_postprocessing.dewe.dewe import load_dataframe, align_dewe_dataframe_to_soe, plot_reactive

# Load Dewesoft data
df = load_dataframe("path/to/data.dxd")

# Align to start of injection (10ms window)
df_aligned = align_dewe_dataframe_to_soe(df, window_ms=10.0)

# Plot with dual Y-axes (HRR calculated from pressure)
fig, ax = plot_reactive(
    df_aligned,
    V_m3=8.5e-3,   # Chamber volume
    gamma=1.35,    # Heat capacity ratio
)
```

Author: Auto-generated from MATLAB Slidemaker toolchain
See Also: heat_release_calulation.py for HRR computation details
"""

import ctypes
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, cast

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except Exception as exc:
    plt = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore[import]
except Exception:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd  # type: ignore[import]
except Exception:
    pd = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray
    from pandas import DataFrame
else:  # pragma: no cover - narrow type fallbacks for runtime
    Axes = Any  # type: ignore[assignment]
    Figure = Any  # type: ignore[assignment]
    NDArray = Any  # type: ignore[assignment]
    DataFrame = Any  # type: ignore[assignment]


def read_with_dwdatareaderlib(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Read Dewesoft .dxd data using the native DWDataReaderLib DLL.

    This is the low-level reader that interfaces directly with Dewesoft's
    DWDataReaderLib64.dll (Windows only). For most use cases, prefer
    `load_dataframe()` which provides a cleaner pandas DataFrame interface.

    Parameters
    ----------
    path : Path
        Path to the .dxd file to read.

    Returns
    -------
    result : dict or None
        Dictionary containing:
        - 'channels': List of channel dicts with 'name', 'data', 'time', 'unit'
        - 'async_channels': Channels with different time bases
        - 'time': Common time vector (numpy array)
        - 'file_info': Sample rate, duration, start time
        - 'dataframe': pandas DataFrame (if pandas available)
    error : str or None
        Error message if reading failed, None on success.

    Notes
    -----
    - Windows only (requires DWDataReaderLib64.dll or DWDataReaderLib.dll)
    - DLL must be in the DeweFileLibrary subdirectory
    - Channels with matching time bases are aligned into a single DataFrame

    See Also
    --------
    load_dataframe : Higher-level function returning a pandas DataFrame
    """

    if np is None:
        return None, "numpy required for DWDataReaderLib (pip install numpy pandas matplotlib)"
    assert np is not None  # Narrow type for type-checkers

    pandas_module = pd

    lib_dir = Path(__file__).resolve().parent / "DeweFileLibrary"
    dll_candidates = ["DWDataReaderLib64.dll", "DWDataReaderLib.dll"]
    dll_path = None
    for candidate in dll_candidates:
        candidate_path = lib_dir / candidate
        if candidate_path.exists():
            dll_path = candidate_path
            break

    if dll_path is None:
        return None, f"DWDataReaderLib DLL not found under {lib_dir}"

    class DWFileInfo(ctypes.Structure):
        _fields_ = [
            ("sample_rate", ctypes.c_double),
            ("start_store_time", ctypes.c_double),
            ("duration", ctypes.c_double),
        ]

    class DWChannel(ctypes.Structure):
        _fields_ = [
            ("index", ctypes.c_int),
            ("name", ctypes.c_char * 100),
            ("unit", ctypes.c_char * 20),
            ("description", ctypes.c_char * 200),
            ("color", ctypes.c_uint),
            ("array_size", ctypes.c_int),
            ("data_type", ctypes.c_int),
        ]

    lib = ctypes.WinDLL(str(dll_path))

    lib.DWInit.restype = ctypes.c_int
    lib.DWDeInit.restype = ctypes.c_int
    lib.DWOpenDataFile.restype = ctypes.c_int
    lib.DWOpenDataFile.argtypes = [ctypes.c_char_p, ctypes.POINTER(DWFileInfo)]
    lib.DWCloseDataFile.restype = ctypes.c_int
    lib.DWGetChannelListCount.restype = ctypes.c_int
    lib.DWGetChannelList.restype = ctypes.c_int
    lib.DWGetChannelList.argtypes = [ctypes.POINTER(DWChannel)]
    lib.DWGetScaledSamplesCount.restype = ctypes.c_longlong
    lib.DWGetScaledSamplesCount.argtypes = [ctypes.c_int]
    lib.DWGetScaledSamples.restype = ctypes.c_int
    lib.DWGetScaledSamples.argtypes = [
        ctypes.c_int,
        ctypes.c_longlong,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]

    DWSTAT_OK = 0

    def _decode_char_array(buffer: ctypes.Array[Any]) -> str:  # type: ignore[type-arg]
        return bytes(buffer).split(b"\x00", 1)[0].decode("utf-8", errors="ignore").strip()

    def _read_samples(ch_index: int, total: int) -> Tuple[NDArray, NDArray]:
        chunk = 500_000
        data = np.empty(total, dtype=np.float64)
        timestamps = np.empty(total, dtype=np.float64)
        offset = 0
        while offset < total:
            count = min(chunk, total - offset)
            data_buf = (ctypes.c_double * count)()
            time_buf = (ctypes.c_double * count)()
            status = lib.DWGetScaledSamples(ch_index, ctypes.c_longlong(offset), count, data_buf, time_buf)
            if status != DWSTAT_OK:
                raise RuntimeError(f"DWGetScaledSamples failed for channel {ch_index} at offset {offset} (status={status})")
            data[offset : offset + count] = np.ctypeslib.as_array(data_buf, shape=(count,))
            timestamps[offset : offset + count] = np.ctypeslib.as_array(time_buf, shape=(count,))
            offset += count
        return data, timestamps

    initiated = False
    opened = False

    try:
        status = lib.DWInit()
        if status != DWSTAT_OK:
            return None, f"DWInit failed with status {status}"
        initiated = True

        file_info = DWFileInfo()
        status = lib.DWOpenDataFile(str(path).encode("utf-8"), ctypes.byref(file_info))
        if status != DWSTAT_OK:
            return None, f"DWOpenDataFile failed with status {status}"
        opened = True

        ch_count = lib.DWGetChannelListCount()
        if ch_count <= 0:
            return None, f"No channels reported in file (count={ch_count})"

        ch_array = (DWChannel * ch_count)()
        status = lib.DWGetChannelList(ch_array)
        if status != DWSTAT_OK:
            return None, f"DWGetChannelList failed with status {status}"

        channels: List[Dict] = []
        async_channels: List[Dict] = []
        skipped: List[Tuple[str, str]] = []
        sync_data: Dict[str, "np.ndarray"] = {}
        time_vector: Optional["np.ndarray"] = None
        base_len: Optional[int] = None
        time_channel_name: Optional[str] = None

        for entry in ch_array:
            ch_name = _decode_char_array(entry.name) or f"channel_{entry.index}"
            ch_unit = _decode_char_array(entry.unit)
            ch_desc = _decode_char_array(entry.description)

            total = lib.DWGetScaledSamplesCount(entry.index)
            if total <= 0:
                skipped.append((ch_name, f"sample_count={total}"))
                continue

            try:
                data_arr, time_arr = _read_samples(entry.index, int(total))
            except RuntimeError as exc:
                skipped.append((ch_name, str(exc)))
                continue

            channel_record = {
                "index": entry.index,
                "name": ch_name,
                "unit": ch_unit,
                "description": ch_desc,
                "array_size": entry.array_size,
                "data_type": entry.data_type,
                "data": data_arr,
                "time": time_arr,
            }
            channels.append(channel_record)

            if time_vector is None:
                time_vector = time_arr.copy()
                base_len = len(time_vector)
                time_channel_name = ch_name
                sync_data[ch_name] = data_arr
                continue

            if base_len is not None and len(time_arr) == base_len:
                if np.allclose(time_arr, time_vector, rtol=0, atol=1e-9):
                    sync_data[ch_name] = data_arr
                    continue

            async_channels.append(channel_record)

        result: Dict[str, object] = {
            "source": str(dll_path),
            "channels": channels,
            "async_channels": async_channels,
            "skipped_channels": skipped,
            "time": time_vector,
            "time_channel": time_channel_name,
            "file_info": {
                "sample_rate": file_info.sample_rate,
                "duration": file_info.duration,
                "start_store_time": file_info.start_store_time,
            },
        }

        if time_vector is not None and sync_data:
            if pandas_module is not None:
                df = pandas_module.DataFrame(sync_data, index=time_vector)
                df.index.name = "time_s"
                result["dataframe"] = df
            else:
                result["dataframe"] = None
            return result, None

        return result, "Synchronous channel set missing or pandas unavailable"

    finally:
        if opened:
            try:
                lib.DWCloseDataFile()
            except Exception:
                pass
        if initiated:
            try:
                lib.DWDeInit()
            except Exception:
                pass


def _dispatch_dewesoft() -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    try:
        import win32com.client  # type: ignore
        import pythoncom  # type: ignore
    except Exception:
        return None, None, "pywin32 not available for COM automation"

    try:
        pythoncom.CoInitialize()
    except Exception:
        pass

    progids = [
        "DewesoftX.Application",
        "DEWESoftX.Application",
        "Dewesoft.Application",
        "DEWESoft.Application",
    ]
    last_err = None
    for pid in progids:
        try:
            app = win32com.client.Dispatch(pid)
            return app, pythoncom, None
        except Exception as exc:
            last_err = exc
    return None, None, f"Cannot start Dewesoft COM. Last error: {last_err}"


def export_via_dewesoftx_csv(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Use DewesoftX COM automation to export DXD to CSV and return CSV path."""

    app, pythoncom, err = _dispatch_dewesoft()
    if app is None:
        return None, err

    try:
        opened = app.LoadFile(str(path))
        if not opened:
            return None, "DewesoftX failed to load file"
    except Exception as exc:
        return None, f"LoadFile error: {exc}"

    out_dir = Path(tempfile.mkdtemp(prefix="dxd_export_"))
    out_csv = out_dir / (path.stem + ".csv")

    try:
        exp = app.Export
        exp.Type = "CSV"
        exp.FileName = str(out_csv)
        try:
            exp.CSV.DecimalSeparator = "."
            exp.CSV.Delimiter = ","
            exp.CSV.IncludeHeader = True
        except Exception:
            pass
        ok = exp.Execute()
        if not ok:
            return None, "DewesoftX export did not succeed"
    except Exception as exc:
        return None, f"Export failed: {exc}"

    if not out_csv.exists():
        csvs = list(out_dir.glob("*.csv"))
        if not csvs:
            return None, "No CSV produced by export"
        return str(csvs[0]), None

    return str(out_csv), None


def plot_dataframe(
    df: DataFrame,
    title: str = "DXD Data",
    criteria: Optional[Iterable[str]] = ("Heat Release", "Chamber Pressure", "Chamber Temperature"),
    ax: Optional[Axes] = None,
    label_prefix: Optional[str] = None,
    alpha: float = 1.0,
    linewidth: float = 1.5,
    legend_outside: bool = True,
    return_fig: bool = False,
    show: bool = False,
) -> Union[Axes, Tuple[Figure, Axes]]:
    """Plot selected columns from a DataFrame onto Matplotlib axes.

    General-purpose plotting utility for Dewesoft data. For combustion-specific
    dual Y-axis plots (pressure, temperature, HRR), use `plot_reactive()` instead.

    Parameters
    ----------
    df : DataFrame
        DataFrame with time-indexed data (index should be 'time_s' or similar).
    title : str
        Plot title. Default "DXD Data".
    criteria : iterable of str, optional
        Column name substrings to filter. Columns containing any of these strings
        (case-insensitive) will be plotted. Default: ("Heat Release", "Chamber Pressure",
        "Chamber Temperature"). Pass None to plot all columns.
    ax : Axes, optional
        Existing Matplotlib axes to plot on. If None, creates a new figure.
    label_prefix : str, optional
        Prefix to add to legend labels (useful for overlaying multiple datasets).
    alpha : float
        Line transparency (0-1). Default 1.0.
    linewidth : float
        Line width. Default 1.5.
    legend_outside : bool
        If True, place legend outside the plot area. Default True.
    return_fig : bool
        If True, return (fig, ax) tuple; otherwise return ax only. Default False.
    show : bool
        If True, call plt.show() after plotting. Default False.

    Returns
    -------
    ax : Axes
        The Matplotlib axes object (if return_fig=False).
    (fig, ax) : tuple
        Figure and axes objects (if return_fig=True).

    Example
    -------
    ```python
    df = load_dataframe("data.dxd")
    fig, ax = plot_dataframe(df, criteria=["Pressure", "Temperature"], return_fig=True)
    plt.show()
    ```

    See Also
    --------
    plot_reactive : Dual Y-axis plot for combustion data with calculated HRR
    """
    if plt is None:
        raise RuntimeError("matplotlib not installed: pip install matplotlib")

    cols = list(df.columns)
    matches = cols if criteria is None else [c for c in cols if any(k.lower() in c.lower() for k in criteria)]
    if not matches:
        raise RuntimeError("No columns to plot")

    # Create or reuse axes
    if ax is None:
        fig_obj, ax_obj = plt.subplots(figsize=(10, 6))
        fig = cast(Figure, fig_obj)
        ax = cast(Axes, ax_obj)
    else:
        fig = cast(Figure, ax.figure)  # type: ignore[attr-defined]

    assert ax is not None  # For type-checkers

    # Plot via pandas, targeting the given axes
    df_plot = df[matches]
    if label_prefix:
        df_plot = df_plot.rename(columns={c: f"{label_prefix}{c}" for c in df_plot.columns})
    df_plot.plot(ax=ax, alpha=float(alpha), linewidth=float(linewidth))
    # Labels/format
    ax.set_title(title)
    ax.set_xlabel("Time" if df.index.name else "Sample")
    ax.set_ylabel("Value")
    ax.grid(True)
    if legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=6)
        fig.tight_layout(rect=(0, 0, 0.82, 1))
    else:
        ax.legend(loc="best")
        fig.tight_layout()
    
    # (optional) show
    if show:
        plt.show()

    return (fig, ax) if return_fig else ax


def resolve_data_path(default_path: str) -> Path:
    """Return a valid data path or raise FileNotFoundError."""
    path = Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def _make_unique_column_name(name: str, existing: set[str]) -> str:
    """Generate a unique column name by appending __N suffix if name already exists."""
    if name not in existing:
        existing.add(name)
        return name
    suffix = 2
    while f"{name}__{suffix}" in existing:
        suffix += 1
    unique = f"{name}__{suffix}"
    existing.add(unique)
    return unique


def _align_to_timebase(
    *,
    base_time: "np.ndarray",
    src_time: "np.ndarray",
    src_data: "np.ndarray",
) -> "np.ndarray":
    """Interpolate channel data to align with a common time base.

    Used internally to synchronize channels with different sample rates
    onto a single time axis for DataFrame construction.
    """
    if np is None:
        raise RuntimeError("numpy is required to align Dewesoft channels")

    if len(src_time) == len(base_time) and np.allclose(src_time, base_time, rtol=0, atol=1e-9):
        return src_data.astype(np.float64, copy=False)

    src_time = np.asarray(src_time, dtype=np.float64).reshape(-1)
    src_data = np.asarray(src_data, dtype=np.float64).reshape(-1)

    if src_time.size != src_data.size:
        raise ValueError(f"Channel time/data length mismatch: time={src_time.size}, data={src_data.size}")

    if src_time.size == 0:
        return np.full_like(base_time, np.nan, dtype=np.float64)

    order = np.argsort(src_time)
    src_time = src_time[order]
    src_data = src_data[order]

    unique_time, unique_idx = np.unique(src_time, return_index=True)
    src_time = unique_time
    src_data = src_data[unique_idx]

    aligned = np.interp(base_time, src_time, src_data).astype(np.float64, copy=False)
    aligned[(base_time < src_time[0]) | (base_time > src_time[-1])] = np.nan
    return aligned


def load_dataframe(path: Path) -> DataFrame:
    """Load Dewesoft data into a pandas DataFrame.

    This is the primary entry point for loading Dewesoft measurement data.
    Supports both native .dxd files (Windows only) and pre-exported .csv files.

    Parameters
    ----------
    path : Path or str
        Path to the data file. Supported formats:
        - `.dxd`: Native Dewesoft format (requires DWDataReaderLib DLL on Windows)
        - `.csv`: Previously exported CSV (expects 'time_s' column or index)

    Returns
    -------
    DataFrame
        pandas DataFrame with:
        - Index: 'time_s' (time in seconds)
        - Columns: All measurement channels (e.g., 'Chamber pressure (BarA)',
          'Temperature acc. Ideal gas law', 'Main Injector - Current Profile')
        - Attrs: Metadata including 'dewe' dict with file_info and channel_meta

    Raises
    ------
    RuntimeError
        If pandas is not installed or DWDataReaderLib fails to load the file.
    ValueError
        If file format is not supported (.dxd or .csv).

    Notes
    -----
    - On Windows, .dxd files are read directly via DWDataReaderLib64.dll
    - On other platforms, pre-export data to CSV using DewesoftX
    - Channels with different sample rates are interpolated to a common time base
    - The DataFrame.attrs['dewe'] contains metadata for debugging

    Example
    -------
    ```python
    from OSCC_postprocessing.dewe.dewe import load_dataframe

    # Load from .dxd file (Windows)
    df = load_dataframe("G:/MeOH_test/Dewe/T2_0001.dxd")

    # Load from pre-exported CSV
    df = load_dataframe("data/exported.csv")

    print(df.columns)  # Available channels
    print(df.index)    # Time vector in seconds
    ```

    See Also
    --------
    align_dewe_dataframe_to_soe : Align data to injection start
    plot_dataframe : Plot selected columns
    plot_reactive : Create dual Y-axis combustion plot
    """

    if pd is None:
        raise RuntimeError("pandas is required to load Dewesoft data into a DataFrame.")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)  # type: ignore[call-arg]
        if "time_s" in df.columns:
            df = df.set_index("time_s")
            df.index.name = "time_s"
        return df

    if path.suffix.lower() != ".dxd":
        raise ValueError(f"Unsupported Dewesoft input type: {path} (expected .dxd or .csv)")

    result, err = read_with_dwdatareaderlib(path)
    if not result:
        if err:
            raise RuntimeError(f"DWDataReaderLib path skipped: {err}")
        raise RuntimeError("DWDataReaderLib did not return data or an error message.")

    time_channel = result.get("time_channel")
    file_info = result.get("file_info", {})
    all_channels = result.get("channels", [])
    print("DWDataReaderLib result:")
    # print(f" - DLL: {result.get('source', 'DWDataReaderLib')}")
    if file_info:
        sr = file_info.get("sample_rate")
        dur = file_info.get("duration")
        if sr:
            print(f" - Sample rate: {sr:.3f} Hz")
        if dur:
            print(f" - Duration: {dur:.3f} s")
    if time_channel:
        print(f" - Time channel: {time_channel}")
    if all_channels:
        preview = ",\n ".join(ch["name"] for ch in all_channels)
        # if len(all_channels) > 8:
        #     preview += ", ..."
        print(f" - Channels loaded: {len(all_channels)} ({preview})")
    if result.get("async_channels"):
        async_names = ", ".join(ch["name"] for ch in result["async_channels"][:6])
        if len(result["async_channels"]) > 6:
            async_names += ", ..."
        print(f" - Async channels (not in DataFrame): {async_names}")
    if result.get("skipped_channels"):
        print(" - Skipped channels:")
        for name, reason in result["skipped_channels"]:
            print(f"   * {name}: {reason}")

    if np is None:
        raise RuntimeError("numpy is required to build a DataFrame from DWDataReaderLib output.")

    if not all_channels:
        raise RuntimeError("No channels were loaded from the DXD file.")

    # Choose the densest time base (highest sample count) to avoid losing information.
    base_channel = max(all_channels, key=lambda ch: len(ch.get("time", [])))
    base_time = np.asarray(base_channel["time"], dtype=np.float64).reshape(-1)
    if base_time.size == 0:
        raise RuntimeError("Base time channel is empty; cannot build DataFrame.")

    df = pd.DataFrame(index=base_time)  # type: ignore[call-arg]
    df.index.name = "time_s"

    existing_names: set[str] = set()
    channel_meta: Dict[str, Dict[str, Any]] = {}

    for ch in all_channels:
        raw_name = cast(str, ch.get("name", "")) or f"channel_{ch.get('index', 'unknown')}"
        col_name = _make_unique_column_name(raw_name, existing_names)

        src_time = np.asarray(ch.get("time", []), dtype=np.float64)
        src_data = np.asarray(ch.get("data", []), dtype=np.float64)

        df[col_name] = _align_to_timebase(base_time=base_time, src_time=src_time, src_data=src_data)
        channel_meta[col_name] = {
            "source_name": raw_name,
            "unit": ch.get("unit", ""),
            "description": ch.get("description", ""),
            "index": ch.get("index"),
            "array_size": ch.get("array_size"),
            "data_type": ch.get("data_type"),
            "original_len": int(len(src_data)),
        }

    df.attrs["dewe"] = {
        "file_info": file_info,
        "time_channel_reported": time_channel,
        "time_base_channel": base_channel.get("name"),
        "channel_meta": channel_meta,
    }

    print(f" - DataFrame shape: {df.shape}")
    print(f" - Columns: {list(df.columns)}")
    return df


def _infer_sample_rate_hz_from_time_s(time_s: "np.ndarray") -> float:
    """Infer sampling rate in Hz from a time vector in seconds."""
    if np is None:
        raise RuntimeError("numpy is required to infer sample rate")

    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if time_s.size < 2:
        raise ValueError("Need at least 2 samples to infer sample rate")

    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot infer sample rate from non-increasing/invalid time base")

    median_dt = float(np.median(dt))
    if median_dt <= 0:
        raise ValueError("Invalid time base (non-positive dt)")
    return 1.0 / median_dt


def _nan_safe_series_values(values: "np.ndarray") -> "np.ndarray":
    """Fill NaN values by interpolation, ensuring a continuous series for gradient calculation."""
    if np is None:
        raise RuntimeError("numpy is required")

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values

    if np.all(~np.isfinite(values)):
        return np.zeros_like(values)

    out = values.copy()
    finite = np.isfinite(out)

    first = int(np.argmax(finite))
    out[:first] = out[first]

    idx = np.arange(out.size)
    out[~finite] = np.interp(idx[~finite], idx[finite], out[finite])
    return out


def plot_reactive(
    dataframes: Union[DataFrame, List[DataFrame]],
    *,
    testpoint_labels: Optional[List[str]] = None,
    chamber_pressure_col: Optional[str] = None,
    chamber_temperature_col: Optional[str] = None,
    time_col: str = "time_ms",
    temperature_scale: float = 0.1,  # Divide temperature by 10 for better scale
    # HRR calculation parameters (matching MATLAB HrrCalc.m)
    V_m3: float = 8.5e-3,  # Chamber volume in m³ (8.5 liters)
    gamma: float = 1.35,  # Heat capacity ratio
    fc_p: float = 1000.0,  # Pre-filter cutoff for pressure [Hz]
    fc_hrr: float = 600.0,  # Post-filter cutoff for HRR [Hz]
    smooth: bool = True,
    smoothing_factor: float = 0.2,
    show_std: bool = False,
    show_individual: bool = False,
    figsize: Tuple[float, float] = (12, 6),
    title: str = "Reactive Combustion Data",
    ax: Optional[Axes] = None,
    return_fig: bool = True,
) -> Union[Axes, Tuple[Figure, Axes]]:
    """
    Plot reactive combustion data with dual Y-axes, matching MATLAB ReactivePlottingFunc.

    Left Y-axis: Chamber Pressure (bar) in red, Chamber Temperature (K/10) in green
    Right Y-axis: Heat Release Rate (kJ/s) in blue - CALCULATED from pressure using hrr_calc

    The HRR is calculated using the constant-volume formula:
        HRR(t) = (V / (gamma - 1)) * dP/dt

    This matches the MATLAB HrrCalc.m implementation exactly.

    Parameters
    ----------
    dataframes : DataFrame or list of DataFrames
        Single dataframe or list of dataframes (one per testpoint/repetition).
        Each dataframe should have columns for pressure and temperature.
    testpoint_labels : list of str, optional
        Labels for each testpoint (e.g., ["T28", "T39", "T40"]).
    chamber_pressure_col : str, optional
        Column name for chamber pressure (in bar). Auto-detected if None.
    chamber_temperature_col : str, optional
        Column name for chamber temperature. Auto-detected if None.
    time_col : str
        Column name for time (in ms). Defaults to "time_ms".
    temperature_scale : float
        Scale factor for temperature (0.1 means divide by 10).
    V_m3 : float
        Chamber volume in m³. Default 8.5e-3 (8.5 liters, matching MATLAB).
    gamma : float
        Heat capacity ratio. Default 1.35 (matching MATLAB).
    fc_p : float
        Butterworth low-pass cutoff for pressure before differentiation [Hz].
    fc_hrr : float
        Butterworth low-pass cutoff for HRR after differentiation [Hz].
    smooth : bool
        Apply Gaussian smoothing to pressure/temperature data.
    smoothing_factor : float
        Smoothing factor (0-1, higher = smoother).
    show_std : bool
        Show standard deviation shading.
    show_individual : bool
        Plot individual traces with transparency.
    figsize : tuple
        Figure size (width, height).
    title : str
        Plot title.
    ax : Axes, optional
        Existing axes to plot on. If None, creates new figure.
    return_fig : bool
        If True, returns (fig, ax) tuple; otherwise returns ax only.

    Returns
    -------
    Axes or (Figure, Axes)
    """
    # Import hrr_calc for calculating heat release rate
    from OSCC_postprocessing.dewe.heat_release_calulation import hrr_calc

    if plt is None:
        raise RuntimeError("matplotlib not installed: pip install matplotlib")
    if pd is None:
        raise RuntimeError("pandas not installed: pip install pandas")
    if np is None:
        raise RuntimeError("numpy not installed: pip install numpy")

    # Normalize input to list of dataframes
    if isinstance(dataframes, pd.DataFrame):
        dfs = [dataframes]
    else:
        dfs = list(dataframes)

    if not dfs:
        raise ValueError("No dataframes provided")

    # Line styles for different testpoints (matching MATLAB)
    line_styles = ["-", "--", ":"]

    # Auto-detect column names from first dataframe
    first_df = dfs[0]

    def find_col(df: DataFrame, keywords: List[str], fallback: Optional[str] = None) -> Optional[str]:
        for col in df.columns:
            col_lower = col.lower()
            if all(kw.lower() in col_lower for kw in keywords):
                return col
        return fallback

    if chamber_pressure_col is None:
        chamber_pressure_col = find_col(first_df, ["chamber", "pressure"])
    if chamber_temperature_col is None:
        chamber_temperature_col = find_col(first_df, ["temperature"])
        if chamber_temperature_col is None:
            chamber_temperature_col = find_col(first_df, ["temp"])

    if chamber_pressure_col is None:
        raise ValueError("Could not find chamber pressure column. Please specify chamber_pressure_col explicitly.")

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create secondary y-axis
    ax2 = ax.twinx()

    # Color scheme matching MATLAB
    colors = {
        "pressure": "red",
        "temperature": "green",
        "heat_release": "blue",
    }

    # Smoothing function using scipy or fallback
    def smooth_data(data: "np.ndarray", factor: float = 0.2) -> "np.ndarray":
        if not smooth:
            return data
        try:
            from scipy.ndimage import gaussian_filter1d
            # Convert smoothing factor to sigma (higher factor = more smoothing)
            sigma = len(data) * factor * 0.1
            sigma = max(1, sigma)  # Minimum sigma of 1
            return gaussian_filter1d(data, sigma=sigma)
        except ImportError:
            # Simple moving average fallback
            window = max(1, int(len(data) * factor * 0.1))
            kernel = np.ones(window) / window
            return np.convolve(data, kernel, mode="same")

    # Plot handles for legend
    pressure_handles = []
    temp_handles = []
    hrr_handles = []

    for idx, df in enumerate(dfs):
        style = line_styles[idx % len(line_styles)]
        label_suffix = f" ({testpoint_labels[idx]})" if testpoint_labels and idx < len(testpoint_labels) else ""

        # Get time axis
        if time_col in df.columns:
            x = df[time_col].to_numpy()
            time_seconds = x / 1000.0  # Convert ms to seconds for hrr_calc
        elif df.index.name == "time_s":
            time_seconds = df.index.to_numpy()
            x = time_seconds * 1000  # Convert to ms for plotting
        elif "time_s" in df.columns:
            time_seconds = df["time_s"].to_numpy()
            x = time_seconds * 1000
        else:
            # Assume 100kHz sampling rate like MATLAB
            time_seconds = np.arange(len(df)) / 100000.0
            x = time_seconds * 1000

        # Plot Chamber Pressure (left axis)
        pressure_data = None
        if chamber_pressure_col and chamber_pressure_col in df.columns:
            pressure_data = df[chamber_pressure_col].to_numpy()
            y = smooth_data(pressure_data.copy(), smoothing_factor)
            h, = ax.plot(x, y, color=colors["pressure"], linestyle=style, linewidth=1.2,
                         label=f"Chamber Pressure (bar){label_suffix}")
            pressure_handles.append(h)

        # Plot Chamber Temperature (left axis, scaled)
        if chamber_temperature_col and chamber_temperature_col in df.columns:
            y = df[chamber_temperature_col].to_numpy() * temperature_scale
            y = smooth_data(y, smoothing_factor)
            h, = ax.plot(x, y, color=colors["temperature"], linestyle=style, linewidth=1.2,
                         label=f"Chamber Temp (K/{int(1/temperature_scale)}){label_suffix}")
            temp_handles.append(h)

        # Calculate and Plot Heat Release Rate (right axis) using hrr_calc
        if pressure_data is not None:
            # Calculate HRR from chamber pressure using the formula:
            # HRR(t) = (V / (gamma - 1)) * dP/dt
            hrr_result = hrr_calc(
                pressure_data,
                time=time_seconds,
                V_m3=V_m3,
                gamma=gamma,
                fc_p=fc_p,
                fc_hrr=fc_hrr,
                return_dataframe=True,
            )
            # hrr_result contains HRR_W in Watts, convert to kJ/s (kW)
            if hasattr(hrr_result, "to_numpy"):
                hrr_kw = hrr_result["HRR_W"].to_numpy() / 1000.0
            else:
                hrr_kw = hrr_result["HRR_W"] / 1000.0

            h, = ax2.plot(x, hrr_kw, color=colors["heat_release"], linestyle=style, linewidth=1.2,
                          label=f"Heat Release (kJ/s){label_suffix}")
            hrr_handles.append(h)

    # Configure axes
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Pressure (bar) / Temperature (K/10)", color="black")
    ax2.set_ylabel("Heat Release Rate (kJ/s)", color=colors["heat_release"])

    ax.tick_params(axis="y", labelcolor="black")
    ax2.tick_params(axis="y", labelcolor=colors["heat_release"])

    # Combined legend
    all_handles = []
    all_labels = []
    if pressure_handles:
        all_handles.append(pressure_handles[0])
        all_labels.append("Chamber Pressure (bar)")
    if temp_handles:
        all_handles.append(temp_handles[0])
        all_labels.append(f"Chamber Temp (K/{int(1/temperature_scale)})")
    if hrr_handles:
        all_handles.append(hrr_handles[0])
        all_labels.append("Heat Release (kJ/s)")

    ax.legend(all_handles, all_labels, loc="upper left", fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return (fig, ax) if return_fig else ax


def align_dewe_dataframe_to_soe(
    df: "DataFrame",
    *,
    injection_current_col: str = "Main Injector - Current Profile",
    grad_threshold: float = 0.05,
    pre_samples: int = 3,
    window_ms: float = 10.0,
    keep_time_index: bool = True,
) -> "DataFrame":
    """Align + truncate a Dewesoft-exported dataframe by Start of Energization (SoE).

    Mirrors the MATLAB logic:
    `idx_zero = find(gradient(injectionCurrent) > threshold, 1) - pre_samples`
    then take ~10 ms window starting at `idx_zero`.

    Operates in *row space* (iloc slicing), but uses `time_s` if available to
    convert `window_ms` into a sample count.
    """

    if pd is None or np is None:
        raise RuntimeError("align_dewe_dataframe_to_soe requires pandas and numpy")

    if injection_current_col not in df.columns:
        raise KeyError(f"Missing column {injection_current_col!r}. Available: {list(df.columns)}")

    # Infer time base (seconds) to convert ms -> samples.
    time_s: Optional["np.ndarray"] = None
    if getattr(df.index, "name", None) == "time_s":
        try:
            time_s = df.index.to_numpy(dtype=np.float64)  # type: ignore[assignment]
        except Exception:
            time_s = None
    if time_s is None and "time_s" in df.columns:
        time_s = df["time_s"].to_numpy(dtype=np.float64)

    if time_s is None:
        raise ValueError("Need a time base to compute window length (df.index named 'time_s' or a 'time_s' column).")

    sample_rate_hz = _infer_sample_rate_hz_from_time_s(time_s)
    window_samples = int(round(sample_rate_hz * (window_ms / 1000.0)))
    window_samples = max(window_samples, 1)

    current = df[injection_current_col].to_numpy(dtype=np.float64)
    current = _nan_safe_series_values(current)
    grad = np.gradient(current)

    candidates = np.flatnonzero(grad > float(grad_threshold))
    if candidates.size == 0:
        # Keep behavior non-fatal so slide_maker can still run for debugging.
        return df.iloc[:window_samples].copy()

    idx_zero = int(candidates[0]) - int(pre_samples)
    idx_zero = max(idx_zero, 0)
    idx_end = min(idx_zero + window_samples, len(df))

    out = df.iloc[idx_zero:idx_end].copy()

    if keep_time_index and getattr(out.index, "name", None) == "time_s":
        # Keep absolute time, but add a convenient aligned time axis.
        t0 = float(out.index[0])
        if "time_ms" in out.columns:
            out = out.drop(columns=["time_ms"])
        out.insert(0, "time_ms", (out.index.to_numpy(dtype=np.float64) - t0) * 1000.0)
    else:
        out = out.reset_index()
        if "time_s" in out.columns:
            t0 = float(out["time_s"].iloc[0])
            if "time_ms" in out.columns:
                out = out.drop(columns=["time_ms"])
            out.insert(0, "time_ms", (out["time_s"].to_numpy(dtype=np.float64) - t0) * 1000.0)

    out.attrs = dict(getattr(df, "attrs", {}))
    out.attrs.setdefault("alignment", {})
    out.attrs["alignment"].update(
        {
            "method": "soe_gradient",
            "injection_current_col": injection_current_col,
            "grad_threshold": float(grad_threshold),
            "pre_samples": int(pre_samples),
            "window_ms": float(window_ms),
            "window_samples": int(window_samples),
            "idx_zero": int(idx_zero),
        }
    )
    return out
