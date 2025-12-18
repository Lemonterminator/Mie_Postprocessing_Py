
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
    """Read Dewesoft data using the native DWDataReaderLib DLL.

    Returns a dict with channel data, dataframe (if pandas available), and metadata.
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
    """
    Plots selected columns from df onto a provided or new Matplotlib Axes.

    - Pass an existing `ax` to overlay on the same plot.
    - Returns `ax` by default; return `(fig, ax)` if `return_fig=True`.
    - Does not call plt.show() unless show=True.
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
    """Load Dewesoft data into a single, tabulated pandas DataFrame.

    Supports:
    - `.dxd`: via `DWDataReaderLib` (Windows DLL)
    - `.csv`: previously-exported tabular data (expects a `time_s` column or index)
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
