file_path = r"G:\Dewe\Test Set 1\BC20250805_Heated_Fuel_T1_0001.dxd"

import ctypes
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union, cast

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except Exception as exc:
    raise RuntimeError("matplotlib not installed: pip install matplotlib") from exc

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
    return_fig: bool = False,
    show: bool = False,
) -> Union[Axes, Tuple[Figure, Axes]]:
    """
    Plots selected columns from df onto a provided or new Matplotlib Axes.

    - Pass an existing `ax` to overlay on the same plot.
    - Returns `ax` by default; return `(fig, ax)` if `return_fig=True`.
    - Does not call plt.show() unless show=True.
    """
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
    df[matches].plot(ax=ax)
    # Labels/format
    ax.set_title(title)
    ax.set_xlabel("Time" if df.index.name else "Sample")
    ax.set_ylabel("Value")
    ax.grid(True)
    # Explicit legend (stable order)
    ax.legend(matches, title="Variables", loc="best")
    
    # Layout & (optional) show
    fig.tight_layout()
    if show:
        plt.show()

    return (fig, ax) if return_fig else ax


def resolve_data_path(default_path: str = file_path) -> Path:
    """Return a valid data path or raise FileNotFoundError."""
    path = Path(default_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def load_dataframe(path: Path) -> DataFrame:
    """Load the DXD file into a dataframe and emit diagnostic details."""
    result, err = read_with_dwdatareaderlib(path)
    if not result:
        if err:
            raise RuntimeError(f"DWDataReaderLib path skipped: {err}")
        raise RuntimeError("DWDataReaderLib did not return data or an error message.")

    dataframe_obj = result.get("dataframe")
    time_channel = result.get("time_channel")
    file_info = result.get("file_info", {})
    all_channels = result.get("channels", [])
    print("DWDataReaderLib result:")
    print(f" - DLL: {result.get('source', 'DWDataReaderLib')}")
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
        preview = ", ".join(ch["name"] for ch in all_channels[:8])
        if len(all_channels) > 8:
            preview += ", ..."
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

    if dataframe_obj is None:
        note = err or "DWDataReaderLib loaded data but pandas is missing; install pandas to build DataFrame."
        raise RuntimeError(f" - Note: {note}")

    if pd is None:
        raise RuntimeError("pandas is required to build a DataFrame from DWDataReaderLib output.")

    df = cast(DataFrame, dataframe_obj)
    print(f" - DataFrame shape: {df.shape}")
    print(f" - Columns: {list(df.columns)}")
    return df


def main() -> None:
    path = resolve_data_path()
    df = load_dataframe(path)

    # First pass: create independent figures/axes, don't show yet
    _fig1, _ax1 = cast(Tuple[Figure, Axes], plot_dataframe(
        df, title=path.name, 
        criteria=["Chamber Pressure", "Chamber Temperature"],
        return_fig=True, show=False
    ))

    df_heat_release = df[[col for col in df.columns if "Heat Release" in col]].copy()
    if "Heat Release" in df_heat_release.columns:
        df_heat_release["Heat Release"] = df_heat_release["Heat Release"] * 100  # scale for visibility

    # plot_dataframe(df, criteria=["Heat Release"], ax=_ax1, show=False)
    _fig2, _ax2 = cast(Tuple[Figure, Axes], plot_dataframe(
        df_heat_release, title=path.name, 
        criteria=["Heat Release"],
        return_fig=True, show=False
    ))


    _fig3, _ax3 = cast(Tuple[Figure, Axes], plot_dataframe(
        df, title=path.name, 
        criteria=["Injector current"],
        return_fig=True, show=False
    ))

    from heat_release_calulation import hrr_calc
    ChmbP_bar = df.get("Chamber pressure", None)
    time_seconds = df.index.to_numpy() if df.index.name == "time_s" else None
    df_hrr = hrr_calc(ChmbP_bar, time=time_seconds, V_m3=8.5e-3, gamma=1.35)
    # First plot (pressure-derived HRR)
    fig4, ax4 = plot_dataframe(
        df_hrr.set_index("time_s").rename(columns={"HRR_W": "Heat Release Rate"}),
        title="HRR",
        criteria=["Heat Release Rate"], return_fig=True
    )
    plt.show()



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc)
        sys.exit(1)
