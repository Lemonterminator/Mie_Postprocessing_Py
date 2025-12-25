import os

try:
    os.environ["MPLBACKEND"] = "TkAgg"
    import matplotlib

    matplotlib.use("TkAgg", force=True)
    HAS_MPL = True
except Exception as exc:
    HAS_MPL = False
    matplotlib = None
    print(f"matplotlib unavailable; skipping plots: {exc}")



from typing import Iterable, Optional, Tuple, cast
from pathlib import Path
import os
import json
try:
    import numpy as np
except Exception as exc:
    print(f"numpy unavailable; skipping slide generation: {exc}")
    raise SystemExit(0)

try:
    from matplotlib.axes import Axes  # type: ignore[import]
    from matplotlib.figure import Figure  # type: ignore[import]
except Exception:
    Axes = object  # type: ignore[assignment]
    Figure = object  # type: ignore[assignment]
from singlehole_pipeline import singlehole_pipeline
from OSCC_postprocessing.binary_ops.functions_bw import keep_largest_component
from OSCC_postprocessing.cine.functions_videos import load_cine_video
from OSCC_postprocessing.motion.optical_flow import compute_farneback_flows
from OSCC_postprocessing.filters.video_filters import gaussian_video_cpu, median_filter_video_auto
from OSCC_postprocessing.playback.video_playback import play_video_cv2, play_videos_side_by_side
from mie_multihole_pipeline import mie_multihole_pipeline
from OSCC_postprocessing.filters.svd_background_removal import (
    svd_foreground_cuda as svd_foreground,
    godec_like,
)
from OSCC_postprocessing.cine.dewe.dewe import *

##########################################
# Manual inputs
Schlieren_dir = r"" # r"G:\MeOH_test\Schlieren"

Mie_dir = r"G:\MeOH_test\Mie"
Luminescence_dir = r"G:\MeOH_test\NFL"
Dewe_dir = r"G:\MeOH_test\Dewe"
testpoints_dictionary = {1:{2, 56}}

save_intermediate_results = False
mode = "sample" # "average all" or "sample"
##########################################
# Format handling functions 

# Funtion to extract testpoint number and repetition number from Dewe filename. 
# File name example: T2_0001.csv
def dewe_name_to_testpoint(name: str) -> int:
    return name.split(".csv")[0].split("_")[0].split("T")[-1]

def dewe_name_to_repetition(name: str) -> int:
    return int(name.split(".csv")[0].split("_")[1])


SCH_dir_path = Path(Schlieren_dir)
Mie_dir_path = Path(Mie_dir)
Luminescence_dir_path = Path(Luminescence_dir)
Dewe_dir_path = Path(Dewe_dir)


# Depending on the naming convention, adjust the processing of strings to Txx_xx
# Modify this function as needed
def raw_name_processing_sch(str) -> str:
    return str.replace(".cine", "").replace("Schlieren Cam_", "")

assert max(len(s) for s in testpoints_dictionary.values()) < 5, "This program handles at most 5 comparisons"

def _iter_files(dir: Path) -> Iterable[Path]:
    return sorted(p for p in dir.iterdir() if p.is_file())


def _load_metadata(files: Iterable[Path]) -> tuple[int, float, tuple[float, float]]:
    env_cx = os.getenv("SCH_CENTRE_X")
    env_cy = os.getenv("SCH_CENTRE_Y")
    env_offset = os.getenv("SCH_OFFSET")
    if env_cx and env_cy and env_offset:
        plumes = int(os.getenv("SCH_PLUMES", "0"))
        centre = (float(env_cx), float(env_cy))
        offset = float(env_offset)
        return plumes, offset, centre

    for file in files:
        if file.suffix.lower() == ".json":
            with file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                plumes = int(data["plumes"])
                offset = float(data["offset"])
                centre = (float(data["centre_x"]) + 4.0, float(data["centre_y"]) - 3.0)
                return plumes, offset, centre
    raise FileNotFoundError(
        
        "No metadata JSON found alongside the cine files.\n"
        "Provide SCH_CENTRE_X, SCH_CENTRE_Y, and SCH_OFFSET environment variables to run without metadata.\n"

        "Plese do the manual calibration in GUI.py first. "
    )




# Processing Schlieren Cine Videos
if not SCH_dir_path.exists():
    raise FileNotFoundError(f"Data dir not found: {SCH_dir_path}")

elif SCH_dir_path != Path("") and SCH_dir_path.exists():
    save_dir_SCH = SCH_dir_path / "Processed_Results"
    save_dir_SCH.mkdir(exist_ok=True)

    rotated_dir_SCH = save_dir_SCH / "Rotated_Videos"
    rotated_dir_SCH.mkdir(exist_ok=True)

    data_dir_SCH = save_dir_SCH / "Postprocessed_Data"
    data_dir_SCH.mkdir(exist_ok=True)

    files = list(_iter_files(SCH_dir_path))
    _, offset, centre = _load_metadata(files)

    cine_files = [f for f in files if f.suffix.lower() == ".cine"]
    if not cine_files:
        raise FileNotFoundError(f"No .cine files found in {SCH_dir_path}")

    for cine_file in cine_files:
        print(str(cine_file.name))
        video = load_cine_video(str(cine_file)).astype(np.float32) / 4096.0
        '''
        singlehole_pipeline("Schlieren", video, offset, centre, cine_file.name, 
                                    rotated_dir_SCH, data_dir_SCH,
                                    save_intermediate_results=save_intermediate_results,
                                    FPS=20)
        '''
    


if not Mie_dir_path.exists():
    raise FileNotFoundError(f"Data dir not found: {Mie_dir_path}")
else: 
    save_dir_Mie = Mie_dir_path / "Processed_Results"
    save_dir_Mie.mkdir(exist_ok=True)

    rotated_dir_Mie = save_dir_Mie / "Rotated_Videos"
    rotated_dir_Mie.mkdir(exist_ok=True)

    data_dir_Mie = save_dir_Mie / "Postprocessed_Data"
    data_dir_Mie.mkdir(exist_ok=True)

    files = list(_iter_files(Mie_dir_path))
    _, offset, centre = _load_metadata(files)

    cine_files = [f for f in files if f.suffix.lower() == ".cine"]
    if not cine_files:
        raise FileNotFoundError(f"No .cine files found in {Mie_dir_path}")

    for cine_file in cine_files:
        video = load_cine_video(str(cine_file)).astype(np.float32) / 4096.0

        singlehole_pipeline("Mie", video, offset, centre, cine_file.name, 
                        rotated_dir_Mie, data_dir_Mie,
                        save_intermediate_results=save_intermediate_results,
                        FPS=20)


if not Dewe_dir_path.exists():
    raise FileNotFoundError(f"Data dir not found: {Dewe_dir_path}")
else: 
    save_dir_Dewe = Dewe_dir_path / "Processed_Results"
    save_dir_Dewe.mkdir(exist_ok=True)

    data_dir_Dewe = save_dir_Dewe / "Postprocessed_Data"
    data_dir_Dewe.mkdir(exist_ok=True)

    plots_dir_Dewe = save_dir_Dewe / "Plots"
    plots_dir_Dewe.mkdir(exist_ok=True)

    files = _iter_files(Dewe_dir_path)

    # Export DXD -> CSV once (skip if already exported).
    for dewe_path in (p for p in files if p.suffix.lower() == ".dxd"):
        name = dewe_path.stem
        out_csv = data_dir_Dewe / f"{name}.csv"
        if out_csv.exists():
            continue
        df = load_dataframe(dewe_path)
        df.to_csv(out_csv)
    
    saved_files = list(_iter_files(data_dir_Dewe))

    names = list(set(f.name for f in saved_files))

    testpoint_correspondence = list(dewe_name_to_testpoint(name) for name in names)
    repetition_correspondence = list(dewe_name_to_repetition(name) for name in names)

    for comparison_set in testpoints_dictionary:
        fig_main: Optional[Figure] = None
        ax_main: Optional[Axes] = None
        fig_current: Optional[Figure] = None
        ax_current: Optional[Axes] = None

        if mode == "average all":
            selected_names = [name for name in names if dewe_name_to_testpoint(name) in 
                              [str(tp) for tp in testpoints_dictionary[comparison_set]]]
        elif mode == "sample":
            # I set to take the first repetition only for each testpoint
            selected_names = [name for name in names if (dewe_name_to_testpoint(name) in 
                              [str(tp) for tp in testpoints_dictionary[comparison_set]] and 
                              dewe_name_to_repetition(name) == 1)]
        
        for name in selected_names:
            df = load_dataframe(data_dir_Dewe / name)
            
            # reset time to start from 0s
            df.index = df.index-df.index[0]

            # df_plot = df # For debugging
              
            df_plot = align_dewe_dataframe_to_soe(
                df,
                injection_current_col="Main Injector - Current Profile",  # names might need to be changed
                grad_threshold=5,
                pre_samples=50,
                window_ms=10.0,
            )
            
            

            if "time_ms" in df_plot.columns:
                df_plot = df_plot.set_index("time_ms")

            label = Path(name).stem

            if HAS_MPL:
                fig_main, ax_main = cast(
                    Tuple[Figure, Axes],
                    plot_dataframe(
                        df_plot,
                        title="Aligned Dewesoft: pressure / temperature / heat release",
                        criteria=[
                            "Chamber pressure",
                            "Chamber gas temperature",
                            "Temperature acc. Ideal gas law",
                            "Heat Release",
                        ],
                        ax=ax_main,
                        label_prefix=f"{label} | ",
                        alpha=0.9,
                        linewidth=1.2,
                        return_fig=True,
                        show=False,
                    ),
                )

                fig_current, ax_current = cast(
                    Tuple[Figure, Axes],
                    plot_dataframe(
                        df_plot,
                        title="Aligned Dewesoft: injector current (filtered)",
                        criteria=["Main Injector - Current Profile/FIR Filter"],
                        ax=ax_current,
                        label_prefix=f"{label} | ",
                        alpha=0.9,
                        linewidth=1.2,
                        return_fig=True,
                        show=False,
                    ),
                )

        if HAS_MPL:
            if fig_main is not None:
                fig_main.savefig(plots_dir_Dewe / f"comparison_{comparison_set}_main.png", dpi=200)
            if fig_current is not None:
                fig_current.savefig(plots_dir_Dewe / f"comparison_{comparison_set}_current.png", dpi=200)

