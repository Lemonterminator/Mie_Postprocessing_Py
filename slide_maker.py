from typing import Iterable
from pathlib import Path
import os
import json
import numpy as np
from singlehole_pipeline import singlehole_pipeline
from OSCC_postprocessing.functions_bw import keep_largest_component
from OSCC_postprocessing.functions_videos import load_cine_video
from OSCC_postprocessing.optical_flow import compute_farneback_flows
from OSCC_postprocessing.video_filters import gaussian_video_cpu, median_filter_video_auto
from OSCC_postprocessing.video_playback import play_video_cv2, play_videos_side_by_side
from mie_multihole_pipeline import mie_multihole_pipeline
from OSCC_postprocessing.svd_background_removal import (
    svd_foreground_cuda as svd_foreground,
    godec_like,
)
from Dewe.dewe import *

##########################################
# Manual inputs
Schlieren_dir = r"" # r"G:\MeOH_test\Schlieren"

Mie_dir = r"G:\MeOH_test\Mie"
Luminescence_dir = r"G:\MeOH_test\NFL"
Dewe_dir = r"G:\MeOH_test\Dewe"
testpoints = {1:{1, 2, 3}, 2:{2,3}}

save_intermediate_results = False
##########################################


SCH_dir_path = Path(Schlieren_dir)
Mie_dir_path = Path(Mie_dir)
Luminescence_dir_path = Path(Luminescence_dir)
Dewe_dir_path = Path(Dewe_dir)


# Depending on the naming convention, adjust the processing of strings to Txx_xx
# Modify this function as needed
def raw_name_processing_sch(str) -> str:
    return str.replace(".cine", "").replace("Schlieren Cam_", "")

assert len(testpoints) < 5, "This program handles at most 5 comparisons"

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

    files = _iter_files(Dewe_dir_path)

    for dewe_path in files:
        # path = resolve_data_dir_path(dewe_path)
        df = load_dataframe(dewe_path)
        name = dewe_path.name.replace(".dxd", "")
        df.to_csv(data_dir_Dewe / f"{name}.csv")
