# Manual inputs
Schlieren_folder = r"C:\Users\LJI008\OneDrive - Wärtsilä Corporation\Documents\Nozzle_temp_impact_SCH"
Mie_folder = r""
Luminescence_folder = r""
Dewe_folder = r""
testpoints = {1:{1, 2, 3}, 2:{2,3}}

save_intermediate_results = True


# Depending on the naming convention, adjust the processing of strings to Txx_xx
# Modify this function as needed
def raw_name_processing_sch(str) -> str:
    return str.replace(".cine", "").replace("Schlieren Cam_", "")

assert len(testpoints) < 5, "This program handles at most 5 comparisons"

from typing import Iterable
from pathlib import Path
import os
import numpy as np
from schlieren_singlehole_pipeline import schlieren_singlehole_pipeline
from mie_postprocessing.functions_bw import keep_largest_component
from mie_postprocessing.functions_videos import load_cine_video
from mie_postprocessing.optical_flow import compute_farneback_flows
from mie_postprocessing.video_filters import gaussian_video_cpu, median_filter_video_auto
from mie_postprocessing.video_playback import play_video_cv2, play_videos_side_by_side
from mie_postprocessing.svd_background_removal import (
    svd_foreground_cuda as svd_foreground,
    godec_like,
)


def _iter_files(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file())




SCH_folder_path = Path(Schlieren_folder)
Mie_folder_path = Path(Mie_folder)
Luminescence_folder_path = Path(Luminescence_folder)
Dewe_folder_path = Path(Dewe_folder)


import json 
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
        "No metadata JSON found alongside the cine files. "
        "Provide SCH_CENTRE_X, SCH_CENTRE_Y, and SCH_OFFSET environment variables to run without metadata."
    )




# Processing Schlieren Cine Videos
if not SCH_folder_path.exists():
    raise FileNotFoundError(f"Data folder not found: {SCH_folder_path}")
else: 
    save_folder_SCH = SCH_folder_path / "Processed_Results"
    save_folder_SCH.mkdir(exist_ok=True)

    rotated_folder_SCH = save_folder_SCH / "Rotated_Videos"
    rotated_folder_SCH.mkdir(exist_ok=True)

    data_SCH = save_folder_SCH / "Postprocessed_Data"
    data_SCH.mkdir(exist_ok=True)

files = _iter_files(SCH_folder_path)

for video_path in files:
    _ , offset, centre = _load_metadata(files)
    # print(f"Metadata: plumes={number_of_plumes}, offset={offset}, centre={centre}")

    # Locate the actual cine files
    cine_files = [f for f in files if f.suffix.lower() == ".cine"]
    if not cine_files:
        raise FileNotFoundError(f"No .cine files found in {SCH_folder_path}")
    # Loop through the cine files, no frame limit
    for cine_file in cine_files:
        video = load_cine_video(str(cine_file)).astype(np.float32) / 4096.0
        schlieren_singlehole_pipeline(video, offset, centre, cine_file.name, 
                                    rotated_folder_SCH, data_SCH,
                                    save_intermediate_results=save_intermediate_results,
                                    FPS=20)
    1




