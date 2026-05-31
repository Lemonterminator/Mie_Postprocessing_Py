# Mie Scattering Multi-Hole Spray Post-Processing Pipeline

Post-processing pipeline for top-view Mie scattering high-speed videos of multi-hole fuel injectors. Processes batches of Phantom `.cine` recordings, segments individual spray plumes, and exports time-resolved spray metrics (penetration, cone angle, area, estimated volume) to CSV with optional preview AVI and boundary-point NPZ files.

---

## Quick Start

```bash
# 1. Activate the virtual environment
.venv\Scripts\activate

# 2. Run with the hardcoded batch (Nozzles 1–8, as configured at the top of the script)
python mie_multi_hole.py

# 3. Run a single nozzle / folder override
python mie_multi_hole.py \
  --parent-folder   "F:\LubeOil\BC20241003_HZ_Nozzle1\cine" \
  --experiment-config "test_matrix_json\Nozzle1.json"

# 4. Write results to a custom location
python mie_multi_hole.py -o "G:\Mie_scattering_top_view_results"
```

Results land in `Mie_scattering_top_view_results\<dataset_name>\<subfolder>\` by default (see [Output Layout](#output-layout)).  
GPU acceleration (CuPy) is used automatically if a CUDA device is available; the pipeline falls back to NumPy otherwise.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | Tested with 3.11 |
| `OSCC_postprocessing` library | Internal library — must be installed in the venv |
| CuPy (optional) | Enables GPU acceleration; falls back to NumPy if absent |
| OpenCV | Required for preview AVI generation |
| NumPy, SciPy, pandas | Core array and tabular data work |

Install dependencies into the bundled venv:
```bash
pip install -e .   # from the repo root, if a pyproject.toml / setup.py is present
```

---

## Input Layout

Each **parent folder** contains one or more **subfolders**, each representing a single test point:

```
F:\LubeOil\BC20241003_HZ_Nozzle1\cine\
├── T01\
│   ├── config.json          ← geometry for all .cine files in this subfolder
│   ├── 001.cine
│   ├── 002.cine
│   └── ...
├── T02\
│   ├── config.json
│   └── *.cine
└── ...
```

### `config.json` (per subfolder)

Describes the injector geometry used by every `.cine` file in that subfolder.

```json
{
  "plumes":        8,
  "centre_x":     512.0,
  "centre_y":     512.0,
  "inner_radius":  80.0,
  "outer_radius": 420.0
}
```

| Field | Type | Description |
|---|---|---|
| `plumes` | int | Number of spray holes / plumes |
| `centre_x`, `centre_y` | float | Nozzle tip pixel coordinates |
| `inner_radius` | float | Inner radius of the annular processing mask (px) |
| `outer_radius` | float | Outer radius; also used for mm/px scaling (`90 mm / outer_radius`) |

### Experiment Config JSON (`test_matrix_json/NozzleN.json`)

Maps test-condition IDs (folder names like `T01`, `Group_01`) to injection parameters (pressure, duration, FPS, umbrella angle, etc.). Loaded once per parent folder by `load_experiment_config`.

---

## Configuration Reference

All processing parameters live as **module-level constants** near the top of `mie_multi_hole.py`. Edit them directly before running a batch.

### Batch paths

```python
parent_folders      = [...]   # List of cine root folders (one per nozzle/dataset)
experiment_configs  = [...]   # Matching list of JSON configs
```

Override at runtime with `--parent-folder` / `--experiment-config` CLI flags or `MIE_PARENT_FOLDER` / `MIE_EXPERIMENT_CONFIG` environment variables.

### Frame and noise settings

| Parameter | Default | Description |
|---|---|---|
| `frame_limit` | `80` | Maximum frames loaded per `.cine` file |
| `noise_floor_multiplier` | `3` | Noise floor multiplier for preprocessing thresholding (use 4 for Nozzle 0) |

### Preprocessing histogram scaling

| Parameter | Default | Description |
|---|---|---|
| `sobel_wsize` | `3` | Sobel filter kernel size |
| `sobel_sigma` | `1` | Gaussian sigma applied before Sobel |
| `threshold` | `0.02` | Absolute intensity threshold after Sobel |
| `q_min_foreground` / `q_max_foreground` | `5` / `99.99` | Percentile clipping for foreground channel |
| `q_min_highpass` / `q_max_highpass` | `5` / `99.9999` | Percentile clipping for highpass channel |

### Angular rotation / resampling

| Parameter | Default | Description |
|---|---|---|
| `angular_bins` | `720` | Number of angular bins for polar resampling (higher = finer plume separation) |
| `interpolation_mode` | `"nearest"` | Rotation interpolation: `"bilinear"`, `"bicubic"`, `"lanczos3"` |
| `border_mode` | `"constant"` | Border fill mode: `"constant"`, `"replicate"`, `"reflect"` |

### Penetration (CDF method)

| Parameter | Default | Description |
|---|---|---|
| `upper_quantile_cdf` | `1 - 5e-3` | Column-wise intensity CDF level that defines penetration depth (99.5 %) |

### Binary (BW) spray metrics

| Parameter | Default | Description |
|---|---|---|
| `nozzle_opening_detection_height` | `20` | Pixel height of the nozzle-tip detection window |
| `nozzle_opening_detection_width` | `30` | Pixel width of the nozzle-tip detection window |
| `thres_penetration_num_pix` | `5` | Minimum spray width (px) for x-axis penetration detection |
| `segment_bw_q_min` / `segment_bw_q_max` | `5` / `99.9` | Percentile range for per-plume BW thresholding |
| `global_bw_threshold` | `0.01` | Fixed threshold applied after plume-wise percentile scaling |
| `repair_bw` | `True` | Apply 3-D morphological closing + largest-component extraction to each plume BW volume |
| `penetration_cleanup_min_len` | `5` | Minimum run length (frames) for valid penetration traces |

### Output toggles

| Parameter | Default | Description |
|---|---|---|
| `save_boundary_points_csv` | `True` | Export plume boundary coordinates to `.npz` files |
| `save_preview_avi` | `True` | Write tiled multi-plume AVI with boundary overlay (async, non-blocking) |
| `preview_playback` | `False` | Open an interactive OpenCV window after each video (blocks pipeline — use for QC only) |
| `preview_fps` | `15` | Playback / AVI frame rate |
| `preview_tile` | `None` | `(rows, cols)` for preview tiling, or `None` for adaptive layout |

### Fallback nozzle defaults

Used when values are absent from both the `.cine` header and the experiment config JSON:

```python
FPS_default                      = 34000
injection_pressure_bar_default   = 2000
control_backpressure_bar_default = 4
umbrella_angle_deg_default       = 180
```

---

## Output Layout

```
Mie_scattering_top_view_results\
└── <dataset_name>\             ← e.g. BC20241003_HZ_Nozzle1
    ├── processing.log          ← append-only timestamped run log
    ├── processing_checkpoint.json  ← last processed file + status
    └── <subfolder>\            ← e.g. T01
        ├── 001.csv             ← per-video spray metrics (one row per frame)
        ├── 001.meta.json       ← per-video scalar metadata
        ├── boundary_points\
        │   └── 001_boundaries.npz   ← plume boundary coordinate arrays
        └── preview\
            └── 001_preview.avi      ← tiled multi-plume video with boundary overlay
```

Override the results root with `--results-base-dir` or `MIE_RESULTS_BASE_DIR`.

### Metrics CSV columns

| Column | Description |
|---|---|
| `frame_idx` | 0-based frame index |
| `cone_angle_proxy_deg` | Spray-wide cone angle proxy (degrees) |
| `occupied_angle_total_deg` | Total angular arc occupied by spray (degrees) |
| `occupied_angle_segment_count` | Number of detected spray segments |
| `penetration_cdf_plume_N` | CDF-based penetration depth for plume N (px) |
| `penetration_cdf(mm)_plume_N` | Same, converted to mm |
| `area_plume_N` | Binary spray area for plume N (px²) |
| `penetration_bw_x_plume_N` | BW x-axis penetration for plume N (px) |
| `penetration_bw_x(mm)_plume_N` | Same, converted to mm |
| `penetration_bw_polar_plume_N` | BW polar penetration for plume N (px) |
| `penetration_bw_polar(mm)_plume_N` | Same, converted to mm |
| `estimated_volume_plume_N` | Estimated spray volume for plume N (axisymmetric approximation) |
| `cone_angle_average_plume_N` | Average cone half-angle for plume N (degrees) |
| `cone_angle_linear_regression_plume_N` | Linear-regression cone angle for plume N (degrees) |
| `nozzle_opening_plume_N` | Detected nozzle opening frame index |
| `nozzle_closing_plume_N` | Detected nozzle closing frame index |

### Metadata JSON fields

Scalar metadata written alongside each CSV:

```json
{
  "plumes": 8,
  "diameter_mm": 0.18,
  "umbrella_angle_deg": 148.0,
  "fps": 34000,
  "chamber_pressure_bar": 60.0,
  "injection_duration_us": 1500.0,
  "injection_pressure_bar": 2000.0,
  "control_backpressure_bar": 4.0,
  "mm_per_px_scale": 0.214,
  "cone_angle_proxy_deg": [...],
  "occupied_angle_total_deg": [...],
  "occupied_angle_segment_count": [...],
  "occupied_angle_segment_widths_deg": [...]
}
```

---

## Processing Architecture

```
main()  →  _resolve_batch_jobs()
              │
              └─ _process_parent_folder()   [async, one per nozzle/dataset]
                    │
                    ├─ load experiment config + nozzle props
                    │
                    └─ for each subfolder (testpoint):
                          │
                          ├─ load config.json  (geometry)
                          ├─ prefetch thread:  _load_cine_to_cpu()  ← disk read (N+1)
                          │
                          └─ _process_cine_file()  [per .cine, main thread]
                                │
                                ├─ _cpu_to_backend()      H2D + fp16 normalisation
                                ├─ mie_multihole_preprocessing()   Sobel + noise floor
                                ├─ mie_multihole_postprocessing()  polar resample + segment
                                ├─ repair_binary_plume_video()     3-D morphology
                                ├─ penetration_cdf_all_plumes()    CDF penetration
                                ├─ _collect_metric_columns()       BW features (thread pool)
                                │
                                └─ I/O writer pool (async):
                                      metrics CSV + metadata JSON
                                      boundary NPZ
                                      preview AVI
```

**Key design decisions:**

- **GPU-CPU hybrid.** All heavy array operations run on the active backend (CuPy on GPU, NumPy on CPU). BW feature extraction (`spary_features_from_bw_video`) is NumPy/SciPy and runs in a `ThreadPoolExecutor` across plumes — it releases the GIL, so threads genuinely overlap.
- **Prefetch pipeline.** A single-worker `ThreadPoolExecutor` reads the next `.cine` from disk while the current file is on the GPU. H2D copies stay on the main thread to keep CuPy's per-thread CUDA context consistent.
- **Async I/O.** A 3-worker writer pool handles CSV, NPZ, and AVI writes. Futures are drained at the end of each parent folder.
- **CuPy memory pool.** Pool blocks are kept warm within a subfolder (faster buffer reuse) and freed at subfolder boundaries to prevent fragmentation across testpoints.
- **Checkpoint / resume.** `processing_checkpoint.json` is written before and after each file. Completed files (CSV + metadata + boundary NPZ all present) are skipped automatically on reruns.

---

## CLI Reference

```
python mie_multi_hole.py [OPTIONS]

Options:
  -p, --parent-folder PATH        Single-run parent folder override
                                  (env: MIE_PARENT_FOLDER)
  -c, --experiment-config PATH    Experiment config JSON for the override run
                                  (env: MIE_EXPERIMENT_CONFIG)
                                  Must be provided together with --parent-folder.
  -o, --results-base-dir PATH     Override the results root directory
                                  (env: MIE_RESULTS_BASE_DIR)
                                  Default: <script_dir>/Mie_scattering_top_view_results
```

**Batch mode** (no flags): processes all `parent_folders` / `experiment_configs` pairs defined at the top of the script in sequence.

---

## Environment Variables

| Variable | Equivalent flag | Description |
|---|---|---|
| `MIE_PARENT_FOLDER` | `-p` | Single-run parent folder |
| `MIE_EXPERIMENT_CONFIG` | `-c` | Single-run experiment config |
| `MIE_RESULTS_BASE_DIR` | `-o` | Results root directory override |

---

## Resume / Rerun Behaviour

The pipeline skips a `.cine` file if **all three** outputs already exist:
1. `<video_name>.csv`
2. `<video_name>.meta.json`
3. `boundary_points/<video_name>_boundaries.npz` (only checked when `save_boundary_points_csv = True`)

If the CSV and metadata exist but the boundary NPZ is missing, the file is **reprocessed** to generate the boundary data without recomputing metrics from scratch.

To force a full rerun, delete the relevant CSV / metadata files.

---

## Known Issues / Notes

- **`Procssing:` typo** in log output (lines 744–745) — cosmetic only, does not affect results.
- **Windows path separator** in subfolder joining (line 1062 uses `"\\"`) — the script is Windows-only as written.
- **`parent_folder` / `experiment_config` module-level scalars** (lines 84–85) are set to the last element of their respective lists. These are superseded by `_resolve_batch_jobs` in normal runs and only matter if you are importing the module directly.
- Phantom `.cine` FPS reading requires `pycine` to be installed; if absent, the fallback value from the experiment config or `FPS_default` is used.
