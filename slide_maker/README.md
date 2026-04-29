# slide_maker

Batch postprocessing and comparison-video pipeline for Mie, Luminescence, Schlieren, and Dewesoft data.

---

## Quick Start

```
cd slide_maker
python process_all.py        # Step 1 — process raw .cine / .dxd files
python comparison_viewer.py  # Step 2 — render comparison video
```

Edit `config.json` before running. See [Configuration](#configuration) below.

---

## Prerequisites

### 1. FFmpeg (required for video output)

`comparison_viewer.py` calls `ffmpeg` via subprocess to encode the final `.mp4`.

**Windows install:**

```
winget install --id=Gyan.FFmpeg -e
```

Or download from https://ffmpeg.org/download.html and add the `bin/` folder to `PATH`.

Verify: `ffmpeg -version`

### 2. Python packages

```
pip install numpy pandas matplotlib opencv-python scikit-image scipy pycine
```

| Package | Used for |
|---|---|
| `numpy` | Array maths throughout |
| `pandas` | CSV loading / averaging |
| `matplotlib` | Plot rendering and animation |
| `opencv-python` | Video frame encode/decode (cv2) |
| `scikit-image` | Image filtering in Mie/Luminescence pipelines |
| `scipy` | Signal processing |
| `pycine` | Reading `.cine` camera files |

### 3. DWDataReaderLib (required for Dewesoft `.dxd` files, Windows only)

The Dewesoft reader wraps a native DLL. Download it once with:

```
python -m OSCC_postprocessing.dewe.download_dwdatareader
```

This places `DWDataReaderLib64.dll` inside `OSCC_postprocessing/dewe/DeweFileLibrary/`.
If you already have processed CSVs from a previous run, this step can be skipped —
`process_all.py` skips `.dxd` conversion if the output CSV already exists.

### 4. CuPy (optional, GPU acceleration)

Some pipeline stages can use CuPy for GPU acceleration. If not installed the code
falls back to NumPy transparently:

```
pip install cupy-cuda12x   # match your CUDA version
```

---

## Repository Layout

```
Mie_Postprocessing_Py/              ← repo root
├── slide_maker/                    ← this folder
│   ├── config.json                 ← edit this first
│   ├── process_all.py              ← Step 1: batch process raw files
│   ├── load_comparison.py          ← locates processed files for comparison
│   ├── comparison_viewer.py        ← Step 2: render comparison video
│   └── matplotlib_video_displayer.py
│
├── OSCC_postprocessing/            ← core library (do not edit)
│   ├── cine/                       ← .cine file reader
│   ├── dewe/                       ← Dewesoft .dxd reader, HRR calc
│   ├── filters/                    ← SVD background removal, bilateral filter
│   ├── analysis/                   ← penetration, cone angle, etc.
│   └── ...
│
├── mie_single_hole.py              ← Mie pipeline entry point
├── luminesence.py                  ← Luminescence pipeline entry point
└── manual_segment.py               ← Schlieren segmentation (run separately)
```

---

## Source Directory Structure

`config.json` has a `"directories"` block pointing to four raw-data folders.
Each folder must follow this layout before running `process_all.py`:

### Mie and Luminescence directories

```
<mie_dir>/                          e.g. G:\MeOH_test\Mie\
├── config.json                     ← calibration file (created by GUI.py)
│     contains: plumes, offset, centre_x, centre_y, inner_radius, outer_radius
├── T2_0001.cine                    ← testpoint 2, repetition 1
├── T2_0002.cine                    ← testpoint 2, repetition 2
├── T56_0001.cine
└── ...
```

After `process_all.py` runs, it creates:

```
<mie_dir>/Processed_Results/
├── Rotated_Videos/
│   ├── T2_0001.npz                 ← rotated video array
│   └── ...
└── Postprocessed_Data/
    ├── T2_0001_metrics.csv         ← Area, Penetration_from_TD, Cone_Angle, ...
    └── ...
```

**File naming convention:** `T{testpoint}_{repetition:04d}.cine`
- Testpoint number comes right after the `T` prefix.
- Repetition is a zero-padded 4-digit number (`_0001`, `_0002`, …).
- The pipeline uses this convention to group repetitions for averaging.

**Calibration file (`config.json` inside the video directory):**
Run `GUI.py` once per camera setup to generate it. Required fields:

| Field | Description |
|---|---|
| `plumes` | Number of spray plumes |
| `offset` | Rotation offset in degrees |
| `centre_x`, `centre_y` | Nozzle centre pixel coordinates |
| `inner_radius` | Inner crop radius (pixels) |
| `outer_radius` | Outer crop radius (pixels) |

### Schlieren directory

Schlieren `.cine` files are **not** processed by `process_all.py`.
Run `manual_segment.py` directly on the Schlieren folder instead.
`manual_segment.py` writes its NPZ output to:

```
<schlieren_dir>/Processed_Results/Rotated_Videos/T{n}_0001.npz
```

`comparison_viewer.py` picks these up automatically.

### Dewesoft directory

```
<dewe_dir>/                         e.g. G:\MeOH_test\Dewe\
├── T2_0001.dxd
├── T56_0001.dxd
└── ...
```

After `process_all.py` runs:

```
<dewe_dir>/Processed_Results/Postprocessed_Data/
├── T2_0001.csv
└── ...
```

If the `.dxd` → `.csv` conversion was done externally, place the CSVs in
`Processed_Results/Postprocessed_Data/` manually. The viewer reads only the CSV.

---

## Configuration (`config.json`)

```jsonc
{
    "directories": {
        "mie":          "G:\\MeOH_test\\Mie",
        "luminescence": "G:\\MeOH_test\\NFL",
        "schlieren":    "G:\\MeOH_test\\Schlieren",
        "dewe":         "G:\\MeOH_test\\Dewe"
    },

    // Each set is one output video. Key = slide title, value = list of testpoint numbers.
    "comparison_sets": {
        "Fuel Temperature - Reactive": [36, 42],
        "Gas Temperature - Reactive":  [36, 35, 37]
    },

    "processing": {
        "mode": "average",          // "sample" (rep 1 only) or "average" (all reps)
        "frame_limit": 200,         // null = no limit
        "save_processed_video_strips": false,
        "saved_video_fps": 20,
        "video_bits": 12            // bit depth of camera sensor
    },

    "hrr_parameters": {
        "V_m3":   8.5e-3,           // chamber volume
        "gamma":  1.35,
        "fc_p":   1000.0,           // pressure low-pass cutoff (Hz)
        "fc_hrr": 600.0             // HRR low-pass cutoff (Hz)
    },

    "alignment": {
        "grad_threshold": 5,        // injection current gradient threshold for SoE detection
        "pre_samples":    50,       // samples to keep before SoE
        "window_ms":      10.0      // x-axis window shown in time plots (ms)
    },

    "column_names": {
        // Lists of fallback column names for each signal.
        // First match found in the CSV wins.
        "chamber_pressure":    ["Chamber pressure (BarA)", "Chamber pressure"],
        "chamber_temperature": ["Temperature acc. Ideal gas law", "Chamber gas temperature"],
        "heat_release":        ["Heat Release", "HRR"],
        "injection_current":   ["Main Injector - Current Profile", "Current Profile"],
        "mie_area":            ["Area", "Spray Area"]
    },

    "output": {
        "video_file": "comparison_output.mp4",
        "fps": 10,
        "debug": false              // true = show window instead of saving
    }
}
```

---

## Modifying the Video Layout

Open `comparison_viewer.py` and edit the two lists near the top of the file:

```python
# Plot rows (left column) — remove, reorder, or add entries.
# Available keys: "pressure_temp_hrr", "mie_area", "injection_current"
PLOT_ROW_KINDS: List[str] = [
    "pressure_temp_hrr",
    "mie_area",
    "injection_current",
]

# Video rows (right columns) — remove, reorder, or add entries.
# Available keys: "mie", "schlieren", "luminescence", "heatmap"
# Rows with no data for any testpoint are automatically hidden.
VIDEO_ROW_KINDS: List[str] = [
    "mie",
    "schlieren",
    "luminescence",
    "heatmap",
]
```

Examples:
- Remove Schlieren: delete `"schlieren"` from `VIDEO_ROW_KINDS`.
- Show only pressure/HRR plot: set `PLOT_ROW_KINDS = ["pressure_temp_hrr"]`.
- Move injection current above pressure: swap the two entries.

No other code changes are needed. Rows that have no data are skipped automatically.

---

## Pipeline Steps

```
GUI.py                     ← calibrate camera once per setup
    ↓
process_all.py             ← batch process all .cine and .dxd files
    ↓
manual_segment.py          ← (Schlieren only) segment spray boundaries
    ↓
comparison_viewer.py       ← render comparison video per comparison_set
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg: command not found` | Install FFmpeg and add to PATH |
| `No config.json found in <dir>` | Run `GUI.py` to calibrate the camera first |
| `DWDataReaderLib DLL not found` | Run `python -m OSCC_postprocessing.dewe.download_dwdatareader` |
| `No .cine files found` | Check `directories.mie` / `directories.luminescence` in config.json |
| Warning: `Dewe CSV not found for T{n}` | Dewesoft file missing or not yet converted — pressure/HRR plots will be blank for that testpoint |
| `pycine` import error | `pip install pycine` |
