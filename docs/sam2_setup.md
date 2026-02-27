# SAM2 Local Setup (Git-Ignored Assets)

This repository keeps SAM2 model assets out of git. Use the bootstrap downloader
to fetch checkpoints/configs into a local folder:

- Default folder: `.models/sam2`
- Git ignored via `.gitignore` (`.models/`)

## Prerequisites

Install core dependencies:

```bash
pip install torch sam2
```

Optional fallback tooling:

```bash
pip install huggingface_hub
```

## Download Command

From repo root:

```bash
python scripts/download_sam2.py --model small
```

Supported model keys:

- `tiny`
- `small` (default)
- `base_plus`
- `large`

Other useful flags:

```bash
python scripts/download_sam2.py --list
python scripts/download_sam2.py --model large --force
python scripts/download_sam2.py --dest .models/sam2
```

## Files Written

The downloader writes:

- `.models/sam2/checkpoints/<sam2 checkpoint>.pt`
- `.models/sam2/configs/<sam2 config>.yaml`
- `.models/sam2/manifest.json`

`manifest.json` tracks available models and an active model key for runtime
resolution.

## Runtime Path Contract (GUI/Backend)

Path resolution order is:

1. Explicit env overrides:
   - `SAM2_CKPT`
   - `SAM2_CONFIG`
2. `SAM2_HOME` + `manifest.json`
   - `SAM2_HOME` defaults to `<repo>/.models/sam2`
3. If neither works, raise a clear setup error instructing user to run
   `python scripts/download_sam2.py --model small`.

Reference resolver:

- `OSCC_postprocessing.utils.resolve_sam2_paths`

Example:

```python
from OSCC_postprocessing.utils import resolve_sam2_paths

paths = resolve_sam2_paths()
print(paths["checkpoint"], paths["config"])
```
