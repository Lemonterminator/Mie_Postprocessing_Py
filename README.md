# Mie Postprocessing Pipeline

This is a Python script for processing .cine video files for Mie scattering analysis. The script performs multi-threaded and multi-process video analysis with support for CUDA acceleration.

## Key Features

- Asynchronous processing of video files
- Multi-threaded and multi-process execution
- Support for CUDA acceleration (when available)
- Processing of multiple plumes in spray analysis
- Penetration profile calculations
- Spray metrics computation

## Main Function

The main function is `main()` which processes video files in a specified directory structure.

## Dependencies

Required packages:
- numpy
- pandas
- scipy
- scikit-image
- opencv-python
- cupy (for CUDA acceleration)

## How to Run

1. Ensure all dependencies are installed
2. Set the parent folder containing .cine files
3. Run: `python main.py -p "path/to/parent/folder"`

## Script Structure

The script processes video files in a nested directory structure:
- Parent folder
  - Subfolder (e.g., T1, T01, Group_01)
    - config.json (contains plume parameters)
    - .cine video files

## Processing Steps

1. Load .cine video files
2. Preprocessing with ring mask filtering
3. Multi-hole processing for spray analysis
4. Penetration profile calculation
5. Spray metrics computation
6. Save results to CSV files

## Note on Execution

The script may encounter system-level restrictions on Windows (Application Control policies) that prevent loading certain DLL files. This is a system-level issue rather than a problem with the script itself.
```

For the packaged Masters-thesis workflow:

```bash
pip install "oscc-postprocessing[masters-thesis]"
```

For local development:

```bash
pip install .
```

## Python API

Then import modules, e.g.:

```python
from OSCC_postprocessing.io.async_plot_saver import AsyncPlotSaver
```

For triangle thresholding, prefer the unified entry point:

```python
from OSCC_postprocessing.analysis.thresholding import triangle_binarize
```

`triangle_binarize_gpu` still exists as a compatibility alias, but new code
should use `triangle_binarize` so CPU/GPU fallback stays consistent across the
package.

## Masters-thesis CLI

Then run:

```bash
oscc-masters-thesis --video path/to/video.cine
```

Useful options:

```bash
oscc-masters-thesis --video path/to/video.cine --output-dir path/to/results --config-path path/to/config.json --mask-path path/to/mask.png --use-gpu auto
```

If `pycine` is missing, install the `masters-thesis` extra.

## Thesis Reproducibility Pipeline

The thesis workflow is wrapped under `pipelines/` as a four-phase chain:

```bash
python pipelines/run_full.py --input-root Mie_scattering_top_view_results
```

The phases are:

- Phase 1 fit archive: `pipelines/fit/run_fit_pipeline.py`
- Phase 2 audits: `pipelines/audit/run_audit_pipeline.py`
- Phase 3 MLP training: `pipelines/train/run_train_pipeline.py`
- Phase 4 thesis artifact assembly: `pipelines/report/run_report_pipeline.py`

Use `--skip-phase fit`, `--skip-phase audit`, `--skip-phase train`, or
`--skip-phase report` for partial reruns. Use `--dry-run` to check command
wiring on machines without `Mie_scattering_top_view_results/`.

See `pipelines/README.md` for archive layout, parent run pointers, generated
LaTeX snippets, and the Phase 2/Phase 4 audit scripts.

