# Masters Thesis Code

This repository contains scripts for processing spray and schlieren videos, extracting flow and spray metrics, and running a weighted analysis pipeline used in the thesis work.

## What is in here

- `main_weighted.py`: Primary entry point that runs the weighted analysis pipeline.
- `opticalFlow.py`: Optical flow computations used for motion/velocity estimates.
- `videoProcessingFunctions.py`: Core video processing helpers used across scripts.
- `functions_videos.py` and `GUI_functions.py`: Utility functions and UI helpers.
- `clustering.py`, `extrapolation.py`, `data_capture.py`: Analysis utilities for clustering, extrapolation, and data collection.
- `spray_origins.json`: Configuration for spray origin positions.
- `Results/`: Example output CSV files.
- `Legacy/`: Older scripts kept for reference.

## Requirements

Install dependencies from:

```
pip install -r requirements.txt
```

## Typical usage

Run the main pipeline:

```
python main_weighted.py
```

## Outputs

The pipeline writes spray metrics as CSV files under `Results/`.

## Notes

- The `Legacy/` folder contains older experiments and is not required for the main pipeline.
