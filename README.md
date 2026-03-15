# oscc-postprocessing

Utilities for Mie imaging post-processing (masking, filters, plotting, I/O).

## Install

From PyPI:

```bash
pip install oscc-postprocessing
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


