# oscc-postprocessing

Utilities for Mie imaging post-processing (masking, filters, plotting, I/O).

Install the base package:

```
pip install .
```

Then import modules, e.g.:

```
from OSCC_postprocessing.io.async_plot_saver import AsyncPlotSaver
```

For the packaged Masters-thesis workflow, install the extra dependencies:

```bash
pip install .[masters-thesis]
```

Then run:

```bash
oscc-masters-thesis --video path/to/video.cine
```

Useful options:

```bash
oscc-masters-thesis --video path/to/video.cine --output-dir path/to/results --config-path path/to/config.json --mask-path path/to/mask.png --use-gpu auto
```


