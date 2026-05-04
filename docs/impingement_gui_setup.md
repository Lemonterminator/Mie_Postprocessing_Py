# Impingement GUI Environment

`impingement_gui.py` depends on the project package plus the MLP artifact directory used for impingement inference.

## 1. Create or reuse the virtual environment

```powershell
cd C:\Users\LJI008\Mie_Postprocessing_Py
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

`tkinter` ships with the standard Windows CPython installer, so it is not installed with `pip`.

## 2. Optional video export

The GUI can run without Manim. If you also want MP4 export:

```powershell
python -m pip install manim
```

If `manim` is missing, the GUI still saves the summary JSON, preview plot, and frame cache.

## 3. Point the GUI at a compatible run

The impingement GUI requires an `engineered_v2` run directory with:

- `train_config_used.json`
- one of `best_model_refinement.pt`, `best_model_stage2.pt`, or `best_model_stage1.pt`

You can set it explicitly before launch:

```powershell
$env:IMPINGEMENT_RUN_DIR="C:\path\to\your\engineered_v2_run"
python .\impingement_gui.py
```

Or launch the GUI and browse to the run directory from the first page.

## 4. Current repository state

As checked on 2026-04-24, this workspace contains:

```text
MLP\runs_mlp\stage2_NLL_penetration_20260317_194155
```

That run is `legacy_raw`, not `engineered_v2`, so it is not sufficient for `impingement_gui.py`.
