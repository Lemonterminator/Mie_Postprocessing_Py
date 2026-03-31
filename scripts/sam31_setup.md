# SAM 3.1 Setup

`SAM 3.1` is not currently a drop-in replacement for the existing `transformers`-based `examples/Sam3.ipynb`.

Important differences:

- `SAM 3.1` weights live in the gated Hugging Face repo `facebook/sam3.1`
- the official runtime is the Meta `sam3` package from GitHub
- Hugging Face explicitly says there is no Transformers integration for `facebook/sam3.1`

## What is included here

- [setup_sam31.ps1](./setup_sam31.ps1): Windows entry point
- [setup_sam31_wsl.sh](./setup_sam31_wsl.sh): actual installer that runs inside WSL
- [run_sam31_smoketest.py](./run_sam31_smoketest.py): smoke test for the official `sam3` runtime

## Why WSL is used

Native Windows install is currently not practical for `SAM 3.1` because the official `sam3` import chain requires `triton`, and `pip install triton` does not provide a native Windows wheel on this machine.

This machine already has:

- WSL2
- Ubuntu 22.04
- NVIDIA GPU visible from WSL

So the supported path here is WSL.

## One-command install

From the repo root in PowerShell:

```powershell
.\scripts\setup_sam31.ps1 -Token "<your_hugging_face_token>"
```

This will:

1. Enter WSL
2. Clone or refresh `facebookresearch/sam3` into `third_party/sam3_official`
3. Create `.venv_sam31_wsl`
4. Install `torch 2.10.0`, `torchvision 0.25.0`, official `sam3`, and `einops`
5. Run [run_sam31_smoketest.py](./run_sam31_smoketest.py)

## Expected runtime blockers

If the token is missing or does not have gated access to `facebook/sam3.1`, the install can still finish but the smoke test will fail when the official builder tries to download:

- `config.json`
- `sam3.1_multiplex.pt`

## Notes about Python version

Official Meta docs for the March 27, 2026 `SAM 3.1` release target Python 3.12+.

This WSL image currently has Python 3.10 by default. The script prefers `python3.12` when available and otherwise falls back to `python3` with a warning.

## Smoke test behavior

The smoke test:

1. Imports the official `sam3` package
2. Builds `build_sam3_predictor(version="sam3.1")`
3. Uses the official sample video asset bundled with the cloned Meta repo
4. Starts a session, adds a `"person"` text prompt, and streams a few frames
5. Writes a JSON summary to `Results/sam31_smoketest.json`
