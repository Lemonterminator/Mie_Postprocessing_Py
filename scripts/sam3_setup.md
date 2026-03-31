# SAM3 Windows Setup

This repo already contains a working local flow in [examples/Sam3.ipynb](../examples/Sam3.ipynb) that expects:

- Python environment: `.venv`
- Local model folder: `.models/sam3_hf`
- Test image: `mask.png`

The missing piece is the gated Hugging Face model download.

## What is required

1. Python 3.12 on Windows
2. Access to the gated model repo: `https://huggingface.co/facebook/sam3`
3. A Hugging Face token that can read that repo
4. NVIDIA GPU is optional, but this machine already has CUDA-enabled PyTorch and will use GPU automatically

## One-command setup

From the repo root in PowerShell:

```powershell
.\scripts\setup_sam3_windows.ps1 -Token "<your_hugging_face_token>"
```

What the script does:

1. Creates `.venv` if missing
2. Installs the repo in editable mode
3. Installs `transformers`, `requests`, and `ipykernel`
4. Registers a notebook kernel named `Python (.venv) - SAM3`
5. Downloads `facebook/sam3` into `.models/sam3_hf`
6. Runs a local smoke test against `mask.png`

If you do not want to put the token on the command line, set it in the shell first:

```powershell
$env:HF_TOKEN = "<your_hugging_face_token>"
.\scripts\setup_sam3_windows.ps1
```

## Manual setup

If you want to run the steps one by one:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install "transformers>=5.4.0" requests ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name mie-postprocessing-sam3 --display-name "Python (.venv) - SAM3"
```

Then download the model:

```powershell
$env:HF_TOKEN = "<your_hugging_face_token>"
.\.venv\Scripts\python.exe .\scripts\download_sam3.py
```

Then validate the local model:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_sam3_smoketest.py
```

Expected result:

- Console prints detection count
- Preview image is written to `Results/sam3_smoketest.png`

## Running the notebook

1. Open [examples/Sam3.ipynb](../examples/Sam3.ipynb)
2. Select kernel `Python (.venv) - SAM3`
3. Run cells from top to bottom

Notes:

- Cell 0 downloads the model from Hugging Face directly and therefore also needs repo access
- Cell 1 is the preferred local workflow and loads from `.models/sam3_hf`
- If `.models/sam3_hf` exists, use Cell 1 onward for reproducible local runs

## Troubleshooting

`401 Unauthorized` or `gated repo`:

- The token is missing, invalid, or does not have access to `facebook/sam3`
- Open `https://huggingface.co/facebook/sam3`, request access, then retry

`Sam3Model` import error:

- `transformers` is too old
- Re-run:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade "transformers>=5.4.0"
```

`examples/Sam3.ipynb` uses CPU instead of GPU:

- Check:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- `True` means CUDA is available
