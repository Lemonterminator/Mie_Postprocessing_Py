#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_sam31_wsl"
SAM3_REPO_DIR="${REPO_ROOT}/third_party/sam3_official"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN="python3.12"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
    echo "[warn] python3.12 not found; falling back to ${PYTHON_BIN}. Official SAM 3.1 docs target Python 3.12+."
  else
    echo "Error: python3.12 or python3 is required in WSL."
    exit 1
  fi
fi

echo "[info] repo_root=${REPO_ROOT}"
echo "[info] python_bin=${PYTHON_BIN}"

if ! "${PYTHON_BIN}" -m ensurepip --version >/dev/null 2>&1; then
  echo "Error: ${PYTHON_BIN} cannot create virtual environments because ensurepip is missing."
  echo "Install the Ubuntu package first, then rerun this script:"
  echo "  sudo apt update && sudo apt install -y python3-venv"
  exit 1
fi

if [[ ! -d "${SAM3_REPO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SAM3_REPO_DIR}")"
  git clone --depth 1 https://github.com/facebookresearch/sam3.git "${SAM3_REPO_DIR}"
else
  git -C "${SAM3_REPO_DIR}" fetch --depth 1 origin main
  git -C "${SAM3_REPO_DIR}" reset --hard origin/main
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  torch==2.10.0 \
  torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${VENV_DIR}/bin/python" -m pip install -e "${SAM3_REPO_DIR}" einops

echo "[info] running SAM 3.1 smoke test"
"${VENV_DIR}/bin/python" "${REPO_ROOT}/scripts/run_sam31_smoketest.py"
