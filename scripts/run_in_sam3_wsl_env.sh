#!/usr/bin/env bash
set -euo pipefail

ENV_PREFIX="${ENV_PREFIX:-/home/linan/.conda/envs/sam3-wsl}"

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Error: missing WSL SAM3 env at ${ENV_PREFIX}" >&2
  exit 1
fi

export PATH="${ENV_PREFIX}/bin:${PATH}"
export CC="${CC:-${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-cc}"
export CXX="${CXX:-${ENV_PREFIX}/bin/x86_64-conda-linux-gnu-c++}"

exec "$@"
