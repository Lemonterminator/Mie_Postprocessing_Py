#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
win_script_dir="$(wslpath -w "$script_dir")"

cmd.exe /C "cd /d \"${win_script_dir}\" && build_thesis.cmd"
