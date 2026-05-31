"""Stitch A / B / C per_fold.csv files into the final Tier-3A summary.

A and B were produced by an earlier `run_family_head_sweep.py` run
(`family_head_sweep_20260528_181949/`). C_family_head_5fold and
C_family_head_modified were produced by the relaunched run after the
architecture-mode bugs were fixed (`family_head_sweep_20260528_212646/`).

This driver collects all four `per_fold.csv` files and re-runs the same
`aggregate_lono_results` + `write_summary_and_verdict` logic that
`run_family_head_sweep.py` would have run if it had owned all four sweeps.

Usage:
    python MLP/MLP_training/ablations/merge_family_head_results.py
    # or override paths:
    python MLP/MLP_training/ablations/merge_family_head_results.py \
        --ab-dir <path-to-AB-sweep-dir> \
        --c-dir  <path-to-C-sweep-dir> \
        --output-dir <where-to-write-merged-summary>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here))                  # ablations/
    sys.path.insert(0, str(_here.parent))           # MLP_training/
    sys.path.insert(0, str(_here.parent / "ood_lono"))

from run_family_head_sweep import (
    MLP_ROOT,
    aggregate_lono_results,
    write_summary_and_verdict,
)


# (run_label, architecture_mode, protocol, sweep-dir name)
DEFAULT_LAYOUT: list[tuple[str, str, str, str, str]] = [
    ("A_single_baseline",       "single",      "lono_5fold",         "AB", "A_single_baseline"),
    ("B_modified_only",         "single",      "lono_modified_only", "AB", "B_modified_only"),
    ("C_family_head_5fold",     "family_head", "lono_5fold",         "C",  "C_family_head_5fold"),
    ("C_family_head_modified",  "family_head", "lono_modified_only", "C",  "C_family_head_modified"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ab-dir", type=Path,
        default=MLP_ROOT / "runs_mlp" / "family_head_sweep_20260528_181949",
        help="Sweep dir containing A_single_baseline/ and B_modified_only/.",
    )
    ap.add_argument(
        "--c-dir", type=Path,
        default=MLP_ROOT / "runs_mlp" / "family_head_sweep_20260528_212646",
        help="Sweep dir containing C_family_head_5fold/ and C_family_head_modified/.",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write merged family_head_lono.csv + summary + verdict. "
             "Defaults to the C sweep dir.",
    )
    args = ap.parse_args()

    ab_dir = args.ab_dir.resolve()
    c_dir = args.c_dir.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else c_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"A/B dir : {ab_dir}")
    print(f"C   dir : {c_dir}")
    print(f"Output  : {output_dir}\n")

    sweep_records: list[tuple[str, str, str, Path | None]] = []
    for run_label, arch_mode, protocol, source, subdir in DEFAULT_LAYOUT:
        root = ab_dir if source == "AB" else c_dir
        per_fold = root / subdir / "per_fold.csv"
        if per_fold.exists():
            print(f"[ok]   {run_label:30s} -> {per_fold}")
            sweep_records.append((run_label, arch_mode, protocol, per_fold))
        else:
            print(f"[miss] {run_label:30s} -> {per_fold} (skipped)")
            sweep_records.append((run_label, arch_mode, protocol, None))

    (output_dir / "merged_sources.json").write_text(json.dumps([
        {
            "run_label": rl,
            "architecture_mode": am,
            "protocol": pr,
            "per_fold_csv": str(pf) if pf else None,
        }
        for (rl, am, pr, pf) in sweep_records
    ], indent=2), encoding="utf-8")

    df = aggregate_lono_results(sweep_records, output_dir)
    write_summary_and_verdict(df, output_dir)
    print(f"\n[done] Wrote merged summary to {output_dir}")


if __name__ == "__main__":
    main()
