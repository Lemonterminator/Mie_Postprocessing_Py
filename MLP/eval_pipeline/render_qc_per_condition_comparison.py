"""Render privacy-safe LV2/LV3 per-condition RMSE diagnostics from one eval run.

The comparison fixes the condition set from the highest-RMSE LV2 conditions,
then plots those same condition keys after LV3 QC gating.  It therefore shows
whether QC reduces the pre-QC error tail without independently cherry-picking
each population's worst cases.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MLP.eval_pipeline.common import read_json, resolve_path, write_json  # noqa: E402
from MLP.eval_pipeline.figures.per_model import (  # noqa: E402
    fig_per_condition_rmse,
    fig_qc_per_condition_rmse_comparison,
)
from MLP.eval_pipeline.figures.style import apply_style  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Layer-1 evalrun directory")
    parser.add_argument("--model", default="thesis_mlp_seed_99")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_style()
    run_dir = resolve_path(args.run)
    model_root = run_dir / "models" / args.model
    lv2_eval_dir = model_root / "lv2" / "full_clean"
    lv3_eval_dir = model_root / "lv3_qc_gated" / "full_clean"
    lv2_meta = read_json(lv2_eval_dir / "metrics.json")
    lv3_meta = read_json(lv3_eval_dir / "metrics.json")
    model_label = lv2_meta["model_label"]

    lv2_fig_dir = run_dir / "figures" / "models" / args.model / "lv2" / "full_clean"
    lv3_fig_dir = run_dir / "figures" / "models" / args.model / "lv3_qc_gated" / "full_clean"
    comparison_dir = run_dir / "figures" / "comparison" / "lv2_vs_lv3_qc" / "full_clean"
    written = []
    written += fig_per_condition_rmse(lv2_eval_dir, lv2_fig_dir, lv2_meta, top_n=args.top_n, dpi=args.dpi, formats=("png",))
    written += fig_per_condition_rmse(lv3_eval_dir, lv3_fig_dir, lv3_meta, top_n=args.top_n, dpi=args.dpi, formats=("png",))
    written += fig_qc_per_condition_rmse_comparison(
        lv2_eval_dir, lv3_eval_dir, comparison_dir, model_label=model_label,
        top_n=args.top_n, dpi=args.dpi, formats=("png",),
    )
    manifest = {
        "run_dir": str(run_dir),
        "model": args.model,
        "comparison": "LV2-selected top-N condition RMSE, paired with the same LV3 QC-gated condition keys",
        "top_n": args.top_n,
        "figures": [str(path) for path in written],
    }
    write_json(comparison_dir / "per_condition_rmse_qc_provenance.json", manifest)
    print(f"[done] rendered {len(written)} privacy-safe per-condition RMSE figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
