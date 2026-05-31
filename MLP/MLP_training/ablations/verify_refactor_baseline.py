"""
Phase 0 regression check.

After each refactor phase, re-train with the baseline seed+config and run:

    python MLP/MLP_training/ablations/verify_refactor_baseline.py \
        --stage1-run <new_stage1_run_dir> \
        --stage2-run <new_stage2_run_dir> \
        --stage3-run <new_stage3_run_dir>

Exit 0 if all metrics are within tolerance, non-zero otherwise.
"""

import argparse
import json
import pathlib
import sys


BASELINE_JSON = pathlib.Path(__file__).parent / "config" / "refactor_baseline.json"


def load_csv_last_test_row(run_dir: pathlib.Path, filename: str = "test_summary.csv") -> dict:
    import csv
    p = run_dir / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    with open(p) as f:
        rows = list(csv.DictReader(f))
    test_rows = [r for r in rows if r.get("split", "test") == "test"]
    row = test_rows[-1] if test_rows else rows[-1]
    return {k: float(v) for k, v in row.items() if v not in ("", None) and k != "split"}


def load_rmse_eval(run_dir: pathlib.Path) -> dict:
    p = run_dir / "post_train_rmse_eval.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get("overall", {})


def check(label: str, got: float, ref: float, tol_rel: float | None = None, tol_abs: float | None = None) -> bool:
    if tol_rel is not None:
        diff_rel = abs(got - ref) / (abs(ref) + 1e-12)
        ok = diff_rel <= tol_rel
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {label}: got={got:.6f}  ref={ref:.6f}  rel_diff={diff_rel:.4f}  tol={tol_rel:.4f}")
        return ok
    if tol_abs is not None:
        diff = abs(got - ref)
        ok = diff <= tol_abs
        marker = "OK" if ok else "FAIL"
        print(f"  [{marker}] {label}: got={got:.6f}  ref={ref:.6f}  abs_diff={diff:.4f}  tol={tol_abs:.4f}")
        return ok
    raise ValueError("Provide tol_rel or tol_abs")


def verify_stage1(run_dir: pathlib.Path, spec: dict) -> bool:
    print(f"\n=== Stage 1: {run_dir.name} ===")
    metrics = load_csv_last_test_row(run_dir)
    ref = spec["test_metrics"]
    tol = spec["tolerance"]
    ok = check("physical_mae_mm", metrics["physical_mae"], ref["physical_mae_mm"],
               tol_rel=tol["physical_mae_mm_rel"])
    return ok


def verify_stage2(run_dir: pathlib.Path, spec: dict) -> bool:
    print(f"\n=== Stage 2: {run_dir.name} ===")
    metrics = load_csv_last_test_row(run_dir)
    ref = spec["test_metrics"]
    tol = spec["tolerance"]
    ok1 = check("nll_scaled", metrics.get("nll_scaled", float("nan")), ref["nll_scaled"],
                tol_abs=tol["nll_scaled_abs"])
    ok2 = check("physical_mae_mm", metrics["physical_mae"], ref["physical_mae_mm"],
                tol_rel=tol["physical_mae_mm_rel"])
    return ok1 and ok2


def verify_stage3(run_dir: pathlib.Path, spec: dict) -> bool:
    print(f"\n=== Stage 3: {run_dir.name} ===")
    rmse = load_rmse_eval(run_dir)
    if not rmse:
        print("  [SKIP] no post_train_rmse_eval.json found — run emit_stage3_headline.py first")
        return True
    ref = spec["rmse_eval"]
    tol = spec["tolerance"]
    ok1 = check("rmse_mm", rmse["rmse_mm"], ref["rmse_mm"],
                tol_abs=tol["rmse_mm_abs"])
    ok2 = check("coverage_1sigma", rmse["coverage_1sigma"], ref["coverage_1sigma"],
                tol_abs=tol["coverage_1sigma_abs"])
    return ok1 and ok2


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify refactor baseline metrics")
    parser.add_argument("--stage1-run", type=pathlib.Path, help="New stage1 run dir")
    parser.add_argument("--stage2-run", type=pathlib.Path, help="New stage2 run dir")
    parser.add_argument("--stage3-run", type=pathlib.Path, help="New stage3 run dir")
    parser.add_argument("--baseline", type=pathlib.Path, default=BASELINE_JSON,
                        help="Baseline JSON (default: refactor_baseline.json)")
    args = parser.parse_args()

    with open(args.baseline) as f:
        baseline = json.load(f)

    results = []

    if args.stage1_run:
        results.append(verify_stage1(args.stage1_run, baseline["stage1"]))
    else:
        print("\n[SKIP] --stage1-run not provided")

    if args.stage2_run:
        results.append(verify_stage2(args.stage2_run, baseline["stage2"]))
    else:
        print("\n[SKIP] --stage2-run not provided")

    if args.stage3_run:
        results.append(verify_stage3(args.stage3_run, baseline["stage3"]))
    else:
        print("\n[SKIP] --stage3-run not provided")

    if not results:
        print("\nNo runs provided — nothing to verify.")
        return 0

    all_ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
