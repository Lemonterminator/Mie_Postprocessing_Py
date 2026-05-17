"""Run a configured suite of Stage-3 distillation ablations.

The suite is intentionally a thin subprocess runner: the training script remains
responsible for a single experiment, while this file handles ordering,
configuration, and failure reporting across multiple experiments.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MLP_ROOT = PROJECT_ROOT / "MLP"
TRAIN_SCRIPT = Path("MLP") / "MLP_training" / "train_stage3_distillation_plus_raw_series.py"
DEFAULT_CONFIG = Path("MLP") / "MLP_training" / "stage3_ablation_suite_config.json"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

COMMON_OPTION_KEYS = {
    "device",
    "series_split",
    "sources",
    "batch_size",
    "epochs",
    "learning_rate",
    "patience",
    "num_workers",
    "pin_memory",
    "precompute_dataset",
    "no_train",
    "save_figures",
    "seed",
    "skip_post_train_eval",
    "eval_split",
    "eval_t_min_ms",
    "eval_t_max_ms",
    "eval_rel_err_floor_mm",
    "eval_output_root",
    "eval_tag",
    "eval_batch_points",
    "eval_fast",
    "eval_save_points",
    "eval_save_plots",
    "eval_max_traj_plots",
    "synthetic_root",
}

BOOLEAN_FALSE_FLAGS = {
    "pin_memory": "--no-pin-memory",
    "precompute_dataset": "--no-precompute-dataset",
    "eval_save_points": "--no-eval-save-points",
    "eval_save_plots": "--no-eval-save-plots",
}

EVAL_SCRIPT = Path("MLP") / "eval" / "inference_rmse_on_series.py"


def path_for_child(path: Path | str) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def cli_flag(key: str) -> str:
    return f"--{key.replace('_', '-')}"


def append_cli_option(cmd: list[str], key: str, value: Any) -> None:
    if value is False and key in BOOLEAN_FALSE_FLAGS:
        cmd.append(BOOLEAN_FALSE_FLAGS[key])
        return
    if value is None or value is False:
        return
    flag = cli_flag(key)
    if value is True:
        cmd.append(flag)
        return
    if isinstance(value, (list, tuple)):
        cmd.append(flag)
        cmd.extend(str(item) for item in value)
        return
    cmd.extend([flag, path_for_child(value) if key.endswith(("root", "dir")) else str(value)])


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config.get("common", {}), dict):
        raise ValueError("Config field 'common' must be an object.")
    if not isinstance(config.get("ablations", []), list):
        raise ValueError("Config field 'ablations' must be a list.")
    return config


def resolve_teacher_run(value: str | Path | None) -> Path:
    if value is None or str(value) in {"", "latest_stage2"}:
        candidates = [
            path
            for path in (MLP_ROOT / "runs_mlp").glob("stage2_engineered_nll_*")
            if (path / "best_model_stage2.pt").exists()
        ]
        if not candidates:
            raise FileNotFoundError("Could not find a stage2 run with best_model_stage2.pt.")
        return max(candidates, key=lambda path: path.stat().st_mtime)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_command(
    *,
    python_exe: Path,
    teacher_run: Path,
    common: dict[str, Any],
    ablation: dict[str, Any],
    lono_holdout: str | None = None,
) -> list[str]:
    ablation_name = str(ablation["name"])
    cmd = [
        str(python_exe),
        str(TRAIN_SCRIPT),
        path_for_child(teacher_run),
    ]

    for key in sorted(COMMON_OPTION_KEYS):
        if key in common:
            append_cli_option(cmd, key, common[key])

    run_name_prefix = ablation.get("run_name_prefix")
    if run_name_prefix:
        cmd.extend(["--run-name-prefix", str(run_name_prefix)])
    cmd.extend(["--ablation-name", ablation_name])

    ablation_args = ablation.get("args", {})
    if not isinstance(ablation_args, dict):
        raise ValueError(f"Ablation '{ablation_name}' field 'args' must be an object.")
    for key, value in ablation_args.items():
        append_cli_option(cmd, key, value)

    if lono_holdout is not None:
        cmd.extend(["--lono-holdout", str(lono_holdout)])

    return cmd


def find_latest_refinement_run(prefix: str, started_at: float) -> Path | None:
    candidates = [
        path
        for path in (MLP_ROOT / "runs_mlp").glob(f"{prefix}_*")
        if path.is_dir() and path.stat().st_mtime >= started_at - 2.0
    ]
    if not candidates:
        candidates = [
            path
            for path in (MLP_ROOT / "runs_mlp").glob(f"{prefix}_*")
            if path.is_dir()
        ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_post_eval(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = run_dir / "post_train_rmse_eval.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_best_result(
    results: list[dict[str, Any]],
    *,
    metric: str,
    mode: str,
) -> dict[str, Any] | None:
    scored = []
    for result in results:
        if int(result.get("returncode", 1)) != 0:
            continue
        overall = (result.get("post_eval") or {}).get("overall", {})
        value = overall.get(metric)
        if isinstance(value, (int, float)):
            scored.append((float(value), result))
    if not scored:
        return None
    reverse = str(mode).lower() == "max"
    return sorted(scored, key=lambda item: item[0], reverse=reverse)[0][1]


def build_winner_eval_command(
    *,
    python_exe: Path,
    run_dir: Path,
    common: dict[str, Any],
    winner_eval: dict[str, Any],
) -> list[str]:
    cmd = [
        str(python_exe),
        str(EVAL_SCRIPT),
        "--refinement-run",
        path_for_child(run_dir),
        "--split",
        str(winner_eval.get("split", common.get("eval_split", common.get("series_split", "clean")))),
        "--t-min-ms",
        str(winner_eval.get("t_min_ms", common.get("eval_t_min_ms", 0.0))),
        "--t-max-ms",
        str(winner_eval.get("t_max_ms", common.get("eval_t_max_ms", 5.0))),
        "--rel-err-floor-mm",
        str(winner_eval.get("rel_err_floor_mm", common.get("eval_rel_err_floor_mm", 5.0))),
        "--batch-points",
        str(winner_eval.get("batch_points", common.get("eval_batch_points", 65536))),
    ]
    device = str(winner_eval.get("device", common.get("device", "auto")))
    if device and device.lower() != "auto":
        cmd.extend(["--device", device])
    output_root = winner_eval.get("output_root", common.get("eval_output_root"))
    if output_root:
        cmd.extend(["--output-root", path_for_child(output_root)])
    synthetic_root = winner_eval.get("synthetic_root", common.get("synthetic_root"))
    if synthetic_root:
        cmd.extend(["--synthetic-root", path_for_child(synthetic_root)])
    tag = winner_eval.get("tag", f"winner_full_{run_dir.name}")
    if tag:
        cmd.extend(["--tag", str(tag)])

    save_points = bool(winner_eval.get("save_points", True))
    save_plots = bool(winner_eval.get("save_plots", True))
    if not save_points:
        cmd.append("--no-save-points")
    if not save_plots:
        cmd.append("--no-save-plots")
    max_traj_plots = winner_eval.get("max_traj_plots")
    if max_traj_plots is not None:
        cmd.extend(["--max-traj-plots", str(max_traj_plots)])
    return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Stage-3 ablation suite from JSON config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--teacher-run", type=Path, default=None, help="Override config common.teacher_run.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None, help="Override config device.")
    parser.add_argument("--synthetic-root", type=Path, default=None,
                        help="Override synthetic data root passed to Stage-3 training and winner evaluation.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config common.seed (passed to every ablation training run).")
    parser.add_argument("--only", nargs="+", default=None, help="Run only these ablation names.")
    parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Also include entries from the optional sensitivity_ablations config block.",
    )
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Run only entries from the optional sensitivity_ablations config block.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--lono-holdout", type=str, default=None,
                        help="If set, hold out experiment_name=<value> as test; "
                             "use leave-one-nozzle-out split. Forwarded to each "
                             "Stage-3 training run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = load_config(config_path)
    common = dict(config.get("common", {}))
    if args.teacher_run is not None:
        common["teacher_run"] = str(args.teacher_run)
    if args.device is not None:
        common["device"] = args.device
    if args.synthetic_root is not None:
        common["synthetic_root"] = str(args.synthetic_root)
    if args.seed is not None:
        common["seed"] = int(args.seed)

    teacher_run = resolve_teacher_run(common.pop("teacher_run", None))
    python_exe = DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)

    requested = set(args.only or [])
    standard_items = list(config["ablations"])
    sensitivity_items = list(config.get("sensitivity_ablations", []))
    if args.sensitivity_only:
        candidate_items = sensitivity_items
    elif args.include_sensitivity:
        candidate_items = standard_items + sensitivity_items
    else:
        candidate_items = standard_items

    ablations = [
        item
        for item in candidate_items
        if item.get("enabled", True) and (not requested or str(item.get("name")) in requested)
    ]
    if requested:
        found = {str(item.get("name")) for item in ablations}
        missing = sorted(requested - found)
        if missing:
            raise KeyError(f"Requested ablation(s) not enabled/found in config: {', '.join(missing)}")
    if not ablations:
        raise ValueError("No enabled ablations selected.")

    print(f"Config: {config_path}")
    print(f"Teacher run: {teacher_run}")
    print(f"Python: {python_exe}")
    print(f"Selected ablations: {', '.join(str(item['name']) for item in ablations)}")

    results: list[dict[str, Any]] = []
    suite_started = datetime.now().strftime("%Y%m%d_%H%M%S")
    selection_config = dict(config.get("selection", {}))
    selection_metric = str(selection_config.get("metric", "rmse_mm"))
    selection_mode = str(selection_config.get("mode", "min"))
    winner_eval_config = dict(config.get("winner_eval", {}))
    for index, ablation in enumerate(ablations, start=1):
        name = str(ablation["name"])
        run_name_prefix = str(ablation.get("run_name_prefix") or "distill_cdf_onset_v2")
        cmd = build_command(
            python_exe=python_exe,
            teacher_run=teacher_run,
            common=common,
            ablation=ablation,
            lono_holdout=args.lono_holdout,
        )
        print()
        print(f"[{index}/{len(ablations)}] {name}")
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)

        start = time.time()
        returncode = 0
        run_dir: Path | None = None
        post_eval: dict[str, Any] | None = None
        if not args.dry_run:
            completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
            returncode = int(completed.returncode)
            run_dir = find_latest_refinement_run(run_name_prefix, start)
            post_eval = load_post_eval(run_dir)
        elapsed_s = time.time() - start
        results.append({
            "name": name,
            "returncode": returncode,
            "elapsed_s": round(elapsed_s, 3),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "post_eval": post_eval,
            "command": cmd,
        })
        if returncode != 0 and not args.continue_on_error:
            break

    best_result = None
    winner_eval_result = None
    if not args.dry_run:
        best_result = select_best_result(results, metric=selection_metric, mode=selection_mode)
        scored_results = [
            result
            for result in results
            if isinstance((result.get("post_eval") or {}).get("overall", {}).get(selection_metric), (int, float))
        ]
        if scored_results:
            reverse = selection_mode.lower() == "max"
            scored_results = sorted(
                scored_results,
                key=lambda result: float((result.get("post_eval") or {}).get("overall", {}).get(selection_metric)),
                reverse=reverse,
            )
            print()
            print("Ablation RMSE ranking:")
            for result in scored_results:
                overall = (result.get("post_eval") or {}).get("overall", {})
                print(
                    f"  {result['name']}: "
                    f"rmse={overall.get('rmse_mm')}  "
                    f"mae={overall.get('mae_mm')}  "
                    f"bias={overall.get('bias_mm')}  "
                    f"p95={overall.get('p95_abs_err_mm')}"
                )
        if best_result is not None:
            overall = (best_result.get("post_eval") or {}).get("overall", {})
            print()
            print(
                f"Best ablation by {selection_metric} ({selection_mode}): "
                f"{best_result['name']} = {overall.get(selection_metric)}"
            )
            if winner_eval_config.get("enabled", True):
                best_run_dir = Path(best_result["run_dir"])
                winner_cmd = build_winner_eval_command(
                    python_exe=python_exe,
                    run_dir=best_run_dir,
                    common=common,
                    winner_eval=winner_eval_config,
                )
                print()
                print("Running full plot evaluation for winner:")
                print(" ".join(shlex.quote(part) for part in winner_cmd), flush=True)
                winner_started = time.time()
                completed = subprocess.run(winner_cmd, cwd=PROJECT_ROOT)
                winner_eval_result = {
                    "returncode": int(completed.returncode),
                    "elapsed_s": round(time.time() - winner_started, 3),
                    "command": winner_cmd,
                }
                if completed.returncode != 0 and not args.continue_on_error:
                    results.append({
                        "name": f"{best_result['name']}_winner_full_eval",
                        "returncode": int(completed.returncode),
                        "elapsed_s": winner_eval_result["elapsed_s"],
                        "command": winner_cmd,
                    })

    if args.dry_run:
        print()
        print("Dry run complete; no suite summary saved.")
    else:
        summary_dir = MLP_ROOT / "runs_mlp" / "stage3_ablation_suites"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"suite_{suite_started}.json"
        summary_path.write_text(
            json.dumps(
                {
                    "config": str(config_path),
                    "teacher_run": str(teacher_run),
                    "dry_run": bool(args.dry_run),
                    "selection": {
                        "metric": selection_metric,
                        "mode": selection_mode,
                        "best": best_result,
                    },
                    "winner_eval": winner_eval_result,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print()
        print(f"Suite summary saved to: {summary_path}")
    failed = [item for item in results if item["returncode"] != 0]
    if failed:
        names = ", ".join(item["name"] for item in failed)
        raise SystemExit(f"Stage3 ablation suite failed: {names}")


if __name__ == "__main__":
    main()
