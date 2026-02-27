import argparse
import json
import re
from pathlib import Path

import pandas as pd


def extract_cine_number(file_path: Path | str) -> int | None:
    path = Path(file_path)
    match = re.search(r"\d+", path.stem)
    if not match:
        return None
    return int(match.group(0))


def compute_injection_duration_us(
    config: dict,
    cine_number: int | None,
    fallback: float | int | None = None,
) -> float | int | None:
    if cine_number is None:
        return fallback

    lookup = config.get("injection_duration_lookup", {})
    formula = lookup.get("formula", {})
    block_expr = formula.get("block")
    rules = formula.get("rules", [])

    if not block_expr or not rules:
        return fallback

    safe_globals = {"__builtins__": {}}
    local_vars = {"cine_number": cine_number}
    try:
        block = eval(str(block_expr), safe_globals, local_vars)
        local_vars["block"] = block
        for rule in rules:
            condition_expr = rule.get("condition")
            result_expr = rule.get("result")
            if not condition_expr or result_expr is None:
                continue
            if bool(eval(str(condition_expr), safe_globals, local_vars)):
                return eval(str(result_expr), safe_globals, local_vars)
    except Exception:
        return fallback

    return fallback


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def backfill(root_folder: Path, config_path: Path, dry_run: bool = False) -> tuple[int, int]:
    config = load_config(config_path)
    updated = 0
    skipped = 0

    for csv_path in sorted(root_folder.rglob("*.csv")):
        cine_number = extract_cine_number(csv_path)
        duration = compute_injection_duration_us(config, cine_number)
        if duration is None:
            skipped += 1
            print(f"Skip (no duration): {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        df["injection_duration_us"] = duration

        if dry_run:
            print(f"[DRY-RUN] Would set injection_duration_us={duration} in {csv_path}")
        else:
            df.to_csv(csv_path, index=False)
            print(f"Updated {csv_path} -> injection_duration_us={duration}")
        updated += 1

    return updated, skipped


def main():
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Backfill injection_duration_us into existing CSV files."
    )
    parser.add_argument(
        "--root-folder",
        type=Path,
        default=repo_root / "BC20220627 - Heinzman DS300 - Mie Top view",
        help="Folder containing CSV files to patch.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "test_matrix_json" / "DS300.json",
        help="Experiment config JSON containing injection_duration_lookup.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing files.",
    )
    args = parser.parse_args()

    if not args.root_folder.exists():
        raise SystemExit(f"Root folder not found: {args.root_folder}")
    if not args.config.exists():
        raise SystemExit(f"Config JSON not found: {args.config}")

    updated, skipped = backfill(args.root_folder, args.config, dry_run=args.dry_run)
    print(f"Done. Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
