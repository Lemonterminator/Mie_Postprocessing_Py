import json
import re
from pathlib import Path

import pandas as pd

# Runtime config (edit these directly)
repo_root = Path(__file__).resolve().parent
folder_name = "BC20241016_HZ_Nozzle8"
config_name = "Nozzle8.json"

injection_pressure_bar_default = 2000
control_backpressure_bar_default = 4
FPS_default = 25000

DRY_RUN = False
REPAIR_FROM_CONFIG = False

ROOT_FOLDER = repo_root / folder_name
CONFIG_PATH = repo_root / "test_matrix_json" / config_name


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


def extract_group_id_from_path(path: Path | str) -> int | None:
    p = Path(path)
    for token in (p.name, p.stem):
        m = re.search(r"[Tt](\d+)", token)
        if m:
            return int(m.group(1))
        if token.isdigit():
            return int(token)
    return None


def build_group_lookup(config: dict) -> dict[int, dict]:
    groups = config.get("test_matrix", {}).get("groups", [])
    lookup: dict[int, dict] = {}
    for g in groups:
        gid = g.get("id")
        if gid is not None:
            lookup[int(gid)] = g
    return lookup


def is_missing_column(df: pd.DataFrame, column: str) -> bool:
    return column not in df.columns or df[column].isna().all()


def backfill(
    root_folder: Path,
    config_path: Path,
    dry_run: bool = False,
    repair_from_config: bool = False,
) -> tuple[int, int]:
    config = load_config(config_path)
    groups_by_id = build_group_lookup(config)
    nozzle_props = config.get("nozzle_properties", {})
    updated = 0
    skipped = 0

    for csv_path in sorted(root_folder.rglob("*.csv")):
        df = pd.read_csv(csv_path)
        changes: list[str] = []

        group_id = extract_group_id_from_path(csv_path.parent)
        group_cfg = groups_by_id.get(group_id or -1, {})

        injection_pressure_value = group_cfg.get("injection_pressure_bar", injection_pressure_bar_default)
        control_backpressure_value = group_cfg.get("control_backpressure", control_backpressure_bar_default)
        fps_value = nozzle_props.get("fps", FPS_default)

        cine_number = extract_cine_number(csv_path)
        duration = compute_injection_duration_us(config, cine_number)
        if duration is not None:
            if is_missing_column(df, "injection_duration_us"):
                df["injection_duration_us"] = duration
                changes.append(f"injection_duration_us={duration}")

        if repair_from_config or is_missing_column(df, "injection_pressure_bar"):
            df["injection_pressure_bar"] = injection_pressure_value
            changes.append(f"injection_pressure_bar={injection_pressure_value}")

        if "control_backpressure_bar" in df.columns:
            if repair_from_config or is_missing_column(df, "control_backpressure_bar"):
                df["control_backpressure_bar"] = control_backpressure_value
                changes.append(f"control_backpressure_bar={control_backpressure_value}")
        elif "control_backpressure" in df.columns:
            if repair_from_config or is_missing_column(df, "control_backpressure"):
                df["control_backpressure"] = control_backpressure_value
                changes.append(f"control_backpressure={control_backpressure_value}")
        else:
            df["control_backpressure_bar"] = control_backpressure_value
            changes.append(f"control_backpressure_bar={control_backpressure_value}")

        if is_missing_column(df, "fps"):
            df["fps"] = fps_value
            changes.append(f"fps={fps_value}")

        if not changes:
            skipped += 1
            print(f"Skip (already populated): {csv_path}")
            continue

        if dry_run:
            change_summary = ", ".join(changes)
            print(f"[DRY-RUN] Would set {change_summary} in {csv_path}")
        else:
            df.to_csv(csv_path, index=False)
            change_summary = ", ".join(changes)
            print(f"Updated {csv_path} -> {change_summary}")
        updated += 1

    return updated, skipped


def main():
    if not ROOT_FOLDER.exists():
        raise SystemExit(f"Root folder not found: {ROOT_FOLDER}")
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config JSON not found: {CONFIG_PATH}")

    updated, skipped = backfill(
        ROOT_FOLDER,
        CONFIG_PATH,
        dry_run=DRY_RUN,
        repair_from_config=REPAIR_FROM_CONFIG,
    )
    print(f"Done. Updated: {updated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
