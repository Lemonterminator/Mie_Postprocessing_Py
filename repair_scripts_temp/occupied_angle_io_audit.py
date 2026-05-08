"""I/O sanity audit for occupied_angle metadata.

Checks for plausible upstream I/O / serialization bugs before blaming the
algorithm:

  1) Schema parity across nozzles: do all meta.json have the same set of keys?
     A failing nozzle missing the occupied_angle keys would point to a stale
     pipeline write.

  2) widths length vs detected_count: if these disagree, serialization is
     corrupting the array.

  3) Sum-vs-total consistency: occupied_angle_total_deg should equal
     sum(widths) (modulo bin rounding). Disagreement is a serializer bug.

  4) File-mtime stratification: do failing nozzles have systematically older
     meta.json (suggesting they were written by an earlier pipeline version)?

  5) Sample a few failing meta.json values to eyeball what is actually stored.
"""
from pathlib import Path
import json
import datetime as dt
from collections import Counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "Mie_scattering_top_view_results"

SUCCESS_NOZZLES = {"BC20241003_HZ_Nozzle1", "BC20241007_HZ_Nozzle4",
                   "BC20241016_HZ_Nozzle8", "BC20241014_HZ_Nozzle3"}
FAIL_NOZZLES = {"BC20220627 - Heinzman DS300 - Mie Top view", "Nozzle0",
                "BC20241017_HZ_Nozzle2", "BC20241010_HZ_Nozzle5",
                "BC20241015_HZ_Nozzle7"}

records = []
for meta_path in sorted(ROOT.rglob("*.meta.json")):
    nozzle = meta_path.parent.parent.name
    cls = "success" if nozzle in SUCCESS_NOZZLES else ("fail" if nozzle in FAIL_NOZZLES else "mixed")
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        records.append({"nozzle": nozzle, "cls": cls, "error": f"json:{e}"})
        continue
    keys = frozenset(m.keys())
    widths = m.get("occupied_angle_segment_widths_deg") or []
    seg_count = m.get("occupied_angle_segment_count")
    total = m.get("occupied_angle_total_deg")
    sum_w = float(np.sum([float(w) for w in widths])) if widths else 0.0
    consistent_count = (seg_count is not None and len(widths) == int(seg_count))
    consistent_total = (total is not None and abs(sum_w - float(total)) < 1.0)  # 1 deg slack
    mtime = dt.datetime.fromtimestamp(meta_path.stat().st_mtime).isoformat(timespec="seconds")
    records.append({"nozzle": nozzle, "cls": cls, "n_keys": len(keys),
                    "has_widths": "occupied_angle_segment_widths_deg" in keys,
                    "has_total": "occupied_angle_total_deg" in keys,
                    "n_widths": len(widths), "seg_count": seg_count,
                    "consistent_count": consistent_count,
                    "consistent_total": consistent_total,
                    "sum_widths": round(sum_w, 1), "total": total,
                    "mtime": mtime, "keys_hash": hash(keys)})

df = pd.DataFrame(records).query("error.isna()" if "error" in pd.DataFrame(records).columns else "index == index")

print("=== 1) Schema parity ===")
schema_by_cls = df.groupby("cls")["keys_hash"].agg(lambda s: s.nunique())
print(schema_by_cls.to_string())
key_sample = df.drop_duplicates("keys_hash").set_index("keys_hash")
for h, row in key_sample.iterrows():
    has = (row["has_widths"], row["has_total"])
    print(f"  hash={h % 100000}  cls={row['cls']}  has_(widths,total)={has}  n_keys={row['n_keys']}")

print("\n=== 2) widths length vs seg_count ===")
mismatch = df[~df["consistent_count"]]
print(f"  Mismatch: {len(mismatch)} / {len(df)}")
if len(mismatch):
    print(mismatch[["nozzle", "cls", "n_widths", "seg_count"]].head(10).to_string())

print("\n=== 3) sum(widths) vs total ===")
inconsistent_total = df[~df["consistent_total"]]
print(f"  Inconsistent: {len(inconsistent_total)} / {len(df)}")
if len(inconsistent_total):
    sample = inconsistent_total[["nozzle", "cls", "sum_widths", "total"]].head(8)
    print(sample.to_string())

print("\n=== 4) mtime stratification ===")
mtime_stats = df.groupby("cls")["mtime"].agg(["min", "max", "count"])
print(mtime_stats.to_string())

print("\n=== 5) Sample failing meta values ===")
for nozzle in sorted(FAIL_NOZZLES):
    sub = df[df["nozzle"] == nozzle]
    if sub.empty:
        continue
    s = sub.iloc[0]
    print(f"  {nozzle}: seg_count={s['seg_count']}, n_widths={s['n_widths']}, "
          f"sum_w={s['sum_widths']}, total={s['total']}, mtime={s['mtime'][:10]}")
