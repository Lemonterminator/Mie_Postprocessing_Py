"""Stratified diagnostic for the occupied_angle detector.

Per-video classification:
  - SUCCESS: detected_count == nominal plumes AND every per-plume width in [5, 60] deg
  - FAILURE: otherwise

Outputs:
  1) Console: per-nozzle SUCCESS / total counts and percentage
  2) Console: within-SUCCESS widths summary (mean, std, range, IQR)
  3) PNG: 2-panel diagnostic
       (a) per-nozzle stacked bar (success vs failure)
       (b) widths histogram for SUCCESS vs FAILURE subsets
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "Mie_scattering_top_view_results"
W_LO, W_HI = 5.0, 60.0  # physically plausible per-plume cone width [deg]

records = []  # one row per video
all_widths_success, all_widths_failure = [], []
for meta_path in sorted(ROOT.rglob("*.meta.json")):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    nozzle = meta_path.parent.parent.name
    n_plumes = int(meta.get("plumes", 0) or 0)
    detected = int(meta.get("occupied_angle_segment_count", 0) or 0)
    widths = list(meta.get("occupied_angle_segment_widths_deg") or [])
    if n_plumes == 0:
        continue
    count_ok = (detected == n_plumes)
    range_ok = bool(widths) and all(W_LO <= float(w) <= W_HI for w in widths)
    success = count_ok and range_ok
    records.append({"nozzle": nozzle, "n_plumes": n_plumes, "detected": detected,
                    "count_ok": count_ok, "range_ok": range_ok, "success": success,
                    "n_widths": len(widths)})
    bucket = all_widths_success if success else all_widths_failure
    bucket.extend(float(w) for w in widths)

df = pd.DataFrame(records)
print(f"Total videos with meta: {len(df)}")
print(f"Overall SUCCESS: {df['success'].sum()} / {len(df)}  ({100*df['success'].mean():.1f}%)")
print(f"  count_ok only:  {df['count_ok'].sum()} ({100*df['count_ok'].mean():.1f}%)")
print(f"  range_ok only:  {df['range_ok'].sum()} ({100*df['range_ok'].mean():.1f}%)")
print()
per_nozzle = df.groupby("nozzle").agg(total=("success", "size"),
                                       success=("success", "sum"),
                                       count_only=("count_ok", "sum"),
                                       range_only=("range_ok", "sum"))
per_nozzle["pct"] = (100 * per_nozzle["success"] / per_nozzle["total"]).round(1)
print("Per-nozzle breakdown:")
print(per_nozzle.to_string())
print()

if all_widths_success:
    arr = np.array(all_widths_success)
    q25, q75 = np.percentile(arr, [25, 75])
    print(f"Within-SUCCESS widths (n={arr.size}):")
    print(f"  mean={arr.mean():.2f}  std={arr.std():.2f}  median={np.median(arr):.2f}")
    print(f"  range=[{arr.min():.1f}, {arr.max():.1f}]  IQR=[{q25:.1f}, {q75:.1f}]")
if all_widths_failure:
    arr_f = np.array(all_widths_failure)
    print(f"Within-FAILURE widths (n={arr_f.size}):")
    print(f"  mean={arr_f.mean():.2f}  std={arr_f.std():.2f}  median={np.median(arr_f):.2f}")
    print(f"  range=[{arr_f.min():.1f}, {arr_f.max():.1f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
nozzles = per_nozzle.index.tolist()
succ = per_nozzle["success"].to_numpy()
fail = per_nozzle["total"].to_numpy() - succ
xpos = np.arange(len(nozzles))
axes[0].bar(xpos, succ, color="#3a7d44", label="success")
axes[0].bar(xpos, fail, bottom=succ, color="#c44536", label="failure")
for i, p in enumerate(per_nozzle["pct"].to_numpy()):
    axes[0].text(i, succ[i] + fail[i] + max(succ.max(), 1) * 0.02, f"{p:.0f}%",
                 ha="center", va="bottom", fontsize=8)
axes[0].set_xticks(xpos); axes[0].set_xticklabels(nozzles, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("# videos"); axes[0].set_title("Detector success per nozzle")
axes[0].legend(loc="upper right"); axes[0].grid(alpha=0.3, axis="y")

bins = np.linspace(0, 360, 121)
axes[1].hist(all_widths_success, bins=bins, color="#3a7d44", alpha=0.75, label=f"SUCCESS (n={len(all_widths_success)})")
axes[1].hist(all_widths_failure, bins=bins, color="#c44536", alpha=0.55, label=f"FAILURE (n={len(all_widths_failure)})")
axes[1].axvspan(W_LO, W_HI, color="grey", alpha=0.1, label=f"sane range [{W_LO},{W_HI}]")
axes[1].set_xlabel("per-plume occupied width [deg]"); axes[1].set_ylabel("count")
axes[1].set_title("Width distribution (success vs failure)")
axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_yscale("log")

fig.tight_layout()
out_png = Path(__file__).with_suffix(".png")
fig.savefig(out_png, dpi=140)
print(f"\nSaved -> {out_png}")
