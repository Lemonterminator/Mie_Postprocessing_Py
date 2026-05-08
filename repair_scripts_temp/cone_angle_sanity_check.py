"""Compare per-plume occupied_angle (image-level scalar) with the quasi-steady
median of per-frame BW cone_angle_average. Outputs an R^2 summary and a scatter
PNG so we can decide whether occupied_angle is a faithful single-value proxy
for the BW cone-angle time series.

Usage: python repair_scripts_temp/cone_angle_sanity_check.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "Mie_scattering_top_view_results"
T_LO_MS, T_HI_MS = 0.3, 1.0  # quasi-steady window
MIN_VALID = 5

records = []
for meta_path in ROOT.rglob("*.meta.json"):
    csv_path = meta_path.with_suffix("").with_suffix(".csv")
    if not csv_path.exists():
        continue
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    n_plumes = int(meta.get("plumes", 0))
    widths = meta.get("occupied_angle_segment_widths_deg") or []
    fps = meta.get("fps")
    # Skip videos where the angular detector found a different plume count.
    if not fps or len(widths) != n_plumes or n_plumes == 0:
        continue
    df = pd.read_csv(csv_path)
    t_ms = df["frame_idx"].to_numpy() * 1000.0 / float(fps)
    in_win = (t_ms >= T_LO_MS) & (t_ms <= T_HI_MS)
    nozzle = meta_path.parent.parent.name
    for p in range(n_plumes):
        col = f"cone_angle_average_plume_{p}"
        if col not in df.columns:
            continue
        ts = df[col].to_numpy()[in_win]
        ts = ts[np.isfinite(ts) & (ts > 0)]
        if ts.size < MIN_VALID:
            continue
        records.append((nozzle, float(widths[p]), float(np.median(ts)), float(np.std(ts))))

print(f"Pairs collected: {len(records)}")
if not records:
    raise SystemExit("No valid (occupied, steady) pairs.")
arr_all = np.array([(o, s, ss) for _, o, s, ss in records])
occ_all, steady_all, std_all = arr_all[:, 0], arr_all[:, 1], arr_all[:, 2]

# Sane-range filter: per-plume cone-angle physically below ~90 deg; per-frame
# steady > 60 deg flags a likely segmentation flip rather than a real plume.
sane = (occ_all < 90.0) & (steady_all < 60.0)
print(f"After sane-range filter (occ<90 & steady<60): {sane.sum()} / {len(records)}"
      f"  ({100*sane.mean():.1f}%)")

def summarize(label, mask):
    o, s, ss = occ_all[mask], steady_all[mask], std_all[mask]
    if o.size < 10:
        print(f"[{label}] too few points"); return
    slope, intercept = np.polyfit(o, s, 1)
    ss_res = float(np.sum((s - (slope * o + intercept)) ** 2))
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(o, s)[0, 1])
    print(f"[{label}] n={o.size}  R^2={r2:.3f}  Pearson r={pearson:.3f}")
    print(f"         fit: steady = {slope:.3f} * occupied + {intercept:.3f}")
    print(f"         occ range [{o.min():.1f},{o.max():.1f}]  steady range [{s.min():.1f},{s.max():.1f}]")
    print(f"         median per-frame std in window: {float(np.median(ss)):.2f} deg")
    return slope, intercept, r2

summarize("ALL    ", np.ones_like(sane, dtype=bool))
fit = summarize("SANE   ", sane)
slope, intercept, r2 = fit
occ, steady = occ_all[sane], steady_all[sane]

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(occ, steady, s=8, alpha=0.35, edgecolor="none")
lim = [min(occ.min(), steady.min()) - 2, max(occ.max(), steady.max()) + 2]
ax.plot(lim, lim, "k--", lw=1, label="y=x")
xs = np.linspace(*lim, 100)
ax.plot(xs, slope * xs + intercept, "r-", lw=1, label=f"fit: y={slope:.2f}x+{intercept:.2f}")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("occupied_angle_segment_widths_deg [deg]")
ax.set_ylabel(f"median per-frame cone_angle_average in [{T_LO_MS},{T_HI_MS}] ms [deg]")
ax.set_title(f"Cone-angle sanity (sane-range): R^2={r2:.3f}, n={occ.size}")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
out_png = Path(__file__).with_suffix(".png")
fig.savefig(out_png, dpi=140)
print(f"Saved scatter -> {out_png}")
