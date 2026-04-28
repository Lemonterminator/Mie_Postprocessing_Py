"""Quantitative analysis of k_sqrt vs k_quarter across the clean-fit archive.

Uses only clean/ subdirectories (post-outlier-filter), rows where
success==True, flag_bad_fit==False, mask_basic==True.

At transition time t0 the two branch amplitudes are:
  quarter-root branch: k_quarter * t0^(1/4)
  sqrt       branch : k_sqrt    * t0^(1/2)
  ratio = sqrt_amp / quarter_amp = (k_sqrt * t0^(1/4)) / k_quarter

Saves Thesis/images/ksqrt_ratio_distribution.png
"""
import pathlib, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "MLP" / "synthetic_data"
OUT_FIG  = ROOT / "Thesis" / "images" / "ksqrt_ratio_distribution.png"

# ── Load clean CSVs (deduplicated per unique plume fit) ───────────────────────
COLS = ["k_sqrt", "k_quarter", "t0", "success", "flag_bad_fit",
        "mask_basic", "mask_outlier", "file_path", "plume_idx"]

dfs = []
for csv in DATA_DIR.rglob("*.csv"):
    # only clean directories, only bw_polar (avoid duplicate penetration sources)
    if "clean" not in csv.parts or "bw_polar" not in csv.parts:
        continue
    if "flagged" in csv.name:
        continue
    try:
        df = pd.read_csv(csv, usecols=lambda c: c in COLS, low_memory=False)
        dfs.append(df)
    except Exception:
        pass

all_data = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded (clean/bw_polar): {len(all_data):,}")

# ── Normalise boolean columns ─────────────────────────────────────────────────
def to_bool(col):
    return col.astype(str).str.strip().str.lower().map({"true": True, "false": False})

all_data["success"]      = to_bool(all_data["success"])
all_data["flag_bad_fit"] = to_bool(all_data["flag_bad_fit"])
all_data["mask_basic"]   = to_bool(all_data["mask_basic"])
all_data["mask_outlier"] = to_bool(all_data["mask_outlier"])

# ── Filter: good fits only ────────────────────────────────────────────────────
# mask_basic=True  → passed basic quality gate
# mask_outlier=True → IS an outlier (exclude)
# flag_bad_fit=True → IS a bad fit (exclude)
accepted = all_data[
    all_data["success"].eq(True) &
    all_data["flag_bad_fit"].eq(False) &
    all_data["mask_basic"].eq(True) &
    all_data["mask_outlier"].eq(False)
].copy()

# Deduplicate: one row per unique plume fit (same file_path + plume_idx)
accepted = accepted.drop_duplicates(subset=["file_path", "plume_idx"])
print(f"Unique accepted plume fits: {len(accepted):,}")

# ── Compute branch amplitude ratio at transition time t0 ─────────────────────
# quarter-root amplitude: k_quarter * t0^(1/4)
# sqrt amplitude        : k_sqrt    * t0^(1/2)
# ratio = k_sqrt * t0^(1/4) / k_quarter
t0  = pd.to_numeric(accepted["t0"],       errors="coerce")
ksq = pd.to_numeric(accepted["k_sqrt"],   errors="coerce")
kqt = pd.to_numeric(accepted["k_quarter"],errors="coerce")

valid = (t0 > 0) & (kqt > 0) & (ksq >= 0)
t0, ksq, kqt = t0[valid], ksq[valid], kqt[valid]

ratio = ksq * (t0 ** 0.25) / kqt

# absolute amplitudes in mm (t0 is in seconds, k_quarter in mm·s^{-1/4})
amp_quarter_mm = kqt * (t0 ** 0.25)
amp_sqrt_mm    = ksq * (t0 ** 0.50)

frac_below_01  = (ratio < 0.1 ).mean()
frac_below_half= (ratio < 0.5 ).mean()
frac_below_one = (ratio < 1.0 ).mean()
median_ratio   = np.median(ratio)
p25 = np.percentile(ratio, 25)
p75 = np.percentile(ratio, 75)
N   = len(ratio)

med_quarter_mm = np.median(amp_quarter_mm)
med_sqrt_mm    = np.median(amp_sqrt_mm)

print(f"\n=== Branch amplitude ratio at t0  (sqrt_amp / quarter_amp) ===")
print(f"  N unique fits in analysis:    {N:,}")
print(f"  Median ratio:                 {median_ratio:.4g}")
print(f"  IQR:                          [{p25:.4g}, {p75:.4g}]")
print(f"  Fraction ratio < 0.1:         {frac_below_01:.1%}")
print(f"  Fraction ratio < 0.5:         {frac_below_half:.1%}")
print(f"  Fraction ratio < 1.0:         {frac_below_one:.1%}")
print(f"\n  Median quarter-root amp at t0: {med_quarter_mm:.2f} mm")
print(f"  Median sqrt amp at t0:         {med_sqrt_mm:.2e} mm")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: log10(ratio) distribution
ax = axes[0]
log_ratio = np.log10(ratio[ratio > 0])
ax.hist(log_ratio, bins=60, color="tab:blue", alpha=0.78, edgecolor="none")
ax.axvline(np.log10(0.5), color="tab:red", lw=1.8, linestyle="--",
           label=rf"ratio $= 0.5$  ({frac_below_half:.0%} below)")
ax.axvline(np.log10(median_ratio), color="tab:orange", lw=1.8, linestyle="-",
           label=f"Median = {median_ratio:.1e}")
ax.set_xlabel(
    r"$\log_{10}\!\left(\frac{k_{\sqrt{\ }}\cdot t_0^{1/4}}{k_{1/4}}\right)$"
    r"  (log amplitude ratio at $t_0$)",
    fontsize=9,
)
ax.set_ylabel("Count")
ax.set_title(
    r"(a)  Branch amplitude ratio at transition time $t_0$  (log scale)" +
    f"\n$N={N:,}$ accepted fits",
    fontsize=9,
)
ax.legend(fontsize=8.5)
ax.grid(True, linestyle="--", alpha=0.4)

# Right: quarter-root amplitude distribution (shows its physical scale)
ax2 = axes[1]
ax2.hist(amp_quarter_mm, bins=60, color="tab:green", alpha=0.78, edgecolor="none")
ax2.axvline(med_quarter_mm, color="tab:orange", lw=1.8, linestyle="-",
            label=f"Median = {med_quarter_mm:.1f} mm")
ax2.set_xlabel(r"Quarter-root branch amplitude at $t_0$  [mm]", fontsize=9)
ax2.set_ylabel("Count")
ax2.set_title(
    r"(b)  Quarter-root branch amplitude $k_{1/4}\cdot t_0^{1/4}$ at transition time" +
    f"\nMedian sqrt amplitude: {med_sqrt_mm:.1e} mm  (negligible by comparison)",
    fontsize=9,
)
ax2.legend(fontsize=8.5)
ax2.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=180, bbox_inches="tight")
print(f"\nFigure saved to {OUT_FIG}")
