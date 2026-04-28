"""Generate a two-panel figure for the thesis illustrating the Hermite crown
parameterisation: left panel shows the baseline profile with control points
and tangent arrows; right panel shows a depth-sweep to illustrate how a single
scalar knob continuously deforms the bowl shape (demonstrating sweep convenience
and C² smoothness).
"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from piston.design_hermite import _quintic_hermite_segment

OUT = ROOT / "Thesis" / "images" / "hermite_crown_profile.png"

# ── Baseline control points ──────────────────────────────────────────────────
r_bore     = 120.0
r_topland  =   5.0

ctrl_pts_base = np.array([
    [  0.0,  3.0],
    [ 55.0, 14.0],
    [100.0,  2.0],
    [115.0,  0.0],
], dtype=float)

seg_len = np.linalg.norm(np.diff(ctrl_pts_base, axis=0), axis=1)
alpha = 0.6
ctrl_vels_base = np.array([
    [seg_len[0] * alpha,  0.0],
    [seg_len[1] * alpha,  0.0],
    [seg_len[2] * alpha, -3.0],
    [seg_len[2] * 0.3,  -0.5],
], dtype=float)
ctrl_accels_base = np.zeros_like(ctrl_pts_base)


def build_profile(ctrl_pts, ctrl_vels, ctrl_accels):
    xs, ds = [], []
    for i in range(len(ctrl_pts) - 1):
        seg = _quintic_hermite_segment(
            ctrl_pts[i], ctrl_pts[i + 1],
            ctrl_vels[i], ctrl_vels[i + 1],
            ctrl_accels[i], ctrl_accels[i + 1],
            num_points=400,
        )
        xs.append(seg[:, 0])
        ds.append(seg[:, 1])
    return np.concatenate(xs), np.concatenate(ds)


# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# ── Left panel: baseline profile with control points and tangents ─────────────
ax = axes[0]
x_base, d_base = build_profile(ctrl_pts_base, ctrl_vels_base, ctrl_accels_base)

ax.plot(x_base, d_base, lw=2.2, color="tab:blue", label="Hermite crown profile")
ax.scatter(ctrl_pts_base[:, 0], ctrl_pts_base[:, 1],
           color="black", zorder=5, s=50, label="Control points")

# Tangent arrows (scaled for readability)
arrow_scale = 0.06
for i, (pt, vel) in enumerate(zip(ctrl_pts_base, ctrl_vels_base)):
    ax.annotate(
        "", xy=(pt[0] + vel[0] * arrow_scale, pt[1] + vel[1] * arrow_scale),
        xytext=pt,
        arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.4),
    )

ax.axvline(r_bore - r_topland, color="gray", linestyle=":", lw=1.2,
           label=f"Topland start ({r_bore - r_topland:.0f} mm)")
ax.axvline(r_bore, color="red", linestyle="--", lw=1.2, label="Bore wall")

ax.set_xlim(-5, r_bore + 8)
ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])  # depth positive downward
ax.invert_yaxis()
ax.set_xlabel("Radial position [mm]")
ax.set_ylabel("Crown depth [mm]  (↓ into piston)")
ax.set_title("(a)  Baseline profile and control-point tangents")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# ── Right panel: nadir-depth sweep ──────────────────────────────────────────
ax2 = axes[1]

nadir_depths = np.linspace(6.0, 22.0, 7)
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.15, 0.85, len(nadir_depths)))

for depth, color in zip(nadir_depths, colors):
    ctrl_pts_sweep = ctrl_pts_base.copy()
    ctrl_pts_sweep[1, 1] = depth          # vary nadir depth only
    x_s, d_s = build_profile(ctrl_pts_sweep, ctrl_vels_base, ctrl_accels_base)
    ax2.plot(x_s, d_s, lw=1.8, color=color,
             label=f"{depth:.0f} mm")

ax2.axvline(r_bore - r_topland, color="gray", linestyle=":", lw=1.2)
ax2.axvline(r_bore, color="red", linestyle="--", lw=1.2)
ax2.invert_yaxis()
ax2.set_xlabel("Radial position [mm]")
ax2.set_ylabel("Crown depth [mm]  (↓ into piston)")
ax2.set_title("(b)  Nadir-depth sweep — single scalar, continuous C² deformation")
ax2.set_xlim(-5, r_bore + 8)

sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(nadir_depths[0], nadir_depths[-1]))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cb.set_label("Nadir depth [mm]")
ax2.grid(True, linestyle="--", alpha=0.4)

fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved to {OUT}")
