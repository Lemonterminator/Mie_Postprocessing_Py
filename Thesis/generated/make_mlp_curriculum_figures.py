"""Generate the Ch.04 method diagrams.

Two figures:
  1. mlp_training_curriculum.png — how Datasets #1/#2/#3 feed the three
     training stages (losses, warm starts, teacher distillation).
  2. mlp_architecture_overview.png — 7-feature input, [512,512,128] Swish
     trunk, (mu, log sigma^2) heads, and the A-scaling reconstruction.

Content mirrors Sections 4.x of 04_trajectory_surrogate_screening.tex.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_ROOT / "Thesis" / "images"

DATA_COLOR = "#EAF1F8"
DATA_EDGE = "#2F6FA3"
STAGE_COLOR = "#FDF3E7"
STAGE_EDGE = "#9A5B38"
OUT_COLOR = "#EDF5ED"
OUT_EDGE = "#4E7B4E"
ARROW = "#444444"


def box(ax, cx, cy, w, h, text, fc, ec, fontsize=8.6, lw=1.4):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, linespacing=1.35)


def arrow(ax, p0, p1, text=None, style="-|>", color=ARROW, lw=1.6, ls="-",
          text_dxy=(0.0, 0.012), fontsize=7.8, connectionstyle="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=connectionstyle,
                                 shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + text_dxy[0], (p0[1] + p1[1]) / 2 + text_dxy[1]
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=fontsize,
                color="#333333", style="italic")


def make_curriculum(out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    dx, sx = 0.215, 0.625          # dataset / stage column centers
    y1, y2, y3 = 0.82, 0.50, 0.18  # stage rows
    dw, dh = 0.36, 0.20            # dataset box size
    sw, sh = 0.40, 0.22            # stage box size

    # ── dataset column ──────────────────────────────────────────────────
    box(ax, dx, y1, dw, dh,
        "Dataset #3 — representative subset\n"
        "599 q1 fits, one per operating\n"
        "condition (median at 5 ms)",
        DATA_COLOR, DATA_EDGE)
    box(ax, dx, y2, dw, dh,
        "Dataset #3 — filtered q1 archive\n"
        "71,700 fitted trajectories\n"
        "(inter-plume spread retained)",
        DATA_COLOR, DATA_EDGE)
    box(ax, dx, y3, dw, dh,
        "Dataset #1 — raw CDF points\n"
        "regime split by raw coverage:\n"
        "reliable ≥70% | uncertain 70–20%\n"
        "| teacher-only <20%",
        DATA_COLOR, DATA_EDGE, fontsize=8.0)

    # ── stage column ────────────────────────────────────────────────────
    box(ax, sx, y1, sw, sh,
        "Stage 1 — MSE baseline\n"
        r"MSE on $\hat{S}=S/A$ + shape penalties"
        "\n(monotonicity, gated concavity)",
        STAGE_COLOR, STAGE_EDGE)
    box(ax, sx, y2, sw, sh,
        "Stage 2 — heteroscedastic NLL\n"
        r"Gaussian NLL on $\hat{S}$: learns $(\hat{\mu},\log\hat{\sigma}^2)$"
        "\n→ clean teacher",
        STAGE_COLOR, STAGE_EDGE)
    box(ax, sx, y3, sw, sh,
        "Stage 3 — censoring-aware distillation\n"
        "raw NLL (reliable) + KD from teacher\n"
        r"(mse$_\mu$ + $\lambda_\sigma$ mse$_{\log\sigma^2}$, $\lambda_\sigma{=}5$)",
        STAGE_COLOR, STAGE_EDGE, fontsize=8.2)

    # data → stage arrows
    for y in (y1, y2, y3):
        arrow(ax, (dx + dw / 2, y), (sx - sw / 2, y))

    # warm-start arrows between stages
    arrow(ax, (sx + sw / 2 + 0.01, y1), (sx + sw / 2 + 0.01, y2),
          "warm start +\nscaler states", connectionstyle="arc3,rad=-0.55",
          text_dxy=(0.115, -0.01))
    arrow(ax, (sx + sw / 2 + 0.01, y2), (sx + sw / 2 + 0.01, y3),
          "warm start", connectionstyle="arc3,rad=-0.55", text_dxy=(0.105, -0.005))
    # frozen teacher arrow (stage2 → stage3 KD)
    arrow(ax, (sx - sw / 2 - 0.012, y2 - sh / 2 + 0.02), (sx - sw / 2 - 0.012, y3 + sh / 2 - 0.02),
          "frozen teacher\n$(\\mu,\\sigma)$", color=STAGE_EDGE, ls="--",
          connectionstyle="arc3,rad=0.45", text_dxy=(-0.085, -0.005))

    # output box
    box(ax, sx, -0.13, 0.46, 0.13,
        "Production screening surrogate\n(5 seeds; deployed for impingement screening)",
        OUT_COLOR, OUT_EDGE)
    arrow(ax, (sx, y3 - sh / 2), (sx, -0.13 + 0.065))

    # shared physics note
    box(ax, dx, -0.13, dw, 0.13,
        "Shared across stages:\n"
        r"7 features; targets scaled by $A=\Delta P^{0.5}\rho_a^{-0.25}d_n^{0.5}$",
        "#F4F4F4", "#888888", fontsize=8.0)

    ax.set_ylim(-0.22, 1.0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out_png}")


def make_architecture_legacy(out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.6), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── input features ──────────────────────────────────────────────────
    feats = [
        r"$t_{\mathrm{norm}}$  (time, min–max to $[0,1]$)",
        r"$\theta_{\mathrm{tilt},z}$  (tilt angle)",
        r"$n_{\mathrm{plumes},z}$  (plume count)",
        r"$t_{i,z}$  (injection duration)",
        r"$P_{\mathrm{cb},z}$  (control backpressure)",
        r"$\log P_{\mathrm{inj},z}$  (residual pressure)",
        r"$\log P_{\mathrm{ch},z}$  (residual pressure)",
    ]
    fx, fw, fh = 0.155, 0.27, 0.082
    top, bot = 0.93, 0.27
    ys = [top - i * (top - bot) / 6 for i in range(7)]
    for y, t in zip(ys, feats):
        box(ax, fx, y, fw, fh, t, DATA_COLOR, DATA_EDGE, fontsize=8.0)
    ax.text(fx, top + 0.085, "7 input features (z-scored)", ha="center",
            fontsize=9.5, weight="bold")

    # ── trunk ───────────────────────────────────────────────────────────
    tx = 0.475
    widths = [("512", 0.30), ("512", 0.30), ("128", 0.20)]
    tys = [0.74, 0.55, 0.36]
    for (units, h), y in zip(widths, tys):
        box(ax, tx, y, 0.135, h, units, "#F3E9F5", "#7B4F8E", fontsize=10)
    ax.text(tx, 0.93, "MLP trunk\nSwish, dropout 0.3", ha="center",
            fontsize=9.5, weight="bold", linespacing=1.3)
    for y0, y1 in zip(tys[:-1], tys[1:]):
        arrow(ax, (tx, y0 - widths[0][1] / 2 + 0.02), (tx, y1 + widths[2][1] / 2 + 0.045), lw=1.4)
    for y in ys:
        arrow(ax, (fx + fw / 2, y), (tx - 0.075, 0.55), lw=0.8)

    # ── heads ───────────────────────────────────────────────────────────
    hx = 0.71
    box(ax, hx, 0.66, 0.13, 0.10, r"$\hat{\mu}$", "#FDF3E7", STAGE_EDGE, fontsize=11)
    box(ax, hx, 0.44, 0.13, 0.10, r"$\log\hat{\sigma}^2$", "#FDF3E7", STAGE_EDGE, fontsize=10)
    arrow(ax, (tx + 0.07, 0.42), (hx - 0.068, 0.66))
    arrow(ax, (tx + 0.07, 0.32), (hx - 0.068, 0.44))
    ax.text(hx, 0.80, "scaled output heads", ha="center", fontsize=9.5, weight="bold")

    # ── A-scaling reconstruction ────────────────────────────────────────
    ax_x = 0.71
    box(ax, ax_x, 0.13, 0.30, 0.11,
        r"amplitude prior  $A=\Delta P^{0.5}\,\rho_a^{-0.25}\,d_n^{0.5}$"
        "\n(computed from condition, not learned)",
        "#F4F4F4", "#888888", fontsize=8.2)
    px = 0.92
    box(ax, px, 0.55, 0.13, 0.16,
        r"$\mu_S = A\,\hat{\mu}$" "\n" r"$\sigma_S = A\,\hat{\sigma}$",
        OUT_COLOR, OUT_EDGE, fontsize=9.5)
    arrow(ax, (hx + 0.068, 0.66), (px - 0.068, 0.60))
    arrow(ax, (hx + 0.068, 0.44), (px - 0.068, 0.50))
    arrow(ax, (ax_x + 0.10, 0.13 + 0.058), (px - 0.03, 0.55 - 0.085),
          color="#888888", ls="--")
    ax.text(px, 0.70, "physical\nprediction (mm)", ha="center", fontsize=9.0,
            weight="bold", linespacing=1.3)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out_png}")


def make_architecture(out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def panel(x, y, w, h, fc, ec, lw=2.0, ls="-", z=1):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.018",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
        ax.add_patch(patch)
        return patch

    def label(x, y, text, size=12, weight="normal", color="#111827", ha="center",
              va="center", linespacing=1.2):
        ax.text(
            x, y, text, ha=ha, va=va, fontsize=size, weight=weight,
            color=color, linespacing=linespacing,
        )

    def flow(p0, p1, text=None, color="#334155", lw=2.2, ls="-", rad=0.0,
             text_offset=(0.0, 0.0), size=9.5):
        arrow(
            ax, p0, p1, text=text, color=color, lw=lw, ls=ls,
            connectionstyle=f"arc3,rad={rad}", text_dxy=text_offset,
            fontsize=size,
        )

    # Input feature block.
    label(0.245, 0.935, "Inputs", size=16, weight="bold")
    label(0.245, 0.895, "7-feature vector", size=9.8, color="#475569")
    panel(0.055, 0.615, 0.38, 0.245, DATA_COLOR, DATA_EDGE, lw=2.2)
    feature_text = (
        r"$t_{\mathrm{norm}}$" "\n"
        r"$\theta_{\mathrm{tilt},z}$,  $n_{\mathrm{plumes},z}$" "\n"
        r"$t_{i,z}$,  $P_{\mathrm{cb},z}$" "\n"
        r"$\log P_{\mathrm{inj},z}$,  $\log P_{\mathrm{ch},z}$"
    )
    label(0.08, 0.82, "z-scored / normalized", size=9.2, color=DATA_EDGE,
          ha="left", weight="bold")
    label(0.08, 0.705, feature_text, size=10.5, ha="left", linespacing=1.35)

    # Shared trunk.
    label(0.745, 0.935, "Trunk", size=16, weight="bold")
    label(0.745, 0.895, "Linear + LN + SiLU", size=9.8, color="#475569")
    panel(0.565, 0.615, 0.36, 0.245, "#F3E8FF", "#7C3AED", lw=2.2)
    layer_y = [0.81, 0.745, 0.68]
    layer_labels = ["512", "512", "128"]
    for y, txt in zip(layer_y, layer_labels):
        panel(0.625, y - 0.025, 0.24, 0.05, "#FBF7FF", "#7C3AED", lw=1.5)
        label(0.745, y, txt, size=12.5)
    for y0, y1 in zip(layer_y[:-1], layer_y[1:]):
        flow((0.745, y0 - 0.027), (0.745, y1 + 0.027), color="#6D28D9", lw=1.4)
    label(0.745, 0.627, "dropout = 0.30", size=8.8, color="#6D28D9", weight="bold")

    # A-scaled heads.
    label(0.745, 0.545, "A-scaled heads", size=15, weight="bold")
    label(0.745, 0.505, "trained in normalized space", size=8.8, color="#475569")
    panel(0.565, 0.395, 0.36, 0.075, "#FFF7ED", "#A16207", lw=1.9)
    label(0.745, 0.432, r"$\hat{\mu}$", size=14)
    panel(0.565, 0.295, 0.36, 0.075, "#FFF7ED", "#A16207", lw=1.9)
    label(0.745, 0.333, r"$\log \hat{\sigma}^{2}$", size=12.5)
    panel(0.565, 0.205, 0.36, 0.06, "#F8FAFC", "#94A3B8", lw=1.5, ls="--")
    label(0.745, 0.235, "onset aux (optional)", size=8.5, color="#64748B",
          linespacing=1.05)

    # Physical reconstruction.
    label(0.245, 0.545, "Condition prior", size=15, weight="bold")
    label(0.245, 0.505, "computed, not learned", size=8.8, color="#475569")
    panel(0.055, 0.335, 0.38, 0.135, "#F8FAFC", "#64748B", lw=1.7, ls="--")
    label(0.245, 0.402,
          r"$A=\Delta P^{0.5}\rho_a^{-0.25}d_n^{0.5}$",
          size=11.5, color="#334155")

    label(0.245, 0.255, "Physical output", size=13.5, weight="bold")
    label(0.245, 0.217, "millimetres", size=8.8, color="#475569")
    panel(0.055, 0.075, 0.38, 0.115, OUT_COLOR, OUT_EDGE, lw=2.2)
    label(0.245, 0.132, r"$\mu_S=A\,\hat{\mu}$     $\sigma_S=A\,\hat{\sigma}$",
          size=12.5)

    # Main flow; keep the arrows sparse so the small slide version stays legible.
    flow((0.435, 0.735), (0.565, 0.735), text="features",
         text_offset=(0.0, 0.02), size=8.5)
    flow((0.745, 0.395), (0.435, 0.155), color="#334155", rad=-0.15)
    flow((0.745, 0.295), (0.435, 0.115), color="#334155", rad=-0.10)

    label(0.50, 0.03,
          r"Primary channels: $(\hat{\mu},\log\hat{\sigma}^2)$; $A$ restores mm scale.",
          size=8.9, color="#475569")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> None:
    make_curriculum(IMAGE_DIR / "mlp_training_curriculum.png")
    make_architecture(IMAGE_DIR / "mlp_architecture_overview.png")


if __name__ == "__main__":
    main()
