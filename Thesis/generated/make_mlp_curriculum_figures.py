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


CUR_FIG_W, CUR_FIG_H = 12.2, 6.1   # curriculum figure canvas (inches)
CUR_Y0, CUR_Y1 = -0.22, 1.0        # curriculum y-limits
BOX_PAD = 0.012                    # FancyBboxPatch pad: drawn edge sits this
                                   # far OUTSIDE the given rect, on every side


def box(ax, cx, cy, w, h, text, fc, ec, fontsize=9.4, lw=1.5):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad={BOX_PAD}",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, linespacing=1.4)


def _pt2y(pt: float) -> float:
    """Convert a text height in points into curriculum-figure y-data units."""
    return pt * (CUR_Y1 - CUR_Y0) / (CUR_FIG_H * 72.0)


def titled_box(ax, cx, cy, w, h, title, body, fc, ec,
               title_size=9.2, body_size=8.3, lw=1.5):
    """Cell with a bold heading line over a smaller detail block.

    Title and body are stacked as one block centred in the box, so cells with
    two and three body lines stay optically aligned with each other.
    """
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad={BOX_PAD}",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    title_h = _pt2y(title_size * 1.5)
    n_body = (body.count("\n") + 1) if body else 0
    body_h = _pt2y(body_size * 1.42) * n_body
    top = cy + (title_h + body_h) / 2
    ax.text(cx, top - title_h / 2, title, ha="center", va="center",
            fontsize=title_size, weight="bold")
    if body:
        ax.text(cx, top - title_h, body, ha="center", va="top",
                fontsize=body_size, linespacing=1.42)


def arrow(ax, p0, p1, text=None, style="-|>", color=ARROW, lw=1.7, ls="-",
          text_dxy=(0.0, 0.012), fontsize=8.4, connectionstyle="arc3,rad=0.0",
          ha="center", mutation_scale=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                                 mutation_scale=mutation_scale,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=connectionstyle,
                                 shrinkA=2, shrinkB=2))
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + text_dxy[0], (p0[1] + p1[1]) / 2 + text_dxy[1]
        ax.text(mx, my, text, ha=ha, va="bottom", fontsize=fontsize,
                color="#333333", style="italic", linespacing=1.3)


def make_curriculum(out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(CUR_FIG_W, CUR_FIG_H), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(CUR_Y0, CUR_Y1)
    ax.axis("off")

    # Column geometry. Every drawn edge sits BOX_PAD outside the rect below,
    # so the left column starts far enough in to keep that padded edge inside
    # the axes, and the data/stage gutter is sized to hold the teacher label.
    dx, sx = 0.1605, 0.684            # dataset / stage column centers
    y1, y2, y3 = 0.845, 0.50, 0.155   # stage rows
    dw, dh = 0.275, 0.215             # dataset box size
    sw, sh = 0.325, 0.215             # stage box size

    # ── dataset column ──────────────────────────────────────────────────
    titled_box(ax, dx, y1, dw, dh,
               "Dataset #3 — representative subset",
               "1,800 SGQR fits across 599 operating\n"
               "conditions (median at 5 ms)",
               DATA_COLOR, DATA_EDGE)
    titled_box(ax, dx, y2, dw, dh,
               "Dataset #3 — filtered SGQR archive",
               "71,700 fitted trajectories\n"
               "(inter-plume spread retained)",
               DATA_COLOR, DATA_EDGE)
    titled_box(ax, dx, y3, dw, dh,
               "Dataset #1 — raw CDF points",
               "regime split by raw coverage:\n"
               "reliable ≥70% | uncertain 70–20%\n"
               "| teacher-only <20%",
               DATA_COLOR, DATA_EDGE)

    # ── stage column ────────────────────────────────────────────────────
    titled_box(ax, sx, y1, sw, sh,
               "Stage 1 — MSE baseline",
               r"MSE on $\hat{S}=S/A$ + shape penalties"
               "\n(monotonicity, gated concavity)",
               STAGE_COLOR, STAGE_EDGE)
    titled_box(ax, sx, y2, sw, sh,
               "Stage 2 — heteroscedastic NLL",
               r"Gaussian NLL on $\hat{S}$: learns $(\hat{\mu},\log\hat{\sigma}^2)$"
               "\n→ clean teacher",
               STAGE_COLOR, STAGE_EDGE)
    titled_box(ax, sx, y3, sw, sh,
               "Stage 3 — censoring-aware distillation",
               "raw NLL (reliable) + KD from teacher\n"
               r"(mse$_\mu$ + $\lambda_\sigma$ mse$_{\log\sigma^2}$, $\lambda_\sigma{=}5$)",
               STAGE_COLOR, STAGE_EDGE)

    # data → stage arrows
    for y in (y1, y2, y3):
        arrow(ax, (dx + dw / 2 + BOX_PAD, y), (sx - sw / 2 - BOX_PAD, y))

    # warm-start arrows between stages
    arrow(ax, (sx + sw / 2 + 0.014, y1), (sx + sw / 2 + 0.014, y2),
          "warm start +\nscaler states", connectionstyle="arc3,rad=-0.5",
          text_dxy=(0.085, -0.012))
    arrow(ax, (sx + sw / 2 + 0.014, y2), (sx + sw / 2 + 0.014, y3),
          "warm start", connectionstyle="arc3,rad=-0.5", text_dxy=(0.078, -0.005))

    # Frozen-teacher arrow (stage 2 → stage 3 KD). Kept straight and drawn just
    # inside the stage column so it reads as leaving stage 2 and entering
    # stage 3; the label is then right-aligned beside it, which leaves the
    # whole gutter as clearance from the dataset column.
    teacher_arrow_x = sx - sw / 2 + 0.020
    arrow(ax, (teacher_arrow_x, y2 - sh / 2 - BOX_PAD), (teacher_arrow_x, y3 + sh / 2 + BOX_PAD),
          "frozen teacher $(\\mu,\\sigma)$\nuncertain / teacher-only KD",
          color=STAGE_EDGE, ls="--",
          connectionstyle="arc3,rad=0.0", text_dxy=(-0.016, -0.032), ha="right",
          fontsize=8.0)

    # output box
    titled_box(ax, sx, -0.13, 0.46, 0.135,
               "Production screening surrogate",
               "(5 seeds; deployed for impingement screening)",
               OUT_COLOR, OUT_EDGE)
    arrow(ax, (sx, y3 - sh / 2 - BOX_PAD), (sx, -0.13 + 0.135 / 2 + BOX_PAD))

    # shared physics note
    titled_box(ax, dx, -0.13, dw, 0.135,
               "Shared across stages",
               r"7 features; targets scaled by"
               "\n"
               r"$A=\Delta P^{0.5}\rho_a^{-0.25}d_n^{0.5}$",
               "#F4F4F4", "#888888")

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

    def panel(x0, y0, x1, y1, fc, ec, lw=2.0, ls="-", z=1, pad=0.014,
              rounding=0.020):
        """Rounded panel whose drawn edge lands exactly on the given rect.

        FancyBboxPatch inflates the rect it is handed by ``pad`` on all four
        sides, so the rect is inset by the same amount first. Every coordinate
        passed to this function is therefore a true visual edge, which lets the
        spacing arithmetic below be read literally.
        """
        patch = FancyBboxPatch(
            (x0 + pad, y0 + pad), (x1 - x0) - 2 * pad, (y1 - y0) - 2 * pad,
            boxstyle=f"round,pad={pad},rounding_size={rounding}",
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
             text_offset=(0.0, 0.0), size=9.5, mutation_scale=14):
        arrow(
            ax, p0, p1, text=text, color=color, lw=lw, ls=ls,
            connectionstyle=f"arc3,rad={rad}", text_dxy=text_offset,
            fontsize=size, mutation_scale=mutation_scale,
        )

    # ── layout grid ─────────────────────────────────────────────────────
    # Two equal-width columns. The gutter is sized to hold the "features"
    # label between the panels rather than on top of one of them.
    L0, L1 = 0.040, 0.4425             # left column: inputs / prior / output
    R0, R1 = 0.5575, 0.960             # right column: trunk / heads
    LC, RC = (L0 + L1) / 2, (R0 + R1) / 2

    # Row anchors. Every row repeats the same title -> subtitle -> panel
    # rhythm (0.036 then 0.032), so the two columns share one baseline grid.
    # The last row bottoms out near y=0.03: savefig(bbox_inches="tight") crops
    # to the axes rectangle, not to the artists, so slack left below the
    # footer would survive into the PNG as a dead band.
    R1_TITLE, R1_SUB, R1_TOP, R1_BOT = 0.955, 0.919, 0.887, 0.622
    INPUT_BOT = R1_BOT - 0.03           # Inputs panel only: extra room below
    R2_TITLE, R2_SUB, R2_TOP = 0.556, 0.520, 0.488
    R3_TITLE, R3_SUB, R3_TOP, R3_BOT = 0.313, 0.277, 0.245, 0.120
    TITLE_SIZE, SUB_SIZE = 14.5, 9.0

    def heading(cx, y_title, y_sub, title, subtitle):
        label(cx, y_title, title, size=TITLE_SIZE, weight="bold")
        label(cx, y_sub, subtitle, size=SUB_SIZE, color="#475569")

    # ── inputs ──────────────────────────────────────────────────────────
    heading(LC, R1_TITLE, R1_SUB, "Inputs", "7-feature vector")
    panel(L0, INPUT_BOT, L1, R1_TOP, DATA_COLOR, DATA_EDGE, lw=2.2)
    feature_text = (
        r"$t_{\mathrm{norm}}$" "\n"
        r"$\theta_{\mathrm{tilt},z}$,  $n_{\mathrm{plumes},z}$" "\n"
        r"$t_{i,z}$,  $P_{\mathrm{cb},z}$" "\n"
        r"$\log P_{\mathrm{inj},z}$,  $\log P_{\mathrm{ch},z}$"
    )
    label(L0 + 0.032, 0.845, "z-scored / normalized", size=9.2, color=DATA_EDGE,
          ha="left", weight="bold")
    # va="top" so the four rows grow downwards from a fixed line, keeping the
    # block optically centred in the panel.
    ax.text(L0 + 0.032, 0.810, feature_text, ha="left", va="top",
            fontsize=10.5, color="#111827", linespacing=1.35)

    # ── shared trunk ────────────────────────────────────────────────────
    heading(RC, R1_TITLE, R1_SUB, "Trunk", "Linear + LN + SiLU")
    panel(R0, R1_BOT, R1, R1_TOP, "#F3E8FF", "#7C3AED", lw=2.2)
    layer_h, layer_gap = 0.042, 0.030
    lx0, lx1 = R0 + 0.068, R1 - 0.068
    layer_tops = [R1_TOP - 0.017 - i * (layer_h + layer_gap) for i in range(3)]
    for top, txt in zip(layer_tops, ["512", "512", "128"]):
        panel(lx0, top - layer_h, lx1, top, "#FBF7FF", "#7C3AED", lw=1.4,
              pad=0.007, rounding=0.011)
        label(RC, top - layer_h / 2, txt, size=12.5)
    # Connectors live in the gaps between layers, so they are fully visible.
    for top, nxt in zip(layer_tops[:-1], layer_tops[1:]):
        flow((RC, top - layer_h), (RC, nxt), color="#6D28D9", lw=1.4,
             mutation_scale=10)
    label(RC, layer_tops[-1] - layer_h - 0.029, "dropout = 0.30", size=8.8,
          color="#6D28D9", weight="bold")

    # ── A-scaled heads ──────────────────────────────────────────────────
    heading(RC, R2_TITLE, R2_SUB, "A-scaled heads", "trained in normalized space")
    head_h, head_gap = 0.070, 0.022
    mu_top = R2_TOP
    sg_top = mu_top - head_h - head_gap
    aux_top = sg_top - head_h - head_gap
    panel(R0, mu_top - head_h, R1, mu_top, "#FFF7ED", "#A16207", lw=1.9)
    label(RC, mu_top - head_h / 2, r"$\hat{\mu}$", size=14)
    panel(R0, sg_top - head_h, R1, sg_top, "#FFF7ED", "#A16207", lw=1.9)
    label(RC, sg_top - head_h / 2, r"$\log \hat{\sigma}^{2}$", size=12.5)
    panel(R0, aux_top - 0.056, R1, aux_top, "#F8FAFC", "#94A3B8", lw=1.5,
          ls="--", pad=0.010, rounding=0.014)
    label(RC, aux_top - 0.028, "onset aux (optional)", size=8.5,
          color="#64748B", linespacing=1.05)

    # ── condition prior ─────────────────────────────────────────────────
    heading(LC, R2_TITLE, R2_SUB, "Condition prior", "computed, not learned")
    prior_bot = R2_TOP - 0.105
    panel(L0, prior_bot, L1, R2_TOP, "#F8FAFC", "#64748B", lw=1.7, ls="--")
    label(LC, (prior_bot + R2_TOP) / 2,
          r"$A=\Delta P^{0.5}\rho_a^{-0.25}d_n^{0.5}$",
          size=11.5, color="#334155")

    # ── physical reconstruction ─────────────────────────────────────────
    heading(LC, R3_TITLE, R3_SUB, "Physical output", "millimetres")
    panel(L0, R3_BOT, L1, R3_TOP, OUT_COLOR, OUT_EDGE, lw=2.2)
    label(LC, (R3_BOT + R3_TOP) / 2,
          r"$\mu_S=A\,\hat{\mu}$     $\sigma_S=A\,\hat{\sigma}$", size=12.5)

    # ── flow; kept sparse so the small slide version stays legible ──────
    mid_y = (R1_TOP + R1_BOT) / 2
    # The gutter is wide enough for this label to sit clear of both panels.
    flow((L1, mid_y), (R0, mid_y), text="features",
         text_offset=(0.0, 0.014), size=7.8)
    # Both head arrows leave the LEFT edge of their box and descend the gutter
    # to land on the RIGHT edge of the output panel: they never cross a box
    # they do not belong to, and they stay right of the "Physical output"
    # heading, which reaches x~0.40.
    # Landing heights straddle the panel mid-line so neither head hits a
    # rounded corner; the differing radii keep the two paths readable as two.
    flow((R0, mu_top - head_h / 2), (L1, R3_BOT + 0.080),
         color="#334155", rad=0.18)
    flow((R0, sg_top - head_h / 2), (L1, R3_BOT + 0.036),
         color="#334155", rad=0.28)

    label(0.50, 0.038,
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
