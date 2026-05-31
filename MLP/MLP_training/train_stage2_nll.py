"""Stage-2 NLL fine-tuning: calibrate heteroscedastic uncertainty.

Why a second stage?
-------------------
Stage-1 sees only representative median trajectories — one sample per
operating condition. These cannot teach the model how variable penetration
actually is across repeated injections under the same condition. Without this
signal the log_var head is unreliable even though the mu head is well trained.

Stage-2 switches to *filtered rows* — all trajectory measurements that pass
quality cuts — so the NLL optimiser sees the full within-condition variance.
The Stage-1 scaler state is reused without refit so the z-scored feature
space is identical between stages; the Stage-1 checkpoint provides a warm
start for the mu head.

NLL loss and its intuition
--------------------------
The Gaussian NLL for one observation y is:

    L_NLL = 0.5 * [log σ²(x,t)  +  (y − μ(x,t))² / σ²(x,t)]

The two terms compete:
- The residual term (y − μ)² / σ² penalises being wrong while claiming small
  uncertainty. Inflating σ² deflates this term.
- The log σ² term penalises predicting excessively large uncertainty.

Together they force the model to predict small variance where trajectories are
repeatable and large variance where they genuinely scatter. This is essential
for downstream calibration: the model must be neither overconfident near the
spray tip nor underconfident in the dense-data region.

Stage-2 also inherits the d1 (monotonicity) and d2 (concavity) shape
penalties from Stage-1 to prevent the NLL optimiser from sacrificing physical
plausibility in exchange for a locally lower uncertainty estimate.

Note: the optional anchor penalty (--stage2-ablation mu_anchor or
mu_sigma_anchor) prevents the predicted mean and std from inflating in the
data-sparse injection onset region, where the NLL optimiser has little
correction signal.

Entry point: python MLP/MLP_training/train_stage2_nll.py <stage1_run_dir>
"""

from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainers.stage2 import Stage2Trainer

if __name__ == "__main__":
    Stage2Trainer().run()
