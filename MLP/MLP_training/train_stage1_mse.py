"""Stage-1 MSE curriculum training: learn a stable mean penetration trend.

Curriculum rationale
--------------------
Each injection trajectory is represented by four fitted log-parameters
(log_k_sqrt, log_k_quarter, log_t0, log_s) from a 1D sigmoid-blended
spray model:

    S(t) = (1 - w(t)) * k_sqrt * sqrt(t)  +  w(t) * k_quarter * t^0.25
    w(t) = sigmoid((t - t0) / s)

Stage-1 trains on *representative* rows — one row per unique operating
condition selected by proximity to the group median penetration at 5 ms.
Using 5 ms (far-time) instead of an earlier timestamp reduces sensitivity
to transient noise when picking the "average" trajectory.

The representative-row strategy deliberately produces a *balanced* view
of condition space: each operating condition contributes exactly one sample
regardless of how many raw repeated measurements were taken. This keeps the
Stage-1 optimiser from overweighting densely sampled conditions.

Architecture: 2 output heads (mu, log_var).
At Stage-1 only the mu head is meaningfully supervised. The log_var head is
regularised toward a prior (var_reg_weight) rather than trained on variance
data, because representative median rows do not provide heteroscedastic
supervision signal. This keeps the output interface compatible with Stage-2
NLL fine-tuning without introducing unstable early variance learning.

Feature scaling strategy
------------------------
Unscaled inputs produce a flat-ellipse loss landscape that impedes gradient
descent. Chosen scheme:
- Time: fixed min-max over [0, 5] ms  →  time_norm in [0, 1].
- Pressures: log(P_inj), log(P_ch), log(ΔP).  Log-scaling compresses the
  order-of-magnitude range and preserves the physically relevant ratios.
- Other continuous inputs (tilt, plumes, diameter, duration, backpressure):
  z-score fitted on the training split only; scaler_state is saved and
  reused at Stage-2 and inference without refit.

Loss terms
----------
    L = MSE(mu, y)
      + λ_var * ||log_var - prior||²       # anchor variance head
      + λ_d1 * max(0, -dmu/dt)²            # monotonicity: S non-decreasing
      + λ_d2 * max(0,  d²mu/dt²)² * gate   # concavity after d2_start_ms

The d2 gate is a soft sigmoid that activates after ~0.5 ms, allowing
physically plausible early-time acceleration while enforcing decelerating
growth at later times.

Note: training loss typically exceeds validation loss because (1) dropout
is active during training, (2) training trajectories span more extreme
conditions due to random splitting, and (3) penalty terms are included in
the reported training figure but not in the test figure.

Entry point: python MLP/MLP_training/train_stage1_mse.py
"""

from __future__ import annotations

from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainers.stage1 import Stage1Trainer

if __name__ == "__main__":
    Stage1Trainer().run()
