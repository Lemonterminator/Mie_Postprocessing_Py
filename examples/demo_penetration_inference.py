"""
Demo: Run inference using the trained penetration MLP and plot mean ± std.

Edit RUN_DIR and params below to your case, then:
  python demo_penetration_inference.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt

from mie_postprocessing.inference_penetration import (
    load_run, frames_to_time, predict_time_range
)


# Path to a training run directory containing best_model.pt, scalers.json, model_config.json
RUN_DIR = Path(r"runs_mlp/penetration_frame")

# Physical parameters (units must match training):
# - chamber_pressure [bar]
# - injection_pressure [bar]
# - injection_duration [microseconds]
# - control_backpressure [bar]
params = dict(
    chamber_pressure=15,
    injection_pressure=2200,
    injection_duration=520,
    control_backpressure=1,
)


def main():
    run = load_run(RUN_DIR)
    # Build time array from frames using the run's frame rate
    fr = float(run['cfg'].get('frame_rate_hz', 34_000.0))
    time_s = frames_to_time(start_frame=0, end_frame=49, frame_rate_hz=fr)
    # Predict corrected penetration mean/std
    out = predict_time_range(run, params, time_s=time_s, output_space='corrected')
    t = out['time_s']; mean = out['mean']; std = out['std']

    # Plot mean and ±1 std band
    plt.figure(figsize=(7, 4))
    plt.plot(t, mean, label='mean', lw=2)
    plt.fill_between(t, mean - std, mean + std, alpha=0.25, label='±1 std')
    plt.xlabel('Time (s)')
    plt.ylabel('Penetration (corrected units)')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

