"""Plot the legacy two-regime branch-amplitude ratio at transition time.

The script scans clean CDF fit rows under ``MLP/synthetic_data/*/cdf/clean``,
computes ``k_sqrt * sqrt(t0) / (k_quarter * t0**0.25)`` for every valid row,
prints simple counts, and saves a histogram to
``MLP/figures/amplitude_ratio_dist.png``.

It is mainly a quick sanity check for archived sqrt-plus-quarter-root fits.
Current q1-only fits keep ``k_sqrt`` at a sentinel value, so their ratios are
expected to be near zero.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "MLP" / "synthetic_data"
DEFAULT_OUT_PATH = PROJECT_ROOT / "MLP" / "figures" / "amplitude_ratio_dist.png"


def collect_ratios(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Collect transition-time branch amplitude ratios from clean CDF fit rows."""
    results = []
    for nozzle_dir in data_dir.glob("*"):
        clean_dir = nozzle_dir / "cdf" / "clean"
        if not clean_dir.exists():
            continue

        for path in clean_dir.glob("*.csv"):
            try:
                df = pd.read_csv(path)
                if df.empty or "k_sqrt" not in df.columns:
                    continue
                for _, row in df.iterrows():
                    # Skip rows that cannot define the transition-time amplitudes.
                    k_sqrt = row.get("k_sqrt", np.nan)
                    k_quarter = row.get("k_quarter", np.nan)
                    t0 = row.get("t0", np.nan)

                    if pd.notna(k_sqrt) and pd.notna(k_quarter) and pd.notna(t0) and t0 > 0:
                        # Compare the two model branches at exactly t0, where
                        # the sigmoid blend switches branches.
                        amp_sqrt = k_sqrt * np.sqrt(t0)
                        amp_quarter = k_quarter * (t0 ** 0.25)
                        ratio = amp_sqrt / amp_quarter if amp_quarter > 0 else np.nan
                        results.append(
                            {
                                "nozzle": nozzle_dir.name,
                                "k_sqrt": k_sqrt,
                                "k_quarter": k_quarter,
                                "t0": t0,
                                "amp_sqrt": amp_sqrt,
                                "amp_quarter": amp_quarter,
                                "ratio": ratio,
                            }
                        )
            except Exception:
                # Keep this one-off audit resilient to a damaged per-folder CSV.
                continue
    return pd.DataFrame(results).dropna(subset=["ratio"])


def plot_ratios(df_res: pd.DataFrame, out_path: Path = DEFAULT_OUT_PATH) -> None:
    """Write the branch-amplitude ratio histogram."""
    plt.figure(figsize=(8, 6))
    plt.hist(df_res["ratio"], bins=np.linspace(0, 2, 100), alpha=0.7, color="steelblue", edgecolor="black")
    plt.axvline(0.5, color="r", linestyle="--", label="Ratio = 0.5")
    plt.axvline(df_res["ratio"].median(), color="green", linestyle="-", label=f"Median = {df_res['ratio'].median():.2f}")
    plt.xlabel("Amplitude Ratio ($k_{sqrt}\\sqrt{t_0} / k_{quarter}t_0^{1/4}$)")
    plt.ylabel("Count")
    plt.title("Distribution of Branch Amplitude Ratio at Transition $t_0$")
    plt.legend()
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    df_res = collect_ratios()
    total_valid = len(df_res)
    count_less_than_half = len(df_res[df_res["ratio"] < 0.5])
    percentage = (count_less_than_half / total_valid) * 100 if total_valid else float("nan")

    print(f"Total accepted samples (clean fits): {total_valid}")
    print(f"Samples where sqrt amp < 0.5 * quarter amp at t0: {count_less_than_half} ({percentage:.1f}%)")
    print(f"Median ratio: {df_res['ratio'].median():.3f}")

    plot_ratios(df_res)
    print(f"Saved distribution plot to {DEFAULT_OUT_PATH}")


if __name__ == "__main__":
    main()

