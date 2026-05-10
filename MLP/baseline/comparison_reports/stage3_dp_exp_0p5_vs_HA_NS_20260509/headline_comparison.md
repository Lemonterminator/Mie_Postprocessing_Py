# External eval (n=795k clean) — ΔP^0.5 vs baselines

| model | rmse_mm | mae_mm | bias_mm | p95_abs_err_mm | cov_1σ | cov_2σ |
|---|---|---|---|---|---|---|
| Hiroyasu-Arai calibrated | 9.974 | 7.686 | 0.625 | 20.493 | 0.536 | 0.886 |
| Naber-Siebers delay | 8.840 | 6.780 | -0.344 | 17.922 | 0.620 | 0.893 |
| Stage-3 MLP anchor_off | 6.696 | 4.875 | -1.149 | 14.118 | 0.745 | 0.979 |
| Stage-3 MLP anchor_off (ΔP^0.5, seed 42) | 6.231 | 4.546 | -0.893 | 13.231 | 0.774 | 0.978 |
| Stage-3 MLP anchor_off (ΔP^0.5, seed 17) | 6.136 | 4.494 | -0.612 | 13.023 | 0.793 | 0.986 |
| Stage-3 MLP anchor_off (ΔP^0.5, seed 99) | 5.948 | 4.383 | -0.074 | 12.533 | 0.792 | 0.985 |
| Stage-3 MLP anchor_off (ΔP^0.5, seed 7) | 6.061 | 4.436 | -0.637 | 12.766 | 0.772 | 0.979 |
| Stage-3 MLP raw_reliable_no_kd (ΔP^0.5, seed 2024) | 6.682 | 5.077 | -1.976 | 13.594 | 0.737 | 0.976 |
| Stage-3 MLP ΔP^0.5 (5-seed mean) | 6.211 | 4.587 | -0.838 | 13.029 | 0.774 | 0.981 |