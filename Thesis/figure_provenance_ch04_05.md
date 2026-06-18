# 图片溯源审查表 — 第 04/05 章 (Figure Provenance Audit — Ch.04 & Ch.05)

目的：记录 `Thesis/latex/sections_en/04_trajectory_surrogate_screening.tex` 和
`05_results.tex` 中出现的、**非 legacy 模型、非 CV(图像分割/光流/Mie 散射)管线**的图片
分别由哪段代码生成，便于审查与精修。

**范围说明**
- ✅ **In scope**（本表 §1 / §2）：当前代理模型(MLP / Stage1-3 / SVGP / residual family-head)与
  数据筛查(CDF 拟合 / 截尾 / 稀疏支撑)产生的图。
- ⛔ **Excluded**（§3）：来自相机原始帧 / penetration_compare / impingement 渲染等 CV 侧的插图，按要求不深追。
- ⚠️ **Unresolved**（§4）：仓库中**找不到**生成字符串的图（疑似 notebook/手工导出后提交），需作者确认。

> 行号为搜索时刻的近似值，精修时请以当前文件为准。多数图先由脚本写入某个 run 目录或
> `Thesis/generated/...`，再被复制/改名进 `Thesis/images/`。

> **一键重建**：`.venv\Scripts\python Thesis\regenerate_thesis_figures.py`
> （先就地净化所有含 `BC########_HZ_` 机密名的图表输入，再按本表重跑全部生成器并
> 复制/改名到 tex 引用路径；`--list` 查看步骤，`--scan-only` 仅查泄漏）。

---

## §1 第 04 章 (04_trajectory_surrogate_screening.tex) — In scope

| # | 图片 (images/…) | 生成脚本 : 行 (函数) | 如何生成 | 内容 / 数据源 | 类别 |
|---|---|---|---|---|---|
| 1 | `fig_spatial_censoring_fraction_by_nozzle_clean.png` | `MLP/curve_fit/reports/audit_cdf_spatial_censoring.py:585` (`save_plots`) | `python … audit_cdf_spatial_censoring.py --plots` | 各喷嘴族空间右截尾比例条形图（raw vs clean vs 估计 FOV 上限） | 数据筛查 |
| 2 | `fig_q1_param_scaling.png` | `MLP/curve_fit/reports/fit_diagnostics.py:311` (`diag_param_distributions`) | `python … fit_diagnostics.py`（或 `fit_raw_data.main()` 调用） | q1 拟合参数(k_quarter,t0,s) vs ΔP 三联散点 + 经验幂律指数 | 当前模型 |
| 3 | `fig_q1_residual_structure.png` | `MLP/curve_fit/reports/fit_diagnostics.py:437` (`diag_residual_structure`) | 同上 | q1 拟合的时窗残差结构与异方差 σ(t) | 当前模型 |
| 4 | `fig_q1_identifiability.png` | `MLP/curve_fit/reports/fit_diagnostics.py:580` (`diag_identifiability`) | 同上 | 逐参数标准误(对数) + 参数相关矩阵热图(Jacobian) | 当前模型 |
| 5 | `filter_survival_cdf_by_nozzle.png` | `MLP/curve_fit/reports/summarize_filter_survival.py:94` (`plot_cdf_survival`) | `python … summarize_filter_survival.py` | 各喷嘴 CDF 拟合-过滤存活率/RMSE 成功率（源 `fit_report.csv`） | 数据筛查 |
| 6 | `fig_sparse_support_topology.png` | 数据来自 `MLP/gradient_stability_diagnostics.py`；出图入口 `Thesis/slides/legacy_notebook_sources/slides_sparse_feature_instability/export_sparse_feature_instability_figures.py:234`（写 `support_topology.png`，后改名加 `fig_` 前缀）。**⚠️ 依赖的旧 checkpoint（`distill_cdf_onset_20260331_194213` 等）已删除，无法重生成；图内无机密标签，现有 PNG 保留即可** | `python …/export_sparse_feature_instability_figures.py` | (chamber,injection) / (chamber,backpressure) 网格上稀疏支撑计数热图 | 数据筛查 |
| 7 | `fig_sparse_diameter_interpolants.png` | 同上脚本 `:246`（写 `diameter_interpolants.png`，后改名）。**⚠️ 同上，checkpoint 已删** | 同上 | 稀疏直径支撑下 MLP 在 t=0.85ms 的非单调插值/梯度伪影 | 数据筛查 |
| 8 | `fig_censor_fov_saturation.png` | `MLP/curve_fit/workflows/cdf_censoring_points.py:705` (`write_condition_plots`) | `python … cdf_censoring_points.py --plots` | 单工况时窗诊断：FOV 饱和触发的右截尾示例（**疑似从 per-condition 批量图中挑选/改名**） | 数据筛查 |
| 9 | `fig_censor_density_drop.png` | 同上 `:705` | 同上 | 单工况时窗诊断：密度塌缩型右截尾示例（**同上，挑选/改名**） | 数据筛查 |
| 10 | `time_windowed_exponents.png` | `MLP/curve_fit/reports/time_windowed_exponent_regression.py:221` (`plot_results`) | `python … time_windowed_exponent_regression.py` | 标度指数(ΔP,ρa,d)随时窗(0.2–1.5ms)演化 + Bernoulli/HA 参考线 | 数据筛查 |
| 11 | `stage2_loss_curves.png` | `MLP/MLP_training/efc/checkpoint.py:122` (`plot_loss_curves`)，由 `trainers/base.py:~310` 调用 | Stage-2 训练自动产出（`train_stage2_nll.py`） | Stage-2 NLL 训练的逐 epoch NLL/MAE/形状惩罚曲线 | 当前模型 |
| 12 | `stage2_toy_inference_2000bar_5bar.png` | `MLP/MLP_training/ablations/toy_inference_from_run.py:~96` (savefig) | 手动：`python … toy_inference_from_run.py <run_dir> --injection-pressure-bar 2000 --chamber-pressure-bar 5 --save-path …` | 2000bar/5bar 工况 Stage-2 预测轨迹 + ±1σ 带 | 当前模型 |
| 13 | `../generated/current/raw_coverage_heatmap.png` | `pipelines/audit/raw_coverage_heatmap.py:223/235` (`write_plot`) | `python … raw_coverage_heatmap.py`（或 `run_audit_pipeline.py`）；**直接引用 generated/current，未经 promote** | raw CDF 覆盖率按工况组×时窗热图 + 70%/20% 阈值等高线 | 数据筛查 |
| 14 | `raw_uncertain_vs_teacher_gaussian_2000bar_5bar.png` | **见 §4（未定位）** | — | 不确定区 raw 观测 vs Stage-2 teacher 高斯采样对比 | ⚠️ |
| 15 | `distillation_loss_curves.png` | `MLP/MLP_training/efc/checkpoint.py:122` (`plot_loss_curves`)，`trainers/base.py:~310` 调用 | Stage-3 蒸馏训练自动产出（`train_stage3_distillation_plus_raw_series.py`） | Stage-3 复合损失(raw NLL+KD MSE+onset+anchor+shape)与 MAE 曲线 | 当前模型 |
| 16 | `../generated/current/alpha_sensitivity_figure.png` | `pipelines/report/alpha_sensitivity_sweep.py` (`write_plot`) | `python … alpha_sensitivity_sweep.py`（或 `run_report_pipeline.py --alpha-sensitivity`） | KD 权重 λ_σ 敏感性扫描对 RMSE/coverage 的影响 | 数据筛查 |
| 17 | `hermite_crown_profile.png` | `piston/generate_hermite_figure.py:117` (savefig) | `python piston/generate_hermite_figure.py`（直接写入 Thesis/images） | 活塞顶面五次 Hermite 样条参数化(基准廓形 + 深度扫描)；几何图、非代理模型 | 当前(几何) |

---

## §2 第 05 章 (05_results.tex) — In scope

| # | 图片 (images/…) | 生成脚本 : 行 (函数) | 如何生成 | 内容 / 数据源 | 类别 |
|---|---|---|---|---|---|
| 1 | `representative_clean_fit.png` | **见 §4（未定位）** | — | 某代表性 clean 工况的 CDF penetration 拟合叠加图 | ⚠️ |
| 2 | `stage3_calibration_coverage.png` | `MLP/eval/calibration_coverage_audit.py:133/172` (`plot_calibration`/`run_audit`) | `python … calibration_coverage_audit.py [--eval-dir …] [--thesis-image-dir …]` | Stage-3 中心区间覆盖率可靠性图 + 按 σ 分位的 sharpness | 当前模型 |
| 3 | `neural_network_fit_results/pred_vs_actual_best.png` | `Thesis/generated/neural_network_fit_results_20260521/make_figures.py:171` (`save_pred_vs_actual`) | `python …/neural_network_fit_results_20260521/make_figures.py` | production MLP seed42 full-clean 预测 vs 实测散点(8万抽样) | 当前模型 |
| 4 | `baseline_comparison_20260521/full_clean_metric_bars_rmse_mae_p95.png` | `Thesis/generated/baseline_comparison_20260521/make_figures.py:296` (`bar_plot`) | `python …/baseline_comparison_20260521/make_figures.py` | HA/NS/production MLP/μ-anchor/SVGP 的 RMSE/MAE/P95 分组条形 | 当前模型* |
| 5 | `baseline_comparison_20260521/production_mlp_per_seed_rmse.png` | 同上 `make_figures.py:334` (`per_seed_rmse_plot`) | 同上 | production MLP 各 seed[7,17,42,99,2024] full-clean vs uncensored RMSE | 当前模型 |
| 6 | `neural_network_fit_results/latest_pred_vs_actual_seed99.png` | `MLP/eval/point_eval_figures.py:616` (`_save_pred_vs_actual`) | `python … point_eval_figures.py --eval-dir <seed99 eval>`（或评估管线自动） | seed99 评估目录的预测 vs 实测散点 | 当前模型 |
| 7 | `neural_network_fit_results/latest_residual_histogram_seed99.png` | `MLP/eval/point_eval_figures.py:619` (`_save_residual_hist`) | 同上 | seed99 残差直方图(±30mm 截断) + 偏差标注 | 当前模型 |
| 8 | `calibration_20260521/reliability_overlay.png` | `MLP/eval/calibration_diagnostics.py:338` (`plot_reliability`) | `python … calibration_diagnostics.py [--output-dir …]` | HA/NS/MLP(5seed)/SVGP 下尾覆盖率可靠性图 ±2% 带 | 当前模型* |
| 9 | `calibration_20260521/crps_sharpness_scatter.png` | `MLP/eval/calibration_diagnostics.py:357` (`plot_crps_sharpness`) | 同上 | 9 个模型的 CRPS vs 预测 σ(sharpness)散点 | 当前模型* |
| 10 | `lono_20260509/lono_rmse_by_fold.png` | `Thesis/generated/make_lono_figures.py:341` | `python Thesis/generated/make_lono_figures.py` | LONO 各折(N0,1,2,4,5) RMSE 分组条形，含 HA/NS 参考线 | 当前模型* |
| 11 | `lono_20260509/lono_coverage_by_fold.png` | `Thesis/generated/make_lono_figures.py:346` | 同上 | LONO 各折 1σ 经验覆盖率 + 名义参考线 | 当前模型 |
| 12 | `svgp_lono_comparison.png` | `Thesis/generated/make_lono_figures.py:337` (`plot_cross_arch_lono_combined`) | 同上 | MLP vs SVGP 跨架构 LONO 双面板(RMSE+coverage) | 当前模型 |
| 13 | `neural_network_fit_results/per_folder_rmse_best.png` | `Thesis/generated/neural_network_fit_results_20260521/make_figures.py:173` (`save_per_folder`) | `python …/make_figures.py` | seed42 各 folder 平均轨迹 RMSE 降序条形(源 `per_folder.csv`) | 当前模型 |
| 14 | `neural_network_fit_results/traj_best_nozzle7_T19.png` | `…/make_figures.py:184` (`save_trajectory`，模板 `traj_best_nozzle{n}_T{t}.png`) | 同上 | 最佳轨迹(最低 RMSE) 实测 vs 预测 + 1σ 带 | 当前模型 |
| 15 | `neural_network_fit_results/traj_worst_nozzle3_T3.png` | `…/make_figures.py:191` (`save_trajectory`，模板 `traj_worst_…`) | 同上 | 最差轨迹(最高 RMSE) 实测 vs 预测 + 1σ 带 | 当前模型 |
| 16 | `stage2_anchor_ablation.png` | `Thesis/generated/make_lono_figures.py:326` | `python Thesis/generated/make_lono_figures.py` | Stage-2 anchor 消融逐折 RMSE/coverage(源 stage2_lono_ablation CSV) | 当前模型 |
| 17 | `neural_network_fit_results/ablation_comparison_best.png` | `Thesis/generated/make_lono_figures.py:331` | 同上 | Stage-3 regime 消融(baseline/raw-reliable/blended/anchor-off)逐折对比 | 当前模型 |
| 18 | `…/stage3_kd_mse_mu_plus_sigma_overlay_baseline.png` | `MLP/curve_fit/reports/eval_stage3_thesis_sync.py:357` (`plot_overlay_with_baseline`) | `python … eval_stage3_thesis_sync.py` | teacher / Stage-3 baseline(fwd_kl) / kd_mse_mu_plus_sigma(w=5) 残差 vs truth | 当前模型 |
| 19 | `…/stage3_kd_mse_mu_plus_sigma_modeA_per_seed.png` | **见 §4（未定位）** | — | modeA 变体各 seed RMSE(数据 `stage3_mse_mu_plus_sigma_modeA_per_seed.csv`) | ⚠️ |
| 20 | `…/stage3_kd_mse_mu_plus_sigma_residual_vs_truth.png` | `MLP/curve_fit/reports/eval_stage3_thesis_sync.py:355` (`plot_residual_vs_truth_by_slice`) | 同 #18 | 按监督 regime / FOV 截尾标志的残差结构 | 当前模型 |
| 21 | `…/stage3_kd_mse_mu_plus_sigma_sigma_calibration.png` | `MLP/curve_fit/reports/eval_stage3_thesis_sync.py:356` (`plot_sigma_calibration`) | 同 #18 | student vs teacher σ 校准 hexbin + σ 残差直方图 | 当前模型 |
| 22 | `residual_family_head_20260608/mlp_family_conditioning_architecture.png` | **Graphviz 图，非绘图脚本**：`Thesis/slides/slides_residual_family_head_production/figs/mlp_family_conditioning_architecture.dot` | `dot -Tpng …/mlp_family_conditioning_architecture.dot -o …png`（旁有 `.svg`/`_graphviz` 变体） | family-conditioning 架构示意图 | 架构图 |
| 23 | `residual_family_head_20260608/delta_l2_sweep.png` | `MLP/MLP_training/ablations/plot_residual_family_head_production.py:246` (`plot_delta_l2_sweep`) | `python … plot_residual_family_head_production.py --figure delta-l2-sweep` | residual δ 收缩扫描：CDF/P50/Q1 RMSE + CDF ECE vs δ-L2 权重 | 当前模型 |
| 24 | `residual_family_head_20260608/residual_svgp_context_rmse_comparison.png` | `…/plot_residual_family_head_production.py:114` (`plot_svgp_context_rmse_comparison`) | `python … --figure svgp-context-rmse` | production SVGP / MLP residual FH / residual multitask SVGP 的 RMSE 对比 | 当前模型 |
| 25 | `residual_family_head_20260608/lono_modified_only_rmse_by_fold.png` | **见 §4（生成器未定位）**；数据 `lono_modified_only_comparison.csv` 来自 `MLP/MLP_training/ablations/run_family_head_sweep.py`（`--protocol lono_modified_only`） | — | modified-only 协议(N0 恒在训练集)逐折 RMSE：single vs family-head | ⚠️ |
| 26 | `residual_family_head_20260608/injector_comparison_dp050.png` | `MLP/MLP_training/ablations/compare_campaign_penetration.py:218` (`plot_comparison`) | `python … compare_campaign_penetration.py --dp-exp 0.50 --output-dir …` | 尺度归一化 S* vs 归一化 t*（动量标度 dp_exp=0.5）参考 vs 其他喷油器残差偏置 | 当前模型 |
| 27 | `residual_family_head_20260608/injector_scatter_all_nozzles.png` | `MLP/MLP_training/ablations/compare_campaign_penetration.py:250` (`plot_raw_scatter`，脚本内名 `scatter_all_nozzles.png`，部署时改名) | 同上 | 各 campaign 尺度归一化点按喷嘴着色散点 | 当前模型 |
| 28 | `residual_family_head_20260608/n0_fewshot_adaptation_curve.png` | `Thesis/generated/make_n0_fewshot_figure.py:86` (savefig) | `python Thesis/generated/make_n0_fewshot_figure.py` | Nozzle0 few-shot：RMSE/bias vs 适配预算 k + in-family LONO 带 + zero-shot 基线 | 当前模型 |

\* 这些图同时把 **HA / NS 经验关联式**作为参考基线画进同一张图 —— 图本身由**当前**评估管线产生，
HA/NS 只是对照线，不属于"legacy 模型图"。若你想把 HA/NS 也算作"legacy"需单独标注。

---

## §3 已排除：CV / 原始图像侧（按要求不深追）

| 图片 | 来源 | 说明 |
|---|---|---|
| `failure_mode_nozzle3/phantom_t3.png … t6.png` | `prepare_images.py`（TIF→PNG 转换，源 `Cam30660-FromFile_0003..0006.tif`，Phantom_Snapshots） | 相机原始帧，CV/原始数据 |
| `failure_mode_nozzle3/nozzle3_t16_plot.png` | `prepare_images.py`（`shutil.copy` 自 penetration_compare 输出） | CV 侧 penetration 提取的对比图 |
| `impingement_preview.png` | 未在 Python 内定位（疑似渲染/外部工具） | 撞壁可视化预览 |
| `impingement_summary.png` | 同上 | 撞壁可视化汇总 |

---

## §4 ⚠️ 未定位 / 需作者确认（仓库中查无生成字符串）

以下图片在整个仓库里**只出现在 `.tex` 引用中**，没有任何 `savefig` / 文件名字符串与之匹配 ——
极可能是 notebook 或一次性脚本生成后**手工改名提交**，建议补一段可复现脚本：

1. **`raw_uncertain_vs_teacher_gaussian_2000bar_5bar.png`**（Ch04）
   - 命名与 `stage2_toy_inference_2000bar_5bar.png` 平行，疑似同一 toy/teacher 可视化的衍生；
     可考虑并入 `toy_inference_from_run.py` 或新增一段 teacher-高斯采样对比脚本。
2. **`representative_clean_fit.png`**（Ch05）
   - 最接近的现存逻辑：`MLP/curve_fit/reports/analyze_cdf_fit_bias.py`（`plot_representative_cases`，
     输出 `representative_plume_panels.png` / `representative_cases.csv`）——但文件名不一致，需确认是否就是它改名而来。
3. **`stage3_kd_mse_mu_plus_sigma_modeA_per_seed.png`**（Ch05）
   - 数据 `Thesis/generated/current/stage3_mse_mu_plus_sigma_modeA_per_seed.csv` 存在；modeA 5-seed 评估由
     `MLP/eval/run_cross_dataset_lv2_lv3_eval.py`（`production_mlp_modeA_5seed`）产出，但**绘图脚本未找到**。
4. **`residual_family_head_20260608/lono_modified_only_rmse_by_fold.png`**（Ch05，§2 #25）
   - 数据(`lono_modified_only_comparison.csv`)可复现，但出图脚本未定位 ——
     `make_lono_figures.py` 只产出 `lono_20260509/*` 与 `stage2_anchor_ablation`/`ablation_comparison_best`，
     不含此图；疑似早期版本或 notebook 生成。

---

## §5 按"生成脚本"分组的复现速查（一次跑可重建一批）

- `MLP/curve_fit/reports/fit_diagnostics.py` → §1 #2,3,4
- `MLP/curve_fit/workflows/cdf_censoring_points.py --plots` → §1 #8,9
- `…/export_sparse_feature_instability_figures.py` → §1 #6,7
- `MLP/MLP_training/efc/checkpoint.py`（训练时自动） → §1 #11,15
- `pipelines/audit/raw_coverage_heatmap.py` / `pipelines/report/alpha_sensitivity_sweep.py` → §1 #13,16（generated/current）
- `Thesis/generated/neural_network_fit_results_20260521/make_figures.py` → §2 #3,13,14,15
- `Thesis/generated/baseline_comparison_20260521/make_figures.py` → §2 #4,5
- `MLP/eval/point_eval_figures.py` → §2 #6,7
- `MLP/eval/calibration_diagnostics.py` → §2 #8,9
- `MLP/eval/calibration_coverage_audit.py` → §2 #2
- `Thesis/generated/make_lono_figures.py` → §2 #10,11,12,16,17
- `MLP/curve_fit/reports/eval_stage3_thesis_sync.py` → §2 #18,20,21（注意：此脚本**不**产 #19 modeA_per_seed）
- `MLP/MLP_training/ablations/plot_residual_family_head_production.py --figure …` → §2 #23,24
- `MLP/MLP_training/ablations/compare_campaign_penetration.py` → §2 #26,27
- `Thesis/generated/make_n0_fewshot_figure.py` → §2 #28
