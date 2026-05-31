# MLP Training — Directory Map

## 默认生产模型 (default production run)

```
python MLP/MLP_training/run_full_pipeline.py             # single seed (seed 42)
python MLP/MLP_training/run_full_pipeline.py --mode C    # 5-seed bootstrap
```

Drives Stage-1 MSE → Stage-2 NLL → Stage-3 distillation ablation suite, picks winner.
Config: `config/full_pipeline_config.json`.

## OOD LONO 泛化实验 (`ood_lono/`)

Leave-one-nozzle-out generalization sweep. Each script runs Stage-1/2/3 with one
nozzle family held out, then evaluates on the held-out nozzle vs HA/NS baselines.

```
python MLP/MLP_training/ood_lono/run_lono_pipeline.py           # full LONO sweep
python MLP/MLP_training/ood_lono/run_stage1_lono_ablation.py    # feature variant × fold
python MLP/MLP_training/ood_lono/run_stage2_lono_ablation.py    # anchor variant × fold
python MLP/MLP_training/ood_lono/run_stage3_lono_ablation.py    # KD variant × fold
```

See `ood_lono/OOD_LONO_PLAN.md` for the full experimental protocol.

## 消融 & 分析工具 (`ablations/`)

In-distribution analysis and comparison scripts. Do not train new models — read
existing run directories and produce comparison tables or figures.

```
python MLP/MLP_training/ablations/emit_stage3_headline.py       # headline CSV for paper
python MLP/MLP_training/ablations/compare_delta_p_exponent_collapse.py
python MLP/MLP_training/ablations/compare_sparse_feature_stability.py
python MLP/MLP_training/ablations/toy_inference_from_run.py     # quick sanity sweep
python MLP/MLP_training/ablations/verify_refactor_baseline.py   # regression check
python MLP/MLP_training/ablations/export_run_manifest.py        # thesis manifest tables
```

## Shared library code (do not move)

| Path | Role |
|------|------|
| `engineered_feature_common.py` | Feature engineering, dataset, split logic |
| `efc/` | Backing modules for engineered_feature_common |
| `trainers/` | Stage-1 and Stage-2 trainer base classes |
| `stage3/` | KD loss strategies |
| `run_stage3_ablation_suite.py` | Stage-3 ablation runner (used by both default + LONO) |
| `train_stage{1,2,3}_*.py` | Individual stage training entry points |
