# Additive Residual Multi-Task SVGP

- Baseline CDF uncensored RMSE: 4.192763 mm
- Baseline P50 observed RMSE: 2.648853 mm
- Passed primary goal: True

## Winner

- Run: C:\Users\Jiang\Documents\Mie_Postprocessing_Py\MLP\runs_mlp\residual_multitask_svgp_prod_existing_full_svgp_l2_1em04_20260531_025634
- Shared base: existing_full_svgp
- Delta L2: 0.0001
- CDF uncensored RMSE: 3.987472 mm
- P50 observed RMSE: 2.076548 mm

## Sweep Table

```csv
kind,shared_base,delta_l2_weight,cdf_rmse_mm,cdf_mae_mm,cdf_bias_mm,cdf_ece,cdf_crps,p50_rmse_mm,q1_observed_rmse_mm,q1_extrapolated_rmse_mm
baseline,existing_full_svgp,,4.192763,2.893807,-0.040461,,,2.648853,2.749035,10.590671
residual,existing_full_svgp,0.0001,3.987471916839798,2.8267596036035783,0.1825934583317243,0.0261602175242646,2.040125742317233,2.0765482997553195,2.134452688337429,10.429313731752222
```

Summary CSV: `C:\Users\Jiang\Documents\Mie_Postprocessing_Py\Thesis\slides\slides_residual_multitask_svgp\residual_multitask_svgp_eval_summary.csv`