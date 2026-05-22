# Stage-3 Fixed-Table Evaluation with HA/NS

Evaluation tables:
- CDF uncensored: `MLP\synthetic_data\cdf_right_censoring_points\cdf_points_uncensored.csv`
- P50 observed: `MLP\synthetic_data\p50_q1_oracle\p50_q1_observed_fit_points.csv`
- Q1 grid: `MLP\synthetic_data\p50_q1_oracle\p50_q1_predictions.csv`

Physics baseline runs:
- hiroyasu_arai_calibrated: `MLP\baseline\Hiroyasu_Arai\outputs\20260521_145756_ha_calibrated_grouped_condition_all_thesis_refresh_20260521`
- naber_siebers_delay: `MLP\baseline\Naber_Siebers\outputs\20260521_145724_ns_delay_grouped_condition_thesis_refresh_20260521`

## CDF Uncensored Point-Level

| model_group | n_runs | rmse | mae | bias | p95 | cov1 | cov2 | cond_rmse_mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| production_mlp | 5 | 4.265 | 3.032 | 0.562 | 8.307 | 0.887 | 0.989 | 4.165 | seed42 (4.217) |
| mu_anchor_raw_uncertain_prodKD | 5 | 4.941 | 3.505 | -0.868 | 10.123 | 0.840 | 0.988 | 5.091 | seed17 (4.870) |
| svgp_stage3 | 1 | 4.193 | 2.894 | -0.040 | 8.461 | 0.716 | 0.945 | 4.114 | seed42 (4.193) |
| hiroyasu_arai_calibrated | 1 | 10.261 | 7.767 | 2.152 | 21.095 | 0.630 | 0.917 | 10.059 | ha_calibrated (10.261) |
| naber_siebers_delay | 1 | 9.286 | 7.195 | 0.628 | 18.408 | 0.659 | 0.927 | 8.949 | ns_delay (9.286) |

## P50 Observed Points

| model_group | n_runs | rmse | mae | bias | p95 | cov1 | cov2 | cond_rmse_mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| q1_oracle | 1 | 1.010 | 0.689 | 0.197 | 2.048 | 0.000 | 0.000 | 0.900 | quarter_only_v1_p50_oracle (1.010) |
| production_mlp | 5 | 2.848 | 1.826 | 0.856 | 6.017 | 0.967 | 1.000 | 2.068 | seed42 (2.794) |
| mu_anchor_raw_uncertain_prodKD | 5 | 3.212 | 2.147 | -0.468 | 7.323 | 0.955 | 1.000 | 2.837 | seed42 (2.967) |
| svgp_stage3 | 1 | 2.649 | 1.603 | 0.358 | 5.900 | 0.940 | 0.996 | 1.855 | seed42 (2.649) |
| hiroyasu_arai_calibrated | 1 | 9.948 | 7.662 | 3.588 | 20.229 | 0.633 | 0.944 | 9.179 | ha_calibrated (9.948) |
| naber_siebers_delay | 1 | 9.048 | 7.206 | 2.622 | 17.362 | 0.645 | 0.967 | 8.148 | ns_delay (9.048) |

## Q1 Oracle Grid, 0-5 ms

| model_group | n_runs | rmse | mae | bias | p95 | cov1 | cov2 | cond_rmse_mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| production_mlp | 5 | 10.086 | 6.533 | 5.845 | 23.496 | 0.709 | 0.899 | 7.040 | seed7 (10.023) |
| mu_anchor_raw_uncertain_prodKD | 5 | 9.669 | 6.288 | 5.266 | 22.728 | 0.740 | 0.916 | 6.877 | seed17 (9.584) |
| svgp_stage3 | 1 | 9.836 | 6.424 | 5.516 | 22.810 | 0.527 | 0.729 | 7.053 | seed42 (9.836) |
| hiroyasu_arai_calibrated | 1 | 29.695 | 22.850 | 19.652 | 59.851 | 0.333 | 0.527 | 27.186 | ha_calibrated (29.695) |
| naber_siebers_delay | 1 | 36.957 | 28.653 | 26.254 | 72.611 | 0.279 | 0.448 | 34.124 | ns_delay (36.957) |

## Q1 Oracle Grid, Extrapolated Region

| model_group | n_runs | rmse | mae | bias | p95 | cov1 | cov2 | cond_rmse_mean | best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| production_mlp | 5 | 10.864 | 7.380 | 6.753 | 24.514 | 0.665 | 0.882 | 7.461 | seed7 (10.794) |
| mu_anchor_raw_uncertain_prodKD | 5 | 10.358 | 6.965 | 6.269 | 23.768 | 0.707 | 0.902 | 7.072 | seed17 (10.272) |
| svgp_stage3 | 1 | 10.591 | 7.236 | 6.410 | 23.701 | 0.459 | 0.683 | 7.465 | seed42 (10.591) |
| hiroyasu_arai_calibrated | 1 | 31.837 | 25.316 | 22.282 | 61.493 | 0.272 | 0.459 | 28.965 | ha_calibrated (31.837) |
| naber_siebers_delay | 1 | 39.818 | 32.216 | 30.378 | 74.495 | 0.217 | 0.360 | 36.586 | ns_delay (39.818) |

## Per-Run Headline

```
               eval_set                    model_group                model_label  n_points  n_conditions   rmse_mm    mae_mm   bias_mm  p95_abs_err_mm  coverage_1sigma  coverage_2sigma  condition_rmse_mean_mm
           p50_observed                      q1_oracle quarter_only_v1_p50_oracle      4213           495  1.009930  0.688959  0.197147        2.048475         0.000000         0.000000                0.900465
         cdf_uncensored                 production_mlp                      seed7    542565           592  4.309562  3.100850  0.737115        8.356879         0.887831         0.989793                4.219200
           p50_observed                 production_mlp                      seed7      4213           495  2.919499  1.927051  0.972699        6.160249         0.967482         0.999525                2.208239
            q1_grid_all                 production_mlp                      seed7     25245           495 10.023386  6.522529  5.892019       23.220746         0.711468         0.904139                7.022534
q1_grid_observed_window                 production_mlp                      seed7      3718           495  2.750107  1.671762  0.702575        6.511392         0.966111         0.999731                1.845204
   q1_grid_extrapolated                 production_mlp                      seed7     21527           495 10.794176  7.360322  6.788305       24.186605         0.667487         0.887629                7.439400
         cdf_uncensored                 production_mlp                     seed17    542565           592  4.231483  2.992354  0.470395        8.233727         0.889005         0.989094                4.124194
           p50_observed                 production_mlp                     seed17      4213           495  2.797001  1.767645  0.768317        5.976484         0.968668         0.999288                2.000155
            q1_grid_all                 production_mlp                     seed17     25245           495 10.149131  6.561803  5.862347       23.608781         0.710042         0.896494                7.081874
q1_grid_observed_window                 production_mlp                     seed17      3718           495  2.678875  1.564396  0.520010        6.366282         0.965035         0.999731                1.740991
   q1_grid_extrapolated                 production_mlp                     seed17     21527           495 10.934154  7.424922  6.785040       24.669054         0.666001         0.878664                7.510778
         cdf_uncensored                 production_mlp                     seed42    542565           592  4.216750  2.981921  0.433254        8.221263         0.891405         0.989723                4.109473
           p50_observed                 production_mlp                     seed42      4213           495  2.794273  1.792725  0.785655        5.879329         0.969380         1.000000                1.996667
            q1_grid_all                 production_mlp                     seed42     25245           495 10.026176  6.462394  5.705251       23.405362         0.713369         0.895464                6.971050
q1_grid_observed_window                 production_mlp                     seed42      3718           495  2.695853  1.639122  0.539951        6.200958         0.965573         1.000000                1.794737
   q1_grid_extrapolated                 production_mlp                     seed42     21527           495 10.799577  7.295437  6.597368       24.407669         0.669810         0.877410                7.381199
         cdf_uncensored                 production_mlp                     seed99    542565           592  4.295211  3.056138  0.585409        8.393292         0.883100         0.989117                4.210908
           p50_observed                 production_mlp                     seed99      4213           495  2.827536  1.804339  0.877427        5.949828         0.966770         0.999288                2.049971
            q1_grid_all                 production_mlp                     seed99     25245           495 10.106566  6.545369  5.847454       23.573976         0.706160         0.900376                7.050183
q1_grid_observed_window                 production_mlp                     seed99      3718           495  2.707755  1.621541  0.583552        6.343677         0.965573         0.999731                1.818498
   q1_grid_extrapolated                 production_mlp                     seed99     21527           495 10.886586  7.395780  6.756600       24.633892         0.661356         0.883216                7.471271
         cdf_uncensored                 production_mlp                   seed2024    542565           592  4.274437  3.029779  0.583747        8.331281         0.881491         0.988453                4.163025
           p50_observed                 production_mlp                   seed2024      4213           495  2.900231  1.837382  0.875290        6.117770         0.962022         0.999525                2.085100
            q1_grid_all                 production_mlp                   seed2024     25245           495 10.126882  6.573795  5.919369       23.672534         0.703347         0.898554                7.075973
q1_grid_observed_window                 production_mlp                   seed2024      3718           495  2.763940  1.639373  0.589263        6.490677         0.960463         0.999462                1.798624
   q1_grid_extrapolated                 production_mlp                   seed2024     21527           495 10.906270  7.426036  6.839950       24.674740         0.658940         0.881126                7.501122
         cdf_uncensored mu_anchor_raw_uncertain_prodKD                      seed7    542565           592  4.923534  3.488898 -0.848491       10.066414         0.838983         0.987837                5.082841
           p50_observed mu_anchor_raw_uncertain_prodKD                      seed7      4213           495  3.170912  2.145041 -0.477505        7.394932         0.958462         0.999763                2.837260
            q1_grid_all mu_anchor_raw_uncertain_prodKD                      seed7     25245           495  9.692790  6.306854  5.320249       22.787834         0.743038         0.916934                6.904133
q1_grid_observed_window mu_anchor_raw_uncertain_prodKD                      seed7      3718           495  3.652268  2.311786 -0.559392        8.825834         0.928725         1.000000                3.412932
   q1_grid_extrapolated mu_anchor_raw_uncertain_prodKD                      seed7     21527           495 10.386183  6.996855  6.335742       23.806659         0.710968         0.902587                7.107041
         cdf_uncensored mu_anchor_raw_uncertain_prodKD                     seed17    542565           592  4.870308  3.457074 -1.029203        9.975970         0.854489         0.989362                5.028201
           p50_observed mu_anchor_raw_uncertain_prodKD                     seed17      4213           495  3.049062  2.034615 -0.628177        6.886714         0.962972         1.000000                2.683281
            q1_grid_all mu_anchor_raw_uncertain_prodKD                     seed17     25245           495  9.583779  6.226326  5.137038       22.527757         0.747237         0.926758                6.809892
q1_grid_observed_window mu_anchor_raw_uncertain_prodKD                     seed17      3718           495  3.572617  2.289547 -0.735257        8.706039         0.940828         1.000000                3.348501
   q1_grid_extrapolated mu_anchor_raw_uncertain_prodKD                     seed17     21527           495 10.271703  6.906260  6.151262       23.503709         0.713801         0.914108                7.017441
         cdf_uncensored mu_anchor_raw_uncertain_prodKD                     seed42    542565           592  4.894902  3.466848 -0.561011       10.000869         0.841785         0.986885                5.023107
           p50_observed mu_anchor_raw_uncertain_prodKD                     seed42      4213           495  2.966747  1.985299 -0.030818        6.764355         0.964633         1.000000                2.497800
            q1_grid_all mu_anchor_raw_uncertain_prodKD                     seed42     25245           495  9.826704  6.395138  5.449247       23.116658         0.732303         0.908061                6.986192
q1_grid_observed_window mu_anchor_raw_uncertain_prodKD                     seed42      3718           495  3.730826  2.373350 -0.207306        9.096633         0.924422         1.000000                3.491668
   q1_grid_extrapolated mu_anchor_raw_uncertain_prodKD                     seed42     21527           495 10.527963  7.089754  6.426209       24.240285         0.699122         0.892182                7.194727
         cdf_uncensored mu_anchor_raw_uncertain_prodKD                     seed99    542565           592  5.031662  3.605230 -1.419967       10.269810         0.829843         0.987452                5.194933
           p50_observed mu_anchor_raw_uncertain_prodKD                     seed99      4213           495  3.275420  2.305138 -1.013163        7.082702         0.964396         1.000000                2.983158
            q1_grid_all mu_anchor_raw_uncertain_prodKD                     seed99     25245           495  9.648530  6.242170  5.070965       22.701845         0.741573         0.914953                6.850503
q1_grid_observed_window mu_anchor_raw_uncertain_prodKD                     seed99      3718           495  3.781150  2.480857 -1.179288        8.934960         0.935987         1.000000                3.601322
   q1_grid_extrapolated mu_anchor_raw_uncertain_prodKD                     seed99     21527           495 10.329735  6.891799  6.150467       23.765010         0.707995         0.900265                7.005982
         cdf_uncensored mu_anchor_raw_uncertain_prodKD                   seed2024    542565           592  4.985328  3.508788 -0.480284       10.300754         0.834838         0.986650                5.125725
           p50_observed mu_anchor_raw_uncertain_prodKD                   seed2024      4213           495  3.595554  2.264610 -0.190977        8.486645         0.924045         0.999525                3.183718
            q1_grid_all mu_anchor_raw_uncertain_prodKD                   seed2024     25245           495  9.594208  6.267106  5.350718       22.506076         0.735868         0.915706                6.835708
q1_grid_observed_window mu_anchor_raw_uncertain_prodKD                   seed2024      3718           495  3.725111  2.372078 -0.037763        9.265730         0.920925         1.000000                3.443462
   q1_grid_extrapolated mu_anchor_raw_uncertain_prodKD                   seed2024     21527           495 10.273764  6.939829  6.281381       23.524754         0.703907         0.901147                7.035723
         cdf_uncensored                    svgp_stage3                     seed42    542565           592  4.192763  2.893807 -0.040461        8.461266         0.715804         0.944600                4.113945
           p50_observed                    svgp_stage3                     seed42      4213           495  2.648853  1.603132  0.357814        5.899526         0.940423         0.995965                1.855245
            q1_grid_all                    svgp_stage3                     seed42     25245           495  9.836484  6.424120  5.515533       22.810068         0.527471         0.729134                7.053243
q1_grid_observed_window                    svgp_stage3                     seed42      3718           495  2.749035  1.722741  0.335535        6.245917         0.922539         0.994890                2.008036
   q1_grid_extrapolated                    svgp_stage3                     seed42     21527           495 10.590671  7.236110  6.410187       23.701437         0.459237         0.683235                7.465358
         cdf_uncensored       hiroyasu_arai_calibrated              ha_calibrated    542565           592 10.260732  7.767321  2.152303       21.094529         0.630378         0.917191               10.059311
           p50_observed       hiroyasu_arai_calibrated              ha_calibrated      4213           495  9.947693  7.662376  3.588360       20.229481         0.633278         0.943983                9.178854
            q1_grid_all       hiroyasu_arai_calibrated              ha_calibrated     25245           495 29.695159 22.850260 19.651747       59.850916         0.332739         0.527352               27.186311
q1_grid_observed_window       hiroyasu_arai_calibrated              ha_calibrated      3718           495 10.893935  8.575175  4.424438       21.800314         0.686659         0.922539               10.169671
   q1_grid_extrapolated       hiroyasu_arai_calibrated              ha_calibrated     21527           495 31.837152 25.315758 22.281707       61.493004         0.271612         0.459098               28.964911
         cdf_uncensored            naber_siebers_delay                   ns_delay    542565           592  9.286084  7.195253  0.628084       18.408244         0.659451         0.927096                8.948502
           p50_observed            naber_siebers_delay                   ns_delay      4213           495  9.048470  7.205746  2.621648       17.361901         0.644671         0.967482                8.148339
            q1_grid_all            naber_siebers_delay                   ns_delay     25245           495 36.956999 28.653293 26.254480       72.611340         0.279144         0.448128               34.124155
q1_grid_observed_window            naber_siebers_delay                   ns_delay      3718           495  9.697488  8.024800  2.382030       17.433539         0.637708         0.955621                8.986351
   q1_grid_extrapolated            naber_siebers_delay                   ns_delay     21527           495 39.817997 32.216109 30.377571       74.494570         0.217216         0.360478               36.585911
```
