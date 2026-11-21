# Resumen benchmark DriftRegressor vs Random Forest

Valores = RMSE del modelo / RMSE del RF clásico (media geométrica). < 1 es mejor que el RF clásico. En negrita, el mejor modelo cuando supera al RF.


## Sintetico — Walk-forward (6 reentrenamientos)

| Dataset | Drift | AUC data drift | concept gain | ADF_error | ADF_hybrid | DR_hybrid | DR_mmd | DR_recent_error | RF_pesos_exp | RF_ventana |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| covariate_shift | data fuerte (extrapolación) | 0.97 | -0.01 | **0.98** | 1.02 | 1.20 | 1.20 | 1.20 | 0.99 | 1.09 |
| friedman_abrupt | concept abrupto | 0.52 | 0.25 | **0.81** | 0.84 | 1.01 | 1.04 | 0.93 | 0.91 | 0.82 |
| friedman_gradual_data | data + concept gradual | 0.99 | 0.06 | 0.91 | **0.91** | 1.08 | 1.02 | 1.08 | 0.96 | 0.92 |
| friedman_recurring | concept recurrente + data | 0.71 | 0.04 | 0.97 | 0.97 | **0.80** | 0.97 | 0.98 | 1.01 | 1.03 |
| healthcare_both | data + concept | 0.78 | 0.57 | **0.56** | 0.57 | 0.95 | 0.93 | 0.88 | 0.90 | 0.58 |
| healthcare_concept | concept gradual | 0.49 | 0.57 | 0.54 | **0.53** | 0.88 | 0.98 | 0.77 | 0.88 | 0.57 |
| healthcare_data | data | 0.78 | 0.01 | 1.03 | 1.12 | 1.20 | 1.20 | 1.17 | 1.04 | 1.18 |
| healthcare_none | ninguno (control) | 0.49 | 0.01 | **0.98** | 1.00 | 1.02 | 1.00 | 1.00 | 1.01 | 1.08 |
| highdim_mixed | data + concept abrupto/incremental | 0.90 | 0.33 | 0.95 | 0.92 | 1.12 | 1.09 | 1.08 | 0.96 | **0.91** |
| incremental | concept incremental | 0.50 | 0.03 | **0.69** | 0.70 | 0.86 | 1.00 | 0.80 | 0.80 | 0.71 |
| local_drift | concept local | 0.50 | 0.44 | 0.71 | 0.71 | 0.89 | 1.02 | 0.82 | 0.89 | **0.69** |


**Agregado:**

| Modelo | RMSE / RF (geo) | Datasets mejor que RF (de 11) | % pasos mejor que RF | Rank medio | p Wilcoxon |
|---|---:|---:|---:|---:|---:|
| ADF_error | 0.810 | 10 | 80 % | 2.09 | 0.005 |
| ADF_hybrid | 0.823 | 9 | 69 % | 2.55 | 0.024 |
| RF_ventana | 0.846 | 7 | 55 % | 3.82 | 0.067 |
| RF_pesos_exp | 0.938 | 8 | 67 % | 4.36 | 0.024 |
| RF_clasico | 1.00 | — | — | 5.09 | — |
| DR_recent_error | 0.965 | 6 | 46 % | 5.36 | 0.520 |
| DR_mmd | 1.038 | 5 | 38 % | 6.27 | 0.320 |
| DR_hybrid | 0.993 | 5 | 39 % | 6.45 | 0.898 |



## Sintetico — Split estático 80/20 (protocolo README)

| Dataset | Drift | AUC data drift | concept gain | ADF_error | ADF_hybrid | DR_hybrid | DR_mmd | DR_recent_error | RF_pesos_exp | RF_ventana |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| covariate_shift | data fuerte (extrapolación) | 0.97 | -0.01 | **0.98** | 1.02 | 1.19 | 1.16 | 1.17 | 1.01 | 1.09 |
| friedman_abrupt | concept abrupto | 0.52 | 0.25 | 0.66 | 0.63 | 0.90 | 1.02 | 0.75 | 0.80 | **0.57** |
| friedman_gradual_data | data + concept gradual | 0.99 | 0.06 | **0.92** | 0.92 | 1.16 | 1.01 | 1.16 | 0.96 | 0.92 |
| friedman_recurring | concept recurrente + data | 0.71 | 0.04 | 1.01 | 1.01 | **0.74** | 1.02 | 1.15 | 1.00 | 1.16 |
| healthcare_both | data + concept | 0.78 | 0.57 | 0.28 | **0.28** | 0.90 | 0.88 | 0.81 | 0.86 | 0.31 |
| healthcare_concept | concept gradual | 0.49 | 0.57 | **0.32** | **0.32** | 0.81 | 0.99 | 0.65 | 0.80 | 0.35 |
| healthcare_data | data | 0.78 | 0.01 | 1.01 | 1.07 | 1.21 | 1.22 | 1.21 | 1.04 | 1.23 |
| healthcare_none | ninguno (control) | 0.49 | 0.01 | **0.99** | 1.01 | 1.05 | 1.02 | 1.02 | 1.01 | 1.06 |
| highdim_mixed | data + concept abrupto/incremental | 0.90 | 0.33 | 1.06 | 0.78 | 1.31 | 1.02 | 1.06 | 0.92 | **0.70** |
| incremental | concept incremental | 0.50 | 0.03 | **0.61** | 0.61 | 0.79 | 0.99 | 0.74 | 0.78 | 0.61 |
| local_drift | concept local | 0.50 | 0.44 | 0.50 | 0.50 | 0.71 | 1.02 | 0.51 | 0.80 | **0.49** |


**Agregado:**

| Modelo | RMSE / RF (geo) | Datasets mejor que RF (de 11) | % pasos mejor que RF | Rank medio | p Wilcoxon |
|---|---:|---:|---:|---:|---:|
| ADF_error | 0.691 | 8 | 82 % | 2.23 | 0.032 |
| ADF_hybrid | 0.677 | 7 | 76 % | 2.77 | 0.042 |
| RF_ventana | 0.701 | 7 | 64 % | 3.91 | 0.083 |
| RF_pesos_exp | 0.903 | 7 | 79 % | 4.00 | 0.054 |
| RF_clasico | 1.00 | — | — | 4.82 | — |
| DR_recent_error | 0.898 | 5 | 52 % | 5.55 | 0.320 |
| DR_hybrid | 0.958 | 6 | 52 % | 6.18 | 0.520 |
| DR_mmd | 1.028 | 3 | 24 % | 6.55 | 0.067 |



## Diagnóstico de drift

| dataset | kind | drift | n | p | data_drift_auc | mmd2_ini_fin | concept_gain | recency_ratio |
|---|---|---|---|---|---|---|---|---|
| air_quality | real | sensor drift (concept) | 7344 | 10 | 1.0 | 0.207 | 0.034 | 0.949 |
| bike_sharing | real | contexto oculto + estacional | 17379 | 11 | 0.974 | 0.182 | 0.383 | 1.239 |
| gas_drift | real | sensor drift fuerte (data + concept) | 13910 | 134 | 1.0 | 0.145 | 0.007 | 1.924 |
| beijing_pm25 | real | estacional + emisiones | 13919 | 12 | 0.824 | 0.013 | 0.151 | 1.023 |
| electricity | real | concept drift de mercado | 15104 | 6 | 1.0 | 0.052 | 0.406 | 1.422 |
| healthcare_none | sintetico | ninguno (control) | 5000 | 4 | 0.49 | 0.0 | 0.005 | 1.039 |
| healthcare_data | sintetico | data | 5000 | 4 | 0.777 | 0.091 | 0.006 | 1.079 |
| healthcare_concept | sintetico | concept gradual | 5000 | 4 | 0.49 | 0.0 | 0.57 | 1.806 |
| healthcare_both | sintetico | data + concept | 5000 | 4 | 0.777 | 0.091 | 0.573 | 1.788 |
| friedman_abrupt | sintetico | concept abrupto | 5000 | 10 | 0.519 | 0.0 | 0.251 | 1.605 |
| friedman_gradual_data | sintetico | data + concept gradual | 5000 | 10 | 0.991 | 0.339 | 0.055 | 1.885 |
| friedman_recurring | sintetico | concept recurrente + data | 5000 | 10 | 0.714 | 0.013 | 0.043 | 1.005 |
| covariate_shift | sintetico | data fuerte (extrapolación) | 5000 | 8 | 0.965 | 0.227 | -0.005 | 2.003 |
| highdim_mixed | sintetico | data + concept abrupto/incremental | 5000 | 20 | 0.902 | 0.066 | 0.335 | 1.286 |
| local_drift | sintetico | concept local | 5000 | 6 | 0.504 | -0.0 | 0.436 | 1.834 |
| incremental | sintetico | concept incremental | 5000 | 10 | 0.497 | 0.0 | 0.034 | 1.349 |