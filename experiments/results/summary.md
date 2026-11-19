# Resumen benchmark DriftRegressor vs Random Forest

Valores = RMSE del modelo / RMSE del RF clásico (media geométrica). < 1 es mejor que el RF clásico. En negrita, el mejor modelo cuando supera al RF.


## Sintetico — Walk-forward (6 reentrenamientos)

| Dataset | Drift | AUC data drift | concept gain | DR_hybrid | DR_mmd | DR_recent_error | DR_recent_refit* | RF_pesos_exp | RF_ventana |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| covariate_shift | data fuerte (extrapolación) | 0.97 | -0.01 | 1.19 | 1.19 | 1.19 | 1.05 | **0.97** | 1.08 |
| friedman_abrupt | concept abrupto | 0.52 | 0.25 | 0.99 | 1.04 | 0.94 | 0.88 | 0.90 | **0.82** |
| friedman_gradual_data | data + concept gradual | 0.99 | 0.06 | 1.10 | 1.04 | 1.11 | 0.96 | 0.97 | **0.93** |
| friedman_recurring | concept recurrente + data | 0.71 | 0.04 | **0.82** | 0.98 | 0.98 | 0.97 | 1.02 | 1.02 |
| healthcare_both | data + concept | 0.78 | 0.57 | 0.95 | 0.92 | 0.86 | 0.66 | 0.88 | **0.57** |
| healthcare_concept | concept gradual | 0.49 | 0.57 | 0.95 | 0.99 | 0.78 | 0.66 | 0.88 | **0.57** |
| healthcare_data | data | 0.78 | 0.01 | 1.29 | 1.18 | 1.13 | 1.08 | 1.01 | 1.11 |
| healthcare_none | ninguno (control) | 0.49 | 0.01 | 1.04 | 0.99 | 1.01 | **0.99** | 1.01 | 1.07 |
| highdim_mixed | data + concept abrupto/incremental | 0.90 | 0.33 | 1.10 | 1.08 | 1.07 | 0.99 | 0.96 | **0.90** |
| incremental | concept incremental | 0.50 | 0.03 | 0.88 | 1.00 | 0.83 | 0.75 | 0.83 | **0.72** |
| local_drift | concept local | 0.50 | 0.44 | 0.89 | 1.02 | 0.83 | 0.76 | 0.89 | **0.70** |


**Agregado:**

| Modelo | RMSE / RF (geo) | Datasets mejor que RF (de 11) | % pasos mejor que RF | Rank medio | p Wilcoxon |
|---|---:|---:|---:|---:|---:|
| DR_recent_refit* | 0.873 | 9 | 57 % | 2.36 | 0.054 |
| RF_ventana | 0.839 | 7 | 54 % | 2.64 | 0.067 |
| RF_pesos_exp | 0.936 | 8 | 68 % | 3.45 | 0.014 |
| DR_recent_error | 0.965 | 6 | 40 % | 4.55 | 0.465 |
| RF_clasico | 1.00 | — | — | 4.55 | — |
| DR_mmd | 1.036 | 5 | 37 % | 5.09 | 0.240 |
| DR_hybrid | 1.008 | 6 | 33 % | 5.36 | 1.000 |



## Sintetico — Split estático 80/20 (protocolo README)

| Dataset | Drift | AUC data drift | concept gain | DR_hybrid | DR_mmd | DR_recent_error | DR_recent_refit* | RF_pesos_exp | RF_ventana |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| covariate_shift | data fuerte (extrapolación) | 0.97 | -0.01 | 1.23 | 1.17 | 1.25 | 1.09 | 1.00 | 1.08 |
| friedman_abrupt | concept abrupto | 0.52 | 0.25 | 0.87 | 0.99 | 0.74 | 0.60 | 0.78 | **0.56** |
| friedman_gradual_data | data + concept gradual | 0.99 | 0.06 | 1.18 | 1.02 | 1.18 | 0.96 | 0.98 | **0.93** |
| friedman_recurring | concept recurrente + data | 0.71 | 0.04 | **0.72** | 1.01 | 1.13 | 1.12 | 1.01 | 1.15 |
| healthcare_both | data + concept | 0.78 | 0.57 | 0.85 | 0.89 | 0.76 | 0.44 | 0.85 | **0.32** |
| healthcare_concept | concept gradual | 0.49 | 0.57 | 0.76 | 1.00 | 0.62 | 0.46 | 0.80 | **0.36** |
| healthcare_data | data | 0.78 | 0.01 | 1.16 | 1.12 | 1.11 | 1.03 | 1.01 | 1.03 |
| healthcare_none | ninguno (control) | 0.49 | 0.01 | 1.01 | 0.98 | 0.99 | **0.98** | 1.01 | 1.08 |
| highdim_mixed | data + concept abrupto/incremental | 0.90 | 0.33 | 1.34 | 1.05 | 1.10 | 1.10 | 0.92 | **0.71** |
| incremental | concept incremental | 0.50 | 0.03 | 1.00 | 0.99 | 0.80 | 0.70 | 0.80 | **0.67** |
| local_drift | concept local | 0.50 | 0.44 | 0.55 | 1.01 | 0.54 | **0.49** | 0.80 | 0.50 |


**Agregado:**

| Modelo | RMSE / RF (geo) | Datasets mejor que RF (de 11) | % pasos mejor que RF | Rank medio | p Wilcoxon |
|---|---:|---:|---:|---:|---:|
| RF_ventana | 0.699 | 7 | 67 % | 2.55 | 0.083 |
| DR_recent_refit* | 0.768 | 7 | 64 % | 2.73 | 0.175 |
| RF_pesos_exp | 0.902 | 7 | 67 % | 3.64 | 0.042 |
| RF_clasico | 1.00 | — | — | 4.36 | — |
| DR_recent_error | 0.897 | 6 | 52 % | 4.45 | 0.278 |
| DR_hybrid | 0.940 | 6 | 45 % | 5.09 | 0.638 |
| DR_mmd | 1.019 | 4 | 33 % | 5.18 | 0.320 |



## Real — Walk-forward (6 reentrenamientos)

| Dataset | Drift | AUC data drift | concept gain | DR_hybrid | DR_mmd | DR_recent_error | DR_recent_refit* | RF_pesos_exp | RF_ventana |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| air_quality | sensor drift (concept) | 1.00 | 0.03 | 1.03 | 1.03 | 1.06 | 1.05 | 1.03 | 1.08 |
| beijing_pm25 | estacional + emisiones | 0.82 | 0.15 | 1.06 | **0.99** | 1.08 | 1.08 | 1.01 | 1.20 |
| bike_sharing | contexto oculto + estacional | 0.97 | 0.38 | 1.05 | 1.01 | 0.88 | 0.87 | 0.97 | **0.86** |
| electricity | concept drift de mercado | 1.00 | 0.41 | 1.06 | **0.91** | 1.13 | 1.09 | 0.98 | 1.03 |
| gas_drift | sensor drift fuerte (data + concept) | 1.00 | 0.01 | 1.26 | 1.17 | 1.25 | 1.10 | 1.01 | 1.11 |


**Agregado:**

| Modelo | RMSE / RF (geo) | Datasets mejor que RF (de 5) | % pasos mejor que RF | Rank medio | p Wilcoxon |
|---|---:|---:|---:|---:|---:|
| RF_clasico | 1.00 | — | — | 2.60 | — |
| DR_mmd | 1.020 | 2 | 40 % | 3.00 | 0.625 |
| RF_pesos_exp | 1.002 | 2 | 60 % | 3.20 | 1.000 |
| DR_recent_refit* | 1.034 | 1 | 33 % | 4.00 | 0.625 |
| RF_ventana | 1.049 | 1 | 37 % | 4.60 | 0.438 |
| DR_hybrid | 1.088 | 0 | 23 % | 5.00 | 0.062 |
| DR_recent_error | 1.073 | 1 | 27 % | 5.60 | 0.438 |



## Real — Split estático 80/20 (protocolo README)

| Dataset | Drift | AUC data drift | concept gain | DR_hybrid | DR_mmd | DR_recent_error | DR_recent_refit* | RF_pesos_exp | RF_ventana |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| air_quality | sensor drift (concept) | 1.00 | 0.03 | 0.98 | 0.94 | 0.92 | **0.91** | 1.04 | 0.98 |
| beijing_pm25 | estacional + emisiones | 0.82 | 0.15 | 1.09 | 1.01 | 1.06 | 1.06 | **0.98** | 1.14 |
| bike_sharing | contexto oculto + estacional | 0.97 | 0.38 | 1.03 | 0.98 | 0.63 | **0.58** | 0.99 | 0.72 |
| electricity | concept drift de mercado | 1.00 | 0.41 | 1.23 | **0.91** | 1.04 | 1.01 | 1.02 | 1.07 |
| gas_drift | sensor drift fuerte (data + concept) | 1.00 | 0.01 | 1.89 | 1.43 | 1.50 | 1.45 | 1.09 | 1.20 |


**Agregado:**

| Modelo | RMSE / RF (geo) | Datasets mejor que RF (de 5) | % pasos mejor que RF | Rank medio | p Wilcoxon |
|---|---:|---:|---:|---:|---:|
| DR_recent_refit* | 0.961 | 2 | 40 % | 2.80 | 1.000 |
| DR_mmd | 1.038 | 3 | 60 % | 3.00 | 0.812 |
| RF_clasico | 1.00 | — | — | 3.40 | — |
| RF_pesos_exp | 1.024 | 2 | 40 % | 3.80 | 0.312 |
| DR_recent_error | 0.990 | 2 | 40 % | 4.00 | 1.000 |
| RF_ventana | 1.004 | 2 | 40 % | 4.60 | 0.812 |
| DR_hybrid | 1.209 | 1 | 20 % | 6.40 | 0.125 |



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