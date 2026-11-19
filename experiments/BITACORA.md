# Bitácora — Benchmark DriftRegressor vs Random Forest

**Período:** 18–19 nov 2026 · **Objetivo:** medir qué tan bueno es `DriftRegressor` (ensamble por bloques
temporales) frente al `RandomForestRegressor` convencional, en datasets sintéticos y reales con
data drift y concept drift, con varios tipos de variables.

---

## 1. Registro cronológico

| # | Paso | Resultado |
|---|---|---|
| 1 | Revisión del repo (`block_ensemble.py`, `mmd.py`, `generate_data.py`, `tune_and_compare.py`) | Modelo solo de **regresión**; 3 modos de peso (`mmd`, `recent_error`, `hybrid`). |
| 2 | Entorno: `venv_drift_forest` con Python 3.14 + `requirements.txt` (el Python 3.10 global no tenía `pyarrow` ni `optuna`) | OK — numpy 2.5.5, sklearn 1.9.1, optuna 5.1.0. |
| 3 | Reproducción del README (`python -m src.drift_adaptive_forest.models.block_ensemble`) | **Idéntico**: RF 6.4003 · DR mmd 5.9404 · recent_error 5.0854 · hybrid 5.8175. |
| 4 | Descarga de 5 datasets reales a `data/real/<nombre>/` (UCI + OpenML) | Air Quality, Bike Sharing, Gas Sensor Drift, Beijing PM2.5, Electricity (Elec2). |
| 5 | Prueba rápida en `healthcare_both`: un **RF entrenado solo con el 20 % más reciente** saca RMSE 2.01 vs 4.58 de `DR_recent_error` | Hipótesis: `recent_error` desperdicia el 15 % más reciente (solo lo usa para validar). Se añadió la variante experimental `DR_recent_refit*`. |
| 6 | Verificación de bugs de inferencia | Confirmados 2 problemas (ver §4). |
| 7 | Benchmark completo: 11 escenarios sintéticos × 3 semillas + 5 reales, protocolos estático y walk-forward, 7 modelos | ~2.2 h de cómputo. `experiments/results/benchmark_*.csv`. |
| 8 | Tuning con Optuna (20 trials por familia, TimeSeriesSplit 4) en 5 sintéticos + 5 reales | La sesión se cortó a mitad; se reanudó solo con los 3 reales faltantes (`tuned_static_part2.csv`). |
| 9 | Resumen y conclusiones | `experiments/results/summary.md` + esta bitácora. |

---

## 2. Diseño experimental

**Modelos** (todos con 100 árboles por bosque, profundidad libre):

| Modelo | Qué es |
|---|---|
| `RF_clasico` | RandomForest con todo el histórico — **referencia** |
| `DR_mmd`, `DR_recent_error`, `DR_hybrid` | `DriftRegressor(n_blocks=5, beta=15)`, exactamente como en el README |
| `RF_ventana` | RF solo con el 20 % más reciente (línea base trivial de recencia) |
| `RF_pesos_exp` | RF con `sample_weight` exponencial, vida media = 20 % del histórico |
| `DR_recent_refit*` | **Experimental** (no está en la librería): `recent_error` + reentrenar el último bloque incluyendo la cola de validación |

**Protocolos:** (a) *estático* 80/20 como en el README; (b) *walk-forward*: 6 reentrenamientos con
ventana expansiva (orígenes en 40 %, 50 %, …, 90 %), cada uno evaluado en el 10 % siguiente.

**Métrica principal:** RMSE del modelo / RMSE del RF clásico (media geométrica). **< 1 = mejor que el RF.**

**Diagnóstico de drift (independiente de los modelos):**
- *AUC data drift*: clasificador que distingue el 30 % inicial vs final de X (0.5 = sin drift).
- *concept gain*: cuánto mejora un RF al añadir el tiempo `t` como feature (KFold aleatorio). > 0 ⇒ P(Y|X) cambia.

### Datasets

| Dataset | Tipo | n | vars | Drift |
|---|---|---:|---:|---|
| healthcare_none / data / concept / both | sintético (generador del repo) | 5 000 | 4 | control / data / concept / ambos |
| friedman_abrupt | sintético | 5 000 | 10 | concept abrupto en t=0.6 |
| friedman_gradual_data | sintético | 5 000 | 10 | data + concept gradual |
| friedman_recurring | sintético | 5 000 | 10 | concept recurrente A→B→A→B con firma en X |
| covariate_shift | sintético | 5 000 | 8 | solo data drift fuerte (2σ) |
| highdim_mixed | sintético | 5 000 | 20 | data + concept abrupto + incremental |
| local_drift | sintético | 5 000 | 6 | concept solo en una región de X |
| incremental | sintético | 5 000 | 10 | coeficientes que rotan continuamente |
| air_quality (UCI 360) | real | 7 344 | 10 | deriva de sensores químicos |
| bike_sharing (UCI 275) | real | 17 379 | 11 | crecimiento del sistema (oculto) + estacional |
| gas_drift (UCI 270) | real | 13 910 | 134 | deriva de sensores durante 36 meses |
| beijing_pm25 (UCI 381) | real | 13 919 | 12 | estacional + cambio de emisiones |
| electricity (Elec2) | real | 15 104 | 6 | concept drift de mercado |

Notas: en Bike Sharing se excluyó `yr` para que el crecimiento actúe como contexto oculto. Beijing y
Electricity se submuestrearon 1 de cada 3 filas para que las matrices del MMD quepan en memoria.

---

## 3. Resultados

### 3.1 Sintéticos — walk-forward (3 semillas × 6 pasos)

| Dataset | DR_mmd | DR_recent | DR_hybrid | DR_refit* | RF_pesos_exp | RF_ventana |
|---|---:|---:|---:|---:|---:|---:|
| healthcare_none (control) | 0.99 | 1.01 | 1.04 | 0.99 | 1.01 | 1.07 |
| healthcare_data | 1.18 | 1.13 | 1.29 | 1.08 | 1.01 | 1.11 |
| healthcare_concept | 0.99 | 0.78 | 0.95 | 0.66 | 0.88 | **0.57** |
| healthcare_both | 0.92 | 0.86 | 0.95 | 0.66 | 0.88 | **0.57** |
| friedman_abrupt | 1.04 | 0.94 | 0.99 | 0.88 | 0.90 | **0.82** |
| friedman_gradual_data | 1.04 | 1.11 | 1.10 | 0.96 | 0.97 | **0.93** |
| friedman_recurring | 0.98 | 0.98 | **0.82** | 0.97 | 1.02 | 1.02 |
| covariate_shift | 1.19 | 1.19 | 1.19 | 1.05 | **0.97** | 1.08 |
| highdim_mixed | 1.08 | 1.07 | 1.10 | 0.99 | 0.96 | **0.90** |
| local_drift | 1.02 | 0.83 | 0.89 | 0.76 | 0.89 | **0.70** |
| incremental | 1.00 | 0.83 | 0.88 | 0.75 | 0.83 | **0.72** |
| **Media geométrica** | **1.036** | **0.965** | **1.008** | 0.873 | 0.936 | 0.839 |
| Datasets mejor que RF (de 11) | 5 | 6 | 6 | 9 | 8 | 7 |

En el split estático (protocolo README) el patrón es el mismo, más marcado: `RF_ventana` 0.70,
`DR_refit*` 0.77, `DR_recent_error` 0.90, `DR_hybrid` 0.94, `DR_mmd` 1.02.

### 3.2 Reales — walk-forward (6 pasos)

| Dataset | DR_mmd | DR_recent | DR_hybrid | DR_refit* | RF_pesos_exp | RF_ventana |
|---|---:|---:|---:|---:|---:|---:|
| air_quality | 1.03 | 1.06 | 1.03 | 1.05 | 1.03 | 1.08 |
| bike_sharing | 1.01 | 0.88 | 1.05 | 0.87 | 0.97 | **0.86** |
| gas_drift | 1.17 | 1.25 | 1.26 | 1.10 | 1.01 | 1.11 |
| beijing_pm25 | **0.99** | 1.08 | 1.06 | 1.08 | 1.01 | 1.20 |
| electricity | **0.91** | 1.13 | 1.06 | 1.09 | 0.98 | 1.03 |
| **Media geométrica** | 1.020 | 1.073 | 1.088 | 1.034 | 1.002 | 1.049 |

En reales **ningún modelo supera de forma consistente al RF clásico** (rank medio: RF clásico 2.6, el mejor).
Split estático real: `DR_recent_error` gana en bike (0.63) y air_quality (0.92), `DR_mmd` en electricity
(0.91); pero todos los DR pierden fuerte en gas_drift (1.43–1.89).

### 3.3 Con hiperparámetros optimizados (Optuna, split estático, RMSE test)

| Dataset | RF clásico opt | RF ventana opt | DR opt | DR vs RF clásico |
|---|---:|---:|---:|---:|
| healthcare_data | **1.256** | 1.330 | 1.344 | +7.0 % |
| healthcare_both | 6.218 | **2.165** | 3.017 | −51.5 % |
| friedman_abrupt | 3.705 | **3.564** | 3.724 | +0.5 % |
| covariate_shift | **1.758** | 1.795 | 1.914 | +8.9 % |
| highdim_mixed | 4.971 | **4.947** | 5.035 | +1.3 % |
| air_quality | 0.498 | **0.482** | 0.501 | +0.8 % |
| bike_sharing | 121.8 | **92.5** | 116.0 | −4.8 % |
| gas_drift | 0.605 | **0.570** | 0.947 | +56.5 % |
| beijing_pm25 | 68.34 | **68.26** | 69.04 | +1.0 % |
| electricity | 0.0762 | 0.0841 | **0.0731** | −4.0 % |

Con tuning, **el DR optimizado supera al RF clásico optimizado en 3 de 10 datasets** (healthcare_both,
bike_sharing, electricity). **El RF de ventana optimizado le gana al DR en 9 de 10.** Optuna eligió
`beta` alto en modo `mmd` (≈57 en healthcare_both y covariate_shift), lo que confirma que el `beta=15`
por defecto es demasiado débil para la escala del MMD² (ver §4, punto 2).

---

## 4. Problemas encontrados en el modelo

1. **`recent_error`/`hybrid` nunca entrenan con el 15 % más reciente.** Esa cola solo se usa para medir
   errores. Bajo concept drift son justo los datos más valiosos. Evidencia: `DR_recent_refit*` (que solo
   reentrena el último bloque con esa cola) mejora a `DR_recent_error` en los 11 sintéticos
   (0.965 → 0.873).
2. **En modo `mmd`, `beta` no está normalizado.** Los MMD² típicos son 0.01–0.2, así que con `beta=15`
   los pesos quedan casi uniformes (beijing: `[0.21, 0.19, 0.18, 0.21, 0.21]`). En ese caso el DR es un
   promedio de 5 bosques entrenados con 1/5 de los datos cada uno, lo que es peor que un RF con todo.
   `hybrid` sí normaliza.
3. **MMD solo ve P(X).** No detecta concept drift puro (`healthcare_concept`, `friedman_abrupt`,
   `local_drift`: AUC ≈ 0.5 y `DR_mmd` ≈ 1.0).
4. **Bug: `predict` con 1 fila falla en `mmd`/`hybrid`** (`ValueError: Se necesitan al menos 2 muestras`).
   No sirve para scoring fila a fila en producción.
5. **Predicción transductiva:** en `mmd`/`hybrid` la predicción de una fila cambia según qué otras filas
   lleguen en el mismo lote (verificado: −0.36 vs −0.44 para la misma fila). Los resultados dependen
   del tamaño y composición del lote de test.
6. **`mmd.py` tiene en `__main__` una ruta absoluta de otra máquina** (`C:\Users\mauri\...`).

---

## 5. Conclusiones

- **El README sobreestima la ventaja.** Su −26.6 % se midió solo en `healthcare_both`, el escenario más
  favorable (concept drift fuerte y monotónico), contra un RF que no tiene ningún mecanismo de recencia.
  Ahí el DR sí gana (−51 % optimizado), pero **un RF entrenado solo con datos recientes gana aún más**
  (−65 %).
- **Sintéticos con concept drift:** el DR (`recent_error`) mejora al RF clásico, pero las líneas base
  triviales de recencia (`RF_ventana`, `RF_pesos_exp`) son iguales o mejores en casi todos los casos.
- **Sin concept drift (solo data drift o sin drift):** el DR es **peor** que el RF (hasta +29 %),
  porque cada sub-bosque ve menos datos. Esto incluye `covariate_shift`, que es el caso para el que se
  pensó el modo `mmd`.
- **Datos reales:** resultado mixto y sin significancia estadística. El DR solo gana con claridad en
  bike_sharing (`recent_error`) y electricity (`mmd`); pierde en gas_drift, donde acumular todo el
  histórico es lo que mejor funciona.
- **El único escenario donde el DR aporta algo que las líneas base no tienen** es el
  **concept drift recurrente** (`friedman_recurring`: `hybrid` 0.82 vs `RF_ventana` 1.02), porque
  puede reutilizar bloques antiguos parecidos al presente.

## 6. Recomendaciones

1. Incluir `RF_ventana` y `RF_pesos_exp` como líneas base obligatorias en el README.
2. Reentrenar con la cola de validación después de calcular los pesos (variante `DR_recent_refit*`).
3. Normalizar las penalizaciones MMD (como en `hybrid`) para que `beta` signifique lo mismo en cada dataset.
4. Calcular los pesos con una referencia fija (p. ej. la ventana más reciente del train) en lugar del lote
   de test: arregla el bug de 1 fila y la dependencia del lote.
5. Enfocar el modelo en drift recurrente, que es donde tiene ventaja estructural.

---

## 7. Cómo reproducir

```powershell
.\venv_drift_forest\Scripts\python.exe -I experiments\benchmark_drift.py --suite all --seeds 0 1 2
.\venv_drift_forest\Scripts\python.exe -I experiments\tune_benchmark.py --trials 20
.\venv_drift_forest\Scripts\python.exe -I experiments\summarize_benchmark.py
```

Archivos: `experiments/synthetic_scenarios.py`, `experiments/real_datasets.py`,
`experiments/benchmark_drift.py`, `experiments/tune_benchmark.py`, `experiments/summarize_benchmark.py`,
resultados en `experiments/results/`.
