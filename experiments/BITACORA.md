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

---

## 8. Rama `mejoras-modelo-drift` (19 nov 2026)

| # | Paso | Resultado |
|---|---|---|
| 10 | Nueva clase `AdaptiveDriftForest` (`models/adaptive_ensemble.py`): ventanas recientes anidadas + bloques, error normalizado, reentrenamiento con la cola, pesos fijos desde `fit` | Corrige los 5 problemas de §4. Verificado: predice 1 fila y es independiente del lote (dif. 1e-15). |
| 11 | Benchmark de `ADF_error` y `ADF_hybrid` con los mismos datasets, semillas y protocolos | `experiments/results/benchmark_*_adf.csv` |
| 12 | Resultado | Sintéticos walk-forward 0.809 (10/11, p=0.002); reales walk-forward 0.983 y estático 0.919. Mejor modelo en rank medio en los 4 cortes. Detalle en `MEJORAS.md`. |
| 13 | Validación con semillas nuevas 3–5 (sintéticos, 8 modelos) | ADF_error walk-forward 0.810 (10/11, p = 0.005), 80/20 0.691. Resultado estable; `experiments/results/summary_holdout.md`. Limitación: mismos generadores. |
| 14 | README actualizado (benchmark ampliado, advertencia sobre el resultado original, estructura del repo) | `README.md` |

---

## 9. Finanzas cuantitativas (20 nov 2026)

| # | Paso | Resultado |
|---|---|---|
| 15 | Experimento de volatilidad a 5 días: VIX, S&P 500, PEN/USD (BCRPData), SVXY; luego USD/JPY, Brent, bono 10 años, Bitcoin, VIX + estructura de plazos. Walk-forward con embargo de 5 días, sin tuning, diseño fijado antes de ver resultados. | `experiments/finance_experiment.py`, `experiments/results/finance_results*.csv` |
| 16 | Incidente: un parámetro `--out` mal aplicado sobrescribió `finance_results.csv`. Se corrigió y se regeneró todo; los números coinciden exactamente con los reportados antes (semillas fijas, datos en caché). | — |
| 17 | HAR-X (OLS con las mismas variables que los árboles) gana a AdaptiveDriftForest en 51/54 bloques y al RF clásico en 49/54. La ventaja previa de los árboles en S&P 500 venía de las variables (VIX), no de los árboles ni del manejo del drift. RF de ventana pierde contra RF clásico en 40/54 bloques (p = 0.0005): en volatilidad, recortar memoria perjudica. | `experiments/figuras/04_*` a `06_*` |

### Hipótesis registrada ANTES de correr (paso 18)

Motivación (surge de ver los resultados del paso 17, por eso se valida fuera de muestra): los árboles no extrapolan
niveles nuevos de volatilidad, cosa que sí hace un modelo lineal. Se propone un modelo híbrido:
`HARX + bosque sobre los residuos del HARX` (el lineal captura el nivel, el bosque la corrección no lineal).

- Modelos nuevos: `HARX+RF_resid` (RF, `min_samples_leaf=20`) y `HARX+ADF_resid` (AdaptiveDriftForest,
  `min_samples_leaf=20`). La hoja mínima de 20 se fija a priori porque los residuos son casi ruido.
- **H1:** `HARX+ADF_resid` tiene menor RMSE que `HARX` en walk-forward.
- **H2:** `HARX+ADF_resid` tiene menor RMSE que `HARX+RF_resid` (el manejo del drift aporta sobre la corrección).
- **Confirmación:** solo cuenta lo que pase en 4 series NO usadas hasta ahora: EUR/USD, oro (GC=F),
  Nasdaq-100 (^NDX) y Russell 2000 (^RUT), mismo protocolo, mismas variables, sin cambios después de ver resultados.
  Los 9 objetivos ya vistos se reportan solo como exploratorios.
- Transparencia: la prueba rápida del código usó un solo corte de EUR/USD (filas 2000–2300), así que se vio un dato
  puntual de una serie de confirmación antes de la corrida. No se cambió nada del diseño después de eso.

## 10. Resultado de la hipótesis registrada (20 nov 2026)

| # | Paso | Resultado |
|---|---|---|
| 18 | Híbridos HAR-X + bosque sobre residuos en las 9 series exploradas | Exploratorio: media geométrica 1.003 (ambos híbridos) contra 0.995 del HAR-X. Sin mejora. |
| 19 | Confirmación en EUR/USD, oro, Nasdaq-100 y Russell 2000 | **H1 no se confirma** (10/24 bloques, p = 0.54; 0.970 contra 0.959 del HAR-X). **H2 sugerente, no significativa** (17/24, p = 0.064). HAR-X mejor que HAR en 19/24 (p = 0.007). |
| 20 | Documentación | `FINANZAS.md`, figuras `04` a `08` en `experiments/figuras/`. |

## 11. Exploratorio: incertidumbre de política económica (EPU) (20 nov 2026)

| # | Paso | Resultado |
|---|---|---|
| 21 | EPU diario de EE. UU. (policyuncertainty.com, 1985–2026), rezagado 1 día. `experiments/eda_epu.py`, figuras `09` a `12`. | Exploratorio, genera hipótesis. |
| 22 | Relación EPU–VIX | La correlación móvil de 1 año va de −0.49 a 0.93 (mediana 0.50): la relación misma tiene drift. Negativa en 2017–2018; desacople desde los aranceles de 2025 (EPU alto, VIX normal). |
| 23 | Adelanto o atraso | En S&P 500 la correlación máxima está en h = −5 días y en el bono en h = −10: la volatilidad de mercado se mueve antes que el EPU. |
| 24 | HAR-X + EPU fuera de muestra (lineal) | Ayuda en 29/72 bloques; solo mejora SVXY (−1.6 %). Sin información adicional a 5 días. |
| 25 | EPU del bloque contra beneficio de adaptarse | Correlación 0.01 (ADF/RF) y −0.03 (ventana/RF): el nivel de EPU no indica cuándo conviene acortar la memoria. No se sigue con la idea de usarlo como interruptor de memoria a 5 días. |

## 12. Macro mensual tipo FRED-MD + EPU (20 nov 2026) — diseño registrado ANTES de correr

Datos: 25 series mensuales de FRED vía API (actividad, empleo, vivienda, consumo, dinero, crédito, tasas, precios),
transformadas con los códigos de McCracken y Ng (2016), más dos spreads; EPU mensual oficial (índice de noticias).
Muestra 1961–2026 (~790 meses). Datos revisados actuales, no en tiempo real (limitación conocida).

- **Objetivos** (h = 1, 3, 6, 12 meses): inflación IPC anualizada `(1200/h)·log(IPC_{t+h}/IPC_t)`, crecimiento
  de la producción industrial (misma forma) y cambio del desempleo `u_{t+h} − u_t`. 12 tareas.
- **Regresores en t:** 4 rezagos propios del objetivo (mensual t, t−1, t−2 y 12 meses) + panel transformado (27).
  Variante "+EPU": log EPU, media de 3 meses y desvío respecto de su media de 12 meses.
- **Modelos:** AR (referencia, OLS con rezagos propios), DI (Stock y Watson: AR + 3 componentes principales del
  panel, ajustados solo con el train), ARX (OLS con todo), RF clásico, RF ventana, DriftRegressor original,
  AdaptiveDriftForest. Versiones +EPU de AR, DI, ARX, RF clásico y AdaptiveDriftForest.
- **Protocolo:** walk-forward (6 bloques, orígenes 40 %–90 %) y 80/20, embargo de h meses, 3 semillas, sin tuning.
- **H3 (drift):** AdaptiveDriftForest tiene menor RMSE que RF clásico (72 bloques walk-forward, prueba de signo).
- **H4 (canal macro del EPU):** DI+EPU tiene menor RMSE que DI en producción industrial y desempleo a h = 6 y 12
  (24 bloques, prueba de signo). Mismo contraste para AR+EPU contra AR.
- Se reportan todas las tareas; hay comparaciones múltiples, así que un p cercano a 0.05 se trata como débil.

| # | Paso | Resultado |
|---|---|---|
| 26 | Corrida macro (12 tareas × 12 modelos × 7 cortes). La PC entró en suspensión a las 18:10 y pausó el cálculo; terminó sin errores. | `experiments/results/macro_results.csv` |
| 27 | H3 | **No se confirma**: AdaptiveDriftForest < RF clásico en 27/72 bloques (p = 0.044, en contra). |
| 28 | H4 | **No se confirma**: DI+EPU < DI en 12/24 (p = 1.0); AR+EPU < AR en 14/24 (p = 0.54). |
| 29 | Exploratorio | RF ventana < RF clásico en 21/72 (p = 0.0005), igual que en finanzas. RF clásico gana al AR en desempleo (0.88–0.95); sin el bloque COVID, 0.94 y 11/20 bloques. Documentado en `MACRO.md`. |

## 13. Demanda de matrícula universitaria (IPEDS, EE. UU.) (20 nov 2026) — diseño registrado ANTES de correr

Datos: matrícula de otoño por institución (IPEDS/NCES), 2002–2023, total de estudiantes (`EFTOTLT` desde 2008;
`EFRACE24` en 2002–2007, verificado contra 2008). Panel balanceado: 3 196 instituciones con ≥ 100 alumnos todos
los años. Limitación conocida: **sesgo de supervivencia** (se excluyen las que cerraron, que es parte del drift real);
el sector se toma del directorio 2023.

- **Objetivo:** crecimiento log de la matrícula a h = 1 y 2 años, `log E_{t+h} − log E_t`.
- **Variables en t:** crecimiento de los últimos 3 años (t, t−1, t−2), crecimiento medio de 3 años, tamaño (log),
  tipo de institución (pública / privada sin fines / con fines de lucro) y nivel (4 años / 2 años / < 2 años),
  crecimiento promedio nacional y del mismo tipo de institución en t.
- **Protocolo:** origen rodante anual. Para cada año de origen o (2012 en adelante) se entrena con todas las filas
  cuyo objetivo ya se conocía en o (`t + h ≤ o`) y se pronostica `t = o`. 11 cortes para h=1, 10 para h=2.
  Sin tuning; 2 semillas.
- **Modelos:** ingenuo (crecimiento 0), tendencia (promedio de 3 años), ETS (Holt amortiguado sobre el log, por
  institución), OLS global, LightGBM global (parámetros por defecto), RF clásico, RF ventana, DriftRegressor
  original, AdaptiveDriftForest.
- **Métrica principal:** RMSE del crecimiento log; secundaria: MAE.
- **H5:** AdaptiveDriftForest < RF clásico (21 cortes, prueba de signo).
- **H6:** AdaptiveDriftForest < LightGBM.
- **H7:** AdaptiveDriftForest < ETS.
- **H8:** la ventaja de AdaptiveDriftForest sobre RF clásico es mayor en privadas con fines de lucro que en públicas.

| # | Paso | Resultado |
|---|---|---|
| 30 | Corrida matrícula (21 cortes, 9 modelos, ~50 min) | `experiments/results/matricula_results.csv` |
| 31 | H5 | **Se confirma**: AdaptiveDriftForest < RF clásico en 20/21 cortes (MAE 21/21), p < 0.001. |
| 32 | H6 | **No se confirma**: < LightGBM en 8/21 (MAE 7/21). En promedio gana a h=2 (0.983 contra 1.001) pero no corte a corte. |
| 33 | H7 | **Se confirma**: < ETS en 21/21. ETS es peor que el ingenuo en los 21 cortes. |
| 34 | H8 | **No se confirma** (9/21). |
| 35 | Corrección de estilo | El naranja de alerta se estaba usando como color de serie (ETS, HAR-X, DI+EPU); se cambió a azul medio (#4F6D8A) según la paleta CCDE. |

## 14. Costos operativos (20 nov 2026)

Matrícula IPEDS, h = 1, origen 2022: 54 000 filas de entrenamiento × 13 variables, 14 núcleos. Cada modelo en un proceso
aparte; RAM = pico del conjunto de trabajo de Windows. `experiments/costos_operativos.py`, `results/costos_operativos.csv`.

| Modelo | Entrenar | CPU total | Predecir 3 196 filas | Predecir 1 fila | RAM extra | Modelo guardado |
|---|---:|---:|---:|---:|---:|---:|
| OLS global | 0.02 s | 0.0 s | 0.001 s | 0.2 ms | 13 MB | < 0.1 MB |
| LightGBM | 0.20 s | 1.7 s | 0.004 s | 1.1 ms | 11 MB | 0.3 MB |
| RF clásico | 6.6 s | 68 s | 0.06 s | 31 ms | 503 MB | 472 MB |
| DriftRegressor original | 5.4 s | 52 s | 0.19 s | 150 ms | 428 MB | 405 MB |
| AdaptiveDriftForest | 37.6 s | 386 s | 0.47 s | 289 ms | 2 793 MB | 1 444 MB |

Nota: la primera medición de RAM daba 0 por un error de tipos en ctypes (handle truncado a 32 bits); se corrigió.

## 15. Versión barata del "MMD-split": tiempo en los primeros niveles (20 nov 2026) — registrado ANTES de correr

Propuesta del usuario: cambiar el criterio de CART en los 3 primeros niveles para minimizar MSE + η·MMD (datos viejos
contra recientes dentro de cada hijo) y luego actuar como RF clásico. Antes de implementarla se prueba una versión barata
que captura la misma intuición (adaptación local por región):

- **RF+t:** RF clásico con el tiempo t como variable adicional.
- **Part3+RF:** árbol de 3 niveles sobre [X, t] (hoja mínima = 5 % de las filas) que define hasta 8 regiones;
  dentro de cada región, un RF clásico sobre X. Los datos futuros tienen t mayor que todo el entrenamiento, así que
  caen en la rama más reciente de cualquier corte en t.
- **Datos:** sintéticos `local_drift`, `healthcare_both`, `friedman_abrupt`, `incremental` y controles sin concept
  drift `healthcare_none` y `covariate_shift` (semillas 0–2, walk-forward y 80/20, mismos cortes que §7);
  reales `bike_sharing` y `electricity`; matrícula IPEDS (mismo protocolo que §13, semillas 0–1).
- **H9:** Part3+RF tiene menor RMSE que RF clásico en `local_drift` (walk-forward, 18 bloques).
- **H10:** en los controles, RF+t y Part3+RF no empeoran más de 2 % respecto de RF clásico (media geométrica).
- **H11:** en matrícula, Part3+RF o RF+t tienen menor RMSE que RF clásico (21 cortes).
- **Regla de decisión:** se implementa el MMD-split real solo si se cumplen H9 y H11.

| # | Paso | Resultado |
|---|---|---|
| 36 | Corrida (`experiments/tiempo_experiment.py`). RF clásico recalculado coincide con corridas previas (dif. < 1e-15). | `tiempo_results.csv`, `tiempo_matricula_results.csv` |
| 37 | H9 | **No se confirma** por la regla registrada: Part3+RF < RF clásico en 10/18 bloques de `local_drift` (p = 0.81), aunque la media geométrica mejora 31 % (los bloques previos al drift no pueden mejorar). |
| 38 | H10 | Se cumple para RF+t (1.00 y 1.01 en controles); **falla** para Part3+RF (1.08 en `covariate_shift`). |
| 39 | H11 | **No se confirma**: en matrícula RF+t < RF clásico en 6/21, Part3+RF en 8/21. AdaptiveDriftForest gana a RF+t en 21/21. |
| 40 | Decisión | Por la regla registrada, **no se implementa todavía el MMD-split**. Hallazgo exploratorio: RF+t (un solo RF con t) es el mejor modelo en `local_drift` (0.58), Bike Sharing (0.81) y empata con ADF en `healthcare_both`, a ~1/6 del costo; pero no sirve en matrícula ni Electricity. Part3+RF casi no corta en t (0–1.2 de 7 cortes): limitar el tiempo a los 3 primeros niveles es lo que falla. |

## 16. Comparación con métodos de la literatura (20 nov 2026) — registrado ANTES de correr

Motivación: el paper (`paper/main.tex`, Discusión) declara que no se comparó con métodos de flujos de datos.

- **Modelos nuevos:** Adaptive Random Forest para regresión (`river.forest.ARFRegressor`, 10 árboles, detectores
  ADWIN por defecto) y Hoeffding Adaptive Tree para regresión (`river.tree.HoeffdingAdaptiveTreeRegressor`), ambos
  con `StandardScaler` en línea como recomienda la documentación de River; LightGBM (parámetros por defecto); RF+t.
- **Protocolo:** el mismo banco y cortes de §7 (11 sintéticos × semillas 0–2, 5 reales con semilla 0, walk-forward
  y 80/20) y la matrícula de §13 (semillas 0–1). Los modelos de River aprenden fila a fila en orden temporal hasta el
  origen y predicen el bloque de prueba **sin aprender de él** (equivale a reentrenar en cada origen, igual que los
  modelos por lotes). Sin tuning.
- **H12:** AdaptiveDriftForest (modo error) tiene menor RMSE que ARF en walk-forward sintético: cociente geométrico
  < 1 y gana en la mayoría de los 11 escenarios.
- **H13:** AdaptiveDriftForest tiene menor RMSE que LightGBM en walk-forward sintético (mismo criterio).
- **H14:** en matrícula, AdaptiveDriftForest tiene menor RMSE que ARF (21 cortes, prueba de signo).
- Datos reales del banco: se reportan, sin hipótesis (5 conjuntos no dan potencia).

| # | Paso | Resultado |
|---|---|---|
| 41 | Corrida (`experiments/comparacion_literatura.py`) | `lit_benchmark.csv`, `lit_matricula.csv` |
| 42 | H12 | **Se confirma**: ADF-E < ARF en 10/11 sintéticos; cociente ADF/ARF 0.814. |
| 43 | H13 | **Se confirma**: ADF-E < LightGBM en 7/11 sintéticos; cociente 0.821. |
| 44 | H14 | **No se confirma** (en contra): en matrícula ARF < ADF-E en 16/21 cortes (p = 0.027), aunque en media geométrica difieren < 1 %. |
| 45 | Reales del banco (sin hipótesis) | LightGBM tiene el menor cociente medio (0.968, mejor que RF en 4/5); ADF-E 0.983; ARF 1.27 y HAT 1.43 (peores que RF en 5/5). |
| 46 | Paper v2 | `paper/main_v2.tex` (copia de `main.tex` con secciones nuevas), `paper/generar_tablas_v2.py`, `paper/referencias_v2.bib` (10 referencias nuevas, 9 verificadas en Crossref; `ke2017` marcada VERIFICAR). Se quitó el bloque de copyright de la portada a pedido del usuario. El original `main.tex` no se modificó. |
