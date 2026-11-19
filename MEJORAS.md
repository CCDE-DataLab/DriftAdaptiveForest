# Mejoras al modelo — AdaptiveDriftForest

Rama `mejoras-modelo-drift`. Nueva clase en
[src/drift_adaptive_forest/models/adaptive_ensemble.py](src/drift_adaptive_forest/models/adaptive_ensemble.py).
El `DriftRegressor` original queda intacto para comparar. Diagnóstico de partida en [CONCLUSIONES.md](CONCLUSIONES.md).

## Qué cambia respecto al DriftRegressor

| Problema del original | Solución en `AdaptiveDriftForest` |
|---|---|
| Solo combina bloques de 1/K de los datos; sin drift pierde contra el RF | Los candidatos incluyen **ventanas recientes anidadas** (10 %, 20 %, 35 %, 60 %, 100 %) además de los bloques. La ventana del 100 % es el RF clásico, así que el ensamble puede quedarse con la mejor línea base |
| `recent_error` nunca entrena con el 15 % más reciente | Los pesos se estiman sin esa cola y luego los candidatos se **reentrenan incluyéndola** |
| `beta` en modo `mmd` depende de la escala del MMD² | Penalización = MSE / MSE mínimo, así **`beta` significa lo mismo en todo dataset** (beta=20: un candidato 5 % peor recibe e⁻¹ del peso) |
| `predict` falla con 1 fila y depende del lote | **Pesos fijos desde `fit`**; predicción determinista fila a fila |
| MMD contra el test (transductivo) | Modo `hybrid` opcional: MMD de cada candidato contra la cola reciente, no contra el test |

```python
from src.drift_adaptive_forest.models.adaptive_ensemble import AdaptiveDriftForest

modelo = AdaptiveDriftForest(weighting="error", beta=20.0, random_state=42)
modelo.fit(X_train, y_train)          # X ordenado en el tiempo (o pasar timestamps=)
modelo.predict(X_nuevo)               # funciona con 1 fila
modelo.weights_summary()              # p. ej. {'ventana_0.10': 1.0}
```

## Resultados

Mismo benchmark de [experiments/BITACORA.md](experiments/BITACORA.md), sin tuning (valores por defecto).
Error relativo = RMSE / RMSE del RF clásico (media geométrica). Menos de 1 = mejor.

| | ADF_error (nuevo) | RF ventana reciente | DriftRegressor original (mejor modo) |
|---|---:|---:|---:|
| Sintéticos, walk-forward | **0.809** | 0.839 | 0.965 |
| Sintéticos, split 80/20 | **0.698** | 0.699 | 0.897 |
| Reales, walk-forward | **0.983** | 1.049 | 1.020 |
| Reales, split 80/20 | **0.919** | 1.004 | 0.990 |

- **Sintéticos (walk-forward):** mejor que el RF clásico en 10 de 11 escenarios, 81 % de los pasos,
  Wilcoxon p = 0.002. Rank medio 2.0, el mejor de los 9 modelos.
- **Sin drift o solo data drift** ya no pierde: `healthcare_none` 0.98, `healthcare_data` 1.00,
  `covariate_shift` 0.98 (el original perdía hasta +29 %).
- **Escenario del README** (`healthcare_both`): 0.53, mejor que el RF de ventana (0.57).
- **Reales:** primer modelo que supera en promedio al RF clásico en ambos protocolos. `gas_drift`
  pasó de 1.25 a 1.00. La ventaja en reales es modesta y sin significancia estadística (5 datasets).

## Límites

- En **drift recurrente** el `DriftRegressor` original con `hybrid` sigue siendo mejor
  (`friedman_recurring`: 0.82 vs 0.97). En **Electricity**, el `DR_mmd` original también (0.91 vs 1.05):
  ahí ayuda adaptar los pesos a cada lote de test.
- Costo: entrena ~2× los candidatos (antes y después de incluir la cola). En `gas_drift` (134 variables)
  son ~90 s por ajuste.

## Siguientes pasos posibles

1. Validar los pesos con varios cortes recientes en vez de uno (menos varianza en la elección).
2. Modo opcional que adapte pesos al lote de test (como el original) para casos tipo Electricity.
3. Tuning con Optuna del nuevo modelo (`beta`, `window_fracs`, `val_frac`).

Reproducir:
```powershell
.\venv_drift_forest\Scripts\python.exe -I experiments\benchmark_drift.py --suite all --seeds 0 1 2 --models ADF_error ADF_hybrid --no-diag --tag adf
.\venv_drift_forest\Scripts\python.exe -I experiments\summarize_benchmark.py
```
