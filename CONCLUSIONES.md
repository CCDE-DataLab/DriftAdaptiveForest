# Conclusiones — DriftRegressor vs Random Forest

Benchmark de 11 escenarios sintéticos (4–20 variables, 3 semillas) y 5 datasets reales con drift
documentado (Air Quality, Bike Sharing, Gas Sensor Drift, Beijing PM2.5, Electricity).
Detalle completo, metodología y registro de pasos en [experiments/BITACORA.md](experiments/BITACORA.md).

## Resumen

El `DriftRegressor` solo supera con claridad al Random Forest convencional cuando hay **concept drift
fuerte**. Aun ahí, un **RF entrenado solo con los datos más recientes** iguala o supera al modelo
adaptativo. Sin concept drift, el modelo adaptativo es **peor** que el RF convencional.

## Resultados

Error relativo = RMSE del modelo / RMSE del RF convencional (media geométrica). Menos de 1 = mejor que el RF.

| | Adaptativo (mejor modo) | RF con datos recientes |
|---|---|---|
| Sintéticos, walk-forward | `recent_error`: 0.97 | 0.84 |
| Reales, walk-forward | `mmd`: 1.02 | 1.05 |

- **Escenario del README (data + concept drift), optimizado con Optuna:** RF convencional 6.22,
  adaptativo 3.02, RF con datos recientes 2.16.
- **Solo data drift o sin drift:** el adaptativo es hasta 29 % peor, porque cada sub-bosque ve solo
  una fracción de los datos.
- **Datos reales:** ningún modelo supera de forma consistente al RF convencional. El adaptativo gana en
  Bike Sharing y Electricity, y pierde fuerte en Gas Drift (+56 % de error, optimizado).
- **Con Optuna:** el adaptativo supera al RF convencional en 3 de 10 datasets; el RF de ventana reciente
  le gana al adaptativo en 9 de 10.
- **Donde sí aporta algo propio:** concept drift recurrente (`friedman_recurring`: `hybrid` 0.82 vs
  RF reciente 1.02), porque puede reutilizar bloques antiguos parecidos al presente.

El −26.6 % del README se midió en el escenario más favorable y sin comparar contra un RF de datos recientes.

## Problemas encontrados en el modelo

1. En `recent_error` e `hybrid`, el 15 % más reciente se usa solo para calcular pesos y nunca para
   entrenar. Una variante que reentrena el último bloque con esa cola mejoró en los 11 sintéticos
   (0.97 → 0.87).
2. En modo `mmd`, `beta=15` es demasiado débil para la escala del MMD² y deja pesos casi uniformes.
   Optuna eligió `beta` ≈ 57.
3. **Bug:** `predict` con una sola fila falla en `mmd` y `hybrid`.
4. La predicción de una fila cambia según qué otras filas lleguen en el mismo lote.
5. El MMD solo detecta cambios en X; no ve concept drift puro.

## Recomendaciones

1. Agregar al README la comparación contra un RF de ventana reciente y un RF con pesos exponenciales.
2. Reentrenar con la cola de validación después de calcular los pesos.
3. Normalizar las penalizaciones MMD para que `beta` signifique lo mismo en cada dataset.
4. Calcular los pesos contra una referencia fija (la ventana más reciente del train) en lugar del lote de test.
5. Enfocar el modelo en drift recurrente, donde tiene ventaja estructural.
