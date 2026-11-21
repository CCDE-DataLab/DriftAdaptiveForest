# Macro mensual (FRED) + incertidumbre de política (EPU)

Diseño e hipótesis registrados antes de correr en [experiments/BITACORA.md](experiments/BITACORA.md) §12.
Código: [macro_data.py](experiments/macro_data.py), [macro_experiment.py](experiments/macro_experiment.py),
[analizar_macro.py](experiments/analizar_macro.py). Figuras `13` a `15` en [experiments/figuras/](experiments/figuras/).

## Resumen

- **Ninguna de las dos hipótesis se confirma.** AdaptiveDriftForest no supera al RF clásico (de hecho pierde
  en 45 de 72 bloques) y el EPU no mejora los pronósticos de producción ni de desempleo.
- **Hallazgo consistente con finanzas:** recortar la memoria perjudica. El RF de ventana pierde contra el RF
  clásico en 51 de 72 bloques (p = 0.0005).
- **Exploratorio:** el RF clásico es el mejor modelo en **desempleo** (0.88–0.95 contra el AR a 3–12 meses),
  y mantiene ~6 % de ventaja aun sin el bloque del COVID-19.

## Diseño

- **Datos:** 25 series mensuales de FRED (API) transformadas como en FRED-MD (McCracken y Ng, 2016), más dos
  spreads; EPU mensual oficial (índice de noticias, Baker, Bloom y Davis). Muestra 1961–2026, ~780 meses.
- **Objetivos** a h = 1, 3, 6 y 12 meses: inflación IPC anualizada, crecimiento de la producción industrial
  y cambio en la tasa de desempleo.
- **Modelos:** AR (referencia), índice de difusión de Stock y Watson (AR + 3 factores), ARX lineal con todo,
  RF clásico, RF ventana, DriftRegressor original y AdaptiveDriftForest; versiones con EPU.
- **Protocolo:** walk-forward con 6 bloques de prueba (1987–2025) y embargo de h meses; 3 semillas; sin tuning.

## Resultados (walk-forward, RMSE / RMSE del AR; menos de 1 = mejor)

| Tarea | Índice de difusión | ARX | RF clásico | RF ventana | AdaptiveDriftForest |
|---|---:|---:|---:|---:|---:|
| Inflación h=1 | 1.01 | 1.09 | **0.98** | 1.10 | 1.02 |
| Inflación h=3 | 1.06 | 1.19 | 1.06 | 1.37 | 1.11 |
| Inflación h=6 | 1.08 | 1.32 | 1.09 | 1.35 | 1.03 |
| Inflación h=12 | 1.07 | 1.29 | 1.09 | 1.09 | 1.02 |
| Producción h=1 | 1.01 | 1.06 | **0.96** | 0.99 | 1.02 |
| Producción h=3 | 1.09 | 1.07 | 1.00 | 1.03 | 1.00 |
| Producción h=6 | 1.15 | 1.13 | 1.05 | 1.05 | 1.11 |
| Producción h=12 | 1.12 | 1.13 | 1.06 | 1.15 | 1.33 |
| Desempleo h=1 | 0.98 | 0.96 | **0.95** | 0.99 | 0.98 |
| Desempleo h=3 | 0.98 | 0.96 | **0.89** | 0.95 | 0.96 |
| Desempleo h=6 | 1.03 | 1.02 | **0.88** | 0.95 | 1.02 |
| Desempleo h=12 | 1.05 | 1.09 | **0.92** | 0.98 | 1.11 |
| **Media geométrica** | 1.051 | 1.105 | **0.990** | 1.074 | 1.056 |

| Hipótesis registrada | Resultado | Bloques | p (signo) |
|---|---|---:|---:|
| H3: AdaptiveDriftForest mejor que RF clásico | **No se confirma** (va en contra) | 27/72 | 0.044 |
| H4: índice de difusión + EPU mejor que sin EPU (producción y desempleo, h = 6 y 12) | **No se confirma** | 12/24 | 1.00 |
| H4: AR + EPU mejor que AR (mismas tareas) | **No se confirma** | 14/24 | 0.54 |

## Interpretación

1. **Con pocos datos el modelo adaptativo se debilita.** Con 300–700 meses de entrenamiento, la cola de
   validación del 15 % es de 45–100 meses, demasiado poco para elegir bien la memoria. Sus pesos saltan de un
   bloque a otro y pierde contra el RF clásico. En macro mensual el modelo no es recomendable.
2. **Recortar memoria perjudica, como en finanzas.** Es el resultado más robusto de todo el proyecto: en datos
   reales con ciclos que se repiten (recesiones, inflación), el histórico completo vale más que lo reciente.
3. **Los árboles ganan en desempleo.** Es coherente con una dinámica no lineal (el desempleo sube rápido en las
   recesiones y baja lento en las recuperaciones), que un AR lineal no captura. Fuera del COVID-19 la ventaja baja
   a 6 % y gana en 11 de 20 bloques, así que es una pista, no una conclusión.
4. **En el COVID-19 los árboles ganan por la misma razón por la que pierden en finanzas:** no extrapolan.
   Ante saltos extremos (abril de 2020), el AR proyecta rebotes enormes y los árboles se quedan en rangos
   históricos (figura 14).
5. **El EPU no aporta a estos horizontes** con este diseño: los efectos están dentro de ±5 % y sin dirección clara.

## Limitaciones

- Datos revisados actuales, no las versiones disponibles en cada fecha (ALFRED). Esto favorece a todos los modelos por igual,
  pero sobrestima la precisión real.
- Panel reducido: 25 series contra ~127 de FRED-MD. Puede explicar por qué el índice de difusión no supera al AR,
  a diferencia de la literatura.
- El bloque 2019–2025 incluye el COVID-19 y pesa mucho en el RMSE.
- Prueba de signo por bloques, sin corrección por correlación entre tareas ni por comparaciones múltiples.
