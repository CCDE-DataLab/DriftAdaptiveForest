# Finanzas cuantitativas — ¿ayuda un modelo con drift a predecir volatilidad?

Experimento sobre 13 series financieras con cambios de régimen. Registro paso a paso en
[experiments/BITACORA.md](experiments/BITACORA.md) (secciones 9 y 10). Código:
[finance_data.py](experiments/finance_data.py), [finance_experiment.py](experiments/finance_experiment.py),
[graficar_analisis.py](experiments/graficar_analisis.py). Figuras en [experiments/figuras/](experiments/figuras/).

## Resumen

**En volatilidad a 5 días, un modelo lineal (HAR-X) supera a todos los modelos de árboles, con o sin
manejo de drift.** Recortar la memoria (ventanas recientes) empeora el pronóstico. El `AdaptiveDriftForest`
evita ese daño, porque aprende a usar ventanas largas, pero no supera al RF clásico de forma significativa
ni se acerca al HAR-X. La hipótesis de un híbrido lineal + bosque sobre residuos **no se confirmó** en
series nuevas.

## Diseño

- **Objetivo:** log de la volatilidad realizada de los próximos 5 días hábiles (para el VIX, su nivel en t+5).
- **Predictores (solo información hasta t):** volatilidad propia a 1, 5 y 22 días, nivel y cambio del VIX,
  retornos a 5 días de S&P 500, cobre y dólar global, y caída desde el máximo del S&P 500.
- **Protocolo:** walk-forward con ventana expansiva (6 bloques de prueba) y split 80/20, con **embargo de 5 días**
  para que el objetivo del entrenamiento no se solape con el test. Sin tuning. Diseño fijado antes de ver resultados.
- **Fuentes públicas sin API key:** Yahoo Finance y BCRPData (tipo de cambio PEN/USD, código `PD04638PD`).

| Modelo | Qué es |
|---|---|
| Persistencia | La volatilidad de los últimos 5 días como pronóstico |
| HAR | Regresión lineal sobre volatilidad propia a 1, 5 y 22 días (Corsi, 2009). **Referencia** |
| HAR-X | Regresión lineal con **todas** las variables que reciben los árboles |
| RF clásico, RF ventana | Random Forest con todo el histórico / con el 20 % más reciente |
| DriftRegressor | Modelo original del repo (`recent_error`) |
| AdaptiveDriftForest | Modelo mejorado de [MEJORAS.md](MEJORAS.md) |
| HAR-X + RF / ADF sobre residuos | Híbridos: el lineal da el nivel y el bosque una corrección no lineal |

## Resultados (error relativo al HAR, walk-forward; menos de 1 = mejor)

### 9 series exploradas

| Serie | HAR-X | RF clásico | RF ventana | AdaptiveDriftForest |
|---|---:|---:|---:|---:|
| VIX (nivel) | 1.00 | 1.10 | 1.25 | 1.15 |
| VIX + estructura de plazos | 1.00 | 1.12 | 1.37 | 1.24 |
| S&P 500 | **0.88** | 0.91 | 1.02 | 0.92 |
| PEN/USD | 1.01 | 1.07 | 1.14 | 1.08 |
| SVXY | 1.05 | 1.06 | 1.07 | 1.11 |
| USD/JPY | 0.99 | 1.09 | 1.15 | 1.08 |
| Brent | 1.00 | 1.06 | 1.12 | 1.06 |
| Bono 10 años | 1.01 | 1.07 | 1.09 | 1.08 |
| Bitcoin | 1.03 | 1.08 | 1.23 | 1.13 |
| **Media geométrica** | **0.995** | 1.060 | 1.156 | 1.090 |

Comparación bloque a bloque (54 bloques = 9 series × 6):
- HAR-X gana a AdaptiveDriftForest en **51/54** y al RF clásico en **49/54**.
- RF ventana pierde contra RF clásico en **40/54** (p = 0.0005): en volatilidad, recortar la memoria perjudica.
- AdaptiveDriftForest contra RF clásico: **27/54**, un empate.

### Confirmación en 4 series nunca usadas (hipótesis registrada antes de correr)

| Serie | HAR-X | RF clásico | AdaptiveDriftForest | HAR-X + RF resid. | HAR-X + ADF resid. |
|---|---:|---:|---:|---:|---:|
| EUR/USD | 1.01 | 1.11 | 1.06 | 1.02 | 1.01 |
| Oro | **0.99** | 1.06 | 1.06 | 1.02 | 1.03 |
| Nasdaq-100 | 0.92 | 1.00 | 0.98 | 0.94 | **0.92** |
| Russell 2000 | **0.92** | 0.99 | 1.02 | 0.95 | 0.93 |
| **Media geométrica** | **0.959** | 1.037 | 1.031 | 0.980 | 0.970 |

| Hipótesis | Resultado | Bloques | p (signo) |
|---|---|---:|---:|
| H1: HAR-X + ADF residual mejor que HAR-X | **No se confirma** | 10/24 | 0.54 |
| H2: ADF residual mejor que RF residual | Sugerente, **no significativa** | 17/24 | 0.064 |
| (No registrada) HAR-X mejor que HAR | Se confirma | 19/24 | 0.007 |
| (No registrada) Añadir un RF residual al HAR-X empeora | Se confirma | 18/24 | 0.023 |

## Interpretación

1. **El drift en volatilidad es sobre todo de nivel, y la dinámica es estable y casi lineal.** Un modelo lineal
   en rezagos se adapta solo a niveles nuevos; los árboles no extrapolan fuera de lo que vieron. Es el mismo
   patrón del escenario sintético `covariate_shift`.
2. **La memoria larga es la correcta.** La volatilidad tiene memoria larga, y por eso las ventanas cortas fallan.
   El AdaptiveDriftForest lo aprende solo (ver los pesos en `06_finanzas_pesos_*.png`), y por eso no sufre el daño
   del RF de ventana. En Bitcoin durante el COVID-19 eligió la ventana del 10 % y le costó caro.
3. **La información cruzada sí aporta:** VIX, cobre y dólar mejoran al HAR (HAR-X gana en 19 de 24 bloques en
   series nuevas). La ventaja que al principio parecía de los árboles era de estas variables.
4. **El manejo del drift muestra una señal débil,** no concluyente: AdaptiveDriftForest le gana al RF clásico en
   16/24 bloques nuevos, y como corrección residual en 17/24.

## Limitaciones

- Solo horizonte de 5 días y solo volatilidad. Con horizontes largos o retornos el resultado puede cambiar.
- Los bloques de prueba son grandes (~2 años); no miden la velocidad de adaptación tras un quiebre puntual.
- Datos de Yahoo con tramos de precio repetido (USD/JPY 2008, Brent 2009), que dan volatilidad 0. Están en el
  entrenamiento y deberían perjudicar más a los modelos lineales; no se limpiaron para no cambiar el diseño.
- La prueba de signo por bloques no corrige la correlación entre bloques ni entre series; falta un
  test de Diebold-Mariano.
- Se vio un corte puntual de EUR/USD en una prueba de código antes de la corrida de confirmación (ver bitácora).

## Siguientes pasos sugeridos

1. Test de Diebold-Mariano por serie.
2. Medir la adaptación tras quiebres con bloques cortos alrededor de eventos (SVXY 2018, COVID-19).
3. Horizontes de 22 días y retornos, donde la dinámica es menos lineal.
4. Datos macro con inestabilidad documentada (FRED-MD mensual o el panel de Stock y Watson).
