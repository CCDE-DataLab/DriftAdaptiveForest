# DriftAdaptiveForest

Framework experimental en Python (compatible con Scikit-Learn) para mitigar **Data Drift** ($P(X)$ cambia en el tiempo) y **Concept Drift** ($P(Y \mid X)$ cambia en el tiempo) en problemas de **regresión temporal** utilizando algoritmos basados en árboles de decisión.

---

## Motivación

En datos tabulares con evolución temporal, entrenar un `RandomForestRegressor` clásico (o incluso pasar `sample_weight` con decaimiento exponencial) presenta dos limitaciones:
1. El **bootstrap uniforme** sigue incluyendo un alto porcentaje de muestras históricas obsoletas en cada árbol, reduciendo el tamaño efectivo de muestra reciente.
2. Los árboles de decisión **no extrapolan** fuera del rango de entrenamiento y promedian regímenes históricos incompatibles con el presente.

`DriftAdaptiveForest` explora tres estrategias progresivas para resolver este problema.

---

## Estrategias del Framework

### 1. Estrategia A — Bootstrap Temporal Probabilístico
Wrapper sobre `RandomForestRegressor` que reemplaza el muestreo bootstrap uniforme antes de entrenar cada árbol por una distribución multinomial basada en la distancia temporal $\Delta t_i = t_{\max} - t_i$:

$$w_i = \exp(-\alpha \cdot \Delta t_i), \qquad p_i = \frac{w_i}{\sum_{j=1}^N w_j}$$

* Controla el balance sesgo–varianza mediante el decaimiento $\alpha$ y monitorea el **Tamaño Efectivo de Muestra (ESS)** de Kish ($\text{ESS} = 1 / \sum p_i^2$) para evitar que los árboles se colapsen en unas pocas muestras recientes.

### 2. Estrategia B — Ensamble Temporal por Bloques con Pesos Dinámicos
Divide el histórico en $K$ ventanas temporales disjuntas ($W_1, \dots, W_K$) y entrena un sub-bosque independiente $f_k(x)$ por bloque. La predicción final es una combinación convexa:

$$\hat{y}(x) = \sum_{k=1}^K \lambda_k f_k(x), \qquad \lambda_k \ge 0, \quad \sum_{k=1}^K \lambda_k = 1$$

Los pesos $\lambda_k$ se calculan dinámicamente frente al presente mediante:
* **Data Drift ($P(X)$):** Distancia $\widehat{\text{MMD}}^2(X_{W_k}, X_{\text{presente}})$ con kernel RBF (no requiere etiquetas futuras).
* **Concept Drift ($P(Y \mid X)$):** Error reciente $\text{MSE}(f_k, W_{\text{val\_reciente}})$ sobre una ventana reciente etiquetada.
* Si un bloque histórico sufre drift severo, su peso colapsa hacia cero vía *softmax* negativo.

### 3. Hipótesis Avanzada (Experimental) — MMD-Split en el Nodo
Evaluación de una función de costo híbrida en los **primeros niveles** del árbol (`depth <= 2` o `3`):

$$\mathcal{L}_{\text{híbrida}}(j, s) = \mathcal{L}_{\text{MSE}}(S_L, S_R) + \eta \cdot \Omega_{\text{MMD}}(S_L, S_R, R)$$

Para hacer viable su costo computacional frente a $O(N^3)$, se contempla aproximar el kernel RBF mediante **Random Fourier Features (RFF / Teorema de Bochner)**, reduciendo el cálculo de $\widehat{\text{MMD}}^2$ a diferencias incrementales de promedios.

---

## Estructura del Proyecto

```text
DriftAdaptiveForest/
├── docs/                                  # Notas matemáticas y análisis de viabilidad (ej. MMD-Split)
├── experiments/                           # Scripts de experimentos, tuning y resultados
├── notebooks/                             # Exploración visual de drift y prototipos
├── src/
│   └── drift_adaptive_forest/
│       ├── data/                          # Generadores sintéticos con drift (escenario Salud) y loaders
│       ├── evaluation/                    # Validación cruzada temporal (TimeSeriesSplit) y estabilidad
│       ├── metrics/                       # Estimadores de MMD (exacto, RFF) y métricas de drift
│       ├── models/                        # Estimadores compatibles con Scikit-Learn (Estrategias A, B y experimental)
│       └── utils/                         # Utilidades de decaimiento temporal, ESS y validación
└── tests/                                 # Pruebas unitarias y compatibilidad con Scikit-Learn
```

---

## Hoja de Ruta

1. **Paso 1 — Laboratorio Sintético (`data/`):** Generador reproducible en dominio clínico (estancia hospitalaria) con control de Data Drift, Concept Drift y relación no lineal.
2. **Paso 2 — Núcleo de MMD (`metrics/`):** Implementación de $\widehat{\text{MMD}}^2$ con kernel RBF y heurística de la mediana para el ancho de banda $\gamma$.
3. **Paso 3 — Modelos (`models/`):** Implementación de la Estrategia B y la Estrategia A heredando de `BaseEstimator` y `RegressorMixin`.
4. **Paso 4 — Evaluación Temporal (`evaluation/`):** Benchmark comparativo frente a `RandomForestRegressor` clásico usando ventanas temporales expansivas/deslizantes.
5. **Paso 5 — Prototipo MMD-Split (`docs/` + `models/`):** Validación computacional y empírica de la hipótesis híbrida con RFF.
