# DriftAdaptiveForest

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Compatible-F7931E?logo=scikit-learn&logoColor=white)
![Optuna](https://img.shields.io/badge/Tuning-Optuna%20%2B%20TimeSeriesSplit-blueviolet)
![Status](https://img.shields.io/badge/Status-Active%20Research-success)

Framework modular en Python (compatible con **Scikit-Learn** y **Optuna**) para mitigar **Data Drift** ($P(X)$ cambia en el tiempo) y **Concept Drift** ($P(Y \mid X)$ cambia en el tiempo) en problemas de **regresión temporal** usando algoritmos basados en árboles.

---

## Por qué falla un Random Forest clásico ante Drift

Entrenar un `RandomForestRegressor` clásico sobre datos temporales (incluso usando `sample_weight` exponencial) presenta dos limitaciones estructurales:
1. El **bootstrap uniforme** sigue mezclando muestras históricas obsoletas dentro de cada árbol, diluyendo el peso de las observaciones recientes.
2. Los árboles **no extrapolan** fuera del rango de entrenamiento y promedian regímenes pasados incompatibles con el presente cuando cambia $P(Y \mid X)$.

---

## Arquitectura de la Estrategia Implementada (`DriftRegressor`)

En lugar de mezclar todas las épocas en un solo bosque, `DriftRegressor` entrena un sub-bosque experto por cada bloque temporal y asigna **pesos dinámicos** en tiempo de inferencia según qué tan válido sigue siendo cada bloque frente al presente:

```mermaid
flowchart LR
    A["X_train, y_train<br/>(Ordenado por fecha)"] --> B{"Modo de peso"}
    B -- "mmd" --> C["K bloques temporales<br/>W_1 ... W_K"]
    B -- "recent_error / hybrid" --> D["Reserva cola reciente W_val<br/>(15% final de train)"]
    D --> C
    C --> E["K Random Forests<br/>f_1 ... f_K"]
    E --> F["Calculo de penalizacion p_k<br/>MMD(W_k, X_test) y/o MSE(f_k, W_val)"]
    F --> G["Softmax Negativo<br/>lambda_k = exp(-beta * p_k)"]
    G --> H["Prediccion ponderada<br/>y_hat = sum(lambda_k * f_k(X_test))"]
```

---

## Formulación Matemática

### 1. Datos Sintéticos con Drift Dual (`generate_data.py`)
Simula ingresos hospitalarios (`dias_estancia` como target no lineal sobre `edad`, `comorbilidad`, `severidad_ingreso` y `ocupacion_hospitalaria`) sobre el reloj normalizado $t_{\text{norm}} \in [0, 1]$:
* **Data Drift ($P(X)$):** Las medias y parámetros de forma de las covariables evolucionan con tendencia y ondas estacionales.
* **Concept Drift ($P(Y \mid X)$):** En `drift_type="both"`, una transición sigmoide centrada en $t_{\text{norm}} = 0.6$ cambia el protocolo clínico:

$$\alpha_t = \frac{1}{1 + \exp\big(-15(t_{\text{norm}} - 0.6)\big)}$$

$$y_t = (1 - \alpha_t)\,f_{\text{vieja}}(X_t) + \alpha_t\,f_{\text{nueva}}(X_t) + \epsilon_t$$

### 2. Pesos Dinámicos por Bloque (`block_ensemble.py`)
Dado un conjunto histórico dividido en $K$ ventanas contiguas $W_1, \dots, W_K$ con sub-bosques $f_1, \dots, f_K$:

* **Modo `"mmd"` (detecta Data Drift en $X$):**
  $$p_k = \max\Big(0,\; \widehat{\text{MMD}}^2\big(X_{W_k}^{\text{scaled}}, X_{\text{test}}^{\text{scaled}}\big)\Big)$$
* **Modo `"recent_error"` (detecta Concept Drift en $Y \mid X$):**
  $$e_k = \text{MSE}(f_k, W_{\text{val}}), \qquad p_k = \frac{e_k}{\bar{e} + \epsilon}$$
* **Modo `"hybrid"` (combina ambos con $\alpha \in [0, 1]$):**
  $$p_k = \alpha \frac{\text{MMD}^2_k}{\overline{\text{MMD}^2} + \epsilon} + (1 - \alpha) \frac{e_k}{\bar{e} + \epsilon}$$

En los tres modos, las penalizaciones se transforman en pesos convexos mediante **softmax negativo** y se combinan en la predicción final:

$$\lambda_k = \frac{\exp(-\beta \cdot p_k)}{\sum_{j=1}^{K} \exp(-\beta \cdot p_j)}, \qquad \hat{y}(x) = \sum_{k=1}^{K} \lambda_k \, f_k(x)$$

---

## Resultados Experimentales Observados

Evaluación temporal estricta (`4,000` filas iniciales para entrenamiento/CV, `1,000` filas futuras para test):

### Escenario A: Solo Data Drift moderado (`healthcare_drift.parquet`)
| Modelo | Configuración | RMSE (Test) | Hallazgo |
| :--- | :--- | :---: | :--- |
| **RF Clásico** | 100 árboles | **1.3073** | Al ser fija $f(X)$, aprovecha las 4,000 muestras juntas |
| **DriftRegressor (`mmd`)** | `n_blocks=5, beta=10` | 1.4761 | `beta=10` dejó 46% de peso a bloques viejos |
| **DriftRegressor (`mmd`)** | `n_blocks=5, beta=50` | 1.3874 | Colapsó bloques 1–3 al 10% total; mejora clara vs `beta=10` |

### Escenario B: Data Drift + Concept Drift (`healthcare_both_drift.parquet`)

**Comparativa base (`n_blocks=5, beta=15`):**
| Modelo | RMSE (Test) | Pesos asignados por bloque `[B1, B2, B3, B4, B5]` |
| :--- | :---: | :--- |
| **RF Clásico** | 6.4003 | *(Peso uniforme sobre toda la historia)* |
| **DriftRegressor (`mmd`)** | 5.9404 | `[0.1024, 0.1734, 0.1180, 0.2923, 0.3140]` |
| **DriftRegressor (`hybrid`)** | 5.8175 | `[0.0000, 0.0008, 0.0000, 0.2302, 0.7690]` |
| **DriftRegressor (`recent_error`)** | **5.0854** | `[0.0000, 0.0000, 0.0000, 0.0000, 1.0000]` |

**Comparativa optimizada con Optuna + `TimeSeriesSplit` (4 folds):**
| Modelo | RMSE | MAE | $R^2$ | Mejora en RMSE |
| :--- | :---: | :---: | :---: | :---: |
| **RF Clásico (Optuna)** | 6.3865 | 4.3541 | -0.9377 | — |
| **DriftRegressor (Optuna)** | **4.6901** | **3.2892** | **-0.0450** | **-26.6%** (-1.70 días) |

> **Conclusión clave:** Cuando solo cambia $P(X)$ levemente, el RF clásico aprovecha todo el histórico; pero cuando ocurre **Data + Concept Drift**, `DriftRegressor` supera ampliamente al RF clásico al aislar regímenes y colapsar el peso de las épocas obsoletas.

---

## Instalación y Uso Rápido

```powershell
# 1. Crear y activar el entorno virtual del proyecto
python -m venv venv_drift_forest
.\venv_drift_forest\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Generar datos sintéticos (Data + Concept Drift)
python -m src.drift_adaptive_forest.data.generate_data

# 4. Probar DriftRegressor vs RF Clásico
python -m src.drift_adaptive_forest.models.block_ensemble

# 5. Ejecutar optimización temporal con Optuna
python -m experiments.tune_and_compare
```

---

## Estructura y Hoja de Ruta

```text
DriftAdaptiveForest/
├── data/raw/                              # Datasets sintéticos (.parquet)
├── experiments/tune_and_compare.py        # Benchmark y tuning con Optuna + TimeSeriesSplit
├── src/drift_adaptive_forest/
│   ├── data/generate_data.py              # Simulador clínico con Data y Concept Drift
│   ├── metrics/mmd.py                     # Estimador empírico de MMD^2 (RBF + heurística mediana)
│   └── models/block_ensemble.py           # Estrategia B: DriftRegressor compatible con sklearn
├── requirements.txt
└── README.md
```

* [x] **Paso 1:** Generador sintético con Data Drift y Concept Drift (`generate_data.py`).
* [x] **Paso 2:** Estimador $\widehat{\text{MMD}}^2$ con kernel RBF y heurística de la mediana (`mmd.py`).
* [x] **Paso 3:** Estrategia B — `DriftRegressor` con modos `"mmd"`, `"recent_error"` y `"hybrid"` (`block_ensemble.py`).
* [x] **Paso 4:** Optimización y validación temporal con Optuna + `TimeSeriesSplit` (`tune_and_compare.py`).
* [ ] **Paso 5 (Próximo):** Estrategia A — Bootstrap Temporal Probabilístico ($w_i = e^{-\alpha \Delta t_i}$ con control de ESS).
* [ ] **Paso 6 (Experimental):** Prototipo de **MMD-Split** en los primeros niveles del árbol usando *Random Fourier Features* (RFF):
  $$\mathcal{L}_{\text{hibrida}}(j, s) = \mathcal{L}_{\text{MSE}}(S_L, S_R) + \eta \cdot \Omega_{\text{MMD}}(S_L, S_R, R)$$

---

## Licencia

Copyright (c) 2026 Mauricio Aguirre Stornaiuolo y Piero Alexander Simeon Bustillos. Todos los derechos reservados.
El código es visible, pero **no es de código abierto**. Puedes leerlo y citarlo; para usarlo, copiarlo o integrarlo
en tus proyectos, **contáctanos primero** abriendo un [issue](https://github.com/CCDE-DataLab/DriftAdaptiveForest/issues)
con quién eres y para qué lo usarás. Los miembros del CCDE pueden usarlo con fines de aprendizaje sin solicitud previa.
Condiciones completas en [LICENSE](LICENSE).
