# Especificación de tareas pendientes — AdaptiveDriftForest

Documento para el agente (o la persona) que continúe el desarrollo. Resume lo que ya se sabe, las críticas
con su evidencia y las tareas concretas con criterio de aceptación. El detalle de cada experimento está en
[experiments/BITACORA.md](experiments/BITACORA.md); los resultados por dominio en [MEJORAS.md](MEJORAS.md),
[MATRICULA.md](MATRICULA.md), [FINANZAS.md](FINANZAS.md) y [MACRO.md](MACRO.md); el paper en
[paper/main_v2.tex](paper/main_v2.tex).

## 1. Estado actual en una página

`AdaptiveDriftForest` (`src/drift_adaptive_forest/models/adaptive_ensemble.py`) entrena bosques aleatorios en
ventanas recientes anidadas (10, 20, 35, 60 y 100 % del histórico) y en 5 bloques históricos; pondera cada
candidato por su error sobre el 15 % más reciente (softmax de `MSE/min MSE` con `beta=20`) y luego reentrena los
candidatos con peso incluyendo esa cola.

| Dónde gana | Dónde empata | Dónde pierde |
|---|---|---|
| Concept drift persistente en sintéticos (0.52–0.83 del RF) | Matrícula frente a LightGBM y ARF (< 1 %) | Volatilidad financiera: HAR-X lineal gana 51/54 bloques |
| Sintéticos frente a RF (10/11), ARF (10/11), LightGBM (7/11) | Gas Drift frente a RF | Macro mensual: RF le gana 45/72 bloques (pocos datos) |
| Matrícula frente a RF (20/21) y ETS (21/21) | | Datos reales del banco: LightGBM tiene menor error medio |
| No pierde sin drift (0.98 del RF) | | Drift recurrente: el `DriftRegressor` original (`hybrid`) es mejor |
| | | Costo: ~190× más lento y ~250× más RAM que LightGBM |

Regla práctica (hipótesis, no probada): conviene cuando el cambio es persistente, hay muchas observaciones
(decenas de miles) y la relación es no lineal. En series que vuelven a su media (volatilidad, inflación,
desempleo) conviene el histórico completo y un modelo lineal de referencia.

## 2. Críticas con evidencia

| # | Crítica | Evidencia | Archivo |
|---|---|---|---|
| C1 | Costo de memoria y cómputo muy alto | 54 332 filas × 13 variables: ajuste 38 s, 2.8 GB de RAM extra, modelo de 1.4 GB, 289 ms por fila (LightGBM: 0.2 s, 11 MB, 0.3 MB, 1 ms) | `experiments/results/costos_operativos.csv` |
| C2 | Árboles sin límite de profundidad (`min_samples_leaf=1`) | El tamaño del modelo crece con las filas; con 1 M de filas podría superar 20 GB (estimación no medida) | `adaptive_ensemble.py` |
| C3 | Se entrenan ~10 bosques y cada uno dos veces (antes y después de la cola) | Ver `fit()`; solo se descartan candidatos con peso < 1e-3 | `adaptive_ensemble.py` |
| C4 | Elección de memoria ruidosa con pocos datos (un solo corte de validación) | Macro: 300–700 meses de entrenamiento, pierde contra RF 45/72; los pesos saltan entre bloques | `MACRO.md`, figuras `06_*` |
| C5 | Una sola memoria para todo el espacio de X | En drift local, RF con el tiempo como variable (RF+t) obtiene 0.58 contra 0.72 del ADF | `experiments/results/tiempo_results.csv` |
| C6 | No reutiliza regímenes que reaparecen | `friedman_recurring`: DR `hybrid` 0.82 contra ADF 0.97 | `benchmark_*.csv` |
| C7 | LightGBM es mejor en datos reales del banco y empata en demanda | Reales: 0.968 contra 0.983; matrícula: diferencias < 1 % | `lit_benchmark.csv`, `matricula_results.csv` |
| C8 | Los árboles no extrapolan niveles nuevos | Finanzas y VIX: el HAR-X lineal supera a todos los árboles | `FINANZAS.md` |
| C9 | Sin intervalos de incertidumbre | No implementado | — |

## 3. Tareas, en orden de prioridad

Cada tarea que implique un experimento debe registrar diseño e hipótesis en `experiments/BITACORA.md` **antes** de
ejecutarlo (ver §4).

### T1. Configuración liviana del AdaptiveDriftForest (atiende C1, C2, C3)

- Probar `min_samples_leaf ∈ {5, 20}`, `n_estimators ∈ {50, 100}` y `max_depth ∈ {None, 20}` (los parámetros ya
  existen en el constructor). Añadir un umbral configurable para no reentrenar candidatos con peso menor a, por
  ejemplo, 0.05.
- Medir costo con `experiments/costos_operativos.py` (añadir las variantes a `MODELOS`).
- Medir precisión con `experiments/benchmark_drift.py --models ...` (sintéticos y reales) y con el protocolo de
  `experiments/matricula_experiment.py`.
- **Aceptación:** en el corte de 54 332 filas, RAM extra < 300 MB y modelo guardado < 100 MB, con un cociente
  de RMSE (media geométrica) que no empeore más de 1 % respecto al ADF actual en walk-forward sintético y en
  matrícula.

### T2. AdaptiveDriftForest con LightGBM como modelo base (atiende C1, C7)

- Añadir un parámetro `base_estimator` (por defecto, el RF actual) para usar `LGBMRegressor` en cada ventana y bloque.
- **Aceptación:** comparar contra LightGBM solo y contra el ADF actual en sintéticos (11 escenarios × 3 semillas),
  reales del banco y matrícula. Hipótesis a registrar: la versión con LightGBM tiene menor RMSE que LightGBM solo
  en la mayoría de los escenarios sintéticos y no pierde más de 1 % en matrícula, con un costo de ajuste menor
  a 10× el de LightGBM.

### T3. Pesos con varios cortes de validación (atiende C4)

- Estimar los errores de cada candidato en 2–3 cortes recientes (no solo la cola del 15 %) y promediar.
- **Aceptación:** menor varianza de los pesos entre reentrenamientos consecutivos y sin pérdida de precisión en
  sintéticos; probar también en macro mensual (`experiments/macro_experiment.py`), donde hoy pierde.

### T4. Tiempo como variable dentro del ensamble (atiende C5)

- Combinar RF+t (el tiempo como covariable, en todos los niveles del árbol) con la ponderación del ADF.
  No limitar el uso del tiempo a los primeros niveles: la versión de 3 niveles (Part3+RF) casi nunca dividió por
  el tiempo (0.4 de 6 divisiones) y no mejoró.
- **Aceptación:** mejora sobre el ADF en `local_drift` sin empeorar en matrícula ni en los controles sin drift
  (`healthcare_none`, `covariate_shift`) más de 2 %.

### T5. Memoria de regímenes (atiende C6)

- Reactivar bloques antiguos cuyo error reciente y cuya MMD contra la cola reciente sean bajos.
- **Aceptación:** igualar al `DriftRegressor` `hybrid` en `friedman_recurring` sin empeorar el resto.

### T6. Intervalos de predicción (atiende C9)

- Conformal adaptativo (Gibbs y Candès, 2021; verificar la cita) con los residuos de la cola reciente.
- **Aceptación:** cobertura empírica a ±3 puntos del nivel nominal (por ejemplo 90 %) en walk-forward sintético y
  en matrícula, reportando ancho medio.

### T7. MMD-split (divisiones de árbol sensibles al drift) — solo si T4 funciona

- Por la regla registrada en la bitácora §15, no se implementa hasta que la versión simplificada muestre beneficio.

## 4. Reglas del proyecto (obligatorias)

1. **Hipótesis antes de correr.** Registrar diseño, modelos, métricas e hipótesis en `experiments/BITACORA.md`
   antes de ejecutar; no cambiarlos después. Los resultados que no se anticiparon se reportan como exploratorios.
2. **Cifras rastreables.** Ningún número en documentos o en el paper se escribe a mano: sale de los CSV de
   `experiments/results/` (el paper usa `paper/generar_tablas.py` y `paper/generar_tablas_v2.py`).
3. **Comparar siempre contra:** RF clásico, RF de ventana reciente, LightGBM y el modelo de referencia del dominio
   (HAR en volatilidad, AR o índice de difusión en macro, ingenuo y ETS en demanda).
4. **Protocolos temporales estrictos:** walk-forward y 80/20; con objetivos a horizonte h, embargo de h periodos.
5. **No sobrescribir archivos del usuario.** Si hay que cambiar un documento suyo (por ejemplo `paper/main.tex`),
   crear una versión nueva numerada.
6. **Sin citas inventadas.** Verificar cada referencia nueva (DOI en Crossref); si no se puede, marcarla
   `% VERIFICAR`. Pendientes: `ke2017` y `harries1999` en `paper/referencias_v2.bib`.
7. **Gráficos con la paleta CCDE:** #0A2540, #1E3D59, #17A2B8, #E2F0F9; #F5A623 solo para quiebres, umbrales y
   alertas, nunca como color de una serie. Implementación de referencia: `experiments/graficar_analisis.py`.
8. **Commits:** sin líneas `Co-Authored-By` ni coautores externos; autor único, el usuario.
9. **Credenciales:** nunca escribir claves en el repositorio. La clave de FRED se lee de `FRED_API_KEY`.

## 5. Entorno y comandos

```powershell
# Entorno (Python 3.14, Windows)
.\venv_drift_forest\Scripts\python.exe -m pip install -r requirements.txt lightgbm statsmodels river openpyxl

# Datos (data/real/ no se versiona; se descargan con estos scripts)
.\venv_drift_forest\Scripts\python.exe -I experiments\finance_data.py
$env:FRED_API_KEY = "<clave>"; .\venv_drift_forest\Scripts\python.exe -I experiments\macro_data.py
# IPEDS: descargar EF2002A..EF2023A y HD2023 de https://nces.ed.gov/ipeds/datacenter/data/ a data/real/matricula/ipeds/

# Experimentos principales
.\venv_drift_forest\Scripts\python.exe -I experiments\benchmark_drift.py --suite all --seeds 0 1 2
.\venv_drift_forest\Scripts\python.exe -I experiments\matricula_experiment.py --seeds 0 1
.\venv_drift_forest\Scripts\python.exe -I experiments\comparacion_literatura.py
.\venv_drift_forest\Scripts\python.exe -I experiments\costos_operativos.py
.\venv_drift_forest\Scripts\python.exe -I paper\generar_tablas_v2.py
```

Notas prácticas:
- Las corridas largas (30–60 min) se pausan si Windows entra en suspensión; desactivarla mientras corren.
- Los modelos de River (ARF, HAT) aprenden fila a fila; `comparacion_literatura.py` los evalúa en una sola pasada.
- El paper no compila en esta máquina (no hay LaTeX); se compila en Overleaf con pdfLaTeX + biber.
- Pendientes del usuario: correo noreply real en `git config user.email` (los commits recientes tienen un texto de
  ejemplo) y revocar la clave de FRED que se compartió en una conversación.
