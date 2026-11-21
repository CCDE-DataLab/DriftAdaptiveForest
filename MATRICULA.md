# Demanda de matrícula universitaria (IPEDS, EE. UU.)

Diseño e hipótesis registrados antes de correr en [experiments/BITACORA.md](experiments/BITACORA.md) §13.
Código: [matricula_data.py](experiments/matricula_data.py), [matricula_experiment.py](experiments/matricula_experiment.py),
[analizar_matricula.py](experiments/analizar_matricula.py). Figuras `16` a `18` en [experiments/figuras/](experiments/figuras/).

## Resumen

- **Primer dataset real donde el modelo adaptativo gana con claridad a su base:** AdaptiveDriftForest supera al
  RF clásico en **20 de 21** cortes (RMSE) y **21 de 21** (MAE), y a ETS en 21 de 21.
- **Frente a LightGBM es un empate.** Gana en promedio a 2 años, pero pierde en la mayoría de los cortes individuales.
- **La matrícula es casi impredecible más allá del "se queda igual":** los mejores modelos mejoran al ingenuo
  solo 1–3 %. Los modelos que extrapolan tendencias (ETS, tendencia de 3 años) son **15–25 % peores** que el ingenuo.

## Diseño

- **Datos:** matrícula de otoño 2002–2023 de 3 196 instituciones (1 620 públicas, 1 253 privadas sin fines de lucro,
  323 con fines de lucro), del IPEDS del Departamento de Educación de EE. UU.
- **Objetivo:** crecimiento log de la matrícula a 1 y 2 años.
- **Variables:** crecimiento de los 3 últimos años, tamaño, tipo y nivel de institución, crecimiento promedio
  nacional y del mismo tipo.
- **Protocolo:** origen rodante anual 2012–2022 (21 cortes en total), sin tuning, 2 semillas.

## Resultados (error relativo al pronóstico ingenuo "la matrícula no cambia"; menos de 1 = mejor)

| Modelo | RMSE h=1 | RMSE h=2 | MAE h=1 | MAE h=2 |
|---|---:|---:|---:|---:|
| Tendencia (promedio 3 años) | 1.136 | 1.245 | 1.117 | 1.200 |
| ETS (Holt amortiguado) | 1.170 | 1.158 | 1.179 | 1.149 |
| OLS global | 1.003 | 1.031 | 1.016 | 1.060 |
| LightGBM | **0.988** | 1.001 | **0.980** | 0.997 |
| RF clásico | 1.002 | 1.008 | 1.004 | 1.004 |
| RF ventana reciente | 1.013 | 1.001 | 1.023 | 0.987 |
| DriftRegressor original | 0.995 | 0.991 | 0.991 | 0.979 |
| **AdaptiveDriftForest** | 0.991 | **0.983** | 0.988 | **0.969** |

| Hipótesis registrada | Resultado | Cortes | p (signo) |
|---|---|---:|---:|
| H5: AdaptiveDriftForest mejor que RF clásico | **Se confirma** | 20/21 (MAE 21/21) | < 0.001 |
| H6: AdaptiveDriftForest mejor que LightGBM | **No se confirma** | 8/21 (MAE 7/21) | 0.38 |
| H7: AdaptiveDriftForest mejor que ETS | **Se confirma** | 21/21 | < 0.001 |
| H8: ventaja mayor en privadas con fines de lucro | **No se confirma** | 9/21 | 0.66 |

## Interpretación

1. **Extrapolar tendencias es lo peor.** La matrícula pública creció hasta 2011 y luego cayó. ETS y la tendencia
   siguieron proyectando crecimiento justo después del cambio (en 2013, ETS es 50 % peor que el ingenuo). Es concept
   drift en su forma más clara: la tendencia pasada dejó de valer.
2. **Elegir la memoria ayuda frente a un RF fijo.** Con 22 000–57 000 filas, el modelo adaptativo tiene datos
   suficientes para elegir bien, y le gana al RF clásico de forma consistente. Es el caso de "muchos datos + cambio
   de tendencia" que anticipaba la regla de la sección de finanzas.
3. **El COVID-19 favoreció a los bosques:** al pronosticar 2020, RF y AdaptiveDriftForest mejoraron al ingenuo en
   ~9 % mientras LightGBM fue peor que el ingenuo. En el rebote de 2021 ganó LightGBM.
4. **LightGBM es el rival a batir en demanda.** Su ventaja viene del boosting (corrige errores en secuencia), no del
   manejo del drift. La idea natural es combinar ambos: elegir la memoria al estilo AdaptiveDriftForest pero con
   LightGBM como modelo base.

## Limitaciones

- Sesgo de supervivencia: solo instituciones que existieron los 22 años; las que cerraron (parte real del drift,
  sobre todo en el sector con fines de lucro) no están.
- El tipo de institución se toma del directorio 2023; algunas cambiaron de sector.
- Hay crecimientos extremos (fusiones o errores de reporte); por eso se registró el MAE como métrica secundaria.
  Las conclusiones no cambian con MAE.
- Datos anuales: 11 cortes por horizonte. Las mejoras frente al ingenuo son pequeñas en magnitud.
