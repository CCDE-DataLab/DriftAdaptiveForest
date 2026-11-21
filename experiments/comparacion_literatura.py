"""Comparación con métodos de la literatura: ARF y HAT (River), LightGBM y RF+t. Diseño en BITACORA §16.

Los modelos de River aprenden fila a fila en orden temporal; en cada origen predicen el bloque de prueba
sin aprender de él. Como el flujo es determinista, una sola pasada equivale a reentrenar en cada origen.

Salidas (no coinciden con el patrón benchmark_*.csv para no alterar las tablas del paper original):
  experiments/results/lit_benchmark.csv, experiments/results/lit_matricula.csv

Uso: venv_drift_forest/Scripts/python.exe -I experiments/comparacion_literatura.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from river import forest, preprocessing, tree  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import root_mean_squared_error  # noqa: E402

from experiments.benchmark_drift import splits  # noqa: E402
from experiments.real_datasets import DATASETS  # noqa: E402
from experiments.synthetic_scenarios import SCENARIOS  # noqa: E402

warnings.filterwarnings("ignore")
RES = ROOT / "experiments" / "results"


def arf(seed):
    return preprocessing.StandardScaler() | forest.ARFRegressor(n_models=10, seed=seed)


def hat(seed):
    return preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeRegressor(seed=seed)


RIVER = {"ARF": arf, "HAT": hat}


def river_una_pasada(make, X, y, seed, tareas):
    """tareas: lista de (clave, a, b). Aprende [0, a) en orden y predice [a, b) sin aprender."""
    cols = [f"x{j}" for j in range(X.shape[1])]
    filas = [dict(zip(cols, r)) for r in X.tolist()]
    m = make(seed)
    aprendidas, out = 0, {}
    for clave, a, b in sorted(tareas, key=lambda z: z[1]):
        for i in range(aprendidas, a):
            m.learn_one(filas[i], float(y[i]))
        aprendidas = max(aprendidas, a)
        out[clave] = np.array([m.predict_one(filas[i]) for i in range(a, b)], dtype=float)
    return out


def lgbm(Xtr, ytr, Xte, seed):
    return LGBMRegressor(random_state=seed, verbose=-1).fit(Xtr, ytr).predict(Xte)


def rf_t(Xtr, ytr, ttr, Xte, tte, seed):
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    return rf.fit(np.column_stack([Xtr, ttr]), ytr).predict(np.column_stack([Xte, tte]))


def banco(rows):
    jobs = [(n, "sintetico", lambda g=g, s=s: g(s), s) for n, (g, _) in SCENARIOS.items() for s in (0, 1, 2)]
    jobs += [(n, "real", f, 0) for n, (f, _) in DATASETS.items()]
    for name, kind, make, seed in jobs:
        X, y = make()
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(X)
        t = np.arange(n) / n
        tareas = [((p, s), a, b) for p in ("static", "walk") for s, a, b in splits(n, p)]
        for mod, mk in RIVER.items():
            t0 = time.perf_counter()
            preds = river_una_pasada(mk, X, y, seed, tareas)
            secs = time.perf_counter() - t0
            for (p, s), a, b in tareas:
                rows.append({"dataset": name, "kind": kind, "seed": seed, "protocol": p, "step": s, "model": mod,
                             "rmse": root_mean_squared_error(y[a:b], preds[(p, s)]), "secs": secs})
        for (p, s), a, b in tareas:
            for mod in ("LightGBM", "RF+t"):
                t0 = time.perf_counter()
                pr = (lgbm(X[:a], y[:a], X[a:b], seed) if mod == "LightGBM"
                      else rf_t(X[:a], y[:a], t[:a], X[a:b], t[a:b], seed))
                rows.append({"dataset": name, "kind": kind, "seed": seed, "protocol": p, "step": s, "model": mod,
                             "rmse": root_mean_squared_error(y[a:b], pr), "secs": time.perf_counter() - t0})
        pd.DataFrame(rows).to_csv(RES / "lit_benchmark.csv", index=False)
        print(f"{name} seed={seed} listo", flush=True)


def matricula(rows):
    from experiments.matricula_data import panel_balanceado
    from experiments.matricula_experiment import FEATS, HORIZONTES, ORIGEN_INICIAL, construir_filas
    m, h = panel_balanceado()
    for horizonte in HORIZONTES:
        f = construir_filas(m, h, horizonte)
        origenes = list(range(ORIGEN_INICIAL, max(m.columns) - horizonte + 1))
        preds = {mod: {o: [] for o in origenes} for mod in RIVER}
        for mod, mk in RIVER.items():
            for seed in (0, 1):
                modelo = mk(seed)
                aprendido_hasta = None  # último año t ya aprendido
                for o in origenes:
                    nuevos = f[(f.t + horizonte <= o) & ((f.t > aprendido_hasta) if aprendido_hasta else True)]
                    for xr, yr in zip(nuevos[FEATS].to_dict("records"), nuevos["y"].to_numpy()):
                        modelo.learn_one(xr, float(yr))
                    aprendido_hasta = o - horizonte
                    te = f[f.t == o]
                    preds[mod][o].append(np.array([modelo.predict_one(xr) for xr in te[FEATS].to_dict("records")]))
                print(f"matrícula h={horizonte} {mod} seed={seed} listo", flush=True)
        for o in origenes:
            te = f[f.t == o]
            yte = te["y"].to_numpy()
            for mod in RIVER:
                err = yte - np.mean(preds[mod][o], axis=0)
                rows.append({"h": horizonte, "origen": o, "model": mod,
                             "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))})
        pd.DataFrame(rows).to_csv(RES / "lit_matricula.csv", index=False)


if __name__ == "__main__":
    banco([])
    matricula([])
    print("\nResultados: lit_benchmark.csv y lit_matricula.csv")
