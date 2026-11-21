"""Versión barata del MMD-split: el tiempo como variable en los primeros niveles. Diseño en BITACORA §15.

Modelos:
  RF_clasico  RandomForest sobre X (referencia, se recalcula para verificar que coincide con corridas previas)
  RF+t        RandomForest sobre [X, t]
  Part3+RF    árbol de 3 niveles sobre [X, t] (hoja mínima 5 %) -> un RF clásico por región

Uso: venv_drift_forest/Scripts/python.exe -I experiments/tiempo_experiment.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import root_mean_squared_error  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from experiments.benchmark_drift import splits  # noqa: E402
from experiments.real_datasets import DATASETS  # noqa: E402
from experiments.synthetic_scenarios import SCENARIOS  # noqa: E402

warnings.filterwarnings("ignore")
RES = ROOT / "experiments" / "results"
SINTETICOS = ["local_drift", "healthcare_both", "friedman_abrupt", "incremental", "healthcare_none", "covariate_shift"]
REALES = ["bike_sharing", "electricity"]


def _rf(seed):
    return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)


class ParticionTiempoRF:
    """Árbol poco profundo sobre [X, t] que define regiones; un RF clásico sobre X en cada región."""

    def __init__(self, depth=3, min_leaf_frac=0.05, random_state=0):
        self.depth, self.min_leaf_frac, self.random_state = depth, min_leaf_frac, random_state

    def fit(self, X, y, t):
        Z = np.column_stack([X, t])
        self.arbol_ = DecisionTreeRegressor(max_depth=self.depth, random_state=self.random_state,
                                            min_samples_leaf=max(30, int(self.min_leaf_frac * len(X)))).fit(Z, y)
        hojas = self.arbol_.apply(Z)
        self.rfs_ = {h: _rf(self.random_state).fit(X[hojas == h], y[hojas == h]) for h in np.unique(hojas)}
        feat = self.arbol_.tree_.feature
        self.cortes_t_ = int((feat == X.shape[1]).sum())  # cuántos de los cortes usan el tiempo
        self.cortes_ = int((feat >= 0).sum())
        return self

    def predict(self, X, t):
        hojas = self.arbol_.apply(np.column_stack([X, t]))
        out = np.empty(len(X))
        for h in np.unique(hojas):
            out[hojas == h] = self.rfs_[h].predict(X[hojas == h])
        return out


def ajustar(nombre, Xtr, ytr, ttr, Xte, tte, seed):
    if nombre == "RF_clasico":
        return _rf(seed).fit(Xtr, ytr).predict(Xte), None
    if nombre == "RF+t":
        return _rf(seed).fit(np.column_stack([Xtr, ttr]), ytr).predict(np.column_stack([Xte, tte])), None
    m = ParticionTiempoRF(random_state=seed).fit(Xtr, ytr, ttr)
    return m.predict(Xte, tte), {"cortes": m.cortes_, "cortes_en_t": m.cortes_t_}


MODELOS = ["RF_clasico", "RF+t", "Part3+RF"]


def benchmark_ordenado(rows):
    jobs = [(n, "sintetico", lambda g=SCENARIOS[n][0], s=s: g(s), s) for n in SINTETICOS for s in (0, 1, 2)]
    jobs += [(n, "real", DATASETS[n][0], 0) for n in REALES]
    for name, kind, make, seed in jobs:
        X, y = make()
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(X)
        t = np.arange(n) / n
        for protocol in ("static", "walk"):
            for step, a, b in splits(n, protocol):
                for mod in MODELOS:
                    p, info = ajustar(mod, X[:a], y[:a], t[:a], X[a:b], t[a:b], seed)
                    rows.append({"dataset": name, "kind": kind, "seed": seed, "protocol": protocol, "step": step,
                                 "model": mod, "rmse": root_mean_squared_error(y[a:b], p),
                                 "info": json.dumps(info) if info else ""})
        print(f"{name} seed={seed} listo", flush=True)
        pd.DataFrame(rows).to_csv(RES / "tiempo_results.csv", index=False)


def matricula(rows_m):
    from experiments.matricula_data import panel_balanceado
    from experiments.matricula_experiment import FEATS, HORIZONTES, ORIGEN_INICIAL, construir_filas
    m, h = panel_balanceado()
    for horizonte in HORIZONTES:
        f = construir_filas(m, h, horizonte)
        for origen in range(ORIGEN_INICIAL, max(m.columns) - horizonte + 1):
            tr, te = f[f.t + horizonte <= origen], f[f.t == origen]
            Xtr, ytr, Xte, yte = tr[FEATS].to_numpy(), tr["y"].to_numpy(), te[FEATS].to_numpy(), te["y"].to_numpy()
            ttr, tte = tr["t"].to_numpy(float), te["t"].to_numpy(float)
            for mod in MODELOS:
                ps, infos = [], []
                for seed in (0, 1):
                    p, info = ajustar(mod, Xtr, ytr, ttr, Xte, tte, seed)
                    ps.append(p)
                    infos.append(info)
                err = yte - np.mean(ps, axis=0)
                rows_m.append({"h": horizonte, "origen": origen, "model": mod,
                               "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
                               "info": json.dumps(infos[0]) if infos[0] else ""})
            print(f"matrícula h={horizonte} origen={origen} listo", flush=True)
            pd.DataFrame(rows_m).to_csv(RES / "tiempo_matricula_results.csv", index=False)


if __name__ == "__main__":
    benchmark_ordenado([])
    matricula([])
    print("\nResultados: tiempo_results.csv y tiempo_matricula_results.csv")
