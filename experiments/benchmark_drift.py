"""Benchmark DriftRegressor vs Random Forest en escenarios sintéticos y reales.

Protocolos (ambos estrictamente temporales, sin mezclar futuro en el entrenamiento):
  static : entrena con el primer 80 % y evalúa en el 20 % final (protocolo del README).
  walk   : walk-forward con ventana expansiva. Orígenes en 40 %, 50 %, ..., 90 % del dataset;
           en cada paso se entrena con todo el pasado y se evalúa en el 10 % siguiente.

Modelos (mismos hiperparámetros de árbol: 100 árboles, profundidad libre):
  RF_clasico       RandomForestRegressor con todo el histórico (la referencia)
  RF_ventana       RF solo con el 20 % más reciente del histórico (línea base de recencia)
  RF_pesos_exp     RF con sample_weight exponencial (vida media = 20 % del histórico)
  DR_mmd / DR_recent_error / DR_hybrid   DriftRegressor(n_blocks=5, beta=15) como en el README
  DR_recent_refit* variante experimental: recent_error + reentrenar el último bloque con la cola
  ADF_error / ADF_hybrid  AdaptiveDriftForest (versión mejorada, ver models/adaptive_ensemble.py)

Uso (desde la raíz del repo):
  venv_drift_forest/Scripts/python.exe -I experiments/benchmark_drift.py --suite all --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: E402
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments.real_datasets import DATASETS  # noqa: E402
from experiments.synthetic_scenarios import SCENARIOS  # noqa: E402
from src.drift_adaptive_forest.metrics.mmd import compute_mmd_square  # noqa: E402
from src.drift_adaptive_forest.models.adaptive_ensemble import AdaptiveDriftForest  # noqa: E402
from src.drift_adaptive_forest.models.block_ensemble import DriftRegressor  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
RESULTS_DIR = ROOT / "experiments" / "results"
N_TREES = 100
RECENT_FRAC = 0.20


# ---------------------------------------------------------------- modelos
def _rf(seed):
    return RandomForestRegressor(n_estimators=N_TREES, random_state=seed, n_jobs=-1)


def rf_clasico(Xtr, ytr, Xte, seed):
    return _rf(seed).fit(Xtr, ytr).predict(Xte), None


def rf_ventana(Xtr, ytr, Xte, seed):
    w = max(200, int(RECENT_FRAC * len(Xtr)))
    return _rf(seed).fit(Xtr[-w:], ytr[-w:]).predict(Xte), None


def rf_pesos_exp(Xtr, ytr, Xte, seed):
    n = len(Xtr)
    age = (n - 1 - np.arange(n)) / n  # 0 = más reciente, ~1 = más antiguo
    sw = 0.5 ** (age / RECENT_FRAC)
    return _rf(seed).fit(Xtr, ytr, sample_weight=sw).predict(Xte), None


class DriftRegressorRefit(DriftRegressor):
    """Variante EXPERIMENTAL (no es parte de la librería).

    En modos recent_error/hybrid el DriftRegressor original reserva el 15 % más reciente
    solo para calcular pesos y nunca entrena con él. Aquí, tras calcular los pesos, se
    reentrena el último bloque incluyendo esa cola para no desperdiciar los datos más nuevos.
    """

    def fit(self, X, y, timestamps=None):
        super().fit(X, y, timestamps)
        if self.weighting_mode in ("recent_error", "hybrid"):
            X = np.asarray(X)
            y = np.asarray(y)
            if timestamps is not None:
                orden = np.argsort(np.asarray(timestamps))
                X, y = X[orden], y[orden]
            n_tr = len(X) - max(20, int(len(X) * self.recent_val_frac))
            start = np.array_split(np.arange(n_tr), self.n_blocks)[-1][0]
            rf = RandomForestRegressor(n_estimators=self.n_estimators_per_block, max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf,
                                       random_state=self.random_state, n_jobs=-1)
            self.models_[-1] = rf.fit(X[start:], y[start:])
            self.block_features_[-1] = self.scaler_.transform(X[start:])
        return self


def drift_regressor(mode, cls=DriftRegressor):
    def _fit_predict(Xtr, ytr, Xte, seed):
        dr = cls(n_blocks=5, n_estimators_per_block=N_TREES, beta=15.0,
                 weighting_mode=mode, random_state=seed)
        pred = dr.fit(Xtr, ytr).predict(Xte)
        return pred, [round(float(w), 4) for w in dr.last_weights_]
    return _fit_predict


def adaptive_forest(weighting):
    def _fit_predict(Xtr, ytr, Xte, seed):
        m = AdaptiveDriftForest(weighting=weighting, n_estimators=N_TREES, random_state=seed)
        pred = m.fit(Xtr, ytr).predict(Xte)
        return pred, m.weights_summary()
    return _fit_predict


MODELS = {
    "RF_clasico": rf_clasico,
    "RF_ventana": rf_ventana,
    "RF_pesos_exp": rf_pesos_exp,
    "DR_mmd": drift_regressor("mmd"),
    "DR_recent_error": drift_regressor("recent_error"),
    "DR_hybrid": drift_regressor("hybrid"),
    "DR_recent_refit*": drift_regressor("recent_error", DriftRegressorRefit),
    "ADF_error": adaptive_forest("error"),
    "ADF_hybrid": adaptive_forest("hybrid"),
}


# ---------------------------------------------------------------- protocolos
def splits(n: int, protocol: str):
    if protocol == "static":
        yield 0, int(0.8 * n), n
    else:
        for step, o in enumerate(np.arange(0.4, 0.95, 0.1)):
            a, b = int(round(o * n)), min(n, int(round((o + 0.1) * n)))
            yield step, a, b


def run_dataset(name, kind, drift_label, X, y, seed, rows_out, models=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    for protocol in ("static", "walk"):
        for step, a, b in splits(len(X), protocol):
            Xtr, ytr, Xte, yte = X[:a], y[:a], X[a:b], y[a:b]
            for mname, fn in MODELS.items():
                if models and mname not in models:
                    continue
                t0 = time.perf_counter()
                pred, weights = fn(Xtr, ytr, Xte, seed)
                rows_out.append({
                    "dataset": name, "kind": kind, "drift": drift_label, "seed": seed,
                    "protocol": protocol, "step": step, "model": mname,
                    "n_train": a, "n_test": b - a,
                    "rmse": root_mean_squared_error(yte, pred),
                    "mae": mean_absolute_error(yte, pred),
                    "r2": r2_score(yte, pred),
                    "secs": time.perf_counter() - t0,
                    "weights": json.dumps(weights) if weights is not None else "",
                })


# ---------------------------------------------------------------- diagnóstico de drift
def drift_diagnostics(X, y, seed=0):
    """Mide cuánto drift hay realmente, sin depender de los modelos comparados.

    data_drift_auc  : AUC de un clasificador que distingue el 30 % inicial del 30 % final de X
                      (0.5 = sin data drift, 1.0 = distribuciones separables).
    mmd2_ini_fin    : MMD² (RBF, mediana) entre esos dos tramos estandarizados.
    concept_gain    : 1 - RMSE(RF con X + t) / RMSE(RF con X), con KFold aleatorio. Si conocer
                      el instante t mejora la predicción dado X, P(Y|X) cambia con el tiempo.
                      (En series reales también captura autocorrelación de corto plazo.)
    recency_ratio   : RMSE en el 20 % final de un RF entrenado en [0,40 %) / RF entrenado en [40,80 %).
                      >1 indica que los datos viejos sirven menos que los recientes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    a = int(0.3 * n)
    Xd = np.vstack([X[:a], X[-a:]])
    yd = np.r_[np.zeros(a), np.ones(a)]
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    auc = cross_val_score(clf, Xd, yd, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          scoring="roc_auc").mean()

    sc = StandardScaler().fit(X[:a])
    rng = np.random.default_rng(seed)
    i0 = rng.choice(a, size=min(a, 1500), replace=False)
    i1 = rng.choice(a, size=min(a, 1500), replace=False)
    mmd2 = compute_mmd_square(sc.transform(X[:a][i0]), sc.transform(X[-a:][i1]))

    t = (np.arange(n) / n)[:, None]
    kf = KFold(5, shuffle=True, random_state=seed)
    reg = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    rmse_x = root_mean_squared_error(y, cross_val_predict(reg, X, y, cv=kf))
    rmse_xt = root_mean_squared_error(y, cross_val_predict(reg, np.hstack([X, t]), y, cv=kf))

    i40, i80 = int(0.4 * n), int(0.8 * n)
    e_old = root_mean_squared_error(y[i80:], _rf(seed).fit(X[:i40], y[:i40]).predict(X[i80:]))
    e_new = root_mean_squared_error(y[i80:], _rf(seed).fit(X[i40:i80], y[i40:i80]).predict(X[i80:]))

    return {
        "n": n, "p": X.shape[1], "data_drift_auc": auc, "mmd2_ini_fin": mmd2,
        "concept_gain": 1 - rmse_xt / rmse_x, "recency_ratio": e_old / e_new,
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["synthetic", "real", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", nargs="*", default=None, help="limitar a estos datasets")
    ap.add_argument("--tag", default="")
    ap.add_argument("--models", nargs="*", default=None, help="correr solo estos modelos")
    ap.add_argument("--no-diag", action="store_true", help="omitir diagnóstico de drift")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    res_path = RESULTS_DIR / f"benchmark_{args.suite}{tag}.csv"
    diag_path = RESULTS_DIR / f"diagnostics_{args.suite}{tag}.csv"

    jobs = []
    if args.suite in ("synthetic", "all"):
        for name, (gen, label) in SCENARIOS.items():
            for seed in args.seeds:
                jobs.append((name, "sintetico", label, lambda g=gen, s=seed: g(s), seed))
    if args.suite in ("real", "all"):
        for name, (loader, label) in DATASETS.items():
            jobs.append((name, "real", label, loader, 0))
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]

    rows, diags = [], []
    t_start = time.perf_counter()
    for i, (name, kind, label, make, seed) in enumerate(jobs, 1):
        X, y = make()
        if not args.no_diag and seed == jobs[[j[0] for j in jobs].index(name)][4]:  # diagnóstico 1 vez por dataset
            d = drift_diagnostics(X, y, seed=seed)
            diags.append({"dataset": name, "kind": kind, "drift": label, **d})
            pd.DataFrame(diags).to_csv(diag_path, index=False)
        run_dataset(name, kind, label, X, y, seed, rows, args.models)
        pd.DataFrame(rows).to_csv(res_path, index=False)
        print(f"[{i}/{len(jobs)}] {name} seed={seed} n={len(X)} p={X.shape[1]} "
              f"({time.perf_counter() - t_start:.0f}s acumulados)", flush=True)

    print(f"\nResultados: {res_path}\nDiagnósticos: {diag_path}")


if __name__ == "__main__":
    main()
