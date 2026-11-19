"""Comparación con hiperparámetros optimizados (Optuna + TimeSeriesSplit), como en el README,
pero en varios datasets y con el mismo presupuesto de trials para cada familia de modelo.

Familias:
  RF_clasico_opt  espacio del README (max_depth, min_samples_leaf, max_features)
  RF_ventana_opt  mismo espacio + fracción de ventana reciente (0.05-1.0; 1.0 == RF clásico)
  DR_opt          espacio del README para DriftRegressor (n_blocks, beta, modo, etc.)

Protocolo: train = primer 80 %, test = 20 % final. Tuning solo con el train (TimeSeriesSplit 4).
n_estimators se fija en 100 (por bloque en DR) para acotar tiempo de cómputo.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/tune_benchmark.py --trials 20
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
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from experiments.real_datasets import DATASETS  # noqa: E402
from experiments.synthetic_scenarios import SCENARIOS  # noqa: E402
from src.drift_adaptive_forest.models.block_ensemble import DriftRegressor  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
RESULTS_DIR = ROOT / "experiments" / "results"
N_TREES = 100

DEFAULT_SYNTH = ["healthcare_data", "healthcare_both", "friedman_abrupt", "covariate_shift", "highdim_mixed"]


class WindowRF:
    """RF entrenado solo con la fracción más reciente del histórico."""

    def __init__(self, window_frac=1.0, **rf_params):
        self.window_frac = window_frac
        self.rf_params = rf_params

    def fit(self, X, y):
        w = max(100, int(self.window_frac * len(X)))
        self.rf_ = RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, **self.rf_params).fit(X[-w:], y[-w:])
        return self

    def predict(self, X):
        return self.rf_.predict(X)


def tree_space(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.5, 1.0),
    }


def build(family, p, seed):
    if family == "RF_clasico_opt":
        return RandomForestRegressor(n_estimators=N_TREES, n_jobs=-1, random_state=seed, **p)
    if family == "RF_ventana_opt":
        p = dict(p)
        return WindowRF(window_frac=p.pop("window_frac"), random_state=seed, **p)
    return DriftRegressor(n_estimators_per_block=N_TREES, random_state=seed, **p)


def space(family, trial):
    if family == "RF_clasico_opt":
        return tree_space(trial)
    if family == "RF_ventana_opt":
        return {**tree_space(trial), "window_frac": trial.suggest_float("window_frac", 0.05, 1.0)}
    return {
        "n_blocks": trial.suggest_int("n_blocks", 3, 8),
        "beta": trial.suggest_float("beta", 2.0, 60.0, log=True),
        "weighting_mode": trial.suggest_categorical("weighting_mode", ["mmd", "recent_error", "hybrid"]),
        "recent_val_frac": trial.suggest_float("recent_val_frac", 0.10, 0.25),
        "alpha_hybrid": trial.suggest_float("alpha_hybrid", 0.1, 0.9),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }


def cv_rmse(family, p, X, y, seed):
    out = []
    for tr, va in TimeSeriesSplit(n_splits=4).split(X):
        out.append(root_mean_squared_error(y[va], build(family, p, seed).fit(X[tr], y[tr]).predict(X[va])))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="tuned_static.csv")
    args = ap.parse_args()

    jobs = [(n, "sintetico", (lambda g=SCENARIOS[n][0]: g(args.seed))) for n in DEFAULT_SYNTH]
    jobs += [(n, "real", f) for n, (f, _) in DATASETS.items()]
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / args.out
    rows = []
    t0 = time.perf_counter()
    for name, kind, make in jobs:
        X, y = make()
        X, y = np.asarray(X, float), np.asarray(y, float)
        cut = int(0.8 * len(X))
        Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]
        for family in ("RF_clasico_opt", "RF_ventana_opt", "DR_opt"):
            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=args.seed))
            study.optimize(lambda tr: cv_rmse(family, space(family, tr), Xtr, ytr, args.seed),
                           n_trials=args.trials)
            model = build(family, study.best_params, args.seed).fit(Xtr, ytr)
            pred = model.predict(Xte)
            weights = getattr(model, "last_weights_", None)
            rows.append({
                "dataset": name, "kind": kind, "model": family,
                "cv_rmse": study.best_value,
                "rmse": root_mean_squared_error(yte, pred),
                "mae": mean_absolute_error(yte, pred), "r2": r2_score(yte, pred),
                "best_params": json.dumps(study.best_params),
                "weights": json.dumps([round(float(w), 4) for w in weights]) if weights is not None else "",
            })
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"{name:15s} {family:15s} cv={study.best_value:.4f} test={rows[-1]['rmse']:.4f} "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"\nResultados: {out_path}")


if __name__ == "__main__":
    main()
