"""Experimento macro mensual (tipo Stock y Watson / FRED-MD) con y sin EPU. Diseño registrado en BITACORA §12.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/macro_experiment.py --seeds 0 1 2
(los datos ya deben estar en caché; ver experiments/macro_data.py)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import root_mean_squared_error  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments.benchmark_drift import (adaptive_forest, drift_regressor, rf_clasico,  # noqa: E402
                                         rf_ventana, splits)
from experiments.macro_data import load_epu_monthly, load_raw, transform  # noqa: E402

warnings.filterwarnings("ignore")
RES = ROOT / "experiments" / "results"
HORIZONTES = (1, 3, 6, 12)
OBJETIVOS = {"inflacion": "CPIAUCSL", "produccion": "INDPRO", "desempleo": "UNRATE"}


def build(objetivo: str, h: int, raw: pd.DataFrame, panel: pd.DataFrame, epu: pd.Series):
    """Devuelve X (DataFrame), y, nombres de columnas AR, panel y EPU, y fechas."""
    serie = raw[OBJETIVOS[objetivo]]
    if objetivo == "desempleo":
        m = serie.diff()
        y = serie.shift(-h) - serie
        a12 = serie - serie.shift(12)
    else:
        m = 1200 * np.log(serie).diff()
        y = (1200 / h) * np.log(serie.shift(-h) / serie)
        a12 = 100 * np.log(serie / serie.shift(12))
    ar = pd.DataFrame({"ar_m0": m, "ar_m1": m.shift(1), "ar_m2": m.shift(2), "ar_12": a12})
    le = np.log(epu).reindex(raw.index)
    ep = pd.DataFrame({"epu": le, "epu3": le.rolling(3).mean(), "epu_dev": le - le.rolling(12).mean()})
    X = pd.concat([ar, panel, ep], axis=1)
    X = X.loc["1961-01-01":]
    ok = X.notna().all(axis=1) & y.reindex(X.index).notna()
    return X[ok], y.reindex(X.index)[ok], list(ar.columns), list(panel.columns), list(ep.columns)


def make_models(cols, c_ar, c_panel, c_epu):
    idx = {c: i for i, c in enumerate(cols)}
    I_ar = [idx[c] for c in c_ar]
    I_pan = [idx[c] for c in c_panel]
    I_epu = [idx[c] for c in c_epu]

    def ols(I):
        def fn(Xtr, ytr, Xte, seed):
            return LinearRegression().fit(Xtr[:, I], ytr).predict(Xte[:, I]), None
        return fn

    def di(extra):
        def fn(Xtr, ytr, Xte, seed):
            sc = StandardScaler().fit(Xtr[:, I_pan])
            pca = PCA(n_components=3, random_state=0).fit(sc.transform(Xtr[:, I_pan]))
            Ftr, Fte = pca.transform(sc.transform(Xtr[:, I_pan])), pca.transform(sc.transform(Xte[:, I_pan]))
            Ztr = np.hstack([Xtr[:, I_ar + extra], Ftr])
            Zte = np.hstack([Xte[:, I_ar + extra], Fte])
            return LinearRegression().fit(Ztr, ytr).predict(Zte), None
        return fn

    def sub(fn, I):
        def g(Xtr, ytr, Xte, seed):
            return fn(Xtr[:, I], ytr, Xte[:, I], seed)
        return g

    base, con_epu = I_ar + I_pan, I_ar + I_pan + I_epu
    return {
        "AR": ols(I_ar), "AR+EPU": ols(I_ar + I_epu),
        "DI": di([]), "DI+EPU": di(I_epu),
        "ARX": ols(base), "ARX+EPU": ols(con_epu),
        "RF_clasico": sub(rf_clasico, base), "RF_clasico+EPU": sub(rf_clasico, con_epu),
        "RF_ventana": sub(rf_ventana, base),
        "DR_recent_error": sub(drift_regressor("recent_error"), base),
        "ADF_error": sub(adaptive_forest("error"), base), "ADF_error+EPU": sub(adaptive_forest("error"), con_epu),
    }


LINEALES = {"AR", "AR+EPU", "DI", "DI+EPU", "ARX", "ARX+EPU"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="macro_results.csv")
    args = ap.parse_args()

    raw = load_raw()
    panel = transform(raw)
    epu = load_epu_monthly()
    rows = []
    for objetivo in OBJETIVOS:
        for h in HORIZONTES:
            X, y, c_ar, c_pan, c_epu = build(objetivo, h, raw, panel, epu)
            cols = list(X.columns)
            Xa, ya, fechas = X.to_numpy(float), y.to_numpy(float), X.index
            models = make_models(cols, c_ar, c_pan, c_epu)
            for protocol in ("walk", "static"):
                for step, a, b in splits(len(Xa), protocol):
                    tr_end = a - h  # embargo: el objetivo de las últimas h filas se solapa con el test
                    for seed in args.seeds:
                        for name, fn in models.items():
                            if seed != args.seeds[0] and name in LINEALES:
                                continue
                            pred, w = fn(Xa[:tr_end], ya[:tr_end], Xa[a:b], seed)
                            rows.append({"objetivo": objetivo, "h": h, "protocol": protocol, "step": step,
                                         "seed": seed, "model": name, "n_train": tr_end, "n_test": b - a,
                                         "test_from": f"{fechas[a]:%Y-%m}", "test_to": f"{fechas[b - 1]:%Y-%m}",
                                         "rmse": root_mean_squared_error(ya[a:b], pred),
                                         "weights": json.dumps(w) if w is not None else ""})
            print(f"{objetivo:10s} h={h:2d} n={len(Xa)} p={Xa.shape[1]} {fechas[0]:%Y-%m}->{fechas[-1]:%Y-%m}",
                  flush=True)
            pd.DataFrame(rows).to_csv(RES / args.out, index=False)
    print(f"\nResultados: {RES / args.out}")


if __name__ == "__main__":
    main()
