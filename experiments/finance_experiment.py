"""Experimento financiero: ¿ayuda un modelo con drift a predecir volatilidad? (diseño fijado antes de ver resultados)

Objetivos (horizonte H = 5 días hábiles, todos en logaritmos):
  vix      log(VIX_{t+5})
  spx_rv   log de la volatilidad realizada anualizada de S&P 500 en t+1..t+5
  pen_rv   idem para PEN/USD (BCRPData)
  svxy_rv  idem para SVXY (ETF corto en VIX; cambió de -1x a -0.5x en feb-2018: quiebre estructural conocido)

Predictores (solo información hasta t): rezagos de la propia serie, VIX, retornos 5d de S&P, cobre y dólar global.

Protocolo: walk-forward con ventana expansiva (orígenes 40 %..90 %, test = 10 % siguiente) y split 80/20.
EMBARGO: al entrenar se descartan las últimas H filas antes del origen, porque su objetivo se solapa con el test.
Sin tuning: se usan los hiperparámetros por defecto de cada modelo.

Modelos: Persistencia, HAR (OLS sobre rezagos de la propia serie; estándar en volatilidad), HARX (OLS con TODAS
         las variables que reciben los árboles), RF clásico,
         RF ventana (20 % reciente), DR_recent_error (original), ADF_error y ADF_hybrid (AdaptiveDriftForest).

Uso: venv_drift_forest/Scripts/python.exe -I experiments/finance_experiment.py --seeds 0 1 2
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
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import root_mean_squared_error  # noqa: E402

from experiments.benchmark_drift import (adaptive_forest, drift_regressor, rf_clasico,  # noqa: E402
                                         rf_pesos_exp, rf_ventana, splits)
from experiments.finance_data import load_all  # noqa: E402
from src.drift_adaptive_forest.models.adaptive_ensemble import AdaptiveDriftForest  # noqa: E402

warnings.filterwarnings("ignore")
H = 5
EPS = 1e-4
RES = ROOT / "experiments" / "results"


def _fwd_rv(r: pd.Series) -> pd.Series:
    """Vol realizada anualizada de t+1..t+H."""
    return np.sqrt((r ** 2).rolling(H).mean().shift(-H) * 252)


def _bwd_rv(r: pd.Series, w: int) -> pd.Series:
    return np.sqrt((r ** 2).rolling(w).mean() * 252)


def build(target: str, raw: pd.DataFrame):
    """Devuelve (X, y, columna_persistencia, columnas_HAR, fechas)."""
    raw = raw.loc[raw["vix"].notna() & raw["spx"].notna()]  # solo días de mercado de EE. UU. abierto
    todas = ["vix", "spx", "copper", "dxy", "svxy", "pen", "btc", "jpy", "brent", "tnx", "vix3m",
             "eur", "gold", "ndx", "rut"]
    px = raw[todas].ffill()
    extra = ["vix3m"] if target == "vix_ts" else ([target.split("_")[0]] if target.endswith("_rv") else [])
    px = px.dropna(subset=["vix", "spx", "copper", "dxy", "pen"] + extra)
    r = np.log(px).diff()
    r["tnx"] = px["tnx"].diff()  # rendimiento: cambio en puntos porcentuales, no retorno log
    f = pd.DataFrame(index=px.index)
    f["l_vix"] = np.log(px["vix"])
    f["l_vix_chg5"] = f["l_vix"] - f["l_vix"].shift(5)
    f["l_vix_ma22"] = f["l_vix"].rolling(22).mean()
    f["l_vix_ma5"] = f["l_vix"].rolling(5).mean()
    for k in ("spx", "pen", "svxy", "btc", "jpy", "brent", "tnx", "eur", "gold", "ndx", "rut"):
        if k in r:
            f[f"lrv1_{k}"] = np.log(r[k].abs() * np.sqrt(252) + EPS)
            f[f"lrv5_{k}"] = np.log(_bwd_rv(r[k], 5) + EPS)
            f[f"lrv22_{k}"] = np.log(_bwd_rv(r[k], 22) + EPS)
    f["ret5_spx"] = r["spx"].rolling(5).sum()
    f["ret5_cu"] = r["copper"].rolling(5).sum()
    f["ret5_dxy"] = r["dxy"].rolling(5).sum()
    f["dd_spx"] = np.log(px["spx"] / px["spx"].rolling(252, min_periods=60).max())

    f["l_ts"] = np.log(px["vix"] / px["vix3m"])  # estructura de plazos: >0 = backwardation (estrés)
    if target in ("vix", "vix_ts"):
        y = np.log(px["vix"]).shift(-H)
        cols = ["l_vix", "l_vix_chg5", "l_vix_ma5", "l_vix_ma22", "ret5_spx", "lrv5_spx", "lrv22_spx",
                "ret5_cu", "ret5_dxy", "dd_spx"]
        if target == "vix_ts":
            cols += ["l_ts"]
        persist, har = "l_vix", ["l_vix", "l_vix_ma5", "l_vix_ma22"]
    else:
        k = target.split("_")[0]
        y = np.log(_fwd_rv(r[k]) + EPS)
        own = [f"lrv1_{k}", f"lrv5_{k}", f"lrv22_{k}"]
        cols = own + ["l_vix", "l_vix_chg5", "ret5_spx", "ret5_cu", "ret5_dxy", "dd_spx"]
        if k != "spx":
            cols += ["lrv5_spx"]
        persist, har = f"lrv5_{k}", own
    X = f[cols]
    ok = X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok], persist, har, X.index[ok]


def make_models(columns, persist, har):
    ip, ih = columns.index(persist), [columns.index(c) for c in har]

    def persistencia(Xtr, ytr, Xte, seed):
        return Xte[:, ip], None

    def har_ols(Xtr, ytr, Xte, seed):
        return LinearRegression().fit(Xtr[:, ih], ytr).predict(Xte[:, ih]), None

    def har_x(Xtr, ytr, Xte, seed):
        return LinearRegression().fit(Xtr, ytr).predict(Xte), None

    def residual(make_forest):
        """HARX para el nivel + bosque sobre los residuos (corrección no lineal). Hipótesis del paso 18."""
        def fn(Xtr, ytr, Xte, seed):
            lin = LinearRegression().fit(Xtr, ytr)
            forest = make_forest(seed).fit(Xtr, ytr - lin.predict(Xtr))
            w = forest.weights_summary() if hasattr(forest, "weights_summary") else None
            return lin.predict(Xte) + forest.predict(Xte), w
        return fn

    return {
        "Persistencia": persistencia,
        "HAR": har_ols,
        "HARX": har_x,
        "RF_clasico": rf_clasico,
        "RF_ventana": rf_ventana,
        "RF_pesos_exp": rf_pesos_exp,
        "DR_recent_error": drift_regressor("recent_error"),
        "ADF_error": adaptive_forest("error"),
        "ADF_hybrid": adaptive_forest("hybrid"),
        "HARX+RF_resid": residual(lambda s: RandomForestRegressor(n_estimators=100, min_samples_leaf=20,
                                                                  random_state=s, n_jobs=-1)),
        "HARX+ADF_resid": residual(lambda s: AdaptiveDriftForest(n_estimators=100, min_samples_leaf=20,
                                                                 random_state=s)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--targets", nargs="*", default=["vix", "spx_rv", "pen_rv", "svxy_rv"])
    ap.add_argument("--models", nargs="*", default=None, help="correr solo estos modelos")
    ap.add_argument("--out", default="finance_results.csv")
    args = ap.parse_args()

    raw = load_all()
    rows = []
    for target in args.targets:
        X, y, persist, har, dates = build(target, raw)
        cols = list(X.columns)
        Xa, ya = X.to_numpy(float), y.to_numpy(float)
        print(f"\n{target}: n={len(Xa)} p={Xa.shape[1]} {dates.min():%Y-%m-%d} -> {dates.max():%Y-%m-%d}", flush=True)
        models = make_models(cols, persist, har)
        for protocol in ("walk", "static"):
            for step, a, b in splits(len(Xa), protocol):
                tr_end = a - H  # embargo
                for seed in args.seeds:
                    for mname, fn in models.items():
                        if args.models and mname not in args.models:
                            continue
                        if seed > args.seeds[0] and mname in ("Persistencia", "HAR", "HARX"):
                            continue
                        pred, w = fn(Xa[:tr_end], ya[:tr_end], Xa[a:b], seed)
                        rows.append({"target": target, "protocol": protocol, "step": step, "seed": seed,
                                     "model": mname, "n_train": tr_end, "n_test": b - a,
                                     "test_from": f"{dates[a]:%Y-%m-%d}", "test_to": f"{dates[b - 1]:%Y-%m-%d}",
                                     "rmse": root_mean_squared_error(ya[a:b], pred),
                                     "weights": json.dumps(w) if w is not None else ""})
                print(f"  {protocol} paso {step} ok", flush=True)
        pd.DataFrame(rows).to_csv(RES / args.out, index=False)
    print(f"\nResultados: {RES / args.out}")


if __name__ == "__main__":
    main()
