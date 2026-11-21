"""Pronóstico de demanda de matrícula universitaria (IPEDS). Diseño registrado en BITACORA §13.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/matricula_experiment.py --seeds 0 1
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
from joblib import Parallel, delayed  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from experiments.benchmark_drift import adaptive_forest, drift_regressor, rf_clasico, rf_ventana  # noqa: E402
from experiments.matricula_data import panel_balanceado  # noqa: E402

warnings.filterwarnings("ignore")
RES = ROOT / "experiments" / "results"
HORIZONTES = (1, 2)
ORIGEN_INICIAL = 2012


def construir_filas(m: pd.DataFrame, h: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    L = np.log(m)
    años = list(m.columns)
    g = L.diff(axis=1)
    nac = g.mean(axis=0)
    ctrl_media = g.groupby(h["CONTROL"]).mean()
    filas = []
    for t in años:
        if t - 3 < años[0]:
            continue
        d = pd.DataFrame({
            "unitid": m.index, "t": t,
            "g0": g[t], "g1": g[t - 1], "g2": g[t - 2], "g3m": (L[t] - L[t - 3]) / 3,
            "log_size": L[t],
            "c_pub": (h["CONTROL"] == 1).astype(float), "c_np": (h["CONTROL"] == 2).astype(float),
            "c_fp": (h["CONTROL"] == 3).astype(float),
            "n_4": (h["ICLEVEL"] == 1).astype(float), "n_2": (h["ICLEVEL"] == 2).astype(float),
            "n_menos2": (h["ICLEVEL"] == 3).astype(float),
            "g_nacional": nac[t], "g_mismo_tipo": h["CONTROL"].map(ctrl_media[t]).to_numpy(),
            "control": h["CONTROL"].to_numpy(),
            "y": (L[t + horizonte] - L[t]) if t + horizonte in años else np.nan,
        })
        filas.append(d)
    f = pd.concat(filas, ignore_index=True)
    return f.sort_values(["t", "unitid"]).reset_index(drop=True)  # orden temporal (lo usan RF ventana y ADF)


FEATS = ["g0", "g1", "g2", "g3m", "log_size", "c_pub", "c_np", "c_fp", "n_4", "n_2", "n_menos2",
         "g_nacional", "g_mismo_tipo"]


def _ets_una(serie: np.ndarray, horizonte: int) -> float:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        fit = ExponentialSmoothing(serie, trend="add", damped_trend=True,
                                   initialization_method="estimated").fit()
        return float(fit.forecast(horizonte)[-1] - serie[-1])
    except Exception:
        return 0.0


def ets_panel(m: pd.DataFrame, origen: int, horizonte: int, unitids) -> np.ndarray:
    L = np.log(m.loc[unitids, [c for c in m.columns if c <= origen]]).to_numpy()
    pred = Parallel(n_jobs=-1, batch_size=64)(delayed(_ets_una)(row, horizonte) for row in L)
    return np.asarray(pred)


def lgbm(Xtr, ytr, Xte, seed):
    return LGBMRegressor(random_state=seed, verbose=-1).fit(Xtr, ytr).predict(Xte), None


MODELOS_ML = {
    "OLS_global": lambda Xtr, ytr, Xte, s: (LinearRegression().fit(Xtr, ytr).predict(Xte), None),
    "LightGBM": lgbm,
    "RF_clasico": rf_clasico,
    "RF_ventana": rf_ventana,
    "DR_recent_error": drift_regressor("recent_error"),
    "ADF_error": adaptive_forest("error"),
}
DETERMINISTAS = {"OLS_global", "Ingenuo", "Tendencia", "ETS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="matricula_results.csv")
    args = ap.parse_args()

    m, h = panel_balanceado()
    filas_pred = []
    resumen = []
    for horizonte in HORIZONTES:
        f = construir_filas(m, h, horizonte)
        for origen in range(ORIGEN_INICIAL, max(m.columns) - horizonte + 1):
            tr = f[(f.t + horizonte <= origen)]
            te = f[f.t == origen]
            Xtr, ytr, Xte, yte = tr[FEATS].to_numpy(), tr["y"].to_numpy(), te[FEATS].to_numpy(), te["y"].to_numpy()
            preds = {
                "Ingenuo": np.zeros(len(te)),
                "Tendencia": te["g3m"].to_numpy() * horizonte,
                "ETS": ets_panel(m, origen, horizonte, te["unitid"].to_numpy()),
            }
            for seed in args.seeds:
                for nombre, fn in MODELOS_ML.items():
                    if nombre in DETERMINISTAS and seed != args.seeds[0]:
                        continue
                    p, _ = fn(Xtr, ytr, Xte, seed)
                    preds[nombre] = preds.get(nombre, 0) + p / (1 if nombre in DETERMINISTAS else len(args.seeds))
            for nombre, p in preds.items():
                err = yte - p
                for ctrl in (0, 1, 2, 3):
                    mask = np.ones(len(te), bool) if ctrl == 0 else (te["control"].to_numpy() == ctrl)
                    resumen.append({"h": horizonte, "origen": origen, "model": nombre,
                                    "control": ["todas", "pública", "sin fines de lucro", "con fines de lucro"][ctrl],
                                    "n": int(mask.sum()), "n_train": len(tr),
                                    "rmse": float(np.sqrt(np.mean(err[mask] ** 2))),
                                    "mae": float(np.mean(np.abs(err[mask])))})
            print(f"h={horizonte} origen={origen} train={len(tr)} test={len(te)}", flush=True)
            pd.DataFrame(resumen).to_csv(RES / args.out, index=False)
    print(f"\nResultados: {RES / args.out}")


if __name__ == "__main__":
    main()
