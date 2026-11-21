"""Evalúa H5-H8 (BITACORA §13) y genera las figuras 16-18 del experimento de matrícula.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/analizar_matricula.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from experiments.graficar_analisis import ACENTO, ALERTA, AZUL_CLARO, GRIS, NAVY, SOPORTE, guardar  # noqa: E402

RES = ROOT / "experiments" / "results"
ESTILO = {
    "Tendencia": (AZUL_CLARO, "v", "Tendencia (promedio 3 años)"),
    "ETS": ("#4F6D8A", "X", "ETS (Holt amortiguado)"),
    "OLS_global": (GRIS, "^", "OLS global"),
    "LightGBM": (SOPORTE, "P", "LightGBM"),
    "RF_clasico": ("#5B7C99", "D", "RF clásico"),
    "RF_ventana": (NAVY, "s", "RF ventana reciente"),
    "DR_recent_error": ("#9DBCD4", "h", "DriftRegressor original"),
    "ADF_error": (ACENTO, "o", "AdaptiveDriftForest"),
}
ORDEN = list(ESTILO)


def geo(s):
    return float(np.exp(np.mean(np.log(s))))


def cargar():
    r = pd.read_csv(RES / "matricula_results.csv")
    base = r[r.model == "Ingenuo"].set_index(["h", "origen", "control"])[["rmse", "mae"]]
    r = r.join(base, on=["h", "origen", "control"], rsuffix="_ing")
    r["r_rmse"] = r["rmse"] / r["rmse_ing"]
    r["r_mae"] = r["mae"] / r["mae_ing"]
    return r


def fig_resumen(r):
    t = r[r.control == "todas"].groupby(["model", "h"])[["r_rmse", "r_mae"]].agg(geo).unstack("h")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(ORDEN))[::-1]
    for ax, met, tit in ((axes[0], "r_rmse", "RMSE"), (axes[1], "r_mae", "MAE (robusto a extremos)")):
        ax.axvline(1, color=GRIS, ls="--", lw=1)
        for hh, mk, desp in ((1, "o", 0.12), (2, "s", -0.12)):
            vals = [t.loc[mo, (met, hh)] for mo in ORDEN]
            cols = [ESTILO[mo][0] for mo in ORDEN]
            ax.scatter(vals, y + desp, c=cols, marker=mk, s=46, edgecolor="white", linewidth=0.5,
                       label=f"h = {hh} año{'s' if hh > 1 else ''}")
        ax.set_yticks(y, [ESTILO[mo][2] for mo in ORDEN])
        ax.set_title(tit, loc="left", color=NAVY, fontweight="bold")
        ax.set_xlabel("error / error del ingenuo (menos de 1 = mejor)")
        ax.grid(axis="y", visible=False)
    axes[1].legend(loc="lower right")
    fig.suptitle("Matrícula universitaria (3 196 instituciones, 2012–2023): error relativo al pronóstico ingenuo",
                 x=0.01, ha="left", color=NAVY, fontweight="bold", fontsize=11, y=1.02)
    guardar(fig, "16_matricula_resumen.png")
    return t


def fig_tiempo(r):
    d = r[(r.control == "todas") & (r.h == 1)]
    fig, ax = plt.subplots(figsize=(10.5, 4))
    for mo in ("ETS", "LightGBM", "RF_clasico", "ADF_error", "Tendencia"):
        g = d[d.model == mo].sort_values("origen")
        c, mk, lab = ESTILO[mo]
        ax.plot(g.origen + 1, g.r_mae, color=c, marker=mk, lw=1.5, ms=6, label=lab)
    ax.axhline(1, color=GRIS, ls="--", lw=1)
    ax.axvspan(2019.5, 2021.5, color=ALERTA, alpha=0.12, lw=0)
    ax.text(2020.5, 0.98, "COVID-19", transform=ax.get_xaxis_transform(), ha="center", va="top",
            color="#9A6200", fontsize=8.5)
    ax.set_xlabel("año pronosticado (h = 1)")
    ax.set_ylabel("MAE / MAE del ingenuo")
    ax.set_title("¿Cuándo gana cada modelo? Matrícula a 1 año", loc="left", color=NAVY, fontweight="bold")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="x", visible=False)
    guardar(fig, "17_matricula_en_el_tiempo.png")


def fig_sector(r):
    sec = ["pública", "sin fines de lucro", "con fines de lucro"]
    d = r[r.control.isin(sec)].pivot_table(index=["h", "origen", "control"], columns="model", values="rmse")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(sec))
    for k, (rival, c) in enumerate((("RF_clasico", "#5B7C99"), ("LightGBM", SOPORTE), ("ETS", "#4F6D8A"))):
        vals = [geo((d["ADF_error"] / d[rival]).xs(s, level="control")) for s in sec]
        ax.bar(x + (k - 1) * 0.26, np.array(vals) - 1, bottom=1, width=0.24, color=c,
               label=f"AdaptiveDriftForest / {ESTILO[rival][2]}")
    ax.axhline(1, color=GRIS, lw=1)
    ax.set_xticks(x, ["Pública", "Privada sin fines de lucro", "Privada con fines de lucro"])
    ax.set_ylabel("cociente de RMSE\n(bajo 1 = gana el modelo nuevo)")
    ax.set_title("¿Dónde ayuda más el modelo adaptativo? (por tipo de institución)", loc="left",
                 color=NAVY, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    ax.grid(axis="x", visible=False)
    guardar(fig, "18_matricula_por_sector.png")


def main():
    r = cargar()
    t = fig_resumen(r)
    print("Error relativo al ingenuo (media geométrica de los cortes):")
    print(t.round(3).to_string())
    fig_tiempo(r)
    fig_sector(r)
    w = r[r.control == "todas"].pivot_table(index=["h", "origen"], columns="model", values="rmse")
    wm = r[r.control == "todas"].pivot_table(index=["h", "origen"], columns="model", values="mae")
    print("\nHipótesis registradas (RMSE; entre paréntesis MAE):")
    for hip, rival in (("H5", "RF_clasico"), ("H6", "LightGBM"), ("H7", "ETS")):
        k, n = int((w["ADF_error"] < w[rival]).sum()), len(w)
        km = int((wm["ADF_error"] < wm[rival]).sum())
        print(f"  {hip} ADF_error < {rival:10s}: {k}/{n} p={binomtest(k, n).pvalue:.4f}   "
              f"(MAE {km}/{n} p={binomtest(km, n).pvalue:.4f})")
    d = r.pivot_table(index=["h", "origen", "control"], columns="model", values="rmse")
    lr = np.log(d["ADF_error"] / d["RF_clasico"]).unstack("control")
    dif = lr["con fines de lucro"] - lr["pública"]
    k = int((dif < 0).sum())
    print(f"  H8 ventaja ADF/RF mayor en con fines de lucro que en públicas: {k}/{len(dif)} "
          f"p={binomtest(k, len(dif)).pvalue:.4f} | media log-cociente: con fines {lr['con fines de lucro'].mean():+.3f}, "
          f"públicas {lr['pública'].mean():+.3f}")
    print("\nOtras (no registradas):")
    for a, b in (("LightGBM", "ETS"), ("RF_ventana", "RF_clasico"), ("ETS", "Ingenuo"), ("LightGBM", "RF_clasico")):
        k = int((w[a] < w[b]).sum())
        km = int((wm[a] < wm[b]).sum())
        print(f"      {a} < {b}: RMSE {k}/{len(w)}  MAE {km}/{len(w)}")


if __name__ == "__main__":
    main()
