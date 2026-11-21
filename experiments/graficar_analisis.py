"""Gráficos de análisis del benchmark (sintético, real y financiero) con la paleta oficial CCDE.

Salida: experiments/figuras/*.png
Uso:    venv_drift_forest/Scripts/python.exe -I experiments/graficar_analisis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = ROOT / "experiments" / "results"
OUT = ROOT / "experiments" / "figuras"

NAVY, SOPORTE, ACENTO, SUAVE, ALERTA = "#0A2540", "#1E3D59", "#17A2B8", "#E2F0F9", "#F5A623"
AZUL_CLARO, GRIS = "#9DBCD4", "#7A8FA3"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": GRIS,
    "axes.labelcolor": NAVY, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": SOPORTE, "ytick.color": SOPORTE, "legend.frameon": False, "legend.fontsize": 8.5,
    "axes.grid": True, "grid.color": SUAVE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "figure.dpi": 100,
})

# estilo fijo por modelo (color + marcador, para no depender solo del color)
ESTILO = {
    "ADF_error": (ACENTO, "o", "AdaptiveDriftForest (nuevo)"),
    "ADF_hybrid": (SOPORTE, "P", "AdaptiveDriftForest hybrid"),
    "RF_ventana": (NAVY, "s", "RF ventana reciente"),
    "RF_clasico": (GRIS, "D", "RF clásico"),
    "DR_recent_error": (AZUL_CLARO, "D", "DriftRegressor original"),
    "DR_mmd": (GRIS, "^", "DriftRegressor mmd"),
    "Persistencia": (AZUL_CLARO, "v", "Persistencia"),
    "HARX": ("#4F6D8A", "X", "HAR-X (lineal, mismas variables)"),
    "HARX+RF_resid": ("#5B7C99", "h", "HAR-X + RF sobre residuos"),
    "HARX+ADF_resid": ("#0E7C8C", "*", "HAR-X + ADF sobre residuos"),
}


def guardar(fig, nombre):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / nombre, dpi=150)
    plt.close(fig)
    print("guardada", OUT / nombre)


def geo(x):
    return float(np.exp(np.mean(np.log(x))))


# ------------------------------------------------------------------ 1 y 2: ratios vs RF clásico
def cargar_benchmark():
    df = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("benchmark_*.csv"))], ignore_index=True)
    key = ["dataset", "seed", "protocol", "step"]
    ref = df[df.model == "RF_clasico"].set_index(key)["rmse"].rename("ref")
    d = df.join(ref, on=key)
    d["ratio"] = d["rmse"] / d["ref"]
    return d


def dotplot(ax, tabla, modelos, titulo, xlim=None, ticks=None):
    orden = tabla["ADF_error"].sort_values(ascending=False).index.tolist()
    y = np.arange(len(orden))
    ax.axvline(1.0, color=GRIS, ls="--", lw=1)
    for m in modelos:
        c, mk, lab = ESTILO[m]
        ax.scatter(tabla.loc[orden, m], y, color=c, marker=mk, s=46 if m == "ADF_error" else 30,
                   label=lab, zorder=3 if m == "ADF_error" else 2, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, orden)
    ax.set_xscale("log")
    ticks = ticks or [0.3, 0.5, 0.7, 1.0, 1.3]
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_title(titulo, loc="left", color=NAVY, fontweight="bold")
    ax.set_xlabel("RMSE / RMSE del RF clásico (menos de 1 = mejor)")
    ax.grid(axis="y", visible=False)


def fig_ratios(d):
    modelos = ["ADF_error", "RF_ventana", "DR_recent_error", "DR_mmd"]
    for kind, nombre, protos in (("sintetico", "01_sinteticos_vs_rf.png", ["walk", "static"]),
                                 ("real", "02_reales_vs_rf.png", ["walk", "static"])):
        sub = d[d.kind == kind]
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * sub.dataset.nunique() + 2.0), sharey=True)
        for ax, proto in zip(axes, protos):
            t = (sub[sub.protocol == proto].groupby(["dataset", "model"]).ratio.apply(geo)
                 .unstack("model"))
            xl, tk = ((0.25, 1.45), [0.3, 0.5, 0.7, 1.0, 1.3]) if kind == "sintetico" else                      ((0.55, 1.6), [0.6, 0.8, 1.0, 1.2, 1.5])
            dotplot(ax, t, modelos, "Walk-forward" if proto == "walk" else "Split 80/20", xl, tk)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
        fig.subplots_adjust(bottom=0.2 if kind == "sintetico" else 0.3)
        fig.suptitle("Sintéticos: error relativo al RF clásico" if kind == "sintetico"
                     else "Datos reales: error relativo al RF clásico", x=0.01, ha="left",
                     color=NAVY, fontweight="bold", fontsize=11)
        guardar(fig, nombre)


# ------------------------------------------------------------------ pesos apilados
def color_candidato(nombre):
    if nombre.startswith("bloque"):
        return GRIS
    f = float(nombre.split("_")[1])
    azules = {0.10: "#CFE3F1", 0.20: AZUL_CLARO, 0.35: ACENTO, 0.60: SOPORTE, 1.00: NAVY}
    return azules.get(f, ACENTO)


def pesos_apilados(ax, filas, titulo, xlabels):
    nombres = []
    for w in filas:
        for k in w:
            if k not in nombres:
                nombres.append(k)
    nombres.sort(key=lambda n: (n.startswith("bloque"), float(n.split("_")[1]) if n.startswith("ventana") else 0, n))
    base = np.zeros(len(filas))
    x = np.arange(len(filas))
    for n in nombres:
        h = np.array([w.get(n, 0.0) for w in filas])
        ax.bar(x, h, bottom=base, color=color_candidato(n), width=0.72, edgecolor="white", linewidth=0.4)
        base += h
    ax.set_xticks(x, xlabels)
    ax.set_ylim(0, 1)
    ax.set_title(titulo, loc="left", color=NAVY, fontweight="bold")
    ax.grid(axis="x", visible=False)
    return nombres


def leyenda_pesos(fig, ax):
    from matplotlib.patches import Patch
    items = [("ventana 10 %", "#CFE3F1"), ("ventana 20 %", AZUL_CLARO), ("ventana 35 %", ACENTO),
             ("ventana 60 %", SOPORTE), ("ventana 100 % (= RF clásico)", NAVY), ("bloque histórico", GRIS)]
    ax.legend(handles=[Patch(color=c, label=l) for l, c in items], ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.22))


def fig_pesos_sinteticos():
    df = pd.read_csv(RES / "benchmark_synthetic_adf.csv")
    df = df[(df.model == "ADF_error") & (df.protocol == "walk") & (df.seed == 0)]
    casos = [("healthcare_none", "Sin drift"), ("healthcare_both", "Data + concept drift"),
             ("friedman_recurring", "Concept drift recurrente"), ("covariate_shift", "Solo data drift fuerte")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, (ds, tit) in zip(axes, casos):
        g = df[df.dataset == ds].sort_values("step")
        pesos_apilados(ax, [json.loads(w) for w in g.weights], tit, [f"{40 + 10 * s}%" for s in g.step])
        ax.set_xlabel("origen del reentrenamiento\n(% de la serie)")
    axes[0].set_ylabel("peso en el ensamble")
    leyenda_pesos(fig, axes[1])
    fig.suptitle("¿Cuánta memoria usa el modelo? Pesos de AdaptiveDriftForest en cada reentrenamiento",
                 x=0.01, y=1.08, ha="left", color=NAVY, fontweight="bold", fontsize=11)
    guardar(fig, "03_pesos_sinteticos.png")


# ------------------------------------------------------------------ finanzas
EVENTOS = {"svxy_rv": [("2018-02-05", "SVXY pasa de −1x a −0.5x")],
           "vix_ts": [("2008-09-15", "Lehman"), ("2020-03-16", "COVID-19")],
           "jpy_rv": [("2016-09-21", "BoJ: control de curva"), ("2022-12-20", "BoJ amplía banda")],
           "brent_rv": [("2014-11-27", "OPEP no recorta"), ("2020-03-09", "guerra de precios"),
                        ("2022-02-24", "Ucrania")],
           "tnx_rv": [("2020-03-16", "COVID-19"), ("2022-03-16", "Fed inicia alzas")],
           "btc_rv": [("2020-03-12", "COVID-19"), ("2022-11-11", "quiebra FTX")],
           "vix": [("2008-09-15", "Lehman"), ("2020-03-16", "COVID-19")],
           "spx_rv": [("2008-09-15", "Lehman"), ("2020-03-16", "COVID-19")],
           "pen_rv": [("2008-09-15", "Lehman"), ("2020-03-16", "COVID-19")]}
NOMBRES = {"vix": "VIX (nivel a 5 días)", "spx_rv": "Volatilidad S&P 500 (5 días)",
           "pen_rv": "Volatilidad PEN/USD (5 días)", "svxy_rv": "Volatilidad SVXY (5 días)",
           "vix_ts": "VIX + estructura de plazos", "jpy_rv": "Volatilidad USD/JPY (5 días)",
           "brent_rv": "Volatilidad Brent (5 días)", "tnx_rv": "Volatilidad bono 10 años (5 días)",
           "btc_rv": "Volatilidad Bitcoin (5 días)",
           "eur_rv": "Volatilidad EUR/USD (5 días)", "gold_rv": "Volatilidad oro (5 días)",
           "ndx_rv": "Volatilidad Nasdaq-100 (5 días)", "rut_rv": "Volatilidad Russell 2000 (5 días)"}
GRUPO_A = ["vix", "spx_rv", "pen_rv", "svxy_rv"]
GRUPO_B = ["vix_ts", "jpy_rv", "brent_rv", "tnx_rv", "btc_rv"]


def cargar_fin(patron="finance_results*.csv"):
    f = pd.concat([pd.read_csv(p) for p in sorted(RES.glob(patron))], ignore_index=True)
    medias = f.groupby(["target", "protocol", "step", "model"]).agg(
        rmse=("rmse", "mean"), test_from=("test_from", "first"), test_to=("test_to", "first"),
        weights=("weights", "first")).reset_index()
    har = medias[medias.model == "HAR"].set_index(["target", "protocol", "step"])["rmse"].rename("har")
    medias = medias.join(har, on=["target", "protocol", "step"])
    medias["ratio"] = medias["rmse"] / medias["har"]
    return f, medias


def fig_fin_resumen(m, modelos=None, nombre="04_finanzas_vs_har.png",
                    titulo="Finanzas: error relativo al HAR (línea naranja = HAR, la referencia estándar en volatilidad)"):
    modelos = modelos or ["Persistencia", "HARX", "RF_clasico", "RF_ventana", "DR_recent_error",
                          "ADF_error", "ADF_hybrid"]
    modelos = [x for x in modelos if x in set(m.model)]
    estilo = dict(ESTILO)
    estilo["RF_clasico"] = (GRIS, "D", "RF clásico")
    targets = [t for t in NOMBRES if t in set(m.target)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.62 * len(targets) + 2.6), sharey=True)
    for ax, proto in zip(axes, ("walk", "static")):
        sub = m[m.protocol == proto]
        t = sub.groupby(["target", "model"]).ratio.apply(geo).unstack("model").loc[targets]
        y = np.arange(len(targets))[::-1]
        ax.axvline(1.0, color=ALERTA, ls="--", lw=1.3, zorder=1)
        for mod in modelos:
            c, mk, lab = estilo[mod]
            ax.scatter(t[mod], y, color=c, marker=mk, s=110 if mk == "*" else (50 if mod == "ADF_error" else 32),
                       label=lab,
                       zorder=3 if mod == "ADF_error" else 2, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y, [NOMBRES[k] for k in targets])
        ax.set_title("Walk-forward" if proto == "walk" else "Split 80/20", loc="left",
                     color=NAVY, fontweight="bold")
        ax.set_xlabel("RMSE / RMSE del HAR (menos de 1 = mejor que HAR)")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=3)
    fig.suptitle(titulo, x=0.01, ha="left", color=NAVY, fontweight="bold", fontsize=11)
    guardar(fig, nombre)


def fig_fin_series(m, grupo, nombre):
    from experiments.finance_data import load_all
    from experiments.finance_experiment import build
    raw = load_all()
    targets = [t for t in grupo if t in set(m.target)]
    fig, axes = plt.subplots(len(targets), 1, figsize=(11, 2.5 * len(targets)))
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        X, y, *_ = build(t, raw)
        ax.plot(y.index, y.values, color=NAVY, lw=0.7, zorder=2)
        w = m[(m.target == t) & (m.protocol == "walk")]
        for step, g in w.groupby("step"):
            g = g.set_index("model")
            a, b = pd.Timestamp(g.test_from.iloc[0]), pd.Timestamp(g.test_to.iloc[0])
            r_adf, r_rf = g.loc["ADF_error", "ratio"], g.loc["RF_clasico", "ratio"]
            mejor = r_adf < 1
            ax.axvspan(a, b, color=ACENTO if mejor else GRIS, alpha=0.16 if mejor else 0.12, lw=0)
            ax.text(a + (b - a) / 2, 0.97, f"ADF/HAR\n{r_adf:.2f}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, color=ACENTO if mejor else SOPORTE,
                    fontweight="bold" if mejor else "normal")
        for fecha, lab in EVENTOS.get(t, []):
            f = pd.Timestamp(fecha)
            if y.index.min() <= f <= y.index.max():
                ax.axvline(f, color=ALERTA, ls="--", lw=1.4, zorder=3)
                ax.text(f, 0.03, " " + lab, transform=ax.get_xaxis_transform(), color="#9A6200",
                        fontsize=8, va="bottom")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.30 * (hi - lo))
        ax.set_title(NOMBRES[t], loc="left", color=NAVY, fontweight="bold")
        ax.set_ylabel("log")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Bloques de prueba walk-forward: en azul-verde donde AdaptiveDriftForest supera al HAR",
                 x=0.01, ha="left", color=NAVY, fontweight="bold", fontsize=11, y=1.0)
    fig.tight_layout()
    guardar(fig, nombre)


def fig_fin_pesos(m, grupo, nombre):
    targets = [t for t in grupo if t in set(m.target)]
    fig, axes = plt.subplots(1, len(targets), figsize=(3.3 * len(targets), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        g = m[(m.target == t) & (m.protocol == "walk") & (m.model == "ADF_error")].sort_values("step")
        pesos_apilados(ax, [json.loads(w) for w in g.weights], NOMBRES[t].split(" (")[0],
                       [pd.Timestamp(d).strftime("%y") for d in g.test_from])
        ax.set_xlabel("año de inicio del bloque de prueba")
    axes[0].set_ylabel("peso en el ensamble")
    leyenda_pesos(fig, axes[len(axes) // 2])
    fig.suptitle("Finanzas: memoria que elige el modelo en cada reentrenamiento",
                 x=0.01, y=1.08, ha="left", color=NAVY, fontweight="bold", fontsize=11)
    guardar(fig, nombre)


def main():
    d = cargar_benchmark()
    fig_ratios(d)
    fig_pesos_sinteticos()
    if list(RES.glob("finance_results*.csv")):
        _, m = cargar_fin()
        fig_fin_resumen(m)
        fig_fin_series(m, GRUPO_A, "05_finanzas_series_y_bloques_A.png")
        fig_fin_series(m, GRUPO_B, "05_finanzas_series_y_bloques_B.png")
        fig_fin_pesos(m, GRUPO_A, "06_finanzas_pesos_A.png")
        fig_fin_pesos(m, GRUPO_B, "06_finanzas_pesos_B.png")
        if "HARX+ADF_resid" in set(m.model):
            fig_fin_resumen(m, ["HARX", "RF_clasico", "ADF_error", "HARX+RF_resid", "HARX+ADF_resid"],
                            "07_finanzas_hibridos_exploratorio.png",
                            "Exploratorio (9 series ya vistas): modelos híbridos HAR-X + bosque sobre residuos")
    if list(RES.glob("finance_holdout_results.csv")):
        _, mh = cargar_fin("finance_holdout_results.csv")
        fig_fin_resumen(mh, ["Persistencia", "HARX", "RF_clasico", "RF_ventana", "ADF_error",
                             "HARX+RF_resid", "HARX+ADF_resid"],
                        "08_finanzas_confirmacion.png",
                        "Confirmación (4 series nunca usadas): error relativo al HAR")


if __name__ == "__main__":
    main()
