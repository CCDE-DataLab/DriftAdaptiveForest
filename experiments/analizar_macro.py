"""Evalúa las hipótesis registradas (BITACORA §12) y genera las figuras 13-15 del experimento macro.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/analizar_macro.py
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

from experiments.graficar_analisis import (ACENTO, ALERTA, AZUL_CLARO, GRIS, NAVY, SOPORTE,  # noqa: E402
                                           guardar)

RES = ROOT / "experiments" / "results"
NOMBRE = {"inflacion": "Inflación", "produccion": "Producción industrial", "desempleo": "Desempleo"}
ESTILO = {
    "DI": (NAVY, "s", "Índice de difusión (Stock-Watson)"),
    "ARX": (GRIS, "^", "ARX (lineal, todas las variables)"),
    "RF_clasico": (AZUL_CLARO, "D", "RF clásico"),
    "RF_ventana": (SOPORTE, "v", "RF ventana reciente"),
    "ADF_error": (ACENTO, "o", "AdaptiveDriftForest"),
    "DI+EPU": ("#4F6D8A", "X", "Índice de difusión + EPU"),
}


def cargar():
    f = pd.read_csv(RES / "macro_results.csv")
    m = f.groupby(["objetivo", "h", "protocol", "step", "model"]).agg(
        rmse=("rmse", "mean"), test_from=("test_from", "first"), test_to=("test_to", "first")).reset_index()
    ar = m[m.model == "AR"].set_index(["objetivo", "h", "protocol", "step"])["rmse"].rename("ar")
    m = m.join(ar, on=["objetivo", "h", "protocol", "step"])
    m["ratio"] = m["rmse"] / m["ar"]
    return m


def geo(s):
    return float(np.exp(np.mean(np.log(s))))


def signo(w, a, b, filtro=None):
    d = w if filtro is None else w[filtro]
    k, n = int((d[a] < d[b]).sum()), len(d)
    return k, n, binomtest(k, n).pvalue


def fig_resumen(m):
    modelos = ["DI", "ARX", "RF_clasico", "RF_ventana", "ADF_error", "DI+EPU"]
    t = m[m.protocol == "walk"].groupby(["objetivo", "h", "model"]).ratio.apply(geo).unstack("model")
    filas = [(o, h) for o in NOMBRE for h in (1, 3, 6, 12)]
    y = np.arange(len(filas))[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    ax.axvline(1.0, color=ALERTA, ls="--", lw=1.3)
    for mod in modelos:
        c, mk, lab = ESTILO[mod]
        ax.scatter([t.loc[f, mod] for f in filas], y, color=c, marker=mk, s=60 if mod == "ADF_error" else 36,
                   label=lab, zorder=3 if mod == "ADF_error" else 2, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, [f"{NOMBRE[o]} · h={h}" for o, h in filas])
    for k in (4, 8):
        ax.axhline(len(filas) - k - 0.5, color=GRIS, lw=0.6)
    ax.set_xscale("log")
    tk = [0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
    ax.set_xticks(tk, [str(v) for v in tk])
    ax.minorticks_off()
    ax.set_xlabel("RMSE / RMSE del AR (menos de 1 = mejor que el AR)  ·  walk-forward, 6 bloques 1987–2025")
    ax.set_title("Macro mensual (FRED): error relativo al modelo AR", loc="left", color=NAVY, fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    guardar(fig, "13_macro_vs_ar.png")
    return t


def fig_tiempo(m):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, o in zip(axes, NOMBRE):
        d = m[(m.protocol == "walk") & (m.objetivo == o) & (m.h == 12)]
        for mod in ("DI", "RF_clasico", "ADF_error"):
            g = d[d.model == mod].sort_values("step")
            c, mk, lab = ESTILO[mod]
            ax.plot(g.test_from.str[:4], g.ratio, color=c, marker=mk, lw=1.4, ms=6, label=lab)
        ax.axhline(1, color=ALERTA, ls="--", lw=1.2)
        ax.set_title(f"{NOMBRE[o]} a 12 meses", loc="left", color=NAVY, fontweight="bold")
        ax.set_xlabel("año de inicio del bloque de prueba")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("RMSE / RMSE del AR")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    fig.suptitle("¿En qué periodos ganan o pierden los modelos? (la línea naranja es el AR)",
                 x=0.01, ha="left", color=NAVY, fontweight="bold", fontsize=11, y=1.04)
    guardar(fig, "14_macro_en_el_tiempo.png")


def fig_epu(m):
    w = m[m.protocol == "walk"].pivot_table(index=["objetivo", "h", "step"], columns="model", values="rmse")
    pares = [("AR+EPU", "AR", NAVY, "s"), ("DI+EPU", "DI", ACENTO, "o"), ("ARX+EPU", "ARX", GRIS, "^"),
             ("ADF_error+EPU", "ADF_error", AZUL_CLARO, "D")]
    filas = [(o, h) for o in NOMBRE for h in (1, 3, 6, 12)]
    y = np.arange(len(filas))[::-1]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axvline(1.0, color=ALERTA, ls="--", lw=1.3)
    for con, sin, c, mk in pares:
        r = (w[con] / w[sin]).groupby(level=[0, 1]).apply(geo)
        ax.scatter([r.loc[f] for f in filas], y, color=c, marker=mk, s=40, label=f"{con} / {sin}",
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(y, [f"{NOMBRE[o]} · h={h}" for o, h in filas])
    for k in (4, 8):
        ax.axhline(len(filas) - k - 0.5, color=GRIS, lw=0.6)
    ax.set_xlabel("RMSE con EPU / RMSE sin EPU  (menos de 1 = el EPU ayuda)")
    ax.set_title("¿Aporta el EPU al pronóstico macro? (walk-forward)", loc="left", color=NAVY, fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=4)
    guardar(fig, "15_macro_epu.png")


def main():
    m = cargar()
    t = fig_resumen(m)
    print("Walk-forward, RMSE / AR (media geométrica de 6 bloques):")
    cols = ["DI", "DI+EPU", "ARX", "RF_clasico", "RF_ventana", "DR_recent_error", "ADF_error", "ADF_error+EPU"]
    print(t[cols].round(3).to_string())
    print("\nMedia geométrica por modelo:", t.apply(lambda c: geo(c)).round(3).sort_values().to_dict())
    fig_tiempo(m)
    fig_epu(m)
    w = m[m.protocol == "walk"].pivot_table(index=["objetivo", "h", "step"], columns="model", values="rmse")
    print("\nHipótesis registradas:")
    k, n, p = signo(w, "ADF_error", "RF_clasico")
    print(f"  H3  ADF_error < RF_clasico: {k}/{n}  p = {p:.4f}")
    sel = w.index.get_level_values(0).isin(["produccion", "desempleo"]) & w.index.get_level_values(1).isin([6, 12])
    for a, b in (("DI+EPU", "DI"), ("AR+EPU", "AR")):
        k, n, p = signo(w, a, b, sel)
        print(f"  H4  {a} < {b} (producción y desempleo, h=6 y 12): {k}/{n}  p = {p:.4f}")
    print("\nOtras comparaciones (no registradas):")
    for a, b in (("DI", "AR"), ("ADF_error", "DI"), ("RF_clasico", "DI"), ("RF_ventana", "RF_clasico"),
                 ("DI+EPU", "DI"), ("ADF_error+EPU", "ADF_error")):
        k, n, p = signo(w, a, b)
        print(f"      {a} < {b}: {k}/{n}  p = {p:.4f}")


if __name__ == "__main__":
    main()
