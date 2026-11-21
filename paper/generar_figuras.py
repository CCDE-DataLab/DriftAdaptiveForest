"""Genera las figuras del paper (paper/figuras/*.pdf) a partir de los datos y resultados del repo.

Uso (desde la raíz del repo):
  venv_drift_forest/Scripts/python.exe paper/generar_figuras.py

Paleta categórica validada (dataviz, modo claro): azul, naranja, aqua, amarillo, en ese orden fijo.
Cada serie lleva además una forma de marcador distinta y leyenda, para no depender solo del color.
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

from experiments.summarize_benchmark import load, ratios, table  # noqa: E402

OUT = Path(__file__).resolve().parent / "figuras"
RES = ROOT / "experiments" / "results"

# Paleta oficial CCDE. Reglas: 70-80 % del gráfico en azules/neutros; ACENTO para la serie
# destacada (el modelo propuesto, lo reciente); ALERTA solo para quiebres, cambios de régimen y énfasis.
# NAVY y SOPORTE casi no se distinguen entre sí (validador: ΔE 9): nunca van como categorías vecinas.
NAVY = "#0A2540"        # azul principal: títulos, texto, serie base
SOPORTE = "#1E3D59"     # azul de soporte: serie secundaria, ejes
ACENTO = "#17A2B8"      # acento tecnológico: serie destacada
SUAVE = "#E2F0F9"       # fondo suave: bandas, cuadrícula
ALERTA = "#F5A623"      # alerta/destacado: quiebres, umbrales
AZUL_CLARO = "#9DBCD4"  # derivado claro de la familia azul (4.ª categoría en barras apiladas)
GRIS = "#7A8FA3"        # gris azulado para líneas de referencia
TINTA, TINTA2, GRILLA = NAVY, SOPORTE, SUAVE

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.edgecolor": GRIS,
    "axes.labelcolor": TINTA,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": TINTA2,
    "ytick.color": TINTA2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def guardar(fig, nombre):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / nombre)
    import os
    if os.environ.get("PREVIEW_DIR"):  # vista previa PNG opcional para revisar las figuras
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / nombre.replace(".pdf", ".png"), dpi=110)
    plt.close(fig)
    print(f"  figuras/{nombre}")


# ---------------------------------------------------------------- Figura 1: qué es el drift
def fig_motivacion():
    df = pd.read_parquet(ROOT / "data" / "raw" / "healthcare_both_drift.parquet")
    df = df.sort_values("fecha_ingreso").reset_index(drop=True)
    n = len(df)
    t = np.arange(n) / n

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.5), gridspec_kw={"wspace": 0.32})

    # (a) data drift: media móvil de la severidad al ingreso
    roll = df["severidad_ingreso"].rolling(250, center=True, min_periods=50).mean()
    a.plot(t, df["severidad_ingreso"], ".", ms=1.2, color=SOPORTE, alpha=0.12, rasterized=True, zorder=1)
    a.plot(t, roll, color=NAVY, lw=2, zorder=2)
    a.axvspan(0, 0.4, color=SUAVE, alpha=0.9, lw=0, zorder=0)
    a.axvspan(0.8, 1.0, color=ACENTO, alpha=0.12, lw=0, zorder=0)
    a.set_xlabel("Tiempo normalizado $t$")
    a.set_ylabel("Severidad al ingreso")
    a.set_title("(a) Data drift: cambia $P(X)$", loc="left")
    a.text(0.02, 0.97, "media móvil", transform=a.transAxes, va="top", color=TINTA2, fontsize=8)

    # (b) concept drift: misma severidad, distinta estancia
    bins = np.arange(1, 10.5, 1.0)
    for mask, color, marker, etiqueta in (
        (t < 0.4, SOPORTE, "o", "Primer 40 % (antiguo)"),
        (t >= 0.8, ACENTO, "s", "Último 20 % (nuevo)"),
    ):
        sub = df[mask]
        cat = pd.cut(sub["severidad_ingreso"], bins)
        g = sub.groupby(cat, observed=True)["dias_estancia"].agg(["mean", "count"])
        g = g[g["count"] >= 15]
        centros = np.array([iv.mid for iv in g.index])
        b.plot(centros, g["mean"], color=color, lw=2, marker=marker, ms=5,
               markeredgecolor="white", markeredgewidth=0.8, label=etiqueta)
    b.set_xlabel("Severidad al ingreso")
    b.set_ylabel("Días de estancia (media)")
    b.set_title("(b) Concept drift: cambia $P(Y\\mid X)$", loc="left")
    b.legend(loc="upper left", handlelength=1.6, fontsize=7.8, title="Protocolo", title_fontsize=7.8)
    for ax in (a, b):
        ax.grid(axis="y", color=GRILLA, lw=0.6)
        ax.set_axisbelow(True)
    guardar(fig, "motivacion_drift.pdf")


# ---------------------------------------------------------------- Figura 2: candidatos
def fig_candidatos():
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    n_tr = 0.85
    filas = []
    for f in (1.00, 0.60, 0.35, 0.20, 0.10):
        filas.append((f"Ventana {int(f * 100)} %", n_tr - f * n_tr, n_tr, ACENTO))
    bordes = np.linspace(0, n_tr, 6)
    for k in range(5):
        filas.append((f"Bloque {k + 1}", bordes[k], bordes[k + 1], SOPORTE))
    for i, (nombre, s, e, color) in enumerate(filas):
        y = len(filas) - i
        ax.barh(y, e - s, left=s, height=0.62, color=color, edgecolor="white", linewidth=1.5)
        ax.text(-0.015, y, nombre, ha="right", va="center", fontsize=8, color=TINTA)
    ax.axvspan(n_tr, 1.0, color=SUAVE, alpha=1.0, lw=0)
    ax.axvline(n_tr, color=ALERTA, lw=1.2, ls=(0, (3, 2)))
    ax.text((n_tr + 1) / 2, len(filas) - 1, "Cola\nreciente\n(15 %)", ha="center", va="center",
            fontsize=8, color=TINTA)
    ax.annotate("", xy=(1.0, 0.25), xytext=(n_tr, 0.25),
                arrowprops=dict(arrowstyle="<->", color=TINTA2, lw=0.8))
    ax.text((n_tr + 1) / 2, 0.45, "valida pesos;\nluego se reentrena\ncon ella", ha="center", va="bottom",
            fontsize=7, color=TINTA2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.1, len(filas) + 0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["pasado", "", "", "", "presente"])
    ax.set_xlabel("Histórico de entrenamiento (ordenado en el tiempo)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ACENTO, label="Ventanas recientes anidadas (ADF)"),
                       Patch(color=SOPORTE, label="Bloques contiguos (DR y ADF)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    guardar(fig, "candidatos.pdf")


# ---------------------------------------------------------------- Figura 3: resultados por dataset
def fig_resultados():
    df, _ = load()
    d = ratios(df)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.9), gridspec_kw={"width_ratios": [11, 5.6], "wspace": 0.75})
    series = (
        ("RF_ventana", "RF ventana 20 %", NAVY, "o"),
        ("DR_best", "DriftRegressor (mejor modo)", ALERTA, "s"),
        ("ADF_error", "AdaptiveDriftForest", ACENTO, "D"),
    )
    for ax, kind, titulo in ((axes[0], "sintetico", "Sintéticos"), (axes[1], "real", "Reales")):
        ratio, _ = table(d, "walk", kind)
        ratio = ratio.assign(DR_best=ratio[["DR_mmd", "DR_recent_error", "DR_hybrid"]].min(axis=1))
        ratio = ratio.sort_values("ADF_error", ascending=False)
        y = np.arange(len(ratio))
        for i in y:
            vals = ratio.iloc[i][[s[0] for s in series]]
            ax.plot([vals.min(), vals.max()], [i, i], color=GRILLA, lw=1.4, zorder=1)
        for j, (col, etiqueta, color, marker) in enumerate(series):
            ax.scatter(ratio[col], y + (j - 1) * 0.0, s=34, color=color, marker=marker, zorder=3,
                       edgecolors="white", linewidths=0.8, label=etiqueta)
        ax.axvline(1, color=GRIS, lw=0.9, ls=(0, (3, 2)), zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([s.replace("_", " ") for s in ratio.index], fontsize=8)
        ax.set_xscale("log")
        ticks = [0.5, 0.7, 1.0, 1.3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{v:g}" for v in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.45, 1.4)
        ax.set_title(titulo, loc="left")
        ax.grid(axis="x", color=GRILLA, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_xlabel("RMSE / RMSE del RF convencional (log)")
    axes[1].set_xlabel("RMSE / RMSE del RF (log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    guardar(fig, "resultados_por_dataset.pdf")


# ---------------------------------------------------------------- Figura 4: pesos del ADF
GRUPOS = (
    ("Ventanas cortas (10–20 %)", ACENTO),
    ("Ventanas medias (35–60 %)", SOPORTE),
    ("Histórico completo (100 %)", AZUL_CLARO),
    ("Bloques antiguos", ALERTA),
)


def _grupo(nombre):
    if nombre.startswith("bloque"):
        return GRUPOS[3][0]
    f = float(nombre.split("_")[1])
    if f >= 1.0:
        return GRUPOS[2][0]
    return GRUPOS[0][0] if f <= 0.2 else GRUPOS[1][0]


def fig_pesos():
    sint = pd.read_csv(RES / "benchmark_synthetic_adf.csv")
    real = pd.read_csv(RES / "benchmark_real_adf.csv")
    casos = (
        (sint, "healthcare_concept", "Concept drift en t ≈ 0.6"),
        (sint, "healthcare_none", "Sin drift (control)"),
        (real, "gas_drift", "Gas Drift (datos reales)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (dfc, ds, titulo) in zip(axes, casos):
        sub = dfc[(dfc.dataset == ds) & (dfc.model == "ADF_error") & (dfc.protocol == "walk") & (dfc.seed == 0)]
        sub = sub.sort_values("step")
        mat = np.zeros((len(sub), len(GRUPOS)))
        for i, w in enumerate(sub["weights"]):
            for nombre, peso in json.loads(w).items():
                mat[i, [g[0] for g in GRUPOS].index(_grupo(nombre))] += peso
        x = np.arange(len(sub))
        base = np.zeros(len(sub))
        for j, (nombre, color) in enumerate(GRUPOS):
            ax.bar(x, mat[:, j], bottom=base, width=0.72, color=color, edgecolor="white", linewidth=1.0,
                   label=nombre)
            base += mat[:, j]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{40 + 10 * s}" for s in sub["step"]], fontsize=7.5)
        ax.set_title(titulo, loc="left", fontsize=8.5)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Peso en el ensamble")
    axes[1].set_xlabel("Origen del paso walk-forward (% de la muestra)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=7.8,
               handlelength=1.2, columnspacing=1.0)
    guardar(fig, "pesos_adf.pdf")


# ---------------------------------------------------------------- Figura 5: galería de escenarios
# (nombre, título, zonas de cambio verdaderas del generador como [(inicio, fin)], explicación breve)
GALERIA = (
    ("healthcare_none", "Sin drift (control)", [], "Nada cambia: el pasado sigue siendo válido."),
    ("healthcare_data", "Solo data drift", [], "La regla no cambia; el error sube donde X sale de lo visto."),
    ("covariate_shift", "Data drift fuerte (2σ)", [], "X sale del rango visto: hay que extrapolar."),
    ("friedman_abrupt", "Concept drift abrupto", [(0.6, 0.6)], "La regla cambia de golpe en t = 0.6."),
    ("healthcare_concept", "Concept drift gradual", [(0.5, 0.7)], "Transición sigmoide entre dos protocolos."),
    ("friedman_gradual_data", "Data + concept gradual", [(0.35, 0.75)], "X y la regla cambian de forma progresiva."),
    ("friedman_recurring", "Concept drift recurrente", [(0.25, 0.5), (0.75, 1.0)], "A → B → A → B: el régimen antiguo vuelve."),
    ("local_drift", "Concept drift local", [(0.55, 1.0)], "Solo cambia una región de X (x0 > 0.6)."),
)


def _indicadores(X, y, seed=0, ventana=400):
    """Desplazamiento de X (en σ del tramo inicial) y error relativo de un modelo 'antiguo'."""
    from sklearn.ensemble import RandomForestRegressor

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(X)
    a = int(0.3 * n)
    mu, sd = X[:a].mean(0), X[:a].std(0) + 1e-12
    Z = (X - mu) / sd
    desplaz = pd.DataFrame(Z).rolling(ventana, center=True, min_periods=ventana // 4).mean().abs().max(axis=1)

    idx = np.arange(n)
    train = idx[(idx < a) & (idx % 2 == 0)]  # filas pares del 30 % inicial
    m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=-1)
    m.fit(X[train], y[train])
    err2 = (m.predict(X) - y) ** 2
    err2[train] = np.nan  # solo filas que el modelo no vio
    rmse = np.sqrt(pd.Series(err2).rolling(ventana, center=True, min_periods=ventana // 4).mean())
    base = np.sqrt(np.nanmean(err2[:a]))  # error fuera de muestra dentro del tramo inicial
    return idx / n, desplaz.to_numpy(), (rmse / base).to_numpy()


def fig_galeria():
    from experiments.synthetic_scenarios import SCENARIOS

    fig = plt.figure(figsize=(6.5, 8.9))
    outer = fig.add_gridspec(4, 2, hspace=0.8, wspace=0.28)
    for k, (ds, titulo, zonas, texto) in enumerate(GALERIA):
        X, y = SCENARIOS[ds][0](0)
        t, desp, err = _indicadores(X, y)
        inner = outer[k // 2, k % 2].subgridspec(2, 1, hspace=0.12)
        ax1 = fig.add_subplot(inner[0])
        ax2 = fig.add_subplot(inner[1], sharex=ax1)
        for ax in (ax1, ax2):
            for s, e in zonas:
                if s == e:
                    ax.axvline(s, color=ALERTA, lw=1.3, ls=(0, (3, 2)))
                else:
                    ax.axvspan(s, e, color=ALERTA, alpha=0.16, lw=0)
            ax.axvspan(0, 0.3, color=SUAVE, alpha=1.0, lw=0)
            ax.grid(axis="y", color=GRILLA, lw=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
        ax1.plot(t, desp, color=ACENTO, lw=1.6)
        ax1.set_ylim(0, max(2.6, np.nanmax(desp) * 1.1))
        ax1.set_ylabel("Δ X (σ)", fontsize=7.5)
        ax1.tick_params(labelbottom=False)
        ax2.plot(t, err, color=NAVY, lw=1.6)
        ax2.axhline(1, color=GRIS, lw=0.8, ls=(0, (2, 2)))
        ax2.set_ylim(0.6, max(2.5, np.nanmax(err) * 1.08))
        ax2.set_ylabel("Error\nrelativo", fontsize=7.5)
        ax2.set_xlim(0, 1)
        ax2.set_xticks([0, 0.3, 0.6, 1.0])
        if k >= 6:
            ax2.set_xlabel("Tiempo normalizado $t$", fontsize=8)
        ax1.set_title(f"({chr(97 + k)}) {titulo}", loc="left", fontsize=8.5, pad=13)
        ax1.text(0, 1.04, texto, transform=ax1.transAxes, ha="left", va="bottom", fontsize=6.8,
                 color=TINTA2)
    guardar(fig, "galeria_escenarios.pdf")


def main():
    print("Escribiendo figuras en", OUT)
    fig_galeria()
    fig_motivacion()
    fig_candidatos()
    fig_resultados()
    fig_pesos()


if __name__ == "__main__":
    main()
