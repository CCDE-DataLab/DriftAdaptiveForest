"""Genera las tablas LaTeX del paper a partir de experiments/results/*.csv.

Ninguna cifra de las tablas se escribe a mano: todas salen de los CSV del benchmark.
Reutiliza las funciones de experiments/summarize_benchmark.py (mismas definiciones de
ratio, datasets ganados, rank medio y prueba de Wilcoxon que summary.md).

Uso (desde la raíz del repo):
  venv_drift_forest/Scripts/python.exe paper/generar_tablas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.summarize_benchmark import RES, load, overall, ratios, table  # noqa: E402

OUT = Path(__file__).resolve().parent / "tablas"

# Modelos que aparecen en las tablas por dataset, en este orden, con su encabezado corto.
MODELOS = [
    ("RF_ventana", "RF-V"),
    ("RF_pesos_exp", "RF-E"),
    ("DR_mmd", "DR-M"),
    ("DR_recent_error", "DR-R"),
    ("DR_hybrid", "DR-H"),
    ("ADF_error", "ADF-E"),
    ("ADF_hybrid", "ADF-H"),
]
NOMBRE = dict(MODELOS) | {"RF_clasico": "RF", "DR_recent_refit*": "DR-R+"}

ETIQUETA_DRIFT = {
    "healthcare_none": "Ninguno (control)",
    "healthcare_data": "Data",
    "healthcare_concept": "Concept gradual",
    "healthcare_both": "Data + concept",
    "friedman_abrupt": "Concept abrupto",
    "friedman_gradual_data": "Data + concept gradual",
    "friedman_recurring": "Concept recurrente",
    "covariate_shift": "Data fuerte (2$\\sigma$)",
    "highdim_mixed": "Data + concept mixto",
    "local_drift": "Concept local",
    "incremental": "Concept incremental",
    "air_quality": "Deriva de sensores",
    "bike_sharing": "Contexto oculto + estacional",
    "gas_drift": "Deriva de sensores fuerte",
    "beijing_pm25": "Estacional + emisiones",
    "electricity": "Mercado eléctrico",
}


def tt(name: str) -> str:
    return "\\texttt{" + name.replace("_", "\\_") + "}"


def f2(v: float) -> str:
    return f"{v:.2f}"


def write(name: str, body: str) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text("% Generado por paper/generar_tablas.py. No editar a mano.\n" + body, encoding="utf-8")
    print(f"  tablas/{name}")


# ------------------------------------------------------------------ tabla de datasets
def tabla_datasets(diags: pd.DataFrame) -> None:
    filas = []
    for kind, titulo in (("sintetico", "Sintéticos (3 semillas cada uno)"), ("real", "Reales")):
        filas.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{titulo}}}}} \\\\")
        for r in diags[diags.kind == kind].itertuples():
            filas.append(
                f"{tt(r.dataset)} & {ETIQUETA_DRIFT[r.dataset]} & {r.n} & {r.p} & "
                f"{r.data_drift_auc:.2f} & {r.concept_gain:.2f} & {r.recency_ratio:.2f} \\\\"
            )
        if kind == "sintetico":
            filas.append("\\addlinespace")
    body = (
        "\\begin{tabularx}{\\textwidth}{l X r r r r r}\n\\toprule\n"
        "Dataset & Tipo de drift & {$n$} & {$p$} & {AUC} & {CG} & {RR} \\\\\n\\midrule\n"
        + "\n".join(filas)
        + "\n\\bottomrule\n\\end{tabularx}\n"
    )
    write("datasets.tex", body)


# ------------------------------------------------------------------ ratios por dataset
def tabla_ratios(d: pd.DataFrame, kind: str, protocol: str, archivo: str) -> None:
    ratio, _ = table(d, protocol, kind)
    cols = [m for m, _ in MODELOS]
    filas = []
    for ds in ratio.index:
        vals = ratio.loc[ds, cols]
        best = vals.min()
        celdas = []
        for m in cols:
            v = vals[m]
            s = f2(v)
            celdas.append(f"\\underline{{{s}}}" if v == best and v < 1 else s)
        filas.append(f"{tt(ds)} & " + " & ".join(celdas) + " \\\\")
    # media geométrica sobre datasets
    geo = [float(np.exp(np.log(ratio[m]).mean())) for m in cols]
    filas.append("\\midrule")
    filas.append("Media geométrica & " + " & ".join(f"{v:.3f}" for v in geo) + " \\\\")
    head = " & ".join("{" + c + "}" for _, c in MODELOS)
    body = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l " + "r" * len(cols) + "}\n\\toprule\n"
        f"Dataset & {head} \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular*}\n"
    )
    write(archivo, body)


# ------------------------------------------------------------------ agregado (4 cortes)
def tabla_agregado(d: pd.DataFrame) -> None:
    paneles = [
        ("sintetico", "walk", "Panel A. Sintéticos, walk-forward (11 escenarios $\\times$ 3 semillas $\\times$ 6 pasos)"),
        ("sintetico", "static", "Panel B. Sintéticos, partición 80/20 (11 escenarios $\\times$ 3 semillas)"),
        ("real", "walk", "Panel C. Reales, walk-forward (5 datasets $\\times$ 6 pasos)"),
        ("real", "static", "Panel D. Reales, partición 80/20 (5 datasets)"),
    ]
    filas = []
    for i, (kind, protocol, titulo) in enumerate(paneles):
        _, out = overall(d, protocol, kind)
        n_ds = d[(d.kind == kind)].dataset.nunique()
        if i:
            filas.append("\\addlinespace")
        filas.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{titulo}}}}} \\\\")
        for r in out.itertuples():
            nombre = NOMBRE.get(r.modelo, r.modelo)
            if r.modelo == "RF_clasico":
                filas.append(f"{nombre} (referencia) & 1.000 & {{--}} & {{--}} & {r.rank_medio:.2f} & {{--}} \\\\")
            else:
                filas.append(
                    f"{nombre} & {r.ratio_medio:.3f} & {int(r.datasets_ganados)}/{n_ds} & "
                    f"{100 * r.win_rate_pasos:.0f} & {r.rank_medio:.2f} & {r.p_wilcoxon:.3f} \\\\"
                )
    body = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r c r r r}\n\\toprule\n"
        "Modelo & {Ratio RMSE} & {Datasets $<1$} & {\\% pasos $<1$} & {Rank medio} & {$p$ Wilcoxon} \\\\\n"
        "\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular*}\n"
    )
    write("agregado.tex", body)


# ------------------------------------------------------------------ Optuna
def tabla_optuna() -> None:
    t = pd.read_csv(RES / "tuned_static_all.csv")
    piv = t.pivot(index="dataset", columns="model", values="rmse")
    params = t[t.model == "DR_opt"].set_index("dataset")["best_params"].map(json.loads)
    orden = list(dict.fromkeys(t.dataset))
    filas = []
    for ds in orden:
        rf, rv, dr = piv.loc[ds, ["RF_clasico_opt", "RF_ventana_opt", "DR_opt"]]
        best = min(rf, rv, dr)
        p = params[ds]

        def fmt(v):
            s = f"{v:.4f}" if v < 0.2 else (f"{v:.3f}" if v < 10 else f"{v:.2f}")
            return f"\\underline{{{s}}}" if v == best else s

        filas.append(
            f"{tt(ds)} & {fmt(rf)} & {fmt(rv)} & {fmt(dr)} & ${100 * (dr / rf - 1):+.1f}$ & "
            f"{tt(p['weighting_mode'])} & {p['n_blocks']} & {p['beta']:.1f} \\\\"
        )
    body = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r r r r l r r}\n\\toprule\n"
        " & \\multicolumn{3}{c}{RMSE en test} & & \\multicolumn{3}{c}{DR elegido por Optuna} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){6-8}\n"
        "Dataset & {RF} & {RF-V} & {DR} & {DR vs RF (\\%)} & Modo & {$K$} & {$\\beta$} \\\\\n\\midrule\n"
        + "\n".join(filas)
        + "\n\\bottomrule\n\\end{tabular*}\n"
    )
    write("optuna.tex", body)


# ------------------------------------------------------------------ cifras sueltas usadas en el texto
def cifras(d: pd.DataFrame) -> None:
    """Escribe macros \\newcommand con las cifras citadas en el texto, para no copiarlas a mano."""
    macros = {}
    for kind, k in (("sintetico", "Sint"), ("real", "Real")):
        for protocol, p in (("walk", "Walk"), ("static", "Static")):
            _, out = overall(d, protocol, kind)
            ratio, _ = table(d, protocol, kind)
            o = out.set_index("modelo")
            for m, tag in (("ADF_error", "ADFe"), ("ADF_hybrid", "ADFh"), ("RF_ventana", "RFV"),
                           ("RF_pesos_exp", "RFE"), ("DR_mmd", "DRm"), ("DR_recent_error", "DRr"),
                           ("DR_hybrid", "DRh"), ("DR_recent_refit*", "DRrefit")):
                macros[f"r{tag}{k}{p}"] = f"{o.loc[m, 'ratio_medio']:.3f}"
                macros[f"g{tag}{k}{p}"] = f"{int(o.loc[m, 'datasets_ganados'])}"
                macros[f"p{tag}{k}{p}"] = f"{o.loc[m, 'p_wilcoxon']:.3f}"
                macros[f"w{tag}{k}{p}"] = f"{100 * o.loc[m, 'win_rate_pasos']:.0f}"
                macros[f"k{tag}{k}{p}"] = f"{o.loc[m, 'rank_medio']:.2f}"
            macros[f"kRF{k}{p}"] = f"{o.loc['RF_clasico', 'rank_medio']:.2f}"
            for ds in ratio.index:
                dsn = "".join(w.capitalize() for w in ds.split("_")).replace("Pm25", "Pm")
                for m, tag in (("ADF_error", "ADFe"), ("RF_ventana", "RFV"), ("DR_hybrid", "DRh"),
                               ("DR_mmd", "DRm"), ("DR_recent_error", "DRr")):
                    macros[f"x{tag}{dsn}{p}"] = f2(ratio.loc[ds, m])
    # extremos del DR original sin concept drift (peor caso, walk-forward sintético)
    ratio_w, _ = table(d, "walk", "sintetico")
    dr_cols = ["DR_mmd", "DR_recent_error", "DR_hybrid"]
    peor = ratio_w.loc[["healthcare_data", "covariate_shift"], dr_cols].to_numpy().max()
    macros["peorDRsinConcept"] = f"{100 * (peor - 1):.0f}"
    # Subconjuntos para contrastar hipótesis (walk-forward sintético)
    concept = ["healthcare_concept", "healthcare_both", "friedman_abrupt", "friedman_gradual_data",
               "friedman_recurring", "highdim_mixed", "local_drift", "incremental"]
    solo_data = ["healthcare_data", "covariate_shift"]
    rc = ratio_w.loc[concept]
    macros["nConcept"] = str(len(concept))
    macros["hUnoGeo"] = f"{float(np.exp(np.log(rc['DR_recent_error']).mean())):.3f}"
    macros["hUnoGana"] = str(int((rc["DR_recent_error"] < 1).sum()))
    rd = ratio_w.loc[solo_data, "DR_mmd"]
    macros["hDosMin"], macros["hDosMax"] = f2(rd.min()), f2(rd.max())
    mejor_dr = ratio_w[dr_cols].min(axis=1)
    macros["hTresRFV"] = str(int((ratio_w["RF_ventana"] < mejor_dr).sum()))
    macros["hTresRFE"] = str(int((ratio_w["RF_pesos_exp"] < mejor_dr).sum()))
    macros["hTresN"] = str(len(ratio_w))
    macros["hCuatroConceptGeo"] = f"{float(np.exp(np.log(rc['ADF_error']).mean())):.3f}"
    sin_concept = ["healthcare_none", "healthcare_data", "covariate_shift"]
    macros["hCuatroSinMax"] = f2(ratio_w.loc[sin_concept, "ADF_error"].max())
    macros["hCuatroVsRFV"] = str(int((ratio_w["ADF_error"] < ratio_w["RF_ventana"]).sum()))

    ratio_s, _ = table(d, "static", "real")
    g = ratio_s.loc["gas_drift", dr_cols]
    macros["gasDRmin"], macros["gasDRmax"] = f2(g.min()), f2(g.max())

    # Optuna
    t = pd.read_csv(RES / "tuned_static_all.csv")
    piv = t.pivot(index="dataset", columns="model", values="rmse")
    dr_vs_rf = piv["DR_opt"] / piv["RF_clasico_opt"] - 1
    macros["optDRganaRF"] = str(int((dr_vs_rf < 0).sum()))
    macros["optRFVganaDR"] = str(int((piv["RF_ventana_opt"] < piv["DR_opt"]).sum()))
    macros["optN"] = str(len(piv))
    macros["optBothRF"] = f"{piv.loc['healthcare_both', 'RF_clasico_opt']:.2f}"
    macros["optBothRFV"] = f"{piv.loc['healthcare_both', 'RF_ventana_opt']:.2f}"
    macros["optBothDR"] = f"{piv.loc['healthcare_both', 'DR_opt']:.2f}"
    macros["optBothDRpct"] = f"{-100 * dr_vs_rf['healthcare_both']:.1f}"
    macros["optBothRFVpct"] = f"{-100 * (piv.loc['healthcare_both', 'RF_ventana_opt'] / piv.loc['healthcare_both', 'RF_clasico_opt'] - 1):.1f}"
    macros["optGasDRpct"] = f"{100 * dr_vs_rf['gas_drift']:.1f}"

    lineas = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    write("cifras.tex", "\n".join(lineas) + "\n")


def main() -> None:
    df, diags = load()
    d = ratios(df)
    print("Escribiendo tablas en", OUT)
    tabla_datasets(diags)
    tabla_ratios(d, "sintetico", "walk", "sint_walk.tex")
    tabla_ratios(d, "sintetico", "static", "sint_static.tex")
    tabla_ratios(d, "real", "walk", "real_walk.tex")
    tabla_ratios(d, "real", "static", "real_static.tex")
    tabla_agregado(d)
    tabla_optuna()
    cifras(d)


if __name__ == "__main__":
    main()
