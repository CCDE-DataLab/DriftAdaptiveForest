"""Tablas y cifras de la versión 2 del paper (main_v2.tex), leídas de experiments/results/*.csv.

No modifica las salidas de generar_tablas.py: escribe tablas/v2_*.tex, tablas/cifras_v2.tex (macros \\V...)
y figuras/v2_literatura.pdf. Copia a figuras/ las figuras de experiments/figuras que usa main_v2.tex.

Uso (desde la raíz del repo): venv_drift_forest/Scripts/python.exe -I paper/generar_tablas_v2.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

RES = ROOT / "experiments" / "results"
FIGS = ROOT / "experiments" / "figuras"
OUT = Path(__file__).resolve().parent / "tablas"
OUTF = Path(__file__).resolve().parent / "figuras"
MAC: list[str] = []


def geo(s) -> float:
    s = np.asarray(s, float)
    return float(np.exp(np.mean(np.log(s))))


def f2(x) -> str:
    return f"{x:.2f}"


def f3(x) -> str:
    return f"{x:.3f}"


def mac(nombre: str, valor) -> None:
    assert nombre.isalpha(), nombre
    MAC.append(f"\\newcommand{{\\V{nombre}}}{{{valor}}}")


def pval(p: float) -> str:
    """Para usar en modo matemático como $p\V...P$: devuelve '<0.001' o '=0.123'."""
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def signo(a, b):
    """Veces que a < b, total y p bilateral de la prueba de signo."""
    a, b = np.asarray(a), np.asarray(b)
    k, n = int((a < b).sum()), len(a)
    return k, n, binomtest(k, n).pvalue


def escribir(nombre: str, filas: list[str], cols: str, encabezado: str) -> None:
    if cols.startswith("l"):  # la primera columna absorbe el ancho sobrante (tabularx necesita una X)
        cols = "X" + cols[1:]
    txt = [f"\\begin{{tabularx}}{{\\textwidth}}{{{cols}}}", "\\toprule", encabezado + " \\\\", "\\midrule"]
    txt += [f if f == "\\midrule" else f + " \\\\" for f in filas]
    txt += ["\\bottomrule", "\\end{tabularx}"]
    (OUT / f"v2_{nombre}.tex").write_text("\n".join(txt) + "\n", encoding="utf-8")


def subrayar_min(vals: list[float], fmt=f2) -> list[str]:
    m = min(vals)
    return [("\\underline{" + fmt(v) + "}") if v == m else fmt(v) for v in vals]


# --------------------------------------------------------------------------------------- literatura
NOMBRE_DS = {
    "healthcare_none": "Hospital, sin drift", "healthcare_data": "Hospital, data", "healthcare_concept": "Hospital, concept",
    "healthcare_both": "Hospital, data + concept", "friedman_abrupt": "Friedman abrupto",
    "friedman_gradual_data": "Friedman gradual + data", "friedman_recurring": "Friedman recurrente",
    "covariate_shift": "Data fuerte", "highdim_mixed": "Alta dimensión mixto", "local_drift": "Concept local",
    "incremental": "Concept incremental", "air_quality": "Air Quality", "bike_sharing": "Bike Sharing",
    "gas_drift": "Gas Drift", "beijing_pm25": "Beijing PM2.5", "electricity": "Electricity",
}


def literatura():
    f = RES / "lit_benchmark.csv"
    if not f.exists():
        print("lit_benchmark.csv no existe todavía; se omite la tabla de literatura")
        return None
    lit = pd.read_csv(f)
    base = pd.concat([pd.read_csv(RES / n) for n in ("benchmark_synthetic.csv", "benchmark_real.csv",
                                                     "benchmark_synthetic_adf.csv", "benchmark_real_adf.csv")])
    base = base[base.model.isin(["RF_clasico", "RF_ventana", "ADF_error"])]
    key = ["dataset", "seed", "protocol", "step"]
    d = pd.concat([base[key + ["kind", "model", "rmse"]], lit[key + ["kind", "model", "rmse"]]])
    w = d.pivot_table(index=["kind"] + key, columns="model", values="rmse")
    mods = [("ARF", "ARF"), ("HAT", "HAT"), ("LightGBM", "LGBM"), ("RF+t", "RF+t"), ("RF_ventana", "RF-V"),
            ("ADF_error", "ADF-E")]
    res = {}
    for proto in ("walk", "static"):
        x = w.xs(proto, level="protocol")
        r = {m: (x[m] / x["RF_clasico"]).groupby(level=["kind", "dataset"]).apply(geo) for m, _ in mods}
        res[proto] = pd.DataFrame(r)
    filas = []
    t = res["walk"]
    for kind, titulo in (("sintetico", "Sintéticos"), ("real", "Reales")):
        tk = t.loc[kind]
        orden = [d_ for d_ in NOMBRE_DS if d_ in tk.index]
        filas.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{titulo}}}}}")
        for ds in orden:
            filas.append(NOMBRE_DS[ds] + " & " + " & ".join(subrayar_min([tk.loc[ds, m] for m, _ in mods])))
        g = [geo(tk[m]) for m, _ in mods]
        filas.append("\\midrule Media geométrica & " + " & ".join(subrayar_min(g)))
        ganados = [int((tk[m] < 1).sum()) for m, _ in mods]
        filas.append(f"Mejor que RF (de {len(tk)}) & " + " & ".join(str(v) for v in ganados))
        if kind == "sintetico":
            filas.append("\\midrule")
        for (m, corto), v in zip(mods, g):
            mac(f"lit{kind.capitalize()}{''.join(ch for ch in corto if ch.isalpha())}", f3(v))
        for (m, corto), v in zip(mods, ganados):
            mac(f"litGana{kind.capitalize()}{''.join(ch for ch in corto if ch.isalpha())}", v)
    escribir("literatura", filas, "l *{6}{c}", "Conjunto & ARF & HAT & LGBM & RF+t & RF-V & ADF-E")

    # Hipótesis H12-H13 (bitácora §16): ADF-E frente a ARF y LightGBM, walk sintético, por escenario
    ts = t.loc["sintetico"]
    for rival, nom in (("ARF", "ARF"), ("LightGBM", "LGBM")):
        rel = ts["ADF_error"] / ts[rival]
        mac(f"adfVs{nom}Geo", f3(geo(rel)))
        mac(f"adfVs{nom}Gana", int((rel < 1).sum()))
    tr = t.loc["real"]
    for rival, nom in (("ARF", "ARF"), ("LightGBM", "LGBM")):
        mac(f"adfVs{nom}GanaReal", int(((tr["ADF_error"] / tr[rival]) < 1).sum()))
    citadas = {"litHatIncr": ("sintetico", "incremental", "HAT"), "litArfCov": ("sintetico", "covariate_shift", "ARF"),
               "litLgbmCov": ("sintetico", "covariate_shift", "LightGBM"),
               "litLgbmHospData": ("sintetico", "healthcare_data", "LightGBM")}
    for nombre, (k, ds, mo) in citadas.items():
        mac(nombre, f2(t.loc[(k, ds), mo]))
    # tiempo de cómputo de River por conjunto (una pasada)
    secs = lit[lit.model.isin(["ARF", "HAT"])].groupby(["model", "kind"]).secs.mean()
    mac("arfSecsSint", f"{secs.loc[('ARF', 'sintetico')]:.1f}")
    figura_literatura(t)
    return t


def figura_literatura(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.graficar_analisis import ACENTO, AZUL_CLARO, GRIS, NAVY, SOPORTE
    estilo = {"ARF": ("#4F6D8A", "X", "ARF (River)"), "HAT": (AZUL_CLARO, "v", "HAT (River)"),
              "LightGBM": (SOPORTE, "P", "LightGBM"), "RF_ventana": (NAVY, "s", "RF-V"),
              "ADF_error": (ACENTO, "o", "ADF-E")}
    orden = [d for d in NOMBRE_DS if d in t.index.get_level_values(1)]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    y = np.arange(len(orden))[::-1]
    ax.axvline(1, color=GRIS, ls="--", lw=1)
    for m, (c, mk, lab) in estilo.items():
        vals = [t.xs(ds, level="dataset")[m].iloc[0] for ds in orden]
        ax.scatter(vals, y, color=c, marker=mk, s=55 if m == "ADF_error" else 34, label=lab,
                   edgecolor="white", linewidth=0.5, zorder=3 if m == "ADF_error" else 2)
    ax.axhline(len(orden) - 11 - 0.5, color=GRIS, lw=0.6)
    ax.set_yticks(y, [NOMBRE_DS[d] for d in orden])
    ax.set_xscale("log")
    tk = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    ax.set_xticks(tk, [str(v) for v in tk])
    ax.minorticks_off()
    ax.set_xlabel("RMSE / RMSE del RF convencional (walk-forward)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=5, frameon=False, fontsize=8)
    fig.savefig(OUTF / "v2_literatura.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------- matrícula
def matricula():
    from experiments.analizar_matricula import cargar
    r = cargar()
    r = r[r.control == "todas"][["h", "origen", "model", "rmse", "mae"]]
    extra = []
    for fname, mods in (("tiempo_matricula_results.csv", ["RF+t", "Part3+RF"]), ("lit_matricula.csv", ["ARF", "HAT"])):
        if (RES / fname).exists():
            e = pd.read_csv(RES / fname)
            extra.append(e[e.model.isin(mods)][["h", "origen", "model", "rmse", "mae"]])
    r = pd.concat([r] + extra)
    w = {met: r.pivot_table(index=["h", "origen"], columns="model", values=met) for met in ("rmse", "mae")}
    orden = [("Tendencia", "Tendencia (3 años)"), ("ETS", "ETS amortiguado"), ("OLS_global", "OLS global"),
             ("LightGBM", "LightGBM"), ("ARF", "ARF (River)"), ("HAT", "HAT (River)"), ("RF_clasico", "RF"),
             ("RF_ventana", "RF-V"), ("RF+t", "RF+t"), ("Part3+RF", "Part3+RF"), ("DR_recent_error", "DR-R"),
             ("ADF_error", "ADF-E")]
    orden = [o for o in orden if o[0] in w["rmse"].columns]
    vals = {m: [geo(w[met].loc[hh][m] / w[met].loc[hh]["Ingenuo"]) for met in ("rmse", "mae") for hh in (1, 2)]
            for m, _ in orden}
    cols_min = [min(vals[m][j] for m, _ in orden) for j in range(4)]
    filas = []
    for m, nom in orden:
        celdas = [("\\underline{" + f3(v) + "}") if v == cols_min[j] else f3(v) for j, v in enumerate(vals[m])]
        filas.append(nom + " & " + " & ".join(celdas))
    escribir("matricula", filas, "l *{4}{c}", "Modelo & RMSE $h=1$ & RMSE $h=2$ & MAE $h=1$ & MAE $h=2$")
    W = w["rmse"]
    for rival, nom in (("RF_clasico", "RF"), ("LightGBM", "LGBM"), ("ETS", "ETS"), ("ARF", "ARF")):
        if rival in W:
            k, n, p = signo(W["ADF_error"], W[rival])
            mac(f"matAdfVs{nom}", k)
            mac(f"mat{nom}VsAdf", int((W[rival] < W["ADF_error"]).sum()))
            mac(f"matAdfVs{nom}P", pval(p))
    mac("matN", len(W))
    mac("matEtsPeorIng", int((W["ETS"] > W["Ingenuo"]).sum()))
    mac("matAdfHdos", f3(vals["ADF_error"][1]))
    mac("matLgbmHuno", f3(vals["LightGBM"][0]))
    if "RF+t" in W:
        k, n, p = signo(W["RF+t"], W["RF_clasico"])
        mac("matRFtVsRF", k)
        mac("matRFtVsRFP", pval(p))
        k, n, p = signo(W["Part3+RF"], W["RF_clasico"])
        mac("matPartVsRF", k)
        mac("matPartVsRFP", pval(p))
        mac("matAdfVsRFt", signo(W["ADF_error"], W["RF+t"])[0])
    from experiments.analizar_matricula import cargar as cargar_sec
    rs = cargar_sec()
    dsec = rs.pivot_table(index=["h", "origen", "control"], columns="model", values="rmse")
    lr = np.log(dsec["ADF_error"] / dsec["RF_clasico"]).unstack("control")
    k, n, p = signo(lr["con fines de lucro"], lr["pública"])
    mac("matSector", k)
    mac("matSectorP", pval(p))
    yr = w["mae"].loc[1]
    mac("matEtsDosmilTrece", f2(yr.loc[2012, "ETS"] / yr.loc[2012, "Ingenuo"]))
    mac("matAdfCovid", f2(yr.loc[2019, "ADF_error"] / yr.loc[2019, "Ingenuo"]))
    mac("matLgbmCovid", f2(yr.loc[2019, "LightGBM"] / yr.loc[2019, "Ingenuo"]))


# --------------------------------------------------------------------------------------- finanzas
NOM_FIN = {"vix": "VIX (nivel)", "vix_ts": "VIX + estructura de plazos", "spx_rv": "S\\&P 500",
           "pen_rv": "PEN/USD", "svxy_rv": "SVXY", "jpy_rv": "USD/JPY", "brent_rv": "Brent",
           "tnx_rv": "Bono a 10 años", "btc_rv": "Bitcoin", "eur_rv": "EUR/USD", "gold_rv": "Oro",
           "ndx_rv": "Nasdaq-100", "rut_rv": "Russell 2000"}


def finanzas():
    from experiments.graficar_analisis import cargar_fin
    _, m1 = cargar_fin("finance_results*.csv")
    _, m2 = cargar_fin("finance_holdout_results.csv")
    mods = [("HARX", "HAR-X"), ("RF_clasico", "RF"), ("RF_ventana", "RF-V"), ("ADF_error", "ADF-E"),
            ("HARX+ADF_resid", "HAR-X+ADF")]
    filas = []
    for etiqueta, m in (("Series exploradas", m1), ("Series de confirmación", m2)):
        t = m[m.protocol == "walk"].groupby(["target", "model"]).ratio.apply(geo).unstack("model")
        filas.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{etiqueta}}}}}")
        for tg in [k for k in NOM_FIN if k in t.index]:
            filas.append(NOM_FIN[tg] + " & " + " & ".join(subrayar_min([t.loc[tg, c] for c, _ in mods])))
        filas.append("Media geométrica & " + " & ".join(subrayar_min([geo(t[c]) for c, _ in mods])))
        if etiqueta.startswith("Series exploradas"):
            filas.append("\\midrule")
    escribir("finanzas", filas, "l *{5}{c}", "Serie & HAR-X & RF & RF-V & ADF-E & HAR-X+ADF")
    w1 = m1[m1.protocol == "walk"].pivot_table(index=["target", "step"], columns="model", values="rmse")
    w2 = m2[m2.protocol == "walk"].pivot_table(index=["target", "step"], columns="model", values="rmse")
    mac("finN", len(w1))
    mac("finHarxVsAdf", signo(w1["HARX"], w1["ADF_error"])[0])
    mac("finHarxVsRF", signo(w1["HARX"], w1["RF_clasico"])[0])
    k, n, p = signo(w1["RF_ventana"], w1["RF_clasico"])
    mac("finRFVvsRFpierde", n - k)
    mac("finRFVvsRFP", pval(p))
    mac("finAdfVsRF", signo(w1["ADF_error"], w1["RF_clasico"])[0])
    mac("finConfN", len(w2))
    k, n, p = signo(w2["HARX+ADF_resid"], w2["HARX"])
    mac("finHuno", k)
    mac("finHunoP", pval(p))
    k, n, p = signo(w2["HARX+ADF_resid"], w2["HARX+RF_resid"])
    mac("finHdos", k)
    mac("finHdosP", pval(p))
    k, n, p = signo(w2["HARX"], w2["HAR"])
    mac("finHarxVsHarConf", k)
    mac("finHarxVsHarConfP", pval(p))


# --------------------------------------------------------------------------------------- macro
def macro():
    from experiments.analizar_macro import cargar
    m = cargar()
    t = m[m.protocol == "walk"].groupby(["objetivo", "h", "model"]).ratio.apply(geo).unstack("model")
    mods = [("DI", "DI"), ("ARX", "ARX"), ("RF_clasico", "RF"), ("RF_ventana", "RF-V"), ("ADF_error", "ADF-E"),
            ("DI+EPU", "DI+EPU")]
    nom = {"inflacion": "Inflación", "produccion": "Producción industrial", "desempleo": "Desempleo"}
    filas = []
    for o in nom:
        for hh in (1, 3, 6, 12):
            filas.append(f"{nom[o]}, $h={hh}$ & " + " & ".join(subrayar_min([t.loc[(o, hh), c] for c, _ in mods])))
    filas.append("\\midrule Media geométrica & " + " & ".join(subrayar_min([geo(t[c]) for c, _ in mods])))
    escribir("macro", filas, "l *{6}{c}", "Tarea & DI & ARX & RF & RF-V & ADF-E & DI+EPU")
    w = m[m.protocol == "walk"].pivot_table(index=["objetivo", "h", "step"], columns="model", values="rmse")
    k, n, p = signo(w["ADF_error"], w["RF_clasico"])
    mac("macN", n)
    mac("macAdfVsRF", k)
    mac("macAdfVsRFP", pval(p))
    sel = w.index.get_level_values(0).isin(["produccion", "desempleo"]) & w.index.get_level_values(1).isin([6, 12])
    k, n, p = signo(w.loc[sel, "DI+EPU"], w.loc[sel, "DI"])
    mac("macEpuDI", k)
    mac("macEpuN", n)
    mac("macEpuDIP", pval(p))
    k, n, p = signo(w["RF_ventana"], w["RF_clasico"])
    mac("macRFVpierde", n - k)
    mac("macRFVP", pval(p))
    mac("macRFgeo", f3(geo(t["RF_clasico"])))
    sin_ult = m[(m.protocol == "walk") & (m.objetivo == "desempleo") & (m.step < 5)]
    mac("macRFdesSinCovid", f2(geo(sin_ult[sin_ult.model == "RF_clasico"].ratio)))
    des = t.loc["desempleo"]
    mac("macRFdesMin", f2(des["RF_clasico"].min()))
    mac("macRFdesMax", f2(des["RF_clasico"].max()))


# --------------------------------------------------------------------------------------- tiempo y costos
def tiempo():
    t = pd.read_csv(RES / "tiempo_results.csv")
    adf = pd.concat([pd.read_csv(RES / n) for n in ("benchmark_synthetic_adf.csv", "benchmark_real_adf.csv")])
    key = ["dataset", "seed", "protocol", "step"]
    d = pd.concat([t[key + ["model", "rmse"]], adf[adf.model == "ADF_error"][key + ["model", "rmse"]]])
    d = d[d.dataset.isin(t.dataset.unique())]
    w = d[d.protocol == "walk"].pivot_table(index=key, columns="model", values="rmse")
    r = {mo: (w[mo] / w["RF_clasico"]).groupby(level="dataset").apply(geo) for mo in ["RF+t", "Part3+RF", "ADF_error"]}
    r = pd.DataFrame(r)
    filas = []
    for ds in [x for x in NOMBRE_DS if x in r.index]:
        filas.append(NOMBRE_DS[ds] + " & " + " & ".join(subrayar_min([r.loc[ds, c] for c in r.columns])))
    escribir("tiempo", filas, "l *{3}{c}", "Conjunto & RF+t & Part3+RF & ADF-E")
    for c, nom in (("RF+t", "RFt"), ("Part3+RF", "Part")):
        mac(f"tiempoCtrlMax{nom}", f2(max(r.loc["healthcare_none", c], r.loc["covariate_shift", c])))
    import json
    pt = t[(t.model == "Part3+RF") & (t.protocol == "walk")]["info"].map(json.loads)
    mac("tiempoCortesT", f"{np.mean([i['cortes_en_t'] for i in pt]):.1f}")
    mac("tiempoCortes", f"{np.mean([i['cortes'] for i in pt]):.0f}")
    x = w.xs("local_drift", level="dataset")
    k, n, p = signo(x["Part3+RF"], x["RF_clasico"])
    mac("tiempoHnueve", k)
    mac("tiempoHnueveN", n)
    mac("tiempoHnueveP", pval(p))
    mac("tiempoLocalPart", f2(r.loc["local_drift", "Part3+RF"]))
    mac("tiempoLocalRFt", f2(r.loc["local_drift", "RF+t"]))
    mac("tiempoLocalAdf", f2(r.loc["local_drift", "ADF_error"]))


def costos():
    c = pd.read_csv(RES / "costos_operativos.csv").set_index("modelo")
    nom = {"OLS_global": "OLS global", "LightGBM": "LightGBM", "RF_clasico": "RF", "DR_recent_error": "DR-R",
           "ADF_error": "ADF-E"}
    filas = []
    for m, n in nom.items():
        d = c.loc[m]
        filas.append(f"{n} & {d.fit_s:.2f} & {d.cpu_fit_s:.1f} & {d.pred_1fila_ms:.1f} & {d.ram_extra_mb:.0f} & "
                     f"{d.modelo_mb:.1f}")
    escribir("costos", filas, "l *{5}{r}",
             "Modelo & Ajuste (s) & CPU (s) & 1 fila (ms) & RAM extra (MB) & Modelo (MB)")
    mac("costoFilas", f"{int(c.loc['ADF_error', 'filas_train']):,}".replace(",", "\\,"))
    mac("costoAdfRam", f"{c.loc['ADF_error', 'ram_extra_mb'] / 1024:.1f}")
    mac("costoAdfVsLgbmFit", f"{c.loc['ADF_error', 'fit_s'] / c.loc['LightGBM', 'fit_s']:.0f}")
    mac("costoAdfVsLgbmRam", f"{c.loc['ADF_error', 'ram_extra_mb'] / c.loc['LightGBM', 'ram_extra_mb']:.0f}")


def copiar_figuras():
    for nombre in ("08_finanzas_confirmacion.png", "13_macro_vs_ar.png", "17_matricula_en_el_tiempo.png",
                   "09_epu_vs_vix.png", "06_finanzas_pesos_B.png"):
        shutil.copy(FIGS / nombre, OUTF / ("v2_" + nombre))


def main():
    OUT.mkdir(exist_ok=True)
    literatura()
    matricula()
    finanzas()
    macro()
    tiempo()
    costos()
    copiar_figuras()
    (OUT / "cifras_v2.tex").write_text("% Generado por paper/generar_tablas_v2.py\n" + "\n".join(MAC) + "\n",
                                       encoding="utf-8")
    print(f"{len(MAC)} macros escritas en tablas/cifras_v2.tex")


if __name__ == "__main__":
    main()
