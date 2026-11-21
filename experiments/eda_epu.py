"""Análisis exploratorio del índice de incertidumbre de política económica (EPU, Baker, Bloom y Davis).

Fuente: https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv -> data/real/epu/ (no se versiona).
El EPU diario se calcula con noticias del día, así que se rezaga 1 día para no usar información futura.

EXPLORATORIO: estos gráficos generan hipótesis; no son una prueba confirmatoria.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/eda_epu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from experiments.benchmark_drift import splits  # noqa: E402
from experiments.finance_data import load_all  # noqa: E402
from experiments.finance_experiment import H, build  # noqa: E402
from experiments.graficar_analisis import (ACENTO, ALERTA, AZUL_CLARO, GRIS, NAVY, NOMBRES,  # noqa: E402
                                           SOPORTE, cargar_fin, guardar)

EPU_FILE = ROOT / "data" / "real" / "epu" / "All_Daily_Policy_Data.csv"
TARGETS = ["vix", "spx_rv", "pen_rv", "svxy_rv", "jpy_rv", "brent_rv", "tnx_rv", "btc_rv",
           "eur_rv", "gold_rv", "ndx_rv", "rut_rv"]
EVENTOS = [("2001-09-11", "11-S"), ("2008-09-15", "Lehman"), ("2011-08-05", "rebaja EE. UU."),
           ("2016-06-24", "Brexit"), ("2020-03-16", "COVID-19"), ("2025-04-02", "aranceles")]


def load_epu():
    e = pd.read_csv(EPU_FILE)
    idx = pd.to_datetime(pd.DataFrame({"year": e.year, "month": e.month, "day": e.day}))
    s = pd.Series(e["daily_policy_index"].to_numpy(float), index=idx).sort_index()
    s = s[~s.index.duplicated(keep="last")].asfreq("D").ffill()
    le = np.log(s.clip(lower=1.0))
    f = pd.DataFrame({"l_epu5": le.rolling(5).mean(), "l_epu22": le.rolling(22).mean()}).shift(1)  # rezago 1 día
    f["d_epu"] = f["l_epu5"] - f["l_epu22"]  # sorpresa reciente de incertidumbre
    return s, f


def marcar_eventos(ax, lo, hi):
    for fecha, lab in EVENTOS:
        d = pd.Timestamp(fecha)
        if lo <= d <= hi:
            ax.axvline(d, color=ALERTA, ls="--", lw=1.1, zorder=1)
            ax.text(d, 0.97, " " + lab, transform=ax.get_xaxis_transform(), color="#9A6200",
                    fontsize=7.5, va="top", rotation=90)


# --------------------------------------------------------------------------- 09
def fig_epu_vs_vix(raw, epu_f):
    px = raw.loc[raw["vix"].notna() & raw["spx"].notna()]
    d = pd.DataFrame({"l_vix": np.log(px["vix"])}).join(epu_f, how="left").dropna()
    z = (d - d.mean()) / d.std()
    corr = d["l_epu22"].rolling(252).corr(d["l_vix"].rolling(22).mean())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(z.index, z["l_vix"].rolling(22).mean(), color=NAVY, lw=1, label="VIX (log, media 22 días)")
    a1.plot(z.index, z["l_epu22"], color=ACENTO, lw=1, label="EPU EE. UU. (log, media 22 días)")
    a1.set_ylabel("desvíos estándar")
    a1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    a1.set_title("Incertidumbre de política (EPU) y volatilidad esperada (VIX)", loc="left",
                 color=NAVY, fontweight="bold")
    a2.plot(corr.index, corr, color=SOPORTE, lw=1.1)
    a2.axhline(0, color=GRIS, lw=0.8)
    a2.fill_between(corr.index, 0, corr, color=AZUL_CLARO, alpha=0.35, lw=0)
    a2.set_ylabel("correlación\nmóvil 1 año")
    a2.set_ylim(-1, 1)
    a2.set_title("La relación entre ambas cambia con el tiempo", loc="left", color=NAVY, fontweight="bold")
    for ax in (a1, a2):
        marcar_eventos(ax, d.index.min(), d.index.max())
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    guardar(fig, "09_epu_vs_vix.png")
    return corr


# --------------------------------------------------------------------------- 10
def fig_lead_lag(raw, epu_f):
    lags = np.arange(-120, 121, 5)
    series = {"spx_rv": ("S&P 500", NAVY, "o"), "pen_rv": ("PEN/USD", ACENTO, "s"),
              "tnx_rv": ("Bono 10 años", AZUL_CLARO, "D"), "eur_rv": ("EUR/USD", GRIS, "^")}
    out = {}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for t, (lab, c, mk) in series.items():
        X, y, *_ = build(t, raw)
        k = t.split("_")[0]
        rv = X[f"lrv22_{k}"]
        e = epu_f["l_epu22"].reindex(rv.index)
        cc = [e.corr(rv.shift(-h)) for h in lags]
        out[lab] = (lags[int(np.nanargmax(cc))], float(np.nanmax(cc)))
        ax.plot(lags, cc, color=c, marker=mk, ms=3.5, lw=1.2, label=lab)
    ax.axvline(0, color=ALERTA, ls="--", lw=1.2)
    ax.set_xlabel("desfase h en días hábiles  (h > 0: el EPU de hoy contra la volatilidad dentro de h días)")
    ax.set_ylabel("correlación")
    ax.set_title("¿La incertidumbre anticipa la volatilidad o la sigue?", loc="left", color=NAVY, fontweight="bold")
    ax.legend(ncol=4, loc="lower center")
    guardar(fig, "10_epu_adelanto_atraso.png")
    return out


# --------------------------------------------------------------------------- 11
def oos_har_x_epu(raw, epu_f):
    """Walk-forward lineal: HAR-X contra HAR-X + EPU, mismo protocolo y embargo que finance_experiment."""
    rows = []
    for t in TARGETS:
        X, y, *_ = build(t, raw)
        Xe = X.join(epu_f[["l_epu5", "l_epu22", "d_epu"]], how="left")
        ok = Xe.notna().all(axis=1)
        X0, X1, yy = X[ok].to_numpy(float), Xe[ok].to_numpy(float), y[ok].to_numpy(float)
        for proto in ("walk", "static"):
            for step, a, b in splits(len(yy), proto):
                e0 = yy[a:b] - LinearRegression().fit(X0[:a - H], yy[:a - H]).predict(X0[a:b])
                e1 = yy[a:b] - LinearRegression().fit(X1[:a - H], yy[:a - H]).predict(X1[a:b])
                rows.append({"target": t, "protocol": proto, "step": step,
                             "rmse_harx": np.sqrt(np.mean(e0 ** 2)), "rmse_epu": np.sqrt(np.mean(e1 ** 2))})
    r = pd.DataFrame(rows)
    r["ratio"] = r["rmse_epu"] / r["rmse_harx"]
    return r


def fig_oos(r):
    g = (r.groupby(["target", "protocol"]).ratio.apply(lambda s: float(np.exp(np.log(s).mean())))
         .unstack("protocol").reindex(TARGETS))
    wins = (r[r.protocol == "walk"].assign(w=lambda d: d.ratio < 1).groupby("target").w.sum().reindex(TARGETS))
    y = np.arange(len(TARGETS))[::-1]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axvline(1.0, color=ALERTA, ls="--", lw=1.3)
    colores = [ACENTO if v < 1 else GRIS for v in g["walk"]]
    ax.barh(y, g["walk"] - 1, left=1, color=colores, height=0.55, label="walk-forward")
    ax.scatter(g["static"], y, color=NAVY, marker="D", s=26, zorder=3, label="split 80/20")
    for yi, t in zip(y, TARGETS):
        ax.text(max(g.loc[t, "walk"], g.loc[t, "static"]) + 0.002, yi, f"{int(wins[t])}/6",
                va="center", fontsize=8, color=SOPORTE)
    ax.set_yticks(y, [NOMBRES[t].replace(" (5 días)", "") for t in TARGETS])
    ax.set_xlabel("RMSE de HAR-X + EPU / RMSE de HAR-X  (menos de 1 = el EPU ayuda)   ·   n/6 = bloques donde ayuda")
    ax.set_title("¿Agrega información el EPU fuera de muestra? (modelos lineales, exploratorio)",
                 loc="left", color=NAVY, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.grid(axis="y", visible=False)
    guardar(fig, "11_epu_fuera_de_muestra.png")
    return g, wins


# --------------------------------------------------------------------------- 12
def fig_bloques(epu_f):
    _, m1 = cargar_fin("finance_results*.csv")
    _, m2 = cargar_fin("finance_holdout_results.csv")
    m = pd.concat([m1, m2])
    w = m[m.protocol == "walk"].pivot_table(index=["target", "step"], columns="model", values="rmse")
    fechas = m[(m.protocol == "walk") & (m.model == "HAR")].set_index(["target", "step"])[["test_from", "test_to"]]
    w = w.join(fechas)
    e = epu_f["l_epu22"]
    w["epu"] = [e.loc[pd.Timestamp(a):pd.Timestamp(b)].mean() for a, b in zip(w.test_from, w.test_to)]
    w["adf_rf"] = np.log(w["ADF_error"] / w["RF_clasico"])
    w["vent_rf"] = np.log(w["RF_ventana"] / w["RF_clasico"])
    w = w[w.index.get_level_values(0) != "vix_ts"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    res = {}
    for ax, col, tit, c, mk in ((axes[0], "adf_rf", "AdaptiveDriftForest / RF clásico", ACENTO, "o"),
                                (axes[1], "vent_rf", "RF ventana reciente / RF clásico", NAVY, "s")):
        ax.axhline(0, color=ALERTA, ls="--", lw=1.2)
        ax.scatter(w["epu"], w[col], color=c, marker=mk, s=22, alpha=0.8, edgecolor="white", linewidth=0.4)
        b = np.polyfit(w["epu"], w[col], 1)
        xs = np.linspace(w["epu"].min(), w["epu"].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=SOPORTE, lw=1.4)
        rho = w[["epu", col]].corr().iloc[0, 1]
        res[col] = (rho, b[0])
        ax.set_title(f"{tit}\ncorrelación = {rho:.2f}", loc="left", color=NAVY, fontweight="bold")
        ax.set_xlabel("EPU medio del bloque de prueba (log)")
    axes[0].set_ylabel("log del cociente de RMSE\n(bajo 0 = mejor que el RF clásico)")
    fig.suptitle("¿Rinde mejor la adaptación cuando la incertidumbre es alta? (72 bloques, 12 series)",
                 x=0.01, ha="left", color=NAVY, fontweight="bold", fontsize=11, y=1.03)
    guardar(fig, "12_epu_bloques.png")
    return res, w


def main():
    raw = load_all()
    epu, epu_f = load_epu()
    print(f"EPU diario: {len(epu)} días, {epu.index.min():%Y-%m-%d} -> {epu.index.max():%Y-%m-%d}")
    corr = fig_epu_vs_vix(raw, epu_f)
    print("correlación móvil EPU-VIX: min %.2f max %.2f mediana %.2f" % (corr.min(), corr.max(), corr.median()))
    print("lead-lag (h del máximo, corr):", fig_lead_lag(raw, epu_f))
    r = oos_har_x_epu(raw, epu_f)
    r.to_csv(ROOT / "experiments" / "results" / "epu_oos_lineal.csv", index=False)
    g, wins = fig_oos(r)
    print(pd.concat([g.round(4), wins.rename("bloques_walk")], axis=1).to_string())
    k = int((r[r.protocol == "walk"].ratio < 1).sum())
    print(f"total walk: EPU ayuda en {k}/{(r.protocol == 'walk').sum()} bloques")
    res, _ = fig_bloques(epu_f)
    print("bloques (corr, pendiente):", {k: (round(v[0], 3), round(v[1], 3)) for k, v in res.items()})


if __name__ == "__main__":
    main()
