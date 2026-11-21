"""Resume los CSV de benchmark_drift.py en tablas markdown (experiments/results/summary.md).

ratio  = media geométrica de RMSE_modelo / RMSE_RF_clasico sobre (semilla, paso). <1 = mejor que RF.
wins   = % de (semilla, paso) donde el modelo tiene menor RMSE que RF_clasico.
rank   = ranking medio por dataset (1 = mejor RMSE medio).
Wilcoxon: test pareado sobre el log-ratio medio por dataset (una observación por dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

RES = ROOT / "experiments" / "results"
REF = "RF_clasico"


def load(holdout: bool = False):
    """holdout=False: todo salvo las corridas *_holdout (semillas 3-5); True: solo esas."""
    files = [p for p in sorted(RES.glob("benchmark_*.csv")) if ("holdout" in p.name) == holdout]
    frames = [pd.read_csv(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    diags = pd.concat([pd.read_csv(p) for p in sorted(RES.glob("diagnostics_*.csv"))], ignore_index=True)
    return df, diags


def ratios(df):
    key = ["dataset", "seed", "protocol", "step"]
    ref = df[df.model == REF].set_index(key)["rmse"].rename("rmse_ref")
    d = df.join(ref, on=key)
    d["log_ratio"] = np.log(d["rmse"] / d["rmse_ref"])
    d["win"] = (d["rmse"] < d["rmse_ref"]).astype(float)
    return d


def table(d, protocol, kind):
    sub = d[(d.protocol == protocol) & (d.kind == kind)]
    g = sub.groupby(["dataset", "model"]).agg(ratio=("log_ratio", lambda s: float(np.exp(s.mean()))),
                                              rmse=("rmse", "mean"))
    ratio = g["ratio"].unstack("model")
    rank = g["rmse"].unstack("model").rank(axis=1)
    return ratio, rank


def fmt_ratio_table(ratio, diags, kind):
    models = [m for m in ratio.columns if m != REF]
    order = list(ratio.index)
    dg = diags.set_index("dataset")
    lines = ["| Dataset | Drift | AUC data drift | concept gain | " + " | ".join(models) + " |",
             "|---|---|---:|---:|" + "---:|" * len(models)]
    for ds in order:
        best = ratio.loc[ds, models].min()
        cells = []
        for m in models:
            v = ratio.loc[ds, m]
            s = f"{v:.2f}"
            cells.append(f"**{s}**" if v == best and v < 1 else s)
        lines.append(f"| {ds} | {dg.loc[ds, 'drift']} | {dg.loc[ds, 'data_drift_auc']:.2f} | "
                     f"{dg.loc[ds, 'concept_gain']:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def overall(d, protocol, kind):
    sub = d[(d.protocol == protocol) & (d.kind == kind)]
    per_ds = sub.groupby(["dataset", "model"])["log_ratio"].mean().unstack("model")
    _, rank = table(d, protocol, kind)
    rows = []
    for m in per_ds.columns:
        if m == REF:
            rows.append((m, 1.0, np.nan, np.nan, rank[m].mean(), np.nan))
            continue
        vals = per_ds[m].values
        try:
            p = wilcoxon(vals).pvalue
        except ValueError:
            p = np.nan
        wins_ds = int((vals < 0).sum())
        win_rate = sub[sub.model == m]["win"].mean()
        rows.append((m, float(np.exp(vals.mean())), wins_ds, win_rate, rank[m].mean(), p))
    out = pd.DataFrame(rows, columns=["modelo", "ratio_medio", "datasets_ganados", "win_rate_pasos",
                                      "rank_medio", "p_wilcoxon"]).sort_values("rank_medio")
    n_ds = per_ds.shape[0]
    lines = [f"| Modelo | RMSE / RF (geo) | Datasets mejor que RF (de {n_ds}) | % pasos mejor que RF | "
             "Rank medio | p Wilcoxon |", "|---|---:|---:|---:|---:|---:|"]
    for r in out.itertuples():
        if r.modelo == REF:
            lines.append(f"| {r.modelo} | 1.00 | — | — | {r.rank_medio:.2f} | — |")
        else:
            lines.append(f"| {r.modelo} | {r.ratio_medio:.3f} | {int(r.datasets_ganados)} | {100 * r.win_rate_pasos:.0f} % "
                         f"| {r.rank_medio:.2f} | {r.p_wilcoxon:.3f} |")
    return "\n".join(lines), out


def main():
    holdout = "--holdout" in sys.argv
    df, diags = load(holdout)
    d = ratios(df)
    parts = ["# Resumen benchmark DriftRegressor vs Random Forest\n",
             "Valores = RMSE del modelo / RMSE del RF clásico (media geométrica). < 1 es mejor que el RF clásico. "
             "En negrita, el mejor modelo cuando supera al RF.\n"]
    for kind in ("sintetico", "real"):
        if kind not in set(d.kind):
            continue
        for protocol, title in (("walk", "Walk-forward (6 reentrenamientos)"),
                                ("static", "Split estático 80/20 (protocolo README)")):
            ratio, _ = table(d, protocol, kind)
            ov, _ = overall(d, protocol, kind)
            parts.append(f"\n## {kind.capitalize()} — {title}\n")
            parts.append(fmt_ratio_table(ratio, diags, kind))
            parts.append("\n\n**Agregado:**\n")
            parts.append(ov)
            parts.append("\n")
    parts.append("\n## Diagnóstico de drift\n")
    dg = diags.round(3).astype(str)
    parts.append("| " + " | ".join(dg.columns) + " |")
    parts.append("|" + "---|" * len(dg.columns))
    parts.extend("| " + " | ".join(r) + " |" for r in dg.values.tolist())
    out = RES / ("summary_holdout.md" if holdout else "summary.md")
    out.write_text("\n".join(parts), encoding="utf-8")
    print("\n".join(parts))


if __name__ == "__main__":
    main()
