"""Panel macro mensual tipo FRED-MD descargado con la API de FRED + EPU mensual oficial.

La clave se lee de la variable de entorno FRED_API_KEY (nunca se guarda en el repo).
Cache en data/real/fred_md/ y data/real/epu/ (no se versionan).

Transformaciones (código tcode de McCracken y Ng, 2016):
  1 nivel · 2 primera diferencia · 4 log · 5 Δlog · 6 Δ²log

Uso: FRED_API_KEY=... venv_drift_forest/Scripts/python.exe -I experiments/macro_data.py
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FRED_DIR = ROOT / "data" / "real" / "fred_md"
EPU_XLSX = ROOT / "data" / "real" / "epu" / "US_Policy_Uncertainty_Data.xlsx"

# serie FRED -> (tcode, grupo)
SERIES = {
    "INDPRO": (5, "producción"), "CUMFNS": (2, "producción"), "W875RX1": (5, "producción"),
    "PAYEMS": (5, "empleo"), "UNRATE": (2, "empleo"), "CE16OV": (5, "empleo"), "UEMPMEAN": (2, "empleo"),
    "AWHMAN": (1, "empleo"), "CES0600000008": (6, "empleo"),
    "HOUST": (4, "vivienda"), "PERMIT": (4, "vivienda"),
    "DPCERA3M086SBEA": (5, "consumo"),
    "M2SL": (6, "dinero"), "BUSLOANS": (6, "crédito"), "NONREVSL": (6, "crédito"),
    "FEDFUNDS": (2, "tasas"), "TB3MS": (2, "tasas"), "GS10": (2, "tasas"), "AAA": (2, "tasas"), "BAA": (2, "tasas"),
    "CPIAUCSL": (6, "precios"), "CPILFESL": (6, "precios"), "PCEPI": (6, "precios"), "PPIACO": (6, "precios"),
    "WTISPLC": (6, "precios"),
}


def fetch_fred(series_id: str, key: str) -> pd.Series:
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
           f"&api_key={key}&file_type=json&frequency=m&aggregation_method=avg")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                timeout=60) as r:
        obs = json.loads(r.read())["observations"]
    s = pd.Series({pd.Timestamp(o["date"]): (np.nan if o["value"] == "." else float(o["value"])) for o in obs})
    return s.rename(series_id).dropna()


def load_raw(refresh: bool = False) -> pd.DataFrame:
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("FRED_API_KEY")
    cols = {}
    for sid in SERIES:
        f = FRED_DIR / f"{sid}.csv"
        if f.exists() and not refresh:
            cols[sid] = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
            continue
        if not key:
            raise RuntimeError("Falta FRED_API_KEY para descargar " + sid)
        s = fetch_fred(sid, key)
        s.to_csv(f)
        cols[sid] = s
    return pd.DataFrame(cols).sort_index()


def load_epu_monthly() -> pd.Series:
    e = pd.read_excel(EPU_XLSX, sheet_name="Main News Index")
    e = e[pd.to_numeric(e["Month"], errors="coerce").notna()]
    idx = pd.to_datetime(pd.DataFrame({"year": e["Year"].astype(int), "month": e["Month"].astype(int), "day": 1}))
    return pd.Series(e["News_Based_Policy_Uncert_Index"].to_numpy(float), index=idx, name="EPU").sort_index()


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for sid, (tc, _) in SERIES.items():
        x = raw[sid]
        if tc == 1:
            out[sid] = x
        elif tc == 2:
            out[sid] = x.diff()
        elif tc == 4:
            out[sid] = np.log(x)
        elif tc == 5:
            out[sid] = np.log(x).diff()
        elif tc == 6:
            out[sid] = np.log(x).diff().diff()
    t = pd.DataFrame(out)
    t["SPREAD_10Y_FF"] = raw["GS10"] - raw["FEDFUNDS"]
    t["SPREAD_BAA_10Y"] = raw["BAA"] - raw["GS10"]
    return t


if __name__ == "__main__":
    raw = load_raw(refresh=True)
    for c in raw.columns:
        s = raw[c].dropna()
        print(f"{c:16s} {len(s):5d} meses  {s.index.min():%Y-%m} -> {s.index.max():%Y-%m}")
    epu = load_epu_monthly()
    print(f"EPU              {len(epu):5d} meses  {epu.index.min():%Y-%m} -> {epu.index.max():%Y-%m}")
