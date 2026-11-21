"""Descarga y cachea series financieras públicas en data/real/finanzas/ (no se versiona).

Fuentes (sin API key):
  Yahoo Finance (endpoint chart v8): ^VIX, ^GSPC, HG=F (cobre), DX-Y.NYB (dólar global), SVXY (ETF corto en VIX),
  BTC-USD, USDJPY=X, BZ=F (Brent), ^TNX (bono 10 años), ^VIX3M; validación: EURUSD=X, GC=F, ^NDX, ^RUT.
  BCRPData (BCR del Perú): PD04638PD = tipo de cambio interbancario venta (S/ por US$), diario.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/finance_data.py
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FIN_DIR = Path(__file__).resolve().parents[1] / "data" / "real" / "finanzas"
UA = {"User-Agent": "Mozilla/5.0"}
START = "2004-01-01"

YAHOO = {"vix": "^VIX", "spx": "^GSPC", "copper": "HG=F", "dxy": "DX-Y.NYB", "svxy": "SVXY",
         "btc": "BTC-USD", "jpy": "USDJPY=X", "brent": "BZ=F", "tnx": "^TNX", "vix3m": "^VIX3M",
         "eur": "EURUSD=X", "gold": "GC=F", "ndx": "^NDX", "rut": "^RUT"}
BCRP = {"pen": "PD04638PD"}
MESES = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6, "Jul": 7, "Ago": 8,
         "Set": 9, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_yahoo(symbol: str) -> pd.Series:
    p1 = int(pd.Timestamp(START).timestamp())
    p2 = int(time.time())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}"
           f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplits")
    res = json.loads(_get(url))["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
    ind = res["indicators"]
    close = (ind["adjclose"][0]["adjclose"] if "adjclose" in ind else ind["quote"][0]["close"])
    s = pd.Series(close, index=ts, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def fetch_bcrp(code: str) -> pd.Series:
    url = (f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{code}/json/"
           f"{pd.Timestamp(START).year}-1-1/{pd.Timestamp.today():%Y-%m-%d}".replace("-0", "-"))
    data = json.loads(_get(url))
    idx, vals = [], []
    for p in data["periods"]:
        d, m, y = p["name"].split(".")
        yy = int(y)
        idx.append(pd.Timestamp(year=2000 + yy if yy < 70 else 1900 + yy, month=MESES[m], day=int(d)))
        v = p["values"][0]
        vals.append(np.nan if v in ("n.d.", "") else float(v))
    return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float).dropna()


def load_all(refresh: bool = False) -> pd.DataFrame:
    FIN_DIR.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, sym in {**YAHOO, **BCRP}.items():
        f = FIN_DIR / f"{name}.csv"
        if f.exists() and not refresh:
            s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            s = fetch_yahoo(sym) if name in YAHOO else fetch_bcrp(sym)
            s.rename(name).to_csv(f)
        cols[name] = s
        print(f"{name:7s} {len(s):6d} filas  {s.index.min():%Y-%m-%d} -> {s.index.max():%Y-%m-%d}")
    return pd.DataFrame(cols).sort_index()


if __name__ == "__main__":
    df = load_all(refresh=True)
    print(df.tail())
