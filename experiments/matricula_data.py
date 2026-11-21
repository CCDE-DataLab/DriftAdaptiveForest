"""Panel de matrícula de otoño por institución (IPEDS, NCES, EE. UU.), 2002-2023.

Archivos en data/real/matricula/ipeds/ (no se versionan), descargados de
https://nces.ed.gov/ipeds/datacenter/data/EF{año}A.zip y HD2023.zip.

Total de estudiantes: EFALEVEL = 1 ("todos los estudiantes").
  2008-2023: columna EFTOTLT.   2002-2007: columna EFRACE24 (total hombres + mujeres;
  verificado contra 2008: correlación 0.984 por institución).
Se prefiere la versión revisada (*_rv.csv) cuando existe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parents[1] / "data" / "real" / "matricula" / "ipeds"
AÑOS = range(2002, 2024)
CONTROL = {1: "pública", 2: "privada sin fines de lucro", 3: "privada con fines de lucro"}
NIVEL = {1: "4 años o más", 2: "2 años", 3: "menos de 2 años"}


def _archivo(y: int) -> Path:
    for nombre in (f"ef{y}a_rv.csv", f"ef{y}a_RV.csv", f"ef{y}a.csv"):
        if (DIR / nombre).exists():
            return DIR / nombre
    raise FileNotFoundError(y)


def cargar_matricula() -> pd.DataFrame:
    """Matriz institución × año con la matrícula total (NaN si no reportó)."""
    cols = {}
    for y in AÑOS:
        d = pd.read_csv(_archivo(y), encoding="latin-1", low_memory=False)
        d.columns = d.columns.str.upper().str.strip().str.replace("﻿", "")
        d = d[d["EFALEVEL"] == 1]
        total = d["EFTOTLT"] if "EFTOTLT" in d.columns else d["EFRACE24"]
        cols[y] = pd.Series(pd.to_numeric(total, errors="coerce").to_numpy(), index=d["UNITID"].to_numpy())
    return pd.DataFrame(cols)


def cargar_directorio() -> pd.DataFrame:
    h = pd.read_csv(DIR / "HD2023.csv", encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    h.columns = h.columns.str.upper().str.strip().str.replace("﻿", "")
    h = h.set_index("UNITID")[["INSTNM", "STABBR", "CONTROL", "ICLEVEL", "SECTOR"]]
    return h


def panel_balanceado(min_alumnos: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Instituciones con matrícula >= min_alumnos en los 22 años (sesgo de supervivencia: ver BITACORA)."""
    m = cargar_matricula()
    m = m[(m.notna().all(axis=1)) & (m.min(axis=1) >= min_alumnos)]
    h = cargar_directorio()
    m = m[m.index.isin(h.index)]
    h = h.loc[m.index]
    h = h[h["CONTROL"].isin(CONTROL) & h["ICLEVEL"].isin(NIVEL)]
    return m.loc[h.index], h


if __name__ == "__main__":
    m, h = panel_balanceado()
    print(f"Instituciones en panel balanceado: {len(m)}  años: {m.columns.min()}-{m.columns.max()}")
    print(h["CONTROL"].map(CONTROL).value_counts().to_string())
    tot = m.groupby(h["CONTROL"].map(CONTROL)).sum().T
    print((tot / 1e6).round(2).iloc[::3].to_string())
