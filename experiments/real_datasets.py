"""Loaders de datasets reales de regresión con drift documentado.

Los archivos se descargan en ``data/real/<nombre>/`` (ver README de experimentos).
Cada loader devuelve (X, y) ordenados cronológicamente, X numérico.

  air_quality   UCI 360. CO(GT) a partir de 5 sensores químicos + clima (2004-2005).
                Drift de sensor conocido (De Vito et al., 2008): concept drift.
  bike_sharing  UCI 275. Alquileres por hora 2011-2012. Se excluye `yr`, así el crecimiento
                del sistema actúa como contexto oculto (concept drift) + estacionalidad (data drift).
  gas_drift     UCI 270. 128 features de 16 sensores de gas en 36 meses (10 batches),
                objetivo log(concentración). Drift de sensores muy fuerte (data + concept).
  beijing_pm25  UCI 381. PM2.5 horario 2010-2014 desde clima. Estacional + cambio de emisiones.
                Submuestreado 1 de cada 3 filas para que el MMD quepa en memoria.
  electricity   Elec2 (Harries 1999, OpenML 151). Demanda NSW 1996-1998 cada 30 min.
                Benchmark clásico de concept drift. Submuestreado 1 de cada 3 filas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REAL_DIR = Path(__file__).resolve().parents[1] / "data" / "real"


def air_quality(root: Path = REAL_DIR):
    df = pd.read_csv(root / "air_quality" / "AirQualityUCI.csv", sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df = df.replace(-200, np.nan)
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S")
    df = df.assign(ts=ts, hour=ts.dt.hour, weekday=ts.dt.weekday).sort_values("ts")
    feats = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)",
             "T", "RH", "AH", "hour", "weekday"]
    df = df.dropna(subset=feats + ["CO(GT)"]).reset_index(drop=True)
    return df[feats].astype(float), df["CO(GT)"].astype(float)


def bike_sharing(root: Path = REAL_DIR):
    df = pd.read_csv(root / "bike_sharing" / "hour.csv").sort_values("instant").reset_index(drop=True)
    feats = ["season", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit",
             "temp", "atemp", "hum", "windspeed"]
    return df[feats].astype(float), df["cnt"].astype(float)


def gas_drift(root: Path = REAL_DIR):
    rows = []
    for b in range(1, 11):
        with open(root / "gas_drift" / f"batch{b}.dat", encoding="ascii") as fh:
            for line in fh:
                parts = line.split()
                if not parts:
                    continue
                gas, conc = parts[0].split(";")
                feats = np.zeros(128)
                for tok in parts[1:]:
                    k, v = tok.split(":")
                    feats[int(k) - 1] = float(v)
                rows.append((int(gas), float(conc), b, feats))
    gas = np.array([r[0] for r in rows])
    conc = np.array([r[1] for r in rows])
    F = np.vstack([r[3] for r in rows])
    X = pd.DataFrame(F, columns=[f"s{j}" for j in range(128)])
    for g in range(1, 7):
        X[f"gas_{g}"] = (gas == g).astype(float)
    y = pd.Series(np.log1p(conc), name="log_conc")
    return X, y


def beijing_pm25(root: Path = REAL_DIR, step: int = 3):
    df = pd.read_csv(root / "beijing_pm25" / "PRSA_data_2010.1.1-2014.12.31.csv").sort_values("No")
    df = df.dropna(subset=["pm2.5"]).reset_index(drop=True).iloc[::step].reset_index(drop=True)
    X = df[["month", "hour", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]].astype(float)
    for c in ["NE", "NW", "SE", "cv"]:
        X[f"cbwd_{c}"] = (df["cbwd"] == c).astype(float)
    return X, df["pm2.5"].astype(float)


def electricity(root: Path = REAL_DIR, step: int = 3):
    cols = ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "class"]
    df = pd.read_csv(root / "electricity" / "electricity.arff", comment="@", header=None, names=cols)
    df = df.iloc[::step].reset_index(drop=True)  # ya viene en orden cronológico
    X = df[["day", "period", "nswprice", "vicprice", "vicdemand", "transfer"]].astype(float)
    return X, df["nswdemand"].astype(float)


DATASETS = {
    "air_quality": (air_quality, "sensor drift (concept)"),
    "bike_sharing": (bike_sharing, "contexto oculto + estacional"),
    "gas_drift": (gas_drift, "sensor drift fuerte (data + concept)"),
    "beijing_pm25": (beijing_pm25, "estacional + emisiones"),
    "electricity": (electricity, "concept drift de mercado"),
}
