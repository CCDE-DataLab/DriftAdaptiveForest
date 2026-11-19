"""Escenarios sintéticos de regresión con distintos tipos de drift.

Cada generador devuelve (X, y) ordenados en el tiempo (fila 0 = más antigua),
con X como DataFrame numérico y y como Serie. El reloj normalizado es
t = linspace(0, 1, n), igual que en ``generate_data``.

Escenarios (tipo de drift real que contiene cada uno):
  healthcare_none        sin drift (control)
  healthcare_data        solo data drift  P(X)
  healthcare_concept     solo concept drift P(Y|X) (sigmoide en t=0.6)
  healthcare_both        data + concept drift (el del README)
  friedman_abrupt        concept drift abrupto en t=0.6 (permutación de roles, Ikonomovska 2011), 10 vars
  friedman_gradual_data  concept drift gradual t∈[0.35,0.75] + data drift en 5 vars
  friedman_recurring     conceptos A→B→A→B con firma en X (recurrente)
  covariate_shift        solo data drift fuerte (2σ) con P(Y|X) fijo -> obliga a extrapolar
  highdim_mixed          20 vars correlacionadas, data drift + drift abrupto (t=0.7) + incremental
  local_drift            concept drift solo en la región x0>0.6 (invisible para MMD)
  incremental            coeficientes que rotan continuamente en el tiempo
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.drift_adaptive_forest.data.generate_data import generate_data

N_DEFAULT = 5000


def _frame(X: np.ndarray, y: np.ndarray, prefix: str = "x") -> tuple[pd.DataFrame, pd.Series]:
    cols = [f"{prefix}{j}" for j in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def healthcare(drift_type: str, n: int = N_DEFAULT, seed: int = 0):
    df = generate_data(n_samples=n, drift_type=drift_type, random_state=seed)
    X = df.drop(columns=["paciente_id", "fecha_ingreso", "dias_estancia"]).astype(float)
    y = df["dias_estancia"].astype(float)
    return X.reset_index(drop=True), y.reset_index(drop=True)


def _friedman_a(X):
    return 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]


def _friedman_b(X):
    # Concepto B: permutación de los roles de las variables (Ikonomovska et al., 2011)
    return 10 * np.sin(np.pi * X[:, 3] * X[:, 4]) + 20 * (X[:, 1] - 0.5) ** 2 + 10 * X[:, 0] + 5 * X[:, 2]


def friedman_abrupt(n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = rng.uniform(0, 1, size=(n, 10))
    y = np.where(t < 0.6, _friedman_a(X), _friedman_b(X)) + rng.normal(0, 1.0, n)
    return _frame(X, y)


def friedman_gradual_data(n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = rng.uniform(0, 1, size=(n, 10))
    # data drift: las 5 primeras variables desplazan su media de 0.29 a 0.71
    for j in range(5):
        X[:, j] = rng.beta(2 + 3 * t, 5 - 3 * t)
    mix = np.clip((t - 0.35) / 0.40, 0, 1)
    y = (1 - mix) * _friedman_a(X) + mix * _friedman_b(X) + rng.normal(0, 1.0, n)
    return _frame(X, y)


def friedman_recurring(n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    concept_b = ((t >= 0.25) & (t < 0.5)) | (t >= 0.75)
    X = rng.uniform(0, 1, size=(n, 10))
    # firma del concepto en X: bajo B, x6 y x7 se concentran en valores altos
    X[concept_b, 6] = rng.uniform(0.4, 1.0, concept_b.sum())
    X[concept_b, 7] = rng.uniform(0.4, 1.0, concept_b.sum())
    y = np.where(concept_b, _friedman_b(X), _friedman_a(X)) + rng.normal(0, 1.0, n)
    return _frame(X, y)


def covariate_shift(n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = rng.normal(0, 1, size=(n, 8))
    X[:, :4] += 2.0 * t[:, None]  # la media se desplaza 2σ en 4 variables
    y = (
        2 * np.sin(1.5 * X[:, 0])
        + 0.5 * X[:, 1] ** 2
        + 1.5 * X[:, 2] * X[:, 3]
        + X[:, 4]
        + 0.5 * X[:, 5] * X[:, 6]
        + rng.normal(0, 0.5, n)
    )
    return _frame(X, y)


def highdim_mixed(n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = 20
    t = np.linspace(0, 1, n)
    cov = 0.5 ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    X[:, :5] += 1.5 * t[:, None]  # data drift en media
    X[:, 5:10] *= (1 + 1.0 * t)[:, None]  # data drift en varianza
    sign = np.where(t < 0.7, 1.0, -1.0)  # drift abrupto: cambio de signo
    w2 = 1.0 + 2.0 * t  # drift incremental
    y = (
        2.0 * sign * np.tanh(X[:, 0])
        + 1.5 * sign * X[:, 1]
        + w2 * np.abs(X[:, 2])
        + 0.8 * X[:, 3] ** 2
        + np.sin(2 * X[:, 4])
        + 0.7 * X[:, 5] * X[:, 6]
        + rng.normal(0, 1.0, n)
    )
    return _frame(X, y)


def local_drift(n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = rng.uniform(0, 1, size=(n, 6))
    base = 5 * X[:, 0] + 4 * np.sin(np.pi * X[:, 1]) + 3 * X[:, 2] * X[:, 3] + 2 * X[:, 4]
    region = (X[:, 0] > 0.6) & (t > 0.55)
    y = base + region * (8 * (X[:, 1] - 0.5) + 3.0) + rng.normal(0, 0.7, n)
    return _frame(X, y)


def incremental(n: int = N_DEFAULT, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    X = rng.normal(0, 1, size=(n, 10))
    w0 = rng.normal(0, 1.5, 10)
    w1 = rng.normal(0, 1.5, 10)
    W = (1 - t)[:, None] * w0 + t[:, None] * w1
    y = (W * X).sum(axis=1) + 0.5 * X[:, 0] * X[:, 1] + rng.normal(0, 0.5, n)
    return _frame(X, y)


SCENARIOS = {
    "healthcare_none": (lambda seed: healthcare("none", seed=seed), "ninguno (control)"),
    "healthcare_data": (lambda seed: healthcare("data", seed=seed), "data"),
    "healthcare_concept": (lambda seed: healthcare("concept", seed=seed), "concept gradual"),
    "healthcare_both": (lambda seed: healthcare("both", seed=seed), "data + concept"),
    "friedman_abrupt": (lambda seed: friedman_abrupt(seed=seed), "concept abrupto"),
    "friedman_gradual_data": (lambda seed: friedman_gradual_data(seed=seed), "data + concept gradual"),
    "friedman_recurring": (lambda seed: friedman_recurring(seed=seed), "concept recurrente + data"),
    "covariate_shift": (lambda seed: covariate_shift(seed=seed), "data fuerte (extrapolación)"),
    "highdim_mixed": (lambda seed: highdim_mixed(seed=seed), "data + concept abrupto/incremental"),
    "local_drift": (lambda seed: local_drift(seed=seed), "concept local"),
    "incremental": (lambda seed: incremental(seed=seed), "concept incremental"),
}
