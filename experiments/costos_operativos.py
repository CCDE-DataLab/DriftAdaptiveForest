"""Costos operativos: tiempo, CPU, RAM máxima y tamaño del modelo guardado.

Caso: matrícula IPEDS, h = 1, origen 2022 (el corte más grande: ~57 000 filas × 13 variables).
Cada modelo corre en un proceso aparte para medir la RAM sin interferencias.
RAM = pico del conjunto de trabajo (PeakWorkingSetSize) de Windows.

Uso: venv_drift_forest/Scripts/python.exe -I experiments/costos_operativos.py
"""

from __future__ import annotations

import ctypes
import json
import pickle
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELOS = ["OLS_global", "LightGBM", "RF_clasico", "DR_recent_error", "ADF_error"]
TMP = Path(__import__("tempfile").gettempdir()) / "drift_costos"


def preparar_datos():
    import numpy as np
    from experiments.matricula_data import panel_balanceado
    from experiments.matricula_experiment import FEATS, construir_filas
    m, h = panel_balanceado()
    f = construir_filas(m, h, 1)
    tr, te = f[f.t + 1 <= 2022], f[f.t == 2022]
    TMP.mkdir(exist_ok=True)
    np.save(TMP / "Xtr.npy", tr[FEATS].to_numpy()); np.save(TMP / "ytr.npy", tr["y"].to_numpy())
    np.save(TMP / "Xte.npy", te[FEATS].to_numpy())


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def pico_mb() -> float:
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = wintypes.HANDLE  # sin esto el handle se trunca a 32 bits
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo falló")
    return c.PeakWorkingSetSize / 2 ** 20


def hijo(nombre: str):
    import warnings
    warnings.filterwarnings("ignore")
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    import numpy as np
    from src.drift_adaptive_forest.models.adaptive_ensemble import AdaptiveDriftForest
    from src.drift_adaptive_forest.models.block_ensemble import DriftRegressor

    Xtr, ytr, Xte = np.load(TMP / "Xtr.npy"), np.load(TMP / "ytr.npy"), np.load(TMP / "Xte.npy")
    base = pico_mb()
    modelo = {
        "OLS_global": lambda: LinearRegression(),
        "LightGBM": lambda: LGBMRegressor(random_state=0, verbose=-1),
        "RF_clasico": lambda: RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1),
        "DR_recent_error": lambda: DriftRegressor(n_blocks=5, beta=15.0, weighting_mode="recent_error", random_state=0),
        "ADF_error": lambda: AdaptiveDriftForest(random_state=0),
    }[nombre]()
    t0, c0 = time.perf_counter(), time.process_time()
    modelo.fit(Xtr, ytr)
    t_fit, c_fit = time.perf_counter() - t0, time.process_time() - c0
    t0 = time.perf_counter()
    modelo.predict(Xte)
    t_pred = time.perf_counter() - t0
    t0 = time.perf_counter()
    modelo.predict(Xte[:1])
    t_1 = time.perf_counter() - t0
    print(json.dumps({"modelo": nombre, "filas_train": len(Xtr), "fit_s": t_fit, "cpu_fit_s": c_fit,
                      "pred_3196_s": t_pred, "pred_1fila_ms": 1000 * t_1,
                      "ram_extra_mb": pico_mb() - base, "ram_pico_mb": pico_mb(),
                      "modelo_mb": len(pickle.dumps(modelo)) / 2 ** 20}))


def main():
    preparar_datos()
    filas = []
    for nombre in MODELOS:
        out = subprocess.run([sys.executable, "-I", __file__, "--hijo", nombre], capture_output=True, text=True)
        linea = [x for x in out.stdout.splitlines() if x.startswith("{")]
        if not linea:
            print(nombre, "ERROR", out.stderr[-500:])
            continue
        filas.append(json.loads(linea[-1]))
        d = filas[-1]
        print(f"{nombre:16s} fit {d['fit_s']:7.2f}s  CPU {d['cpu_fit_s']:7.1f}s  pred {d['pred_3196_s']:6.3f}s  "
              f"1 fila {d['pred_1fila_ms']:7.1f}ms  RAM extra {d['ram_extra_mb']:7.0f}MB  pico {d['ram_pico_mb']:6.0f}MB  "
              f"modelo {d['modelo_mb']:8.1f}MB", flush=True)
    import pandas as pd
    pd.DataFrame(filas).to_csv(ROOT / "experiments" / "results" / "costos_operativos.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--hijo":
        hijo(sys.argv[2])
    else:
        main()
