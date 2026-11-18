import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


def generate_data(n_samples:int=5000,start_date:str="2023-01-01", 
    drift_type:str="data",path:str | Path | None = None,random_state:int = 42) ->pd.DataFrame:
    """
    Esta funcion genera datos sinteticos para el proyecto de Drift Adaptative Forest
    """
    rng = np.random.default_rng(random_state)
    t_norm = np.linspace(0, 1, n_samples)



    fechas = pd.to_datetime(start_date) + pd.to_timedelta(
    np.sort(rng.uniform(0, 730, size=n_samples)), unit="D"
    )

    if drift_type in ("data","both"):
        media_edad_t = 45 + 15 * t_norm + 5 * np.sin(4 * np.pi * t_norm)
        media_severidad_t = 4.0 + 3.0 * t_norm + 1.5 * np.sin(4 * np.pi * t_norm)
        a_ocupacion_t = 2.5 + 4.5*t_norm
        lam_morbilidad = 1.5 + 1.0 * t_norm
    else:
        media_edad_t = 45
        media_severidad_t = 4
        a_ocupacion_t = 2.5
        lam_morbilidad = 1.5

    
    edad = np.clip(rng.normal(loc=media_edad_t, scale=12,size=n_samples), 18, 95)

    comorbilidad = rng.poisson(lam=lam_morbilidad,size=n_samples)


    
    severidad_ingreso = np.clip(rng.normal(loc=media_severidad_t,scale=1.5,size=n_samples),1,10)

    ocupacion_hospitalaria = np.clip(rng.beta(a_ocupacion_t,2.5,size=n_samples),0.3,1.0)


    ruido = rng.normal(0, 1.2, size=n_samples)

    estancia_vieja = (
        2.0
        + 0.05 * edad
        + 0.8 * comorbilidad
        + comorbilidad**2
        + severidad_ingreso ** (1 / 2)
        + 3.0 * ocupacion_hospitalaria ** (1 / 3)
    )

    if drift_type in ("concept", "both"):
        # Nuevo protocolo clinico a partir de t_norm ~ 0.6:
        # reduce el impacto cuadratico de comorbilidad y aumenta sensibilidad a severidad/saturacion
        alpha_t = 1.0 / (1.0 + np.exp(-15.0 * (t_norm - 0.6)))
        estancia_nueva = (
            1.0
            + 0.02 * edad
            + 0.3 * comorbilidad
            + 0.25 * (comorbilidad**2)
            + 1.8 * severidad_ingreso
            - 2.0 * ocupacion_hospitalaria
            + np.where(ocupacion_hospitalaria > 0.82, 4.5, 0.0)
        )
        base_estancia = (1.0 - alpha_t) * estancia_vieja + alpha_t * estancia_nueva
    else:
        base_estancia = estancia_vieja

    dias_estancia = np.clip(base_estancia + ruido, 1, None)

    
    
    df = pd.DataFrame({
    "paciente_id": [f"PAC_{i:05d}" for i in range(n_samples)],
    "fecha_ingreso": fechas,
    "edad": edad.round(0).astype(int),
    "comorbilidad": comorbilidad,
    "dias_estancia": dias_estancia.round(1),
    "severidad_ingreso":severidad_ingreso.round(1),
    "ocupacion_hospitalaria":ocupacion_hospitalaria.round(2)
    })
    df = df.sort_values("fecha_ingreso", ascending=True).reset_index(drop=True)



    if path is not None:
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        ruta = path
        df.to_parquet(path=ruta,index=False)

    return df




if __name__ == "__main__":
    df_both = generate_data(drift_type="both", path=RAW_DIR / "healthcare_both_drift.parquet")
    print(f"Dataset 'both' guardado exitosamente. Filas y columnas: {df_both.shape}")