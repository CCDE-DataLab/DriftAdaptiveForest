import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


def load_data(ruta:Path)->pd.DataFrame:
    df = pd.read_parquet(ruta)
    return df

def estimate_rbf_gamma(X_ini:np.ndarray,X_fin:np.ndarray)-> float:

    Z = np.vstack([X_ini,X_fin])
    rng = np.random.default_rng(42)

    if len(Z) > 1000:

        idx = rng.choice(len(Z), size=1000, replace=False)
        Z_sub = Z[idx]
    else:
        Z_sub = Z

    dists_sq = pdist(Z_sub, metric="sqeuclidean")
    dists_pos = dists_sq[dists_sq>0]
    if len(dists_pos) == 0:
        return 1.0

    mediana_sq = np.median(dists_pos)

    gamma = 1 / (2*mediana_sq)

    return float(gamma)


def compute_mmd_square(X_ini:np.ndarray,X_fin:np.ndarray,gamma: float | None = None, unbiased: bool = True) ->float:
    
    P = X_ini
    Q = X_fin
    
    if gamma is None:
        gamma = estimate_rbf_gamma(X_ini=X_ini,X_fin=X_fin)

    
    #Matrices de similitud RBF
    K_PP = rbf_kernel(P, gamma=gamma)
    K_QQ = rbf_kernel(Q,gamma=gamma)
    K_PQ = rbf_kernel(P,Q,gamma=gamma)


    term_pq = K_PQ.mean()
    if not unbiased:
        term_pp_biased = K_PP.mean()
        term_qq_biased = K_QQ.mean()
        mmd = term_pp_biased + term_qq_biased - 2.0*term_pq
    else:
        n, m = len(P), len(Q)
        if n<=1 or m<=1:
            raise ValueError("Se necesitan al menos 2 muestras por grupo para MMD insesgado")
        
        term_pp_unbiased = (K_PP.sum() - np.trace(K_PP)) / (n * (n - 1))
        term_qq_unbiased = (K_QQ.sum() - np.trace(K_QQ)) / (m * (m - 1))
        mmd = term_pp_unbiased + term_qq_unbiased - 2.0*term_pq


    return float(mmd)



if __name__ == "__main__":
    df = load_data(ruta=r"C:\Users\mauri\Documents\CCDE\DriftAdaptiveForest\data\raw\healthcare_drift.parquet") #creemos luego un config para las rutas y poder importarlas como librerias
    
    print(f"Datos cargados exitosamente! Las columnas son las siguientes:{df.columns.to_list()}\n")
    df = df.copy()
    columns_drop = ["paciente_id","fecha_ingreso","dias_estancia"]
    df = df.drop(columns=columns_drop,axis=1)
    print(f"Columnas {columns_drop} eliminadas exitosamente.\n")
    x_ini = df[:500]
    x_fin = df[-500:]
    print("Iniciando proceso de estandarizacion")
    scaler = StandardScaler()
    X_ini_s = scaler.fit_transform(x_ini)
    X_fin_s = scaler.transform(x_fin)
    print("Conjunto de datos iniciales y finales estandarizados con exito\n")
    
    
    print("Calculando DataDrift con MMD\n")    
    mmd = compute_mmd_square(X_ini=X_ini_s,X_fin=X_fin_s)
    print(f"MMD calculado con exito.\nMMD:{mmd:.4f}")












