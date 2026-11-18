from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import root_mean_squared_error
from src.drift_adaptive_forest.metrics.mmd import compute_mmd_square,load_data
from scipy.special import softmax


class DriftRegressor(BaseEstimator,RegressorMixin):

    def __init__(self,n_blocks:int = 5,n_estimators_per_block: int = 100,beta:float = 50.0, random_state:int | None = 42):
        self.n_blocks = n_blocks
        self.n_estimators_per_block = n_estimators_per_block
        self.beta = beta
        self.random_state = random_state


    def fit(self,X,y,timestamps=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if timestamps is not None:
            orden = np.argsort(timestamps)
            X = X[orden]
            y = y[orden]

        bloques_idx = np.array_split(np.arange(len(X)),self.n_blocks)


        self.models_ = []
        self.block_features_ = []
        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(X)

        for bloque in bloques_idx:
            X_bloque = X[bloque]
            y_bloque = y[bloque]

            rf = RandomForestRegressor(n_estimators=self.n_estimators_per_block,random_state=self.random_state)
            rf.fit(X_bloque,y_bloque)
            
            X_bloque_scaled = X_scaled[bloque]

            self.models_.append(rf)
            self.block_features_.append(X_bloque_scaled)
        
        return self


    def compute_weights(self,X_scaled:np.ndarray) -> np.ndarray:

        self.mmd_results_ = []
        
        
        for bloque in self.block_features_:
            mmd = compute_mmd_square(X_ini = bloque, X_fin=X_scaled)
            mmd = max(0.0,mmd)
            self.mmd_results_.append(mmd)

        mmd_array = np.asarray(self.mmd_results_)
        weights = softmax(-self.beta*mmd_array)
        self.last_weights_ = weights
        
        return self.last_weights_


    def predict(self,X):

        check_is_fitted(self,["models_","block_features_"])

        X_test = np.asarray(X)
        self.pred_list = []
        for model in self.models_:
            pred = model.predict(X_test)
            self.pred_list.append(pred)

        preds = np.asarray(self.pred_list)

        
        X_test_scaled = self.scaler_.transform(X_test)
            
        weights = self.compute_weights(X_scaled=X_test_scaled)

        y_pred = np.average(preds,axis=0,weights=weights)


        return y_pred


if __name__ == "__main__":
    df = load_data(ruta = r"C:\Users\mauri\Documents\CCDE\DriftAdaptiveForest\data\raw\healthcare_drift.parquet")
    df = df.sort_values("fecha_ingreso",ascending=True).reset_index(drop=True)
    columns_drop = ["paciente_id","fecha_ingreso","dias_estancia"]
    X = df.drop(columns=columns_drop,axis=1)
    y = df["dias_estancia"]
    X_train = X[:4000]
    y_train = y[:4000]
    X_test = X[-1000:]
    y_test = y[-1000:]

    print("Entrenando con modelo clasico de Random Forest:\n")
    rf = RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1) #sugiereme hiperparametros para este modelo
    rf.fit(X_train,y_train)
    y_pred_rf = rf.predict(X_test)
    rmse_rf = root_mean_squared_error(y_true=y_test,y_pred=y_pred_rf)
    print(f"Modelo clasico entrenado con exito.\nResultados del modelo:{rmse_rf:.4f}\n")

    print("Entrenando modelo con Drift adaptativo:\n")

    dr = DriftRegressor()
    dr.fit(X = X_train,y = y_train)
    y_pred_dr = dr.predict(X_test)
    rmse_dr = root_mean_squared_error(y_true=y_test,y_pred=y_pred_dr)
    print(f"Modelo adaptativo entrenado con exito.\nResultados del modelo:{rmse_dr:.4f}\n")
    print(f"Resultados del modelo Drift Adaptativo:\nMMD:{dr.mmd_results_}\nPesos asignados:{dr.last_weights_}")






