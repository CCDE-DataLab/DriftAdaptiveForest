from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.drift_adaptive_forest.metrics.mmd import compute_mmd_square, load_data


class DriftRegressor(BaseEstimator, RegressorMixin):
    """Ensamble temporal por bloques con pesos dinámicos por MMD y/o error reciente."""

    VALID_MODES = ("mmd", "recent_error", "hybrid")

    def __init__(
        self,
        n_blocks: int = 5,
        n_estimators_per_block: int = 100,
        beta: float = 10.0,
        weighting_mode: str = "mmd",
        recent_val_frac: float = 0.15,
        alpha_hybrid: float = 0.5,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int | None = 42,
    ):
        self.n_blocks = n_blocks
        self.n_estimators_per_block = n_estimators_per_block
        self.beta = beta
        self.weighting_mode = weighting_mode
        self.recent_val_frac = recent_val_frac
        self.alpha_hybrid = alpha_hybrid
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y, timestamps=None):
        if self.weighting_mode not in self.VALID_MODES:
            raise ValueError(
                f"weighting_mode debe ser uno de {self.VALID_MODES}, se recibió '{self.weighting_mode}'."
            )

        X = np.asarray(X)
        y = np.asarray(y)

        if timestamps is not None:
            orden = np.argsort(np.asarray(timestamps))
            X = X[orden]
            y = y[orden]

        # En modo 'recent_error' o 'hybrid', reservamos la cola más reciente como validación
        # para medir qué bloques siguen prediciendo bien P(Y|X) sin data leakage.
        if self.weighting_mode in ("recent_error", "hybrid"):
            val_size = max(20, int(len(X) * self.recent_val_frac))
            if val_size >= len(X) - self.n_blocks * 2:
                raise ValueError("Muy pocas muestras para separar validación reciente y entrenar bloques.")
            X_train_blocks, y_train_blocks = X[:-val_size], y[:-val_size]
            X_val_rec, y_val_rec = X[-val_size:], y[-val_size:]
        else:
            X_train_blocks, y_train_blocks = X, y
            X_val_rec, y_val_rec = None, None

        bloques_idx = np.array_split(np.arange(len(X_train_blocks)), self.n_blocks)

        self.models_ = []
        self.block_features_ = []
        self.recent_mse_ = []
        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(X_train_blocks)

        for bloque in bloques_idx:
            X_bloque = X_train_blocks[bloque]
            y_bloque = y_train_blocks[bloque]

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators_per_block,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_bloque, y_bloque)

            self.models_.append(rf)
            self.block_features_.append(X_scaled[bloque])

            if X_val_rec is not None:
                pred_val = rf.predict(X_val_rec)
                self.recent_mse_.append(float(mean_squared_error(y_val_rec, pred_val)))

        return self

    def compute_weights(self, X_scaled: np.ndarray) -> np.ndarray:
        self.mmd_results_ = []

        if self.weighting_mode in ("mmd", "hybrid"):
            for bloque in self.block_features_:
                mmd = compute_mmd_square(X_ini=bloque, X_fin=X_scaled)
                self.mmd_results_.append(max(0.0, mmd))
            mmd_array = np.asarray(self.mmd_results_, dtype=float)

        if self.weighting_mode == "mmd":
            penalties = mmd_array
        elif self.weighting_mode == "recent_error":
            mse_array = np.asarray(self.recent_mse_, dtype=float)
            penalties = mse_array / (mse_array.mean() + 1e-8)
        else:  # "hybrid"
            mse_array = np.asarray(self.recent_mse_, dtype=float)
            mmd_norm = mmd_array / (mmd_array.mean() + 1e-8)
            mse_norm = mse_array / (mse_array.mean() + 1e-8)
            penalties = self.alpha_hybrid * mmd_norm + (1.0 - self.alpha_hybrid) * mse_norm

        weights = softmax(-self.beta * penalties)
        self.last_weights_ = weights
        return self.last_weights_

    def predict(self, X):
        check_is_fitted(self, ["models_", "block_features_"])

        X_test = np.asarray(X)
        preds = np.asarray([model.predict(X_test) for model in self.models_])

        X_test_scaled = self.scaler_.transform(X_test)
        weights = self.compute_weights(X_scaled=X_test_scaled)

        return np.average(preds, axis=0, weights=weights)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[3] / "data" / "raw" / "healthcare_both_drift.parquet"
    df = load_data(ruta=data_path)
    df = df.sort_values("fecha_ingreso", ascending=True).reset_index(drop=True)

    columns_drop = ["paciente_id", "fecha_ingreso", "dias_estancia"]
    X = df.drop(columns=columns_drop)
    y = df["dias_estancia"]

    X_train, y_train = X[:4000], y[:4000]
    X_test, y_test = X[-1000:], y[-1000:]

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rmse_rf = root_mean_squared_error(y_test, rf.predict(X_test))
    print(f"RF Clásico RMSE: {rmse_rf:.4f}\n")

    for mode in ("mmd", "recent_error", "hybrid"):
        dr = DriftRegressor(n_blocks=5, beta=15.0, weighting_mode=mode)
        dr.fit(X_train, y_train)
        rmse_dr = root_mean_squared_error(y_test, dr.predict(X_test))
        print(f"DriftRegressor ({mode}) RMSE: {rmse_dr:.4f}")
        print(f"  Pesos: {np.round(dr.last_weights_, 4)}")
