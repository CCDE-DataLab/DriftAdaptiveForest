from pathlib import Path
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.drift_adaptive_forest.data.generate_data import generate_data
from src.drift_adaptive_forest.metrics.mmd import load_data
from src.drift_adaptive_forest.models.block_ensemble import DriftRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "healthcare_both_drift.parquet"


def temporal_cv_rmse(model_fn, X_train, y_train, n_splits: int = 4) -> float:
    """Evalúa un estimador con TimeSeriesSplit y retorna el RMSE promedio."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses = []
    for tr_idx, va_idx in tscv.split(X_train):
        model = model_fn()
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        preds = model.predict(X_train.iloc[va_idx])
        rmses.append(root_mean_squared_error(y_train.iloc[va_idx], preds))
    return float(np.mean(rmses))


def tune_random_forest(X_train, y_train, n_trials: int = 20) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250, step=50),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
        return temporal_cv_rmse(lambda: RandomForestRegressor(**params), X_train, y_train)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_drift_regressor(X_train, y_train, n_trials: int = 25) -> dict:
    def objective(trial):
        params = {
            "n_blocks": trial.suggest_int("n_blocks", 3, 8),
            "n_estimators_per_block": trial.suggest_int("n_estimators_per_block", 50, 150, step=25),
            "beta": trial.suggest_float("beta", 2.0, 60.0, log=True),
            "weighting_mode": trial.suggest_categorical("weighting_mode", ["mmd", "recent_error", "hybrid"]),
            "recent_val_frac": trial.suggest_float("recent_val_frac", 0.10, 0.25),
            "alpha_hybrid": trial.suggest_float("alpha_hybrid", 0.1, 0.9),
            "max_depth": trial.suggest_int("max_depth", 4, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "random_state": 42,
        }
        return temporal_cv_rmse(lambda: DriftRegressor(**params), X_train, y_train)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def main():
    if not DATA_PATH.exists():
        generate_data(drift_type="both", path=DATA_PATH)

    df = load_data(ruta=DATA_PATH)
    df = df.sort_values("fecha_ingreso", ascending=True).reset_index(drop=True)

    X = df.drop(columns=["paciente_id", "fecha_ingreso", "dias_estancia"])
    y = df["dias_estancia"]

    X_train, y_train = X.iloc[:4000], y.iloc[:4000]
    X_test, y_test = X.iloc[4000:], y.iloc[4000:]

    print("1. Optimizando RandomForestRegressor clásico con Optuna (TimeSeriesSplit)...")
    best_rf_params = tune_random_forest(X_train, y_train, n_trials=20)
    rf_tuned = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    rf_tuned.fit(X_train, y_train)
    pred_rf = rf_tuned.predict(X_test)

    print("2. Optimizando DriftRegressor (Estrategia B) con Optuna (TimeSeriesSplit)...")
    best_dr_params = tune_drift_regressor(X_train, y_train, n_trials=25)
    dr_tuned = DriftRegressor(**best_dr_params, random_state=42)
    dr_tuned.fit(X_train, y_train)
    pred_dr = dr_tuned.predict(X_test)

    print("\n=== RESULTADOS EN TEST FUTURO (últimas 1,000 filas, Data + Concept Drift) ===")
    for name, preds in [("RF Clásico (Optuna)", pred_rf), ("DriftRegressor (Optuna)", pred_dr)]:
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"{name:25s} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    print("\nMejores hiperparámetros RF Clásico:", best_rf_params)
    print("Mejores hiperparámetros DriftRegressor:", best_dr_params)
    print("Pesos finales asignados por DriftRegressor:", np.round(dr_tuned.last_weights_, 4))


if __name__ == "__main__":
    main()
