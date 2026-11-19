"""AdaptiveDriftForest: versión mejorada del DriftRegressor.

Corrige los problemas detectados en el benchmark (ver experiments/BITACORA.md):

1. Candidatos que contienen a las líneas base. Además de los K bloques históricos (útiles en drift
   recurrente) se entrenan ventanas recientes anidadas (p. ej. 10 %, 20 %, 35 %, 60 %, 100 % del
   histórico). La ventana del 100 % es el RF clásico y las cortas son el RF de ventana, así que el
   ensamble puede degenerar en la mejor línea base si eso es lo que mejor funciona.
2. Pesos por error normalizado. p_k = MSE_k / min_j MSE_j, de modo que beta tiene el mismo significado
   en cualquier dataset (beta=20 => un candidato 5 % peor recibe exp(-1) del peso del mejor).
3. Reentrenamiento con la cola de validación. Los pesos se estiman con modelos que no vieron el 15 %
   más reciente, y luego los candidatos se reentrenan incluyéndolo (no se desperdician los datos nuevos).
4. Pesos fijos tras fit. La predicción no depende del lote de test: funciona con una sola fila y es
   determinista fila a fila. Opcionalmente (weighting="hybrid") se añade un término MMD entre cada
   candidato y la cola reciente (no contra el test).
"""

from __future__ import annotations

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.drift_adaptive_forest.metrics.mmd import compute_mmd_square


class AdaptiveDriftForest(BaseEstimator, RegressorMixin):
    VALID_WEIGHTING = ("error", "hybrid")

    def __init__(
        self,
        n_blocks: int = 5,
        window_fracs: tuple = (0.1, 0.2, 0.35, 0.6, 1.0),
        n_estimators: int = 100,
        beta: float = 20.0,
        val_frac: float = 0.15,
        weighting: str = "error",
        alpha_mmd: float = 0.3,
        refit: bool = True,
        min_weight: float = 1e-3,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        random_state: int | None = 42,
        n_jobs: int = -1,
    ):
        self.n_blocks = n_blocks
        self.window_fracs = window_fracs
        self.n_estimators = n_estimators
        self.beta = beta
        self.val_frac = val_frac
        self.weighting = weighting
        self.alpha_mmd = alpha_mmd
        self.refit = refit
        self.min_weight = min_weight
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ helpers
    def _new_rf(self):
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def _candidates(self, n_tr: int, min_size: int):
        """Lista de (nombre, tipo, inicio, fin, frac) sobre el rango [0, n_tr)."""
        cands, seen = [], set()
        for f in sorted(set(self.window_fracs)):
            size = n_tr if f >= 1.0 else max(min_size, int(round(f * n_tr)))
            start = max(0, n_tr - size)
            if (start, n_tr) not in seen:
                seen.add((start, n_tr))
                cands.append((f"ventana_{f:.2f}", "ventana", start, n_tr, f))
        if self.n_blocks and self.n_blocks > 1:
            for b, idx in enumerate(np.array_split(np.arange(n_tr), self.n_blocks)):
                start, end = int(idx[0]), int(idx[-1]) + 1
                if (start, end) not in seen and end - start >= min_size:
                    seen.add((start, end))
                    cands.append((f"bloque_{b + 1}", "bloque", start, end, None))
        return cands

    # ------------------------------------------------------------------ API
    def fit(self, X, y, timestamps=None):
        if self.weighting not in self.VALID_WEIGHTING:
            raise ValueError(f"weighting debe ser uno de {self.VALID_WEIGHTING}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if timestamps is not None:
            orden = np.argsort(np.asarray(timestamps), kind="stable")
            X, y = X[orden], y[orden]

        n = len(X)
        val = max(30, int(n * self.val_frac))
        n_tr = n - val
        min_size = max(30, int(0.03 * n))
        if n_tr < 2 * min_size:
            raise ValueError("Muy pocas muestras para separar validación reciente y entrenar candidatos.")

        cands = self._candidates(n_tr, min_size)
        X_val, y_val = X[n_tr:], y[n_tr:]

        # 1) error de cada candidato sobre la cola reciente (sin leakage)
        mse, models_val = [], []
        for _, _, s, e, _ in cands:
            m = self._new_rf().fit(X[s:e], y[s:e])
            models_val.append(m)
            mse.append(mean_squared_error(y_val, m.predict(X_val)))
        mse = np.asarray(mse)
        penalty = mse / (mse.min() + 1e-12)

        if self.weighting == "hybrid":
            scaler = StandardScaler().fit(X[:n_tr])
            rng = np.random.default_rng(self.random_state)
            Zv = scaler.transform(X_val)
            mmd = []
            for _, _, s, e, _ in cands:
                idx = np.arange(s, e)
                if len(idx) > 1000:
                    idx = rng.choice(idx, 1000, replace=False)
                mmd.append(max(0.0, compute_mmd_square(scaler.transform(X[idx]), Zv)))
            mmd = np.asarray(mmd)
            penalty = (1 - self.alpha_mmd) * penalty + self.alpha_mmd * (1 + mmd / (mmd.mean() + 1e-12))
            self.mmd_ = mmd

        w = softmax(-self.beta * (penalty - penalty.min()))
        w[w < self.min_weight] = 0.0
        w = w / w.sum()

        # 2) reentrenar los candidatos con peso > 0 incluyendo la cola reciente
        self.models_, self.weights_, self.candidate_names_ = [], [], []
        for (name, kind, s, e, f), m, wk in zip(cands, models_val, w):
            if wk == 0:
                continue
            if self.refit:
                if kind == "ventana":
                    size = n if f >= 1.0 else e - s
                    s2, e2 = n - size, n
                else:  # bloque: solo el último se extiende hasta el final
                    s2, e2 = s, (n if e == n_tr else e)
                m = self._new_rf().fit(X[s2:e2], y[s2:e2])
            self.models_.append(m)
            self.weights_.append(wk)
            self.candidate_names_.append(name)

        self.weights_ = np.asarray(self.weights_)
        self.all_candidates_ = [c[0] for c in cands]
        self.val_mse_ = mse
        self.all_weights_ = w
        return self

    def predict(self, X):
        check_is_fitted(self, ["models_", "weights_"])
        X = np.asarray(X, dtype=float)
        preds = np.asarray([m.predict(X) for m in self.models_])
        return np.average(preds, axis=0, weights=self.weights_)

    def weights_summary(self) -> dict:
        check_is_fitted(self, ["models_", "weights_"])
        return {n: round(float(w), 4) for n, w in zip(self.candidate_names_, self.weights_)}
