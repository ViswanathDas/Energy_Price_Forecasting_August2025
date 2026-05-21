"""
Forecasting Models — Energy Price Forecasting
==============================================
Baseline and gradient-boosted models for day-ahead price prediction.

Models included:
  1. Naive baseline — "same hour yesterday" (lag-24 forecast)
  2. Linear regression — interpretable, fast, good sanity check
  3. Gradient Boosted Trees (LightGBM or sklearn) — best accuracy

The philosophy: start simple, understand it, then add complexity.
Each model outputs the same interface so you can swap them easily.
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings


# ── Naive Baseline ─────────────────────────────────────────────────────────────

class NaiveBaseline:
    """
    Predicts tomorrow's price = same hour yesterday (lag-24).
    Unbeatable benchmark in electricity markets — surprisingly hard to beat.
    """

    def __init__(self, lag: int = 24):
        self.lag = lag

    def fit(self, X, y):
        # No training needed — just needs 'lag_24h' in features
        self._lag_col = f"lag_{self.lag}h"
        if self._lag_col not in X.columns:
            raise ValueError(f"NaiveBaseline needs column '{self._lag_col}' in X")
        return self

    def predict(self, X):
        return X[self._lag_col].values

    def __repr__(self):
        return f"NaiveBaseline(lag={self.lag}h)"


# ── Linear Regression ─────────────────────────────────────────────────────────

class LinearModel:
    """
    Ridge regression with feature scaling.
    Useful for understanding which features matter and as a fast baseline.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._scaler = None
        self._model = None

    def fit(self, X, y):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X_scaled, y)
        self._feature_names = list(X.columns)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def feature_importance(self, top_n: int = 15) -> pd.Series:
        """Return top-N features by absolute coefficient magnitude."""
        coefs = pd.Series(
            np.abs(self._model.coef_), index=self._feature_names
        ).sort_values(ascending=False)
        return coefs.head(top_n)

    def __repr__(self):
        return f"LinearModel(alpha={self.alpha})"


# ── Gradient Boosted Trees ─────────────────────────────────────────────────────

class GradientBoostedModel:
    """
    LightGBM (preferred) with sklearn GradientBoostingRegressor as fallback.
    Best accuracy for tabular time-series features.

    LightGBM install: pip install lightgbm
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None
        self._use_lgbm = False

    def fit(self, X, y):
        self._feature_names = list(X.columns)

        try:
            import lightgbm as lgb
            self._use_lgbm = True
            self._model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError:
            warnings.warn(
                "LightGBM not installed — using sklearn GradientBoostingRegressor.\n"
                "Install with: pip install lightgbm"
            )
            from sklearn.ensemble import GradientBoostingRegressor
            self._model = GradientBoostingRegressor(
                n_estimators=min(self.n_estimators, 200),
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
            )

        self._model.fit(X, y)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    def feature_importance(self, top_n: int = 15) -> pd.Series:
        """Return top-N features by model importance."""
        if self._model is None:
            raise RuntimeError("Call fit() before feature_importance().")
        if self._use_lgbm:
            imp = self._model.booster_.feature_importance(importance_type="gain")
        else:
            imp = self._model.feature_importances_

        series = pd.Series(imp, index=self._feature_names)
        series = series / series.sum()   # normalise to sum = 1
        return series.sort_values(ascending=False).head(top_n)

    def __repr__(self):
        backend = "LightGBM" if self._use_lgbm else "sklearn GBM"
        return f"GradientBoostedModel({backend}, n={self.n_estimators})"
