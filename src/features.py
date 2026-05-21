"""
Feature Engineering — Energy Price Forecasting
===============================================
Calendar, lag, and rolling features for electricity price prediction.

Key insight: electricity prices are driven by:
  - Time-of-day (morning/evening peaks)
  - Day-of-week (weekday vs weekend)
  - Seasonality (winter heating, summer cooling)
  - Recent price history (autocorrelation, mean-reversion)
  - Volatility regime (spikes cluster)
  - Generation mix: renewable share suppresses prices (merit-order effect);
    high non-renewable load tends to correlate with elevated prices
"""

import numpy as np
import pandas as pd
from typing import Optional


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based features from the DatetimeIndex.

    Adds: hour, day_of_week, month, quarter, is_weekend,
          hour_sin, hour_cos, month_sin, month_cos (cyclic encoding)
    """
    df = df.copy()
    idx = df.index

    # Raw calendar
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek        # 0=Mon, 6=Sun
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_peak_hour"] = ((idx.hour >= 7) & (idx.hour <= 21)).astype(int)

    # Cyclic encoding (sin/cos) — avoids discontinuity at midnight / year-end
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] = [1, 2, 3, 6, 12, 24, 48, 168],
    col: str = "price",
) -> pd.DataFrame:
    """
    Add lagged price features.

    Default lags (in hours):
      1h, 2h, 3h, 6h, 12h — short-term autocorrelation
      24h — same-hour yesterday
      48h — same-hour two days ago
      168h — same-hour last week
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = [6, 24, 72, 168],
    col: str = "price",
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, max over multiple windows.

    Windows in hours: 6h (quarter-day), 24h (day), 72h (3 days), 168h (week)
    These capture trend and volatility regime.
    """
    df = df.copy()
    for w in windows:
        rolling = df[col].shift(1).rolling(w)  # shift(1) avoids data leakage
        df[f"roll_mean_{w}h"] = rolling.mean()
        df[f"roll_std_{w}h"]  = rolling.std()
        df[f"roll_min_{w}h"]  = rolling.min()
        df[f"roll_max_{w}h"]  = rolling.max()
    return df


def add_price_diff_features(df: pd.DataFrame, col: str = "price") -> pd.DataFrame:
    """
    Add first and second differences + percentage change.
    Captures rate of change and acceleration in prices.
    """
    df = df.copy()
    df["price_diff_1h"] = df[col].diff(1)
    df["price_diff_24h"] = df[col].diff(24)
    df["price_pct_1h"] = df[col].pct_change(1) * 100
    df["price_pct_24h"] = df[col].pct_change(24) * 100
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "price",
    horizon: int = 24,
    drop_na: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build full feature matrix X and target y.

    The target is the price 'horizon' steps ahead (default: 24h ahead).
    All feature engineering is applied; rows with NaN dropped.

    Parameters
    ----------
    df : DataFrame with 'price' column and DatetimeIndex
    horizon : int  Forecast horizon in hours
    drop_na : bool Drop rows with any NaN feature

    Returns
    -------
    X : pd.DataFrame  Feature matrix
    y : pd.Series     Target (future price)
    """
    df = df.copy()

    # Target: price 24h ahead
    df["target"] = df[target_col].shift(-horizon)

    # Add features
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_price_diff_features(df)
    df = add_generation_features(df)   # no-op if generation columns absent

    # Replace inf/-inf (from pct_change crossing zero, renewable_share edge cases)
    df = df.replace([np.inf, -np.inf], np.nan)

    if drop_na:
        df = df.dropna()

    feature_cols = [c for c in df.columns if c not in [target_col, "target"]]
    X = df[feature_cols]
    y = df["target"]

    return X, y


def train_test_split_temporal(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float = 0.2,
) -> tuple:
    """
    Split by time (not random) — critical for time series.
    Uses the last 'test_fraction' of data as the test set.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    n = len(X)
    split = int(n * (1 - test_fraction))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def add_generation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add generation mix features from SMARD data.

    These capture the merit-order effect: when renewables are high
    relative to total load, prices tend to fall (sometimes negative).
    Non-renewable generation proxies for demand-side pressure.

    Expects columns: nuclear, non_renewable, renewable (MW).
    Silently skips if columns are not present.
    """
    gen_cols = {"nuclear", "non_renewable", "renewable"}
    available = gen_cols & set(df.columns)
    if not available:
        return df

    df = df.copy()

    if "renewable" in df.columns and "non_renewable" in df.columns:
        total_gen = df["renewable"].fillna(0) + df["non_renewable"].fillna(0) + df.get("nuclear", pd.Series(0, index=df.index)).fillna(0)
        # Renewable share: high share means price suppression
        df["renewable_share"] = df["renewable"] / total_gen.replace(0, float("nan"))
        # Lag features for generation (same hour yesterday)
        df["renewable_lag24h"]     = df["renewable"].shift(24)
        df["non_renewable_lag24h"] = df["non_renewable"].shift(24)
        # Rolling mean to capture regime (24h window)
        df["renewable_roll24h"]    = df["renewable"].shift(1).rolling(24).mean()

    return df
