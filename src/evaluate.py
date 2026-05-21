"""
Evaluation Metrics and Plots — Energy Price Forecasting
========================================================
Standard metrics for electricity price forecasting evaluation.

Key metrics:
  MAE  — Mean Absolute Error [EUR/MWh]  (most intuitive)
  RMSE — Root Mean Squared Error        (penalises spikes)
  MAPE — Mean Absolute Percentage Error (careful with near-zero prices)
  Skill score — vs naive baseline (positive = better than naive)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional


# ── Metrics ───────────────────────────────────────────────────────────────────

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred))**2)))


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    """MAPE clipped at eps to avoid division by near-zero prices."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def skill_score(mae_model: float, mae_baseline: float) -> float:
    """
    Skill score vs baseline. >0 is better than baseline, 1.0 is perfect.
    SS = 1 - (MAE_model / MAE_baseline)
    """
    return 1.0 - (mae_model / mae_baseline)


def evaluate_all(
    y_true,
    predictions: dict,
    baseline_name: str = "Naive",
) -> pd.DataFrame:
    """
    Evaluate multiple models and return results table.

    Parameters
    ----------
    y_true : array-like   True prices
    predictions : dict    {model_name: y_pred_array}
    baseline_name : str   Which model to use as baseline for skill score

    Returns
    -------
    pd.DataFrame  with columns MAE, RMSE, MAPE, Skill vs naive
    """
    rows = []
    mae_naive = mae(y_true, predictions[baseline_name]) if baseline_name in predictions else None

    for name, y_pred in predictions.items():
        m = mae(y_true, y_pred)
        r = rmse(y_true, y_pred)
        mp = mape(y_true, y_pred)
        ss = skill_score(m, mae_naive) if mae_naive is not None and name != baseline_name else float("nan")
        rows.append({"Model": name, "MAE [EUR/MWh]": round(m, 2), "RMSE [EUR/MWh]": round(r, 2),
                     "MAPE [%]": round(mp, 1), "Skill vs Naive": round(ss, 3)})

    df = pd.DataFrame(rows).set_index("Model")
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(
    y_true: pd.Series,
    predictions: dict,
    title: str = "Day-Ahead Price Forecast vs Actual",
    n_days: int = 14,
    save_path: Optional[str] = None,
):
    """
    Line plot of actual prices vs model forecasts over the last n_days of test set.
    """
    n = min(n_days * 24, len(y_true))
    idx = y_true.index[-n:]
    y_true_plot = y_true.iloc[-n:]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(idx, y_true_plot.values, color="black", lw=1.2, label="Actual", zorder=5)

    colours = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for (name, y_pred), colour in zip(predictions.items(), colours):
        ax.plot(idx, np.asarray(y_pred)[-n:], lw=1.0, alpha=0.8, color=colour, label=name)

    ax.set_ylabel("Price [EUR/MWh]")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=30, ha="right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_error_distribution(
    y_true,
    predictions: dict,
    save_path: Optional[str] = None,
):
    """Histogram of forecast errors for each model."""
    y_true = np.asarray(y_true)
    n_models = len(predictions)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        errors = y_true - np.asarray(y_pred)
        ax.hist(errors, bins=60, color="#3498db", alpha=0.75, edgecolor="white")
        ax.axvline(0, color="black", lw=1.0, linestyle="--")
        ax.axvline(errors.mean(), color="crimson", lw=1.2, label=f"Mean: {errors.mean():.1f}")
        ax.set_title(f"{name} — Error Distribution")
        ax.set_xlabel("Error [EUR/MWh]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Count")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(
    model,
    title: str = "Feature Importance",
    top_n: int = 15,
    save_path: Optional[str] = None,
):
    """Horizontal bar chart of feature importance (models that support it)."""
    if not hasattr(model, "feature_importance"):
        print(f"{model} does not support feature_importance()")
        return None

    imp = model.feature_importance(top_n=top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    colours = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp)))[::-1]
    imp.plot(kind="barh", ax=ax, color=colours)
    ax.set_title(title)
    ax.set_xlabel("Relative Importance")
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_seasonality(df: pd.DataFrame, col: str = "price", save_path: Optional[str] = None):
    """
    Three panels: hour-of-day profile, day-of-week profile, month profile.
    Shows the structural seasonality in electricity prices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Hour of day
    hourly = df.groupby(df.index.hour)[col].agg(["mean", "std"])
    h_idx = np.arange(len(hourly))
    h_mean = hourly["mean"].values
    h_std = hourly["std"].values
    axes[0].fill_between(h_idx, h_mean - h_std, h_mean + h_std, alpha=0.2, color="steelblue")
    axes[0].plot(h_idx, h_mean, color="steelblue", lw=1.5)
    axes[0].set_title("Hour-of-Day Profile")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Avg Price [EUR/MWh]")
    axes[0].set_xticks(range(0, len(hourly), 4))
    axes[0].set_xticklabels(hourly.index[::4].tolist())
    axes[0].grid(True, alpha=0.25)

    # Day of week
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    daily = df.groupby(df.index.dayofweek)[col].mean()
    colours = ["#3498db"] * 5 + ["#e74c3c"] * 2
    axes[1].bar(dow_labels, daily.values, color=colours, edgecolor="white")
    axes[1].set_title("Day-of-Week Profile")
    axes[1].set_ylabel("Avg Price [EUR/MWh]")
    axes[1].grid(True, axis="y", alpha=0.25)

    # Month of year — use only months present in the dataset
    monthly = df.groupby(df.index.month)[col].agg(["mean", "std"])
    month_labels_all = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    present_months = monthly.index.tolist()
    present_labels = [month_labels_all[m - 1] for m in present_months]
    p_idx = np.arange(len(present_months))
    m_mean = monthly["mean"].values
    m_std = monthly["std"].values
    axes[2].fill_between(p_idx, m_mean - m_std, m_mean + m_std, alpha=0.2, color="darkorange")
    axes[2].plot(p_idx, m_mean, color="darkorange", lw=1.5, marker="o", ms=4)
    axes[2].set_title("Month-of-Year Profile")
    axes[2].set_xticks(p_idx)
    axes[2].set_xticklabels(present_labels, rotation=30, ha="right", fontsize=8)
    axes[2].set_ylabel("Avg Price [EUR/MWh]")
    axes[2].grid(True, alpha=0.25)

    plt.suptitle("Electricity Price Seasonality", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_autocorrelation(df: pd.DataFrame, col: str = "price", lags: int = 168, save_path=None):
    """
    ACF plot to visualise autocorrelation structure.
    168 hours = 1 week — shows daily (lag-24) and weekly (lag-168) structure.
    """
    from pandas.plotting import autocorrelation_plot
    series = df[col].dropna()
    acf_vals = [series.autocorr(lag=i) for i in range(1, lags + 1)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(1, lags + 1), acf_vals, color="steelblue", alpha=0.6, width=1.0)
    ax.axhline(0, color="black", lw=0.8)
    # Confidence band (95%)
    conf = 1.96 / np.sqrt(len(series))
    ax.axhline(conf, color="crimson", lw=0.8, linestyle="--", label="95% CI")
    ax.axhline(-conf, color="crimson", lw=0.8, linestyle="--")
    # Mark key lags
    for lag in [24, 48, 72, 168]:
        if lag <= lags:
            ax.axvline(lag, color="orange", lw=0.8, alpha=0.5)
    ax.set_xlabel("Lag [hours]")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Electricity Price Autocorrelation (up to 1 week)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
