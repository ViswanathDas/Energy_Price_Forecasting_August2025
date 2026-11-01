"""
Energy Price Forecasting — Main Analysis Script
================================================
Usage:
    cd energy-price-forecasting
    pip install -r requirements.txt
    python notebooks/run_analysis.py

Data source: SMARD.de monthly CSV files (Bundesnetzagentur, CC BY 4.0)
Place the monthly CSVs in data/raw/smard/ and the script will load them all.
Alternatively, point DATA_FILE to a single legacy CSV.

What it does:
  1. Loads 13 months of NL day-ahead prices + generation mix (SMARD)
  2. EDA: seasonality, autocorrelation, price distribution
  3. Builds features (calendar, lags, rolling stats, generation mix)
  4. Trains 3 models: Naive baseline, Linear Regression, Gradient Boosted
  5. Evaluates on held-out test set (last 20% by time)
  6. Generates plots → saves to data/processed/plots/
  7. Prints a clean results table
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_loader import load_smard_folder, load_prices
from src.features import build_feature_matrix, train_test_split_temporal
from src.model import NaiveBaseline, LinearModel, GradientBoostedModel
from src.evaluate import (evaluate_all, plot_actual_vs_predicted,
                          plot_error_distribution, plot_feature_importance,
                          plot_seasonality, plot_autocorrelation)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

# Primary data source: folder of SMARD monthly CSVs
SMARD_FOLDER = "data/raw/smard"

# Legacy fallback: single CSV (used only if SMARD_FOLDER is empty or missing)
DATA_FILE    = "data/raw/day_ahead_prices.csv"
COUNTRY_COL  = None   # e.g. "NL" or None to auto-detect

FORECAST_HORIZON = 24    # hours ahead to forecast (24 = day-ahead)
TEST_FRACTION    = 0.20  # fraction of data used for testing (last 20%)

SAVE_PLOTS = True        # save plots to data/processed/plots/
SHOW_PLOTS = False       # set True if you want interactive windows

# ══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 65)
    print("  Energy Price Forecasting Pipeline")
    print("=" * 65)

    # ── Paths ──────────────────────────────────────────────────────────────────
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / DATA_FILE
    plot_dir = base_dir / "data" / "processed" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def savefig(name):
        return str(plot_dir / name) if SAVE_PLOTS else None

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    smard_dir = base_dir / SMARD_FOLDER
    if smard_dir.exists() and any(smard_dir.glob("*.csv")):
        df = load_smard_folder(smard_dir, freq="h")
    else:
        print("  SMARD folder not found — falling back to legacy CSV")
        df = load_prices(base_dir / DATA_FILE, country_col=COUNTRY_COL, freq="h")

    print(f"\nDataset overview:")
    print(f"  Records  : {len(df):,} hourly observations")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Mean price: {df['price'].mean():.1f} EUR/MWh")
    print(f"  Std dev   : {df['price'].std():.1f} EUR/MWh")
    print(f"  Min / Max : {df['price'].min():.1f} / {df['price'].max():.1f} EUR/MWh")

    # ── 2. EDA Plots ───────────────────────────────────────────────────────────
    print("\n[2/6] Generating EDA plots...")

    fig_season = plot_seasonality(df, save_path=savefig("01_seasonality.png"))
    print("  ✓ Seasonality plot")

    fig_acf = plot_autocorrelation(df, save_path=savefig("02_autocorrelation.png"))
    print("  ✓ Autocorrelation plot")

    # Price distribution + time series overview
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    # Full time series
    df["price"].plot(ax=axes[0], lw=0.5, color="steelblue", alpha=0.7)
    axes[0].set_title("Day-Ahead Electricity Prices")
    axes[0].set_ylabel("Price [EUR/MWh]")
    axes[0].grid(True, alpha=0.25)
    # Distribution
    df["price"].hist(ax=axes[1], bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_title("Price Distribution")
    axes[1].set_xlabel("Price [EUR/MWh]")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.25)
    plt.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(savefig("00_overview.png"), dpi=150, bbox_inches="tight")
    print("  ✓ Overview plot")

    # ── 3. Feature engineering ─────────────────────────────────────────────────
    print(f"\n[3/6] Building features (horizon = {FORECAST_HORIZON}h ahead)...")
    X, y = build_feature_matrix(df, target_col="price", horizon=FORECAST_HORIZON, drop_na=True)
    print(f"  Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"  Features: {', '.join(X.columns[:8])} ...")

    # ── 4. Train/test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, test_fraction=TEST_FRACTION)
    print(f"\n[4/6] Split:")
    print(f"  Train: {len(X_train):,} samples ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"  Test : {len(X_test):,} samples  ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    # ── 5. Train models ────────────────────────────────────────────────────────
    print("\n[5/6] Training models...")

    models = {
        "Naive (lag-24h)": NaiveBaseline(lag=24),
        "Linear Regression": LinearModel(alpha=1.0),
        "Gradient Boosted": GradientBoostedModel(n_estimators=300, learning_rate=0.05),
    }

    predictions_test = {}
    for name, model in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        predictions_test[name] = model.predict(X_test)
        print("done")

    # ── 6. Evaluation ──────────────────────────────────────────────────────────
    print("\n[6/6] Evaluation on test set:")
    results = evaluate_all(y_test.values, predictions_test, baseline_name="Naive (lag-24h)")
    print()
    print(results.to_string())

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating result plots...")

    fig_ts = plot_actual_vs_predicted(
        y_test, predictions_test,
        title=f"Day-Ahead Price Forecast vs Actual ({FORECAST_HORIZON}h horizon)",
        n_days=14,
        save_path=savefig("03_forecast_vs_actual.png"),
    )
    print("  ✓ Forecast vs actual")

    fig_err = plot_error_distribution(
        y_test.values, predictions_test,
        save_path=savefig("04_error_distribution.png"),
    )
    print("  ✓ Error distribution")

    fig_imp = plot_feature_importance(
        models["Gradient Boosted"],
        title="Feature Importance — Gradient Boosted Model",
        save_path=savefig("05_feature_importance.png"),
    )
    print("  ✓ Feature importance")

    if SHOW_PLOTS:
        plt.show()

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Pipeline complete.")
    if SAVE_PLOTS:
        print(f"  Plots saved to: {plot_dir}")
    print()
    best_model = results["MAE [EUR/MWh]"].idxmin()
    best_mae = results.loc[best_model, "MAE [EUR/MWh]"]
    naive_mae = results.loc["Naive (lag-24h)", "MAE [EUR/MWh]"]
    skill = results.loc[best_model, "Skill vs Naive"]
    print(f"  Best model  : {best_model}")
    print(f"  Best MAE    : {best_mae:.2f} EUR/MWh")


if __name__ == "__main__":
    main()
