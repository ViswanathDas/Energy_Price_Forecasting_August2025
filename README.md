# Energy Price Forecasting

[![tests](https://github.com/ViswanathDas/Energy_Price_Forecasting_August2025/actions/workflows/tests.yml/badge.svg)](https://github.com/ViswanathDas/Energy_Price_Forecasting_August2025/actions/workflows/tests.yml)

Day-ahead electricity price forecasting for the Dutch (NL) market using machine learning.
Built as a practical engineering portfolio project demonstrating time-series feature engineering,
generation mix modelling, and systematic model evaluation.

## What It Does

- Loads 13 months of NL day-ahead prices plus generation mix (nuclear, non-renewable, renewable) from SMARD.de
- Exploratory analysis: hour-of-day and monthly seasonality, autocorrelation structure, price distribution including negative-price hours
- Feature engineering: cyclic calendar features, lag features, rolling statistics, generation mix features (merit-order effect)
- Three models: naive lag-24h baseline, linear regression, gradient boosted trees
- Evaluation on held-out test set: MAE, RMSE, and skill score vs naive baseline
- Plots saved to `data/processed/plots/`

## Quick Start

```bash
pip install -r requirements.txt

# Place SMARD monthly CSVs in data/raw/smard/
# Download from: https://www.smard.de/en/downloadcenter/download-market-data/
python notebooks/run_analysis.py
```

## Data Source

**SMARD.de** — Bundesnetzagentur, CC BY 4.0
(https://www.smard.de)

Monthly CSV files with hourly data. Each file contains:
- `Date (UTC)` — ISO 8601 timestamp
- `Nuclear`, `Non-Renewable`, `Renewable` — generation mix (MW)
- `Day Ahead Auction (NL)` — NL day-ahead price (EUR/MWh)

The dataset covers September 2024 to September 2025 (13 months, 9,480 hourly observations).
This period contains 602 hours with negative prices — real market events driven by high renewable
generation outpacing demand, particularly in spring and summer.

## Project Structure

```
energy-price-forecasting/
├── notebooks/
│   └── run_analysis.py     main script
├── src/
│   ├── data_loader.py      SMARD loader + legacy ENTSO-E support
│   ├── features.py         calendar, lag, rolling, generation mix features
│   ├── model.py            Naive, Linear, GradientBoosted wrappers
│   └── evaluate.py         metrics and plot functions
├── data/
│   ├── raw/smard/          monthly SMARD CSVs (gitignored)
│   └── processed/plots/    generated plots
└── requirements.txt
```

## Key Engineering Choices

**Temporal train/test split** — data is split by time (not shuffled). Shuffling leaks future
prices into training features and produces misleadingly good results.

**Cyclic encoding** — hour and month encoded as sin/cos pairs so the model treats 23:00 and
00:00 as adjacent, not opposite ends of a range.

**Generation mix features** — renewable share is the dominant driver of the merit-order effect.
High renewable output pushes expensive thermal plants off the margin and suppresses (or inverts)
prices. Adding nuclear, non-renewable, and renewable generation as lagged features gives the model
a physics-based signal beyond calendar patterns alone.

**MAPE not used** — the dataset includes negative and near-zero prices. Percentage error is
undefined or explodes near zero, so MAE and skill score vs naive are the primary metrics here.

**Lag-24h naive baseline** — "tomorrow = same hour today" is a strong benchmark in electricity
markets. Beating it by 30%+ on MAE with simple models is a meaningful result.

## Results

Evaluated on 13 months of NL day-ahead data (Sep 2024 to Sep 2025).
Test set: final 20% by time (approximately Jul to Sep 2025, 1,670 samples).

| Model | MAE [EUR/MWh] | RMSE [EUR/MWh] | Skill vs Naive |
|-------|:---:|:---:|:---:|
| Naive (lag-24h) | 29.09 | 43.68 | 0.000 |
| Linear Regression | 20.52 | 29.13 | 0.295 |
| Gradient Boosted | **20.01** | **28.38** | **0.312** |

The gradient boosted model outperforms linear regression on this dataset — the opposite of the
earlier 7-month ENTSO-E run where linear was slightly better. The reason is straightforward:
with more training data and additional generation mix features, the ensemble has enough signal
to learn non-linear relationships (renewable oversupply events, demand spikes, seasonal
interaction effects). Both models improve on naive by roughly 30% on MAE.

The higher absolute error compared to the earlier run reflects the test period: summer 2025
included extreme price spikes up to 873 EUR/MWh and extended negative-price windows, making
this a harder forecasting problem than a typical winter period.

**What the feature importance plot shows:**
The gradient boosted model ranks lag-24h and lag-168h (same hour last week) as the strongest
features, followed by rolling mean and renewable generation lag. The renewable share feature
appears in the top 10, confirming the merit-order signal is being used.

## Limitations

- The model forecasts each hour 24 hours ahead from the information available at that hour. The real day-ahead auction closes at noon on D-1 for all 24 hours of day D, so a production setup would forecast all hours of the next day from a single cut-off.
- Generation mix features use the actual generation at forecast time, not the published generation forecasts that a trader would use for D+1.
- MAPE is reported but not used to rank models, because prices near or below zero make it explode (373 percent for the naive model on this test set).
- Single train/test split. A rolling-origin backtest would give a more robust comparison.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

The tests check that the target is the price 24 hours ahead, that changing future prices does not change past features (no leakage), that the split is strictly temporal, and that the naive and linear results in the table above are reproduced.

## Data

Source: SMARD.de (Bundesnetzagentur), licensed CC BY 4.0. The monthly CSV files are included in `data/raw/smard/` so the pipeline runs out of the box.

## Earlier Run (ENTSO-E, 7 months)

For comparison, the earlier pipeline run on a shorter ENTSO-E dataset (Sep 2024 to Mar 2025):

| Model | MAE | Skill |
|-------|:---:|:---:|
| Naive | 11.61 | 0.000 |
| Linear Regression | 7.47 | 0.357 |
| Gradient Boosted | 8.03 | 0.308 |

Linear outperformed GBM with limited data — a real example of the bias-variance tradeoff in
practice. With a longer training window and richer features, GBM overtook it.
