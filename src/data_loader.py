"""
Data Loader — Energy Price Forecasting
=======================================
Loads and cleans day-ahead electricity price data.

Supported formats (auto-detected):
  1. SMARD.de monthly CSV export (Kaggle dataset)
     3-row header: attribution / column names / units
     Columns: 'Date (UTC)', Nuclear, Non-Renewable, Renewable, 'Day Ahead Auction (NL)'
     Use load_smard_folder() to load all monthly files at once.
  2. ENTSO-E Transparency Platform CSV export
     Columns: 'DateTime', 'Price' (EUR/MWh)
  3. Generic two-column CSV: timestamp + price
  4. ENTSO-E with country column: 'DateTime', 'NL', 'DE', ...

Usage
-----
    from src.data_loader import load_prices

    df = load_prices("data/raw/day_ahead_prices.csv")
    # Returns DataFrame with DatetimeIndex and column 'price'
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import warnings


# ── column name aliases (auto-detection) ──────────────────────────────────────
_TIME_ALIASES = [
    "datetime", "date", "time", "timestamp", "period",
    "mtu", "mtu (utc)", "mtu (cet/cest)",
]
_PRICE_ALIASES = [
    "price", "eur/mwh", "day-ahead price [eur/mwh]",
    "nl", "germany", "de", "price (eur/mwh)", "day ahead price",
    "day-ahead auction prices in eur/mwhe", "value",
]


def _find_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Return the first column whose lowercase name matches an alias."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower_cols:
            return lower_cols[alias]
    return None


def load_prices(
    filepath: str | Path,
    country_col: Optional[str] = None,
    freq: str = "h",
    fill_method: str = "ffill",
) -> pd.DataFrame:
    """
    Load electricity price CSV into a clean DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.
    country_col : str, optional
        If the CSV has multiple country columns, pick this one (e.g. 'NL').
        If None, auto-detects the first numeric column after the timestamp.
    freq : str
        Desired output frequency ('h' = hourly, '15min', 'D' etc.).
        Data is resampled to this frequency if needed.
    fill_method : str
        How to fill missing values: 'ffill', 'bfill', 'interpolate', or 'drop'.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (UTC), single column 'price' [EUR/MWh].
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Put your CSV in data/raw/ and update the path in run_analysis.py"
        )

    # ── Read file ──────────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(filepath, sep=None, engine="python")
    except Exception as e:
        raise ValueError(f"Could not read {filepath}: {e}")

    df_raw.columns = df_raw.columns.str.strip()

    # ── Find timestamp column ──────────────────────────────────────────────────
    time_col = _find_column(df_raw, _TIME_ALIASES)
    if time_col is None:
        # Try first column
        time_col = df_raw.columns[0]
        warnings.warn(f"Timestamp column not found by name — using first column: '{time_col}'")

    # ── Parse timestamps ───────────────────────────────────────────────────────
    try:
        # ENTSO-E often uses 'DD.MM.YYYY HH:MM' or 'YYYY-MM-DD HH:MM'
        # infer_datetime_format was deprecated in pandas 2.0; pd.to_datetime
        # now infers formats automatically without that argument.
        timestamps = pd.to_datetime(df_raw[time_col], dayfirst=False)
    except Exception:
        timestamps = pd.to_datetime(df_raw[time_col], dayfirst=True)

    # ── Find price column ──────────────────────────────────────────────────────
    if country_col is not None:
        price_col = country_col
    else:
        price_col = _find_column(df_raw, _PRICE_ALIASES)
        if price_col is None:
            # Use first numeric column that isn't the timestamp
            numeric_cols = [c for c in df_raw.columns if c != time_col
                            and pd.api.types.is_numeric_dtype(df_raw[c])]
            if not numeric_cols:
                # Try to coerce
                for c in df_raw.columns:
                    if c != time_col:
                        try:
                            pd.to_numeric(df_raw[c].str.replace(",", "."), errors="raise")
                            numeric_cols.append(c)
                        except Exception:
                            pass
            if not numeric_cols:
                raise ValueError(
                    f"Could not find a numeric price column in {filepath}.\n"
                    f"Columns found: {list(df_raw.columns)}\n"
                    f"Set 'country_col' to the correct column name."
                )
            price_col = numeric_cols[0]
            warnings.warn(f"Price column auto-detected as: '{price_col}'")

    # ── Clean price values ─────────────────────────────────────────────────────
    prices_raw = df_raw[price_col].copy()
    if prices_raw.dtype == object:
        prices_raw = prices_raw.str.replace(",", ".").str.replace(" ", "")
    prices = pd.to_numeric(prices_raw, errors="coerce")

    # ── Build clean DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({"price": prices.values}, index=timestamps)
    df.index.name = "datetime"
    df = df.sort_index()

    # Remove timezone if mixed, localise to UTC
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC")
    else:
        df.index = df.index.tz_localize("UTC", ambiguous="infer", nonexistent="shift_forward")

    # ── Resample to target frequency ───────────────────────────────────────────
    df = df.resample(freq).mean()

    # ── Handle missing values ──────────────────────────────────────────────────
    n_missing = df["price"].isna().sum()
    if n_missing > 0:
        print(f"  Missing values: {n_missing} ({100*n_missing/len(df):.1f}%)")
        if fill_method == "ffill":
            df["price"] = df["price"].ffill()
        elif fill_method == "bfill":
            df["price"] = df["price"].bfill()
        elif fill_method == "interpolate":
            df["price"] = df["price"].interpolate(method="time")
        elif fill_method == "drop":
            df = df.dropna()
        else:
            raise ValueError(
                f"Unknown fill_method: '{fill_method}'. "
                f"Choose from: 'ffill', 'bfill', 'interpolate', 'drop'."
            )

    print(f"  Loaded {len(df):,} records from {df.index[0]} to {df.index[-1]}")
    print(f"  Price range: {df['price'].min():.1f} – {df['price'].max():.1f} EUR/MWh")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SMARD.de loader (Kaggle dataset — monthly CSV files)
# ══════════════════════════════════════════════════════════════════════════════

# ======================================================================
# SMARD.de loader (Kaggle dataset, CC BY 4.0, Bundesnetzagentur)
# ======================================================================

def load_smard_file(filepath) -> "pd.DataFrame":
    """
    Load a single SMARD.de monthly CSV file.

    Format: 3-row header (attribution / column names / units), then hourly data.
    Columns: Date (UTC), Nuclear, Non-Renewable, Renewable, Day Ahead Auction (NL)

    Returns DataFrame with DatetimeIndex (UTC) and columns:
      price, nuclear, non_renewable, renewable
    """
    filepath = Path(filepath)

    df_raw = pd.read_csv(
        filepath,
        skiprows=[0, 2],
        encoding="utf-8-sig",
        sep=",",
    )
    df_raw.columns = df_raw.columns.str.strip()

    timestamps = pd.to_datetime(df_raw["Date (UTC)"], utc=True)

    price_col = next(
        (c for c in df_raw.columns if "Day Ahead Auction" in c),
        None,
    )
    if price_col is None:
        raise ValueError(
            f"Could not find price column in {filepath.name}. "
            f"Columns: {list(df_raw.columns)}"
        )

    def _col(name):
        col = df_raw.get(name)
        if col is None:
            return np.full(len(timestamps), np.nan)
        return pd.to_numeric(col, errors="coerce").values

    df = pd.DataFrame(
        {
            "price":         _col(price_col),
            "nuclear":       _col("Nuclear"),
            "non_renewable": _col("Non-Renewable"),
            "renewable":     _col("Renewable"),
        },
        index=timestamps,
    )
    df.index.name = "datetime"
    return df.sort_index()


def load_smard_folder(folder, freq="h", fill_method="ffill") -> "pd.DataFrame":
    """
    Load all SMARD monthly CSV files from a folder and combine into one DataFrame.

    Parameters
    ----------
    folder      : Path to directory containing monthly CSV files.
    freq        : Resample target (default 'h').
    fill_method : 'ffill', 'bfill', 'interpolate', or 'drop'.

    Returns
    -------
    pd.DataFrame with DatetimeIndex (UTC): price, nuclear, non_renewable, renewable
    """
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    print(f"  Found {len(csv_files)} monthly CSV files in {folder.name}/")

    frames = []
    for f in csv_files:
        try:
            chunk = load_smard_file(f)
            frames.append(chunk)
        except Exception as e:
            warnings.warn(f"  Skipped {f.name}: {e}")

    if not frames:
        raise ValueError("No files could be loaded from the SMARD folder.")

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    df = df.resample(freq).mean()

    n_missing = df["price"].isna().sum()
    if n_missing > 0:
        print(f"  Missing price values: {n_missing} ({100*n_missing/len(df):.1f}%)")
        if fill_method == "ffill":
            df = df.ffill()
        elif fill_method == "bfill":
            df = df.bfill()
        elif fill_method == "interpolate":
            df = df.interpolate(method="time")
        elif fill_method == "drop":
            df = df.dropna()

    print(f"  Loaded {len(df):,} hourly records from {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Price range: {df['price'].min():.1f} to {df['price'].max():.1f} EUR/MWh")
    print(f"  Columns: {list(df.columns)}")

    return df
