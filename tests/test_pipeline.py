"""Leakage checks and a regression test for the README results table."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_smard_folder
from src.features import build_feature_matrix, train_test_split_temporal
from src.model import NaiveBaseline, LinearModel
from src.evaluate import mae

ROOT = Path(__file__).resolve().parents[1]
SMARD = ROOT / "data" / "raw" / "smard"


@pytest.fixture(scope="module")
def data():
    if not any(SMARD.glob("*.csv")):
        pytest.skip("SMARD data not present")
    return load_smard_folder(SMARD, freq="h")


def test_target_is_price_24h_ahead(data):
    X, y = build_feature_matrix(data, target_col="price", horizon=24, drop_na=True)
    t = y.index[100]
    assert y.loc[t] == pytest.approx(data["price"].loc[t + pd.Timedelta(hours=24)])


def test_features_do_not_use_future_prices(data):
    """Changing prices after time t must not change the features at time t."""
    X, _ = build_feature_matrix(data, target_col="price", horizon=24, drop_na=False)
    cut = data.index[len(data) // 2]
    tampered = data.copy()
    tampered.loc[tampered.index > cut, "price"] += 1000.0
    X2, _ = build_feature_matrix(tampered, target_col="price", horizon=24, drop_na=False)
    rows = X.index[X.index <= cut][-200:]
    pd.testing.assert_frame_equal(X.loc[rows], X2.loc[rows])


def test_split_is_temporal(data):
    X, y = build_feature_matrix(data, target_col="price", horizon=24, drop_na=True)
    X_tr, X_te, _, _ = train_test_split_temporal(X, y, test_fraction=0.2)
    assert X_tr.index.max() < X_te.index.min()


def test_readme_results_naive_and_linear(data):
    X, y = build_feature_matrix(data, target_col="price", horizon=24, drop_na=True)
    X_tr, X_te, y_tr, y_te = train_test_split_temporal(X, y, test_fraction=0.2)
    naive = NaiveBaseline().fit(X_tr, y_tr)
    lin = LinearModel().fit(X_tr, y_tr)
    mae_naive = mae(y_te.values, naive.predict(X_te))
    mae_lin = mae(y_te.values, lin.predict(X_te))
    assert mae_naive == pytest.approx(29.09, abs=0.05)
    assert mae_lin == pytest.approx(20.52, abs=0.05)
    assert 1 - mae_lin / mae_naive == pytest.approx(0.295, abs=0.005)
