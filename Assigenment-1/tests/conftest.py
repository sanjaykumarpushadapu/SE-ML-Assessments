"""Shared fixtures. The tiny dataset here is synthetic and used for unit tests only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


@pytest.fixture(scope="session")
def tiny_df() -> pd.DataFrame:
    """Small synthetic table (1000 rows, 50 frauds) used only by the tests, not the Kaggle data."""
    rng = np.random.default_rng(config.SEED)
    n, n_fraud = 1000, 50
    y = np.array([0] * (n - n_fraud) + [1] * n_fraud)
    X = rng.normal(0, 1, size=(n, 28)) + y[:, None] * 1.5
    df = pd.DataFrame(X, columns=config.V_COLS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = rng.exponential(60, n)
    df["Class"] = y
    return df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)


@pytest.fixture(scope="session")
def tiny_parts(tiny_df):
    """Train/validation/test parts of the tiny table, made with the real split filter."""
    from pipeline.features import build_features
    from pipeline.split import split
    X, y = build_features(tiny_df)
    return split(X, y)


@pytest.fixture(scope="session")
def tiny_model(tiny_parts):
    """A logistic regression pipeline fitted on the tiny training part with the real train filter."""
    from pipeline.train import train
    return train(tiny_parts["X_train"], tiny_parts["y_train"], "logistic_regression")


@pytest.fixture()
def valid_payload(tiny_df) -> dict:
    """One valid API request body taken from the tiny table."""
    row = tiny_df[config.FEATURES].iloc[0]
    return {k: float(v) for k, v in row.items()}
