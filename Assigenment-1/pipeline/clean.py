"""Filter 2: remove duplicate rows and rows with missing values."""
import numpy as np
import pandas as pd

import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy; the input frame is not modified."""
    needed = config.FEATURES + [config.TARGET]
    # drop_duplicates and dropna return new frames, so the caller's data stays unchanged.
    out = df.drop_duplicates().dropna(subset=needed).reset_index(drop=True)
    if out.empty:
        raise ValueError("No training rows remain after cleaning.")
    if any(not pd.api.types.is_numeric_dtype(out[column])
           or pd.api.types.is_complex_dtype(out[column]) for column in config.FEATURES):
        raise ValueError("Training features must be real numeric values.")
    if not np.isfinite(out[config.FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Training features must be finite.")
    if (out["Amount"] < 0).any():
        raise ValueError("Amount must be nonnegative.")
    # The label must be exactly 0 or 1, otherwise the data is not what we expect.
    if set(out[config.TARGET].unique()) != {0, 1}:
        raise ValueError("Class column must contain both 0 and 1, and no other labels.")
    out[config.TARGET] = out[config.TARGET].astype(int)
    return out
