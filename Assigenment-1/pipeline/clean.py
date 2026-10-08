"""Filter 2: remove duplicate rows and rows with missing values."""
import pandas as pd

import config


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy; the input frame is not modified."""
    needed = config.FEATURES + [config.TARGET]
    # drop_duplicates and dropna return new frames, so the caller's data stays unchanged.
    out = df.drop_duplicates().dropna(subset=needed).reset_index(drop=True)
    # The label must be exactly 0 or 1, otherwise the data is not what we expect.
    if not set(out[config.TARGET].unique()) <= {0, 1}:
        raise ValueError("Class column must contain only 0 and 1.")
    out[config.TARGET] = out[config.TARGET].astype(int)
    return out
