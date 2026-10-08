"""Filter 1: read the raw CSV and check that the expected columns exist."""
from pathlib import Path

import pandas as pd

import config


def ingest(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Return the raw transactions; fail loudly if the file or columns are missing."""
    path = Path(path)
    # Stop with a clear message instead of continuing with made-up or empty data.
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Upload creditcard.csv first.")
    df = pd.read_csv(path)
    # A wrong or damaged file should fail here, not later inside the model.
    missing = set(config.FEATURES + [config.TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df
