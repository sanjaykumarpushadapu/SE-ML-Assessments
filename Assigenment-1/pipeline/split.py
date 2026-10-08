"""Filter 4: stratified 60/20/20 split into train, validation and test sets."""
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def split(X: pd.DataFrame, y: pd.Series, seed: int = config.SEED) -> dict:
    """Return a dict with X_/y_ for train, val and test."""
    # Step 1: keep 20% aside as the test set. stratify=y keeps the same fraud share in each part,
    # which matters because fraud is only about 0.17% of the rows.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=seed)
    # Step 2: split the remaining 80% into 60% train and 20% validation (0.25 of 80% = 20%).
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=0.25, stratify=y_rest, random_state=seed)
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test}
