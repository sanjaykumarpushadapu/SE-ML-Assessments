"""Filter 3: pick the model inputs (X) and the label (y).

Scaling is not done here. It is part of the model pipeline in train.py so that it
is fitted on the training set only (BR-003) and saved with the model (BR-004).
"""
import pandas as pd

import config


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the feature table and the label column."""
    # X holds the 30 input columns as floats; y holds the fraud label.
    return df[config.FEATURES].astype(float), df[config.TARGET]
