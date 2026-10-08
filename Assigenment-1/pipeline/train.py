"""Filter 5: fit a model. The scaler and the classifier are one sklearn Pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import config

CANDIDATES = ("logistic_regression", "random_forest")  # the two models we compare


def make_estimator(name: str, seed: int = config.SEED):
    """Return an unfitted classifier by name (this is the filter we can swap)."""
    if name == "logistic_regression":
        # class_weight="balanced" gives the rare fraud class a larger weight during training.
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        # max_depth and min_samples_leaf limit the tree size so training stays fast and
        # the forest does not memorise single transactions.
        return RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2,
                                      class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed)
    raise ValueError(f"Unknown model name: {name}")


def build_model(name: str) -> Pipeline:
    """Unfitted scaler + classifier pipeline."""
    # RobustScaler uses the median and quartiles, so a few very large Amount values
    # do not distort it. The other columns (V1..V28) are passed through unchanged.
    scale = ColumnTransformer([("scale", RobustScaler(), config.SCALED_COLS)],
                              remainder="passthrough")
    # Scaler + classifier in one object: the API loads this object, so serving applies
    # exactly the same scaling as training (BR-004).
    return Pipeline([("prep", scale), ("clf", make_estimator(name))])


def train(X_train: pd.DataFrame, y_train: pd.Series, name: str) -> Pipeline:
    """Fit scaler and classifier on the training set only and return the pipeline."""
    model = build_model(name)
    model.fit(X_train, y_train)  # the scaler is fitted on training data only (no leakage)
    return model


def cross_val_proba(X: pd.DataFrame, y: pd.Series, name: str,
                    folds: int = config.CV_FOLDS, seed: int = config.SEED):
    """Out-of-fold fraud probabilities: every row is scored by a model that never saw it.

    The scaler and classifier are refitted inside each fold, so there is no leakage (BR-003).
    Scoring all training + validation rows this way gives about four times as many frauds
    to choose the threshold on as the validation set alone.
    """
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return cross_val_predict(build_model(name), X, y, cv=cv, method="predict_proba")[:, 1]
