"""Tests for the pipeline filters and the rules BR-002 to BR-006, BR-010 and BR-011."""
import inspect
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import config
from pipeline import clean as clean_mod, evaluate, features, ingest as ingest_mod, split as split_mod
from pipeline import train as train_mod
from pipeline import run_pipeline as rp


# ---- BR-002 threshold chosen on validation ---------------------------------------------
def test_BR002_threshold_meets_targets_on_validation(tiny_parts):
    """On out-of-fold validation scores the threshold keeps FPR within budget and reaches the target recall."""
    proba = train_mod.cross_val_proba(tiny_parts["X_train"], tiny_parts["y_train"],
                                      "logistic_regression")
    thr, val_m = evaluate.evaluate_validation(tiny_parts["y_train"], proba)
    assert val_m["fpr"] <= config.FPR_BUDGET
    assert val_m["recall"] >= config.TARGET_RECALL
    assert val_m["threshold"] == thr


def test_BR002_threshold_is_highest_recall_within_fpr_budget():
    """The threshold takes the most recall the FPR budget allows, with the fewest false alarms for it."""
    y = [0] * 10 + [1] * 4
    proba = [0.1] * 8 + [0.95, 0.5] + [0.9, 0.8, 0.7, 0.05]
    # max_fpr 0.1 allows one false alarm (0.95): recall 3/4 is reached at threshold 0.7.
    assert evaluate.choose_threshold(y, proba, max_fpr=0.1) == 0.7
    # max_fpr 0.2 would allow the 0.5 alarm too, but that adds no recall, so 0.7 stays.
    assert evaluate.choose_threshold(y, proba, max_fpr=0.2) == 0.7
    # With no FPR limit every fraud is caught.
    assert evaluate.choose_threshold(y, proba, max_fpr=1.0) == 0.05


def test_BR002_unreachable_recall_raises():
    """If no candidate reaches the target recall on validation, the pipeline must stop and not guess."""
    fitted = {"a": (None, 0.5, {"recall": 0.80, "fpr": 0.01, "pr_auc": 0.9})}
    with pytest.raises(ValueError):
        rp.pick_winner(fitted)


def test_BR002_threshold_not_hard_coded_in_api():
    """The API must read the threshold from MLflow and must not contain its own fixed value."""
    import api.main as api_main
    src = inspect.getsource(api_main)
    assert "params[\"threshold\"]" in src and ">= 0." not in src


# ---- BR-003 no leakage -----------------------------------------------------------------
def test_BR003_splits_are_disjoint_and_stratified(tiny_parts):
    """Train, validation and test must not share rows and must have the same fraud share."""
    idx = [set(tiny_parts[f"X_{k}"].index) for k in ("train", "val", "test")]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    rates = [tiny_parts[f"y_{k}"].mean() for k in ("train", "val", "test")]
    assert max(rates) - min(rates) < 0.01
    assert sum(len(i) for i in idx) == 1000


def test_BR003_scaler_fitted_on_train_only(tiny_model, tiny_parts, tiny_df):
    """The scaler must use the training median of Amount, not the median of all data (no leakage)."""
    scaler = tiny_model.named_steps["prep"].named_transformers_["scale"]
    train_median = tiny_parts["X_train"]["Amount"].median()
    all_median = tiny_df["Amount"].median()
    assert scaler.center_[1] == pytest.approx(train_median)
    assert train_median != pytest.approx(all_median)


# ---- BR-004 same preprocessing in training and serving ---------------------------------
def test_BR004_preprocessing_saved_inside_model(tiny_model, tiny_parts, tmp_path):
    """The scaler is saved inside the model, and a saved and reloaded model gives identical results."""
    assert "prep" in tiny_model.named_steps
    path = tmp_path / "m.pkl"
    joblib.dump(tiny_model, path)
    reloaded = joblib.load(path)
    X = tiny_parts["X_test"]
    np.testing.assert_allclose(reloaded.predict_proba(X), tiny_model.predict_proba(X))


# ---- BR-005 / BR-006 metrics and honest targets ----------------------------------------
def test_BR005_metrics_include_required_measures():
    """Metrics must include recall, precision, F1, PR-AUC and the confusion matrix, and not accuracy."""
    m = evaluate.compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    for key in ("recall", "precision", "f1", "pr_auc", "tn", "fp", "fn", "tp"):
        assert key in m
    assert "accuracy" not in m
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)


def test_BR006_targets_flag_is_false_when_target_missed():
    """When the targets are missed, the result must say so (False) instead of hiding it."""
    bad = evaluate.compute_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    flags = evaluate.targets_met(bad)
    assert flags["recall_ok"] is False and flags["fpr_ok"] is False


def test_BR006_test_set_not_used_for_threshold():
    """The threshold is chosen on validation data first; the test set is used only afterwards."""
    src = inspect.getsource(rp.run_pipeline)
    assert src.index("evaluate_validation") < src.index("evaluate_test")
    assert "choose_threshold" not in src


# ---- BR-010 pipe-and-filter ------------------------------------------------------------
def test_BR010_filters_do_not_mutate_their_input(tiny_df):
    """Filters must return new data and leave the input table unchanged."""
    before = tiny_df.copy(deep=True)
    clean_mod.clean(tiny_df)
    features.build_features(tiny_df)
    pd.testing.assert_frame_equal(tiny_df, before)


def test_BR010_each_filter_is_a_separate_function():
    """Each filter is its own function in its own module."""
    for fn in (ingest_mod.ingest, clean_mod.clean, features.build_features, split_mod.split,
               train_mod.train, evaluate.evaluate_validation, evaluate.evaluate_test):
        assert callable(fn)
    assert len({f.__module__ for f in (ingest_mod.ingest, clean_mod.clean, split_mod.split)}) == 3


def test_BR010_run_pipeline_chains_filters_in_order(monkeypatch, tiny_df, tmp_path):
    """run_pipeline must call the filters in the order ingest, clean, features, split, train, evaluate."""
    calls = []

    def spy(name, fn):
        def wrapper(*a, **k):
            calls.append(name)
            return fn(*a, **k)
        return wrapper

    for name in ("ingest", "clean", "build_features", "split", "train", "evaluate_validation",
                 "evaluate_test"):
        monkeypatch.setattr(rp, name, spy(name, getattr(rp, name)))
    path = tmp_path / "creditcard.csv"
    tiny_df.to_csv(path, index=False)
    rp.run_pipeline(path, candidates=("logistic_regression",), track=False)
    order = ["ingest", "clean", "build_features", "split", "train", "evaluate_validation",
             "evaluate_test"]
    assert calls == order


def test_BR010_ingest_fails_loudly(tmp_path):
    """ingest must raise an error for a missing file and for a file with wrong columns."""
    with pytest.raises(FileNotFoundError):
        ingest_mod.ingest(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        ingest_mod.ingest(bad)


# ---- BR-011 seeds ----------------------------------------------------------------------
def test_BR011_seed_is_42_everywhere():
    """The configured seed and the models' random_state must be 42."""
    assert config.SEED == 42
    for name in train_mod.CANDIDATES:
        assert train_mod.make_estimator(name).random_state == 42


def test_BR011_same_seed_gives_same_split_and_results(tiny_df, tmp_path):
    """Running the pipeline twice must give exactly the same test results."""
    path = tmp_path / "c.csv"
    tiny_df.to_csv(path, index=False)
    a = rp.run_pipeline(path, candidates=("logistic_regression",), track=False)
    b = rp.run_pipeline(path, candidates=("logistic_regression",), track=False)
    assert json.dumps(a["test"], sort_keys=True) == json.dumps(b["test"], sort_keys=True)


def test_BR006_winner_is_highest_validation_recall_then_pr_auc():
    """The winner has the highest validation recall; PR-AUC breaks ties; below-target candidates are skipped."""
    fitted = {"a": (None, 0.5, {"recall": 0.85, "fpr": 0.010, "pr_auc": 0.99}),
              "b": (None, 0.5, {"recall": 0.93, "fpr": 0.015, "pr_auc": 0.8}),
              "c": (None, 0.5, {"recall": 0.93, "fpr": 0.014, "pr_auc": 0.7}),
              "d": (None, 0.5, {"recall": 0.91, "fpr": 0.012, "pr_auc": 0.9})}
    assert rp.pick_winner(fitted) == "b"


def test_BR006_fraud_loss_reduction_is_share_of_fraud_money_caught():
    """Loss reduction = Amount of caught frauds / Amount of all frauds; genuine blocked money is reported too."""
    m = evaluate.loss_metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5, [100.0, 300.0, 50.0, 70.0])
    assert m["fraud_amount_total"] == 400.0 and m["fraud_amount_caught"] == 100.0
    assert m["loss_reduction"] == 0.25 and m["genuine_amount_blocked"] == 50.0
