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
def test_BR002_threshold_meets_target_recall_on_validation(tiny_model, tiny_parts):
    """The chosen threshold must reach the target recall on the validation set."""
    thr, val_m = evaluate.evaluate_validation(tiny_model, tiny_parts["X_val"], tiny_parts["y_val"])
    assert val_m["recall"] >= config.VALIDATION_RECALL_TARGET
    assert val_m["threshold"] == thr


def test_BR002_validation_margin_does_not_change_acceptance_target(monkeypatch, tiny_model, tiny_parts):
    monkeypatch.setattr(config, "VALIDATION_RECALL_TARGET", 0.95)
    _, metrics = evaluate.evaluate_validation(tiny_model, tiny_parts["X_val"], tiny_parts["y_val"])
    assert metrics["recall"] >= config.TARGET_RECALL
    assert config.TARGET_RECALL == 0.90
    assert evaluate.targets_met({"recall": 0.90, "fpr": 0.01})["recall_ok"] is True


def test_BR002_recall_margin_respects_false_positive_limit():
    labels = [1] * 20 + [0] * 100
    probabilities = [0.95] * 18 + [0.55, 0.1] + [0.6] * 10 + [0.01] * 90
    threshold = evaluate.choose_threshold(labels, probabilities, target_recall=0.95, max_fpr=0.02)
    metrics = evaluate.compute_metrics(labels, probabilities, threshold)
    assert metrics["recall"] == 0.90
    assert metrics["fpr"] == 0.0


def test_BR002_false_positive_limit_is_strict():
    labels = [1] * 20 + [0] * 100
    probabilities = [0.95] * 18 + [0.55, 0.1] + [0.6] * 2 + [0.01] * 98
    threshold = evaluate.choose_threshold(labels, probabilities, target_recall=0.95, max_fpr=0.02)
    assert threshold == 0.95


def test_BR002_recall_margin_is_used_when_feasible():
    labels = [1] * 20 + [0] * 100
    probabilities = [0.95] * 18 + [0.55, 0.1] + [0.6] + [0.01] * 99
    threshold = evaluate.choose_threshold(labels, probabilities, target_recall=0.95, max_fpr=0.02)
    metrics = evaluate.compute_metrics(labels, probabilities, threshold)
    assert metrics["recall"] == 0.95
    assert metrics["fpr"] == 0.01


@pytest.mark.parametrize("max_fpr", [None, 0.02])
def test_BR002_unreachable_recall_raises(max_fpr):
    """If the target recall cannot be reached, choose_threshold must raise an error and not guess."""
    with pytest.raises(ValueError):
        evaluate.choose_threshold([0, 1, 1], [0.9, 0.1, 0.2], target_recall=1.01, max_fpr=max_fpr)


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


def test_BR003_boosted_candidate_fits_only_training_data(tiny_parts):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        model = train_mod.train(tiny_parts["X_train"], tiny_parts["y_train"], "hist_gradient_boosting")
        probabilities = model.predict_proba(tiny_parts["X_val"])
    assert model.named_steps["clf"].early_stopping is False
    assert model.named_steps["clf"].class_weight == "balanced"
    assert model.named_steps["prep"].named_transformers_["scale"].center_[1] == pytest.approx(
        tiny_parts["X_train"]["Amount"].median())
    assert probabilities.shape == (len(tiny_parts["X_val"]), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


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


@pytest.mark.parametrize("column,value,message", [
    ("V1", np.inf, "finite"),
    ("V1", -np.inf, "finite"),
    ("V1", "invalid", "numeric"),
    ("Amount", -1.0, "nonnegative"),
    ("Class", 2, "Class"),
])
def test_BR010_clean_rejects_invalid_training_values(tiny_df, column, value, message):
    invalid = tiny_df.copy(deep=True)
    if isinstance(value, str):
        invalid[column] = invalid[column].astype(object)
    invalid.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        clean_mod.clean(invalid)


@pytest.mark.parametrize("label", [0, 1])
def test_BR010_clean_requires_both_classes(tiny_df, label):
    with pytest.raises(ValueError, match="both 0 and 1"):
        clean_mod.clean(tiny_df[tiny_df[config.TARGET] == label])


def test_BR010_clean_rejects_empty_dataset(tiny_df):
    with pytest.raises(ValueError, match="No training rows"):
        clean_mod.clean(tiny_df.iloc[:0])


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


def test_BR011_native_and_docker_artifact_paths_are_separate(monkeypatch, tmp_path):
    import importlib

    try:
        with monkeypatch.context() as settings:
            settings.delenv("MLFLOW_ARTIFACT_DIR", raising=False)
            importlib.reload(config)
            assert config.ARTIFACT_DIR == config.ROOT / "mlruns-native"
            assert config.ARTIFACT_DIR != config.ROOT / "mlruns"
            docker_artifacts = tmp_path / "docker-mlruns"
            settings.setenv("MLFLOW_ARTIFACT_DIR", str(docker_artifacts))
            importlib.reload(config)
            assert config.ARTIFACT_ROOT == docker_artifacts.as_uri()
    finally:
        importlib.reload(config)


def test_BR011_same_seed_gives_same_split_and_results(tiny_df, tmp_path):
    """Running the pipeline twice must give exactly the same test results."""
    path = tmp_path / "c.csv"
    tiny_df.to_csv(path, index=False)
    a = rp.run_pipeline(path, candidates=("logistic_regression",), track=False)
    b = rp.run_pipeline(path, candidates=("logistic_regression",), track=False)
    assert json.dumps(a["test"], sort_keys=True) == json.dumps(b["test"], sort_keys=True)


def test_BR006_winner_is_lowest_validation_fpr_then_pr_auc():
    """The winner has the lowest validation false-positive rate; PR-AUC only breaks ties."""
    fitted = {"a": (None, 0.5, {"recall": 0.95, "fpr": 0.04, "pr_auc": 0.9}),
              "b": (None, 0.5, {"recall": 0.95, "fpr": 0.01, "pr_auc": 0.8}),
              "c": (None, 0.5, {"recall": 0.95, "fpr": 0.01, "pr_auc": 0.7})}
    assert rp.pick_winner(fitted) == "b"


@pytest.mark.parametrize("boosted_recall", [0.93, 0.95])
def test_BR006_winner_prefers_recall_within_false_positive_budget(boosted_recall):
    fitted = {"baseline": (None, 0.5, {"recall": 0.90, "fpr": 0.001, "pr_auc": 0.8}),
              "boosted": (None, 0.1, {"recall": boosted_recall, "fpr": 0.019, "pr_auc": 0.85}),
              "too_many_false_alarms": (None, 0.1, {"recall": 0.99, "fpr": 0.02, "pr_auc": 0.9})}
    assert rp.pick_winner(fitted) == "boosted"


def test_BR006_fraud_loss_reduction_is_share_of_fraud_money_caught():
    """Loss reduction = Amount of caught frauds / Amount of all frauds; genuine blocked money is reported too."""
    m = evaluate.loss_metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5, [100.0, 300.0, 50.0, 70.0])
    assert m["fraud_amount_total"] == 400.0 and m["fraud_amount_caught"] == 100.0
    assert m["loss_reduction"] == 0.25 and m["genuine_amount_blocked"] == 50.0
