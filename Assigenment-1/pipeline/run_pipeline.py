"""Chains the filters: ingest, clean, features, split, train, evaluate (BR-010).

This file only connects the filters; the real work happens inside each filter.
"""
import json
import logging

import pandas as pd

import config
from pipeline.clean import clean
from pipeline.evaluate import evaluate_test, evaluate_validation, targets_met
from pipeline.features import build_features
from pipeline.ingest import ingest
from pipeline.split import split
from pipeline.train import CANDIDATES, cross_val_proba, train

log = logging.getLogger("pipeline")


def pick_winner(fitted: dict) -> str:
    """Candidate with the highest validation recall; PR-AUC breaks ties.

    Every candidate's threshold already keeps its validation false-positive rate within
    the FPR budget (goal G2), so recall decides (goal G1). A candidate that does not reach
    the target recall on validation is not eligible; if none does, the pipeline stops
    instead of registering a model that is known to miss the target.
    """
    ok = [n for n in fitted if fitted[n][2]["recall"] >= config.TARGET_RECALL]
    if not ok:
        raise ValueError(f"No candidate reaches recall {config.TARGET_RECALL} with FPR <= "
                         f"{config.FPR_BUDGET} on validation.")
    return max(ok, key=lambda n: (fitted[n][2]["recall"], fitted[n][2]["pr_auc"]))


def run_pipeline(data_path=config.DATA_PATH, candidates=CANDIDATES, track: bool = True) -> dict:
    """Run every stage, pick the candidate with the highest validation recall, test it once."""
    # Each stage hands its output to the next one.
    raw = ingest(data_path)
    log.info("[1/6 ingest]   rows=%d columns=%d", *raw.shape)

    df = clean(raw)
    log.info("[2/6 clean]    rows=%d (removed %d), fraud=%d (%.3f%%)", len(df),
             len(raw) - len(df), df[config.TARGET].sum(), 100 * df[config.TARGET].mean())

    X, y = build_features(df)
    log.info("[3/6 features] X=%s", X.shape)

    parts = split(X, y)
    log.info("[4/6 split]    train=%d val=%d test=%d", len(parts["X_train"]),
             len(parts["X_val"]), len(parts["X_test"]))

    # Training and validation rows together form the development set; the test set stays aside.
    # Each candidate is scored on it with CV_FOLDS-fold cross-validation (out-of-fold, so every row is
    # scored by a model that did not see it) and the threshold is chosen on those scores. The
    # final model is then fitted on the whole development set.
    X_dev = pd.concat([parts["X_train"], parts["X_val"]])
    y_dev = pd.concat([parts["y_train"], parts["y_val"]])
    fitted = {}
    for name in candidates:
        model = train(X_dev, y_dev, name)
        threshold, val_m = evaluate_validation(y_dev, cross_val_proba(X_dev, y_dev, name))
        fitted[name] = (model, threshold, val_m)
        log.info("[5/6 train]    %-19s val recall=%.3f fpr=%.4f precision=%.3f pr_auc=%.4f "
                 "threshold=%.4f", name, val_m["recall"], val_m["fpr"], val_m["precision"],
                 val_m["pr_auc"], threshold)

    # The winner is chosen from validation results only. Only now is the test set used, once.
    winner = pick_winner(fitted)
    model, threshold, val_m = fitted[winner]
    test_m = evaluate_test(model, parts["X_test"], parts["y_test"], threshold)
    checks = targets_met(test_m)  # real values against the targets, nothing is adjusted
    log.info("[6/6 evaluate] winner=%s test recall=%.3f precision=%.3f f1=%.3f pr_auc=%.4f fpr=%.4f",
             winner, test_m["recall"], test_m["precision"], test_m["f1"], test_m["pr_auc"],
             test_m["fpr"])
    log.info("Targets on test set: recall>=%.2f -> %s, fpr<%.2f -> %s", config.TARGET_RECALL,
             checks["recall_ok"], config.MAX_FPR, checks["fpr_ok"])

    result = {"winner": winner, "threshold": threshold, "fpr_budget": config.FPR_BUDGET,
              "val": val_m, "test": test_m,
              "targets_met": checks,
              "candidates": {n: {"threshold": t, "val": m} for n, (_, t, m) in fitted.items()}}

    if track:  # track=False lets the tests run the pipeline without MLflow
        from pipeline.registry import log_candidate
        for name, (m, t, vm) in fitted.items():
            is_win = name == winner
            # Every candidate is logged for comparison; only the winner is registered.
            _, version = log_candidate(name, m, X_dev, t, vm,
                                       test_m if is_win else None, register=is_win)
            if is_win:
                result["model_version"] = version
                log.info("Registered %s version %s with alias '%s'", config.MODEL_NAME,
                         version, config.ALIAS)
        # Save results and one fraud + one genuine test row for the API test calls.
        config.ARTIFACTS_DIR.mkdir(exist_ok=True)
        (config.ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(result, indent=2))
        samples = {
            "fraud": parts["X_test"][parts["y_test"] == 1].iloc[0].to_dict(),
            "genuine": parts["X_test"][parts["y_test"] == 0].iloc[0].to_dict()}
        (config.ARTIFACTS_DIR / "sample_requests.json").write_text(json.dumps(samples))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_pipeline()
