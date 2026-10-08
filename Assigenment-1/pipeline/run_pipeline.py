"""Chains the filters: ingest, clean, features, split, train, evaluate (BR-010).

This file only connects the filters; the real work happens inside each filter.
"""
import json
import logging

import config
from pipeline.clean import clean
from pipeline.evaluate import evaluate_test, evaluate_validation, targets_met
from pipeline.features import build_features
from pipeline.ingest import ingest
from pipeline.split import split
from pipeline.train import CANDIDATES, train

log = logging.getLogger("pipeline")


def pick_winner(fitted: dict) -> str:
    """Prefer the recall margin within the FPR budget, using validation only."""
    feasible = [name for name, (_, _, metrics) in fitted.items()
                if metrics["recall"] >= config.TARGET_RECALL and metrics["fpr"] < config.MAX_FPR]
    preferred = [name for name in feasible
                 if fitted[name][2]["recall"] >= config.VALIDATION_RECALL_TARGET]
    if preferred:
        return min(preferred, key=lambda name: (fitted[name][2]["fpr"], -fitted[name][2]["pr_auc"]))
    if feasible:
        return min(feasible, key=lambda name: (-fitted[name][2]["recall"], fitted[name][2]["fpr"],
                                              -fitted[name][2]["pr_auc"]))
    return min(fitted, key=lambda name: (fitted[name][2]["fpr"], -fitted[name][2]["pr_auc"]))


def run_pipeline(data_path=config.DATA_PATH, candidates=CANDIDATES, track: bool = True,
                 demonstration: bool = True) -> dict:
    """Run every stage, select on validation recall/FPR, then test the frozen choice once."""
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

    # Train every candidate on the training set and judge it on the validation set.
    fitted = {}
    for name in candidates:
        model = train(parts["X_train"], parts["y_train"], name)
        threshold, val_m = evaluate_validation(model, parts["X_val"], parts["y_val"])
        fitted[name] = (model, threshold, val_m)
        log.info("[5/6 train]    %-19s val recall=%.3f precision=%.3f pr_auc=%.4f threshold=%.4f",
                 name, val_m["recall"], val_m["precision"], val_m["pr_auc"], threshold)

    # The winner is chosen from validation results only. Only now is the test set used, once.
    winner = pick_winner(fitted)
    model, threshold, val_m = fitted[winner]
    test_m = evaluate_test(model, parts["X_test"], parts["y_test"], threshold)
    checks = targets_met(test_m)  # real values against the targets, nothing is adjusted
    log.info("[6/6 evaluate] winner=%s test recall=%.3f precision=%.3f f1=%.3f pr_auc=%.4f fpr=%.4f",
             winner, test_m["recall"], test_m["precision"], test_m["f1"], test_m["pr_auc"],
             test_m["fpr"])
    log.info("Minimum acceptance on test set: recall>=%.2f -> %s, fpr<%.2f -> %s", config.MIN_ACCEPTANCE_RECALL,
             checks["recall_ok"], config.MAX_FPR, checks["fpr_ok"])
    log.info("Desired recall goal: recall>=%.2f -> %s", config.TARGET_RECALL,
             test_m["recall"] >= config.TARGET_RECALL)

    result = {"winner": winner, "threshold": threshold, "val": val_m, "test": test_m,
              "targets_met": checks,
              "acceptance_policy": {"revision": config.RECALL_POLICY_REVISION,
                                    "minimum_recall": config.MIN_ACCEPTANCE_RECALL,
                                    "desired_recall": config.TARGET_RECALL},
              "selection_policy": {"validation_recall_target": config.VALIDATION_RECALL_TARGET,
                                   "minimum_recall": config.TARGET_RECALL, "max_fpr": config.MAX_FPR},
              "candidates": {n: {"threshold": t, "val": m} for n, (_, t, m) in fitted.items()}}

    if track:  # track=False lets the tests run the pipeline without MLflow
        from pipeline.registry import log_candidate
        if demonstration and not all(checks.values()):
            log.warning("Demonstration only: model-quality acceptance failed; "
                        "the champion is not an accepted production model.")
        for name, (m, t, vm) in fitted.items():
            is_win = name == winner
            # Every candidate is logged for comparison; only the winner is registered.
            _, version = log_candidate(name, m, parts["X_train"], t, vm,
                                       test_m if is_win else None, register=is_win,
                                       allow_failed_acceptance=demonstration)
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
