"""Filter 6: choose the threshold on validation data and compute the metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

import config


def choose_threshold(y_true, proba, target_recall: float = config.TARGET_RECALL,
                     max_fpr: float | None = None) -> float:
    """Prefer the recall target; with an FPR cap, fall back to the best feasible recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("Target recall must be between 0 and 1.")
    if max_fpr is not None:
        false_positive_rates, recalls, operating_thresholds = roc_curve(y_true, proba, drop_intermediate=False)
        feasible = np.flatnonzero((false_positive_rates < max_fpr)
                                  & (recalls >= config.TARGET_RECALL)
                                  & np.isfinite(operating_thresholds))
        preferred = feasible[recalls[feasible] >= target_recall]
        if len(preferred):
            return float(operating_thresholds[preferred[0]])
        if len(feasible):
            best_recall = recalls[feasible].max()
            return float(operating_thresholds[feasible[recalls[feasible] == best_recall][0]])
    # precision_recall_curve lists every possible threshold with its recall.
    # Recall goes down as the threshold goes up.
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    # All thresholds that still reach the target recall.
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        raise ValueError(f"Recall {target_recall} cannot be reached on this data.")
    # The last one is the highest such threshold, i.e. the fewest false alarms
    # while still meeting the recall target.
    return float(thresholds[ok[-1]])


def compute_metrics(y_true, proba, threshold: float) -> dict:
    """Recall, precision, F1, PR-AUC, ROC-AUC, FPR and the confusion matrix (BR-005)."""
    # The label is fraud when the probability reaches the threshold (BR-001).
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y_true, pred, zero_division=0)),        # frauds caught
        "precision": float(precision_score(y_true, pred, zero_division=0)),  # flagged that are fraud
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),  # threshold-free, good for rare classes
        "roc_auc": float(roc_auc_score(y_true, proba)),
        # false-positive rate = genuine transactions wrongly flagged / all genuine
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def targets_met(metrics: dict) -> dict:
    """Compare measured values with the targets; nothing is tuned here (BR-006)."""
    flags = {"recall_ok": metrics["recall"] >= config.TARGET_RECALL,
             "fpr_ok": metrics["fpr"] < config.MAX_FPR}
    if "loss_reduction" in metrics:  # only the test report has the Amount-based loss figures
        flags["loss_ok"] = metrics["loss_reduction"] >= config.MIN_LOSS_REDUCTION
    return flags


def loss_metrics(y_true, proba, threshold: float, amount) -> dict:
    """Fraud loss measured in money (goal G4), assuming every flagged fraud is stopped.

    Loss = the Amount of the fraud transactions. Without the model every fraud goes through;
    with the model the flagged ones are blocked, so the loss reduction is the share of
    fraud money that is caught. The Amount of genuine transactions that are wrongly blocked
    is reported too, because that is the cost of the false alarms.
    """
    y = np.asarray(y_true)
    flagged = np.asarray(proba) >= threshold
    amt = np.asarray(amount, dtype=float)
    total = float(amt[y == 1].sum())
    caught = float(amt[(y == 1) & flagged].sum())
    return {"fraud_amount_total": total, "fraud_amount_caught": caught,
            "loss_reduction": caught / total if total else 0.0,
            "genuine_amount_blocked": float(amt[(y == 0) & flagged].sum())}


def evaluate_validation(model, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[float, dict]:
    """Pick the threshold on the validation set and report validation metrics."""
    proba = model.predict_proba(X_val)[:, 1]  # column 1 = probability of fraud
    threshold = choose_threshold(y_val, proba, target_recall=config.VALIDATION_RECALL_TARGET,
                                 max_fpr=config.MAX_FPR)
    return threshold, compute_metrics(y_val, proba, threshold)


def evaluate_test(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """Final report on the test set with the threshold already fixed."""
    # The threshold comes from validation; the test set is never used to choose anything.
    proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba, threshold)
    metrics.update(loss_metrics(y_test, proba, threshold, X_test["Amount"]))  # fraud loss in money (G4)
    return metrics
