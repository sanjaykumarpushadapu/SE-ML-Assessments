"""Filter 6: choose the threshold on validation data and compute the metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

import config


def choose_threshold(y_true, proba, max_fpr: float = config.FPR_BUDGET) -> float:
    """Threshold with the highest recall whose false-positive rate stays within max_fpr.

    The false-positive rate is measured on tens of thousands of genuine rows, so it is a
    stable estimate. Recall is measured on a few hundred frauds and is much noisier. The
    old rule (highest threshold with recall just >= target) sat exactly on the edge of the
    recall target and fell short on new data. Fixing the FPR instead and taking all the
    recall it allows leaves headroom above the recall target whenever the model has it.
    """
    # roc_curve lists every threshold with its FPR and recall (TPR); both fall as the
    # threshold rises. The label is fraud when proba >= threshold, as in compute_metrics.
    fpr, recall, thresholds = roc_curve(y_true, proba)
    ok = np.where(fpr <= max_fpr)[0]
    # Highest recall within the FPR budget; argmax takes the first such point, which is the
    # highest threshold with that recall, i.e. the fewest false alarms for it.
    return float(thresholds[ok[np.argmax(recall[ok])]])


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


def evaluate_validation(y_val, proba) -> tuple[float, dict]:
    """Pick the threshold on validation scores and report validation metrics.

    proba are out-of-fold probabilities (train.cross_val_proba), so no row was scored by a
    model that was trained on it. Whether the recall target is reached is checked when the
    winner is picked.
    """
    threshold = choose_threshold(y_val, proba)
    return threshold, compute_metrics(y_val, proba, threshold)


def evaluate_test(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict:
    """Final report on the test set with the threshold already fixed."""
    # The threshold comes from validation; the test set is never used to choose anything.
    proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, proba, threshold)
    metrics.update(loss_metrics(y_test, proba, threshold, X_test["Amount"]))  # fraud loss in money (G4)
    return metrics
