"""
Performance evaluation on the held-out test set (Step 7 of the pipeline).

Accuracy is recorded for completeness but is deliberately *not* used to draw
conclusions: on imbalanced data a classifier can score highly simply by
predicting the majority class. The metrics that drive the analysis are those
focused on the minority (phishing) class:

* **Recall**    - proportion of phishing sites detected (false negatives are costly)
* **Precision** - proportion of phishing predictions that are correct
* **F1-score**  - harmonic mean of the two
* **ROC-AUC**   - overall separability, threshold-free
* **PR-AUC**    - precision-recall area, the more informative AUC under imbalance
* **MCC**       - balanced single-figure summary that is robust to skew
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "mcc",
    "balanced_accuracy",
    "accuracy",
]


def evaluate(y_true, y_pred, y_scores=None) -> dict:
    """Compute the full metric set for one experimental configuration.

    ``y_scores`` are continuous scores for the positive class. When omitted the
    threshold-free metrics are returned as ``NaN`` rather than being silently
    approximated from hard labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if y_scores is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_scores)
        results["pr_auc"] = average_precision_score(y_true, y_scores)
    else:
        results["roc_auc"] = float("nan")
        results["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    results.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    return results


def format_metrics(results: dict, decimals: int = 4) -> dict:
    """Round metric values for presentation in tables."""
    return {
        k: (round(v, decimals) if isinstance(v, float) else v)
        for k, v in results.items()
    }
