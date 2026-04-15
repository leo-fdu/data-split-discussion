from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


METRIC_NAMES = ["roc_auc", "pr_auc", "accuracy", "mae", "rmse", "r2"]


def empty_metrics() -> dict[str, float]:
    return {name: np.nan for name in METRIC_NAMES}


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    context: str,
    positive_label: int | float,
) -> dict[str, float]:
    metrics = empty_metrics()
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        warnings.warn(
            f"{context}: only one class present in y_true; returning NaN for ROC-AUC and PR-AUC.",
            stacklevel=2,
        )
        return metrics

    binary_true = (y_true == positive_label).astype(int)
    metrics["roc_auc"] = float(roc_auc_score(binary_true, y_score))
    metrics["pr_auc"] = float(average_precision_score(binary_true, y_score))
    return metrics


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context: str,
) -> dict[str, float]:
    metrics = empty_metrics()
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    try:
        metrics["r2"] = float(r2_score(y_true, y_pred))
    except ValueError:
        warnings.warn(
            f"{context}: R2 is undefined for this split; returning NaN.",
            stacklevel=2,
        )
        metrics["r2"] = np.nan

    return metrics
