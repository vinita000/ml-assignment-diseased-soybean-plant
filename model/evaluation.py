"""Cross-validated evaluation.

This is the main methodological difference from a single hold-out split. Every
model is scored with repeated stratified k-fold cross-validation — by default
5 folds repeated 3 times, so 15 fits per model — and reported as mean plus or
minus standard deviation across those runs.

The reason is that a single 80/20 split on a dataset this size puts about 114
rows in the test set. Shifting the random seed can move accuracy by several
percentage points, which is enough to reorder the models entirely. Reporting the
spread alongside the mean makes it visible when two models are genuinely tied
rather than separated by split luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, cross_val_predict,
                                     StratifiedKFold)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def _score_fold(y_true, y_pred, y_proba, is_binary: bool) -> dict:
    """The six required metrics for one fold."""
    average = "binary" if is_binary else "weighted"
    try:
        if is_binary:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                average="weighted",
                                labels=np.arange(y_proba.shape[1]))
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=average,
                                     zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def cross_validate_model(pipeline, X, y, is_binary: bool,
                         n_splits: int = 5, n_repeats: int = 3,
                         random_state: int = 42) -> pd.DataFrame:
    """Score one pipeline across every fold. Returns one row per fold."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipeline.fit(X_tr, y_tr)
        rows.append({
            "fold": fold,
            **_score_fold(y_te, pipeline.predict(X_te),
                          pipeline.predict_proba(X_te), is_binary),
        })
    return pd.DataFrame(rows)


def evaluate_all(pipelines: dict, X, y, is_binary: bool,
                 n_splits: int = 5, n_repeats: int = 3,
                 random_state: int = 42):
    """Cross-validate every pipeline.

    Returns (summary, per_fold) where summary has one row per model with mean and
    standard deviation for each metric, and per_fold holds the raw fold scores for
    plotting the spread.
    """
    summary_rows, per_fold_frames = [], []

    for name, pipeline in pipelines.items():
        folds = cross_validate_model(pipeline, X, y, is_binary,
                                     n_splits, n_repeats, random_state)
        folds.insert(0, "model", name)
        per_fold_frames.append(folds)

        row = {"ML Model Name": name}
        for metric in METRIC_NAMES:
            row[metric] = folds[metric].mean()
            row[f"{metric} SD"] = folds[metric].std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("ML Model Name")
    per_fold = pd.concat(per_fold_frames, ignore_index=True)
    return summary, per_fold


def pooled_confusion(pipeline, X, y, class_names: list[str],
                     n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """Confusion matrix pooled over out-of-fold predictions.

    Every row gets predicted exactly once, by a model that did not see it during
    training, so the matrix covers the whole dataset instead of one 20% slice.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    y_pred = cross_val_predict(pipeline, X, y, cv=splitter)
    return confusion_matrix(y, y_pred, labels=list(range(len(class_names))))


def format_summary(summary: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Collapse mean and SD into 'mean ± sd' strings for display."""
    out = pd.DataFrame(index=summary.index)
    for metric in METRIC_NAMES:
        out[metric] = [
            f"{m:.{decimals}f} ± {s:.{decimals}f}"
            for m, s in zip(summary[metric], summary[f"{metric} SD"])
        ]
    return out
