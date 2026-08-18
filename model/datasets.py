"""Dataset loading and validation.

Adds one capability over a plain read_csv: columns that are stored as numbers but
are really coded nominal levels can be coerced to categorical. The soybean data
codes every observation as a small integer — `precip` is 0, 1 or 2 for below
normal, normal, above normal. Left as numbers, a scaler would treat the gap from
0 to 1 as a real distance and the models would read ordering into labels that
have none.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_FEATURES = 12
MIN_INSTANCES = 500


@dataclass(frozen=True)
class Dataset:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    target_col: str
    class_names: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    origin: str = "local"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    def summary(self) -> dict:
        counts = self.target.value_counts()
        return {
            "instances": int(len(self.frame)),
            "features": int(self.features.shape[1]),
            "classes": self.n_classes,
            "smallest_class": int(counts.min()),
            "largest_class": int(counts.max()),
            "numeric": len(self.numeric_cols),
            "categorical": len(self.categorical_cols),
            "missing_cells": int(self.frame.isna().sum().sum()),
            "rows_with_missing": int(self.frame.isna().any(axis=1).sum()),
            "origin": self.origin,
        }


class DatasetError(ValueError):
    """Raised when a CSV does not satisfy the assignment's constraints."""


def load(csv_path: str,
         target_col: str,
         drop_cols: list[str] | None = None,
         categorical_max_unique: int = 0,
         origin: str = "local",
         rename: dict | None = None) -> Dataset:
    """Read a CSV and return a validated Dataset.

    categorical_max_unique: a numeric column with at most this many distinct
    values is treated as categorical. 0 disables the behaviour.
    """
    if not os.path.exists(csv_path):
        raise DatasetError(f"No such file: {csv_path}")

    frame = pd.read_csv(csv_path)
    # Normalise column names before anything else, so a file taken from the
    # remote source and one taken from the snapshot present an identical schema.
    frame = frame.rename(columns=lambda c: c.replace(".", "_").replace(" ", "_"))
    if rename:
        frame = frame.rename(columns=rename)
    if drop_cols:
        frame = frame.drop(columns=[c for c in drop_cols if c in frame.columns])

    if target_col not in frame.columns:
        raise DatasetError(
            f"Target column {target_col!r} not found. "
            f"Available columns: {list(frame.columns)[:15]}"
        )

    features = frame.drop(columns=[target_col])
    target = frame[target_col].astype(str)

    if features.shape[1] < MIN_FEATURES:
        raise DatasetError(
            f"The assignment requires at least {MIN_FEATURES} features; "
            f"this file has {features.shape[1]}."
        )
    if len(frame) < MIN_INSTANCES:
        raise DatasetError(
            f"The assignment requires at least {MIN_INSTANCES} instances; "
            f"this file has {len(frame)}."
        )
    if target.nunique() < 2:
        raise DatasetError("The target column has fewer than two distinct values.")

    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    if categorical_max_unique > 0:
        coded = [c for c in numeric_cols
                 if features[c].nunique(dropna=True) <= categorical_max_unique]
        numeric_cols = [c for c in numeric_cols if c not in coded]
        categorical_cols = categorical_cols + coded

    smallest = target.value_counts().min()
    if smallest < 2:
        raise DatasetError(
            f"Class {target.value_counts().idxmin()!r} has only {smallest} "
            "instance, so the data cannot be split with stratification."
        )

    return Dataset(
        frame=frame,
        features=features,
        target=target,
        target_col=target_col,
        class_names=sorted(target.unique()),
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        origin=origin,
    )
