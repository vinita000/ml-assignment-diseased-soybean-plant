"""Modelling package for ML Assignment 2.

    sources.py      remote acquisition, integrity checking, offline fallback
    datasets.py     loading and constraint validation
    classifiers.py  preprocessing pipeline and the five estimators
    evaluation.py   cross-validated metrics
"""

from . import classifiers, datasets, evaluation, sources

__all__ = ["sources", "datasets", "classifiers", "evaluation"]
