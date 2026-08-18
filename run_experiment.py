"""run_experiment.py — cross-validate every model and print the comparison table.

    python run_experiment.py
    python run_experiment.py --csv data/other.csv --target label --folds 10

Nothing is pickled. The Streamlit app rebuilds the models from source at startup,
which keeps the repository free of binary artifacts and removes the version-skew
failure that happens when a model pickled by one scikit-learn release is loaded by
another. On a dataset this size a full refit takes a couple of seconds.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from model import classifiers, datasets, evaluation, sources

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-validate the assignment's five classifiers.")
    parser.add_argument("--csv", default=None,
                        help="local CSV; omit to fetch the configured remote source")
    parser.add_argument("--target", default=None)
    parser.add_argument("--offline", action="store_true",
                        help="skip the network and use the committed snapshot")
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache and re-download")
    parser.add_argument("--categorical-max-unique", type=int, default=10)
    parser.add_argument("--drop", nargs="*", default=[])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--holdout", type=float, default=0.2,
                        help="fraction reserved as test_data.csv for the app")
    args = parser.parse_args()

    # ---------------- acquire ----------------
    source = sources.DEFAULT_SOURCE
    if args.csv:
        csv_path, origin = args.csv, "local"
        target_col = args.target or source.target_col
    else:
        print(f"Source: {source.name}\n  {source.url}")
        csv_path, origin = sources.acquire(source,
                                           allow_network=not args.offline,
                                           force_refresh=args.refresh)
        target_col = args.target or source.target_col
        print(f"  {sources.describe_origin(origin)}\n")

    try:
        ds = datasets.load(csv_path, target_col, args.drop,
                           categorical_max_unique=args.categorical_max_unique,
                           origin=origin, rename=source.rename)
    except datasets.DatasetError as exc:
        raise SystemExit(f"Dataset problem: {exc}")

    info = ds.summary()
    print(f"Loaded {csv_path}")
    for key, value in info.items():
        print(f"  {key:<14}{value}")
    print()

    encoder = LabelEncoder()
    y = encoder.fit_transform(ds.target)
    class_names = [str(c) for c in encoder.classes_]

    # ---------------- hold out a slice for the app ----------------
    from sklearn.model_selection import train_test_split
    idx_train, idx_test = train_test_split(
        np.arange(len(ds.frame)), test_size=args.holdout,
        random_state=classifiers.RANDOM_STATE, stratify=y)
    ds.frame.iloc[idx_test].to_csv(os.path.join(HERE, "test_data.csv"),
                                   index=False)
    print(f"wrote test_data.csv ({len(idx_test)} rows held out for the app)\n")

    # ---------------- cross-validate ----------------
    pipelines = classifiers.build_all(ds.numeric_cols, ds.categorical_cols,
                                     ds.n_classes, info['smallest_class'])
    total_fits = args.folds * args.repeats
    print(f"Cross-validating {len(pipelines)} models "
          f"({args.folds} folds x {args.repeats} repeats = {total_fits} fits each)\n")

    summary, per_fold = evaluation.evaluate_all(
        pipelines, ds.features, y, ds.is_binary, args.folds, args.repeats,
        classifiers.RANDOM_STATE)

    display = evaluation.format_summary(summary)
    print("=" * 100)
    print(f"COMPARISON TABLE  (mean +/- SD over {total_fits} folds)")
    print("=" * 100)
    print(display.to_string())

    print("\n--- markdown for README.md ---\n")
    print(display.reset_index().to_markdown(index=False))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    summary.round(4).to_csv(os.path.join(HERE, "results", "metrics_summary.csv"))
    per_fold.round(4).to_csv(os.path.join(HERE, "results", "metrics_per_fold.csv"),
                             index=False)

    for metric in evaluation.METRIC_NAMES:
        best = summary[metric].idxmax()
        print(f"Best {metric:<10}{best}  ({summary.loc[best, metric]:.4f})")

    # ---------------- spread across folds ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, metric in zip(axes.ravel(), evaluation.METRIC_NAMES):
        data = [per_fold[per_fold["model"] == m][metric].to_numpy()
                for m in pipelines]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(pipelines) + 1),
                      [m.split(" (")[0] for m in pipelines])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Spread across {total_fits} cross-validation folds", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "metric_spread.png"),
                dpi=120, bbox_inches="tight")
    print("\nwrote results/metric_spread.png")

    # ---------------- pooled confusion matrices ----------------
    fig, axes = plt.subplots(1, len(pipelines), figsize=(4.4 * len(pipelines), 4.2))
    for ax, (name, pipeline) in zip(np.atleast_1d(axes), pipelines.items()):
        cm = evaluation.pooled_confusion(pipeline, ds.features, y, class_names,
                                         args.folds, classifiers.RANDOM_STATE)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Out-of-fold confusion matrices (every row predicted once)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "confusion_matrices.png"),
                dpi=120, bbox_inches="tight")
    print("wrote results/confusion_matrices.png")

    # ---------------- per-model reports ----------------
    print("\n" + "=" * 100)
    print("OUT-OF-FOLD CLASSIFICATION REPORTS")
    print("=" * 100)
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True,
                               random_state=classifiers.RANDOM_STATE)
    for name, pipeline in pipelines.items():
        y_pred = cross_val_predict(pipeline, ds.features, y, cv=splitter)
        print(f"\n{name}\n{'-' * len(name)}")
        print(classification_report(y, y_pred, target_names=class_names,
                                    zero_division=0))

    print("Done. Tables and figures are in results/.")


if __name__ == "__main__":
    main()
