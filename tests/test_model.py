"""Self-tests for the modelling package.

Run with:  python -m unittest discover -s tests -v

Deliberately uses the standard library's unittest rather than pytest, so the
test suite adds no dependency beyond what the app already needs.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import classifiers, datasets, evaluation, sources  # noqa: E402


def _toy_frame(n=600, n_features=14, n_classes=2, seed=0):
    """A small frame that satisfies the assignment's constraints."""
    rng = np.random.default_rng(seed)
    data = {f"f{i}": rng.normal(size=n) for i in range(n_features - 1)}
    data["cat"] = rng.choice(["a", "b", "c"], n)
    frame = pd.DataFrame(data)
    signal = frame["f0"] * 2 + frame["f1"]
    frame["label"] = pd.qcut(signal, n_classes,
                             labels=[f"c{i}" for i in range(n_classes)]).astype(str)
    return frame


class TestDatasetLoading(unittest.TestCase):

    def setUp(self):
        self.path = "_toy_test.csv"
        _toy_frame().to_csv(self.path, index=False)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_loads_and_reports_shape(self):
        ds = datasets.load(self.path, "label")
        self.assertEqual(ds.summary()["instances"], 600)
        self.assertEqual(ds.summary()["features"], 14)
        self.assertTrue(ds.is_binary)

    def test_splits_numeric_and_categorical(self):
        ds = datasets.load(self.path, "label")
        self.assertIn("cat", ds.categorical_cols)
        self.assertNotIn("cat", ds.numeric_cols)
        self.assertEqual(len(ds.numeric_cols) + len(ds.categorical_cols), 14)

    def test_rejects_missing_target(self):
        with self.assertRaises(datasets.DatasetError):
            datasets.load(self.path, "not_a_column")

    def test_rejects_too_few_features(self):
        _toy_frame(n=600, n_features=5).to_csv(self.path, index=False)
        with self.assertRaises(datasets.DatasetError):
            datasets.load(self.path, "label")

    def test_rejects_too_few_rows(self):
        _toy_frame(n=100).to_csv(self.path, index=False)
        with self.assertRaises(datasets.DatasetError):
            datasets.load(self.path, "label")

    def test_rejects_missing_file(self):
        with self.assertRaises(datasets.DatasetError):
            datasets.load("no_such_file.csv", "label")


class TestRemoteSource(unittest.TestCase):
    """The acquisition layer must degrade, never crash."""

    def test_snapshot_matches_recorded_digest(self):
        self.assertTrue(sources.verify_snapshot(),
                        "committed snapshot no longer matches the upstream digest")

    def test_offline_never_downloads(self):
        """With the network disabled, only local origins are acceptable."""
        path, origin = sources.acquire(allow_network=False)
        self.assertIn(origin, {"cache", "snapshot"})
        self.assertNotEqual(origin, "download")
        self.assertTrue(os.path.exists(path))

    def test_unreachable_host_falls_back(self):
        dead = sources.RemoteSource(
            name="dead", url="https://nonexistent.invalid/x.csv", sha256="0" * 64,
            target_col="disease", snapshot="soybean.csv", description="")
        _, origin = sources.acquire(dead)
        self.assertEqual(origin, "snapshot")

    def test_digest_mismatch_rejects_download(self):
        bad = sources.RemoteSource(
            name="bad", url=sources.SOYBEAN.url, sha256="0" * 64,
            target_col="disease", snapshot="soybean.csv", description="")
        _, origin = sources.acquire(bad)
        self.assertEqual(origin, "snapshot",
                         "a mismatched download must not be used")

    def test_digest_is_stable(self):
        self.assertEqual(sources.digest(b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223"
                         "b00361a396177a9cb410ff61f20015ad")


class TestCategoricalCoercion(unittest.TestCase):

    def setUp(self):
        self.path = "_coerce_test.csv"
        frame = _toy_frame()
        frame["coded"] = (frame["f0"] > 0).astype(int)
        frame.to_csv(self.path, index=False)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_low_cardinality_numeric_becomes_categorical(self):
        ds = datasets.load(self.path, "label", categorical_max_unique=5)
        self.assertIn("coded", ds.categorical_cols)
        self.assertNotIn("coded", ds.numeric_cols)

    def test_disabled_by_default(self):
        ds = datasets.load(self.path, "label")
        self.assertIn("coded", ds.numeric_cols)

    def test_dots_in_column_names_are_normalised(self):
        frame = _toy_frame()
        frame = frame.rename(columns={"f0": "plant.stand"})
        frame.to_csv(self.path, index=False)
        ds = datasets.load(self.path, "label")
        self.assertIn("plant_stand", ds.features.columns)


class TestPipelines(unittest.TestCase):

    def test_scales_hyperparameters_to_class_count(self):
        """A shallow tree cripples a many-class problem; depth must adapt."""
        few = classifiers.base_estimators(n_classes=2, smallest_class=50)
        many = classifiers.base_estimators(n_classes=19, smallest_class=8)
        self.assertEqual(few["Decision Tree"].max_depth, 6)
        self.assertIsNone(many["Decision Tree"].max_depth)

    def test_k_stays_below_smallest_class(self):
        est = classifiers.base_estimators(n_classes=19, smallest_class=8)
        self.assertLess(est["kNN"].n_neighbors, 8)

    def test_builds_all_required_models(self):
        pipes = classifiers.build_all(["f0", "f1"], ["cat"])
        self.assertEqual(list(pipes), classifiers.MODEL_ORDER)
        self.assertEqual(len(pipes), 5)

    def test_preprocessing_is_inside_the_pipeline(self):
        """Guards against the leakage bug of scaling before the split."""
        for name, pipe in classifiers.build_all(["f0"], []).items():
            self.assertIn("preprocess", pipe.named_steps, name)
            self.assertIn("classifier", pipe.named_steps, name)

    def test_omits_categorical_branch_when_unused(self):
        pre = classifiers.build_preprocessor(["f0", "f1"], [])
        self.assertEqual([n for n, _, _ in pre.transformers], ["numeric"])


class TestEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        frame = _toy_frame()
        cls.X = frame.drop(columns=["label"])
        cls.y = pd.factorize(frame["label"])[0]
        cls.pipes = classifiers.build_all(
            [c for c in cls.X.columns if c != "cat"], ["cat"])

    def test_returns_all_six_metrics(self):
        summary, per_fold = evaluation.evaluate_all(
            {"kNN": self.pipes["kNN"]}, self.X, self.y, True,
            n_splits=3, n_repeats=1)
        for metric in evaluation.METRIC_NAMES:
            self.assertIn(metric, summary.columns)
            self.assertIn(f"{metric} SD", summary.columns)

    def test_metrics_are_in_valid_range(self):
        summary, _ = evaluation.evaluate_all(
            {"kNN": self.pipes["kNN"]}, self.X, self.y, True,
            n_splits=3, n_repeats=1)
        row = summary.iloc[0]
        for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1"]:
            self.assertGreaterEqual(row[metric], 0.0, metric)
            self.assertLessEqual(row[metric], 1.0, metric)
        self.assertGreaterEqual(row["MCC"], -1.0)
        self.assertLessEqual(row["MCC"], 1.0)

    def test_fold_count_matches_request(self):
        _, per_fold = evaluation.evaluate_all(
            {"kNN": self.pipes["kNN"]}, self.X, self.y, True,
            n_splits=4, n_repeats=2)
        self.assertEqual(len(per_fold), 8)

    def test_multiclass_path_runs(self):
        frame = _toy_frame(n_classes=4, seed=1)
        X = frame.drop(columns=["label"])
        y = pd.factorize(frame["label"])[0]
        pipes = classifiers.build_all(
            [c for c in X.columns if c != "cat"], ["cat"])
        summary, _ = evaluation.evaluate_all(
            {"Naive Bayes": pipes["Naive Bayes"]}, X, y, False,
            n_splits=3, n_repeats=1)
        self.assertFalse(np.isnan(summary.iloc[0]["AUC"]))

    def test_pooled_confusion_covers_every_row(self):
        cm = evaluation.pooled_confusion(self.pipes["Naive Bayes"], self.X,
                                         self.y, ["c0", "c1"], n_splits=3)
        self.assertEqual(cm.sum(), len(self.y))

    def test_format_summary_produces_mean_and_sd(self):
        summary, _ = evaluation.evaluate_all(
            {"kNN": self.pipes["kNN"]}, self.X, self.y, True,
            n_splits=3, n_repeats=1)
        display = evaluation.format_summary(summary)
        self.assertIn("±", display.iloc[0]["Accuracy"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
