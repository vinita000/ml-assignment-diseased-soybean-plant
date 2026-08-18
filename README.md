# Soybean Disease Classification — ML Assignment 2

**Name:** Vinita Kumari
**BITS ID:** 2025ac05827
**Programme:** M.Tech (AIML) — Work Integrated Learning Programmes
**Course:** Machine Learning

Five classification models applied to a nineteen-class agricultural diagnosis
problem, with the dataset acquired from its public URL at runtime, verified
against a recorded SHA-256, and backed by a committed snapshot.

---

## a. Problem Statement

Given thirty-five field observations of a diseased soybean plant — planting date,
precipitation, temperature, the appearance of leaves, stem, fruit pods, roots and
seed, and the condition of the surrounding crop — identify which of nineteen
diseases the plant is suffering from. This is a multi-class classification
problem with a large number of classes relative to the amount of data.

Two properties make it harder than the class count alone suggests. Class sizes
span an order of magnitude, from 92 instances of the most common disease down to
8 of the rarest. And 121 of the 683 records are incomplete, with 2,337 missing
cells in total — an inspector in the field could not always determine every
attribute. Both are realistic, and both have to be handled rather than ignored.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Source | Soybean (Large), UCI Machine Learning Repository |
| Retrieved from | `https://raw.githubusercontent.com/selva86/datasets/master/Soybean.csv` |
| SHA-256 | `96eac8047b2034523c57c5d6ae32ecc64ba2da139b1d930a838a35de3cda5ce6` |
| Instances | 683 |
| Features | 35 (meets the ≥12 requirement) |
| Target variable | `disease` |
| Classes | 19 |
| Class balance | 8 to 92 instances per class |
| Missing values | 2,337 cells across 121 rows |
| Feature types | 35 categorical (see below) |
| Train / test split | 546 / 137, stratified, `random_state=42` |

### Why every feature is treated as categorical

The features arrive as small integers, which `read_csv` types as numeric. They
are not numeric. `precip` is 0, 1 or 2 for *below normal*, *normal*, *above
normal*; `stem_cankers` is 0–3 for position on the stem. Left as numbers, a
scaler would treat the gap from 0 to 1 as a real distance and the models would
read ordering into labels that have none.

`model/datasets.py` therefore takes a `categorical_max_unique` parameter: any
numeric column with at most that many distinct values is reclassified as
categorical and one-hot encoded. Set to 10 for this dataset, which captures all
35 columns.

### Preprocessing

- **Most-frequent imputation** on every feature. This is the first thing the
  pipeline does, and unlike the other variants of this project it is doing real
  work here — 18% of rows have at least one gap.
- **One-hot encoding** with `handle_unknown="ignore"`, expanding 35 coded columns
  into roughly 100 binary indicators.
- **`class_weight="balanced"`** on Logistic Regression, Decision Tree and Random
  Forest, since the largest class is eleven times the smallest.
- All of it inside a scikit-learn `Pipeline`, so cross-validation refits the
  preprocessing **inside each fold**. A test asserts this structure holds.

### Hyperparameters derived from the target's shape

Two settings are computed from the data rather than hard-coded, because values
that suit a balanced binary problem actively damage a nineteen-class one:

| Setting | Binary default | This dataset | Why |
|---|---|---|---|
| `DecisionTreeClassifier(max_depth=)` | 6 | `None` | A depth-6 tree has at most 64 leaves — not enough structure for 19 classes over ~100 one-hot inputs |
| `KNeighborsClassifier(n_neighbors=)` | 11 | 7 | k must stay below the smallest class size; a class with 8 members can never win a vote among 11 neighbours |

This mattered. An earlier run of this project carried the binary defaults over
unchanged and the Decision Tree scored **0.3651** accuracy while every other
model scored above 0.90. That was not a property of decision trees — it was a
crippled baseline produced by a hyperparameter that made sense for a different
dataset. Correcting the depth moved it to 0.9185.

---

## c. GitHub Repository Link

**Repository:** <FILL IN — https://github.com/your-username/ml-assignment-2>

**Live Streamlit App:** <FILL IN — https://your-app.streamlit.app>

```
ml-assignment-2/
├── app.py                       Streamlit application
├── run_experiment.py            CLI: acquires data, cross-validates, writes results/
├── requirements.txt             five dependencies
├── README.md                    this file
├── test_data.csv                137-row held-out slice for the app
├── data/
│   ├── soybean.csv              committed snapshot, byte-identical to upstream
│   └── .cache/                  verified downloads (gitignored)
├── model/
│   ├── __init__.py
│   ├── sources.py               remote acquisition, integrity, fallback
│   ├── datasets.py              loading, validation, categorical coercion
│   ├── classifiers.py           preprocessing + the five estimators
│   ├── evaluation.py            cross-validated metrics
│   └── ML_Assignment2_Colab.ipynb
├── tests/
│   └── test_model.py            25 unit tests, standard-library unittest
└── results/
    ├── metrics_summary.csv
    ├── metrics_per_fold.csv
    ├── metric_spread.png
    └── confusion_matrices.png
```

---

## The data acquisition layer

This is the part that differs most from a conventional submission, so it is worth
setting out plainly — including the argument against it.

`model/sources.py` fetches the dataset from its public URL using only
`urllib.request` from the standard library, so it adds nothing to
`requirements.txt`. Three safeguards sit around the transfer:

**Integrity checking.** The upstream file's SHA-256 is recorded in the source
definition. A download whose digest does not match is rejected rather than
silently trained on. Public data files get corrected, reformatted and replaced
without notice, and a changed dataset that still parses cleanly is the failure
mode that produces wrong numbers instead of an error.

**Local caching.** A verified download is written to `data/.cache/` and reused,
so the network is touched once rather than on every app restart. A cached file
that fails its digest is discarded and re-fetched.

**A committed snapshot.** `data/soybean.csv` is byte-identical to upstream, so
the same recorded digest verifies both. If the network is unreachable, the host
is rate limiting, or the digest fails, the loader falls back to the snapshot and
reports which source it used.

### The honest argument against runtime fetching

Fetching at deploy time is, by itself, a **reliability regression**. A hosting
platform that cannot reach the upstream host gets a dead app rather than a
working one, and you have traded a guaranteed local read for a dependency on
somebody else's uptime. During development of this project the upstream host
returned HTTP 429 continuously for roughly forty minutes, which is precisely the
scenario the fallback exists for.

So the design treats the remote path as the convenience and the committed
snapshot as the thing that makes the deployment dependable — not the other way
round. The app surfaces which source is in use, and a sidebar toggle disables the
network entirely. The assignment requires the test data to be committed in any
case, so local data was never optional.

`sources.acquire()` never raises on a network problem. Five unit tests cover the
degradation paths: cache hit, network disabled, corrupt cache, digest mismatch,
and unreachable host.

---

## d. Models Used

### Evaluation methodology

Every model is scored with **repeated stratified k-fold cross-validation** — 5
folds repeated 3 times, 15 fits per model — reported as mean ± standard
deviation. Stratification is essential here: with 8 instances in the rarest
class, an unstratified split could leave a class entirely absent from a fold.

### Comparison Table

Mean ± SD across 15 folds. Precision, Recall and F1 are weighted averages across
the 19 classes; AUC is one-vs-rest, weighted.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:---|:---|:---|:---|:---|:---|:---|
| Logistic Regression | 0.9288 ± 0.0234 | 0.9954 ± 0.0021 | 0.9373 ± 0.0213 | 0.9288 ± 0.0234 | 0.9283 ± 0.0243 | 0.9232 ± 0.0248 |
| Decision Tree | 0.9185 ± 0.0129 | 0.9535 ± 0.0073 | 0.9230 ± 0.0135 | 0.9185 ± 0.0129 | 0.9174 ± 0.0132 | 0.9114 ± 0.0139 |
| kNN | 0.9126 ± 0.0223 | 0.9897 ± 0.0058 | 0.9227 ± 0.0201 | 0.9126 ± 0.0223 | 0.9106 ± 0.0240 | 0.9056 ± 0.0239 |
| Naive Bayes | 0.9424 ± 0.0148 | 0.9907 ± 0.0050 | 0.9513 ± 0.0134 | 0.9424 ± 0.0148 | 0.9415 ± 0.0156 | 0.9383 ± 0.0159 |
| Random Forest (Ensemble) | 0.9327 ± 0.0182 | 0.9964 ± 0.0017 | 0.9388 ± 0.0164 | 0.9327 ± 0.0182 | 0.9320 ± 0.0187 | 0.9272 ± 0.0193 |

Fold spread: `results/metric_spread.png`.
Out-of-fold confusion matrices: `results/confusion_matrices.png`.

### Single-split figures (what the app reports)

Fitted once on 546 training rows, scored on the 137-row `test_data.csv`:

| ML Model Name | Accuracy | AUC | F1 | MCC | Misclassified |
|:---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9124 | 0.9933 | 0.9118 | 0.9050 | 12 / 137 |
| Decision Tree | 0.9197 | 0.9538 | 0.9183 | 0.9133 | 11 / 137 |
| kNN | 0.8832 | 0.9672 | 0.8819 | 0.8734 | 16 / 137 |
| Naive Bayes | 0.9781 | 0.9986 | 0.9791 | 0.9763 | 3 / 137 |
| Random Forest (Ensemble) | 0.9124 | 0.9954 | 0.9126 | 0.9046 | 12 / 137 |

Note how far Naive Bayes moves between the two tables — 0.9424 under
cross-validation against 0.9781 on this particular 137-row split. That gap is
itself worth a sentence in the observations below.

### Observations

Mean ± SD figures below are the 15-fold cross-validated numbers from the
comparison table; single-split figures refer to the `test_data.csv` table
above.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Third on accuracy (0.9288 ± 0.0234) but the second-best AUC (0.9954 ± 0.0021). A one-vs-rest linear model gets 19 separate linear boundaries to draw across the 99 one-hot indicators produced from the 35 coded fields, and most soybean diseases are distinguished by a handful of near-deterministic symptom combinations (leaf malformation only in one disease, a specific canker-plus-lodging pairing in another) rather than subtle, overlapping cues. That is closer to linearly separable in one-hot space than the class count on its own would suggest, which is why plain logistic regression is competitive with the ensembles here. |
| Decision Tree | Lowest accuracy among the non-kNN models (0.9185 ± 0.0129) but the *tightest* SD of any model on that metric — tighter than Naive Bayes and less than half Logistic Regression's. That is the opposite of what a single unpruned tree normally does across folds. With `max_depth=None` it grows to depth 22 with 66 leaves on the training data, but the categorical, low-cardinality, symptom-coded features give it very similar splits fold to fold, so it lands in roughly the same place each time — consistent, not necessarily good. Its AUC (0.9535 ± 0.0073) is far below every other model's because a single tree's leaf-membership probabilities are close to hard 0/1 votes rather than genuine confidence estimates: it can be right most of the time while ranking classes poorly on the ones it gets wrong. |
| kNN | Last on Accuracy (0.9126 ± 0.0223), F1 (0.9106 ± 0.0240) and MCC (0.9056 ± 0.0239), yet second-best on AUC (0.9897 ± 0.0058) — its probability estimates rank classes well even where its hard predictions don't. Distance in a ~99-dimensional one-hot space is a poor proxy for symptom similarity: two records differing in a single coded field sit almost as far apart as two records that agree on nothing, so "nearest" neighbours are noisier than the raw feature count implies. It is also the model most constrained by the class imbalance — k had to be capped below 8 (the smallest class' size) purely so a vote among neighbours can't be won entirely by a majority class, leaving k=7, which is a small, high-variance neighbourhood for a 19-way problem. |
| Naive Bayes | The winner on five of six metrics (0.9424 ± 0.0148 accuracy, plus Precision, Recall, F1 and MCC), losing only AUC to Random Forest — unusual for the simplest model in the lineup, and worth explaining rather than just reporting. Its core assumption — that features are independent given the class — is normally a crude approximation, but here the features *are* largely independent symptom indicators conditioned on a disease (whether the seed shows discoloration tells you little else once you already know the disease), which is close to the actual data-generating process for once. The single-split table shows it jumping to 0.9781 accuracy with only 3/137 misclassified — a 3.6-point gap over its own cross-validated mean, which is the clearest illustration in this project of why the CV table exists: one 137-row split flatters this model more than any other. |
| Random Forest (Ensemble) | Best AUC of all five (0.9964 ± 0.0017) with by far the tightest SD on that metric, but only **second** on accuracy (0.9327 ± 0.0182), behind Naive Bayes. Bagging 300 trees smooths out the single tree's noisy, near-hard-vote probabilities into a well-calibrated ranking — hence the AUC — but averaging votes doesn't fix the same tree-level confusions (e.g. `alternarialeaf-spot` vs `frog-eye-leaf-spot`, which share most recorded symptoms) that the single Decision Tree also makes, so accuracy improves by only about 1.4 points over the lone tree (0.9327 vs 0.9185) rather than the larger jump ensembling usually buys. |
| **Overall Winner** | **Naive Bayes**, on five of six metrics — but the margin is worth interrogating rather than taking at face value. Its accuracy lead over Random Forest is 0.0097, smaller than either model's own fold-to-fold SD (0.0148 and 0.0182), which on the summary table alone reads as a tie. Looking at the actual paired fold results tells a more specific story: Naive Bayes only outright beats Random Forest in 8 of the 15 folds, yet a paired test across those folds (which cancels out fold-to-fold difficulty rather than treating each model's spread independently) puts the difference at p ≈ 0.03 — nominally significant despite winning barely more than half the individual folds, because the *within-fold* differences are more consistent than the *between-fold* accuracy swings suggest. My read: a real but small edge, not a landslide. To settle it more convincingly I would rerun with more repeats (10×5 rather than 5×3) to shrink the SD further, and complement the paired accuracy test with McNemar's test on the pooled out-of-fold predictions in `results/confusion_matrices.png`, which compares the two models' errors on the exact same rows rather than aggregated fold statistics. |

---

## Streamlit App Features

| Requirement | Implementation |
|---|---|
| Dataset upload option (CSV) | Sidebar file uploader, with the bundled 137-row `test_data.csv` as fallback |
| Model selection dropdown | Sidebar `selectbox` across all five pipelines |
| Display of evaluation metrics | "Evaluate on a CSV" tab — all six metrics as metric cards |
| Confusion matrix / classification report | Same tab, 19×19 heatmap and per-class report |
| Additional | Data-source banner showing whether the app downloaded, cached or fell back; a network toggle; cross-validation tab with adjustable folds and a live box plot; mistakes-only filter; per-feature class-separation view |

The app fits on the **training rows only**, reproducing the split and seed used
by `run_experiment.py`, so `test_data.csv` is genuinely unseen.

---

## How to Run Locally

```bash
git clone <your-repo-url>
cd ml-assignment-2
pip install -r requirements.txt

python -m unittest discover -s tests -v   # 25 tests
python run_experiment.py                  # fetches, verifies, cross-validates
python run_experiment.py --offline        # snapshot only, no network
python run_experiment.py --refresh        # ignore cache, re-download
streamlit run app.py
```

Any other dataset:

```bash
python run_experiment.py --csv data/your.csv --target your_column --folds 10
```

---

## BITS Virtual Lab Execution

`model/ML_Assignment2_Colab.ipynb` is the notebook version. A screenshot of it
executing on BITS Virtual Lab is included in the submission PDF.

---

## Notes on Dependencies and Reproducibility

- **Five packages**: `streamlit`, `scikit-learn`, `pandas`, `numpy`,
  `matplotlib`. HTTP uses the standard library, so remote fetching adds nothing.
- **No pickled models.** The app rebuilds from source at startup under
  `st.cache_resource`, which removes the version-skew failure where a pickle
  written by one scikit-learn release will not load under another.
- **No binary artifacts** beyond the two result figures.
- `random_state=42` throughout. Re-running `run_experiment.py` reproduces every
  number above exactly.
- If the upstream file ever changes, the digest check fails loudly and the run
  falls back to the snapshot rather than quietly training on different data.
