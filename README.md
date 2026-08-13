# Phishing Website Detection under Class Imbalance

Implementation for an MSc dissertation investigating whether class imbalance
treatment techniques improve machine learning performance for phishing website
detection.

Seven treatment techniques are compared against an untreated baseline, across
three classifiers and two naturally imbalanced datasets, giving 48
configurations. The central finding is that no technique improves on the
untreated baseline.

---

## Setup

Python 3.12.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
.venv/bin/pip install -r requirements.txt          # macOS / Linux
```

Package versions are pinned to those reported in Section 4.11: pandas 2.2.3,
numpy 1.26.4, scikit-learn 1.4.2, imbalanced-learn 0.12.3, statsmodels 0.14.2,
SHAP 0.46.0.

Both datasets are downloaded automatically on first run and cached under
`data/`. They are not committed to this repository, since they are published
separately by their authors.

---

## Running

The full pipeline runs with a single command:

```bash
python run_pipeline.py
```

This executes the 48-configuration matrix at full data scale, generates the
result tables, runs the statistical tests, computes SHAP attributions and
writes the figures. Runtime is roughly two hours, dominated by the Support
Vector Machine configurations.

| Flag | Purpose |
|---|---|
| `--quick` | One dataset, three methods, tree classifiers only; for checking the setup |
| `--skip-experiments` | Reuse the existing `results/results.csv` and re-run analysis onwards |
| `--no-shap` | Skip the SHAP stage, which refits once per treatment method |
| `--all-methods-shap` | Compare all eight conditions in SHAP rather than four |
| `--seed N` | Random seed; defaults to 42 |

### Hyperparameter selection

`run_pipeline.py` does not search for hyperparameters. It reads the selections
recorded in `results/results_multiseed.csv`, which were produced by grid search
on a 20,000-instance stratified subsample under three seeds. Grid search at full
scale would take roughly 48 hours, and holding the parameters fixed also
isolates the effect of sample size from the effect of re-selection.

That file is committed, so the pipeline runs without it being regenerated. To
reproduce it:

```bash
python src/experiment.py --seeds 42 1 2 --output results_multiseed.csv
```

Runtime is roughly 69 minutes.

---

## Datasets

| Dataset | Instances | Features | Phishing | Ratio |
|---|---|---|---|---|
| Vrbančič et al. (2020) | 88,647 | 93 after cleaning | 34.57% | 1:1.89 |
| URL-Phish (2025) | 116,600 | 22 | 14.24% | 1:6.02 |

Both are imbalanced as published, so no artificial skew is introduced. Two
earlier candidates, the UCI Phishing Websites dataset (44.31% phishing) and the
Hannousse and Yahiouche benchmark (50.00%), were rejected because they are
approximately balanced and therefore leave imbalance treatment with nothing to
correct. Their loaders are retained in `src/data_loader.py` so this can be
verified independently.

The URL-Phish file as distributed contains 116,600 rows, which differs from the
111,660 described in the accompanying publication. The measured values are used
throughout; see Section 5.3.

---

## What is compared

**Treatment techniques:** no treatment (baseline), random oversampling, random
undersampling, SMOTE, ADASYN, SMOTEENN, SMOTETomek, cost-sensitive learning.

**Classifiers:** Decision Tree, Random Forest, Support Vector Machine.

**Metrics:** precision, recall, F1, ROC-AUC, PR-AUC, MCC, balanced accuracy.
Accuracy is recorded but not used to draw conclusions, since a classifier can
score highly on imbalanced data by predicting the majority class.

**Statistical tests:** Friedman for overall comparisons, post-hoc Wilcoxon with
Holm-Bonferroni correction for pairs, and McNemar on paired predictions.

Resampling is applied inside each cross-validation fold using an
`imblearn.pipeline.Pipeline`, so synthetic instances never enter the partition a
model is validated against.

---

## Results summary

| | Mean F1 |
|---|---|
| No treatment (baseline) | **0.9306** |
| SMOTETomek | 0.9241 |
| Cost-sensitive learning | 0.9240 |
| SMOTE | 0.9234 |
| Random oversampling | 0.9220 |
| SMOTEENN | 0.9188 |
| Random undersampling | 0.9027 |
| ADASYN | 0.8946 |

Every technique raised recall and lowered precision relative to the untreated
baseline, and none improved mean F1. Random Forest was the strongest classifier
at 0.9359 mean F1, holding the highest rank in every matched block.

The full-scale matrix is executed under a single seed, which provides six
matched blocks for the treatment comparison. A Wilcoxon signed-rank test over
six pairs cannot return a two-sided p-value below 0.031, while Holm correction
across 21 pairs requires 0.0024, so no pairwise comparison between techniques
can reach significance. The ranking above rests on the Friedman test and mean
ranks rather than on pairwise separation. This is a limitation of the design and
is discussed in Section 6.5.2.

---

## Repository layout

```
code/
├── run_pipeline.py              end-to-end runner
├── requirements.txt
├── src/
│   ├── config.py                experimental constants and paths
│   ├── data_loader.py           dataset loading, stratified subsampling
│   ├── preprocessing.py         feature cleaning, stratified train-test split
│   ├── imbalance.py             the seven treatment techniques
│   ├── models.py                classifiers, grids, leakage-safe pipeline
│   ├── evaluation.py            metric computation
│   ├── experiment.py            experiment runner, reduced and full scale
│   ├── analysis.py              result tables
│   ├── statistical_tests.py     Friedman, Wilcoxon, McNemar, Holm-Bonferroni
│   └── explainability.py        SHAP attribution and heatmap figures
├── extras/
│   └── url_features.py          URL feature extraction and its validation
├── results/
│   ├── results_multiseed.csv    hyperparameter selection, three seeds
│   ├── results.csv              full-scale run, 48 configurations
│   ├── comparison.csv           reduced scale against full scale
│   ├── tables/                  result tables as CSV and Markdown
│   ├── figures/                 SHAP heatmaps
│   ├── statistical_tests/       Friedman, post-hoc, McNemar output
│   ├── shap/                    SHAP importance and rank comparison
│   ├── predictions/             per-configuration test predictions
│   └── checkpoints/             per-configuration results, for resuming
└── models/                      sample data and metadata
```

Prediction files are committed so that the McNemar tests can be reproduced from
a clone without re-running the matrix. Trained model binaries and cached
datasets are not, since both regenerate from the code.

---

## Feature extraction

The URL-Phish dataset publishes feature values but not the code that produced
them. The definitions were recovered by inspection and validated against the
published data:

```bash
python extras/url_features.py --samples 2000
```

This extracts all 22 features for 2,000 URLs drawn from the dataset and compares
them against the stored values, giving 44,000 individual comparisons. The
recovered definitions are documented in Appendix B.1 of the dissertation.

To extract features for a single URL:

```bash
python extras/url_features.py --url https://example.com/login
```

---

## Reproducing individual stages

Modules can be run on their own. From the `code/` directory:

```bash
# Full-scale matrix only
python src/experiment.py --full

# A subset of configurations
python src/experiment.py --full --datasets urlphish --classifiers random_forest

# Statistical tests on the full-scale results
python src/statistical_tests.py --full

# SHAP analysis and figures
python src/explainability.py --full
python src/explainability.py --figures

# Result tables
python src/analysis.py --results results.csv
```

Each full-scale configuration is checkpointed to `results/checkpoints/` on
completion, so an interrupted run resumes rather than restarting.

---

## Notes on reproducibility

All randomness is seeded. Re-executing under the same seed on the same machine
reproduces the stored results exactly: confusion matrices are identical and the
largest metric discrepancy is one unit in the last place of a double-precision
float. Results are not guaranteed bit-identical across different hardware, where
a different linear algebra implementation may accumulate floating-point
operations in a different order.

The hyperparameter selection subsample and the full-scale test partition are
disjoint, so no instance used to select hyperparameters appears in the partition
on which the resulting models were evaluated.
