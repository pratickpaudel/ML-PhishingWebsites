# Implementation Guide

Working implementation of the experimental procedure described in Chapter 3 and
Chapter 4: a comparison of class imbalance treatment techniques for machine
learning based phishing website detection.

Each stage below corresponds to one node in the experimental procedure diagram
(`../figures/Figure_3_1_Experimental_Procedure.png`).

---

## 1. Setup

Python 3.11 is used. From the `code/` directory:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Package versions are pinned to match those reported in Chapter 4 (scikit-learn
1.3.0, imbalanced-learn 0.11.0, pandas 2.0.3, numpy 1.24.3, SHAP 0.42.1).

---

## 2. Run everything

The whole procedure can be executed with a single command:

```bash
.venv/bin/python run_subsample.py
```

Runtime is roughly 5 minutes for the full 48-configuration sweep. Useful flags:

| Flag | Purpose |
|---|---|
| `--quick` | Small subset, for checking the setup works |
| `--skip-experiments` | Re-run only the analysis on existing `results.csv` |
| `--with-shap` | Also run the SHAP explainability stage |
| `--ratio 0.05` | Use a different induced imbalance ratio |

For the statistically stronger version, run the sweep under several seeds
(see [section 8](#8-repeated-runs-recommended)):

```bash
.venv/bin/python src/experiment.py --seeds 42 1 2 --output results_multiseed.csv
```

Everything below explains what each stage does and how to run it on its own.

---

## 3. Stage-by-stage

All individual modules are run from the `src/` directory.

### Step 1-2: Dataset loading and preprocessing

```bash
cd src
../.venv/bin/python data_loader.py
```

`data_loader.py` fetches both benchmarks and caches them in `data/`:

| Dataset | As published | % phishing | Ratio |
|---|---|---|---|
| **Vrbančič et al. (2020)** | 88,647 rows, 111 features | 34.57% | 1:1.89 |
| **URL-Phish (2025)** | 116,600 rows, 22 features | 14.24% | 1:6.02 |

Both are **naturally imbalanced as published**, so no artificial skew is
introduced. This matters because applying SMOTE or cost-sensitive weighting to
balanced data leaves the techniques with nothing to correct.

Two earlier candidates were examined and rejected for exactly that reason. Their
loaders are retained so the claim can be reproduced:

| Rejected | % phishing | Ratio | Why |
|---|---|---|---|
| UCI Phishing Websites | 44.31% | 1:1.26 | Close to balanced |
| Hannousse & Yahiouche | 50.00% | 1:1.00 | Balanced by design |

**Label convention.** Phishing is the positive class (`1`), legitimate is `0`.
Every recall, precision, F1 and PR-AUC figure refers to the phishing class.

**Stratified subsampling.** Both datasets are reduced to 20,000 rows
(`SUBSAMPLE_SIZE` in `config.py`) because the SVM scales roughly quadratically
in training set size and the matrix is repeated across three seeds. Sampling is
proportional within each class, so the imbalance ratio is preserved exactly
(34.57% and 14.23% after reduction). Only volume is reduced.

**Induced imbalance is not used.** `MINORITY_RATIO` is `None`; the
`induce_imbalance` function is retained only for the optional severity
sensitivity analysis.

`preprocessing.py` then drops zero-variance and duplicate columns, handles any
non-finite values, and produces a **stratified 80/20 split** with
`random_state=42`. The test set is separated before any treatment or scaling.

### Step 3: Imbalance treatment

`imbalance.py` provides the seven techniques plus an untreated baseline:

| Method | Family |
|---|---|
| Random Oversampling | Data-level (resampling) |
| Random Undersampling | Data-level (resampling) |
| SMOTE | Data-level (synthetic) |
| ADASYN | Data-level (synthetic) |
| SMOTEENN | Hybrid |
| SMOTETomek | Hybrid |
| Cost-Sensitive Learning | Algorithm-level |

The first six return an imbalanced-learn sampler. Cost-sensitive learning is
different in kind — it changes the training objective rather than the data — so
it returns `None` and is applied through the classifier's `class_weight`
instead. Resampling and class weighting are therefore never combined.

### Step 4-5: Classifier selection, training and tuning

`models.py` builds the pipeline that keeps the experiment leakage-free:

```python
ImbPipeline([
    ("scaler",     StandardScaler()),   # fitted per fold
    ("sampler",    <sampler>),          # resamples that fold's training part only
    ("classifier", <estimator>),
])
```

This pipeline is passed to `GridSearchCV` with `StratifiedKFold(5)`. Because the
sampler lives *inside* the pipeline, resampling is re-executed within each fold
and applied only to that fold's training partition — synthetic samples never
reach the validation partition. Doing the resampling before cross-validation
would inflate the scores.

Tuned hyperparameters:

| Classifier | Parameters |
|---|---|
| Decision Tree | `max_depth`, `min_samples_split`, `criterion` |
| Random Forest | `n_estimators`, `max_depth`, `max_features` |
| SVM | `C`, `gamma`, `kernel` |

Selection uses F1 on the minority class. The winning configuration is refit on
the full training set (`refit=True`).

### Step 6: Evaluation

`evaluation.py` computes precision, recall, F1, ROC-AUC, PR-AUC, MCC, balanced
accuracy and the confusion matrix on the untouched test set. Accuracy is
recorded but not used to draw conclusions, since a majority-class predictor
scores highly on imbalanced data.

Threshold-free metrics use `predict_proba` where available and
`decision_function` for the SVM.

### Step 7: Run the experimental matrix

```bash
../.venv/bin/python experiment.py
```

Runs 2 datasets × 8 methods × 3 classifiers = 48 configurations (the 42 reported
in Chapter 4, plus 6 untreated baselines for reference). Outputs:

* `results/results.csv` — one row per configuration
* `results/predictions/*.npz` — per-configuration test predictions, needed
  because McNemar's test operates on paired predictions rather than summary
  scores

Subsets can be run with `--datasets`, `--methods`, `--classifiers`, `--ratio`.

### Step 8: Comparative performance analysis

```bash
../.venv/bin/python analysis.py
```

Generates the Chapter 5 tables as both `.csv` and `.md` (paste-ready):

| Output | Content |
|---|---|
| `table_5_1/5_2_performance_*` | Every configuration, per dataset |
| `table_5_3_by_classifier` | Mean performance per classifier |
| `table_5_4_by_method` | Mean performance per imbalance method |
| `table_5_5_best_worst` | Best and worst configuration per dataset |
| `table_treatment_effect` | Change vs the untreated baseline |

`table_treatment_effect` is the one that isolates the contribution of imbalance
treatment itself, by comparing each treated configuration against the same
classifier trained on untreated data.

### Step 9: Statistical significance testing

```bash
../.venv/bin/python statistical_tests.py
```

* **Friedman test** — whether classifiers differ overall, and whether imbalance
  methods differ overall, using matched blocks and mean ranks.
* **Post-hoc Wilcoxon** signed-rank tests with **Holm-Bonferroni** correction,
  identifying which specific pairs differ. Holm is used rather than plain
  Bonferroni because it is uniformly more powerful at the same error rate.
* **McNemar's test** — paired comparison on identical test instances. The exact
  binomial version is used when there are fewer than 25 discordant cases, where
  the chi-squared approximation is unreliable.

The McNemar comparisons are chosen to be informative rather than merely
top-ranked: best vs untreated baseline (did treatment change behaviour?), best
vs the best of each other classifier, and best vs worst.

### Step 10: SHAP explainability (Section 3.10)

```bash
# global + local attributions for one configuration
../.venv/bin/python explainability.py --dataset uci --method smote --classifier random_forest --plot

# does treatment change which features the model relies on?
../.venv/bin/python explainability.py --compare --dataset uci --classifier random_forest
```

`TreeExplainer` is used for Decision Tree and Random Forest (exact and fast).
The SVM falls back to `KernelExplainer`, which is an approximation and is **much
slower** — around 4 minutes for 40 instances. Prefer the tree models for SHAP
analysis, or reduce `sample_size`.

Local explanations report feature values on their **original scale**, not the
standardised values the model sees, so they are readable.

### Step 11: Explainability dashboard (Section 3.10)

The dashboard needs a persisted model, so train one per dataset first. The
configuration is chosen automatically from the experiment results, so run the
experiments beforehand:

```bash
cd src
../.venv/bin/python persist_models.py
cd ..
.venv/bin/streamlit run dashboard.py
```

This writes `models/{dataset}_model.joblib` alongside the feature ordering, a
sample of held-out test instances and the metrics the configuration achieved. The
model files are excluded from version control because they are large and
regenerable.

The dashboard has three views: prediction with local SHAP attribution, global
feature importance, and the comparative results of the study.

**URL input.** For URL-Phish, all 22 features are lexical, so the dashboard
accepts a URL typed directly. The feature definitions were recovered from the
published data and are checked against it:

```bash
cd src
../.venv/bin/python url_features.py --samples 2000   # verify extraction
../.venv/bin/python url_features.py --url "http://smbc-card565.club"
```

All 22 features reproduce the published values exactly, which is what makes URL
input valid rather than merely plausible. The Vrbancic feature set includes
domain registration and hosting attributes that cannot be derived from a URL, so
instances for that dataset are selected from the held-out test set instead.

**A caveat worth knowing.** The model scores real URLs from the dataset
accurately, but URLs following classic phishing conventions, such as a login path
on a raw IP address, are often scored as legitimate. The dataset's phishing
samples are dominated by abuse of free hosting and site-builder services, while
its legitimate samples are mostly established institutional domains, so the model
learned that narrower distinction. This is a property of the training data, and
the dashboard displays a corresponding caution.

---

## 4. Project layout

```
code/
├── run_pipeline.py          # end-to-end orchestrator
├── requirements.txt
├── dashboard.py             # Streamlit explainability dashboard
├── src/
│   ├── config.py            # all experimental constants
│   ├── data_loader.py       # loading + induced imbalance
│   ├── preprocessing.py     # cleaning + stratified split
│   ├── imbalance.py         # the seven techniques
│   ├── models.py            # pipeline, grids, GridSearchCV
│   ├── evaluation.py        # metrics
│   ├── experiment.py        # the configuration sweep
│   ├── analysis.py          # Chapter 5 tables
│   ├── statistical_tests.py # Friedman, Wilcoxon, McNemar
│   ├── explainability.py    # SHAP
│   ├── persist_models.py    # trains and saves the model the dashboard serves
│   └── url_features.py      # URL feature extraction, with verification
├── data/                    # cached datasets (downloaded on first run)
├── models/                  # persisted models for the dashboard
├── results/                 # results.csv, tables, test outputs
└── figures/                 # SHAP plots
```

---

## 5. Reproducibility

`RANDOM_STATE = 42` is applied to the induced downsampling, the train-test
split, the cross-validation folds, every sampler, and every classifier. Deleting
`results/` and re-running `run_subsample.py` reproduces identical numbers.

---

## 6. Results obtained

Both datasets used at their published class ratios, reduced to 20,000 rows by
stratified sampling, replicated across seeds 42, 1 and 2. F1 on the phishing
class was used for hyperparameter selection. Values below are means over the
three replications.

**By classifier** (treated configurations only):

| Classifier | Precision | Recall | F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|---|
| Decision Tree | 0.8683 | 0.9216 | 0.8934 | 0.9498 | 0.8456 | 0.8608 |
| **Random Forest** | **0.9036** | 0.9398 | **0.9204** | **0.9907** | **0.9749** | **0.8962** |
| Support Vector Machine | 0.8440 | **0.9491** | 0.8926 | 0.9853 | 0.9520 | 0.8600 |

**By imbalance method** (averaged over classifiers and datasets):

| Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| No Treatment (Baseline) | **0.9286** | 0.9067 | **0.9172** | **0.9432** |
| Random Oversampling | 0.8884 | 0.9305 | 0.9083 | 0.9286 |
| Random Undersampling | 0.8267 | **0.9493** | 0.8824 | 0.9171 |
| SMOTE | 0.8912 | 0.9320 | 0.9108 | 0.9323 |
| ADASYN | 0.8539 | 0.9421 | 0.8941 | 0.9220 |
| SMOTEENN | 0.8555 | 0.9462 | 0.8982 | 0.9103 |
| SMOTETomek | 0.8912 | 0.9329 | 0.9112 | 0.9316 |
| Cost-Sensitive Learning | 0.8970 | 0.9250 | 0.9099 | 0.9269 |

**Best and worst configurations:**

| Dataset | Best | F1 | Worst | F1 | Gap |
|---|---|---|---|---|---|
| Vrbančič (1:1.89) | Random Forest + SMOTE | 0.9420 | SVM + ADASYN | 0.9001 | 0.0419 |
| URL-Phish (1:6.02) | Random Forest + SMOTE | 0.9167 | Decision Tree + Random Undersampling | 0.8115 | 0.1052 |

**Statistical tests** (42 blocks for classifiers, 18 for methods):

| Test | Statistic | p | Significant |
|---|---|---|---|
| Friedman — classifiers | χ² = 57.57 | < 0.001 | yes |
| Friedman — imbalance methods | χ² = 43.70 | < 0.001 | yes |
| Post-hoc classifier pairs | — | — | 2 of 3 |
| Post-hoc method pairs | — | — | 8 of 21 |

Mean Friedman ranks (1 = best): Random Forest 1.05, SVM 2.41, Decision Tree
2.55. For methods, SMOTE and SMOTETomek tie best at 2.64, and Random
Undersampling is worst at 5.94.

**SHAP.** On URL-Phish with Random Forest, the three highest-ranked features
(`is_https`, `entropy`, `digit_ratio`) hold identical rank across all treatment
methods, while mid-ranked features move by up to five positions. On Vrbančič,
`time_domain_activation` and `qty_slash_url` are similarly invariant, but weaker
features shift by as much as eleven positions. Imbalance treatment therefore
perturbs the ordering of weaker features without changing which evidence
dominates the decision.

---

## 7. Interpreting these results

Three points need care in the write-up.

**Treatment trades precision for recall.** Every technique raised recall over the
untreated baseline (+0.018 to +0.043 mean), but all of them lowered precision, so
the baseline retains the highest mean F1 (0.9172). This is not a null result — it
is the central trade-off. The choice of technique should follow from the relative
cost of a missed phishing site versus a false alarm; for phishing detection recall
usually dominates, which favours treatment despite the precision cost. Note that
this same pattern appeared on the earlier balanced datasets under induced
imbalance, so it is a robust finding rather than an artefact of one design.

**Imbalance severity governs how much technique choice matters.** The best-to-worst
F1 gap is 0.1052 on URL-Phish (1:6.02) but only 0.0419 on Vrbančič (1:1.89) — two
and a half times larger. Pairing datasets with different natural skew is what makes
this observable, and it is the strongest argument for the two-dataset design.

**Random undersampling remains the clearest failure case.** It produced the
highest recall (0.9493) and the lowest precision (0.8267), because discarding most
of the legitimate training data removes the evidence needed to rule phishing out.
It is the worst method by mean Friedman rank and forms the worst configuration on
URL-Phish.

---

## 8. Repeated runs

Each seed repeats the entire matrix as an independent replication: it changes the
stratified subsample, the train-test split, the CV folds, the samplers and the
classifiers. Every replication becomes an additional matched block for the
Friedman and post-hoc tests.

```bash
cd src
../.venv/bin/python experiment.py --seeds 42 1 2 --output results_multiseed.csv
../.venv/bin/python analysis.py --results results_multiseed.csv
../.venv/bin/python statistical_tests.py --results results_multiseed.csv
```

144 configurations. The analysis code averages metrics across replications
automatically, so each condition still appears once. McNemar's test stays
*within* a replication, because it requires both models to have been evaluated on
identical test instances.

**Runtime.** Roughly 70 minutes in total. Background execution is unreliable in
short-lived environments, so it is safer to run in chunks and merge:

```bash
for ds in vrbancic urlphish; do
  for s in 42 1 2; do
    ../.venv/bin/python experiment.py --seeds $s --datasets $ds \
        --output part_${ds}_s${s}.csv
  done
done
```

Then concatenate the `part_*.csv` files into `results_multiseed.csv`.

**Why replication matters.** On an earlier single-seed run, comparing seven
methods across only six blocks (2 datasets × 3 classifiers) left 0 of 21 pairs
significant after Holm correction, despite a significant overall Friedman test.
Three replications raise this to 18 blocks and recover the ability to separate
individual methods.

State in the dissertation that the experiment was repeated under three random
seeds, that reported metrics are means over replications, and that each
replication contributed an additional block to the significance tests.

---

## 9. Optional sensitivity analysis

The two datasets already differ in natural imbalance (1:1.89 and 1:6.02), which
covers the severity question directly. If a finer-grained analysis is wanted,
`induce_imbalance` can downsample the phishing class further:

```bash
../.venv/bin/python experiment.py --ratio 0.05 --classifiers random_forest --output results_r05.csv
```

This is not used in the main study, since both datasets are imbalanced as
published and inducing further skew would reintroduce the artificiality the
dataset selection was intended to avoid.
