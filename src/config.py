"""
Central configuration for the phishing detection experiments.

All experimental constants live here so that a single change propagates
through the whole pipeline. This supports the reproducibility requirement
described in Chapter 4.
"""

import os
from pathlib import Path

# joblib probes for physical cores and emits a noisy traceback in containerised
# environments where that probe fails. Declaring the count up front avoids it.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_DIR / "data"
RESULTS_DIR = CODE_DIR / "results"
FIGURES_DIR = CODE_DIR / "figures"
MODELS_DIR = CODE_DIR / "models"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20          # stratified 80/20 train-test split
CV_FOLDS = 5              # stratified 5-fold cross-validation
SCORING = "f1"            # metric used to select hyperparameters
ALPHA = 0.05              # significance level for statistical tests

# Induced imbalance is NOT used in the main experiments: both datasets in
# DATASETS are already imbalanced as published (Vrbancic ~1:1.89, URL-Phish
# ~1:6.02). This is retained only for the optional severity sensitivity
# analysis described in Chapter 6.
MINORITY_RATIO = None

# Ratios used for the optional sensitivity analysis (Chapter 6).
SENSITIVITY_RATIOS = [0.05, 0.10, 0.20]

DATASETS = ["vrbancic", "urlphish"]

# Stratified subsampling size. Both datasets are large (88k and 116k rows) and
# the SVM scales roughly quadratically in training set size, which makes the
# full data impractical across a repeated 144-configuration sweep. Subsampling
# preserves the natural class ratio exactly; only the volume is reduced.
# Set to None to use the datasets in full.
SUBSAMPLE_SIZE = 20_000

IMBALANCE_METHODS = [
    "none",                     # baseline: no treatment
    "random_oversampling",
    "random_undersampling",
    "smote",
    "adasyn",
    "smoteenn",
    "smotetomek",
    "cost_sensitive",
]

CLASSIFIERS = ["decision_tree", "random_forest", "svm"]

# The 42-configuration matrix in Chapter 4 excludes the "none" baseline.
# It is included above so a reference point is always available.
CORE_IMBALANCE_METHODS = [m for m in IMBALANCE_METHODS if m != "none"]

# ---------------------------------------------------------------------------
# Display names (used in tables and figures)
# ---------------------------------------------------------------------------
METHOD_LABELS = {
    "none": "No Treatment (Baseline)",
    "random_oversampling": "Random Oversampling",
    "random_undersampling": "Random Undersampling",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
    "smoteenn": "SMOTEENN",
    "smotetomek": "SMOTETomek",
    "cost_sensitive": "Cost-Sensitive Learning",
}

CLASSIFIER_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
}

DATASET_LABELS = {
    "vrbancic": "Vrbancic et al. (1:1.89)",
    "urlphish": "URL-Phish (1:6.02)",
    # Rejected candidates.
    "uci": "UCI Phishing Websites",
    "hannousse": "Hannousse & Yahiouche",
}
