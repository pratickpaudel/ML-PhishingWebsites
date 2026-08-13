"""Central configuration. All experimental constants live here so a single
change propagates through the pipeline.
"""

import os
from pathlib import Path

# joblib probes for physical cores and emits a noisy traceback in containerised
# environments where that probe fails. Declaring the count up front avoids it.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = CODE_DIR / "data"
RESULTS_DIR = CODE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
STATS_DIR = RESULTS_DIR / "statistical_tests"
SHAP_DIR = RESULTS_DIR / "shap"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
MODELS_DIR = CODE_DIR / "models"

for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, STATS_DIR,
           SHAP_DIR, PREDICTIONS_DIR, CHECKPOINTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
TEST_SIZE = 0.20          # stratified 80/20 train-test split
CV_FOLDS = 5              # stratified 5-fold cross-validation
SCORING = "f1"            # metric used to select hyperparameters
ALPHA = 0.05              # significance level for statistical tests

# Induced imbalance is NOT used in the main experiments: both datasets in
MINORITY_RATIO = None

# Ratios used for the optional sensitivity analysis (Chapter 6).
SENSITIVITY_RATIOS = [0.05, 0.10, 0.20]

DATASETS = ["vrbancic", "urlphish"]

# Stratified subsampling size. Both datasets are large (88k and 116k rows) and
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