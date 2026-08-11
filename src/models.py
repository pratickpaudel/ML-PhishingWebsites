"""
Classifiers, hyperparameter grids and the leakage-safe training pipeline
(Steps 5-6 of the pipeline).

The central methodological device here is ``imblearn.pipeline.Pipeline``. By
placing scaling, resampling and the classifier inside a single pipeline object
and passing that object to ``GridSearchCV``, the resampling is executed
independently within each cross-validation fold and applied only to that fold's
training partition. Synthetic samples therefore never leak into the validation
partition, which would otherwise produce optimistically biased estimates.
"""

from __future__ import annotations

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import CV_FOLDS, RANDOM_STATE, SCORING
from imbalance import get_sampler, uses_class_weight


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
def get_classifier(name: str, class_weight=None, random_state: int = RANDOM_STATE):
    """Instantiate a classifier.

    ``class_weight`` is set to ``"balanced"`` only for the cost-sensitive
    condition; it stays ``None`` for every resampling condition so that the two
    strategies are not unintentionally combined.
    """
    if name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight=class_weight,
            random_state=random_state,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    if name == "svm":
        # decision_function is used for threshold-free metrics, so probability
        # estimation (which requires expensive internal cross-validation) is
        # left disabled.
        return SVC(
            class_weight=class_weight,
            random_state=random_state,
            probability=False,
        )

    raise ValueError(f"Unknown classifier '{name}'.")


# ---------------------------------------------------------------------------
# Hyperparameter grids (Section 4.5)
# ---------------------------------------------------------------------------
PARAM_GRIDS = {
    "decision_tree": {
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 10],
        "classifier__criterion": ["gini", "entropy"],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [None, 20],
        "classifier__max_features": ["sqrt", "log2"],
    },
    "svm": {
        "classifier__C": [1, 10],
        "classifier__gamma": ["scale", 0.01],
        "classifier__kernel": ["rbf"],
    },
}


def get_param_grid(classifier_name: str) -> dict:
    return dict(PARAM_GRIDS[classifier_name])


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def build_pipeline(
    classifier_name: str,
    imbalance_method: str,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Assemble scaling, optional resampling and the classifier.

    Step order is significant:

    1. ``scaler``     - fitted on the fold's training partition only.
    2. ``sampler``    - resamples that training partition only (omitted for the
                        baseline and for cost-sensitive learning).
    3. ``classifier`` - trained on the treated training partition.
    """
    class_weight = "balanced" if uses_class_weight(imbalance_method) else None
    steps = [("scaler", StandardScaler())]

    sampler = get_sampler(imbalance_method, random_state=random_state)
    if sampler is not None:
        steps.append(("sampler", sampler))

    steps.append(
        ("classifier", get_classifier(classifier_name, class_weight, random_state))
    )
    return ImbPipeline(steps=steps)


def build_search(
    classifier_name: str,
    imbalance_method: str,
    cv_folds: int = CV_FOLDS,
    scoring: str = SCORING,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
) -> GridSearchCV:
    """Wrap the pipeline in a stratified grid search.

    Stratified folds are used because standard k-fold can produce folds with
    very few minority instances on imbalanced data, which destabilises the
    hyperparameter selection.
    """
    pipeline = build_pipeline(classifier_name, imbalance_method, random_state)
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=get_param_grid(classifier_name),
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,       # retrain the best configuration on the full training set
        error_score="raise",
    )


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------
def get_scores(fitted_model, X) -> np.ndarray:
    """Continuous scores for the positive (phishing) class.

    ROC-AUC and PR-AUC are threshold-free and therefore need a ranking rather
    than hard labels. ``predict_proba`` is preferred when available, otherwise
    ``decision_function`` is used (the SVM case).
    """
    if hasattr(fitted_model, "predict_proba"):
        try:
            return fitted_model.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            pass

    return fitted_model.decision_function(X)
