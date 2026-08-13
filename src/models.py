"""Classifiers, hyperparameter grids and the training pipeline.

Scaling, resampling and the classifier are wrapped in a single
imblearn Pipeline so resampling runs inside each CV fold and applies only to
that fold's training partition. Without this, synthetic samples leak into
validation and the estimates are optimistically biased.
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
def get_classifier(name: str, class_weight=None, random_state: int = RANDOM_STATE):
    """Instantiate a classifier."""
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
        return SVC(
            class_weight=class_weight,
            random_state=random_state,
            probability=False,
        )

    raise ValueError(f"Unknown classifier '{name}'.")


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
def build_pipeline(
    classifier_name: str,
    imbalance_method: str,
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    """Assemble scaling, optional resampling and the classifier."""
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
    """Wrap the pipeline in a stratified grid search."""
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
def get_scores(fitted_model, X) -> np.ndarray:
    """Continuous scores for the positive (phishing) class."""
    if hasattr(fitted_model, "predict_proba"):
        try:
            return fitted_model.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            pass

    return fitted_model.decision_function(X)