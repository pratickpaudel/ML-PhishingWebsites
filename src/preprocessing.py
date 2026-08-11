"""
Preprocessing and the stratified train-test split (Steps 2-3 of the pipeline).

Two points of methodological importance are enforced here:

1. The split is **stratified**, so the phishing proportion is preserved in both
   the training and test partitions.
2. The test set is separated **before** any imbalance treatment or scaling is
   applied, and is never touched again until final evaluation. Feature scaling
   is fitted inside the cross-validation pipeline (see ``models.py``), not on
   the full dataset, which prevents information leaking from test to train.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Basic structural cleaning applied identically to both datasets.

    * Drops constant (zero-variance) columns, which carry no information.
    * Drops exact duplicate columns.
    * Replaces any infinite values and fills residual gaps with the column
      median so that downstream estimators receive finite input.
    """
    X = X.copy()

    # Remove zero-variance columns.
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    # Remove duplicated columns (same values under a different name).
    X = X.loc[:, ~X.T.duplicated()]

    # Guard against non-finite values.
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    if X.isnull().any().any():
        X = X.fillna(X.median(numeric_only=True))

    return X


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Return a stratified train-test split.

    The stratification guarantees that the minority (phishing) class appears in
    the test set in the same proportion as in the full dataset.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def prepare(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
):
    """Clean the features and produce the train-test split in one call."""
    X = clean_features(X)
    return split_data(X, y, random_state=random_state)


def split_summary(y_train, y_test) -> dict:
    """Summary of the split, used to evidence stratification in Chapter 5."""
    return {
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train_phishing": int((y_train == 1).sum()),
        "test_phishing": int((y_test == 1).sum()),
        "train_phishing_pct": round(100 * (y_train == 1).mean(), 2),
        "test_phishing_pct": round(100 * (y_test == 1).mean(), 2),
    }
