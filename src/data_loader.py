"""
Dataset loading for the phishing detection experiments (Step 1 of the pipeline).

Two **naturally imbalanced** benchmark datasets are used in the study:

* ``vrbancic``  - Vrbancic et al. (2020), Data in Brief.
                  88,647 instances, 111 features, 34.57% phishing (~1:1.89).
* ``urlphish``  - URL-Phish (2025), Mendeley.
                  116,600 URLs, 22 numeric features, 14.24% phishing (~1:6.02).

Both datasets exhibit class imbalance as published, so no artificial skew is
introduced. This matters because the research question concerns the treatment of
class imbalance: applying SMOTE or cost-sensitive weighting to an already
balanced dataset leaves the techniques with nothing to correct.

Two earlier candidates were rejected for exactly that reason and are retained
below only so that the claim can be reproduced:

* ``uci``       - UCI Phishing Websites: 44.31% phishing (1:1.26).
* ``hannousse`` - Hannousse & Yahiouche: 50.00% phishing (1:1.00), balanced by
                  design.

Label convention
----------------
Throughout the project the **phishing class is the positive class (1)** and the
legitimate class is the negative class (0). Recall, precision, F1 and PR-AUC are
therefore all reported with respect to the phishing class, which is the
minority-class problem of interest.
"""

from __future__ import annotations

import urllib.request

import numpy as np
import pandas as pd

from config import DATA_DIR, RANDOM_STATE

# --- Datasets used in the study -------------------------------------------
VRBANCIC_FILE = DATA_DIR / "vrbancic_dataset_full.csv"
VRBANCIC_URL = (
    "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/"
    "master/dataset_full.csv"
)

URLPHISH_FILE = DATA_DIR / "urlphish_dataset.csv"
URLPHISH_URL = (
    "https://data.mendeley.com/public-files/datasets/65z9twcx3r/"
    "files/0e9c55e4-9adb-43f5-8403-1bbd143ebdb6/file_downloaded"
)

# --- Rejected candidates, kept for verification only ----------------------
UCI_CACHE = DATA_DIR / "uci_phishing.csv"
HANNOUSSE_FILE = DATA_DIR / "dataset_B_05_2020.csv"
HANNOUSSE_URL = (
    "https://data.mendeley.com/public-files/datasets/c2gw7fy2j4/"
    "files/575316f4-ee1d-453e-a04f-7b950915b61b/file_downloaded"
)


def _download_if_missing(path, url: str) -> None:
    """Fetch a dataset on first use and cache it under ``data/``.

    An explicit User-Agent is sent because both GitHub and Mendeley reject
    urllib's default agent with HTTP 403.
    """
    if path.exists():
        return

    print(f"  downloading {path.name} ...", flush=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; phishing-detection-research)"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        path.write_bytes(response.read())


# ---------------------------------------------------------------------------
# Loaders for the datasets used in the study
# ---------------------------------------------------------------------------
def load_vrbancic() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Vrbancic et al. (2020) phishing dataset.

    The target column ``phishing`` already follows the project convention
    (1 = phishing, 0 = legitimate), so no remapping is needed. All 111 features
    are numeric.
    """
    _download_if_missing(VRBANCIC_FILE, VRBANCIC_URL)

    df = pd.read_csv(VRBANCIC_FILE)
    y = df["phishing"].astype(int)
    y.name = "phishing"
    X = df.drop(columns=["phishing"])
    return X, y


def load_urlphish() -> tuple[pd.DataFrame, pd.Series]:
    """Load the URL-Phish (2025) dataset.

    The ``url``, ``dom`` and ``tld`` columns are string identifiers rather than
    model features and are dropped, leaving 22 numeric lexical and structural
    features. The ``label`` column already uses 1 for phishing.

    Note that the published file contains 116,600 rows with 16,600 phishing
    instances, which differs from the 111,660 / 11,660 figures quoted in the
    accompanying paper. The counts measured from the data are used here.
    """
    _download_if_missing(URLPHISH_FILE, URLPHISH_URL)

    df = pd.read_csv(URLPHISH_FILE)
    y = df["label"].astype(int)
    y.name = "phishing"
    X = df.drop(columns=["label"])
    return X, y


# ---------------------------------------------------------------------------
# Loaders for the rejected candidates (not part of the study)
# ---------------------------------------------------------------------------
def load_uci() -> tuple[pd.DataFrame, pd.Series]:
    """Load the UCI Phishing Websites dataset (rejected: near balanced).

    The raw target uses ``-1`` for phishing and ``1`` for legitimate; it is
    remapped so that phishing is ``1``.
    """
    if UCI_CACHE.exists():
        df = pd.read_csv(UCI_CACHE)
    else:
        from ucimlrepo import fetch_ucirepo

        repo = fetch_ucirepo(id=327)
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        df.to_csv(UCI_CACHE, index=False)

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].map({-1: 1, 1: 0}).astype(int)
    y.name = "phishing"
    return X, y


def load_hannousse() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Hannousse & Yahiouche benchmark (rejected: balanced by design)."""
    _download_if_missing(HANNOUSSE_FILE, HANNOUSSE_URL)

    df = pd.read_csv(HANNOUSSE_FILE)
    y = df["status"].map({"phishing": 1, "legitimate": 0}).astype(int)
    y.name = "phishing"
    X = df.drop(columns=["status", "url"])
    return X, y


LOADERS = {
    "vrbancic": load_vrbancic,
    "urlphish": load_urlphish,
    # Retained so the "these datasets are balanced" claim can be reproduced.
    "uci": load_uci,
    "hannousse": load_hannousse,
}


# ---------------------------------------------------------------------------
# Stratified subsampling
# ---------------------------------------------------------------------------
def stratified_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    n_samples: int,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reduce the dataset to ``n_samples`` rows, preserving the class ratio.

    Both classes are sampled in proportion to their existing frequency, so the
    natural imbalance is carried through unchanged. This is used purely to keep
    training times tractable: the SVM has roughly quadratic complexity in the
    number of training instances, which makes the full 88k-116k row datasets
    impractical across a repeated 144-configuration sweep.

    Returns the data unchanged when ``n_samples`` is at least the dataset size.
    """
    if n_samples is None or n_samples >= len(y):
        return X.reset_index(drop=True), y.reset_index(drop=True)

    rng = np.random.RandomState(random_state)
    keep_parts = []

    for label in (0, 1):
        idx = y[y == label].index.to_numpy()
        # Proportional allocation, with at least one instance per class.
        n_take = int(round(n_samples * len(idx) / len(y)))
        n_take = max(1, min(n_take, len(idx)))
        keep_parts.append(rng.choice(idx, size=n_take, replace=False))

    keep = np.concatenate(keep_parts)
    keep.sort()

    return (
        X.loc[keep].reset_index(drop=True),
        y.loc[keep].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Controlled imbalance (not used in the main study)
# ---------------------------------------------------------------------------
def induce_imbalance(
    X: pd.DataFrame,
    y: pd.Series,
    minority_ratio: float,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    """Randomly downsample the phishing class to a target proportion.

    Retained for the optional severity sensitivity analysis discussed in
    Chapter 6. It is **not** applied in the main experiments, because both
    datasets are already imbalanced as published.

    Only the minority (phishing) class is reduced; every legitimate instance is
    retained, so no data is fabricated.
    """
    if not 0 < minority_ratio < 0.5:
        raise ValueError("minority_ratio must be between 0 and 0.5 (exclusive)")

    rng = np.random.RandomState(random_state)
    pos_idx = y[y == 1].index.to_numpy()
    neg_idx = y[y == 0].index.to_numpy()

    # n_pos / (n_pos + n_neg) = ratio  ->  n_pos = ratio * n_neg / (1 - ratio)
    n_pos_target = int(round(minority_ratio * len(neg_idx) / (1 - minority_ratio)))
    n_pos_target = max(1, min(n_pos_target, len(pos_idx)))

    keep_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)
    keep = np.concatenate([neg_idx, keep_pos])
    keep.sort()

    return X.loc[keep].reset_index(drop=True), y.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def load_dataset(
    name: str,
    minority_ratio: float | None = None,
    random_state: int = RANDOM_STATE,
    subsample: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a dataset by name.

    Parameters
    ----------
    name
        One of ``"vrbancic"``, ``"urlphish"`` (used in the study) or ``"uci"``,
        ``"hannousse"`` (rejected candidates).
    minority_ratio
        If given, the phishing class is downsampled to this proportion. Left as
        ``None`` for the main experiments, since both datasets are already
        imbalanced.
    random_state
        Controls the subsampling draw. Varying it across repeated runs is what
        makes each replication an independent sample.
    subsample
        If given, reduce to this many rows while preserving the class ratio.
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {list(LOADERS)}.")

    X, y = LOADERS[name]()

    # Drop identifier and other non-numeric columns.
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    if minority_ratio is not None:
        X, y = induce_imbalance(X, y, minority_ratio, random_state=random_state)

    if subsample is not None:
        X, y = stratified_subsample(X, y, subsample, random_state=random_state)

    return X, y


def describe(
    name: str,
    minority_ratio: float | None = None,
    random_state: int = RANDOM_STATE,
    subsample: int | None = None,
) -> dict:
    """Return summary statistics used for the dataset table in Chapter 4."""
    X, y = load_dataset(
        name, minority_ratio, random_state=random_state, subsample=subsample
    )
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return {
        "dataset": name,
        "instances": len(y),
        "features": X.shape[1],
        "phishing": n_pos,
        "legitimate": n_neg,
        "phishing_pct": round(100 * n_pos / len(y), 2),
        "imbalance_ratio": f"1:{round(n_neg / n_pos, 2)}",
        "missing_values": int(X.isnull().sum().sum()),
    }


if __name__ == "__main__":
    from config import DATASETS, SUBSAMPLE_SIZE

    print("Datasets used in the study (as published):")
    for ds in DATASETS:
        print("  ", describe(ds))

    print(f"\nAfter stratified subsampling to {SUBSAMPLE_SIZE} rows:")
    for ds in DATASETS:
        print("  ", describe(ds, subsample=SUBSAMPLE_SIZE))

    print("\nRejected candidates (retained to evidence the balance claim):")
    for ds in ("uci", "hannousse"):
        print("  ", describe(ds))