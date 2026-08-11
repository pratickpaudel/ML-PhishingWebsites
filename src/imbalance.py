"""
Imbalance treatment techniques (Step 4 of the pipeline).

Seven strategies are compared, covering the three families identified in the
literature review:

* **Data-level, resampling**      - random oversampling, random undersampling
* **Data-level, synthetic**       - SMOTE, ADASYN
* **Data-level, hybrid**          - SMOTEENN, SMOTETomek
* **Algorithm-level, weighting**  - cost-sensitive learning

The first six return an imbalanced-learn *sampler* that is placed inside the
cross-validation pipeline. Cost-sensitive learning is different in kind: it
changes the training objective rather than the data, so it returns ``None``
here and is applied through the classifier's ``class_weight`` parameter
(see ``models.py``). A ``"none"`` baseline is also provided.
"""

from __future__ import annotations

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from config import RANDOM_STATE

# Methods that work by weighting the loss rather than resampling the data.
ALGORITHM_LEVEL = {"cost_sensitive"}

# Methods that leave the training distribution untouched.
NO_TREATMENT = {"none"}


def get_sampler(method: str, random_state: int = RANDOM_STATE):
    """Return the sampler for ``method``, or ``None`` if no resampling applies.

    Returning ``None`` for ``"none"`` and ``"cost_sensitive"`` is deliberate:
    the pipeline builder omits the resampling step entirely in those cases.
    """
    if method in NO_TREATMENT or method in ALGORITHM_LEVEL:
        return None

    if method == "random_oversampling":
        return RandomOverSampler(random_state=random_state)

    if method == "random_undersampling":
        return RandomUnderSampler(random_state=random_state)

    if method == "smote":
        return SMOTE(random_state=random_state)

    if method == "adasyn":
        return ADASYN(random_state=random_state)

    if method == "smoteenn":
        return SMOTEENN(random_state=random_state)

    if method == "smotetomek":
        return SMOTETomek(random_state=random_state)

    raise ValueError(f"Unknown imbalance method '{method}'.")


def uses_class_weight(method: str) -> bool:
    """Whether ``method`` is applied through the classifier's class weights."""
    return method in ALGORITHM_LEVEL


def method_family(method: str) -> str:
    """Category label used in the comparison tables of Chapter 5."""
    return {
        "none": "Baseline",
        "random_oversampling": "Data-level (resampling)",
        "random_undersampling": "Data-level (resampling)",
        "smote": "Data-level (synthetic)",
        "adasyn": "Data-level (synthetic)",
        "smoteenn": "Hybrid",
        "smotetomek": "Hybrid",
        "cost_sensitive": "Algorithm-level",
    }[method]
