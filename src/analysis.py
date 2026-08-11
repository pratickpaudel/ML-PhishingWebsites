"""
Result aggregation and table generation (Step 8 of the pipeline).

Turns the raw ``results.csv`` produced by ``experiment.py`` into the summary
tables required by Chapter 5:

* Table 5.1 / 5.2 - per-dataset performance of every configuration
* Table 5.3        - mean performance by classifier
* Table 5.4        - mean performance by imbalance method
* Table 5.5        - best and worst configuration per dataset

Every table is written both as a CSV (for inspection) and as a Markdown file
(for pasting into the dissertation).
"""

from __future__ import annotations

import argparse

import pandas as pd

from config import (
    CLASSIFIER_LABELS,
    DATASET_LABELS,
    METHOD_LABELS,
    RESULTS_DIR,
)

# Metrics reported in the dissertation tables, in presentation order.
REPORT_METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]

# Ordering used so tables read consistently rather than alphabetically.
METHOD_ORDER = [
    "none",
    "random_oversampling",
    "random_undersampling",
    "smote",
    "adasyn",
    "smoteenn",
    "smotetomek",
    "cost_sensitive",
]
CLASSIFIER_ORDER = ["decision_tree", "random_forest", "svm"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_results(results_file: str = "results.csv") -> pd.DataFrame:
    """Load the results table, dropping any configurations that failed."""
    df = pd.read_csv(RESULTS_DIR / results_file)
    if "error" in df.columns:
        failed = df["error"].notna().sum()
        if failed:
            print(f"Warning: {failed} configuration(s) failed and were excluded.")
        df = df[df["error"].isna()].drop(columns=["error"])
    return df


def _ordered(df: pd.DataFrame, col: str, order: list[str]) -> pd.DataFrame:
    """Sort ``col`` by a fixed presentation order rather than alphabetically."""
    present = [v for v in order if v in df[col].unique()]
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=present, ordered=True)
    return df.sort_values(col)


def _write(df: pd.DataFrame, name: str, index: bool = False) -> None:
    """Write a table as both CSV and Markdown."""
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=index)
    with open(RESULTS_DIR / f"{name}.md", "w") as fh:
        fh.write(df.to_markdown(index=index))
    print(f"  wrote {name}.csv / {name}.md")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table_per_dataset(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Full configuration-level results for one dataset (Tables 5.1 / 5.2).

    If the sweep was repeated under several seeds, metrics are averaged across
    replications so that each classifier/method pair appears exactly once.
    """
    sub = df[df["dataset"] == dataset].copy()

    if "seed" in sub.columns and sub["seed"].nunique() > 1:
        sub = (
            sub.groupby(["classifier", "imbalance_method"], as_index=False)[REPORT_METRICS]
            .mean()
        )

    sub = _ordered(sub, "imbalance_method", METHOD_ORDER)
    sub = sub.sort_values(["classifier", "imbalance_method"])

    out = sub[["classifier", "imbalance_method"] + REPORT_METRICS].copy()
    out["classifier"] = out["classifier"].map(CLASSIFIER_LABELS)
    out["imbalance_method"] = out["imbalance_method"].map(METHOD_LABELS)
    out[REPORT_METRICS] = out[REPORT_METRICS].round(4)

    return out.rename(
        columns={"classifier": "Classifier", "imbalance_method": "Imbalance method"}
    )


def table_by_classifier(df: pd.DataFrame, exclude_baseline: bool = True) -> pd.DataFrame:
    """Mean performance per classifier across all methods (Table 5.3)."""
    sub = df[df["imbalance_method"] != "none"] if exclude_baseline else df
    agg = sub.groupby("classifier")[REPORT_METRICS].mean().round(4).reset_index()
    agg = _ordered(agg, "classifier", CLASSIFIER_ORDER)
    agg["classifier"] = agg["classifier"].map(CLASSIFIER_LABELS)
    return agg.rename(columns={"classifier": "Classifier"})


def table_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Mean performance per imbalance method across classifiers (Table 5.4)."""
    agg = df.groupby("imbalance_method")[REPORT_METRICS].mean().round(4).reset_index()
    agg = _ordered(agg, "imbalance_method", METHOD_ORDER)
    agg["imbalance_method"] = agg["imbalance_method"].map(METHOD_LABELS)
    return agg.rename(columns={"imbalance_method": "Imbalance method"})


def table_best_worst(
    df: pd.DataFrame, metric: str = "f1", exclude_baseline: bool = True
) -> pd.DataFrame:
    """Best and worst configuration per dataset (Table 5.5)."""
    sub = df[df["imbalance_method"] != "none"] if exclude_baseline else df

    # With repeated runs, rank conditions by their mean across replications
    # rather than letting a single lucky seed decide the winner.
    if "seed" in sub.columns and sub["seed"].nunique() > 1:
        sub = (
            sub.groupby(["dataset", "classifier", "imbalance_method"], as_index=False)[
                REPORT_METRICS
            ]
            .mean()
        )

    rows = []

    for dataset, group in sub.groupby("dataset"):
        best = group.loc[group[metric].idxmax()]
        worst = group.loc[group[metric].idxmin()]

        def describe(r):
            return (
                f"{CLASSIFIER_LABELS[r['classifier']]} + "
                f"{METHOD_LABELS[r['imbalance_method']]}"
            )

        rows.append(
            {
                "Dataset": DATASET_LABELS.get(dataset, dataset),
                "Best configuration": describe(best),
                "Best F1": round(best["f1"], 4),
                "Best recall": round(best["recall"], 4),
                "Best PR-AUC": round(best["pr_auc"], 4),
                "Worst configuration": describe(worst),
                "Worst F1": round(worst["f1"], 4),
                "Worst recall": round(worst["recall"], 4),
                "Worst PR-AUC": round(worst["pr_auc"], 4),
                "F1 gap": round(best["f1"] - worst["f1"], 4),
            }
        )

    return pd.DataFrame(rows)


def table_treatment_effect(df: pd.DataFrame) -> pd.DataFrame:
    """Change in performance relative to the untreated baseline.

    This isolates the contribution of imbalance treatment itself, which is the
    central question of the dissertation: each treated configuration is compared
    against the same classifier trained on untreated data.
    """
    # Each treated run is matched to the baseline from the *same* replication,
    # so the seed must be part of the key when the sweep has been repeated.
    key_cols = ["dataset", "classifier"]
    if "seed" in df.columns:
        key_cols.append("seed")

    baseline = (
        df[df["imbalance_method"] == "none"]
        .set_index(key_cols)[REPORT_METRICS]
        .sort_index()
    )
    if baseline.empty:
        return pd.DataFrame()

    treated = df[df["imbalance_method"] != "none"].copy()
    rows = []

    for _, r in treated.iterrows():
        key = tuple(r[c] for c in key_cols)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        rows.append(
            {
                "Dataset": DATASET_LABELS.get(r["dataset"], r["dataset"]),
                "Classifier": CLASSIFIER_LABELS[r["classifier"]],
                "Imbalance method": METHOD_LABELS[r["imbalance_method"]],
                "delta_recall": r["recall"] - base["recall"],
                "delta_precision": r["precision"] - base["precision"],
                "delta_f1": r["f1"] - base["f1"],
                "delta_pr_auc": r["pr_auc"] - base["pr_auc"],
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    delta_cols = ["delta_recall", "delta_precision", "delta_f1", "delta_pr_auc"]

    # Average the deltas over replications so one row describes one condition.
    out = (
        out.groupby(["Dataset", "Classifier", "Imbalance method"], as_index=False)[delta_cols]
        .mean()
        .round(4)
    )
    return out.sort_values("delta_f1", ascending=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def generate_all(results_file: str = "results.csv", metric: str = "f1") -> dict:
    """Generate every Chapter 5 table and write it to the results directory."""
    df = load_results(results_file)
    tables = {}

    print("Generating tables:")

    for i, dataset in enumerate(sorted(df["dataset"].unique()), start=1):
        t = table_per_dataset(df, dataset)
        tables[f"table_5_{i}_{dataset}"] = t
        _write(t, f"table_5_{i}_performance_{dataset}")

    t3 = table_by_classifier(df)
    tables["table_5_3_classifiers"] = t3
    _write(t3, "table_5_3_by_classifier")

    t4 = table_by_method(df)
    tables["table_5_4_methods"] = t4
    _write(t4, "table_5_4_by_method")

    t5 = table_best_worst(df, metric=metric)
    tables["table_5_5_best_worst"] = t5
    _write(t5, "table_5_5_best_worst")

    t6 = table_treatment_effect(df)
    if not t6.empty:
        tables["treatment_effect"] = t6
        _write(t6, "table_treatment_effect")

    _print_summary(t3, t4, t5, t6)
    return tables


def _print_summary(t3, t4, t5, t6) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY OF RESULTS")
    print("=" * 78)

    print("\nMean performance by classifier (treated configurations only):")
    print(t3.to_string(index=False))

    print("\nMean performance by imbalance method:")
    print(t4.to_string(index=False))

    print("\nBest and worst configuration per dataset:")
    for _, r in t5.iterrows():
        print(f"\n  {r['Dataset']}")
        print(f"    best : {r['Best configuration']}")
        print(f"           F1={r['Best F1']}  recall={r['Best recall']}  PR-AUC={r['Best PR-AUC']}")
        print(f"    worst: {r['Worst configuration']}")
        print(f"           F1={r['Worst F1']}  recall={r['Worst recall']}  PR-AUC={r['Worst PR-AUC']}")
        print(f"    F1 gap between best and worst: {r['F1 gap']}")

    if t6 is not None and not t6.empty:
        print("\nLargest gains over the untreated baseline (by F1):")
        print(t6.head(5).to_string(index=False))
        print("\nLargest losses relative to the untreated baseline (by F1):")
        print(t6.tail(5).to_string(index=False))

        print("\nMean change in recall by method (treatment effect):")
        recall_effect = (
            t6.groupby("Imbalance method")["delta_recall"]
            .mean()
            .round(4)
            .sort_values(ascending=False)
        )
        print(recall_effect.to_string())

    print("\n" + "=" * 78)
    print(f"All tables written to {RESULTS_DIR}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 5 result tables.")
    parser.add_argument("--results", default="results.csv")
    parser.add_argument("--metric", default="f1")
    args = parser.parse_args()
    generate_all(results_file=args.results, metric=args.metric)


if __name__ == "__main__":
    main()
