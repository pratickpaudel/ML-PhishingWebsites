"""
Statistical significance testing (Step 9 of the pipeline).

Two complementary tests are applied:

* **Friedman test** - a non-parametric test for differences across more than two
  related groups. It is used to ask whether the classifiers differ overall, and
  whether the imbalance treatment methods differ overall, by ranking their
  performance within each experimental block.

* **McNemar's test** - a paired test on the *same* test instances. It compares
  two fitted models by counting the cases where one is correct and the other is
  wrong, which is more informative than comparing summary metrics because it
  accounts for the correlation between predictions on shared data.

Where the Friedman test is significant, post-hoc pairwise Wilcoxon signed-rank
tests with Holm-Bonferroni correction identify *which* pairs differ. Holm is
used rather than plain Bonferroni because it is uniformly more powerful while
still controlling the family-wise error rate.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

from config import ALPHA, CLASSIFIER_LABELS, METHOD_LABELS, RESULTS_DIR

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FULL_DIR = RESULTS_DIR / "full"


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------
def load_predictions(config_id: str) -> dict[str, np.ndarray]:
    """Load the saved test-set predictions for one configuration."""
    path = PREDICTIONS_DIR / f"{config_id}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved predictions for '{config_id}'. Run experiment.py first."
        )
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------
def mcnemar_test(config_a: str, config_b: str, alpha: float = ALPHA) -> dict:
    """Compare two configurations on their shared test set.

    The two models must have been evaluated on identical test instances, which
    holds whenever they share the same dataset and imbalance ratio (the split is
    driven by a fixed random seed).
    """
    a = load_predictions(config_a)
    b = load_predictions(config_b)

    if not np.array_equal(a["y_true"], b["y_true"]):
        raise ValueError(
            "McNemar's test requires identical test sets; "
            f"'{config_a}' and '{config_b}' differ."
        )

    y_true = a["y_true"]
    correct_a = a["y_pred"] == y_true
    correct_b = b["y_pred"] == y_true

    # 2x2 contingency table of agreement/disagreement.
    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))

    table = [[both_correct, a_only], [b_only, both_wrong]]

    # The exact binomial test is used when the discordant counts are small,
    # where the chi-squared approximation is unreliable.
    discordant = a_only + b_only
    use_exact = discordant < 25
    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    return {
        "model_a": config_a,
        "model_b": config_b,
        "both_correct": both_correct,
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "both_wrong": both_wrong,
        "n_discordant": discordant,
        "test": "exact binomial" if use_exact else "chi-squared (continuity corrected)",
        "statistic": round(float(result.statistic), 4),
        "p_value": float(result.pvalue),
        "significant": bool(result.pvalue < alpha),
        "favours": (
            "model_a" if a_only > b_only else "model_b" if b_only > a_only else "tie"
        ),
    }


def mcnemar_top_models(
    results: pd.DataFrame,
    metric: str = "f1",
    top_n: int = 2,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Run McNemar's test between the best configurations within each dataset.

    Comparisons are made within a dataset because McNemar's test requires the
    same test instances.
    """
    rows = []
    for dataset, group in results.groupby("dataset"):
        best = group.nlargest(top_n, metric)
        for cfg_a, cfg_b in itertools.combinations(best["config_id"], 2):
            row = mcnemar_test(cfg_a, cfg_b, alpha=alpha)
            row["dataset"] = dataset
            row["comparison"] = f"top-{top_n} by {metric}"
            rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_key_comparisons(
    all_results: pd.DataFrame,
    metric: str = "f1",
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Run the McNemar comparisons that carry the most interpretive weight.

    Ranking the top two configurations often pairs two nearly identical models
    (for example SMOTE and SMOTETomek can yield the same predictions when no
    Tomek links are removed), which produces a vacuous test. The comparisons
    below are chosen instead because each answers a distinct question:

    1. *best vs untreated baseline, same classifier* - did imbalance treatment
       actually change predictive behaviour?
    2. *best vs best of each other classifier* - is the leading classifier
       genuinely better, not just marginally ahead on a summary metric?
    3. *best vs worst* - how large is the spread across the design space?

    ``all_results`` should include the untreated baseline rows.
    """
    rows = []

    # Comparisons must stay within a single (dataset, seed) combination, because
    # a different seed produces a different train-test split and McNemar's test
    # requires both models to have been evaluated on identical instances.
    group_keys = ["dataset"]
    if "seed" in all_results.columns:
        group_keys.append("seed")

    for keys, group in all_results.groupby(group_keys):
        keys = keys if isinstance(keys, tuple) else (keys,)
        dataset = keys[0]
        seed = keys[1] if len(keys) > 1 else None

        treated = group[group["imbalance_method"] != "none"]
        if treated.empty:
            continue

        best = treated.loc[treated[metric].idxmax()]
        best_id = best["config_id"]

        def add(cfg_b, label):
            if cfg_b == best_id:
                return
            try:
                row = mcnemar_test(best_id, cfg_b, alpha=alpha)
            except FileNotFoundError:
                return
            row["dataset"] = dataset
            if seed is not None:
                row["seed"] = seed
            row["comparison"] = label
            rows.append(row)

        # 1. Best treated configuration vs the untreated baseline.
        baseline = group[
            (group["imbalance_method"] == "none")
            & (group["classifier"] == best["classifier"])
        ]
        if not baseline.empty:
            add(baseline.iloc[0]["config_id"], "best vs untreated baseline")

        # 2. Best configuration vs the best of each competing classifier.
        for clf, clf_group in treated.groupby("classifier"):
            if clf == best["classifier"]:
                continue
            add(
                clf_group.loc[clf_group[metric].idxmax()]["config_id"],
                f"best vs best {clf}",
            )

        # 3. Best vs worst overall.
        add(treated.loc[treated[metric].idxmin()]["config_id"], "best vs worst")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------
def friedman_test(
    results: pd.DataFrame,
    group_col: str,
    block_cols: list[str],
    metric: str = "f1",
    alpha: float = ALPHA,
) -> dict:
    """Test whether the levels of ``group_col`` differ in ``metric``.

    Parameters
    ----------
    group_col
        The factor being compared, e.g. ``"classifier"`` or ``"imbalance_method"``.
    block_cols
        Columns identifying the matched blocks. For a classifier comparison the
        blocks are (dataset, imbalance method), so each classifier is measured
        once per block.
    """
    pivot = results.pivot_table(
        index=block_cols,
        columns=group_col,
        values=metric,
    ).dropna()

    if pivot.shape[0] < 2 or pivot.shape[1] < 3:
        return {
            "comparison": group_col,
            "metric": metric,
            "error": (
                "Friedman's test needs at least 3 groups and 2 blocks; "
                f"got {pivot.shape[1]} groups and {pivot.shape[0]} blocks."
            ),
        }

    samples = [pivot[c].to_numpy() for c in pivot.columns]
    statistic, p_value = friedmanchisquare(*samples)

    # Mean rank per group (rank 1 = best), the conventional Friedman summary.
    mean_ranks = pivot.rank(axis=1, ascending=False).mean().sort_values()

    return {
        "comparison": group_col,
        "metric": metric,
        "n_groups": int(pivot.shape[1]),
        "n_blocks": int(pivot.shape[0]),
        "statistic": round(float(statistic), 4),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "mean_ranks": {k: round(float(v), 3) for k, v in mean_ranks.items()},
        "best_group": mean_ranks.index[0],
    }


# ---------------------------------------------------------------------------
# Post-hoc pairwise comparisons
# ---------------------------------------------------------------------------
def holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> list[bool]:
    """Return per-hypothesis significance flags under Holm-Bonferroni control.

    P-values are sorted ascending and compared against ``alpha / (m - i)``.
    Testing stops at the first non-rejection, and all remaining hypotheses are
    retained, which is what preserves the family-wise error rate.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    flags = [False] * m

    for i, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - i):
            flags[idx] = True
        else:
            break

    return flags


def posthoc_wilcoxon(
    results: pd.DataFrame,
    group_col: str,
    block_cols: list[str],
    metric: str = "f1",
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction."""
    pivot = results.pivot_table(
        index=block_cols,
        columns=group_col,
        values=metric,
    ).dropna()

    pairs, raw_p, stats_ = [], [], []
    for a, b in itertools.combinations(pivot.columns, 2):
        x, y = pivot[a].to_numpy(), pivot[b].to_numpy()
        if np.allclose(x, y):
            # Wilcoxon is undefined when every difference is zero.
            statistic, p = 0.0, 1.0
        else:
            statistic, p = wilcoxon(x, y)
        pairs.append((a, b))
        stats_.append(float(statistic))
        raw_p.append(float(p))

    flags = holm_bonferroni(raw_p, alpha=alpha)

    return pd.DataFrame(
        {
            "group_a": [p[0] for p in pairs],
            "group_b": [p[1] for p in pairs],
            "mean_a": [round(pivot[p[0]].mean(), 4) for p in pairs],
            "mean_b": [round(pivot[p[1]].mean(), 4) for p in pairs],
            "statistic": stats_,
            "p_value": raw_p,
            "significant_holm": flags,
        }
    ).sort_values("p_value")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_all_tests(
    results_file: str = "results.csv",
    metric: str = "f1",
    exclude_baseline: bool = True,
    alpha: float = ALPHA,
    verbose: bool = True,
) -> dict:
    """Run the full statistical analysis and write the output tables."""
    all_results = pd.read_csv(RESULTS_DIR / results_file)

    if "error" in all_results.columns:
        all_results = all_results[all_results["error"].isna()]

    # The Friedman/post-hoc analysis compares treatment methods, so the
    # untreated baseline is excluded there but retained for McNemar comparisons.
    results = (
        all_results[all_results["imbalance_method"] != "none"]
        if exclude_baseline
        else all_results
    )

    # When the sweep has been repeated under several seeds, each seed forms an
    # additional matched block. This is what lifts the post-hoc tests out of the
    # very low power regime that a single replication suffers from.
    extra_block = ["seed"] if "seed" in results.columns and results["seed"].nunique() > 1 else []

    friedman_classifiers = friedman_test(
        results, "classifier", ["dataset", "imbalance_method"] + extra_block, metric, alpha
    )
    friedman_methods = friedman_test(
        results, "imbalance_method", ["dataset", "classifier"] + extra_block, metric, alpha
    )

    friedman_df = pd.DataFrame(
        [
            {
                "comparison_group": "Classifiers",
                "metric": metric,
                "n_groups": friedman_classifiers.get("n_groups"),
                "n_blocks": friedman_classifiers.get("n_blocks"),
                "statistic": friedman_classifiers.get("statistic"),
                "p_value": friedman_classifiers.get("p_value"),
                "significant": friedman_classifiers.get("significant"),
                "best_group": friedman_classifiers.get("best_group"),
            },
            {
                "comparison_group": "Imbalance methods",
                "metric": metric,
                "n_groups": friedman_methods.get("n_groups"),
                "n_blocks": friedman_methods.get("n_blocks"),
                "statistic": friedman_methods.get("statistic"),
                "p_value": friedman_methods.get("p_value"),
                "significant": friedman_methods.get("significant"),
                "best_group": friedman_methods.get("best_group"),
            },
        ]
    )
    friedman_df.to_csv(RESULTS_DIR / "friedman_tests.csv", index=False)

    posthoc_clf = posthoc_wilcoxon(
        results, "classifier", ["dataset", "imbalance_method"] + extra_block, metric, alpha
    )
    posthoc_mth = posthoc_wilcoxon(
        results, "imbalance_method", ["dataset", "classifier"] + extra_block, metric, alpha
    )
    posthoc_clf.to_csv(RESULTS_DIR / "posthoc_classifiers.csv", index=False)
    posthoc_mth.to_csv(RESULTS_DIR / "posthoc_methods.csv", index=False)

    mcnemar_df = mcnemar_key_comparisons(all_results, metric=metric, alpha=alpha)
    mcnemar_df.to_csv(RESULTS_DIR / "mcnemar_tests.csv", index=False)

    if verbose:
        _print_report(
            friedman_classifiers, friedman_methods, posthoc_clf, posthoc_mth, mcnemar_df, metric
        )

    return {
        "friedman_classifiers": friedman_classifiers,
        "friedman_methods": friedman_methods,
        "posthoc_classifiers": posthoc_clf,
        "posthoc_methods": posthoc_mth,
        "mcnemar": mcnemar_df,
    }


def _print_report(fried_clf, fried_mth, posthoc_clf, posthoc_mth, mcnemar_df, metric):
    def fmt_p(p):
        return "< 0.001" if p < 0.001 else f"{p:.4f}"

    print("=" * 78)
    print(f"STATISTICAL SIGNIFICANCE TESTING  (metric = {metric}, alpha = {ALPHA})")
    print("=" * 78)

    for title, res, labels in (
        ("Classifiers", fried_clf, CLASSIFIER_LABELS),
        ("Imbalance methods", fried_mth, METHOD_LABELS),
    ):
        print(f"\nFriedman test - {title}")
        if "error" in res:
            print(f"  {res['error']}")
            continue
        print(f"  H0: no difference in {metric} across {title.lower()}")
        print(f"  chi-squared = {res['statistic']}, p = {fmt_p(res['p_value'])}, "
              f"blocks = {res['n_blocks']}, groups = {res['n_groups']}")
        print(f"  -> H0 {'REJECTED' if res['significant'] else 'NOT rejected'} at alpha = {ALPHA}")
        print("  Mean ranks (1 = best):")
        for name, rank in res["mean_ranks"].items():
            print(f"    {labels.get(name, name):28s} {rank}")

    for title, df in (("classifiers", posthoc_clf), ("imbalance methods", posthoc_mth)):
        sig = df[df["significant_holm"]]
        print(f"\nPost-hoc Wilcoxon (Holm-corrected) - {title}: "
              f"{len(sig)} of {len(df)} pairs significant")
        for _, r in sig.iterrows():
            print(f"    {r['group_a']} ({r['mean_a']}) vs {r['group_b']} ({r['mean_b']}) "
                  f"p = {fmt_p(r['p_value'])}")

    print("\nMcNemar's test - paired comparisons on identical test sets")
    if mcnemar_df.empty:
        print("  No comparisons available.")
    else:
        for _, r in mcnemar_df.iterrows():
            print(f"  [{r['dataset']}] {r['comparison']}")
            print(f"    A: {r['model_a']}")
            print(f"    B: {r['model_b']}")
            print(f"    contingency: both correct = {r['both_correct']}, "
                  f"A only = {r['a_correct_b_wrong']}, B only = {r['b_correct_a_wrong']}, "
                  f"both wrong = {r['both_wrong']}")
            print(f"    {r['test']}: statistic = {r['statistic']}, p = {fmt_p(r['p_value'])}")
            print(f"    -> {'significant' if r['significant'] else 'not significant'} "
                  f"(favours {r['favours']})")

    print("\n" + "=" * 78)
    print(f"Tables written to {RESULTS_DIR}")
    print("=" * 78)




# ---------------------------------------------------------------------------
# Full-scale testing
# ---------------------------------------------------------------------------
# The full-scale run uses a single seed, so the matched blocks available to the
# rank-based tests come only from (dataset x classifier) and
# (dataset x imbalance method) combinations, not from replications as well.
# Power therefore differs sharply between the three tests and the report says so:
#
# * McNemar is unaffected and in fact stronger. It is paired on a single test
#   set and needs no replications, and the full-scale test partitions (17,730
#   and 23,320 instances) are four to six times larger.
# * The classifier comparison has 14 blocks across 3 groups, which is adequate.
# * The method comparison has only 6 blocks across 7 groups. A Wilcoxon
#   signed-rank test over 6 pairs cannot return a two-sided p below 0.031,
#   while Holm-Bonferroni over 21 pairs demands 0.0024 at its tightest. No
#   pairwise method comparison can reach significance whatever the data show.
#   That is a property of the design, not a finding, and must not be reported
#   as evidence of equivalence.


def _min_attainable_p(n_blocks: int) -> float:
    """Smallest two-sided p a Wilcoxon signed-rank test can return.

    With n matched pairs the most extreme outcome is all differences sharing a
    sign, of two-sided probability 2 / 2**n. Below roughly 8 blocks this floor
    collides with any multiple-comparison correction.
    """
    return 2 / (2 ** n_blocks)


def _annotate_power(posthoc: pd.DataFrame, n_blocks: int, alpha: float) -> pd.DataFrame:
    """Flag comparisons that could not have reached significance either way.

    Without this a reader cannot distinguish "no difference was found" from
    "no difference could have been found" - very different claims.
    """
    if posthoc.empty or n_blocks == 0:
        return posthoc
    posthoc = posthoc.copy()
    posthoc["n_blocks"] = n_blocks
    posthoc["test_underpowered"] = _min_attainable_p(n_blocks) > alpha / len(posthoc)
    return posthoc


def run_full_tests(
    results_file: str = "results.csv",
    metric: str = "f1",
    alpha: float = ALPHA,
    exclude_baseline: bool = True,
) -> dict:
    """Friedman, post-hoc Wilcoxon and McNemar tests on the full-scale results.

    Reads results/full/ and writes back to it using the same filenames the
    reduced-scale tests use, so the two sets are directly comparable.
    """
    global PREDICTIONS_DIR

    path = FULL_DIR / results_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/experiment.py --full"
        )

    results = pd.read_csv(path)
    if "error" in results.columns:
        results = results[results["error"].isna()]

    treated = (
        results[results["imbalance_method"] != "none"] if exclude_baseline else results
    )

    friedman = pd.DataFrame([
        friedman_test(treated, "classifier", ["dataset", "imbalance_method"],
                      metric=metric, alpha=alpha),
        friedman_test(treated, "imbalance_method", ["dataset", "classifier"],
                      metric=metric, alpha=alpha),
    ])

    posthoc_clf = posthoc_wilcoxon(
        treated, "classifier", ["dataset", "imbalance_method"], metric=metric, alpha=alpha
    )
    posthoc_mth = posthoc_wilcoxon(
        treated, "imbalance_method", ["dataset", "classifier"], metric=metric, alpha=alpha
    )

    blocks = {
        r["comparison"]: int(r["n_blocks"])
        for _, r in friedman.iterrows()
        if pd.notna(r.get("n_blocks"))
    }
    posthoc_clf = _annotate_power(posthoc_clf, blocks.get("classifier", 0), alpha)
    posthoc_mth = _annotate_power(posthoc_mth, blocks.get("imbalance_method", 0), alpha)

    # McNemar reads the full-scale prediction dumps rather than results/.
    previous = PREDICTIONS_DIR
    PREDICTIONS_DIR = FULL_DIR / "predictions"
    try:
        mcnemar_df = mcnemar_key_comparisons(results, metric=metric, alpha=alpha)
    finally:
        PREDICTIONS_DIR = previous

    if not mcnemar_df.empty:
        mcnemar_df = mcnemar_df.sort_values("p_value").reset_index(drop=True)
        mcnemar_df["significant_holm"] = holm_bonferroni(
            mcnemar_df["p_value"].tolist(), alpha=alpha
        )
        total = (
            mcnemar_df["both_correct"] + mcnemar_df["a_correct_b_wrong"]
            + mcnemar_df["b_correct_a_wrong"] + mcnemar_df["both_wrong"]
        )
        # The effect size McNemar does not report: on large test sets a tiny
        # disagreement can be highly significant, so both must be read together.
        mcnemar_df["disagreement_pct"] = (
            100 * mcnemar_df["n_discordant"] / total
        ).round(2)

    friedman.to_csv(FULL_DIR / "friedman_tests.csv", index=False)
    posthoc_clf.to_csv(FULL_DIR / "posthoc_classifiers.csv", index=False)
    posthoc_mth.to_csv(FULL_DIR / "posthoc_methods.csv", index=False)
    mcnemar_df.to_csv(FULL_DIR / "mcnemar_tests.csv", index=False)

    _print_full_report(friedman, posthoc_clf, posthoc_mth, mcnemar_df, alpha)
    print(f"\nWritten to {FULL_DIR}")
    return {
        "friedman": friedman,
        "posthoc_classifiers": posthoc_clf,
        "posthoc_methods": posthoc_mth,
        "mcnemar": mcnemar_df,
    }


def _fmt_p(p) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def _print_full_report(friedman, posthoc_clf, posthoc_mth, mcnemar_df, alpha) -> None:
    print("=" * 78)
    print("STATISTICAL TESTS - FULL SCALE (single seed)")
    print("=" * 78)

    print("\nFRIEDMAN")
    for _, r in friedman.iterrows():
        if pd.notna(r.get("error")):
            print(f"  {r['comparison']}: {r['error']}")
            continue
        mark = "significant" if r["significant"] else "not significant"
        print(f"  {r['comparison']}: chi2={r['statistic']}, p={_fmt_p(r['p_value'])} "
              f"({mark}), {r['n_blocks']} blocks, best = {r['best_group']}")
        print(f"    mean ranks: {r['mean_ranks']}")

    for name, df in (("CLASSIFIERS", posthoc_clf), ("METHODS", posthoc_mth)):
        print(f"\nPOST-HOC WILCOXON - {name}")
        if df.empty:
            print("  no comparisons")
            continue
        if bool(df["test_underpowered"].iloc[0]):
            n = int(df["n_blocks"].iloc[0])
            print(f"  WARNING: {n} blocks over {len(df)} pairs. The smallest attainable")
            print(f"  two-sided p is {_min_attainable_p(n):.4f}, but Holm requires "
                  f"{alpha / len(df):.4f} at its tightest.")
            print("  No pair can reach significance regardless of the data.")
        for _, r in df.head(6).iterrows():
            mark = "*" if r["significant_holm"] else " "
            print(f"  {mark} {str(r['group_a']):22s} vs {str(r['group_b']):22s} "
                  f"p={_fmt_p(r['p_value'])}  ({r['mean_a']} vs {r['mean_b']})")

    print("\nMcNEMAR")
    if mcnemar_df.empty:
        print("  no comparisons")
        return
    for dataset, group in mcnemar_df.groupby("dataset"):
        print(f"  {dataset}")
        for _, r in group.iterrows():
            mark = "significant" if r["significant_holm"] else "not significant"
            print(f"    {r['comparison']}: p={_fmt_p(r['p_value'])} ({mark})")
            print(f"      {r['n_discordant']} discordant, "
                  f"{r['disagreement_pct']}% of test set, favours {r['favours']}")
    print("\n  Full-scale test partitions are 4-6x larger than at reduced scale, so")
    print("  significance here does not imply a larger effect. Read the")
    print("  disagreement percentage alongside every p-value.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run statistical significance tests.")
    parser.add_argument("--results", default=None,
                        help="Defaults to results.csv, read from results/full/ with --full.")
    parser.add_argument("--metric", default="f1")
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="Test the full-scale results in results/full/.")
    args = parser.parse_args()

    if args.full:
        run_full_tests(
            results_file=args.results or "results.csv",
            metric=args.metric,
            exclude_baseline=not args.include_baseline,
        )
    else:
        run_all_tests(
            results_file=args.results or "results.csv",
            metric=args.metric,
            exclude_baseline=not args.include_baseline,
        )