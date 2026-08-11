"""
Experiment runner (Steps 1-7 executed across the full comparison matrix).

Each configuration is one (dataset, imbalance method, classifier) triple. For
every configuration the runner:

1. loads the dataset and induces the controlled imbalance ratio,
2. cleans the features and produces the stratified train-test split,
3. fits a grid search whose pipeline resamples inside each CV fold,
4. refits the winning hyperparameters on the full treated training set,
5. evaluates once on the untouched test set.

Per-configuration test predictions are persisted alongside the metrics because
McNemar's test operates on paired predictions rather than summary scores.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd

from config import (
    CLASSIFIERS,
    CORE_IMBALANCE_METHODS,
    DATASETS,
    IMBALANCE_METHODS,
    MINORITY_RATIO,
    RANDOM_STATE,
    RESULTS_DIR,
    SUBSAMPLE_SIZE,
)
from data_loader import load_dataset
from evaluation import evaluate
from imbalance import method_family
from models import build_pipeline, build_search, get_scores
from preprocessing import prepare, split_summary

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


def config_id(
    dataset: str,
    method: str,
    classifier: str,
    ratio: float | None,
    seed: int = RANDOM_STATE,
) -> str:
    """Stable identifier used for result rows and prediction filenames.

    The seed is part of the identifier so that repeated runs do not overwrite
    one another's saved predictions. ``ratio`` is ``None`` in the main
    experiments, where each dataset's own class distribution is used, and is
    recorded as ``rnative`` in that case.
    """
    ratio_tag = "native" if ratio is None else f"{int(round(ratio * 100))}"
    return f"{dataset}__{method}__{classifier}__r{ratio_tag}__s{seed}"


def run_configuration(
    dataset: str,
    method: str,
    classifier: str,
    minority_ratio: float = MINORITY_RATIO,
    save_predictions: bool = True,
    seed: int = RANDOM_STATE,
) -> dict:
    """Execute a single experimental configuration and return its metrics.

    ``seed`` drives the induced downsampling, the train-test split, the
    cross-validation folds, the sampler and the classifier. Repeating the sweep
    under different seeds therefore produces independent replications rather
    than identical re-runs.
    """
    started = time.time()

    X, y = load_dataset(
        dataset,
        minority_ratio=minority_ratio,
        random_state=seed,
        subsample=SUBSAMPLE_SIZE,
    )
    X_train, X_test, y_train, y_test = prepare(X, y, random_state=seed)

    search = build_search(classifier, method, random_state=seed)
    with warnings.catch_warnings():
        # Convergence and sampling warnings are expected on some folds and are
        # not informative once the configuration completes.
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_scores = get_scores(best_model, X_test)

    metrics = evaluate(y_test, y_pred, y_scores)

    cid = config_id(dataset, method, classifier, minority_ratio, seed)
    if save_predictions:
        np.savez_compressed(
            PREDICTIONS_DIR / f"{cid}.npz",
            y_true=np.asarray(y_test),
            y_pred=np.asarray(y_pred),
            y_scores=np.asarray(y_scores),
        )

    row = {
        "config_id": cid,
        "dataset": dataset,
        "imbalance_method": method,
        "method_family": method_family(method),
        "classifier": classifier,
        "minority_ratio": minority_ratio,
        "seed": seed,
        **metrics,
        "cv_best_score": round(float(search.best_score_), 4),
        "best_params": json.dumps(
            {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
        ),
        "runtime_sec": round(time.time() - started, 1),
    }
    row.update(split_summary(y_train, y_test))
    return row


def run_all(
    datasets=None,
    methods=None,
    classifiers=None,
    minority_ratio: float = MINORITY_RATIO,
    include_baseline: bool = True,
    output_name: str = "results.csv",
    verbose: bool = True,
    seeds=None,
) -> pd.DataFrame:
    """Run the full comparison matrix and write the results table to disk.

    Parameters
    ----------
    seeds
        One or more random seeds. Each seed repeats the entire matrix as an
        independent replication, which increases the number of matched blocks
        available to the Friedman and post-hoc tests. Defaults to the single
        project seed.
    """
    datasets = datasets or DATASETS
    classifiers = classifiers or CLASSIFIERS
    seeds = list(seeds) if seeds else [RANDOM_STATE]
    if methods is None:
        methods = IMBALANCE_METHODS if include_baseline else CORE_IMBALANCE_METHODS

    combinations = list(product(seeds, datasets, methods, classifiers))
    total = len(combinations)
    rows = []

    if verbose:
        seed_note = f"{len(seeds)} seed(s) {seeds}" if len(seeds) > 1 else f"seed {seeds[0]}"
        ratio_note = (
            "each dataset's own class distribution"
            if minority_ratio is None
            else f"an induced {minority_ratio:.0%} minority ratio"
        )
        print(f"Running {total} configurations using {ratio_note}, {seed_note}\n")

    for i, (seed, dataset, method, classifier) in enumerate(combinations, start=1):
        if verbose:
            print(
                f"[{i:>3}/{total}] s{seed:<3} {dataset:10s} {method:22s} {classifier:15s}",
                end=" ",
                flush=True,
            )

        try:
            row = run_configuration(
                dataset, method, classifier, minority_ratio, seed=seed
            )
            rows.append(row)
            if verbose:
                print(
                    f"F1={row['f1']:.4f} recall={row['recall']:.4f} "
                    f"PR-AUC={row['pr_auc']:.4f} ({row['runtime_sec']}s)"
                )
        except Exception as exc:  # keep the sweep alive, record the failure
            if verbose:
                print(f"FAILED: {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "config_id": config_id(
                        dataset, method, classifier, minority_ratio, seed
                    ),
                    "dataset": dataset,
                    "imbalance_method": method,
                    "classifier": classifier,
                    "minority_ratio": minority_ratio,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    results = pd.DataFrame(rows)
    out_path = RESULTS_DIR / output_name
    results.to_csv(out_path, index=False)

    if verbose:
        print(f"\nResults written to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Full-scale runs
# ---------------------------------------------------------------------------
# The functions below run the same matrix on the complete datasets (88,647 and
# 116,600 instances) rather than the SUBSAMPLE_SIZE reduction. Two differences
# from the reduced-scale sweep above, both deliberate:
#
# 1. Hyperparameters are not re-searched. Each configuration reuses the
#    selection already recorded in results/results_multiseed.csv and is fitted
#    once, which is what makes the run tractable: grid search at full scale is
#    a multi-day job dominated by the SVM. It is also the cleaner comparison,
#    since re-tuning would confound sample size with re-selection.
# 2. Each configuration is checkpointed on completion, so an interrupted run
#    resumes rather than restarting.
#
# Output goes to results/full/ and never touches results/.

FULL_DIR = RESULTS_DIR / "full"
FULL_ROWS_DIR = FULL_DIR / "rows"
FULL_PREDICTIONS_DIR = FULL_DIR / "predictions"

for _d in (FULL_DIR, FULL_ROWS_DIR, FULL_PREDICTIONS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Metrics carried into the scale comparison table.
COMPARISON_METRICS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]


def load_tuned_params(results_file: str = "results_multiseed.csv") -> dict:
    """Map (dataset, method, classifier, seed) -> tuned hyperparameters.

    Read from the reduced-scale results table, where the selection made by grid
    search was stored with the ``classifier__`` prefix already stripped.
    """
    path = RESULTS_DIR / results_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The full-scale run reuses the hyperparameters "
            "selected by grid search at reduced scale. Regenerate it with:\n"
            "  python src/experiment.py --seeds 42 1 2 --output results_multiseed.csv"
        )

    df = pd.read_csv(path)
    if "error" in df.columns:
        df = df[df["error"].isna()]

    params = {}
    for _, r in df.iterrows():
        if pd.isna(r.get("best_params")):
            continue
        key = (r["dataset"], r["imbalance_method"], r["classifier"], int(r["seed"]))
        params[key] = json.loads(r["best_params"])
    return params


def _resolve_params(tuned: dict, dataset: str, method: str, classifier: str, seed: int):
    """Find tuned parameters for one configuration, falling back across seeds."""
    key = (dataset, method, classifier, seed)
    if key in tuned:
        return tuned[key], f"seed {seed}"

    for (d, m, c, s), p in sorted(tuned.items(), key=lambda kv: kv[0][3]):
        if (d, m, c) == (dataset, method, classifier):
            return p, f"seed {s} (fallback)"

    raise KeyError(
        f"No tuned hyperparameters for {dataset}/{method}/{classifier}."
    )


def full_config_id(dataset: str, method: str, classifier: str, seed: int) -> str:
    """Identifier matching the reduced-scale scheme, tagged as full scale."""
    return f"{dataset}__{method}__{classifier}__rnative__s{seed}__full"


def run_full_configuration(
    dataset: str,
    method: str,
    classifier: str,
    params: dict,
    params_source: str,
    seed: int = RANDOM_STATE,
) -> dict:
    """Fit and evaluate one configuration on the complete dataset."""
    started = time.time()

    # subsample=None is the point of this run: all rows, not SUBSAMPLE_SIZE.
    X, y = load_dataset(dataset, minority_ratio=None, random_state=seed, subsample=None)
    X_train, X_test, y_train, y_test = prepare(X, y, random_state=seed)

    pipeline = build_pipeline(classifier, method, random_state=seed)
    if params:
        pipeline.set_params(**{f"classifier__{k}": v for k, v in params.items()})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_scores = get_scores(pipeline, X_test)
    metrics = evaluate(y_test, y_pred, y_scores)

    cid = full_config_id(dataset, method, classifier, seed)
    np.savez_compressed(
        FULL_PREDICTIONS_DIR / f"{cid}.npz",
        y_true=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_scores=np.asarray(y_scores),
    )

    row = {
        "config_id": cid,
        "dataset": dataset,
        "imbalance_method": method,
        "method_family": method_family(method),
        "classifier": classifier,
        "minority_ratio": None,
        "seed": seed,
        "scale": "full",
        "n_features": X_train.shape[1],
        **metrics,
        "best_params": json.dumps(params),
        "params_source": params_source,
        "runtime_sec": round(time.time() - started, 1),
    }
    row.update(split_summary(y_train, y_test))
    return row


def run_full_all(
    datasets=None,
    methods=None,
    classifiers=None,
    seed: int = RANDOM_STATE,
    tuning_file: str = "results_multiseed.csv",
    output_name: str = "results.csv",
    force: bool = False,
) -> pd.DataFrame:
    """Run the full-scale sweep, skipping configurations already completed."""
    datasets = datasets or DATASETS
    classifiers = classifiers or CLASSIFIERS
    methods = methods or IMBALANCE_METHODS

    tuned = load_tuned_params(tuning_file)
    combinations = list(product(datasets, methods, classifiers))
    total = len(combinations)

    print(f"Full-scale run: {total} configurations, seed {seed}")
    print(f"Checkpointing to {FULL_ROWS_DIR}\n")
    sweep_started = time.time()

    for i, (dataset, method, classifier) in enumerate(combinations, start=1):
        cid = full_config_id(dataset, method, classifier, seed)
        checkpoint = FULL_ROWS_DIR / f"{cid}.json"
        prefix = f"[{i:>2}/{total}] {dataset:10s} {method:22s} {classifier:15s}"

        if checkpoint.exists() and not force:
            print(f"{prefix} skipped (already complete)")
            continue

        print(prefix, end=" ", flush=True)
        try:
            params, source = _resolve_params(tuned, dataset, method, classifier, seed)
            row = run_full_configuration(
                dataset, method, classifier, params, source, seed
            )
            checkpoint.write_text(json.dumps(row, indent=2, default=str))
            print(
                f"F1={row['f1']:.4f} recall={row['recall']:.4f} "
                f"PR-AUC={row['pr_auc']:.4f} ({row['runtime_sec']}s)"
            )
        except Exception as exc:
            # Record the failure rather than letting it vanish, so a missing
            # cell cannot be mistaken for one never attempted.
            print(f"FAILED: {type(exc).__name__}: {exc}")
            checkpoint.with_suffix(".error.json").write_text(
                json.dumps({"config_id": cid, "error": f"{type(exc).__name__}: {exc}"},
                           indent=2)
            )

    results = collect_full(output_name)
    print(f"\nSweep finished in {(time.time() - sweep_started) / 60:.1f} minutes.")
    return results


def collect_full(output_name: str = "results.csv") -> pd.DataFrame:
    """Assemble the checkpointed rows into a single results table."""
    rows = [
        json.loads(p.read_text())
        for p in sorted(FULL_ROWS_DIR.glob("*.json"))
        if not p.name.endswith(".error.json")
    ]
    if not rows:
        print("No completed configurations found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    out = FULL_DIR / output_name
    df.to_csv(out, index=False)
    print(f"Results written to {out} ({len(df)} configurations)")

    failures = list(FULL_ROWS_DIR.glob("*.error.json"))
    if failures:
        print(f"WARNING: {len(failures)} configuration(s) failed; see {FULL_ROWS_DIR}")
    return df


def compare_scales(
    tuning_file: str = "results_multiseed.csv",
    output_name: str = "results.csv",
) -> pd.DataFrame:
    """Compare full-scale results against the reduced-scale tuning run.

    Included because it is the check that justifies reusing the tuned
    hyperparameters: if the ranking of techniques is preserved across a
    four- to six-fold increase in training data, the selections transfer.
    """
    full = collect_full(output_name)
    if full.empty:
        return full

    main = pd.read_csv(RESULTS_DIR / tuning_file)
    if "error" in main.columns:
        main = main[main["error"].isna()]

    # Average over replications so each condition appears once, matching the
    # single-seed full-scale run.
    main_mean = (
        main.groupby(["dataset", "imbalance_method", "classifier"], as_index=False)[
            COMPARISON_METRICS
        ].mean()
    )

    merged = main_mean.merge(
        full[["dataset", "imbalance_method", "classifier"] + COMPARISON_METRICS],
        on=["dataset", "imbalance_method", "classifier"],
        suffixes=("_reduced", "_full"),
    )
    for m in COMPARISON_METRICS:
        merged[f"delta_{m}"] = merged[f"{m}_full"] - merged[f"{m}_reduced"]

    cols = ["dataset", "classifier", "imbalance_method"]
    cols += [f"{m}_reduced" for m in COMPARISON_METRICS]
    cols += [f"{m}_full" for m in COMPARISON_METRICS]
    cols += [f"delta_{m}" for m in COMPARISON_METRICS]
    merged = merged[cols].round(4)

    merged.to_csv(FULL_DIR / "comparison.csv", index=False)
    with open(FULL_DIR / "comparison.md", "w") as fh:
        fh.write(merged.to_markdown(index=False))
    print(f"\nComparison written to {FULL_DIR / 'comparison.csv'}")

    print("\nMean change from reduced scale to full scale:")
    print(merged[[f"delta_{m}" for m in COMPARISON_METRICS]].mean().round(4).to_string())
    print("\nF1 at full scale, by method (mean across classifiers and datasets):")
    print(full.groupby("imbalance_method")["f1"].mean().round(4)
          .sort_values(ascending=False).to_string())
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the phishing detection experiments.")
    parser.add_argument("--full", action="store_true",
                        help="Run at full data scale, reusing tuned hyperparameters.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Subset of datasets.")
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of imbalance methods.")
    parser.add_argument("--classifiers", nargs="*", default=None, help="Subset of classifiers.")
    parser.add_argument("--ratio", type=float, default=MINORITY_RATIO, help="Minority class ratio.")
    parser.add_argument("--no-baseline", action="store_true", help="Exclude the untreated baseline.")
    parser.add_argument("--output", default="results.csv", help="Output CSV filename.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help="Random seeds; each one repeats the whole matrix as a replication.",
    )
    args = parser.parse_args()

    if args.full:
        run_full_all(
            datasets=args.datasets,
            methods=args.methods,
            classifiers=args.classifiers,
            seed=args.seeds[0] if args.seeds else RANDOM_STATE,
            output_name="results.csv",
        )
        compare_scales()
        return

    run_all(
        datasets=args.datasets,
        methods=args.methods,
        classifiers=args.classifiers,
        minority_ratio=args.ratio,
        include_baseline=not args.no_baseline,
        output_name=args.output,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()