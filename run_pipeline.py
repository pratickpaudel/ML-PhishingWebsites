#!/usr/bin/env python
"""
End-to-end pipeline runner (full data scale).

Executes the complete experimental procedure in the order shown in the
methodology diagram, on the complete datasets (88,647 and 116,600 instances):

    dataset loading -> preprocessing -> stratified split -> imbalance treatment
    -> classifier fitting -> evaluation -> comparative analysis
    -> statistical significance testing -> SHAP explainability -> figures

All outputs are written to ``results/full/`` and ``figures/``.

Hyperparameters
---------------
Hyperparameters are not searched here. Each configuration reuses the selection
recorded in ``results/results_multiseed.csv``, which was produced by grid
search on a stratified subsample, and is fitted once. Grid search at full scale
is a multi-day job dominated by the SVM, and reusing the selections also keeps
the comparison clean: re-tuning would confound the effect of sample size with
the effect of re-selection.

If that file is missing, regenerate it with:

    python src/experiment.py --seeds 42 1 2 --output results_multiseed.csv

Usage
-----
    python run_pipeline.py                    # full pipeline
    python run_pipeline.py --quick            # small subset, for a smoke test
    python run_pipeline.py --skip-experiments # re-analyse existing results
    python run_pipeline.py --no-shap          # skip the SHAP stage (the slow one)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DATASETS, RESULTS_DIR  # noqa: E402

FULL_DIR = RESULTS_DIR / "full"


def banner(step: str, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"STEP {step}: {title}")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full experimental pipeline.")
    parser.add_argument("--quick", action="store_true", help="Run a reduced subset.")
    parser.add_argument("--skip-experiments", action="store_true",
                        help="Reuse the existing results/full/results.csv.")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip the SHAP stage, which refits once per method.")
    parser.add_argument("--all-methods-shap", action="store_true",
                        help="Compare all eight conditions in SHAP instead of four.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="results.csv")
    args = parser.parse_args()

    started = time.time()

    # -- Step 1: describe the input data ------------------------------------
    banner("1", "Dataset summary")
    from data_loader import describe

    for ds in DATASETS:
        d = describe(ds)
        print(f"  {ds}: {d['instances']:6d} rows, {d['features']:3d} features, "
              f"{d['phishing_pct']}% phishing (ratio {d['imbalance_ratio']})")

    # -- Steps 2-7: run the experimental matrix -----------------------------
    import experiment

    if not args.skip_experiments:
        banner("2-7", "Experimental matrix (split, treatment, fitting, evaluation)")
        if args.quick:
            experiment.run_full_all(
                datasets=[DATASETS[0]],
                methods=["none", "smote", "random_undersampling"],
                classifiers=["decision_tree", "random_forest"],
                seed=args.seed,
                output_name=args.results,
            )
        else:
            experiment.run_full_all(seed=args.seed, output_name=args.results)
    else:
        banner("2-7", "Experimental matrix (skipped, reusing existing results)")
        if not (FULL_DIR / args.results).exists():
            print(f"  ERROR: {FULL_DIR / args.results} not found.")
            return 1

    # -- Step 8: comparative analysis ---------------------------------------
    banner("8", "Comparative performance analysis")
    experiment.compare_scales(output_name=args.results)

    import analysis

    # analysis.py writes to a fixed set of filenames in its results directory.
    # Redirecting it keeps the tables beside the data they describe.
    analysis.RESULTS_DIR = FULL_DIR
    analysis.generate_all(results_file=args.results)

    # -- Step 9: statistical significance -----------------------------------
    banner("9", "Statistical significance testing")
    import statistical_tests

    try:
        statistical_tests.run_full_tests(results_file=args.results)
    except Exception as exc:
        print(f"  Statistical testing skipped: {type(exc).__name__}: {exc}")
        print("  (expected with --quick, which produces too few matched blocks)")

    # -- Step 10: explainability and figures --------------------------------
    if not args.no_shap:
        banner("10", "SHAP explainability")
        import explainability
        from config import IMBALANCE_METHODS

        methods = IMBALANCE_METHODS if args.all_methods_shap else None
        for ds in ([DATASETS[0]] if args.quick else DATASETS):
            try:
                explainability.compare_across_methods_full(
                    dataset=ds, classifier="random_forest",
                    methods=methods, seed=args.seed,
                )
            except Exception as exc:
                print(f"  SHAP skipped for {ds}: {type(exc).__name__}: {exc}")

        banner("11", "SHAP figures")
        explainability.generate_figures(
            datasets=[DATASETS[0]] if args.quick else DATASETS
        )
    else:
        banner("10-11", "SHAP explainability and figures (skipped)")

    print(f"\nPipeline complete in {(time.time() - started) / 60:.1f} minutes.")
    print(f"Outputs in {FULL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())