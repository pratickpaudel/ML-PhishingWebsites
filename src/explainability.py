"""SHAP analysis and heatmap figures.

Answers two questions the performance metrics cannot: which features drive
predictions, and whether imbalance treatment changes them. TreeExplainer is
used for the tree models; the SVM needs KernelExplainer, which is tractable
only at reduced scale.
"""

from __future__ import annotations

import argparse
import time
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import (
    CLASSIFIERS,
    CLASSIFIER_LABELS,
    DATASET_LABELS,
    FIGURES_DIR,
    METHOD_LABELS,
    MINORITY_RATIO,
    RANDOM_STATE,
    RESULTS_DIR,
    SUBSAMPLE_SIZE,
)
from data_loader import load_dataset
from models import build_pipeline, build_search
from preprocessing import prepare

# Sample sizes keep the kernel-based approximation tractable.
TREE_SAMPLE = 500
KERNEL_SAMPLE = 100
KERNEL_BACKGROUND = 50


class ShapResult(NamedTuple):
    """Bundle of SHAP output and the data it was computed on."""

    values: np.ndarray
    X_scaled: pd.DataFrame
    X_raw: pd.DataFrame
    model: object


def _fit_model(dataset: str, method: str, classifier: str, minority_ratio: float):
    """Refit one configuration and return the model with its train/test data."""
    X, y = load_dataset(
        dataset, minority_ratio=minority_ratio, subsample=SUBSAMPLE_SIZE
    )
    X_train, X_test, y_train, y_test = prepare(X, y)

    search = build_search(classifier, method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    return search.best_estimator_, X_train, X_test, y_test


def _transform_through_pipeline(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every pipeline step except the final classifier."""
    X_out = X
    for name, step in pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            X_out = step.transform(X_out)
    return pd.DataFrame(np.asarray(X_out), columns=X.columns, index=X.index)


def compute_shap_values(
    dataset: str = "vrbancic",
    method: str = "smote",
    classifier: str = "random_forest",
    minority_ratio: float = MINORITY_RATIO,
    sample_size: int | None = None,
):
    """Compute SHAP values for one configuration."""
    import shap

    model, X_train, X_test, _ = _fit_model(dataset, method, classifier, minority_ratio)
    estimator = model.named_steps["classifier"]

    X_test_t = _transform_through_pipeline(model, X_test)
    rng = np.random.RandomState(RANDOM_STATE)

    is_tree = classifier in {"decision_tree", "random_forest"}
    n = sample_size or (TREE_SAMPLE if is_tree else KERNEL_SAMPLE)
    n = min(n, len(X_test_t))
    idx = rng.choice(len(X_test_t), size=n, replace=False)
    X_sample = X_test_t.iloc[idx]
    X_sample_raw = X_test.iloc[idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_tree:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_sample)
        else:
            X_train_t = _transform_through_pipeline(model, X_train)
            bg_idx = rng.choice(
                len(X_train_t), size=min(KERNEL_BACKGROUND, len(X_train_t)), replace=False
            )
            explainer = shap.KernelExplainer(
                estimator.decision_function, X_train_t.iloc[bg_idx]
            )
            shap_values = explainer.shap_values(X_sample, silent=True)

    shap_values = _select_positive_class(shap_values)
    return ShapResult(shap_values, X_sample, X_sample_raw, model)


def _select_positive_class(shap_values) -> np.ndarray:
    """Reduce SHAP output to a 2-D array of contributions to the phishing class."""
    if isinstance(shap_values, list):
        # One array per class; index 1 is the positive (phishing) class.
        return np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


def global_importance(shap_values: np.ndarray, X_sample: pd.DataFrame) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value (global importance)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": X_sample.columns,
                "mean_abs_shap": mean_abs,
                "mean_shap": mean_signed,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda d: d.index + 1)
    )


def local_explanation(
    result: ShapResult,
    instance_index: int = 0,
    top_n: int = 10,
) -> pd.DataFrame:
    """Per-feature contributions for a single website (local explanation)."""
    contributions = result.values[instance_index]
    return (
        pd.DataFrame(
            {
                "feature": result.X_scaled.columns,
                "feature_value": result.X_raw.iloc[instance_index].to_numpy(),
                "shap_value": contributions,
                "direction": np.where(contributions > 0, "towards phishing", "towards legitimate"),
            }
        )
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .drop(columns="abs_shap")
        .reset_index(drop=True)
    )


def compare_across_methods(
    dataset: str = "vrbancic",
    classifier: str = "random_forest",
    methods: list[str] | None = None,
    minority_ratio: float = MINORITY_RATIO,
    top_n: int = 15,
    verbose: bool = True,
    sample_size: int | None = None,
) -> pd.DataFrame:
    """Compare global feature importance across imbalance treatment methods."""
    methods = methods or ["none", "smote", "smoteenn", "random_undersampling"]
    frames = []

    for method in methods:
        if verbose:
            print(f"  SHAP: {dataset} / {method} / {classifier}", flush=True)
        result = compute_shap_values(
            dataset, method, classifier, minority_ratio, sample_size=sample_size
        )
        imp = global_importance(result.values, result.X_scaled)
        imp["imbalance_method"] = method
        frames.append(imp)

    combined = pd.concat(frames, ignore_index=True)

    # Wide ranking table: features as rows, methods as columns.
    ranks = combined.pivot(index="feature", columns="imbalance_method", values="rank")
    baseline = methods[0]
    ranks = ranks.sort_values(baseline).head(top_n)
    ranks["rank_range"] = ranks.max(axis=1) - ranks.min(axis=1)

    out = RESULTS_DIR / f"shap_method_comparison_{dataset}_{classifier}.csv"
    combined.to_csv(RESULTS_DIR / f"shap_importance_{dataset}_{classifier}.csv", index=False)
    ranks.to_csv(out)

    if verbose:
        print(f"\nTop {top_n} features by baseline rank ({dataset}, {CLASSIFIER_LABELS[classifier]}):")
        print(ranks.to_string())
        print(f"\nWritten to {out}")

    return ranks


def plot_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    dataset: str,
    method: str,
    classifier: str,
    max_display: int = 15,
) -> None:
    """Save a SHAP beeswarm summary plot for use as a figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False,
    )
    plt.title(
        f"{DATASET_LABELS.get(dataset, dataset)} - "
        f"{CLASSIFIER_LABELS.get(classifier, classifier)} - "
        f"{METHOD_LABELS.get(method, method)}",
        fontsize=10,
    )
    plt.tight_layout()

    path = FIGURES_DIR / f"shap_summary_{dataset}_{method}_{classifier}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"Figure saved to {path}")


# ---------------------------------------------------------------------------

from config import SHAP_DIR  # noqa: E402

FULL_DIR = SHAP_DIR

# Methods compared by default. A full refit is required per method, so the
# default is four rather than all eight.
FULL_SHAP_METHODS = ["none", "smote", "smoteenn", "random_undersampling"]
SHAP_SUPPORTED_AT_FULL_SCALE = {"decision_tree", "random_forest"}

# Presentation order for the heatmap columns.
_METHOD_ORDER = [
    "none",
    "random_oversampling",
    "random_undersampling",
    "smote",
    "adasyn",
    "smoteenn",
    "smotetomek",
    "cost_sensitive",
]


def _compute_full_shap_one(
    dataset: str,
    method: str,
    classifier: str,
    tuned: dict,
    seed: int = RANDOM_STATE,
    sample_size: int = TREE_SAMPLE,
) -> pd.DataFrame:
    """Fit one configuration at full scale and return its global importances."""
    import shap

    from experiment import _resolve_params

    if classifier not in SHAP_SUPPORTED_AT_FULL_SCALE:
        raise ValueError(
            f"'{classifier}' is not supported at full scale; KernelExplainer is "
            "intractable on the full training partitions. Use the reduced-scale "
            "analysis for the SVM."
        )

    X, y = load_dataset(dataset, minority_ratio=None, random_state=seed, subsample=None)
    X_train, X_test, y_train, _ = prepare(X, y, random_state=seed)

    params, _ = _resolve_params(tuned, dataset, method, classifier, seed)
    pipeline = build_pipeline(classifier, method, random_state=seed)
    if params:
        pipeline.set_params(**{f"classifier__{k}": v for k, v in params.items()})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    estimator = pipeline.named_steps["classifier"]
    X_test_t = _transform_through_pipeline(pipeline, X_test)

    # A sample of the test set is explained rather than all 23,320 rows: the
    # global ranking converges well before that, and the figures show 15 rows.
    rng = np.random.RandomState(RANDOM_STATE)
    n = min(sample_size, len(X_test_t))
    idx = rng.choice(len(X_test_t), size=n, replace=False)
    X_sample = X_test_t.iloc[idx]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = shap.TreeExplainer(estimator).shap_values(X_sample)

    imp = global_importance(_select_positive_class(values), X_sample)
    imp["imbalance_method"] = method
    return imp


def compare_across_methods_full(
    dataset: str = "vrbancic",
    classifier: str = "random_forest",
    methods: list[str] | None = None,
    seed: int = RANDOM_STATE,
    top_n: int = 15,
    sample_size: int = TREE_SAMPLE,
) -> pd.DataFrame:
    """Compare global SHAP importance across methods at full scale."""
    from experiment import load_tuned_params

    methods = methods or FULL_SHAP_METHODS
    tuned = load_tuned_params()
    frames = []

    print(f"{dataset} / {CLASSIFIER_LABELS.get(classifier, classifier)} at full scale")
    for method in methods:
        started = time.time()
        print(f"  {method:22s}", end=" ", flush=True)
        try:
            frames.append(
                _compute_full_shap_one(dataset, method, classifier, tuned, seed, sample_size)
            )
            print(f"done ({time.time() - started:.0f}s)")
        except Exception as exc:
            print(f"FAILED: {type(exc).__name__}: {exc}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(FULL_DIR / f"shap_importance_{dataset}_{classifier}.csv", index=False)

    ranks = combined.pivot(index="feature", columns="imbalance_method", values="rank")
    reference = methods[0] if methods[0] in ranks.columns else ranks.columns[0]
    ranks = ranks.sort_values(reference).head(top_n)
    ranks["rank_range"] = ranks.max(axis=1) - ranks.min(axis=1)
    ranks.to_csv(FULL_DIR / f"shap_method_comparison_{dataset}_{classifier}.csv")

    stable = int((ranks["rank_range"] == 0).sum())
    print(f"  rank stability: {stable}/{len(ranks)} features unchanged, "
          f"max shift {int(ranks['rank_range'].max())} places")
    return ranks


# ---------------------------------------------------------------------------


# Single-hue ramps with monotonic lightness, so the figures stay legible when
# printed without colour.
_RANK_CMAP = LinearSegmentedColormap.from_list(
    "rank", ["#1b3a5c", "#3d6d99", "#7ba3c8", "#b9cfe2", "#e8eef4"]
)
_MAG_CMAP = LinearSegmentedColormap.from_list(
    "mag", ["#f2f0ec", "#d6c9b4", "#b89b6f", "#8f6f3d", "#5c4420"]
)

_HEATMAP_LABELS = {
    "none": "No Treatment",
    "random_oversampling": "Random\nOversampling",
    "random_undersampling": "Random\nUndersampling",
    "smote": "SMOTE",
    "adasyn": "ADASYN",
    "smoteenn": "SMOTEENN",
    "smotetomek": "SMOTETomek",
    "cost_sensitive": "Cost-Sensitive",
}


def _relative_luminance(rgba) -> float:
    """Perceived brightness, used to pick annotation colour."""
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ordered_methods(columns) -> list:
    present = [m for m in _METHOD_ORDER if m in columns]
    return present + [c for c in columns if c not in _METHOD_ORDER]


def _draw_heatmap(matrix, title, cbar_label, path, annotate_as_int, reverse_scale):
    """Render one heatmap, scaling height with row count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(1.35 * n_cols + 3.2, 0.40 * n_rows + 2.0))

    cmap = _RANK_CMAP if reverse_scale else _MAG_CMAP
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        [_HEATMAP_LABELS.get(c, c) for c in matrix.columns],
        fontsize=9, va="top", linespacing=1.25,
    )
    ax.tick_params(axis="x", pad=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=8.5)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    norm = im.norm
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            if np.isnan(v):
                continue
            # Pick text colour from the rendered cell, not the raw value: the
            # colormap is not linear in lightness.
            colour = "white" if _relative_luminance(cmap(norm(v))) < 0.55 else "#1a1a1a"
            ax.text(
                j, i,
                f"{int(round(v))}" if annotate_as_int else f"{v:.3f}",
                ha="center", va="center",
                fontsize=8.5, color=colour, fontweight="medium",
            )

    ax.set_title(title, fontsize=10.5, pad=14, loc="left", color="#1a1a1a")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=2, labelsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


def plot_rank_heatmap(dataset, classifier="random_forest", results_dir=None, top_n=15,
                      suffix="_full"):
    """Feature importance rank under each imbalance method."""
    results_dir = results_dir or FULL_DIR
    path = results_dir / f"shap_method_comparison_{dataset}_{classifier}.csv"
    if not path.exists():
        print(f"  skipped: {path.name} not found")
        return None

    df = pd.read_csv(path).set_index("feature")
    df = df.drop(columns=[c for c in ("rank_range",) if c in df.columns])
    df = df[_ordered_methods(df.columns)]

    # Order by the baseline ranking so all columns share one reference.
    reference = "none" if "none" in df.columns else df.columns[0]
    df = df.sort_values(reference).head(top_n)

    title = (f"SHAP feature importance rank - {DATASET_LABELS.get(dataset, dataset)}\n"
             f"{CLASSIFIER_LABELS.get(classifier, classifier)} "
             f"(rank 1 = most important; identical columns indicate rank stability)")
    _draw_heatmap(df, title, "Importance rank",
                  FIGURES_DIR / f"shap_rank_heatmap_{dataset}_{classifier}{suffix}.png",
                  annotate_as_int=True, reverse_scale=True)
    return df


def plot_magnitude_heatmap(dataset, classifier="random_forest", results_dir=None,
                           top_n=15, suffix="_full"):
    """Mean absolute SHAP value per feature under each imbalance method."""
    results_dir = results_dir or FULL_DIR
    path = results_dir / f"shap_importance_{dataset}_{classifier}.csv"
    if not path.exists():
        print(f"  skipped: {path.name} not found")
        return None

    long = pd.read_csv(path)
    wide = long.pivot(index="feature", columns="imbalance_method", values="mean_abs_shap")
    wide = wide[_ordered_methods(wide.columns)]

    reference = "none" if "none" in wide.columns else wide.columns[0]
    wide = wide.sort_values(reference, ascending=False).head(top_n)

    title = (f"Mean absolute SHAP value - {DATASET_LABELS.get(dataset, dataset)}\n"
             f"{CLASSIFIER_LABELS.get(classifier, classifier)} "
             f"(magnitude of each feature's contribution)")
    _draw_heatmap(wide, title, "Mean |SHAP|",
                  FIGURES_DIR / f"shap_magnitude_heatmap_{dataset}_{classifier}{suffix}.png",
                  annotate_as_int=False, reverse_scale=False)
    return wide


def generate_figures(datasets=None, classifier="random_forest", results_dir=None,
                     top_n=15, suffix="_full") -> None:
    """Generate both heatmaps for every dataset with saved SHAP output."""
    from config import DATASETS

    datasets = datasets or DATASETS
    for ds in datasets:
        print(f"{ds} / {classifier}:")
        ranks = plot_rank_heatmap(ds, classifier, results_dir, top_n, suffix)
        plot_magnitude_heatmap(ds, classifier, results_dir, top_n, suffix)

        # A numeric summary of what the rank figure shows, so the claim in the
        # text can cite a number rather than an impression.
        if ranks is not None and len(ranks.columns) > 1:
            spread = ranks.max(axis=1) - ranks.min(axis=1)
            print(f"  rank stability: {int((spread == 0).sum())}/{len(spread)} features "
                  f"unchanged across methods, max shift {int(spread.max())} places")
    print(f"\nFigures written to {FIGURES_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP explainability analysis.")
    parser.add_argument("--dataset", default="vrbancic")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Instances explained; lower values trade precision for speed.",
    )
    parser.add_argument("--classifier", default="random_forest")
    parser.add_argument("--method", default="smote")
    parser.add_argument("--ratio", type=float, default=MINORITY_RATIO)
    parser.add_argument("--compare", action="store_true", help="Compare across imbalance methods.")
    parser.add_argument("--full", action="store_true",
                        help="Run at full data scale, reusing tuned hyperparameters.")
    parser.add_argument("--figures", action="store_true",
                        help="Generate heatmap figures from saved SHAP output.")
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--plot", action="store_true", help="Save a beeswarm summary figure.")
    args = parser.parse_args()

    if args.figures:
        generate_figures(top_n=15)
        return

    if args.full:
        from config import DATASETS

        for ds in ([args.dataset] if args.dataset else DATASETS):
            compare_across_methods_full(
                dataset=ds, classifier=args.classifier, methods=args.methods,
                sample_size=args.sample_size,
            )
        generate_figures()
        return

    if args.compare:
        compare_across_methods(
            dataset=args.dataset,
            classifier=args.classifier,
            methods=args.methods,
            minority_ratio=args.ratio,
            sample_size=args.sample_size,
        )
        return

    result = compute_shap_values(
        args.dataset, args.method, args.classifier, args.ratio,
        sample_size=args.sample_size,
    )

    print(f"\nGlobal feature importance ({args.dataset} / {args.method} / {args.classifier}):")
    print(global_importance(result.values, result.X_scaled).head(15).to_string(index=False))

    print("\nLocal explanation for test instance 0:")
    print(local_explanation(result).to_string(index=False))

    if args.plot:
        plot_summary(
            result.values, result.X_scaled, args.dataset, args.method, args.classifier
        )


if __name__ == "__main__":
    main()