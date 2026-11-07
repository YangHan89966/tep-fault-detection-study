"""
Step 10: Random run-pairing robustness diagnostic for Step 04 screening.

This script tests whether the original pointwise matched-counterfactual
screening rule depends on pairing Fault 3 development run i with normal
validation run i.  It compares the original matched pairing with repeated
random *derangements* of normal runs (no run remains paired with the same run
number).  The PCA model, development data, post-fault window, direction-ratio
threshold, and Top-K ranking rule are otherwise unchanged.

Place this file in:
    TEP_Fault3_Project/scripts/10_random_pairing_robustness.py

Run from the project root:
    python scripts/10_random_pairing_robustness.py

The script writes only files beginning with ``10_`` and does not overwrite the
original Step 04 or Step 06/06b results.
"""

from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


FAULT_START_SAMPLE = 161
PCA_CUMVAR_TARGET = 0.95
DIRECTION_RATIO_THRESHOLD = 0.95
N_SELECTED_VARIABLES = 5

# Repeated random run matching.  One hundred repeats are enough to reveal
# whether the direction-ratio screen is stable while remaining practical on
# the 250 x 800 x 52 development tensor.
N_RANDOM_PAIRINGS = 100
RANDOM_SEED = 20260714


def require_file(path, instruction):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def load_variable_columns():
    path = require_file(
        TABLE_DIR / "02_variable_list.csv",
        "run scripts/02_prepare_fault3_data.py",
    )
    table = pd.read_csv(path)
    if "variable" not in table.columns:
        raise ValueError(f"Missing 'variable' column in {path}")
    variables = table["variable"].astype(str).tolist()
    if len(variables) != 52:
        raise ValueError(f"Expected 52 variables, found {len(variables)}")
    return variables


def load_data():
    paths = {
        "normal_train": PROCESSED_DIR / "normal_train.pkl",
        "normal_validation": PROCESSED_DIR
        / "normal_validation_runs_1_250.pkl",
        "fault3_development": PROCESSED_DIR
        / "fault3_development_runs_1_250.pkl",
    }
    for path in paths.values():
        require_file(path, "run scripts/02_prepare_fault3_data.py")

    print("Loading data...")
    data = {name: pd.read_pickle(path) for name, path in paths.items()}
    for name, frame in data.items():
        print(f"  {name}: shape={frame.shape}")
    return data


def fit_pca_model(normal_train, variable_cols):
    x = normal_train[variable_cols].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    x_std = (x - mean) / std

    pca = PCA()
    pca.fit(x_std)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    n_pcs = int(np.searchsorted(cumulative, PCA_CUMVAR_TARGET) + 1)

    model = {
        "mean": mean,
        "std": std,
        "components": pca.components_[:n_pcs, :],
        "n_pcs": n_pcs,
        "cum_var": float(cumulative[n_pcs - 1]),
    }
    print(
        f"PCA model: {n_pcs} PCs, cumulative explained variance "
        f"{model['cum_var']:.6f}"
    )
    return model


def compute_residual_tensor(frame, variable_cols, model, expected_runs=None):
    """Return post-fault residual tensor [run, sample, variable]."""
    sorted_frame = frame.sort_values(["simulationRun", "sample"]).reset_index(
        drop=True
    )
    runs = np.sort(sorted_frame["simulationRun"].unique())
    if expected_runs is not None and not np.array_equal(runs, expected_runs):
        raise ValueError("Normal and Fault 3 development run IDs do not match")

    counts = sorted_frame.groupby("simulationRun").size().to_numpy()
    if len(set(counts.tolist())) != 1:
        raise ValueError("Runs do not all contain the same number of samples")
    n_samples = int(counts[0])

    first_samples = sorted_frame.loc[
        sorted_frame["simulationRun"] == runs[0], "sample"
    ].to_numpy()
    expected_samples = np.tile(first_samples, len(runs))
    if not np.array_equal(sorted_frame["sample"].to_numpy(), expected_samples):
        raise ValueError("Sample grids are not identical across runs")

    x = sorted_frame[variable_cols].to_numpy(dtype=float)
    x_std = (x - model["mean"]) / model["std"]
    loadings = model["components"]
    residual = x_std - (x_std @ loadings.T) @ loadings
    tensor = residual.reshape(len(runs), n_samples, len(variable_cols))

    post_mask = first_samples >= FAULT_START_SAMPLE
    if not np.any(post_mask):
        raise ValueError("No post-fault samples found")
    post_samples = first_samples[post_mask]
    return tensor[:, post_mask, :], runs, post_samples


def calculate_metrics(
    normal_tensor,
    fault_tensor,
    normal_run_order,
    variable_cols,
    normal_residual_std,
):
    """Apply the exact Step 04 pointwise delta calculations."""
    delta = fault_tensor - normal_tensor[normal_run_order, :, :]
    mean_delta = delta.mean(axis=(0, 1))
    std_delta = delta.std(axis=(0, 1), ddof=1)
    positive_ratio = (delta > 0).mean(axis=(0, 1))
    negative_ratio = (delta < 0).mean(axis=(0, 1))
    direction_ratio = np.maximum(positive_ratio, negative_ratio)
    sign = np.where(mean_delta >= 0, 1, -1)
    effect_size = np.divide(
        mean_delta,
        normal_residual_std,
        out=np.full_like(mean_delta, np.nan),
        where=normal_residual_std > 0,
    )

    metrics = pd.DataFrame(
        {
            "variable": variable_cols,
            "residual_variable": [f"res_{v}" for v in variable_cols],
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "normal_residual_std": normal_residual_std,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "direction_ratio": direction_ratio,
            "dominant_direction": np.where(mean_delta >= 0, "positive", "negative"),
            "sign_for_positive_evidence": sign,
            "effect_size": effect_size,
            "abs_effect_size": np.abs(effect_size),
            "abs_mean_delta": np.abs(mean_delta),
        }
    )
    return metrics


def apply_step04_selection(metrics):
    """Reproduce the Step 04 stable-filter then Top-K ranking rule."""
    ranked = metrics.sort_values(
        ["direction_ratio", "abs_mean_delta", "abs_effect_size"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    stable = ranked[
        ranked["direction_ratio"] >= DIRECTION_RATIO_THRESHOLD
    ].copy()
    selected = stable.sort_values(
        ["abs_mean_delta", "direction_ratio", "abs_effect_size"],
        ascending=[False, False, False],
    ).head(N_SELECTED_VARIABLES).reset_index(drop=True)
    return ranked, stable, selected


def random_derangement(n, rng):
    """Random permutation with no fixed run-to-run matches."""
    identity = np.arange(n)
    while True:
        permutation = rng.permutation(n)
        if np.all(permutation != identity):
            return permutation


def jaccard(left, right):
    left, right = set(left), set(right)
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def run_random_pairing_experiment(
    normal_tensor,
    fault_tensor,
    variable_cols,
    normal_residual_std,
    original_selected,
):
    rng = np.random.default_rng(RANDOM_SEED)
    n_runs = normal_tensor.shape[0]
    ratio_matrix = np.empty((N_RANDOM_PAIRINGS, len(variable_cols)), dtype=float)
    selection_matrix = np.zeros(
        (N_RANDOM_PAIRINGS, len(variable_cols)), dtype=bool
    )
    repeat_rows = []

    original_set = original_selected["variable"].astype(str).tolist()
    original_mean_delta = None
    start_time = time.time()

    for repeat in range(N_RANDOM_PAIRINGS):
        permutation = random_derangement(n_runs, rng)
        metrics = calculate_metrics(
            normal_tensor,
            fault_tensor,
            permutation,
            variable_cols,
            normal_residual_std,
        )
        _, stable, selected = apply_step04_selection(metrics)
        selected_vars = selected["variable"].astype(str).tolist()

        ratio_matrix[repeat, :] = metrics["direction_ratio"].to_numpy()
        selection_matrix[
            repeat,
            [variable_cols.index(v) for v in selected_vars],
        ] = True

        if original_mean_delta is None:
            original_mean_delta = metrics["mean_delta"].to_numpy()

        overlap = len(set(original_set) & set(selected_vars))
        repeat_rows.append(
            {
                "repeat": repeat + 1,
                "random_seed": RANDOM_SEED,
                "n_fixed_run_pairs": int(
                    np.sum(permutation == np.arange(n_runs))
                ),
                "n_direction_stable_variables": int(len(stable)),
                "n_selected_variables": int(len(selected)),
                "selected_variables": ";".join(selected_vars),
                "original_top5_overlap_count": overlap,
                "original_top5_jaccard": jaccard(original_set, selected_vars),
                "exact_original_top5_recovered": set(selected_vars)
                == set(original_set),
                "median_direction_ratio_all_variables": float(
                    metrics["direction_ratio"].median()
                ),
                "maximum_direction_ratio": float(
                    metrics["direction_ratio"].max()
                ),
                "minimum_selected_direction_ratio": float(
                    selected["direction_ratio"].min()
                )
                if len(selected)
                else np.nan,
            }
        )

        if (repeat + 1) % 10 == 0 or repeat == 0:
            elapsed = time.time() - start_time
            print(
                f"  random pairing {repeat + 1:3d}/{N_RANDOM_PAIRINGS}: "
                f"stable={len(stable):2d}, selected={len(selected):1d}, "
                f"Top-5 overlap={overlap}, elapsed={elapsed:.1f}s"
            )

    repeat_df = pd.DataFrame(repeat_rows)
    return repeat_df, ratio_matrix, selection_matrix, original_mean_delta


def summarize_variables(
    matched_ranked,
    ratio_matrix,
    selection_matrix,
    variable_cols,
):
    matched_by_var = matched_ranked.set_index("variable")
    rows = []
    for j, variable in enumerate(variable_cols):
        ratios = ratio_matrix[:, j]
        matched = matched_by_var.loc[variable]
        rows.append(
            {
                "variable": variable,
                "residual_variable": f"res_{variable}",
                "matched_mean_delta": matched["mean_delta"],
                "matched_abs_mean_delta": matched["abs_mean_delta"],
                "matched_direction_ratio": matched["direction_ratio"],
                "matched_direction_stable": bool(
                    matched["direction_ratio"] >= DIRECTION_RATIO_THRESHOLD
                ),
                "random_direction_ratio_mean": float(np.mean(ratios)),
                "random_direction_ratio_median": float(np.median(ratios)),
                "random_direction_ratio_q025": float(
                    np.quantile(ratios, 0.025)
                ),
                "random_direction_ratio_q975": float(
                    np.quantile(ratios, 0.975)
                ),
                "random_direction_ratio_min": float(np.min(ratios)),
                "random_direction_ratio_max": float(np.max(ratios)),
                "random_pass_095_frequency": float(
                    np.mean(ratios >= DIRECTION_RATIO_THRESHOLD)
                ),
                "random_top5_selection_frequency": float(
                    np.mean(selection_matrix[:, j])
                ),
                "sign_for_positive_evidence": int(
                    matched["sign_for_positive_evidence"]
                ),
            }
        )
    return pd.DataFrame(rows)


def make_global_summary(
    model,
    matched_stable,
    matched_selected,
    repeat_df,
    variable_summary,
    mean_delta_max_abs_difference,
    existing_selected_match,
):
    stable_counts = repeat_df["n_direction_stable_variables"]
    overlap = repeat_df["original_top5_overlap_count"]
    exact = repeat_df["exact_original_top5_recovered"].astype(float)
    enough = (repeat_df["n_selected_variables"] >= N_SELECTED_VARIABLES).astype(
        float
    )

    return {
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "fault_start_sample": FAULT_START_SAMPLE,
        "screening_window": f"samples {FAULT_START_SAMPLE}-960",
        "direction_ratio_threshold": DIRECTION_RATIO_THRESHOLD,
        "top_k": N_SELECTED_VARIABLES,
        "n_random_derangements": N_RANDOM_PAIRINGS,
        "random_seed": RANDOM_SEED,
        "matched_direction_stable_count": int(len(matched_stable)),
        "matched_selected_variables": matched_selected["variable"].tolist(),
        "matched_recomputation_agrees_with_existing_step04": bool(
            existing_selected_match
        ),
        "random_stable_count_median": float(stable_counts.median()),
        "random_stable_count_q025": float(stable_counts.quantile(0.025)),
        "random_stable_count_q975": float(stable_counts.quantile(0.975)),
        "random_stable_count_min": int(stable_counts.min()),
        "random_stable_count_max": int(stable_counts.max()),
        "random_probability_of_selecting_full_top_k": float(enough.mean()),
        "random_original_top5_overlap_median": float(overlap.median()),
        "random_original_top5_overlap_q025": float(overlap.quantile(0.025)),
        "random_original_top5_overlap_q975": float(overlap.quantile(0.975)),
        "random_exact_original_top5_recovery_frequency": float(exact.mean()),
        "maximum_abs_mean_delta_change_across_permutations": float(
            mean_delta_max_abs_difference
        ),
        "variables_passing_095_in_at_least_80_percent_random_pairings": (
            variable_summary.loc[
                variable_summary["random_pass_095_frequency"] >= 0.80,
                "variable",
            ].tolist()
        ),
    }


def plot_direction_ratio_stability(variable_summary, original_selected_vars):
    plot = variable_summary.sort_values(
        "matched_abs_mean_delta", ascending=False
    ).head(15)
    x = np.arange(len(plot))
    med = plot["random_direction_ratio_median"].to_numpy()
    low = plot["random_direction_ratio_q025"].to_numpy()
    high = plot["random_direction_ratio_q975"].to_numpy()
    yerr = np.vstack([med - low, high - med])

    colors = [
        "#D95F59" if v in set(original_selected_vars) else "#4C78A8"
        for v in plot["variable"]
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.errorbar(
        x,
        med,
        yerr=yerr,
        fmt="none",
        ecolor="#777777",
        capsize=3,
        linewidth=1.2,
        label="Random pairing: median and 95% range",
    )
    ax.scatter(
        x,
        med,
        color=colors,
        s=42,
        zorder=3,
    )
    ax.scatter(
        x,
        plot["matched_direction_ratio"],
        facecolors="none",
        edgecolors="#222222",
        marker="D",
        s=52,
        zorder=4,
        label="Matched pairing",
    )
    ax.axhline(
        DIRECTION_RATIO_THRESHOLD,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label="Direction-ratio threshold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot["variable"], rotation=35, ha="right")
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Direction ratio")
    ax.set_title("Matched versus random run pairing: direction stability")
    ax.legend(loc="lower left")
    fig.tight_layout()

    png = FIGURE_DIR / "fig10a_pairing_direction_ratio_stability.png"
    pdf = FIGURE_DIR / "fig10a_pairing_direction_ratio_stability.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_selection_frequency(variable_summary, original_selected_vars):
    plot = variable_summary.sort_values(
        ["random_top5_selection_frequency", "matched_abs_mean_delta"],
        ascending=[False, False],
    ).head(15)
    colors = [
        "#D95F59" if v in set(original_selected_vars) else "#7A9E9F"
        for v in plot["variable"]
    ]

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(
        plot["variable"],
        100 * plot["random_top5_selection_frequency"],
        color=colors,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-5 selection frequency (%)")
    ax.set_title("Variable-selection stability across random run pairings")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()

    png = FIGURE_DIR / "fig10b_pairing_variable_selection_frequency.png"
    pdf = FIGURE_DIR / "fig10b_pairing_variable_selection_frequency.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    print("=" * 80)
    print("Step 10: Random run-pairing robustness diagnostic")
    print("=" * 80)

    variable_cols = load_variable_columns()
    data = load_data()
    model = fit_pca_model(data["normal_train"], variable_cols)

    print("Computing normal validation residual tensor...")
    normal_tensor, runs, post_samples = compute_residual_tensor(
        data["normal_validation"], variable_cols, model
    )
    print("Computing Fault 3 development residual tensor...")
    fault_tensor, fault_runs, fault_post_samples = compute_residual_tensor(
        data["fault3_development"],
        variable_cols,
        model,
        expected_runs=runs,
    )
    if not np.array_equal(post_samples, fault_post_samples):
        raise ValueError("Normal and Fault 3 post-fault sample grids differ")
    if not np.array_equal(runs, fault_runs):
        raise ValueError("Normal and Fault 3 run IDs differ")

    normal_residual_std = normal_tensor.reshape(
        -1, normal_tensor.shape[-1]
    ).std(axis=0, ddof=1)

    print("\nRecomputing the original matched pairing...")
    identity = np.arange(len(runs))
    matched_metrics = calculate_metrics(
        normal_tensor,
        fault_tensor,
        identity,
        variable_cols,
        normal_residual_std,
    )
    matched_ranked, matched_stable, matched_selected = apply_step04_selection(
        matched_metrics
    )
    print(f"  matched direction-stable variables: {len(matched_stable)}")
    print(f"  matched Top-5: {matched_selected['variable'].tolist()}")

    existing_path = TABLE_DIR / "04_fault3_auto_selected_signed_residuals.csv"
    existing_selected_match = False
    if existing_path.exists():
        existing_selected = pd.read_csv(existing_path)
        existing_vars = existing_selected["variable"].astype(str).tolist()
        recomputed_vars = matched_selected["variable"].astype(str).tolist()
        existing_selected_match = existing_vars == recomputed_vars
        if existing_selected_match:
            print("  recomputed Top-5 exactly matches the existing Step 04 table")
        else:
            print("  WARNING: recomputed Top-5 differs from existing Step 04 table")
            print(f"    existing:   {existing_vars}")
            print(f"    recomputed: {recomputed_vars}")

    print(
        f"\nRunning {N_RANDOM_PAIRINGS} random derangements "
        "(zero same-number run pairs)..."
    )
    repeat_df, ratio_matrix, selection_matrix, first_random_mean_delta = (
        run_random_pairing_experiment(
            normal_tensor,
            fault_tensor,
            variable_cols,
            normal_residual_std,
            matched_selected,
        )
    )

    variable_summary = summarize_variables(
        matched_ranked,
        ratio_matrix,
        selection_matrix,
        variable_cols,
    )
    mean_delta_max_abs_difference = float(
        np.max(
            np.abs(
                matched_metrics["mean_delta"].to_numpy()
                - first_random_mean_delta
            )
        )
    )
    global_summary = make_global_summary(
        model,
        matched_stable,
        matched_selected,
        repeat_df,
        variable_summary,
        mean_delta_max_abs_difference,
        existing_selected_match,
    )

    matched_path = TABLE_DIR / "10_matched_pairing_recomputed_metrics.csv"
    repeat_path = TABLE_DIR / "10_random_pairing_repeat_summary.csv"
    variable_path = TABLE_DIR / "10_random_pairing_variable_stability.csv"
    matched_ranked.to_csv(matched_path, index=False, encoding="utf-8-sig")
    repeat_df.to_csv(repeat_path, index=False, encoding="utf-8-sig")
    variable_summary.to_csv(variable_path, index=False, encoding="utf-8-sig")

    fig10a = plot_direction_ratio_stability(
        variable_summary, matched_selected["variable"].tolist()
    )
    fig10b = plot_selection_frequency(
        variable_summary, matched_selected["variable"].tolist()
    )

    summary_path = CONFIG_DIR / "fault3_random_pairing_robustness_summary.json"
    config = {
        "script": "scripts/10_random_pairing_robustness.py",
        "diagnostic_question": (
            "Does Step 04 pointwise direction consistency remain stable when "
            "normal and Fault 3 development runs are not matched by run ID?"
        ),
        "pairing_design": {
            "matched": "Fault 3 run i minus normal run i at the same sample",
            "random": (
                "Fault 3 run i minus a randomly deranged normal run at the same "
                "sample; zero same-ID pairs"
            ),
            "repeats": N_RANDOM_PAIRINGS,
            "seed": RANDOM_SEED,
        },
        "fixed_across_comparisons": {
            "pca_training": "normal_train.pkl",
            "pca_cumulative_variance_target": PCA_CUMVAR_TARGET,
            "development_normal": "normal_validation_runs_1_250.pkl",
            "development_fault": "fault3_development_runs_1_250.pkl",
            "post_fault_samples": (
                f"{int(post_samples.min())}-{int(post_samples.max())}"
            ),
            "direction_ratio_threshold": DIRECTION_RATIO_THRESHOLD,
            "top_k": N_SELECTED_VARIABLES,
            "ranking": (
                "direction_ratio >= 0.95, then descending abs_mean_delta, "
                "direction_ratio, abs_effect_size"
            ),
        },
        "summary": global_summary,
        "outputs": {
            "matched_metrics": str(matched_path.relative_to(PROJECT_ROOT)),
            "repeat_summary": str(repeat_path.relative_to(PROJECT_ROOT)),
            "variable_stability": str(variable_path.relative_to(PROJECT_ROOT)),
            "figures": [
                str(path.relative_to(PROJECT_ROOT))
                for path in [*fig10a, *fig10b]
            ],
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Step 10 summary")
    print("=" * 80)
    print(f"Matched stable-variable count: {len(matched_stable)}")
    print(
        "Random stable-variable count, median [2.5%, 97.5%]: "
        f"{global_summary['random_stable_count_median']:.1f} "
        f"[{global_summary['random_stable_count_q025']:.1f}, "
        f"{global_summary['random_stable_count_q975']:.1f}]"
    )
    print(
        "Probability of obtaining five direction-stable variables: "
        f"{100 * global_summary['random_probability_of_selecting_full_top_k']:.1f}%"
    )
    print(
        "Original Top-5 overlap, median [2.5%, 97.5%]: "
        f"{global_summary['random_original_top5_overlap_median']:.1f} "
        f"[{global_summary['random_original_top5_overlap_q025']:.1f}, "
        f"{global_summary['random_original_top5_overlap_q975']:.1f}]"
    )
    print(
        "Exact original Top-5 recovery frequency: "
        f"{100 * global_summary['random_exact_original_top5_recovery_frequency']:.1f}%"
    )
    print(f"Saved repeat summary: {repeat_path}")
    print(f"Saved variable stability: {variable_path}")
    print(f"Saved configuration and summary: {summary_path}")
    print("Step 10 finished successfully.")


if __name__ == "__main__":
    main()
