"""
Step 14: Repeated random run-level split robustness for the pairing-free method.

Why this analysis is needed
---------------------------
The primary Step 11 result uses the prespecified runs 1--250 for development
and runs 251--500 for final evaluation.  This script checks whether the
pairing-free Top-5 screen and its performance are overly dependent on that one
split.

For each of 20 prespecified repeats, the 500 normal runs and 500 Fault-3 runs
are split independently at the run level into 250 development/calibration runs
and 250 evaluation runs.  The PCA model remains fixed from the separate normal
training set.  Within every repeat:

1. The early-160 run-mean residual screen is recomputed using independent
   normal and Fault-3 development sets and a 2,000-replicate run bootstrap.
2. Stable residuals are ranked by absolute Hedges' g and the Top-5 are frozen.
3. Signs, normal scaling, and the 95th-percentile CUSUM limit are learned only
   from the repeat's development/calibration data.
4. Performance is evaluated on the complementary normal and Fault-3 runs.

This is a split-robustness analysis, not a replacement for the untouched
primary final test.  Every repeat is reported and the original Top-5 and
k=0.05 remain frozen.

Place in:
    TEP_Fault3_Project/scripts/14_repeated_split_robustness.py

Run from project root:
    python scripts/14_repeated_split_robustness.py

Prerequisites: Steps 02 and 11.  Keep
``11_unpaired_runlevel_screening.py`` in the scripts directory.
"""

from pathlib import Path
import gc
import importlib.util
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_WINDOW = "early_160_primary"
WINDOW_START_SAMPLE = 161
WINDOW_END_SAMPLE = 320
N_REPEATS = 20
N_DEVELOPMENT_RUNS = 250
N_BOOTSTRAP_REPLICATES = 2000
RANDOM_SPLIT_SEED = 20260715
PRIMARY_TOP_K = 5
PRIMARY_CUSUM_K = 0.05

SUMMARY_METRICS = [
    "n_direction_stable_variables",
    "overlap_with_primary_top5",
    "normal_run_level_far",
    "fault3_pre_fault_run_far",
    "fault3_run_level_detection_rate",
    "fault3_post_fault_point_detection_rate",
    "median_alarm_delay_min",
    "mean_alarm_delay_min",
]


def load_step11_module():
    path = SCRIPT_DIR / "11_unpaired_runlevel_screening.py"
    if not path.exists():
        raise FileNotFoundError(
            f"Required Step 11 script not found: {path}\n"
            "Keep 11_unpaired_runlevel_screening.py in the scripts directory."
        )
    spec = importlib.util.spec_from_file_location("step11_unpaired", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "load_variable_columns",
        "load_data",
        "fit_pca_model",
        "compute_residual_frame",
        "calculate_run_means",
        "screen_unpaired_runlevel",
        "evaluate_selected_method",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Step 11 script is missing functions: {missing}")
    return module


def require_file(path, instruction):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\nPlease {instruction} first."
        )
    return path


def load_primary_reference():
    selected_path = require_file(
        TABLE_DIR / "11_unpaired_selected_variables.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    selected_all = pd.read_csv(selected_path)
    primary_selected = selected_all[
        selected_all["window"].astype(str) == PRIMARY_WINDOW
    ].sort_values("selection_rank")
    if len(primary_selected) != PRIMARY_TOP_K:
        raise ValueError(
            f"Expected {PRIMARY_TOP_K} primary selected variables, found "
            f"{len(primary_selected)}"
        )
    primary_selected = primary_selected.copy()
    primary_selected["sign_for_positive_evidence"] = primary_selected[
        "sign_for_positive_evidence"
    ].astype(int)

    performance_path = require_file(
        TABLE_DIR / "11_unpaired_cusum_performance.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    performance = pd.read_csv(performance_path)
    reference_rows = performance[
        performance["screening_window"].astype(str) == PRIMARY_WINDOW
    ]
    if len(reference_rows) != 1:
        raise ValueError(
            "Expected exactly one Step 11 primary performance row, found "
            f"{len(reference_rows)}"
        )
    return primary_selected, reference_rows.iloc[0].copy()


def validate_combined_runs(frame, name):
    run_ids = np.sort(frame["simulationRun"].unique())
    if len(run_ids) != 500:
        raise ValueError(f"Expected 500 {name} runs, found {len(run_ids)}")
    counts = frame.groupby("simulationRun").size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != 960:
        raise ValueError(
            f"Expected 960 samples in every {name} run; count range is "
            f"{counts.min()}-{counts.max()}"
        )
    return run_ids


def prepare_combined_residuals(step11, data, variable_cols, model):
    print("Computing PCA residuals once for all 500 normal and Fault-3 runs...")
    normal_dev_res, residual_cols = step11.compute_residual_frame(
        data["normal_validation"], variable_cols, model
    )
    normal_eval_res, _ = step11.compute_residual_frame(
        data["normal_evaluation"], variable_cols, model
    )
    fault_dev_res, _ = step11.compute_residual_frame(
        data["fault3_development"], variable_cols, model
    )
    fault_eval_res, _ = step11.compute_residual_frame(
        data["fault3_final_test"], variable_cols, model
    )
    normal_all = pd.concat(
        [normal_dev_res, normal_eval_res], ignore_index=True
    ).sort_values(["simulationRun", "sample"]).reset_index(drop=True)
    fault_all = pd.concat(
        [fault_dev_res, fault_eval_res], ignore_index=True
    ).sort_values(["simulationRun", "sample"]).reset_index(drop=True)
    normal_ids = validate_combined_runs(normal_all, "normal")
    fault_ids = validate_combined_runs(fault_all, "Fault-3")
    return normal_all, fault_all, residual_cols, normal_ids, fault_ids


def subset_runs(frame, run_ids):
    run_set = set(int(x) for x in run_ids)
    return frame[frame["simulationRun"].isin(run_set)].copy()


def annotate_run_results(frame, repeat_id, split_seed):
    out = frame.copy()
    out.insert(0, "repeat_id", int(repeat_id))
    out.insert(1, "split_seed", int(split_seed))
    return out


def run_one_repeat(
    step11,
    repeat_id,
    normal_all,
    fault_all,
    residual_cols,
    variable_cols,
    normal_ids,
    fault_ids,
    primary_selected,
):
    normal_seed = RANDOM_SPLIT_SEED + 2 * repeat_id
    fault_seed = RANDOM_SPLIT_SEED + 2 * repeat_id + 1
    bootstrap_seed = RANDOM_SPLIT_SEED + 10000 + repeat_id
    normal_rng = np.random.default_rng(normal_seed)
    fault_rng = np.random.default_rng(fault_seed)

    normal_permutation = normal_rng.permutation(normal_ids)
    fault_permutation = fault_rng.permutation(fault_ids)
    normal_development_ids = np.sort(
        normal_permutation[:N_DEVELOPMENT_RUNS]
    )
    normal_evaluation_ids = np.sort(
        normal_permutation[N_DEVELOPMENT_RUNS:]
    )
    fault_development_ids = np.sort(
        fault_permutation[:N_DEVELOPMENT_RUNS]
    )
    fault_evaluation_ids = np.sort(
        fault_permutation[N_DEVELOPMENT_RUNS:]
    )

    normal_development = subset_runs(normal_all, normal_development_ids)
    normal_evaluation = subset_runs(normal_all, normal_evaluation_ids)
    fault_development = subset_runs(fault_all, fault_development_ids)
    fault_evaluation = subset_runs(fault_all, fault_evaluation_ids)

    normal_run_means = step11.calculate_run_means(
        normal_development,
        residual_cols,
        WINDOW_START_SAMPLE,
        WINDOW_END_SAMPLE,
    )
    fault_run_means = step11.calculate_run_means(
        fault_development,
        residual_cols,
        WINDOW_START_SAMPLE,
        WINDOW_END_SAMPLE,
    )
    screening, stable, selected = step11.screen_unpaired_runlevel(
        normal_run_means,
        fault_run_means,
        variable_cols,
        f"repeat_{repeat_id:02d}",
        WINDOW_START_SAMPLE,
        WINDOW_END_SAMPLE,
        bootstrap_seed,
    )
    if len(selected) != PRIMARY_TOP_K:
        raise ValueError(
            f"Repeat {repeat_id} selected {len(selected)} variables instead "
            f"of {PRIMARY_TOP_K}; stable candidate count={len(stable)}"
        )

    result = step11.evaluate_selected_method(
        f"Repeated split {repeat_id:02d}",
        f"repeat_{repeat_id:02d}",
        selected,
        normal_development,
        normal_evaluation,
        fault_evaluation,
    )

    primary_variables = primary_selected["variable"].astype(str).tolist()
    primary_set = set(primary_variables)
    primary_signs = dict(
        zip(
            primary_selected["variable"].astype(str),
            primary_selected["sign_for_positive_evidence"].astype(int),
        )
    )
    selected_variables = selected["variable"].astype(str).tolist()
    selected_set = set(selected_variables)
    overlap = selected_set & primary_set
    matching_signs = sum(
        int(
            selected.loc[
                selected["variable"].astype(str) == variable,
                "sign_for_positive_evidence",
            ].iloc[0]
        )
        == primary_signs[variable]
        for variable in overlap
    )

    performance = {
        "repeat_id": int(repeat_id),
        "normal_split_seed": int(normal_seed),
        "fault_split_seed": int(fault_seed),
        "bootstrap_seed": int(bootstrap_seed),
        "n_direction_stable_variables": int(len(stable)),
        "selected_variables": ";".join(selected_variables),
        "selected_signs": ";".join(
            selected["sign_for_positive_evidence"].astype(int).astype(str)
        ),
        "overlap_with_primary_top5": int(len(overlap)),
        "jaccard_with_primary_top5": float(
            len(overlap) / len(selected_set | primary_set)
        ),
        "exact_primary_top5_set": bool(selected_set == primary_set),
        "exact_primary_top5_order": bool(selected_variables == primary_variables),
        "primary_overlap_sign_agreement": float(
            matching_signs / len(overlap) if overlap else np.nan
        ),
    }
    performance.update(result["metrics"])

    screening = screening.copy()
    screening.insert(0, "repeat_id", int(repeat_id))
    screening.insert(1, "normal_split_seed", int(normal_seed))
    screening.insert(2, "fault_split_seed", int(fault_seed))
    screening.insert(3, "bootstrap_seed", int(bootstrap_seed))
    selected = selected.copy()
    selected.insert(0, "repeat_id", int(repeat_id))
    selected.insert(1, "normal_split_seed", int(normal_seed))
    selected.insert(2, "fault_split_seed", int(fault_seed))
    selected.insert(3, "bootstrap_seed", int(bootstrap_seed))

    split_manifest = {
        "repeat_id": int(repeat_id),
        "normal_split_seed": int(normal_seed),
        "fault_split_seed": int(fault_seed),
        "bootstrap_seed": int(bootstrap_seed),
        "normal_development_runs": ";".join(
            normal_development_ids.astype(str)
        ),
        "normal_evaluation_runs": ";".join(
            normal_evaluation_ids.astype(str)
        ),
        "fault3_development_runs": ";".join(
            fault_development_ids.astype(str)
        ),
        "fault3_evaluation_runs": ";".join(
            fault_evaluation_ids.astype(str)
        ),
        "same_number_development_overlap": int(
            len(
                set(normal_development_ids).intersection(
                    set(fault_development_ids)
                )
            )
        ),
        "pairing_used": False,
    }
    return {
        "performance": performance,
        "screening": screening,
        "selected": selected,
        "normal_runs": annotate_run_results(
            result["normal_run_results"], repeat_id, normal_seed
        ),
        "fault_runs": annotate_run_results(
            result["fault3_run_results"], repeat_id, fault_seed
        ),
        "split_manifest": split_manifest,
    }


def summarize_metrics(performance, primary_reference):
    rows = []
    for metric in SUMMARY_METRICS:
        values = pd.to_numeric(performance[metric], errors="coerce").dropna()
        reference = (
            float(primary_reference[metric])
            if metric in primary_reference.index
            else np.nan
        )
        rows.append(
            {
                "metric": metric,
                "n_repeats_with_value": int(len(values)),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "median": float(values.median()),
                "percentile_2_5": float(values.quantile(0.025)),
                "percentile_97_5": float(values.quantile(0.975)),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "primary_fixed_split_value": reference,
            }
        )
    return pd.DataFrame(rows)


def summarize_variable_stability(
    all_screening, selected_all, variable_cols, primary_selected
):
    primary_signs = dict(
        zip(
            primary_selected["variable"].astype(str),
            primary_selected["sign_for_positive_evidence"].astype(int),
        )
    )
    primary_set = set(primary_signs)
    rows = []
    for variable in variable_cols:
        screening = all_screening[
            all_screening["variable"].astype(str) == variable
        ].copy()
        selected = selected_all[
            selected_all["variable"].astype(str) == variable
        ].copy()
        signs = screening["sign_for_positive_evidence"].astype(int)
        modal_sign = int(signs.mode().iloc[0])
        sign_consistency = float((signs == modal_sign).mean())
        if len(selected):
            selected_sign_consistency = float(
                (
                    selected["sign_for_positive_evidence"].astype(int)
                    == selected[
                        "sign_for_positive_evidence"
                    ].astype(int).mode().iloc[0]
                ).mean()
            )
            median_rank = float(selected["selection_rank"].median())
        else:
            selected_sign_consistency = np.nan
            median_rank = np.nan
        if variable in primary_set:
            primary_sign_agreement = float(
                (signs == primary_signs[variable]).mean()
            )
        else:
            primary_sign_agreement = np.nan
        rows.append(
            {
                "variable": variable,
                "residual_variable": f"res_{variable}",
                "is_primary_top5": variable in primary_set,
                "primary_sign": primary_signs.get(variable, np.nan),
                "stable_frequency": float(
                    screening["direction_stable_95ci"].astype(bool).mean()
                ),
                "top5_selection_frequency": float(len(selected) / N_REPEATS),
                "median_selection_rank_when_selected": median_rank,
                "modal_sign_across_repeats": modal_sign,
                "sign_consistency_across_repeats": sign_consistency,
                "sign_consistency_when_selected": selected_sign_consistency,
                "agreement_with_primary_sign": primary_sign_agreement,
                "median_abs_hedges_g": float(
                    screening["abs_hedges_g"].median()
                ),
                "median_bootstrap_direction_confidence": float(
                    screening["bootstrap_direction_confidence"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["top5_selection_frequency", "median_selection_rank_when_selected"],
        ascending=[False, True],
        na_position="last",
    )


def plot_performance_distribution(performance, primary_reference):
    specs = [
        ("normal_run_level_far", 100.0, "Normal run-level FAR (%)"),
        (
            "fault3_post_fault_point_detection_rate",
            100.0,
            "Post-fault point detection (%)",
        ),
        ("median_alarm_delay_min", 1.0, "Median alarm delay (min)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.8))
    rng = np.random.default_rng(20260715)
    for ax, (metric, scale, ylabel) in zip(axes, specs):
        values = performance[metric].to_numpy(dtype=float) * scale
        ax.boxplot(
            values,
            widths=0.45,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "white"},
        )
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            1 + jitter,
            values,
            s=24,
            color="#4C78A8",
            alpha=0.75,
            zorder=3,
        )
        reference_value = float(primary_reference[metric]) * scale
        ax.axhline(
            reference_value,
            color="#D95F59",
            linestyle="--",
            linewidth=1.4,
            label="Primary fixed split",
        )
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    fig.suptitle("Pairing-free method across 20 independent random run splits")
    fig.tight_layout()
    png = FIGURE_DIR / "fig14a_repeated_split_performance.png"
    pdf = FIGURE_DIR / "fig14a_repeated_split_performance.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_selection_frequency(variable_stability):
    shown = variable_stability[
        variable_stability["top5_selection_frequency"] > 0
    ].copy()
    colors = [
        "#D95F59" if flag else "#4C78A8"
        for flag in shown["is_primary_top5"]
    ]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    ax.bar(
        shown["variable"],
        100 * shown["top5_selection_frequency"],
        color=colors,
    )
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-5 selection frequency across splits (%)")
    ax.set_title("Variable-selection stability under random run splits")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png = FIGURE_DIR / "fig14b_repeated_split_selection_frequency.png"
    pdf = FIGURE_DIR / "fig14b_repeated_split_selection_frequency.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def plot_primary_overlap(performance):
    counts = (
        performance["overlap_with_primary_top5"]
        .value_counts()
        .reindex(range(0, PRIMARY_TOP_K + 1), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(counts.index, counts.values, color="#7A9E9F")
    ax.set_xticks(range(0, PRIMARY_TOP_K + 1))
    ax.set_xlabel("Overlap with frozen primary Top-5")
    ax.set_ylabel("Number of random splits")
    ax.set_title("Top-5 set overlap under repeated random run splits")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    png = FIGURE_DIR / "fig14c_repeated_split_top5_overlap.png"
    pdf = FIGURE_DIR / "fig14c_repeated_split_top5_overlap.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [png, pdf]


def main():
    print("=" * 80)
    print("Step 14: Repeated random run-level split robustness")
    print("=" * 80)
    print(
        f"Repeats={N_REPEATS}, development/evaluation runs="
        f"{N_DEVELOPMENT_RUNS}/{500-N_DEVELOPMENT_RUNS}, "
        f"run-bootstrap replicates={N_BOOTSTRAP_REPLICATES}"
    )
    print(
        "Normal and Fault-3 run splits are generated independently; no "
        "normal/fault run pairing is used."
    )

    step11 = load_step11_module()
    step11.BOOTSTRAP_REPLICATES = N_BOOTSTRAP_REPLICATES
    step11.BOOTSTRAP_BATCH_SIZE = min(
        int(step11.BOOTSTRAP_BATCH_SIZE), N_BOOTSTRAP_REPLICATES
    )
    if not np.isclose(float(step11.CUSUM_K), PRIMARY_CUSUM_K):
        raise ValueError(
            f"Step 11 CUSUM k is {step11.CUSUM_K}, expected {PRIMARY_CUSUM_K}"
        )

    variable_cols = step11.load_variable_columns()
    primary_selected, primary_reference = load_primary_reference()
    print(
        "Frozen primary Top-5: "
        f"{primary_selected['variable'].astype(str).tolist()}"
    )
    data = step11.load_data()
    model = step11.fit_pca_model(data["normal_train"], variable_cols)
    normal_all, fault_all, residual_cols, normal_ids, fault_ids = (
        prepare_combined_residuals(step11, data, variable_cols, model)
    )
    del data
    gc.collect()

    performance_rows = []
    screening_rows = []
    selected_rows = []
    normal_run_rows = []
    fault_run_rows = []
    split_rows = []

    for repeat_id in range(1, N_REPEATS + 1):
        result = run_one_repeat(
            step11,
            repeat_id,
            normal_all,
            fault_all,
            residual_cols,
            variable_cols,
            normal_ids,
            fault_ids,
            primary_selected,
        )
        performance_rows.append(result["performance"])
        screening_rows.append(result["screening"])
        selected_rows.append(result["selected"])
        normal_run_rows.append(result["normal_runs"])
        fault_run_rows.append(result["fault_runs"])
        split_rows.append(result["split_manifest"])
        metrics = result["performance"]
        print(
            f"Repeat {repeat_id:02d}/{N_REPEATS}: stable="
            f"{metrics['n_direction_stable_variables']}, selected="
            f"{metrics['selected_variables']}, overlap="
            f"{metrics['overlap_with_primary_top5']}/5, FAR="
            f"{100*metrics['normal_run_level_far']:.2f}%, run detection="
            f"{100*metrics['fault3_run_level_detection_rate']:.2f}%, point="
            f"{100*metrics['fault3_post_fault_point_detection_rate']:.2f}%, "
            f"delay={metrics['median_alarm_delay_min']} min"
        )

    performance = pd.DataFrame(performance_rows)
    all_screening = pd.concat(screening_rows, ignore_index=True)
    selected_all = pd.concat(selected_rows, ignore_index=True)
    normal_runs_all = pd.concat(normal_run_rows, ignore_index=True)
    fault_runs_all = pd.concat(fault_run_rows, ignore_index=True)
    split_manifest = pd.DataFrame(split_rows)
    metric_summary = summarize_metrics(performance, primary_reference)
    variable_stability = summarize_variable_stability(
        all_screening, selected_all, variable_cols, primary_selected
    )

    paths = {
        "performance": TABLE_DIR / "14_repeated_split_performance.csv",
        "metric_summary": TABLE_DIR / "14_repeated_split_metric_summary.csv",
        "selected_variables": TABLE_DIR
        / "14_repeated_split_selected_variables.csv",
        "variable_stability": TABLE_DIR
        / "14_repeated_split_variable_stability.csv",
        "all_variable_screening": TABLE_DIR
        / "14_repeated_split_all_variable_screening.csv",
        "normal_run_results": TABLE_DIR
        / "14_repeated_split_normal_run_results.csv",
        "fault3_run_results": TABLE_DIR
        / "14_repeated_split_fault3_run_results.csv",
        "split_manifest": TABLE_DIR / "14_repeated_split_manifest.csv",
    }
    performance.to_csv(paths["performance"], index=False, encoding="utf-8-sig")
    metric_summary.to_csv(
        paths["metric_summary"], index=False, encoding="utf-8-sig"
    )
    selected_all.to_csv(
        paths["selected_variables"], index=False, encoding="utf-8-sig"
    )
    variable_stability.to_csv(
        paths["variable_stability"], index=False, encoding="utf-8-sig"
    )
    all_screening.to_csv(
        paths["all_variable_screening"], index=False, encoding="utf-8-sig"
    )
    normal_runs_all.to_csv(
        paths["normal_run_results"], index=False, encoding="utf-8-sig"
    )
    fault_runs_all.to_csv(
        paths["fault3_run_results"], index=False, encoding="utf-8-sig"
    )
    split_manifest.to_csv(
        paths["split_manifest"], index=False, encoding="utf-8-sig"
    )

    figure_paths = []
    figure_paths += plot_performance_distribution(
        performance, primary_reference
    )
    figure_paths += plot_selection_frequency(variable_stability)
    figure_paths += plot_primary_overlap(performance)

    exact_set_count = int(performance["exact_primary_top5_set"].sum())
    all_detected_count = int(
        np.isclose(performance["fault3_run_level_detection_rate"], 1.0).sum()
    )
    config = {
        "script": "scripts/14_repeated_split_robustness.py",
        "analysis_role": (
            "Split-robustness analysis; does not replace the frozen primary "
            "fixed-split final test"
        ),
        "n_repeats": N_REPEATS,
        "n_total_normal_runs": int(len(normal_ids)),
        "n_total_fault3_runs": int(len(fault_ids)),
        "n_development_runs_per_class": N_DEVELOPMENT_RUNS,
        "n_evaluation_runs_per_class": int(500 - N_DEVELOPMENT_RUNS),
        "normal_and_fault_splits_independent": True,
        "pairing_used": False,
        "screening_window": [WINDOW_START_SAMPLE, WINDOW_END_SAMPLE],
        "bootstrap_replicates_per_repeat": N_BOOTSTRAP_REPLICATES,
        "base_random_seed": RANDOM_SPLIT_SEED,
        "primary_top_k_frozen": PRIMARY_TOP_K,
        "primary_cusum_k_frozen": PRIMARY_CUSUM_K,
        "control_limit_rule": (
            "95th percentile of each repeat's normal-development run maxima"
        ),
        "pca_training_data": "data/processed/normal_train.pkl",
        "pca_n_components": int(model["n_pcs"]),
        "pca_cumulative_explained_variance": float(model["cum_var"]),
        "primary_selected_variables": primary_selected[
            "variable"
        ].astype(str).tolist(),
        "primary_selected_signs": primary_selected[
            "sign_for_positive_evidence"
        ].astype(int).tolist(),
        "summary": {
            "exact_primary_top5_set_recovery_count": exact_set_count,
            "all_250_fault_runs_detected_repeat_count": all_detected_count,
            "median_top5_overlap": float(
                performance["overlap_with_primary_top5"].median()
            ),
        },
        "output_tables": [
            str(path.relative_to(PROJECT_ROOT)) for path in paths.values()
        ],
        "output_figures": [
            str(path.relative_to(PROJECT_ROOT)) for path in figure_paths
        ],
    }
    config_path = CONFIG_DIR / "fault3_repeated_split_robustness.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Step 14 summary")
    print("=" * 80)
    print(
        f"Exact frozen Top-5 set recovered: {exact_set_count}/{N_REPEATS} "
        f"({100*exact_set_count/N_REPEATS:.1f}%)"
    )
    print(
        "Median overlap with frozen Top-5: "
        f"{performance['overlap_with_primary_top5'].median():.1f}/5"
    )
    print(
        f"Repeats with 250/250 Fault-3 runs detected: "
        f"{all_detected_count}/{N_REPEATS}"
    )
    print("\nRepeated-split metric summary:")
    print(metric_summary.to_string(index=False))
    print("\nMost frequently selected variables:")
    print(
        variable_stability[
            [
                "variable",
                "is_primary_top5",
                "stable_frequency",
                "top5_selection_frequency",
                "sign_consistency_across_repeats",
            ]
        ].head(12).to_string(index=False)
    )
    print(f"\nSaved configuration: {config_path}")
    print("Step 14 finished successfully.")


if __name__ == "__main__":
    main()
