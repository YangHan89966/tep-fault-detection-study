"""
Step 12: Factorized ablation for the pairing-free Step 11 method.

This script replaces the historical Step 06b evidence with a controlled
ablation based on the prespecified Step 11 primary window
(``early_160_primary``).  All residual signs, the bootstrap-stable residual
set, and the Top-5 set come from the pairing-free run-level screen.

Methods compared
----------------
1. All 52 residuals, no sign unification, CUSUM.
2. All 52 residuals, unpaired development signs, CUSUM.
3. All bootstrap-stable residuals, unpaired signs, CUSUM.
4. Unpaired Top-5 residuals, no sign unification, CUSUM.
5. Unpaired signed Top-5 residuals, no CUSUM.
6. Full unpaired signed Top-5 residual CUSUM.

Prerequisites: run Steps 02, 11 first and keep
``11_unpaired_runlevel_screening.py`` in the scripts directory.

Place in:
    TEP_Fault3_Project/scripts/12_unpaired_factorized_ablation.py

Run from project root:
    python scripts/12_unpaired_factorized_ablation.py

Only files beginning with ``12_`` are written.
"""

from pathlib import Path
import importlib.util
import json
import re

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
        "fit_signed_scaler",
        "construct_local_signed_index",
        "compute_cusum",
        "determine_limit",
        "evaluate_normal",
        "evaluate_fault",
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


def safe_slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def read_step11_selections(variable_cols):
    all_path = require_file(
        TABLE_DIR / "11_unpaired_screening_all_variables.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    selected_path = require_file(
        TABLE_DIR / "11_unpaired_selected_variables.csv",
        "run scripts/11_unpaired_runlevel_screening.py",
    )
    all_metrics = pd.read_csv(all_path)
    selected_all = pd.read_csv(selected_path)

    primary = all_metrics[
        all_metrics["window"].astype(str) == PRIMARY_WINDOW
    ].copy()
    selected = selected_all[
        selected_all["window"].astype(str) == PRIMARY_WINDOW
    ].copy()
    if len(primary) != len(variable_cols):
        raise ValueError(
            f"Expected {len(variable_cols)} primary-window residual rows, "
            f"found {len(primary)}"
        )
    if len(selected) != 5:
        raise ValueError(f"Expected 5 primary Top-K residuals, found {len(selected)}")

    primary_by_var = primary.set_index("variable").loc[variable_cols].reset_index()
    stable_mask = primary_by_var["direction_stable_95ci"].astype(bool)
    stable = primary_by_var[stable_mask].copy()
    if len(stable) < len(selected):
        raise ValueError("Bootstrap-stable set is smaller than the selected Top-5")

    required = {
        "variable",
        "residual_variable",
        "sign_for_positive_evidence",
    }
    for name, table in [
        ("primary all-variable table", primary_by_var),
        ("primary stable table", stable),
        ("primary selected table", selected),
    ]:
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")
        signs = set(table["sign_for_positive_evidence"].astype(int).unique())
        if not signs.issubset({-1, 1}):
            raise ValueError(f"{name} contains invalid signs: {sorted(signs)}")

    return primary_by_var, stable, selected.sort_values("selection_rank")


def sign_dict(table):
    return {
        str(row["residual_variable"]): int(row["sign_for_positive_evidence"])
        for _, row in table.iterrows()
    }


def make_selection_table(residual_cols, signs):
    rows = []
    for col in residual_cols:
        variable = col[4:] if col.startswith("res_") else col
        rows.append(
            {
                "variable": variable,
                "residual_variable": col,
                "sign_for_positive_evidence": 1
                if signs is None
                else int(signs[col]),
            }
        )
    return pd.DataFrame(rows)


def use_raw_local_index(local_df):
    df = local_df.sort_values(["simulationRun", "sample"]).copy()
    df["monitoring_statistic"] = df["local_signed_residual"]
    return df


def run_method(
    step11,
    method,
    normal_validation_res,
    normal_evaluation_res,
    fault_final_res,
):
    name = method["method"]
    residual_cols = method["residual_cols"]
    signs = method["signs_dict"]
    use_cusum = bool(method["use_cusum"])
    selected_table = make_selection_table(residual_cols, signs)

    print("\n" + "=" * 80)
    print(f"Running: {name}")
    print("=" * 80)

    scaler = step11.fit_signed_scaler(normal_validation_res, selected_table)
    validation_local = step11.construct_local_signed_index(
        normal_validation_res, scaler
    )
    normal_local = step11.construct_local_signed_index(
        normal_evaluation_res, scaler
    )
    fault_local = step11.construct_local_signed_index(fault_final_res, scaler)

    if use_cusum:
        validation_stat = step11.compute_cusum(validation_local)
        normal_stat = step11.compute_cusum(normal_local)
        fault_stat = step11.compute_cusum(fault_local)
    else:
        validation_stat = use_raw_local_index(validation_local)
        normal_stat = use_raw_local_index(normal_local)
        fault_stat = use_raw_local_index(fault_local)

    limit, validation_run_max = step11.determine_limit(validation_stat)
    normal_metrics, normal_runs = step11.evaluate_normal(normal_stat, limit)
    fault_metrics, fault_runs = step11.evaluate_fault(fault_stat, limit)

    metrics = {
        "method": name,
        "method_short": method["method_short"],
        "slug": method["slug"],
        "residual_set": method["residual_set"],
        "n_residual_variables": len(residual_cols),
        "direction_unification": signs is not None,
        "sign_source": method["sign_source"],
        "use_cusum": use_cusum,
        "cusum_k": step11.CUSUM_K if use_cusum else np.nan,
        "run_level_percentile": step11.RUN_LEVEL_PERCENTILE,
        "control_limit": limit,
    }
    metrics.update(normal_metrics)
    metrics.update(fault_metrics)
    print(
        f"Normal FAR={100 * metrics['normal_run_level_far']:.2f}%, "
        f"Fault 3 detection="
        f"{100 * metrics['fault3_run_level_detection_rate']:.2f}%, "
        f"post-fault point detection="
        f"{100 * metrics['fault3_post_fault_point_detection_rate']:.2f}%, "
        f"median delay={metrics['median_alarm_delay_min']} min"
    )
    return {
        "metrics": metrics,
        "scaler": scaler,
        "validation_run_max": validation_run_max,
        "normal_run_results": normal_runs,
        "fault3_run_results": fault_runs,
    }


def save_method_outputs(method, result):
    slug = method["slug"]
    paths = {
        "scaler": TABLE_DIR / f"12_{slug}_scaler.csv",
        "validation_run_max": TABLE_DIR
        / f"12_{slug}_validation_run_max.csv",
        "normal_run_results": TABLE_DIR
        / f"12_{slug}_normal_run_results.csv",
        "fault3_run_results": TABLE_DIR
        / f"12_{slug}_fault3_run_results.csv",
    }
    for key, path in paths.items():
        result[key].to_csv(path, index=False, encoding="utf-8-sig")
    return {key: str(path.relative_to(PROJECT_ROOT)) for key, path in paths.items()}


def make_contrasts(metrics_df):
    specs = [
        (
            "Direction unification within unpaired Top-5",
            "Unpaired Top-5 residuals w/o sign + CUSUM",
            "Unpaired signed Top-5 residuals + CUSUM",
            "direction unification",
        ),
        (
            "CUSUM within signed unpaired Top-5",
            "Unpaired signed Top-5 residuals w/o CUSUM",
            "Unpaired signed Top-5 residuals + CUSUM",
            "sequential accumulation",
        ),
        (
            "Top-5 selection after unpaired sign unification",
            "All residuals with unpaired signs + CUSUM",
            "Unpaired signed Top-5 residuals + CUSUM",
            "local Top-K selection",
        ),
        (
            "Bootstrap stability filtering",
            "All residuals with unpaired signs + CUSUM",
            "Bootstrap-stable signed residuals + CUSUM",
            "95% run-bootstrap stability filter",
        ),
        (
            "Top-5 compression within bootstrap-stable set",
            "Bootstrap-stable signed residuals + CUSUM",
            "Unpaired signed Top-5 residuals + CUSUM",
            "absolute Hedges' g ranking and Top-K compression",
        ),
    ]
    indexed = metrics_df.set_index("method")
    rows = []
    for contrast, comparator_name, target_name, component in specs:
        comparator = indexed.loc[comparator_name]
        target = indexed.loc[target_name]
        rows.append(
            {
                "contrast": contrast,
                "comparator": comparator_name,
                "target": target_name,
                "isolated_component": component,
                "delta_normal_run_far_percentage_points": 100
                * (
                    target["normal_run_level_far"]
                    - comparator["normal_run_level_far"]
                ),
                "delta_fault3_detection_percentage_points": 100
                * (
                    target["fault3_run_level_detection_rate"]
                    - comparator["fault3_run_level_detection_rate"]
                ),
                "delta_postfault_point_detection_percentage_points": 100
                * (
                    target["fault3_post_fault_point_detection_rate"]
                    - comparator["fault3_post_fault_point_detection_rate"]
                ),
                "delta_median_delay_min": target["median_alarm_delay_min"]
                - comparator["median_alarm_delay_min"],
            }
        )
    return pd.DataFrame(rows)


def compare_with_step11(metrics_df):
    path = TABLE_DIR / "11_unpaired_cusum_performance.csv"
    if not path.exists():
        return {"available": False}
    step11_performance = pd.read_csv(path)
    reference_rows = step11_performance[
        step11_performance["screening_window"].astype(str) == PRIMARY_WINDOW
    ]
    if len(reference_rows) != 1:
        return {"available": True, "unique_primary_row": False}
    reference = reference_rows.iloc[0]
    proposed = metrics_df[
        metrics_df["method"]
        == "Unpaired signed Top-5 residuals + CUSUM"
    ].iloc[0]
    fields = [
        "normal_run_level_far",
        "fault3_run_level_detection_rate",
        "fault3_post_fault_point_detection_rate",
        "median_alarm_delay_min",
    ]
    differences = {
        field: float(proposed[field] - reference[field]) for field in fields
    }
    exact = all(abs(value) <= 1e-12 for value in differences.values())
    return {
        "available": True,
        "unique_primary_row": True,
        "exact_reproduction_with_tolerance_1e-12": exact,
        "differences": differences,
    }


def plot_results(metrics_df):
    x = np.arange(len(metrics_df))
    labels = metrics_df["method_short"]
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(
        x - width / 2,
        100 * metrics_df["normal_run_level_far"],
        width,
        label="Normal run-level FAR",
        color="#7A9E9F",
    )
    ax.bar(
        x + width / 2,
        100 * metrics_df["fault3_run_level_detection_rate"],
        width,
        label="Fault 3 run-level detection",
        color="#D95F59",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_title("Pairing-free factorized ablation: false alarms and detection")
    ax.legend()
    fig.tight_layout()
    fig1_png = FIGURE_DIR / "fig12a_unpaired_ablation_far_detection.png"
    fig1_pdf = FIGURE_DIR / "fig12a_unpaired_ablation_far_detection.pdf"
    fig.savefig(fig1_png, dpi=300)
    fig.savefig(fig1_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bars = ax.bar(
        x, metrics_df["median_alarm_delay_min"], color="#4C78A8"
    )
    for bar, value in zip(bars, metrics_df["median_alarm_delay_min"]):
        if pd.isna(value):
            bar.set_height(0)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0,
                "No detection",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Median alarm delay (min)")
    ax.set_title("Pairing-free factorized ablation: alarm delay")
    fig.tight_layout()
    fig2_png = FIGURE_DIR / "fig12b_unpaired_ablation_alarm_delay.png"
    fig2_pdf = FIGURE_DIR / "fig12b_unpaired_ablation_alarm_delay.pdf"
    fig.savefig(fig2_png, dpi=300)
    fig.savefig(fig2_pdf)
    plt.close(fig)
    return [fig1_png, fig1_pdf, fig2_png, fig2_pdf]


def main():
    print("=" * 80)
    print("Step 12: Pairing-free factorized ablation")
    print("=" * 80)
    step11 = load_step11_module()
    variable_cols = step11.load_variable_columns()
    primary, stable, selected = read_step11_selections(variable_cols)
    data = step11.load_data()
    model = step11.fit_pca_model(data["normal_train"], variable_cols)

    print("Computing PCA residuals for validation and final evaluation...")
    normal_validation_res, all_residual_cols = step11.compute_residual_frame(
        data["normal_validation"], variable_cols, model
    )
    normal_evaluation_res, _ = step11.compute_residual_frame(
        data["normal_evaluation"], variable_cols, model
    )
    fault_final_res, _ = step11.compute_residual_frame(
        data["fault3_final_test"], variable_cols, model
    )

    all_signs = sign_dict(primary)
    stable_cols = stable["residual_variable"].astype(str).tolist()
    stable_signs = sign_dict(stable)
    selected_cols = selected["residual_variable"].astype(str).tolist()
    selected_signs = sign_dict(selected)

    if set(all_signs) != set(all_residual_cols):
        raise ValueError("Step 11 all-variable sign table does not cover 52 residuals")

    methods = [
        {
            "method": "All residuals + CUSUM",
            "method_short": "All / no sign / CUSUM",
            "residual_set": "all_52",
            "residual_cols": all_residual_cols,
            "signs_dict": None,
            "sign_source": "none",
            "use_cusum": True,
            "purpose": "All-residual no-sign comparator.",
        },
        {
            "method": "All residuals with unpaired signs + CUSUM",
            "method_short": "All / unpaired signs / CUSUM",
            "residual_set": "all_52",
            "residual_cols": all_residual_cols,
            "signs_dict": all_signs,
            "sign_source": "Step 11 primary unpaired run-mean shifts",
            "use_cusum": True,
            "purpose": "All 52 residuals with pairing-free development signs.",
        },
        {
            "method": "Bootstrap-stable signed residuals + CUSUM",
            "method_short": "Stable set / signed / CUSUM",
            "residual_set": "bootstrap_stable_95ci",
            "residual_cols": stable_cols,
            "signs_dict": stable_signs,
            "sign_source": "Step 11 primary unpaired run-bootstrap screen",
            "use_cusum": True,
            "purpose": "All residuals whose unpaired 95% bootstrap CI excludes zero.",
        },
        {
            "method": "Unpaired Top-5 residuals w/o sign + CUSUM",
            "method_short": "Top-5 / no sign / CUSUM",
            "residual_set": "unpaired_top5",
            "residual_cols": selected_cols,
            "signs_dict": None,
            "sign_source": "none",
            "use_cusum": True,
            "purpose": "Direction-unification ablation within the unpaired Top-5.",
        },
        {
            "method": "Unpaired signed Top-5 residuals w/o CUSUM",
            "method_short": "Top-5 / signed / no CUSUM",
            "residual_set": "unpaired_top5",
            "residual_cols": selected_cols,
            "signs_dict": selected_signs,
            "sign_source": "Step 11 primary unpaired run-mean shifts",
            "use_cusum": False,
            "purpose": "Sequential-accumulation ablation within the unpaired Top-5.",
        },
        {
            "method": "Unpaired signed Top-5 residuals + CUSUM",
            "method_short": "Top-5 / signed / CUSUM",
            "residual_set": "unpaired_top5",
            "residual_cols": selected_cols,
            "signs_dict": selected_signs,
            "sign_source": "Step 11 primary unpaired run-mean shifts",
            "use_cusum": True,
            "purpose": "Full pairing-free proposed method.",
        },
    ]
    for method in methods:
        method["slug"] = safe_slug(method["method"])

    metric_rows = []
    output_manifest = {}
    for method in methods:
        result = run_method(
            step11,
            method,
            normal_validation_res,
            normal_evaluation_res,
            fault_final_res,
        )
        metric_rows.append(result["metrics"])
        output_manifest[method["method"]] = save_method_outputs(method, result)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = TABLE_DIR / "12_unpaired_factorized_ablation_results.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    definitions = pd.DataFrame(
        [
            {
                "method": method["method"],
                "method_short": method["method_short"],
                "residual_set": method["residual_set"],
                "n_residual_variables": len(method["residual_cols"]),
                "direction_unification": method["signs_dict"] is not None,
                "sign_source": method["sign_source"],
                "use_cusum": method["use_cusum"],
                "purpose": method["purpose"],
            }
            for method in methods
        ]
    )
    definitions_path = TABLE_DIR / "12_unpaired_ablation_definitions.csv"
    definitions.to_csv(definitions_path, index=False, encoding="utf-8-sig")

    contrasts = make_contrasts(metrics_df)
    contrasts_path = TABLE_DIR / "12_unpaired_ablation_contrasts.csv"
    contrasts.to_csv(contrasts_path, index=False, encoding="utf-8-sig")
    parity = compare_with_step11(metrics_df)
    figure_paths = plot_results(metrics_df)

    config_path = CONFIG_DIR / "fault3_unpaired_factorized_ablation_config.json"
    config = {
        "script": "scripts/12_unpaired_factorized_ablation.py",
        "primary_screening_window": PRIMARY_WINDOW,
        "selection_source": "results/tables/11_unpaired_selected_variables.csv",
        "all_signs_and_stability_source": (
            "results/tables/11_unpaired_screening_all_variables.csv"
        ),
        "n_all_residuals": len(all_residual_cols),
        "n_bootstrap_stable_residuals": len(stable_cols),
        "n_selected_residuals": len(selected_cols),
        "selected_variables": selected["variable"].tolist(),
        "selected_signs": selected[
            "sign_for_positive_evidence"
        ].astype(int).tolist(),
        "pca_n_components": model["n_pcs"],
        "pca_cumulative_explained_variance": model["cum_var"],
        "cusum_k": step11.CUSUM_K,
        "run_level_percentile": step11.RUN_LEVEL_PERCENTILE,
        "methods": definitions.to_dict(orient="records"),
        "step11_full_method_reproduction_check": parity,
        "method_outputs": output_manifest,
        "summary_outputs": {
            "metrics": str(metrics_path.relative_to(PROJECT_ROOT)),
            "definitions": str(definitions_path.relative_to(PROJECT_ROOT)),
            "contrasts": str(contrasts_path.relative_to(PROJECT_ROOT)),
            "figures": [str(p.relative_to(PROJECT_ROOT)) for p in figure_paths],
        },
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Step 12 summary")
    print("=" * 80)
    print(
        metrics_df[
            [
                "method",
                "n_residual_variables",
                "normal_run_level_far",
                "fault3_run_level_detection_rate",
                "fault3_post_fault_point_detection_rate",
                "median_alarm_delay_min",
            ]
        ].to_string(index=False)
    )
    print(f"Step 11 full-method exact reproduction: {parity}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved contrasts: {contrasts_path}")
    print(f"Saved configuration: {config_path}")
    print("Step 12 finished successfully.")


if __name__ == "__main__":
    main()
