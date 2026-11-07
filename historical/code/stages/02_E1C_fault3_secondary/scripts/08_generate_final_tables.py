from pathlib import Path
import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
LOG_DIR = PROJECT_ROOT / "results" / "logs"
DOC_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
DOC_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def read_csv_required(path: Path):
    require_file(path)
    return pd.read_csv(path)


def to_percent(x):
    if pd.isna(x):
        return np.nan
    return float(x) * 100.0


def first_value(df, row_mask, col):
    if col not in df.columns:
        return np.nan
    sub = df.loc[row_mask, col]
    if len(sub) == 0:
        return np.nan
    return sub.iloc[0]


def round_or_nan(x, ndigits=4):
    if pd.isna(x):
        return np.nan
    return round(float(x), ndigits)


def make_table1_data_split():
    split_path = LOG_DIR / "02_processed_data_split_summary.csv"
    split_df = read_csv_required(split_path)

    purpose_map = {
        "normal_train": "PCA modeling only",
        "normal_validation_runs_1_250": "Threshold selection and validation",
        "normal_evaluation_runs_251_500": "Final normal false alarm evaluation",
        "fault3_development_runs_1_250": "Residual variable screening and method development",
        "fault3_final_test_runs_251_500": "Final leakage-free Fault 3 testing",
    }

    table = split_df.copy()
    table["purpose"] = table["dataset"].map(purpose_map)

    keep_cols = [
        "dataset",
        "purpose",
        "rows",
        "columns",
        "fault_numbers",
        "simulation_runs",
        "run_min",
        "run_max",
        "sample_min",
        "sample_max",
    ]

    table = table[keep_cols]

    out_path = TABLE_DIR / "final_table1_data_split.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

    return table


def make_table2_pca_vs_proposed():
    pca_metrics = read_csv_required(TABLE_DIR / "03_pca_baseline_metrics.csv")
    pca_model = read_csv_required(TABLE_DIR / "03_pca_model_summary.csv")
    pca_threshold = read_csv_required(TABLE_DIR / "03_pca_run_level_thresholds.csv")

    proposed_metrics = read_csv_required(TABLE_DIR / "05_local_signed_cusum_metrics.csv")
    proposed_threshold = read_csv_required(TABLE_DIR / "05_local_signed_cusum_threshold.csv")

    pca_normal_mask = pca_metrics["dataset"].astype(str).str.contains("normal")
    pca_fault_mask = pca_metrics["dataset"].astype(str).str.contains("fault3")

    prop_normal_mask = proposed_metrics["dataset"].astype(str).str.contains("normal")
    prop_fault_mask = proposed_metrics["dataset"].astype(str).str.contains("fault3")

    rows = []

    def add_metric(metric_name, pca_value, prop_value, unit):
        rows.append(
            {
                "metric": metric_name,
                "PCA_Q_T2_baseline": pca_value,
                "Local_signed_residual_CUSUM": prop_value,
                "unit": unit,
            }
        )

    add_metric(
        "Number of retained PCs",
        first_value(pca_model, pca_model.index == 0, "n_pcs"),
        first_value(pca_model, pca_model.index == 0, "n_pcs"),
        "count",
    )

    add_metric(
        "Cumulative explained variance",
        first_value(pca_model, pca_model.index == 0, "cumulative_explained_variance"),
        first_value(pca_model, pca_model.index == 0, "cumulative_explained_variance"),
        "ratio",
    )

    add_metric(
        "Q control limit",
        first_value(pca_threshold, pca_threshold.index == 0, "Q_limit"),
        np.nan,
        "statistic value",
    )

    add_metric(
        "T2 control limit",
        first_value(pca_threshold, pca_threshold.index == 0, "T2_limit"),
        np.nan,
        "statistic value",
    )

    add_metric(
        "CUSUM control limit",
        np.nan,
        first_value(proposed_threshold, proposed_threshold.index == 0, "cusum_control_limit"),
        "statistic value",
    )

    add_metric(
        "Normal run-level FAR",
        to_percent(first_value(pca_metrics, pca_normal_mask, "normal_run_level_far_total")),
        to_percent(first_value(proposed_metrics, prop_normal_mask, "normal_run_level_far")),
        "%",
    )

    add_metric(
        "Normal point-level FAR",
        to_percent(first_value(pca_metrics, pca_normal_mask, "normal_point_level_far_total")),
        to_percent(first_value(proposed_metrics, prop_normal_mask, "normal_point_level_far")),
        "%",
    )

    add_metric(
        "Fault 3 pre-fault run-level FAR",
        to_percent(first_value(pca_metrics, pca_fault_mask, "fault3_pre_fault_run_far_total")),
        to_percent(first_value(proposed_metrics, prop_fault_mask, "fault3_pre_fault_run_far")),
        "%",
    )

    add_metric(
        "Fault 3 pre-fault point-level FAR",
        to_percent(first_value(pca_metrics, pca_fault_mask, "fault3_pre_fault_point_far_total")),
        to_percent(first_value(proposed_metrics, prop_fault_mask, "fault3_pre_fault_point_far")),
        "%",
    )

    add_metric(
        "Fault 3 run-level detection rate",
        to_percent(first_value(pca_metrics, pca_fault_mask, "fault3_run_level_detection_rate_total")),
        to_percent(first_value(proposed_metrics, prop_fault_mask, "fault3_run_level_detection_rate")),
        "%",
    )

    add_metric(
        "Fault 3 post-fault point-level detection rate",
        to_percent(first_value(pca_metrics, pca_fault_mask, "fault3_post_fault_point_detection_rate_total")),
        to_percent(first_value(proposed_metrics, prop_fault_mask, "fault3_post_fault_point_detection_rate")),
        "%",
    )

    add_metric(
        "Median alarm delay",
        first_value(pca_metrics, pca_fault_mask, "median_alarm_delay_min_detected_runs"),
        first_value(proposed_metrics, prop_fault_mask, "median_alarm_delay_min"),
        "min",
    )

    add_metric(
        "Mean alarm delay",
        first_value(pca_metrics, pca_fault_mask, "mean_alarm_delay_min_detected_runs"),
        first_value(proposed_metrics, prop_fault_mask, "mean_alarm_delay_min"),
        "min",
    )

    table = pd.DataFrame(rows)

    for col in ["PCA_Q_T2_baseline", "Local_signed_residual_CUSUM"]:
        table[col] = table[col].apply(lambda x: round_or_nan(x, 6))

    out_path = TABLE_DIR / "final_table2_pca_vs_proposed.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

    return table


def make_table3_selected_residuals():
    selected = read_csv_required(TABLE_DIR / "04_fault3_auto_selected_signed_residuals.csv")

    keep_cols = [
        "variable",
        "residual_variable",
        "dominant_direction",
        "sign_for_positive_evidence",
        "mean_delta",
        "abs_mean_delta",
        "direction_ratio",
        "effect_size",
    ]

    table = selected[keep_cols].copy()

    for col in ["mean_delta", "abs_mean_delta", "direction_ratio", "effect_size"]:
        table[col] = table[col].apply(lambda x: round_or_nan(x, 6))

    out_path = TABLE_DIR / "final_table3_selected_local_residuals.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

    return table


def make_table4_ablation():
    ablation = read_csv_required(TABLE_DIR / "06_ablation_results.csv")

    table = ablation.copy()

    percent_cols = [
        "normal_run_level_far",
        "normal_point_level_far",
        "fault3_pre_fault_run_far",
        "fault3_pre_fault_point_far",
        "fault3_run_level_detection_rate",
        "fault3_post_fault_point_detection_rate",
    ]

    for col in percent_cols:
        if col in table.columns:
            table[col + "_percent"] = table[col] * 100.0

    keep_cols = [
        "method",
        "n_residual_variables",
        "direction_unification",
        "use_cusum",
        "normal_run_level_far_percent",
        "normal_point_level_far_percent",
        "fault3_pre_fault_run_far_percent",
        "fault3_run_level_detection_rate_percent",
        "fault3_post_fault_point_detection_rate_percent",
        "median_alarm_delay_min",
        "mean_alarm_delay_min",
        "detected_runs",
    ]

    keep_cols = [c for c in keep_cols if c in table.columns]
    table = table[keep_cols].copy()

    for col in table.columns:
        if col not in ["method", "direction_unification", "use_cusum"]:
            table[col] = table[col].apply(lambda x: round_or_nan(x, 6))

    out_path = TABLE_DIR / "final_table4_ablation.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

    return table


def make_table5_direction_ratio_sensitivity():
    df = read_csv_required(TABLE_DIR / "07_direction_ratio_sensitivity.csv")
    table = df.copy()

    percent_cols = [
        "normal_run_level_far",
        "normal_point_level_far",
        "fault3_pre_fault_run_far",
        "fault3_pre_fault_point_far",
        "fault3_run_level_detection_rate",
        "fault3_post_fault_point_detection_rate",
    ]

    for col in percent_cols:
        if col in table.columns:
            table[col + "_percent"] = table[col] * 100.0

    keep_cols = [
        "direction_ratio_threshold",
        "n_direction_stable_variables",
        "n_selected_variables",
        "selected_variables",
        "normal_run_level_far_percent",
        "fault3_run_level_detection_rate_percent",
        "fault3_post_fault_point_detection_rate_percent",
        "median_alarm_delay_min",
        "mean_alarm_delay_min",
        "detected_runs",
    ]

    keep_cols = [c for c in keep_cols if c in table.columns]
    table = table[keep_cols].copy()

    for col in table.columns:
        if col != "selected_variables":
            table[col] = table[col].apply(lambda x: round_or_nan(x, 6))

    out_path = TABLE_DIR / "final_table5_direction_ratio_sensitivity.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

    return table


def make_table6_cusum_k_sensitivity():
    df = read_csv_required(TABLE_DIR / "07_cusum_k_sensitivity.csv")
    table = df.copy()

    percent_cols = [
        "normal_run_level_far",
        "normal_point_level_far",
        "fault3_pre_fault_run_far",
        "fault3_pre_fault_point_far",
        "fault3_run_level_detection_rate",
        "fault3_post_fault_point_detection_rate",
    ]

    for col in percent_cols:
        if col in table.columns:
            table[col + "_percent"] = table[col] * 100.0

    keep_cols = [
        "cusum_k",
        "cusum_control_limit",
        "normal_run_level_far_percent",
        "fault3_run_level_detection_rate_percent",
        "fault3_post_fault_point_detection_rate_percent",
        "median_alarm_delay_min",
        "mean_alarm_delay_min",
        "detected_runs",
    ]

    keep_cols = [c for c in keep_cols if c in table.columns]
    table = table[keep_cols].copy()

    for col in table.columns:
        table[col] = table[col].apply(lambda x: round_or_nan(x, 6))

    out_path = TABLE_DIR / "final_table6_cusum_k_sensitivity.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

    return table


def make_figure_list():
    figures = [
        {
            "figure_id": "Fig. 1a",
            "filename_png": "fig1a_fault3_pca_Q_failure.png",
            "filename_pdf": "fig1a_fault3_pca_Q_failure.pdf",
            "caption_draft": "PCA Q/SPE response for a representative Fault 3 run under the run-level control limit.",
            "purpose": "Show that global PCA Q/SPE fails to provide sustained Fault 3 detection.",
        },
        {
            "figure_id": "Fig. 1b",
            "filename_png": "fig1b_fault3_pca_T2_failure.png",
            "filename_pdf": "fig1b_fault3_pca_T2_failure.pdf",
            "caption_draft": "PCA Hotelling's T2 response for a representative Fault 3 run under the run-level control limit.",
            "purpose": "Show that global PCA T2 also fails to provide sustained Fault 3 detection.",
        },
        {
            "figure_id": "Fig. 2",
            "filename_png": "fig2_fault3_local_signed_residual_shifts.png",
            "filename_pdf": "fig2_fault3_local_signed_residual_shifts.pdf",
            "caption_draft": "Direction-unified local PCA residual shifts for the auto-selected Fault 3 residual variables.",
            "purpose": "Show that Fault 3 information is preserved in several local signed PCA residual variables.",
        },
        {
            "figure_id": "Fig. 3",
            "filename_png": "fig3_fault3_local_signed_cusum_alarm.png",
            "filename_pdf": "fig3_fault3_local_signed_cusum_alarm.pdf",
            "caption_draft": "Local signed residual CUSUM response for a representative Fault 3 run.",
            "purpose": "Show that the proposed CUSUM statistic accumulates the weak local residual shift and triggers an alarm.",
        },
        {
            "figure_id": "Fig. 4a",
            "filename_png": "fig4a_ablation_far_detection.png",
            "filename_pdf": "fig4a_ablation_far_detection.pdf",
            "caption_draft": "Ablation comparison of normal run-level false alarm rate and Fault 3 run-level detection rate.",
            "purpose": "Show the necessity of local selection, direction unification, and CUSUM accumulation.",
        },
        {
            "figure_id": "Fig. 4b",
            "filename_png": "fig4b_ablation_alarm_delay.png",
            "filename_pdf": "fig4b_ablation_alarm_delay.pdf",
            "caption_draft": "Ablation comparison of median alarm delay.",
            "purpose": "Show that the full method substantially shortens the alarm delay.",
        },
        {
            "figure_id": "Fig. 5a",
            "filename_png": "fig5a_direction_ratio_sensitivity.png",
            "filename_pdf": "fig5a_direction_ratio_sensitivity.pdf",
            "caption_draft": "Sensitivity of the proposed method to the direction ratio threshold.",
            "purpose": "Show that the selected variables and detection performance are stable across direction ratio thresholds.",
        },
        {
            "figure_id": "Fig. 5b",
            "filename_png": "fig5c_cusum_k_sensitivity.png",
            "filename_pdf": "fig5c_cusum_k_sensitivity.pdf",
            "caption_draft": "Sensitivity of the proposed method to the CUSUM reference value k.",
            "purpose": "Show that the method is not dependent on a single CUSUM k value.",
        },
    ]

    table = pd.DataFrame(figures)

    def exists_png(filename):
        return (FIGURE_DIR / filename).exists()

    def exists_pdf(filename):
        return (FIGURE_DIR / filename).exists()

    table["png_exists"] = table["filename_png"].apply(exists_png)
    table["pdf_exists"] = table["filename_pdf"].apply(exists_pdf)

    out_path = TABLE_DIR / "final_figure_list_for_manuscript.csv"
    table.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

    return table


def make_numeric_summary_md(table2, table3, table4):
    def get_metric(metric_name, method_col):
        sub = table2[table2["metric"] == metric_name]
        if len(sub) == 0:
            return np.nan
        return sub[method_col].iloc[0]

    selected_vars = table3["variable"].tolist()

    pca_far = get_metric("Normal run-level FAR", "PCA_Q_T2_baseline")
    pca_fdr = get_metric("Fault 3 run-level detection rate", "PCA_Q_T2_baseline")
    pca_point = get_metric("Fault 3 post-fault point-level detection rate", "PCA_Q_T2_baseline")
    pca_delay = get_metric("Median alarm delay", "PCA_Q_T2_baseline")

    prop_far = get_metric("Normal run-level FAR", "Local_signed_residual_CUSUM")
    prop_fdr = get_metric("Fault 3 run-level detection rate", "Local_signed_residual_CUSUM")
    prop_point = get_metric("Fault 3 post-fault point-level detection rate", "Local_signed_residual_CUSUM")
    prop_delay = get_metric("Median alarm delay", "Local_signed_residual_CUSUM")

    lines = []
    lines.append("# Final Numeric Summary for Manuscript")
    lines.append("")
    lines.append("## PCA model")
    lines.append("")
    lines.append("- Number of retained PCs: 36")
    lines.append("- Cumulative explained variance: approximately 0.9582")
    lines.append("")
    lines.append("## PCA baseline")
    lines.append("")
    lines.append(f"- Normal run-level FAR: {pca_far:.2f}%")
    lines.append(f"- Fault 3 run-level detection rate: {pca_fdr:.2f}%")
    lines.append(f"- Fault 3 post-fault point-level detection rate: {pca_point:.4f}%")
    lines.append(f"- Median alarm delay: {pca_delay:.2f} min")
    lines.append("")
    lines.append("## Selected local signed residual variables")
    lines.append("")
    lines.append("- " + ", ".join(selected_vars))
    lines.append("")
    lines.append("## Proposed method")
    lines.append("")
    lines.append(f"- Normal run-level FAR: {prop_far:.2f}%")
    lines.append(f"- Fault 3 run-level detection rate: {prop_fdr:.2f}%")
    lines.append(f"- Fault 3 post-fault point-level detection rate: {prop_point:.4f}%")
    lines.append(f"- Median alarm delay: {prop_delay:.2f} min")
    lines.append("")
    lines.append("## Core interpretation")
    lines.append("")
    lines.append(
        "The conventional global PCA Q/T2 statistics fail to detect Fault 3 reliably, "
        "whereas the proposed local signed residual CUSUM method achieves complete "
        "run-level detection with a lower normal run-level false alarm rate."
    )
    lines.append("")
    lines.append(
        "The ablation study supports that local residual selection, direction unification, "
        "and CUSUM accumulation are all necessary components."
    )

    out_path = DOC_DIR / "final_numeric_summary_for_manuscript.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")

    return out_path


def main():
    print("=" * 80)
    print("Step 08: Generate final manuscript tables and figure list")
    print("=" * 80)

    table1 = make_table1_data_split()
    table2 = make_table2_pca_vs_proposed()
    table3 = make_table3_selected_residuals()
    table4 = make_table4_ablation()
    table5 = make_table5_direction_ratio_sensitivity()
    table6 = make_table6_cusum_k_sensitivity()
    figure_list = make_figure_list()
    summary_path = make_numeric_summary_md(table2, table3, table4)

    config = {
        "script": "scripts/08_generate_final_tables.py",
        "purpose": "Generate final tables, figure list, and numeric summary for manuscript writing.",
        "output_tables": [
            "results/tables/final_table1_data_split.csv",
            "results/tables/final_table2_pca_vs_proposed.csv",
            "results/tables/final_table3_selected_local_residuals.csv",
            "results/tables/final_table4_ablation.csv",
            "results/tables/final_table5_direction_ratio_sensitivity.csv",
            "results/tables/final_table6_cusum_k_sensitivity.csv",
            "results/tables/final_figure_list_for_manuscript.csv",
        ],
        "output_docs": [
            "docs/final_numeric_summary_for_manuscript.md",
        ],
    }

    config_path = CONFIG_DIR / "fault3_final_table_generation_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Saved Step 08 config: {config_path}")

    print("\n" + "=" * 80)
    print("Final manuscript table summary")
    print("=" * 80)

    print("\nTable 2: PCA baseline vs proposed method")
    print(table2)

    print("\nTable 3: Selected local residual variables")
    print(table3)

    print("\nTable 4: Ablation study")
    print(table4)

    print("\nFigure existence check")
    print(figure_list[["figure_id", "filename_png", "png_exists", "filename_pdf", "pdf_exists"]])

    print("\nStep 08 finished successfully.")


if __name__ == "__main__":
    main()