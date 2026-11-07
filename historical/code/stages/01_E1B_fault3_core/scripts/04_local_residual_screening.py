from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_DIR = PROJECT_ROOT / "configs"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Main settings
# ======================================================================
FAULT_START_SAMPLE = 161
PCA_CUMVAR_TARGET = 0.95

# First criterion: direction consistency
DIRECTION_RATIO_THRESHOLD = 0.95

# Second criterion: among direction-stable residuals, select top variables
# with the largest absolute mean paired residual shift.
N_SELECTED_VARIABLES = 5


def load_variable_columns():
    variable_path = TABLE_DIR / "02_variable_list.csv"
    if not variable_path.exists():
        raise FileNotFoundError(
            f"Variable list not found: {variable_path}. "
            "Please run scripts/02_prepare_fault3_data.py first."
        )

    variable_info = pd.read_csv(variable_path)
    return variable_info["variable"].tolist()


def load_data():
    files = {
        "normal_train": PROCESSED_DIR / "normal_train.pkl",
        "normal_validation": PROCESSED_DIR / "normal_validation_runs_1_250.pkl",
        "fault3_development": PROCESSED_DIR / "fault3_development_runs_1_250.pkl",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Processed file not found: {path}. "
                "Please run scripts/02_prepare_fault3_data.py first."
            )

    print("Loading processed data...")
    data = {name: pd.read_pickle(path) for name, path in files.items()}

    for name, df in data.items():
        print(f"{name}: shape={df.shape}")

    return data


def fit_pca_model(df_train, variable_cols):
    print("\n" + "=" * 80)
    print("Fitting PCA model on normal training data")
    print("=" * 80)

    X_train = df_train[variable_cols].to_numpy(dtype=float)

    mean_ = X_train.mean(axis=0)
    std_ = X_train.std(axis=0, ddof=1)
    std_[std_ == 0] = 1.0

    X_train_std = (X_train - mean_) / std_

    pca_full = PCA()
    pca_full.fit(X_train_std)

    explained_ratio = pca_full.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    n_pcs = int(np.searchsorted(cumulative_ratio, PCA_CUMVAR_TARGET) + 1)
    cum_var = float(cumulative_ratio[n_pcs - 1])

    model = {
        "mean": mean_,
        "std": std_,
        "components": pca_full.components_[:n_pcs, :],
        "eigenvalues": pca_full.explained_variance_[:n_pcs],
        "n_pcs": n_pcs,
        "cum_var": cum_var,
    }

    print(f"Selected PCs: {n_pcs}")
    print(f"Cumulative explained variance: {cum_var:.6f}")

    return model


def compute_variable_residuals(df, variable_cols, model):
    """
    Compute variable-wise PCA reconstruction residuals in standardized space.

    The residual is calculated as:
    standardized data - PCA reconstruction using retained PCs.

    We do not only calculate the global Q statistic here.
    Instead, we keep each variable's reconstruction residual separately.
    """
    X = df[variable_cols].to_numpy(dtype=float)
    X_std = (X - model["mean"]) / model["std"]

    P = model["components"]
    scores = X_std @ P.T
    X_recon = scores @ P

    residual = X_std - X_recon

    residual_cols = [f"res_{v}" for v in variable_cols]

    id_df = df[["faultNumber", "simulationRun", "sample"]].reset_index(drop=True)
    residual_values_df = pd.DataFrame(
        residual,
        columns=residual_cols,
        index=id_df.index,
    )

    residual_df = pd.concat([id_df, residual_values_df], axis=1)

    return residual_df, residual_cols


def screen_direction_stable_residuals(normal_res, fault_res, variable_cols, residual_cols):
    """
    Pair Fault 3 residuals and normal residuals by simulationRun and sample.

    For each residual variable:
    1. Compute paired residual shift:
       delta = Fault 3 residual - normal residual.
    2. Calculate direction consistency:
       positive_ratio, negative_ratio, direction_ratio.
    3. First keep variables with direction_ratio >= 0.95.
    4. Among these direction-stable variables, select the top variables with
       the largest absolute mean paired residual shift.

    This avoids selecting many variables that have stable but extremely tiny shifts.
    """
    print("\n" + "=" * 80)
    print("Screening direction-stable local PCA residual variables")
    print("=" * 80)

    key_cols = ["simulationRun", "sample"]

    normal_post = (
        normal_res[normal_res["sample"] >= FAULT_START_SAMPLE]
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    fault_post = (
        fault_res[fault_res["sample"] >= FAULT_START_SAMPLE]
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    if normal_post[key_cols].equals(fault_post[key_cols]) is False:
        raise ValueError(
            "Normal validation data and Fault 3 development data are not exactly paired "
            "by simulationRun and sample."
        )

    normal_matrix = normal_post[residual_cols].to_numpy(dtype=float)
    fault_matrix = fault_post[residual_cols].to_numpy(dtype=float)

    delta_matrix = fault_matrix - normal_matrix

    rows = []

    for j, var in enumerate(variable_cols):
        delta = delta_matrix[:, j]
        normal_residual = normal_matrix[:, j]

        mean_delta = float(np.mean(delta))
        std_delta = float(np.std(delta, ddof=1))
        normal_std = float(np.std(normal_residual, ddof=1))

        positive_ratio = float(np.mean(delta > 0))
        negative_ratio = float(np.mean(delta < 0))
        direction_ratio = max(positive_ratio, negative_ratio)

        if mean_delta >= 0:
            direction = "positive"
            sign_for_positive_evidence = 1
        else:
            direction = "negative"
            sign_for_positive_evidence = -1

        effect_size = mean_delta / normal_std if normal_std > 0 else np.nan

        rows.append(
            {
                "variable": var,
                "residual_variable": f"res_{var}",
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "normal_residual_std": normal_std,
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "direction_ratio": direction_ratio,
                "dominant_direction": direction,
                "sign_for_positive_evidence": sign_for_positive_evidence,
                "effect_size": effect_size,
                "abs_effect_size": abs(effect_size),
                "abs_mean_delta": abs(mean_delta),
            }
        )

    metrics = pd.DataFrame(rows)

    # Full ranking table: for inspection and supplementary analysis.
    metrics = metrics.sort_values(
        by=["direction_ratio", "abs_mean_delta", "abs_effect_size"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # First filter: direction-stable variables.
    stable = metrics[
        metrics["direction_ratio"] >= DIRECTION_RATIO_THRESHOLD
    ].copy()

    # Second filter: compact local residual subset.
    # Among direction-stable variables, select the top N by absolute mean shift.
    selected = stable.sort_values(
        by=["abs_mean_delta", "direction_ratio", "abs_effect_size"],
        ascending=[False, False, False],
    ).head(N_SELECTED_VARIABLES).reset_index(drop=True)

    # Save all results.
    metrics_path = TABLE_DIR / "04_fault3_residual_direction_metrics.csv"
    stable_path = TABLE_DIR / "04_fault3_direction_stable_residuals.csv"
    selected_path = TABLE_DIR / "04_fault3_auto_selected_signed_residuals.csv"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    stable.to_csv(stable_path, index=False, encoding="utf-8-sig")
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")

    print(f"Saved all residual direction metrics: {metrics_path}")
    print(f"Saved direction-stable residual variables: {stable_path}")
    print(f"Saved selected residual variables: {selected_path}")

    print("\nDirection-stable variable count:")
    print(len(stable))

    print("\nFinal selected variables:")
    if len(selected) == 0:
        print("No variables selected. Please check the threshold.")
    else:
        print(
            selected[
                [
                    "variable",
                    "dominant_direction",
                    "mean_delta",
                    "abs_mean_delta",
                    "direction_ratio",
                    "effect_size",
                    "sign_for_positive_evidence",
                ]
            ]
        )

    return metrics, stable, selected


def plot_selected_signed_residual_shifts(normal_res, fault_res, selected):
    """
    Plot direction-unified paired residual shifts for selected variables.

    For each selected variable:
    signed_delta = sign * (fault residual - normal residual)

    After this direction unification, Fault 3-related shifts should move in
    the positive direction.
    """
    if len(selected) == 0:
        print("No selected variables. Skip plotting.")
        return None

    key_cols = ["simulationRun", "sample"]

    normal_sorted = normal_res.sort_values(key_cols).reset_index(drop=True)
    fault_sorted = fault_res.sort_values(key_cols).reset_index(drop=True)

    if normal_sorted[key_cols].equals(fault_sorted[key_cols]) is False:
        raise ValueError(
            "Normal validation data and Fault 3 development data are not exactly paired "
            "by simulationRun and sample."
        )

    plot_df = normal_sorted[key_cols].copy()

    for _, row in selected.iterrows():
        var = row["variable"]
        res_col = row["residual_variable"]
        sign = int(row["sign_for_positive_evidence"])

        signed_delta = sign * (
            fault_sorted[res_col].to_numpy(dtype=float)
            - normal_sorted[res_col].to_numpy(dtype=float)
        )

        plot_df[var] = signed_delta

    sample_mean = plot_df.groupby("sample")[selected["variable"].tolist()].mean()
    sample_mean = sample_mean.reset_index()

    sample_path = TABLE_DIR / "04_selected_signed_residual_shift_by_sample.csv"
    sample_mean.to_csv(sample_path, index=False, encoding="utf-8-sig")
    print(f"Saved sample-wise signed shifts: {sample_path}")

    plt.figure(figsize=(10, 5.5))

    for var in selected["variable"].tolist():
        plt.plot(sample_mean["sample"], sample_mean[var], label=var)

    plt.axvline(FAULT_START_SAMPLE, linestyle=":", label="Fault start")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Sample")
    plt.ylabel("Direction-unified paired residual shift")
    plt.title("Fault 3 local signed PCA residual shifts")
    plt.legend(ncol=2)
    plt.tight_layout()

    fig_png = FIGURE_DIR / "fig2_fault3_local_signed_residual_shifts.png"
    fig_pdf = FIGURE_DIR / "fig2_fault3_local_signed_residual_shifts.pdf"

    plt.savefig(fig_png, dpi=300)
    plt.savefig(fig_pdf)
    plt.close()

    print(f"Saved figure: {fig_png}")
    print(f"Saved figure: {fig_pdf}")

    return fig_png


def main():
    print("=" * 80)
    print("Step 04: Local signed PCA residual screening for Fault 3")
    print("=" * 80)

    variable_cols = load_variable_columns()
    data = load_data()

    # 1. Fit the same PCA model using normal training data only.
    model = fit_pca_model(data["normal_train"], variable_cols)

    # 2. Compute variable-wise PCA residuals.
    print("\nComputing variable-wise PCA residuals...")
    normal_res, residual_cols = compute_variable_residuals(
        data["normal_validation"], variable_cols, model
    )

    fault3_res, _ = compute_variable_residuals(
        data["fault3_development"], variable_cols, model
    )

    # 3. Screen direction-stable residual variables on development data only.
    metrics, stable, selected = screen_direction_stable_residuals(
        normal_res, fault3_res, variable_cols, residual_cols
    )

    # 4. Plot selected signed residual shifts.
    fig_path = plot_selected_signed_residual_shifts(
        normal_res, fault3_res, selected
    )

    # 5. Save compact config.
    config = {
        "script": "scripts/04_local_residual_screening.py",
        "fault_id": 3,
        "fault_start_sample": FAULT_START_SAMPLE,
        "pca_cumvar_target": PCA_CUMVAR_TARGET,
        "selected_n_pcs": model["n_pcs"],
        "cumulative_explained_variance": model["cum_var"],
        "direction_ratio_threshold": DIRECTION_RATIO_THRESHOLD,
        "n_direction_stable_variables": int(len(stable)),
        "n_selected_variables_target": N_SELECTED_VARIABLES,
        "n_selected_variables_final": int(len(selected)),
        "selection_rule": (
            "First keep residual variables with direction_ratio >= "
            f"{DIRECTION_RATIO_THRESHOLD}; then rank these variables by abs_mean_delta "
            f"and select the top {N_SELECTED_VARIABLES} variables."
        ),
        "screening_data": {
            "normal": "normal_validation_runs_1_250.pkl",
            "fault3": "fault3_development_runs_1_250.pkl",
            "pairing": "simulationRun + sample",
            "screening_samples": f">= {FAULT_START_SAMPLE}",
        },
        "selected_variables": selected["variable"].tolist(),
        "selected_residual_variables": selected["residual_variable"].tolist(),
        "selected_directions": selected["dominant_direction"].tolist(),
        "selected_signs": selected["sign_for_positive_evidence"].astype(int).tolist(),
        "figure": str(fig_path) if fig_path is not None else None,
        "output_tables": {
            "all_metrics": "results/tables/04_fault3_residual_direction_metrics.csv",
            "direction_stable": "results/tables/04_fault3_direction_stable_residuals.csv",
            "selected": "results/tables/04_fault3_auto_selected_signed_residuals.csv",
            "sample_mean_shift": "results/tables/04_selected_signed_residual_shift_by_sample.csv",
        },
    }

    config_path = CONFIG_DIR / "fault3_local_residual_screening_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Saved Step 04 config: {config_path}")

    print("\n" + "=" * 80)
    print("Key local residual screening results")
    print("=" * 80)
    print(f"Selected PCs: {model['n_pcs']}")
    print(f"Cumulative explained variance: {model['cum_var']:.6f}")
    print(f"Direction ratio threshold: {DIRECTION_RATIO_THRESHOLD}")
    print(f"Direction-stable variable count: {len(stable)}")
    print(f"Final selected variable count: {len(selected)}")
    print(f"Selection rule: direction-stable first, then top {N_SELECTED_VARIABLES} by abs_mean_delta")

    if len(selected) > 0:
        for _, row in selected.iterrows():
            print(
                f"{row['variable']}: direction={row['dominant_direction']}, "
                f"direction_ratio={row['direction_ratio']:.4f}, "
                f"mean_delta={row['mean_delta']:.6f}, "
                f"abs_mean_delta={row['abs_mean_delta']:.6f}, "
                f"effect_size={row['effect_size']:.6f}, "
                f"sign={int(row['sign_for_positive_evidence'])}"
            )

    print("\nStep 04 finished successfully.")


if __name__ == "__main__":
    main()