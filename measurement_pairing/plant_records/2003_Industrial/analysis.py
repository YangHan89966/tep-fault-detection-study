#!/usr/bin/env python3
"""Reproduce an exploratory timing-pairing analysis of a public level loop.

Usage:
    python analysis.py --data data/other-L-paper-horch-2003.csv
    python analysis.py --zip /path/to/DataRepositories.zip

Only the documented normal-operation pulp-and-paper level record is analysed.
Positive ell pairs an OP increment at t with a PV/error feature from t + ell.
Each shifted model is fitted independently on the training section. A second,
explicitly labelled analysis freezes the native-pairing coefficients and scale.
No control configuration, causal logging error, or fault-detection result is
inferred. This is exploratory work following an earlier inspection of the data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import zipfile

import numpy as np
import pandas as pd

MEMBER = "Other/other-L-paper-horch-2003/other-L-paper-horch-2003.csv"
EXPECTED_SHA256 = "76da7b972818f9116afc27d31c1afb44196d2f66325a7c5aba27be5a8047b2c0"
LAGS = (-2, -1, 0, 1, 2)
TARGETS = np.arange(3, 901, dtype=int)
GAP = 8


def metrics(observed, predicted):
    """Residual means observed minus predicted; SD/covariance use ddof=1."""
    observed = np.asarray(observed, float)
    residual = observed - np.asarray(predicted, float)
    denom = np.sum((observed - observed.mean()) ** 2)
    assert np.all(np.isfinite(residual)) and denom > 0
    return {
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1 - np.sum(residual ** 2) / denom),
        "residual_mean": float(residual.mean()),
        "residual_sd": float(residual.std(ddof=1)),
        "residual_variance": float(residual.var(ddof=1)),
        "residual_second_moment": float(np.mean(residual ** 2)),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def fit(X, y):
    coef, _, rank, singular = np.linalg.lstsq(X, y, rcond=None)
    assert rank == X.shape[1]
    assert np.all(np.isfinite(coef))
    assert np.allclose(X.T @ (y - X @ coef), 0, atol=1e-8)
    return coef, float(singular[0] / singular[-1])


def split_indices(fraction):
    n_train = int(round((len(TARGETS) - GAP) * fraction))
    train = TARGETS[:n_train]
    omitted = TARGETS[n_train:n_train + GAP]
    test = TARGETS[n_train + GAP:]
    # All dependent OP indices and all possible lagged PV indices are disjoint.
    assert train[-1] + max(LAGS) < test[0] + min(LAGS) - 1
    return train, omitted, test


def load_bytes(args):
    if args.zip:
        with zipfile.ZipFile(args.zip) as archive:
            return archive.read(MEMBER)
    source = Path(args.data) if args.data else Path(__file__).parent / "data" / "other-L-paper-horch-2003.csv"
    return source.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inp = parser.add_mutually_exclusive_group()
    inp.add_argument("--data", help="Original semicolon-delimited CSV")
    inp.add_argument("--zip", help="Original complete SACAC ZIP archive")
    parser.add_argument("--out", "--output", dest="output", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    original = load_bytes(args)
    observed_sha = hashlib.sha256(original).hexdigest()
    assert observed_sha == EXPECTED_SHA256, f"Source bytes changed: {observed_sha}"
    df = pd.read_csv(io.BytesIO(original), sep=";")
    assert df.columns.tolist() == ["Time", "SP", "PV", "OP"]
    assert df.shape == (904, 4), f"Unexpected data shape: {df.shape}"
    assert np.all(np.isfinite(df.to_numpy(float)))
    assert np.array_equal(df.Time.to_numpy(), np.arange(904) * 2)
    assert np.array_equal(df.SP.to_numpy(), np.full(904, 50))

    op = df.OP.to_numpy(float)
    pv = df.PV.to_numpy(float)
    sp = df.SP.to_numpy(float)
    dy = np.r_[np.nan, np.diff(op)]
    dpv = np.r_[np.nan, np.diff(pv)]
    error = sp - pv

    def design(t, ell=0):
        j = t + ell
        return np.column_stack((np.ones(len(t)), dpv[j], error[j]))

    def features(t, ell=0):
        j = t + ell
        return np.column_stack((dy[t], dpv[j], error[j]))

    regression_rows, split_rows, predictions, acf_rows = [], [], [], []
    models = {}
    for fraction, split_name in [(0.5, "primary_50"), (0.4, "sensitivity_40"),
                                 (0.6, "sensitivity_60")]:
        train, omitted, test = split_indices(fraction)
        split_rows.append({
            "split": split_name, "train_fraction_of_nonpurged_targets": fraction,
            "train_n": len(train), "test_n": len(test), "purged_n": len(omitted),
            "train_start_index": int(train[0]), "train_end_index": int(train[-1]),
            "purge_start_index": int(omitted[0]), "purge_end_index": int(omitted[-1]),
            "test_start_index": int(test[0]), "test_end_index": int(test[-1]),
        })
        for ell in LAGS:
            Xtr, Xte = design(train, ell), design(test, ell)
            coef, cond = fit(Xtr, dy[train])
            pred = Xte @ coef
            resid = dy[test] - pred
            models[(split_name, ell)] = coef
            row = {
                "split": split_name, "model": "increment_PI_form",
                "target": "delta_OP", "ell_samples": ell, "offset_seconds": 2 * ell,
                "train_n": len(train), "test_n": len(test),
                "intercept_a": coef[0], "coefficient_b_delta_PV": coef[1],
                "coefficient_c_error": coef[2], "training_design_condition_number": cond,
                **metrics(dy[test], pred),
            }
            regression_rows.append(row)
            if split_name == "primary_50":
                for k, t in enumerate(test):
                    predictions.append({
                        "target_index_zero_based": int(t), "time_seconds": float(df.Time.iloc[t]),
                        "ell_samples": ell, "offset_seconds": 2 * ell,
                        "observed_delta_OP": dy[t], "predicted_delta_OP_refit": pred[k],
                        "residual_refit": resid[k],
                    })
                # Sample ACF with the common lag-zero sum-of-squares denominator.
                centered = resid - resid.mean()
                den = np.dot(centered, centered)
                for k in range(1, 21):
                    acf_rows.append({"ell_samples": ell, "acf_lag_samples": k,
                                     "acf": float(np.dot(centered[:-k], centered[k:]) / den)})

        # Baselines are native-pairing models. Static P predicts OP, whereas the
        # increment models predict delta_OP; their RMSEs are not interchangeable.
        for baseline in ["static_P_form", "increment_P_form", "zero_increment"]:
            if baseline == "static_P_form":
                Xtr = np.column_stack((np.ones(len(train)), pv[train]))
                Xte = np.column_stack((np.ones(len(test)), pv[test]))
                coef, cond = fit(Xtr, op[train]); target = "OP"
                observed = op[test]; pred = Xte @ coef
                a, b, c = coef[0], coef[1], np.nan
            elif baseline == "increment_P_form":
                Xtr, Xte = design(train)[:, :2], design(test)[:, :2]
                coef, cond = fit(Xtr, dy[train]); target = "delta_OP"
                observed = dy[test]; pred = Xte @ coef
                a, b, c = coef[0], coef[1], 0.0
            else:
                a, b, c, cond = 0.0, 0.0, 0.0, np.nan
                target = "delta_OP"; observed = dy[test]; pred = np.zeros(len(test))
            regression_rows.append({
                "split": split_name, "model": baseline, "target": target,
                "ell_samples": 0, "offset_seconds": 0,
                "train_n": len(train), "test_n": len(test),
                "intercept_a": a,
                "coefficient_b_delta_PV": b if baseline != "static_P_form" else np.nan,
                "coefficient_b_PV_static": b if baseline == "static_P_form" else np.nan,
                "coefficient_c_error": c, "training_design_condition_number": cond,
                **metrics(observed, pred),
            })

    train, omitted, test = split_indices(0.5)
    native_coef = models[("primary_50", 0)]
    native_features = features(train)
    means = native_features.mean(axis=0)
    scales = native_features.std(axis=0, ddof=1)
    assert np.all(scales > 0)
    covariance_rows, eigen_rows, frozen_rows = [], [], []
    for ell in LAGS:
        Xtest = features(test, ell)
        Ztest = (Xtest - means) / scales
        # np.cov re-centres the TEST data. This is covariance, not Z.T @ Z / n.
        cov = np.cov(Ztest, rowvar=False, ddof=1)
        eigval, eigvec = np.linalg.eigh(cov)
        assert eigval[0] > 0 and np.all(np.isfinite(eigval))
        for i, eig in enumerate(eigval):
            eigen_rows.append({"ell_samples": ell, "offset_seconds": 2*ell,
                               "rank_ascending": i+1, "eigenvalue": eig})
        pred = design(test, ell) @ native_coef
        m = metrics(dy[test], pred)
        # Independent covariance identity for the frozen affine relation.
        raw_weight = np.array([1.0, -native_coef[1], -native_coef[2]])
        standard_weight = scales * raw_weight
        covariance_implied_variance = float(standard_weight @ cov @ standard_weight)
        assert np.isclose(covariance_implied_variance, m["residual_variance"], rtol=1e-9, atol=1e-12)
        assert np.isclose(m["residual_second_moment"],
                          (len(test)-1)/len(test) * m["residual_variance"] + m["residual_mean"]**2,
                          rtol=1e-10, atol=1e-12)
        frozen_rows.append({"ell_samples": ell, "offset_seconds": 2*ell,
                            "coefficient_source": "native_primary_training_only", **m})
        covariance_rows.append({
            "ell_samples": ell, "offset_seconds": 2 * ell, "test_n": len(test),
            "lambda_min": eigval[0], "lambda_mid": eigval[1], "lambda_max": eigval[2],
            "condition_number_covariance": eigval[-1] / eigval[0],
            "lambda_min_fraction_of_trace": eigval[0] / eigval.sum(),
            "fixed_native_relation_variance_OP_squared": m["residual_variance"],
            "fixed_native_relation_second_moment_OP_squared": m["residual_second_moment"],
        })
        for row in predictions:
            if row["ell_samples"] == ell:
                k = row["target_index_zero_based"] - int(test[0])
                row["predicted_delta_OP_frozen"] = pred[k]
                row["residual_frozen"] = dy[test[k]] - pred[k]

    quantization = []
    for col in ["SP", "PV", "OP"]:
        v = df[col].to_numpy(float)
        diffs = np.diff(np.unique(v))
        quantization.append({
            "column": col, "unique_values": int(len(np.unique(v))),
            "min": float(v.min()), "max": float(v.max()),
            "smallest_positive_distinct_value_difference": float(diffs.min()) if len(diffs) else None,
            "max_abs_deviation_from_0p001_grid": float(np.max(np.abs(v * 1000 - np.round(v * 1000)))),
            "note": "Formatting/grid diagnostic only; this does not establish sensor resolution.",
        })

    regressions = pd.DataFrame(regression_rows)
    primary = regressions[(regressions.split == "primary_50") &
                          (regressions.model == "increment_PI_form")]
    native_rmse = float(primary.loc[primary.ell_samples == 0, "rmse"].iloc[0])
    comparisons = []
    for _, row in primary.iterrows():
        comparisons.append({"ell_samples": int(row.ell_samples),
                            "offset_seconds": int(row.offset_seconds),
                            "rmse_refit": row.rmse, "r2_refit": row.r2,
                            "rmse_ratio_to_native": row.rmse/native_rmse})
    summary = {
        "analysis_status": "exploratory_follow_up_after_prior_data_screening",
        "source_file": "other-L-paper-horch-2003.csv",
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "actual_rows": len(df), "description_txt_rows": 903,
        "time_start_seconds": 0, "time_end_seconds": 1806,
        "sampling_interval_seconds": 2, "constant_SP": 50,
        "source_row_count_note": "CSV contains 904 observations; accompanying TXT says 903.",
        "split_primary": split_rows[0],
        "offset_definition": "Positive ell uses PV and error at t+ell with delta_OP at t.",
        "model": "delta_OP[t] = a + b*(PV[t+ell]-PV[t+ell-1]) + c*(SP[t+ell]-PV[t+ell]) + residual[t]",
        "native_training_coefficients": {"a": float(native_coef[0]), "b": float(native_coef[1]), "c": float(native_coef[2])},
        "native_test_metrics": {k: v for k, v in primary[primary.ell_samples == 0].iloc[0].to_dict().items()
                                if not isinstance(v, float) or np.isfinite(v)},
        "primary_comparisons": comparisons,
        "common_standardization": {"source": "native_primary_training_only",
            "feature_order": ["delta_OP", "delta_PV", "SP_minus_PV"],
            "mean": means.tolist(), "sample_sd": scales.tolist()},
        "covariance_definition": "Sample covariance of test feature vectors standardized by native training SD; re-centred on their own test means; ddof=1.",
        "frozen_relation_definition": "All offsets use native training coefficients a,b,c. Variance and uncentred second moment are reported separately.",
        "interpretation_limits": [
            "Controller configuration and scan/history timing are absent from the available metadata.",
            "PI-form coefficients are empirical estimates, not verified controller gains.",
            "A deliberately shifted pairing is a sensitivity intervention, not evidence of an original historian error.",
            "The sample is one approximately 30-minute normal-operation record; no population-level generalization or fault-detection claim is evaluated.",
            "Positive offsets use future PV for an offline pairing diagnostic, not a deployable online predictor.",
            "Observed coefficients, splits, and shifts were not preregistered before first inspection of this record.",
        ],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    regressions.to_csv(out / "regression_metrics.csv", index=False, float_format="%.14g")
    pd.DataFrame(split_rows).to_csv(out / "split_definitions.csv", index=False)
    pd.DataFrame(predictions).to_csv(out / "test_predictions.csv", index=False, float_format="%.14g")
    pd.DataFrame(acf_rows).to_csv(out / "residual_acf.csv", index=False, float_format="%.14g")
    pd.DataFrame(covariance_rows).to_csv(out / "covariance_metrics.csv", index=False, float_format="%.14g")
    pd.DataFrame(eigen_rows).to_csv(out / "covariance_eigenvalues.csv", index=False, float_format="%.14g")
    pd.DataFrame(frozen_rows).to_csv(out / "frozen_native_relation_metrics.csv", index=False, float_format="%.14g")
    pd.DataFrame(quantization).to_csv(out / "data_grid_diagnostics.csv", index=False, float_format="%.14g")
    pd.DataFrame(comparisons).to_csv(out / "primary_comparisons.csv", index=False, float_format="%.14g")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "source_sha256": summary["source_sha256"],
                      "primary_comparisons": comparisons,
                      "native_training_coefficients": summary["native_training_coefficients"]}, indent=2))


if __name__ == "__main__":
    main()
