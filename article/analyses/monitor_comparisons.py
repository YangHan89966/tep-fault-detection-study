"""Recompute the article table rows supported by the model-comparison experiments."""
import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = "monitor_comparisons/"
VIEWS = ("CI", "PU")


def read_table(archive, path):
    compression = "gzip" if path.endswith(".gz") else None
    with archive.open(ROOT + path) as source:
        return pd.read_csv(source, compression=compression, float_precision="round_trip")


def percent(flags):
    return round(100 * np.mean(flags), 9)


def first_alarm_row(monitor, fault, panel, faults, normals, eight_end):
    row = {"monitor": monitor, "fault": fault, "panel": panel}
    for view, ff, nn in zip(VIEWS, faults, normals):
        assert ff.size and nn.size
        row[f"clean_40h_{view}_pct"] = percent((ff >= 160) & (ff <= 959))
        row[f"clean_8h_{view}_pct"] = percent((ff >= 160) & (ff <= eight_end))
        row[f"normal_48h_{view}_pct"] = percent(nn >= 0)
    return row


def table_frame(d, role, recording, first):
    paired = d.pivot(index=[c for c in ("rep", "case_id", "run_id") if c in d.columns],
                     columns=recording, values=first)
    if paired.isna().any().any():
        raise ValueError(f"Missing recording for {role}")
    return [paired[view].to_numpy() for view in VIEWS]


def from_long(monitor, fault, panel, d, role, rec, first, eight_end=319):
    fault_label = next(r for r in d[role].unique() if r.startswith("fault"))
    normal_label = next(r for r in d[role].unique() if r.startswith("normal"))
    f = table_frame(d[d[role] == fault_label], fault_label, rec, first)
    n = table_frame(d[d[role] == normal_label], normal_label, rec, first)
    return first_alarm_row(monitor, fault, panel, f, n, eight_end)


def matched_row(monitor, d):
    row = {"monitor": monitor, "fault": 9, "panel": "Five fits, shared independent streams"}
    for view in VIEWS:
        total = np.zeros(3, dtype=int)
        for _, g in d[d.view == view].groupby("rep"):
            nn = g[g.role == "normal"]
            ff = g[g.role == "fault"]
            assert len(nn) == 400 and len(ff) == 200
            peaks = np.maximum(nn.pre_max, nn.post40_max)
            limit = np.sort(peaks)[379]
            assert (peaks > limit).sum() == 20
            total += [(ff.pre_max.le(limit) & ff.post40_max.gt(limit)).sum(),
                      (ff.pre_max.le(limit) & ff.post8_max.gt(limit)).sum(),
                      (peaks > limit).sum()]
        for label, val, denom in zip(("clean_40h", "clean_8h", "normal_48h"), total, (1000, 1000, 2000)):
            row[f"{label}_{view}_pct"] = 100 * val / denom
    return row


def sfa(archive, fault, source):
    arrays = []
    for rolefault in (fault, 0):
        path = ROOT + f"representation/sfa/results/{source}_eval_F{rolefault}.npz"
        with np.load(BytesIO(archive.read(path)), allow_pickle=False) as panel:
            # Axes after the recording are dimension split, statistic and chart.
            first = panel["first"][:, :, :, 0, 2, 1]
            assert first.shape == (200 if rolefault else 400, 5, 2)
            arrays.append([first[:, :, i].ravel() for i in range(2)])
    return first_alarm_row("SFA, dominant slow-feature dynamic CUSUM", fault, source, *arrays, 319)


def reproduce(archive):
    rows = []
    four = read_table(archive, "four_metric_search/stats/run_scores.csv.gz")
    for arm, label in (("residual_original78_start0", "Four-metric residual direction search"),
                       ("full_original78_start0", "Four-metric full-space direction search")):
        part = four[four.arm == arm]
        rows.append(from_long(label, 9, "D1", part, "role", "view", "first", 320))
        rows.append(matched_row(label + ", matched 5% FAR", part))
    rows.append(sfa(archive, 9, "D1"))
    xgb = read_table(archive, "representation/xgb/results/D1/summary/per_run.csv.gz").query("trees == 500 and k == 0.5")
    rows.append(from_long("Window-statistics XGBoost", 9, "D1", xgb, "role", "recording", "first_alarm"))
    transfer = read_table(archive, "transfer/summary/run_scores.csv.gz").query("panel == 'D2' and policy == 'D2_rank190'")
    selected = read_table(archive, "transfer/s12/first_alarms.csv").query("panel == 'D2' and threshold_policy == 'D2_rank190'")
    for fault in (3, 15):
        m = "full_mean_signed" if fault == 3 else "full_moment_squared"
        subset = selected.query("fault_id==@fault and method==@m")
        rows.append(from_long("Four-metric residual direction search", fault, "D2", subset,
                              "endpoint_group", "recording", "first_alarm", 320))
        for chart, label in (("Linear_residual", "Identity-metric residual direction"),
                             ("Linear_full", "Identity-metric full-space direction"),
                             ("XGB_500_k0.5", "Window-statistics XGBoost")):
            rows.append(from_long(label, fault, "D2", transfer.query("fault==@fault and chart==@chart"),
                                  "role", "view", "first"))
        rows.append(sfa(archive, fault, "D2"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Measurement_Pairing_Additional_Experiments.zip")
    parser.add_argument("--output", type=Path, help="Write the recomputed percentages as CSV.")
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        result = reproduce(archive)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    else:
        print(result.to_string(index=False))
