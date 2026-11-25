"""Recompute D-feed detection rates, false alarms and recovery from paired first alarms."""
import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


ALARM_FILE = "dfeed_reconstruction_monitoring/results/per_run_first_alarms.csv.gz"
CONDITIONS = ("PU", "RC", "CI")
# Row 160 is the first faulty record. The last row of each window is inclusive.
HORIZONS = {2: 199, 8: 319, 40: 959}
DRAWS = 20_000


def interval(samples):
    return np.percentile(samples[np.isfinite(samples)], [2.5, 97.5])


def summarize(records):
    rates = []
    recovery = []
    # Match the fit and run bootstrap streams in the study's archived analysis.
    fit_sampler = np.random.default_rng(2026112411)
    fit_weights = fit_sampler.multinomial(36, np.full(36, 1 / 36), size=DRAWS)
    fit_weights = fit_weights.astype(float)

    for (fault, role), group in records.groupby(["fault", "role"], sort=True):
        runs = group["physical_id"].unique()
        fits = group["replicate"].unique()
        if len(runs) != 120 or len(fits) != 36:
            raise ValueError(f"Expected 120 paired trajectories and 36 fits for Fault {fault}, {role}")
        # Preserve the archived trajectory and fit ordering for identical bootstrap draws.
        first = {
            arm: group.pivot(index="replicate", columns="physical_id", values=arm)
                      .loc[fits, runs].to_numpy()
            for arm in CONDITIONS
        }
        if any(pd.isna(values).any() for values in first.values()):
            raise ValueError(f"Incomplete fit-by-trajectory panel for Fault {fault}, {role}")

        run_sampler = np.random.default_rng(2026112420 + 2 * fault + (role == "normal"))
        run_weights = run_sampler.multinomial(120, np.full(120, 1 / 120), size=DRAWS)
        horizons = (48,) if role == "normal" else HORIZONS
        for hours in horizons:
            estimates, samples = {}, {}
            for arm, alarm in first.items():
                if role == "normal":
                    event = alarm >= 0
                    endpoint = "FAR48"
                else:
                    event = (alarm >= 160) & (alarm <= HORIZONS[hours])
                    endpoint = f"clean{hours}"
                rate = 100 * event.mean()
                paired_draws = 100 * (np.sum((fit_weights @ event.astype(float)) * run_weights,
                                             axis=1) / event.size)
                lower, upper = interval(paired_draws)
                rates.append(dict(fault=fault, role=role, endpoint=endpoint, arm=arm,
                                  events=int(event.sum()), reconstructions=len(fits),
                                  physical_runs=len(runs), estimate=rate, lower95=lower, upper95=upper))
                estimates[arm], samples[arm] = rate, paired_draws
            if role == "fault":
                gap = estimates["CI"] - estimates["PU"]
                gain = estimates["RC"] - estimates["PU"]
                gaps = samples["CI"] - samples["PU"]
                gains = samples["RC"] - samples["PU"]
                valid = np.abs(gaps) > 1e-12
                lower, upper = interval(100 * gains[valid] / gaps[valid])
                recovery.append(dict(fault=fault, endpoint=endpoint, total_gap_pp=gap,
                                     regained_pp=gain, count_zero_bootstrap_gap=int((~valid).sum()),
                                     estimate=100*gain/gap if gap else np.nan, lower95=lower, upper95=upper))
    return pd.DataFrame(rates), pd.DataFrame(recovery)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Measurement_Pairing_Additional_Experiments.zip")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for rates and recovery")
    args = parser.parse_args()
    with ZipFile(args.archive) as archive, archive.open(ALARM_FILE) as source:
        records = pd.read_csv(source, compression="gzip")
    with threadpool_limits(limits=1, user_api="blas"):
        rates, recovery = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    rates.to_csv(args.output / "dfeed_rates.csv", index=False)
    recovery.to_csv(args.output / "dfeed_recovery.csv", index=False)
