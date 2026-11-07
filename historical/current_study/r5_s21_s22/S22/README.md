# Supplementary Data S22 — fixed-alarm D2 endpoint sensitivity

This historical component includes the saved fault-event records, endpoint protocol, complete endpoint tables and numerical implementation for the post-outcome endpoint analysis. The fixed score-path archive is distributed separately as `TEP_P1_Historical_Fault09_Score_Paths_S30_20261026.zip`; its original and current identities are recorded in `../../EXTERNAL_INPUTS.csv`.

At the fixed 40-hour horizon, the mean Signed point-permutation interaction is -189.4978, +36.1022 and +487.3022 minutes for failure penalties of 2,400, 4,800 and 9,600 minutes. The underlying alarms are unchanged. These are descriptive means over ten overlapping splits. Table 12 and Figure S2 report this analysis.

## Reaggregate the saved events

From `current_study/r5_s21_s22/`, with NumPy available:

```sh
python -B S22/code/analyze_d2_endpoints.py --original S22/inputs/original_D2 --replay S22/inputs/replayed_D2 --protocol S22/protocols/EXPERIMENT_2_D2_ENDPOINT_PROTOCOL_v1.0.json --out NEW_S22_ENDPOINT_RESULTS
```

Use a new output-directory name. The supplied original and replayed event files support this saved-event calculation. The original D2 implementation is retained as a scientific source asset. Its external fixed-score input can be identified with `tools/run_fixed_diagnostic.py --diagnostic D2 --source PATH_TO_S30_ZIP` from the repository root. This supplement does not contain the original 52-variable trajectories or claim independent raw-to-score replication.
