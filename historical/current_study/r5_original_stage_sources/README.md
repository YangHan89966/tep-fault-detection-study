# Historical R5 generation, learning and scoring source

The four stages describe a fixed scientific sequence:

| Stage | Role |
|---|---|
| A | Generate five groups of development/calibration trajectories, 4,575 in total |
| B | Learn, select, calibrate and fix 135 configurations |
| C | Generate one shared 1,000-trajectory test panel after the model identities are fixed |
| D | Evaluate the fixed configurations and summarize saved outcomes |

PCA and the residual basis remain fixed across all five downstream learning replicates. Each fault has 250 shared test trajectories; five fitted-state evaluations do not multiply that physical sample size.

These are scientific source records and protocol definitions. They are not a newly validated portable deployment. Repeating generation or learning requires matching external inputs, the documented numerical environment and explicit configuration of execution paths. Historical creation-time status fields describe their recorded stage and are not evidence that the later study was incomplete.

Allocation, completion and model identities are retained in `../r5_raw_identity/`. For recalculation using saved outcomes, start with `../r5_s21_s22/S21/` and the [root reproduction guide](../../REPRODUCTION_GUIDE.md). File integrity checks, saved-result recalculation and raw-to-score reproduction are different checks.
