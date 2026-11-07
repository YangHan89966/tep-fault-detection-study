# Historical direction experiments

This compact component contains the original S28 scientific code, study definitions, fixed PCA and residual basis, normal-design states, selection and calibration locks, 8,400 selected-system result rows, and eight saved summary tables. The retrospective study covers 20 faults, ten development splits and three representation variants.

Run `python verify_summary.py` from this directory with NumPy and pandas installed. It calls the unchanged historical summarization function and compares four summaries reconstructed from `MAIN_RESULTS.csv` with the supplied tables. The other four tables summarize candidate-level information and are retained as saved results.

All candidate model states, event records and role-separated reference arrays remain in the companion archive `TEP_P1_Historical_Direction_Experiments_S28_20261026.zip`. The code in `code/` supports that complete archive. Its full integrity check is `python code/verify_release.py --archive PATH_TO_COMPLETE_ARCHIVE`. Candidate-level checks and complete refitting require that companion and the original raw inputs identified in `INPUT_ACCESS.json`. This repository does not assert that the companion is currently available online.

Scientific source files are copied unchanged. The `verify_summary.py` wrapper and this README describe this selected distribution. The source archive supplied no standalone license; use follows the applicable project and upstream terms.
