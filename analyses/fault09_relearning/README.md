# Recording-specific Fault 9 relearning

This directory contains the frozen simulation and learning programs and the completed five-replicate study. `study_complete/` contains the trained models, calibration arrays, run identities and 12,000 evaluation rows. The compact source archive supplies the data used by the replay program.

## Study design

Each training replicate uses 384 new physical trajectories, recorded both ways with identical valve positions. The two monitors are fitted separately under the same rules. Each uses its own recording of 78 normal runs for PCA. There are 1,920 development runs across five replicates, 199 separate normal runs for calibration, and a common evaluation panel of 400 normal and 200 Fault 9 runs. The random keys for these 2,719 trajectories are disjoint from those of the earlier 1,199-run panel.

The primary contrast uses the second-moment direction and compares the controller-input and post-update 40 h clean-detection rates. The analysis pairs recordings within a physical run and gives each fitted replicate equal weight. The protocol specifies a positive lower 95% confidence limit for the detection difference. To report an increase within the allowed false-alarm margin, the upper 95% limit for the normal-alarm difference must also be at most +5 percentage points. The intervals use 20,000 whole-run paired bootstrap draws conditional on these five fits and the common calibration panel. The protocol and trajectory keys were recorded before simulation. The original operating points use independent calibration. Leave-one-fit-out and matched 5% false-alarm results are sensitivity analyses.

## Replay and simulation

From this directory, replay the saved results with Python and NumPy:

```sh
python replay_relearning.py F9_PAIRED_RELEARNING_COMPLETE.zip
```

The replay checks 2,719 physical-run identities, the model and threshold locks, all 14 paired contrasts and the 20,000-draw paired bootstrap. It reads saved evaluation endpoints. The original raw trajectories are archived separately.

For a new simulation, use Python 3.12 with NumPy 2.3.5 and a new output directory:

```sh
python runtime/launch.py --work NEW_SIMULATION_OUTPUT --workers 8
```

The launcher checks the simulator on a separate full-length trajectory before the study runs. On Windows, `RUN_FAULT9.cmd` starts the same launcher. On Linux or WSL, use `bash RUN_WSL.sh`. Both accept `--workers`, `--work` and `--status`. If needed, `SETUP_ENV.cmd` on Windows or `bash RUN_WSL.sh --setup-runtime` creates a local environment. Set `F9_PYTHON` to select an existing Python 3.12 interpreter.

Use the same work directory for progress checks and restarts. To pause after a batch, create an empty `STOP_AFTER_BATCH` file in the work directory. Remove it and rerun the launcher to resume. Completed trajectory files are checked on restart. Allow at least 5 GB of disk space for a new simulation. The launcher writes the final `F9_PAIRED_RELEARNING_COMPLETE.zip` in the work directory. Full trajectories remain in the work directory under `raw/` and are archived separately.

The eight starter trajectories supplied with the original running kit are omitted here; the final models and results are in `study_complete/`. Engineering fixtures and the 53-file code lock are retained.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
