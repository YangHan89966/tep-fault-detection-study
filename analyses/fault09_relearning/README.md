# Recording-specific Fault 9 relearning

This directory contains the frozen simulation and learning programs and the completed five-replicate study. `study_complete/` contains the trained models, calibration arrays, run identities and 12,000 evaluation rows. The exact compact source archive is retained for the original replay program.

From this directory, replay the saved results with Python and NumPy:

```sh
python replay_relearning.py F9_PAIRED_RELEARNING_COMPLETE.zip
```

The replay checks 2,719 physical-run identities, the model and threshold locks, all 14 paired contrasts and the 20,000-draw paired bootstrap. It reads saved evaluation endpoints. The original raw trajectories are archived separately.

For a new simulation, use Python 3.12 with NumPy 2.3.5 and a new output directory:

```sh
python runtime/launch.py --work NEW_SIMULATION_OUTPUT --workers 8
```

The launcher performs the engine check before running the frozen design. The eight starter trajectories supplied with the original running kit are omitted here; the final models and results are in `study_complete/`. Engineering fixtures and the 53-file code lock are retained.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
