# Reproducing the saved results

Run the setup commands in the root README first. The integrity, manuscript-table and confirmation-event checks use only the Python standard library. Numerical replays use NumPy; historical calculations also use SciPy and pandas. Component environment records preserve the original calculation versions.

| Check | Command from repository root | Scope |
| --- | --- | --- |
| Distributed files | `python verify_repository.py` | SHA-256 identity of every included file |
| Manuscript tables | `python verify_paper.py` | 83 Data S1 CSVs; 68 SI tables and 6,110 normalized cells |
| Confirmation events | `python verify_confirmation.py` | 1,199 physical runs; 13,200 event rows across 22 monitoring arms |
| Fault 9 and Rust | `python verify_fault09_extensions.py --replay` | Final relearning outcomes and 14 contrasts; separate 194-run Rust comparison |
| Detection–false-alarm grids | `python analyses/detection_far/review/crosscheck_all_grids.py` | Independent recalculation from saved calibration/evaluation maxima |

## Recording and noise coupling

From `analyses/fault09_coupling/`, check all distributed scientific files:

```sh
python verify_distribution.py
```

From `analyses/fault09_coupling/graded_coupling/`, reproduce the saved-score calculation in a new directory:

```sh
python public_replay.py recalculate --output /path/to/new/graded_replay
```

The graded-coupling replay compares eight output files and 35 arrays, and independently verifies 3,600 evaluation event rows. For the recording-by-B component, use its `README.md` for the independent result and bootstrap checks. Its stored-score checks cover 1,920 event rows and 84 paired-bootstrap results. Repeated floating-point calculations can differ in the last decimal places; the component reports state their numerical tolerances.

## Historical calculations

See [historical/REPRODUCTION_GUIDE.md](historical/REPRODUCTION_GUIDE.md) for the dense-validation, sensitivity, risk-time, R5, fixed-score and direction-summary commands. Use new output directories outside the scientific input directories.

## Raw-trajectory reproduction

The commands above recalculate results from supplied frozen scientific inputs. Regenerating all process trajectories or rebuilding scores from every original measurement matrix additionally requires the raw-data archives listed in the component documentation. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).
