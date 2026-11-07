# Historical saved-result reproduction

Install NumPy, SciPy and pandas in the selected Python environment. Original version records are retained under `environment/` and within the components. Use new output directories outside the repository. Commands below start from `historical/`.

```sh
python -B verify_repository.py
python -B current_study/dense_validation/tools/verify_saved_results.py --out /path/to/new/dv-output
python -B current_study/k_sensitivity/saved_results/reaggregate_s18.py --output /path/to/new/s18-output
python -B current_study/f15_risktime/recheck_saved.py --report /path/to/new/s20-report.json
python -B current_study/r5_s21_s22/S21/Stage_D_Return/reproduction_code/reaggregate_d.py --return-root current_study/r5_s21_s22/S21/Stage_D_Return --output /path/to/new/s21-output
python -B current_study/r5_s21_s22/S22/code/analyze_d2_endpoints.py --original current_study/r5_s21_s22/S22/inputs/original_D2 --replay current_study/r5_s21_s22/S22/inputs/replayed_D2 --protocol current_study/r5_s21_s22/S22/protocols/EXPERIMENT_2_D2_ENDPOINT_PROTOCOL_v1.0.json --out /path/to/new/s22-output
python -B fault09_score_paths/code/verify_saved_scores.py fault09_score_paths --report /path/to/new/s30-report.json
python -B direction_experiments/verify_summary.py
```

These calculations use supplied scores, alarm outcomes, calibration states or event records. Raw scoring and repeated fitting require the exact external process inputs identified by the component records. The compact S28 component reconstructs four result summaries; its complete candidate-level payload is in the named companion archive.

The S18 command without `--bootstrap` recalculates endpoints and selection summaries. Add `--bootstrap` to reproduce its recorded N paired intervals. S21 recomputes its recorded paired bootstrap summaries. Scientific input hashes remain fixed.
