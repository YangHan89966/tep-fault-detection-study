# Historical monitoring code and results

This directory contains original scientific code, fitted states, calibration records, saved alarms and numerical summaries used in the Tennessee Eastman monitoring studies. Relative paths inside each scientific component are preserved.

| Directory | Contents |
|---|---|
| `code/` | Historical PCA, residual screening, CUSUM, ablation and robustness implementations |
| `current_study/dense_validation/` | Dense-validation models, alarm outcomes and replay code |
| `current_study/k_sensitivity/` | CUSUM reference-value sensitivity and recorded outcomes |
| `current_study/f15_risktime/` | Fixed Fault-15 score paths, calibration and risk/time calculations |
| `current_study/r5_original_stage_sources/` | Scientific sources for development generation, learning, shared-test generation and evaluation |
| `current_study/r5_s21_s22/` | Five fitted-state results and fixed-event endpoint sensitivity |
| `current_study/reproduction/` | Historical diagnostic code and saved numerical inputs |
| `numerical_evidence/` | Models, calibration arrays, scores and outcomes from four historical studies |
| `fault09_score_paths/` | Fault-9 scalar scores, fixed states and CUSUM replay |
| `direction_experiments/` | Compact S28 code, common inputs, locks and saved summaries |
| `audited_assets/` | Numerical source tables and rendering code retained from earlier analyses |

Start with [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md). Scientific filenames retain their original study labels; figure and table numbers in historical files refer to those analyses. The submitted documents and current tables are in `../paper/`.

The original assessment, dense validation, repeated learning and later independent-stream confirmation are distinct evidence sets. Repeated fitted states share their stated test populations. Their model counts are not counts of independent physical trajectories.

Full process matrices and large raw-trajectory archives are separate inputs. Availability and precise input identities are documented in [INPUT_AVAILABILITY.md](INPUT_AVAILABILITY.md). Original source files are byte-identical to their source archives; local documentation and the S28 summary wrapper are identified separately in `SOURCE_INVENTORY.json`. See the preserved [license scope](LICENSE_SCOPE.md) and component notices.
