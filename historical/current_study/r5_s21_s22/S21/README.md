# Supplementary Data S21 — five downstream learning fits

This historical component retains all 135 configuration results, all 15 per-fit contrasts, the three finite-five-fit mean contrasts, saved alarm records, frozen states, the original protocol and the saved-result aggregation implementation. Numerical arrays and scientific result tables retain their original bytes. Documentation and locator metadata have been updated; FILES.json describes this distribution, while SOURCE_MEMBERS.json records original source identities.

The five fits share the same 250 test runs for each fault and the same 250 normal runs. They are not 1,250 independent test runs. All configurations and adverse outcomes remain available. Five-fit ranges describe these fitted states, and conditional bootstrap intervals do not include population retraining uncertainty.

## Contents

- Stage_D_Return/summary/: complete endpoint, contrast, detection-curve and per-run summaries, plus the saved paired vectors and bootstrap means.
- Stage_D_Return/evaluated/: the 15 fault-by-fit scored records and their source endpoint/event files.
- Stage_D_Return/frozen_models/ and reproduction_assets/: archived fitted states and the fixed PCA assets.
- Stage_D_Return/reproduction_reference/: original learning protocol, seed manifest, configuration table and model lock.
- Stage_D_Return/reproduction_code/: unchanged source implementation.

## Reaggregate saved alarm outcomes

From `current_study/r5_s21_s22/`, in an environment with NumPy and SciPy:

```sh
python -B S21/Stage_D_Return/reproduction_code/reaggregate_d.py --return-root S21/Stage_D_Return --output NEW_S21_REAGGREGATION
```

Use a new output-directory name. This command starts from saved alarm bits; it does not generate trajectories, refit a monitor, or score raw process inputs. The 5,575 raw R5 trajectories are not duplicated in this supplement. The other archived implementation files document the original scoring pipeline and are not instructions to launch a new experiment.
