# Method-to-code map

The [historical implementation map](../docs/HISTORICAL_METHOD_TO_CODE_MAP.md) identifies source for the original assessment, historical diagnostics, DV, reference-value sensitivity and Fault-15 threshold analysis. Its study labels retain their historical meaning.

## Repeated downstream learning (R5)

| Stage | Source directory | Scientific role |
|---|---|---|
| A | `r5_original_stage_sources/TEP_v1.8_Relearning_StageA_v1.0/` | Generate five groups of development/calibration trajectories, 4,575 in total |
| B | `r5_original_stage_sources/TEP_v1.8_Relearning_StageB_v1.0/` | Learn, select and calibrate five sets of 27 configurations |
| C | `r5_original_stage_sources/TEP_v1.8_Relearning_StageC_v1.0/` | Generate the shared 1,000-trajectory test panel after model identities are fixed |
| D | `r5_original_stage_sources/TEP_v1.8_Relearning_StageD_v1.0/` | Score fixed model states and summarize outcomes without refitting |

`r5_raw_identity/ALL_135_CONFIGURATIONS.csv` and `r5_raw_identity/ALL_MODELS_LOCK.json` identify the configurations. In `r5_s21_s22/S21/Stage_D_Return/`, `frozen_models/`, `evaluated/` and `summary/` contain fitted states and outcomes; `reproduction_assets/` and `reproduction_reference/` hold PCA/basis assets and protocol records.

The stage sources document the original numerical sequence. They are not a newly validated portable installation. Raw arrays and environment-specific configuration are separate requirements; preserved expected identities must be checked against the actual inputs.

## Saved-outcome and diagnostic calculations

- DV: `dense_validation/tools/verify_saved_results.py` recalculates saved outcomes; `dense_validation/tools/replay_raw.py` requires separately supplied raw data.
- S18: `k_sensitivity/saved_results/reaggregate_s18.py` uses saved alarm arrays; `k_sensitivity/replay_existing_trajectories.py` exposes the existing-trajectory scoring path.
- S20: `f15_risktime/` and `f15_risktime/supplementary_source/` contain path-based threshold/timing inputs and supplementary reaggregation code.
- S21: `r5_s21_s22/S21/Stage_D_Return/reproduction_code/reaggregate_d.py` recalculates summaries from saved outcomes.
- S22: `r5_s21_s22/S22/code/analyze_d2_endpoints.py` recalculates endpoint alternatives from fixed event records; adjacent `protocols/` and `inputs/` supply those records.

See the [root reproduction guide](../REPRODUCTION_GUIDE.md) for commands and distinctions between input levels. Changing the endpoint applied to saved alarms does not create a new population of simulated trajectories.
