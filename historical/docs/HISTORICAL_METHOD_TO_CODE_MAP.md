> Historical documentation of earlier study stages. Use the [reproduction guide](../REPRODUCTION_GUIDE.md) for the supplied scientific components. Historical manuscript labels retain their original numbering.

# Method-to-code map

| Manuscript analysis | Repository path | Stage |
|---|---|---|
| Data intake and Fault 3 preprocessing | `code/stages/00_data_intake_and_preparation` | 01–02 |
| Fault 3 PCA, local residual screening, CUSUM | `code/stages/01_E1B_fault3_core` | 03–05 |
| Fault 3 ablation and sensitivity | `code/stages/02_E1C_fault3_secondary` | 06–08 |
| Pairing-free screening, repeated splits, fair baselines | `code/stages/03_E1D_pairing_free_fairness` | 10–16 |
| Fault 3 companion analysis | `code/stages/04_E1E_fault3_companion` | 17 |
| Faults 9/15 multi-evidence development and validation | `code/stages/05_E1F_faults9_15_multievidence` | 18–20 |
| 17-fault development freeze and final validation | `code/stages/06_E1G_reserved_multifault` | 21, 22-R |
| Post-freeze synthesis and sensitivity | `code/stages/07_E1H_postfreeze_synthesis` | 23, 23B |
| Robustness, bank audit, joint gate, consistency audit | `code/stages/08_E1I_robustness_bank_audit` | 24–27 |
| Family adaptation ablation | `code/stages/09_U1_family_ablation` | U1 |
| Full-pipeline repeated-split stability | `code/stages/10_U2_selection_stability` | U2 |
| Matched robustness | `code/stages/11_U3_matched_robustness` | U3 |
| Dynamic CVA baseline | `code/stages/12_U4_dynamic_cva` | U4 |
