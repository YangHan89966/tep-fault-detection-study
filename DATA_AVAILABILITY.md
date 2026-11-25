# Data and code

Han, Y. and Zhang, X. (2026). *Data and code for “Measurement–controller pairing and fault detection in the Tennessee Eastman process”*. Zenodo, version 1.1.0. [10.5281/zenodo.24519585](https://doi.org/10.5281/zenodo.24519585).

| File in the Zenodo record | Contents |
| --- | --- |
| `Supplementary_Information_and_Source_Tables.zip` | Submission and extended supplementary information, article Tables 1–4, supplementary Tables S1–S42, and figure PDFs with their plotting code and inputs |
| `Measurement_Pairing_Additional_Experiments.zip` | Monitor comparisons including slow feature analysis and XGBoost, first alarms, and monitoring after reconstructing D-feed flow |
| `Fault9_CVA_and_Dfeed_Reconstruction.zip` | Fault 9 canonical variate analysis and the check of D-feed flow reconstruction against recorded measurements |
| `TEP_Measurement_Pairing.zip` | Family B simulation results and protocols, extended tables and source materials for the plant-record analyses |

Figure 5 uses the refitted monitoring results in `dfeed_reconstruction_monitoring/` inside `Measurement_Pairing_Additional_Experiments.zip`. The [supplementary table source index](article/experiment_sources.csv) identifies the files behind Tables S35–S42. Some refits require raw trajectory files listed in the corresponding experiment's input index.

The [7 November 2026 Zenodo release](https://doi.org/10.5281/zenodo.22084007) contains the earlier Family A simulated trajectories and analysis materials. The historical Tennessee Eastman data are from [Rieth et al. (2017)](https://doi.org/10.7910/DVN/6C3JR1). The pilot-plant source is documented in the [dataset attribution](article/figures/data/DATA_ATTRIBUTION.txt). The industrial level-loop source is the [SACAC benchmark](https://sacac.org.za/resources/).
