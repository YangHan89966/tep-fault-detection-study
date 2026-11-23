# Data availability

For the measurement–controller pairing study, cite the combined data and code archive for Family A and Family B. Han, Y. and Zhang, X. (2026), *Data and code for: Measurement–controller pairing in closed-loop records changes the detectability of difficult Tennessee Eastman faults*, Zenodo, version 2.0.0. [Version DOI](https://doi.org/10.5281/zenodo.24453012).

The record carries forward the original Family A simulation files. The additional ZIP, `TEP_Measurement_Pairing_Data_and_Code_v2.0.0.zip`, contains the extended Supplementary Information, the SI table crosswalk, Family B results and protocols, the scripts and numerical inputs for the main figures, and the plant-record analyses. Its `README.md` gives reproduction commands and its `MANIFEST.csv` lists individual files and SHA-256 checksums.

| Materials | Source |
| --- | --- |
| Family A historical simulation archives, raw trajectories and original Data S1 | [Zenodo, version 1.0.0, 7 November 2026](https://doi.org/10.5281/zenodo.22084007) |
| Current SI table numbers, including the study timeline and Family B tables | [Repository table crosswalk](measurement_pairing/tables/SI_table_crosswalk.csv) |
| Family B summary statistics, rank-95 first alarms and protocol summaries | [Family B files](measurement_pairing/family_b/README.md) |
| Six main figures and numerical plotting sources | [Figure guide](measurement_pairing/README.md) |
| Pilot and industrial source attribution, download and analysis | [Plant-record guide](measurement_pairing/plant_records/README.md) |

The archived Family A confirmation panel identifies 1,199 runs and contains their evaluation events. The independent Fault 9 relearning study identifies 2,719 physical runs and supplies fitted models and per-run results. The separate Rust comparison includes the endpoints for 194 runs. Component manifests identify the raw archives needed to start again from measurement matrices.

The historical process data are from Rieth et al. (2017). [Harvard Dataverse](https://doi.org/10.7910/DVN/6C3JR1).
