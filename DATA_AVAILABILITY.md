# Data availability

The repository supplies source programs, fitted monitoring states, thresholds, run identities, saved events and scores, numerical tables and the V10 scientific documents. Larger included files are restored locally with `python unpack.py`.

The 1,199-run confirmation panel includes a run manifest and saved evaluation events. The Fault 9 relearning study includes 2,719 physical run receipts, fitted models and final per-run results. A receipt identifies a run; it does not contain its measurement matrix. The separate Rust comparison includes the saved endpoints for 194 runs.

Full raw measurement matrices, complete historical candidate-model collections and some original simulator inputs are supplied in separate data archives. Their filenames and available checksums are recorded in the relevant component README and source manifests. The repository's saved-result checks do not require downloading those raw archives.

The associated data deposit has reserved DOI [10.5281/zenodo.22084007](https://doi.org/10.5281/zenodo.22084007). Its publication and current file coverage are managed separately from this repository. Use the file inventories to establish which inputs are present before starting a raw-data reconstruction.

The historical dataset is attributed to Rieth et al., [doi:10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1). Its terms and the upstream simulator notices remain applicable.
