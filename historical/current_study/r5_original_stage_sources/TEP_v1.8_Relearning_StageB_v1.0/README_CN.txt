R5 Stage B: downstream learning, selection and calibration

Scientific role
This stage consumes the 4,575 development/calibration trajectories from Stage A. Five independent development groups are processed with the original normal-training PCA held fixed. For each group, downstream centering/whitening, directions, score scaling, candidate selection and limits are estimated according to the fixed rules. Each group yields nine configurations for each of Faults 3, 9 and 15: 27 per group, 135 overall.

Selection is completed before reading the group's 259 outer-calibration trajectories. The recorded settings include k=0.05, 120-record cumulative-sum covariance, ridge=1e-4, 32 anchor trajectories, 59 inner-reference trajectories with rank 57, and 259 outer-calibration trajectories with rank 247. These settings define the historical study and are not a parameter search for this archive.

Source and inputs
Numerical source is in payload/code/; the protocol and input identities are in payload/reference/. Reproduction requires the matching external process arrays and numerical environment. This stage does not generate or evaluate the shared test panel, relearn PCA, or replace failed calculations with new data.

Interpretation
A fixed model state records what will be evaluated; it is not evidence that a scientific prediction succeeded. Fitted model states, selected configurations and later outcomes are retained elsewhere in the historical archive. Consult the archive-level reproduction guide for saved-result recalculation. The original execution environment was Python 3.12.3 and NumPy 2.2.6; portable execution in a new environment is not established by file-integrity checks alone.
