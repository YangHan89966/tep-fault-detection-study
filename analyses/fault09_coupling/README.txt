FAULT 9 SUPPLEMENTARY ANALYSES
Distribution version 1.1.0 | Prepared 30 October 2026

This archive provides two complementary sensitivity analyses of Tennessee Eastman
Fault 9 monitoring. The recording-mode by B-estimator analysis evaluates whether the
large recording contrast persists after changing B. The graded-noise analysis
examines partial sharing of measurement noise while retaining a fixed historical
monitor. Both reuse previously analyzed trajectories.

CONTENTS
recording_B_cross/  Protocol, models, thresholds, saved scores, per-run events,
                   paired comparisons and independent verification scripts.
graded_coupling/   Five noise-sharing fractions and actual post-update reference,
                   799 per-run score files, protocol, thresholds, bootstrap indices,
                   construction and verification code, and publication figures.
release_docs/      Release notes and archive-level numerical identity checks.

From the extraction directory, run python verify_distribution.py to verify the
release file identities. Read the README in each module for numerical verification
commands and dependencies.
Saved-result verification is available from this archive. Reconstructing measurement
noise in the coupling module additionally uses the normal-calibration, normal-
evaluation and Fault-9 raw-data ZIPs from 26 October 2026 in the same collection.
The B-estimator study's exact 745 source trajectories are identified by its input
manifest; they belong to its historical development and recording-panel design.
These 745 raw trajectories are external dependencies; the archived scores support
output verification, while full relearning additionally requires the original
workspaces. They are distinct from the independent-stream panel used for coupling.

DESIGNS
Recording by B: 306 development runs, 199 normal outer-calibration runs,
120 normal evaluation runs and 120 Fault-9 evaluation runs. Two recording modes,
two B estimators and two direction pools give eight monitored outputs per run.
Graded coupling: 199 normal calibration runs, 400 normal evaluation runs and
200 Fault-9 evaluation runs. A fixed historical monitor is evaluated at shared-noise
fractions 0, 0.25, 0.50, 0.75 and 1, with actual post-update recording as a separate
sixth reference. The shared fraction is constant within a run.

The B experiment relearns the direction pools within each recording pipeline;
the graded experiment retains the historical numerical representation. Their model
states, denominators and outcomes should therefore be interpreted within each design.
Clean detection requires an alarm after fault onset and no pre-fault alarm. Normal
alarms cover the full 48-hour run. All arms report realized normal alarms alongside
detection under the common nominal calibration rule. Difference intervals are
conditional, marginal paired-bootstrap intervals.

PROVENANCE AND RELEASE IDENTITIES
Original scientific protocols and recorded execution identities are preserved.
Public packaging removes machine-specific paths and author-facing working documents.
Module-level derivation records distinguish any public representation of metadata or
portable code from its original hash. Scientific CSV and NPZ/NPY outputs retain their
original bytes. Distribution manifests identify the files delivered here; they are
release checksums, while original frozen hashes remain scientific provenance anchors.
Version 1.1.0 names this distribution and the proposed collection update. Original
protocol version fields and historical study IDs retain their original meanings.

The analyses are post-hoc sensitivity studies on previously analyzed panels. The
graded panel uses distinct keyed random streams; reusing it here adds no independent
confirmation sample. The historical B panel retains its documented stream-dependence
limitations. Sample sizes count physical trajectories, not recording views or arms.

ATTRIBUTION AND LICENSING
Original component copyright and license notices remain applicable. The simulator
notices are in graded_coupling/engine/vendor/LICENSE and README.md. A third-party
license applies to its identified component, not automatically to this whole archive.
This distribution introduces no new license grant for author-created material.
Use the record's applicable licensing information together with component notices.
AI tools assisted with code, analysis protocols, figures and text preparation.

CITATION
Cite the specific Zenodo record version that contains this archive as distribution
version 1.1.0. The record's citation panel supplies its assigned version DOI and
creator metadata. No new DOI is embedded in these files before Zenodo assigns it.
