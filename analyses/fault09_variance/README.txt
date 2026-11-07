TEP P1 Fault 9 recording variance decomposition
Study ID TEP_P1_F9_RECORDING_VARIANCE_DECOMPOSITION_20261103_v1
Date 2026-11-03

PURPOSE
This supplementary analysis explains the difference between normal score
variances 2.200029 at rho=0 and 3.363026 under actual post-update recording.
It reuses all 400 independent-stream normal evaluation trajectories and the
unchanged historical H_D01_moment_A01_signed affine score. It performs no
process integration, model fitting, threshold selection, or fault evaluation.

READ FIRST
REPORT_ZH.txt is the Chinese analysis report.
ANALYSIS_NOTES_EN.txt gives the corresponding scientific interpretation in English.
Historical table identifiers are retained in source files; current Data S1 supplies final numbering.

MAIN FINDINGS
rho=0 variance: 2.2000291801902443
controller-input variance: 1.0003449678721514
post-update variance: 3.3630261611087713
post minus rho=0: 1.162996980918527

All19 analyzer measurements and their event identities are identical between
the two archived recordings across all 400 runs. The11 manipulated variables
are also identical. Replacing the 22 continuous measurements with CI values
therefore gives exactly the full CI endpoint. Replacing XMEAS17 alone gives
variance 2.1730025962985673.

Keeping the rho=0 independent noise while changing the denoised measurements
to post-update values gives variance 3.344702767361623. This first step adds
1.1446735871713787; replacing this common noise with the actual post-update
noise adds 0.01832339374714831. These add exactly to the endpoint gap. The 98.4%
first-step share is conditional on this replacement construction and is not a
general causal fraction. Denoised measurements retain noise effects already
transmitted through the original closed loop.

The direct covariance-block decomposition assigns 1.156856521480261 to the
change in twice the covariance between continuous measurements and manipulated
variables. Negative covariance cancellation weakens, while the continuous
group's marginal score variance is nearly unchanged. This decomposition and
the common-noise decomposition are alternative views; do not add their shares.

CONTENTS
results/ALL41_MEASUREMENT_DIFFERENCE_DECOMPOSITION.csv: all measurement channels,
  original and noise-normalized units, ddof0 as in current Table S1.
results/FLOWS_AND_LEVELS_DIFFERENCE_DECOMPOSITION.csv: ten flows plus two levels.
results/GROUP_SUBSTITUTION_AND_BRIDGE.csv: fixed-score variances and 95% intervals.
results/EXACT_ENDPOINT_BLOCKS.csv: six terms that sum to the endpoint gap.
results/GROUP_ENDPOINT_COVARIANCES.csv: both endpoints for each covariance block.
results/EXACT_TIMING_NOISE_TERMS.csv: exact five-term T/E decomposition.
results/SIGNAL_CHANGE_ALLOCATION.csv: symmetric covariance allocation.
results/PER_RUN_SUFFICIENT_STATISTICS.npz: per-run means and centered crossproducts.
results/BOOTSTRAP_METRICS.npz: 20000 shared whole-run bootstrap weights and estimates.
results/PER_RUN_SCORE_SUMMARY.csv: means and within-run score variances.
results/RECONSTRUCTION_AUDIT.json: 400-run identities, schedule checks and exact replay.
results/INDEPENDENT_AUDIT.json: independent Chan-merge numerical verification.
checkpoints/: 400 per-run sufficient-statistic files and their integrity records.
code/: computation, aggregation and report source.
PROTOCOL.json: scope and analysis definitions, recorded before 400-run computation.
SOURCE_ARCHIVES.json: source filenames, sizes and SHA256 identities.
FILE_MANIFEST.csv: files in this supplemental package and their SHA256 hashes.

UNCERTAINTY
Score variances pool 400 x 960 observations with ddof 1. The20000 paired bootstrap
resamples whole physical runs, including between-run mean differences; it
does not treat time points as independent. The seed is 2026110301. Intervals
are marginal descriptive intervals conditional on the fixed model and the
existing noise realization. This is post hoc analysis of an already analyzed
confirmation panel, not a new independent confirmation sample.

REPRODUCE SUMMARY WITHOUT ORIGINAL RAW TRAJECTORIES
Requires Python 3 and NumPy. Run from an extracted copy of this supplement:

  python code/summarize_decomposition.py --root .
  python code/export_group_covariances.py

The first command reads the included 400 checkpoints and recomputes tables,
bootstrap and SUMMARY.json. It verifies the original computation script hash.
Use a working copy if preserving the distributed output hashes is important.
Word report regeneration additionally requires python-docx and a suitable
CJK font. The report source is code/build_report.py.

RECONSTRUCT FROM ORIGINAL RECORDED DATA
The large source trajectories are not duplicated in this supplemental zip.
Provide the two archives named in SOURCE_ARCHIVES.json and extract the graded
coupling archive. Its root contains code/, assets/, reference/ and scores/.
Then run, substituting your actual local paths:

  python code/compute_decomposition.py --raw-zip PATH/TEP_P1_Normal_Evaluation_Data_v1_20261026.zip --coupling-root PATH/TEP_P1_F9_Graded_Coupling_20261030 --out NEW_OUTPUT --workers 6
  python code/summarize_decomposition.py --root NEW_OUTPUT

When rerunning into a new output directory, also copy this package's code/
directory there before aggregation, so the computation-script hash can be
verified. Start from an empty checkpoint directory if code or source inputs
are changed. The distributed checkpoints belong to the archived script hash.

SCOPE BOUNDARY
This supplement handles the rho=0 versus post-update discrepancy. Fault 15
replacement experiments and detection versus false-alarm curves are separate
analyses and were not performed in this package.
