Fault 9: observed recording contrast on two distinct paired panels
=================================================================
Replay command (Python + NumPy, no simulator):
python replay_f9.py TEP_P1_F9_Graded_Coupling_20261030.zip TEP_P1_Historical_Numerical_Evidence_20261026.zip OUTPUT_DIR

F9_RUN_MAXIMA.csv has one row per original physical run and recording.
Match by (panel, role, run_id), never across panels. Thresholds are separate
for the two recordings and panels; the signed projection is the same archived
historical score (affine map SHA 7a20407a0fafa0028931de8d73c32b2ceae9585175e59dadd836777a97efcf66).
In the confirmation score archive, rho_1 is the replay-verified actual
controller-input endpoint; actual_post_update is the actual post-update
endpoint. No intermediate coupling setting was analyzed.

Each row records a two-sided signed CUSUM, k=.05, initialized to zero only
at the start, 960 records at 3 min. Indexing the original records from 1:
pre_max uses 1–160; post2_max 161–201; post8_max 161–321 (onset through
480 min inclusive); post40_max 161–960; all48_max 1–960; normal48_max
equals all48_max for either normal role, blank for fault runs. Original_h
is the original independently calibrated limit (190th ascending of 199
normal-calibration maxima). For any limit h, clean detection requires
pre_max <= h AND post-window max > h. Evaluation FAR is the proportion of
normal evaluation runs with normal48_max > h. Each fault run stays in the
denominator even after a pre-fault alarm.

F9_ORIGINAL_POINTS.csv: exact original order statistics and aggregate counts;
all per-run normal alarms and fault pre/2h/8h/40h outcomes were compared
with the separate original confirmation event records or original panel-2
Signed_CUSUM_reference records. The confirmation calibration peaks also
reproduce all 398 cells of the archived original two columns. F9_SOURCES.csv
contains per-member SHA-256 identities, checked against the supplied
manifests, and includes those manifests' own SHA-256.
