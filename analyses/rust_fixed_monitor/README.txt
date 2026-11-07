Evidence for an independently implemented recording driver (TEP-Rust)
4 November 2026 — one fixed historical Fault 9 detector; no new training

What ran: the official TEP Studio WebAssembly binary, not a translation by the
authors of this validation. No change to the simulator binary or JavaScript glue.
It is a different implementation of the driver and controller from the Python
tep package. Both descend from the Braatz code and this Rust port comes from
the same research group as the Python tep package. This does not independently
validate a different plant, Ricker controller, or industrial historian. The
within-trajectory comparison tests measurement pairing, not a change in control.

Physical runs: 59 calibration + 60 normal evaluation + 75 Fault 9 evaluation.
Independent assignment of seeds by SHA-256 of role and ordinal, with distinct
32-bit odd values. Roles are disjoint; the simulator uses its historical scalar
generator, so overlaps between underlying streams are possible. This is a
post hoc panel, without external preregistration. No seeds or runs were removed
or replaced. All 194 runs reached the full 960-record horizon and returned status
completed. With trip=0 the simulator continues to emit after a plant trip,
but retains the tripped status. A separate off-panel open-loop positive
control tripped at 3.06 h: trip=0 emitted all 120 scheduled records over 6 h
and still returned tripped; trip=1 returned tripped and stopped at 61 records. A separate
pilot only tested executability; two native replay checks repeat existing test
trajectories and add no samples to the panel. limits.json was written after
calibration and before the evaluation batch. The published original lower bound
or performance of other monitors was not used to set this limit.

Outcome: on the same 75 Fault 9 trajectories the actual controller-input view
detected 70 cleanly within 40 h versus 21 with the standard Rust output. Of
these, 49 were controller-input only and none native only. Both triggered in
2 of the 60 normal evaluation runs; the same observed counts at separate limits
are not proof of equal population false-alarm probabilities. Using the specified delay origin (the first affected record, 161), clean
detections by 8 h were 22 vs 0. Measured from the actual injection time,
the counts by 8 h were 21 vs 0. Counts at 2 h (0 vs 0) and over the
40-h follow-up (70 vs 21) were unchanged. Mean penalized losses using
the specified record-based delay: 980.36 vs 2170.24 min. Normal score
variances: 0.995 vs 2.387 (pooled). The measurement–valve level-loop
correlations over the 60 normal runs had medians 1.000 and 1.000 for the input
view, 0.024 and 0.025 for the native view (12–7, 15–8). See summary.json for
ranges and all outcomes. CI in these files means controller input.

Timing: step numbers are one-based. A controller cycle at step k reads meas[k-1].
The standard Rust Sample returns meas[k], produced by the next model evaluation,
alongside the same current commands. The difference is one model step = 1 s
when Euler dt=1/3600 h. This is not the 3-s offset of tep. Sample.hours labels
step k at (k-1) seconds within floating-point accumulation. We use driver steps
180,360,...,172800 for 960 records. Fault 9 is scheduled at 8 h and first
appears in the 3-min recorded fault flags at record 161. A one-second
ground-truth replay located activation at driver step 28801, t=28800 s.
The first affected record (161) is stamped 28979 s; record 320 is at
57599 s and record 321 at 57779 s. The latter is exactly eight hours
after record 161 but 179 s beyond the literal injection+8 h. One CI run
first alarms on record 321. CLOCK_TRIP_AUDIT.json and summary.json
report both conventions. New measurements belong
to the first integration stage; do not infer that a full second of physical
valve response to the just-issued command was observed. The standard Rust
control loop, timing and all noise are untouched. Both views contain identical
valve columns, taken from the standard record. Off-grid controller actions
continue normally. The nominal native recorder (sample_every=180) reproduced
the high-frequency observer's native record exactly: native_replay.json.

Files: protocol.json specifies the plan, seeds and roles; limits.json supplies
the two locked thresholds. calibration.csv and evaluation.csv give per-run
results. RUN_MANIFEST.csv binds every 960x93 observation matrix and native run
metadata. The 93 columns are 41 controller inputs, 41 native measurements and
11 shared XMV. They are little-endian float64. Paired raw bytes are archived separately in RUST_PAIRED_RECORDS.zip,
under runs/. The compact evidence package includes the executable, scripts,
scores, raw digests, public evidence and pinned Rust source code. The raw ZIP
is optional for re-analysis; the exact output is reproducible without it.

Source and reproducibility:
  * https://github.com/jkitchin/tep-rust/tree/0735ad3b5bb7fdd367a8a31da1db8a209dac2a15
    crates/tepsim/src/sim.rs (step/control/observation), crates/tepsim-control/
    src/lib.rs (3-step fast schedule), crates/tepsim-core/src/rng.rs (float).
  * https://kitchingroup.cheme.cmu.edu/tep-rust/deltas.html (D-010, D-011).
  * https://kitchingroup.cheme.cmu.edu/tep-rust/studio/ (live WASM). The WASM
    reproduced the source project's native one-hour checksum c8a26889992f1719.
    The deployed app.js matched the source in the retrieved Git tree bytewise.
    These checks do not attest the deployed binary's exact source commit.
  * https://github.com/jkitchin/tennessee-eastman-profbraatz/tree/9a6c8e5fcef4a2850778704e7793c87b0a187005
    src/tep/simulator.py: control uses existing measurement at ~312, step()
    advances, and simulate() lines 415,421–425 then read measurements and XMV.
    README/docs specify recording interval; present public issue search
    (is:issue, all statuses) returned no results. Intent is unverified.
  * https://www.mdpi.com/2227-9717/14/16/2569 .
    Koçak et al., Methods §2.1, explicitly identify the Rieth dataset.
  * https://doi.org/10.26434/chemrxiv.10001628/v1 Lyu et al., preprint;
    https://github.com/KitchinHUB/tep-manuscript/blob/main/notebooks/03-generate-new-tep-dataset.ipynb
    uses from tep import TEPSimulator and returns result.measurements and
    result.manipulated_vars[:, :11]. IDV 3,9,15 excluded; no pairing experiment.
    Source versions were not established as identical to ours. Distinguish
    this new external validation set from their Rieth-based training/benchmark.
  * https://doi.org/10.1016/j.softx.2022.101053 pyTEP calls MATLAB Engine.
    https://github.com/kennyuren/COSTEP specifies MATLAB R2024b+, Simulink
    and the Signal Processing Toolbox. Neither engine ran here.
  * https://depts.washington.edu/control/LARRY/TE/download.html distinguishes
    the 2015 revised Simulink model and alternative decentralized controllers.

Reproduce in a clean unpacked directory with Node (ES modules), Python, NumPy.
The protocol.json and historical map are already included. Commands, in order:
  python run_role.py cal 4
  OPENBLAS_NUM_THREADS=1 python score.py cal
  python run_role.py normal 4
  python run_role.py fault 4
  node verify_clock_trip.mjs
  OPENBLAS_NUM_THREADS=1 python score.py evaluation
  node verify_native.mjs
On Windows set OPENBLAS_NUM_THREADS=1 in the environment instead. No network
or Matlab is required for the packaged simulator. This is the portable Rust
check; it is separate from any retraining program or private validation data.
