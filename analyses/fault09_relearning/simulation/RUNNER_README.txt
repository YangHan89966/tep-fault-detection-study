Pinned simulator and full-horizon engineering validation
========================================================
Source: recovered paper-1 independent-stream confirmation results archive
TEP_P1_Independent_Stream_Complete_Results_20261024_recovered_v2.zip
SHA256 923851b6196f0aa5b79cb977a67d56ded778ac472402aec691ea4b9e1bdadebd.
The 30 adapter/engine files in pinned/ are individually identical to the
original freeze. They include upstream license and identity records. The
Python random adapter draws high32/2^32 from successive NumPy Philox
words; its original full physical/controller/measurement equations, one-second
integration, 3-minute recorder, controller-input snapshot and post-update
record are executed unchanged. All score/training logic is external.

Assumed reference environment: CPython 3.12, NumPy 2.3.5. Set BLAS threads
to one before starting the parent driver. No separate TEP installation,
Fortran compiler, GPU or Internet connection is needed for the simulator.

Engineering check only (from the simulation directory):
  python engineering_check.py
No panel key is generated. It reruns the archived 48-hour QA key, checks
all 11 arrays byte-for-byte and compares random-draw counts, RNG trace,
all-step trace and final physical state digest for 172800 steps. Output is
engineering/PORTABLE_ENGINE_QA.json or the explicit first path argument.
Require PASS on the reference machine; investigate failures, never weaken
the gate or replace a failed identity with a different key.

Scientific execution uses only the parent protocol/manifest scheduler.
The imported function run_row(row, raw_dir, protocol_lock_sha256) writes to
raw_dir/<case_id>/, one compressed RAW.npz, RUN_INFO.json and COMPLETE.json
with SHA-256 bindings. Each worker reopens and verifies the engine sources
and runtime once. Re-running an identical complete row verifies hashes and
skips work. An interrupted row reuses exactly its same key; previous files
are retained under incomplete_attempts. Simultaneous writers are excluded
by an OS-backed lock; the file can remain after the OS lock is released.
The caller must verify the frozen manifest, unique stream keys and stage
gate before invoking a worker. Never call this function to score a panel.

Original QA file size 0.85 MB suggests roughly 2.3 GB of compressed raw
data for 2719 runs (actual size depends on trajectories). Allow additional
space for models, working copies, logs, outputs and incomplete attempts.
