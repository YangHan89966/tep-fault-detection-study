R5 Stage A: development and calibration trajectory generation

Scientific role
This stage defines generation of five separate groups of 915 development/calibration trajectories, 4,575 in total. Each group contains normal roles of 40/32/59/259 trajectories and, for each of Faults 3, 9 and 15, 100 fitting and 75 selection trajectories. The shared 1,000-run test panel is allocated separately and is not generated at this stage.

PCA and the residual basis come from the original normal-training assessment and remain fixed. The scientific recording convention retains 960 samples at three-minute intervals over a 48-hour horizon, with 52 process variables, the separately retained twelfth manipulated variable, native times and shutdown diagnostics. Source and allocation definitions are under release/code/, release/assets/ and release/protocol/.

Reproduction requirements
The historical numerical execution used Python 3.12.3, NumPy 2.2.6 and the pinned tep 2.5.0 Python backend. New execution requires the matching environment, source identities and an explicitly configured destination. This archive is not a validated portable generation launcher. Follow the archive-level reproduction guide before running any generation code.

Data identities and limitations
Allocation precedes generation. Existing source identities and seed exclusions must be respected. A resource interruption permits retry of the same allocated trajectory and seed; it does not justify outcome-based replacement. Distinct 32-bit initial states do not prove independent or non-overlapping random substreams. The 4,575 original process arrays are distributed separately from this code archive. Software tests with synthetic inputs are not experimental detection results.
