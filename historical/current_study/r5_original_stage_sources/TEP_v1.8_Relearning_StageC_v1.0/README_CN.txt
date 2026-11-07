R5 Stage C: shared test trajectory generation

Scientific role
This stage defines generation of the 1,000 test trajectories allocated before learning: 250 normal trajectories and 250 each for Faults 3, 9 and 15. All five fitted states use this same test panel. It is not five separate panels of 1,000 trajectories.

The stage requires the completed development data and the fixed identities of all 135 configurations. It does not fit models, select parameters, recalibrate limits or automatically evaluate detection performance. Source is in payload/code/; the study protocol, allocation and model references are in payload/reference/.

Reproduction requirements
The historical execution used Python 3.12.3 and NumPy 2.2.6 with one numerical thread per trajectory process and the pinned simulator source. New execution requires matching source and input identities plus explicit path configuration. This archive is not a newly validated portable generation launcher.

Reusing a completed trajectory requires identity verification. Resource-interrupted generation can only retry the original allocated identity and seed; numerical or integrity failures must not be resolved through outcome-based replacement. Generation completion is separate from model evaluation. Synthetic scheduler tests do not constitute physical-process results.

The original shared test arrays are distributed separately. See the archive-level input inventory and reproduction guide.
