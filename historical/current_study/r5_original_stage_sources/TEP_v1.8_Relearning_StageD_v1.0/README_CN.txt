R5 Stage D: evaluation of fixed model states

Scientific role
This stage scores the shared 1,000-trajectory test panel using five previously fixed sets of 27 configurations. It uses the saved PCA, directions, score scaling, k=0.05 and calibration limits. It performs no trajectory generation, refitting, reselection or recalibration. All 135 configurations and adverse outcomes remain part of the study.

Source and saved outputs
The numerical evaluation and analysis source is in payload/code/; the evaluation specification and expected identities are in payload/. Full saved states and outcomes are in the historical archive's current_study/r5_s21_s22/S21/Stage_D_Return/ directory. Its reproduction_code/reaggregate_d.py entry point recalculates summary statistics from saved alarm records in a new output directory. That calculation is not raw-trajectory rescoring or simulator replication.

Statistical interpretation
The panel contains 250 normal trajectories and 250 trajectories for each fault. Five evaluations do not create 1,250 independent trajectories per fault. For the finite five-state contrasts, the differences for the same physical test trajectory are averaged across fitted states before paired whole-run resampling. The recorded calculations use 20,000 draws and report marginal 95% and three-contrast adjusted 98.3333% intervals. Intervals condition on the fixed PCA and five fitted states; they do not estimate future training-population uncertainty.

Normal run resampling is shared across local monitors. Their individual false-alarm rates are not the risk of an OR-combined alarm bank. The historical execution used Python 3.12.3 and NumPy 2.2.6. New raw scoring requires the matching external arrays and environment; software tests on synthetic inputs are separate from scientific results.
