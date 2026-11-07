# Implementation disclosures and limits of the evidence

## Shared normal-reference roles and the earlier protocol discrepancy

The normal-validation wording in the earlier protocol document differs from the detailed JSON specification and mathematical definition. The implemented method uses the same 250 normal-validation trajectories for the normal reference in the unpaired development screen, evidence scaling and run-maximum limit calibration. Historical method-review records concern this exact implemented state; they do not erase the discrepancy or establish deviation-free execution of the earlier wording.

The implemented roles and discrepancy were documented after development and before access to the reserved evaluation data. This timing must not be backdated as preregistration. Screening-reference and calibration samples are not independent. Calibration-set false-alarm rates are not independent held-out estimates and do not provide an unconditional exact 5% guarantee. The separate 250-run normal-evaluation set is used to assess fixed monitors and must not be reused for selection or recalibration.

## Conditional Top-1 frequencies

Top-1 frequencies are global conditional frequencies given the observed main selection gate and the original pooled standard deviation. They do not measure full-pipeline stability under refitted PCA, repeated confidence-interval estimation, repeated BH screening or repeated evidence-family selection. Stored coordinates and frequencies refer to this conditional construction; they must not be relabelled as the result of an alternative stability procedure.

## Simulator and random initialization

The compatible Python TEP implementation maps four recorded input fields to one scalar random-number generator with a fixed initial state. The four fields are not four independent random streams. An initial-state seed label must not be interpreted as selecting different physical initial conditions when the implementation does not do so.

The new-seed assessment uses the pinned compatible simulator and this four-field-to-one-generator adapter. It is not validation on an external physical process or evidence of exact reproduction using four independent random streams. The original seed manifest and adapter ledger identify the relevant inputs; a separate seed bank is not implied by this documentation. The absence of a seed CSV from a particular source component, or a statement that reserved evaluation had not yet run at that stage, does not establish that its seed values had never been disclosed. Calculations requiring those inputs must use their exact recorded files.

## Shutdown and recorded continuation

The primary analysis retains the specified 960-sample horizon and records shutdown and the subsequent continuation interval for each trajectory. Runs, faults and seeds are not removed because shutdown occurred. Continued controller and measurement updates with zero process-state derivatives after shutdown are not ordinary continuing physical operation.

Separate run-level annotations can mask the continuation interval starting at the first shutdown sample and describe the number of actually observed post-onset samples, whether an alarm occurred and the observed alarm delay. These annotations describe the observed pre-shutdown record. They do not replace primary endpoints, estimate counterfactual detection without shutdown or change adverse primary outcomes.

## Uncertainty, denominators and conclusions

The recorded interval calculations use 20,000 whole-run resamples within the fixed 17-fault panel. For each fault, resampled trajectory indices are shared across the four methods; normal false-alarm resampling indices are shared across all methods and local detectors. Intervals condition on the fixed fault panel, trained states and recorded construction. They do not propagate all model-selection or training uncertainty. Individual 95% intervals are not a simultaneous confidence band.

Endpoints are reported separately. They do not define a composite method ranking or a post hoc noninferiority margin. A method/fault combination can have no trajectories satisfying a no-prefault-alarm condition; its conditional quantity must be null with the denominator reported, rather than dropping the fault and redefining a macro result.

When a conditional bootstrap draw is undefined, retain the corresponding index and valid-draw count. The complete fixed-panel interval is unavailable in that case; undefined draws must not be silently discarded to produce an apparently complete interval.

An implementation or execution PASS reports the stated computational check only. It does not establish confirmation of a predicted statistical direction, publication suitability, industrial validation or universal superiority.
