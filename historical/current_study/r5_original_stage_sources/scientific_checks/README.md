# Historical Stage B model checks

`verify_stage_b_models.py` preserves the independent saved-model checks originally used after Stage B. It checks variance normalization of the axes, the selection among 49 candidates for each of 15 models, the rank-57 inner and rank-247 outer thresholds, and reconstruction of all 135 configurations. It also checks original manifests, source bindings and event chronology. It does not invoke the process simulator or repeat learning.

`STAGE_B_SAVED_MODEL_CHECKS.json` is the unchanged historical result of those checks, including 1,470 recomputed threshold values. It is not a result newly generated for this distribution.

The verifier requires the complete ORIGINAL Stage A release directory, Stage B payload directory, extracted Stage B result directory and Stage B result ZIP. That ZIP has SHA-256 `30c3748e9281404055c2c8d2a1ba6820eb906a090bcfe84fe6e3bd57a03af7fa`. It also requires original manifests and event records bound to those inputs. These complete original input trees are not assembled by this distribution; the cleaned source subsets and the new archive SHA-256 are not interchangeable with them.

Run from this directory with NumPy installed and the four original inputs available:

```sh
python -B verify_stage_b_models.py --return-root ORIGINAL_STAGE_B_RESULTS --stage-b-payload ORIGINAL_STAGE_B_PAYLOAD --stage-a-release ORIGINAL_STAGE_A_RELEASE --original-return-zip ORIGINAL_STAGE_B_ZIP --out NEW_CHECK_REPORT.json
```

The output report must be new and outside all input trees. Syntax and command-line help were checked for this distribution. The full original-input verification was not rerun because its complete input containers were not available. The numerical verification expressions remain unchanged; only transport arguments and the output location were adapted.
