# Rust simulator fixed-monitor comparison

The saved comparison contains 59 calibration, 60 normal-evaluation and 75 fault-evaluation runs. Each physical run has controller-input and native-recording endpoints under the same fixed historical affine score.

Recount the supplied endpoints with Python's standard library:

```sh
python check_rust_results.py
```

The checker verifies the model and protocol identities, calibration thresholds, paired outcomes and the record-321 timing sensitivity. The 40-hour fault table is 21 detected by both views, 49 by controller-input only, 0 by native recording only, and 5 by neither.

`driver.mjs`, the pinned WASM implementation and upstream source archive are included for simulator reproduction. Raw paired time-series records are archived separately; `score.py` uses those records. Component licenses and upstream notices are in `UPSTREAM_LICENSES/`.

`SHA256SUMS.txt` gives paths relative to this directory. Earlier package manifests describe their original source packages. `SOURCE_ARCHIVES.md` records the source archives and the directory mapping.
