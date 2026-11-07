# Paired, recording-specific Fault 9 relearning

Each development replicate has six disjoint roles. For the two recorded versions of the **same** physical trajectories, the code fits two entire pipelines. No data are exchanged between views, except the run identities and the required equality of their eleven valve-output columns. Chart labels M1–M4 and array order:

| Position | Recording | Pool for direction | Chart input |
|---|---|---|---|
| 0, M1 | Controller input (snapshot) | Second moment | Signed |
| 1, M2 | 3 s post-update (native) | Second moment | Signed |
| 2, M3 | Controller input | Mean shift | Signed |
| 3, M4 | Post-update | Mean shift | Signed |

The historical 119/120 versus 38/120 comparison concerns M1 versus M2 on the same 120 Fault 9 evaluation runs. Each monitor's direction and threshold are learned on its own recording. M3/M4 are prespecified comparator arms, not alternatives over which the software selects a winning monitor. This module does not set a new primary endpoint.

## Frozen learning algorithm

1. Fit recording-specific PCA anew on exactly 78 paired normal trajectories per replicate. Standardize all 52 physical variables on the 78 × 960 observations in that view (sample SD, ddof=1); decompose the correlation covariance; retain the smallest number of components explaining at least 95% of total variance. Complete the retained loading matrix to an orthonormal residual basis by QR, canonicalizing complement-column signs. The historical 36/35 retained components are not fixed for the new study.
2. On 40 different normal runs, restrict to the supported residual subspace using the inherited eigenvalue floor, max(1e-10 × trace(C)/r, 1e-14); use the identity when full rank. Estimate symmetric whitening and the covariance B of the 40 first-120-record whitened sums divided by √120. Save/lock the geometry before reading numerical Fault 9 runs. No extra fallback, eigenfloor adjustment, or cross-recording basis alignment.
3. Fit the mean and second-moment candidate pools on the same separate 100 Fault 9 runs. Preserve original first/last post-onset windows and the inherited pre-marker rule: for a constant terminal suffix ≥160 rows, use only available fault records strictly before the marker. Average runs and early/late endpoints equally. There are at most eight candidate directions per pool, with original metric/ridge/normalization/deduplication. Signed two-sided CUSUM is fixed at k=0.05, reset only per run.
4. For each candidate, use 32 normal runs to compute the 90th percentile of run maxima (linear interpolation), 59 separate normal runs for the anchored rank 57 inner limit, and 75 separate Fault 9 runs for selection. Minimize exact summed episode loss in samples within each pool, taking the first candidate on ties. A clean delay is first post-onset crossing minus record 161; pre-alarm or miss costs 800 samples/2,400 min. Both sides receive the same physical runs, sample sizes, tuning space and rules.
5. The outer workflow must lock all five replicates before the shared 199 normal calibration runs. Each of the 20 arms gets its own anchored 190th of 199 maxima. Lock every threshold before scientific evaluation. Do not use this module to explore a different pool, threshold or rank on the held-out data.

## Integration

Python 3, NumPy (rank/quantile API), and the supplied vendor files only. Set numerical library threads to one before Python startup; parallelism should be across independent physical runs, not BLAS-internal thread pools. No simulator is imported.

- `pipeline.fit_replicate(rows, raw_root, model_dir)`: fits and seals one fresh model directory (absent or empty). Every row contains `case_id,rep,role,index,fault_id,key_hex`. Index is 1-based **within that role**. Role names/counts: pca_normal:78, normal_design:40, normal_anchor:32, normal_inner:59, fault_fit:100, fault_select:75. Input file: `raw_root/case_id/RAW.npz`. Optional row field `raw_sha256` enforces a known raw digest. Input-receipt file records all actual raw digests. The caller binds the five physical cohorts' identities, disjointness, source runtime and protocol, and stage ordering.
- `pipeline.raw_pair(raw, fault_id)` returns the original native/post and snapshot/CI arrays as a dict keyed by `native,snapshot`. Checks phase, time, fault flags and XMV equality.
- `pipeline.score_pair(post, ci, model_dir)` returns (960,4) **chart statistics** in the order M1–M4. Do not accrue another CUSUM. `score_batch` accepts two arrays of (n,960,52); it preserves each run's start independently. Chunk batches for memory.
- `pipeline.anchors(model_dir)` → four selected training-normal scale anchors. `pipeline.anchored_limits(peaks, model_dir)` → four final limits from (199,4) maxima, using original anchored floating-point operation order. Strict statistic > limit.
- At first scoring in a process the model lock and all model dependencies are reverified, then cached by absolute model path. Keep each completed model directory immutable.
- A failed fit preserves diagnostic files but never writes a model lock. Do not replace a failed key or silently retry a different design.

The historical PCA in assets is **only a QA reference**. New scientific runs refit PCA. `vendor` contains byte-preserved source functions; SHA-256 and source-archive names are in `assets/SOURCE_BINDING.json`. To repeat the optional checks: `python verify_history.py --archives <directory-with-original-archives> --out check_history.json` and `python verify_synthetic.py --out check_synthetic.json`. Use Python with single-threaded numerical libraries. The historical replay uses original 78 raw pairs plus original archived fit moments, selection masks, calibration scores and evaluation scores. It is not a raw replay of all old 306 development runs. A separate synthetic smoke test exercises the entire new directory-based development pipeline including fresh PCA, rank-deficient CI support and four chart paths. Neither test reads new scientific evaluation trajectories.
