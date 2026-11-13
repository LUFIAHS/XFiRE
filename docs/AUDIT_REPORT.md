# XFiRE v3.3 - technical and scientific audit

Scope: code inspection, a synthetic-data test bench (generated FIRMS-like CSVs), before/after comparison against the untouched v3.3 upload.
**Not available in the audit sandbox:** real FIRMS data, `lightgbm`, `xgboost`, `hyperopt`, a GPU, Windows. Those code paths were read, not run.

## Problems found (prioritised) and status

| # | Sev. | Problem | Status |
|---|---|---|---|
| 1 | Critical | Forecast stage **crashed** whenever LightGBM was absent: the scikit-learn fallback received LightGBM-only parameter names. This also made `test_forecast`, `test_pipeline`, `test_serve` fail on the original. | Fixed (`_HGB_NAMES` translation; tests pass) |
| 2 | Critical | Harmonization report was internally inconsistent: the recommendation rule used one estimator, the "paired comparison" table another (pooled), so text and table could disagree. | Fixed: one estimator for headline, CI and paired delta; printed decision table with every criterion; invariant tested |
| 3 | High | Forecast was always published and badged "6-month forecast" even when it did not beat climatology; coverage judged on one year only. | Fixed: nested, simplest-first model selection (climatology -> ridge persistence -> trees); verdict from paired block bootstrap; pooled coverage with CI; honest dashboard badge |
| 4 | High | Hyperopt inner score could be read as performance; labels claimed "LightGBM + Hyperopt" even when the fallback ran untuned. | Fixed: inner score stored as `inner_validation_mae`; label names the backend that ran and whether tuning happened |
| 5 | High | Learned harmonizer silently unavailable without XGBoost; CUDA chosen from `nvidia-smi` alone. | Fixed: scikit-learn `hist_gbm` stand-in (named as such, same features/offset/protocol); CUDA only after a real test training |
| 6 | High | No spatial hold-out; test-year confirmation compared only with the seasonal ratio. | Fixed: 8-degree block hold-out (cells removed from training and the baseline fit); test must beat the strongest simpler baseline |
| 7 | Medium | Bootstrap CIs, paired deltas and headline errors computed on different row sets. | Fixed (shared helpers in `raster.Evaluator`, new `metrics.py`) |
| 8 | Medium | `box_series` per-cell `np.add.at` loop (slow on global grids). | Fixed (concat + bincount, identical sums, ~7x faster in a benchmark) |
| 9 | Medium | VIIRS platforms dropped by the consistency policy were never compared. | Added `viirs_platforms` diagnostic (S-NPP vs NOAA-20/21) |
| 10 | Medium | Dashboard: forecast and quality shown after secondary material; architecture box hard-coded "XGBoost"/"LightGBM + Hyperopt". | Reordered; architecture moved to an expandable section; labels dynamic |
| 11 | Low | Unused `pillow` dependency. | Removed |

## Data-policy checks (read in code, exercised on synthetic data)
* Missing weeks: global weeks with no kept-satellite detections are flagged and stay NaN (never zero). OK.
* Duplicates: checked only when `acq_time` exists; otherwise a visible warning is raised (it is, in the synthetic run). OK.
* FRP is kept separate from counts; Terra/Aqua and day/night kept apart at ingest; VIIRS reduced to one platform by policy; MODIS trimmed to its consistent period. OK.
* Weeks outside the sensor overlap are not used for fitting ratios. OK.
* Detections are thermal anomalies, not burned area or incidents; wording kept in dashboard and reports.

## Still weak / not verified
* All numbers in this repository come from **synthetic** data. No accuracy claim about real fires is supported yet.
* The validation-year count is small (about 6 forecast origins-years), so the forecast verdict test has low power: "no detectable improvement" is not "proven no improvement".
* The harmonized history that feeds the forecast uses ratios fitted on the whole record (documented limitation: calibration constant, not fire outcomes).
* LightGBM, XGBoost and Hyperopt paths: unit-reviewed only.
* The original summary's real-data figures could not be reproduced (no data).


## Targeted post-audit corrections (2026-11-13)

Only the forecasting module received additional code changes after the audit:

- Hyperopt now reports the best candidate's actual inner-validation MAE separately from the score of the parameters actually selected. It reports the real number of completed trials, failed trials, requested trials, and whether the candidate beat the default by the configured minimum gain. If the search fails or produces no valid candidate, defaults are retained and reported honestly.
- Horizon groups with fewer than 30 earlier-fold validation rows now use seasonal climatology (weight 0) instead of assigning the learned model a weight of 1 without evidence.
- The forecast output now explains that weights are shared within groups (weeks 1-5, 6-13, 14-26). A group weight optimizes pooled error across that group; it does not assert that every individual horizon beats climatology. Thus, a slightly worse week-5 MAE does not by itself prove that a shared 1-5 group weight is a coding error.
- Added focused regression tests for the no-evidence fallback, bounded horizon weights, and the gradient-boosting backend adapter.

**Verification boundary:** syntax checks and the focused regression tests were run in the build environment. Hyperopt is not installed in that environment, and no real FIRMS datasets were supplied with this ZIP. Therefore, the Hyperopt execution path and real-data forecasting accuracy remain unverified here. The user's Windows machine with the project's real FIRMS inputs is the next place to run the full pipeline.
