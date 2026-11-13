# v3.2 change notes

## Architecture (final)
Harmonization = **XGBoost** (baselines: regional block ratio, seasonal ratio). Forecast = **LightGBM + Hyperopt**. Removed: YOLOv12, GRU/LSTM/TCN, **U-Net and PyTorch** (`unet.py`, `test_unet.py`, `requirements-deep.txt`, `--install-deep`), the Poisson/NB regression baseline, Random Forest and XGBoost-forecast challengers (not built). CLI flags renamed: `--train-unet/--unet-*` → `--retrain-harmonization/--folds/--test-years/--margin/--xgb-*`; `unet_model_metrics.csv` → `harmonization_model_metrics.csv`. Internally the cell-level estimate path is still keyed `unet` in results.json/JS (`uc_method` says which method it is).
New: `xgbharm.py`, Hyperopt tuning in `forecast.py`, architecture panel in the dashboard.


## Modified (earlier v3.2 work)
`ingest.py` (per-stage filter counts, day/night + acq_time kept, duplicate detection) · `config.py` (nominal VIIRS date only for a warning) · `harmonize.py` (real VIIRS start, missing-week handling, recent-year weighting, ratio bootstrap, uncertainty bounds) · `raster.py` (Poisson/NB baseline, forward-chaining folds, MAE/RMSE/spatial r/CI, auto regions, gap-aware weights) · `forecast.py` (rolling-origin by origin year, candidates, bias correction from earlier folds, 50/80/90 % conformal bands) · `pipeline.py`, `cli.py`, `serve.py`, `events.py` (rule detector only) · `core.js`, `template.html` (see below) · tests · requirements/pyproject/launch.py (YOLO/ultralytics removed).
## New
`harmeval.py` (protocol, selection rule, reports) · `diagnostics.py` · `forecast_eval.py` · `tests/test_diagnostics.py`, `tests/check_dashboard.js`.
## Removed
`yolo.py`, `deep.py`, `tests/test_yolo.py`, `tests/test_deep.py`, `examples/manual_pytorch.py`, `UPGRADE_v3.1.md`; dependency `ultralytics`; CLI `--train-deep/--train-yolo/--no-yolo/--deep-*`; the Detector toggle and the deep-forecast overlay in the UI.
## Bugs / weaknesses found in v3.1 (verified in code)
1. The reserved test year could veto the model choice (violated "never influences selection").
2. Cross-fit was year-blocked with future years in training (not chronological); the actual fold count was not reported.
3. Forecast validation selected rows by TARGET year, so training rows could contain outcomes after the validation origin.
4. Forecast was fitted in log space and back-transformed with expm1 (median, not mean) → a structural underprediction for over-dispersed counts; no bias correction; blend weight chosen on the scoring folds.
5. Intervals were checked only at 90 %, with few calibration years; coverage by horizon/activity level not reported.
6. VIIRS start hardcoded in Python and JS (and `y>=2012` in the yearly chart); missing weeks were zeros.
7. Map used extent-level ratios while the calendar used area-level ratios → totals disagreed for drawn areas (map now uses the calendar's calibration).
8. Day/night, Terra/Aqua identity and FRP were discarded at ingest; duplicates were not checked.
## Follow-up fixes (after first real run)
* Crash in duplicate detection with pandas 3 (read-only array) fixed.
* Duplicate detection now stores hashes on DISK (memory maps), so the 50 M RAM cap no longer applies in builds; warns when input files overlap in time.
* Warning when the reserved test year has far larger error than the folds (typical of duplicated/mixed-product data in that year) and when a year's VIIRS/MODIS ratio is unusual.
* Forecast: equal-weight ensemble candidate; intervals recalibrated on earlier years, coverage reported on the newest held-out year.
* Dashboard: yearly chart axis fixed; day/night, drift and weighting lines, tiny eras and long texts removed (details remain in the reports).

## Not verified here / limitations
* The original zip contained no FIRMS data; all tests here are synthetic. Your uploaded real-data tables showed a doubled 2024 VIIRS year (duplicated/overlapping files) — rebuild with one file per period. XGBoost and Hyperopt were exercised on synthetic data only; on that data XGBoost did NOT beat the block ratio (the selection rule kept the block ratio), which is expected because the synthetic truth is a regional ratio.
* Day/night calibration and Terra/Aqua differences are diagnosed (ratio method) but not applied; the grid and U-Net have no day/night channel.
* No spatial hold-out; no quantile-LightGBM comparison (conformal only); weather predictors not added; cell-level forecast maps intentionally not created.
* Harmonization uncertainty is shown for history only and is never merged into forecast intervals.
* Duplicate detection is capped at 50 M records (reported when reached) and needs `acq_time`.
* Desktop/mobile layout was not re-tested in a real browser (existing responsive CSS kept; page checked in jsdom only).
