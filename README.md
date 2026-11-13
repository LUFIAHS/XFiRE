# XFiRE v3.3 — burning-activity calendar from NASA FIRMS (MODIS + VIIRS)

```
MODIS + VIIRS observations
  → data quality & observation correction (matching, duplicate removal, observation coverage, missing weeks kept missing)
  → HARMONIZATION: seasonal ratio | regional block ratio | learned Poisson GBM (XGBoost, or scikit-learn `hist_gbm` if XGBoost is missing)
     the simplest method that is not clearly beaten on held-out years is used; the learned model must earn its place
  → harmonized historical fire activity
  → FORECASTING: climatology | ridge persistence | LightGBM (Hyperopt-tuned when available); nested selection, simplest first
  → 26-week outlook: forecasts, uncertainty intervals, validation results
```
Only two learned models: **XGBoost** (MODIS-equivalent counts of VIIRS fire cells) and **LightGBM tuned by Hyperopt** (26-week forecast). Everything else is a transparent statistical baseline (seasonal ratio, regional block ratio, climatology, seasonal-naive). No PyTorch, no GPU requirement (XGBoost can use an NVIDIA GPU with `--device cuda`).
Harmonized values are **MODIS-equivalent estimates, never satellite measurements**; forecasts are model output, never confirmed fires.

## Install and run
```bash
python launch.py          # creates a venv, installs requirements.txt, builds from data/modis + data/viirs, opens the app (run.sh / run.bat / run.command do the same)
```
Manual: `pip install -r requirements.txt`, then `PYTHONPATH=src python -m xfire ...` (or `pip install -e .`).

## Commands
```bash
# build (FIRMS archive CSVs). Keep ONE file per sensor and period (no overlapping downloads).
python -m xfire --modis data/modis/*.csv --viirs data/viirs/*.csv --out output --global --no-hotspots --serve
python -m xfire --modis ... --viirs ... --out output --regions config/regions.json      # smaller/faster: only your areas (recommended on 8 GB RAM)
# re-run only the harmonization comparison / XGBoost, then forecasts (no re-ingest) — e.g. final settings
python -m xfire --out output --retrain-harmonization --folds 8 --test-years 1 --device cuda
# diagnostics + reports without training;  forecast evaluation vs baselines (writes forecast_evaluation.* and 3 CSVs)
python -m xfire --out output --evaluate-harmonization
python -m xfire --out output --evaluate-forecast --forecast-tests 5 --hyperopt-evals 30
python -m xfire --out output --serve --port 8000
```
Options: `--harmonize auto|ratio|block_ratio|xgboost|learned` (force; recorded as forced), `--spatial-folds N`, `--min-win F`, `--threads N`, `--no-xgboost`, `--xgb-rounds`, `--recent-tau auto|off|<years>`, `--hyperopt-evals N`, `--no-hyperopt`, `--satellite-policy`, `--all-fire-types`.

## Harmonization protocol (what the reports mean)
* Reserved test year(s): newest complete overlap year(s); never used for training, tuning, calibration, recent-year weighting or **model selection**.
* Folds: forward-chaining whole years (train on earlier years only), each needing ≥ 3 earlier years; if fewer fit than requested the report says so.
* Choice rule (held-out folds only): the regional block ratio must beat the seasonal ratio, and **XGBoost must beat both baselines** by the margin (default 2 %) with a paired block-bootstrap 90 % interval below 0 and wins in ≥ 60 % of folds; otherwise the simpler method is kept and the reason is written down.
* XGBoost: Poisson objective on the MODIS count of cell-weeks where VIIRS saw fire, starting from the block-ratio estimate (base margin); features: VIIRS this/previous/next week, 8 neighbours, location, season; no year feature; early stopping on the newest training weeks; seasonal mass calibration on training weeks.
* Missing weeks (a satellite with no detections inside its record) are NaN/skipped, never zero. The VIIRS start is read from the data.
* Diagnostics: annual/seasonal ratio with drift test, Terra vs Aqua, day vs night (ratio only, not applied), recent-year weighting (applied only with forward-chaining evidence), handover test, era statistics, duplicate/overlap and unusual-year warnings.

## Forecast
LightGBM on a climatology + persistence offset (trees learn only what those miss). **Hyperopt (TPE)** chooses the objective (Tweedie, Poisson, log-anomaly, or the v3.1-style trees/ridge blend) and the hyper-parameters, plus an optional shrink toward climatology, using only data that precede the backtest (two chronological inner splits). Backtest: rolling-origin by origin year against climatology, seasonal-naive, `forecast_basic` and the v3.1 model on identical rows. Intervals 50/80/90 %: conformal residuals from earlier years, widened until earlier-year coverage reaches target; coverage is reported on the newest held-out year. Drawn areas in the local server reuse the whole-extent parameters (no re-tuning).

## Outputs (in `--out`)
`dashboard.html`, `results.json`, `quality_report.json`, `diag_counts.csv.gz`, **`harmonization_evaluation.md/json`, `annual_ratio_diagnostics.csv`, `sensor_comparison.csv`, `harmonization_model_metrics.csv`**, **`forecast_evaluation.md/json`, `forecast_horizon_errors.csv`, `forecast_bias_diagnostics.csv`, `forecast_interval_diagnostics.csv`** (after `--evaluate-forecast`), `hotspots/`.

## Tests
```bash
PYTHONPATH=src:tests python tests/test_ingest.py      # also: test_diagnostics, test_eval, test_xgbharm, test_forecast, test_pipeline, test_serve
node tests/check_core.js output/results.json          # calendar/map/CSV consistency, sensor jump, missing weeks, trend
node tests/check_dashboard.js output/dashboard.html   # optional: npm i jsdom; renders the page and clicks every download
```
All tests use **synthetic** data: they verify behaviour and leakage rules, not scientific accuracy. Accuracy claims come only from `--evaluate-*` on your real FIRMS data.

## v3.3 audit (read these first)
* `docs/AUDIT_REPORT.md` — what was wrong, what was fixed, what is still unverified.  `docs/EVALUATION_REPORT.md` — before/after, labelled verified / synthetic-only / untested.  `docs/WINDOWS.md` + `run.ps1` — PowerShell guide (CPU, no CUDA assumed).  `UPGRADE_v3.3.md` — change list.
* A learned model is reported under the name of the backend that really ran (`xgboost` or `hist_gbm`), and the forecast label states whether Hyperopt really tuned it.
* The dashboard shows an honest forecast verdict (beats / not better than climatology) with pooled interval coverage; the "6-month forecast" badge is no longer unconditional.
