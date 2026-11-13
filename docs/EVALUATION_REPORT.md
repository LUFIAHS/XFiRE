# XFiRE v3.3 - final evaluation (synthetic bench only)

Bench: 24 years of weekly synthetic FIRMS-like data (3 regions, heterogeneous ratios, injected bursts), MODIS 2002-, VIIRS 2012-. Command lines are in `docs/WINDOWS.md`.

## Verified improvements (tests pass, behaviour observed)
* Forecast no longer crashes without LightGBM (original: crash; v3.3-fixed: runs, all tests pass).
* Harmonization: decision table = report text. The learned model is **not** auto-selected: on the synthetic run `hist_gbm` had 26.4% held-out error vs 20.5% for the seasonal ratio, 0/6 fold wins, and the report chose `ratio`.
* Reserved test year is untouched: corrupting it changes only the test score (test_eval).
* Forecast selection avoids false skill claims: on 20 series without persistence 19 published climatology (mean skill -0.35%); on persistent series (phi = 0.94) it captured about +9% skill (`test_forecast_selection`).
* Pooled interval coverage on the end-to-end run: 50% -> 0.53, 80% -> 0.80, 90% -> 0.89 (90% bootstrap ranges include nominal).
* `box_series`: identical output, about 7x faster (30k cells x 300 weeks, 10 calls: 0.58 s -> 0.08 s).
* Dashboard renders in headless Chromium at 1400 and 390 px width with no page errors and no horizontal scroll.

## Synthetic-only results (do not transfer)
* Seasonal ratio is the right answer on this bench because the generator's ratio is a seasonal constant. Real data may differ; run `--evaluate-harmonization` on real FIRMS files.
* Forecast skill vs climatology: -4% to -9% for the learned candidate on the end-to-end series (not significant; published forecast = climatology).

## Untested changes
* XGBoost / LightGBM / Hyperopt code paths, CUDA detection, Windows scripts (`run.ps1`), any real FIRMS file.

## Remaining limitations
* Few validation years -> low-power verdicts. Forecast inputs use whole-record ratios. Harmonization uncertainty is not propagated into forecast bands. Detections are not burned area.

## Answers
* Harmonization beats baselines? Not shown. On synthetic data it correctly does not; on real data it is unknown until evaluated.
* Forecast beats climatology? Not shown. The system now says so itself.
* Scientifically defensible? The *procedure* (leak-free folds, reserved test year, paired CIs, simplest-first selection, honest verdicts) is. Results on real data have not been produced here.
* Next: run both `--evaluate-*` commands on real MODIS/VIIRS archives with LightGBM/XGBoost/Hyperopt installed; add more validation years; consider propagating harmonization uncertainty.
