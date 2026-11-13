# v3.3 audit changes
Modified: `src/xfire/{forecast,forecast_eval,harmeval,raster,xgbharm,harmonize,diagnostics,pipeline,cli,template.html}`, `README.md`, `requirements.txt`, `pyproject.toml`, `tests/{test_eval,test_xgbharm}.py`.
Added: `src/xfire/metrics.py`, `tests/test_forecast_selection.py`, `run.ps1`, `docs/{AUDIT_REPORT,EVALUATION_REPORT,WINDOWS}.md`, this file.
Removed: dependency `pillow`.
New CLI: `--spatial-folds`, `--min-win`, `--threads`, `--harmonize learned`.
