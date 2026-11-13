# Running XFiRE v3.3 on Windows (ASUS TUF A15, CPU only)

1. Install Python 3.11+ from python.org (tick "Add python.exe to PATH"). Open PowerShell in the project folder.
2. `powershell -ExecutionPolicy Bypass -File .\run.ps1 setup`  (creates `.venv`, installs `requirements.txt`; lightgbm, xgboost, hyperopt, scikit-learn).
3. `powershell -ExecutionPolicy Bypass -File .\run.ps1 test`  (all tests use SYNTHETIC data; takes a few minutes).
4. Put ONE FIRMS archive CSV per sensor/period in `data\modis\` and `data\viirs\`, then `.\run.ps1 build` (add `--regions config\regions.json` instead of `--global` on 8 GB RAM; edit the call in `run.ps1`).
5. `.\run.ps1 evaluate` writes `harmonization_evaluation.md` and `forecast_evaluation.md` in `output\`. **Read these before trusting the dashboard.**
6. `.\run.ps1 serve` -> http://127.0.0.1:8000

Notes
* The GPU is never assumed. `--device auto` selects CUDA only if a 2-round XGBoost training on the GPU actually succeeds; otherwise CPU.
* CPU threads: `$env:OMP_NUM_THREADS` / `--threads N`. Hyperopt trials: `--hyperopt-evals 30` (use 10 for a quick run, `--no-hyperopt` to skip).
* If a package is missing the code falls back: no XGBoost -> scikit-learn `hist_gbm` (named as such); no LightGBM -> scikit-learn HistGradientBoosting. The reports state which backend ran.
* Windows-specific code paths (venv layout, launch scripts) were NOT executed during the audit (Linux sandbox).
