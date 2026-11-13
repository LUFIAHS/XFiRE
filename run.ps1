# XFiRE v3.3 - Windows PowerShell helper (CPU only; CUDA is never assumed).
# Usage (from the project folder):  powershell -ExecutionPolicy Bypass -File .\run.ps1 [setup|test|build|evaluate|serve]
param([string]$Step = "build")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "$PSScriptRoot\src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "6"            # TUF A15 (Ryzen 7 4800H = 8 cores): leave headroom for the OS; lower to 4 if the fan is loud
$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }
switch ($Step) {
  "setup"    { python -m venv .venv; & .\.venv\Scripts\python.exe -m pip install --upgrade pip; & .\.venv\Scripts\python.exe -m pip install -r requirements.txt }
  "test"     { $env:PYTHONPATH = "$PSScriptRoot\src;$PSScriptRoot\tests"
               foreach ($t in "test_ingest","test_diagnostics","test_eval","test_xgbharm","test_forecast","test_forecast_selection","test_pipeline","test_serve") { Write-Host "== $t"; & $py "tests\$t.py"; if ($LASTEXITCODE) { throw "$t failed" } } }
  "build"    { & $py -m xfire --modis (Get-ChildItem data\modis\*.csv).FullName --viirs (Get-ChildItem data\viirs\*.csv).FullName --out output --global --no-hotspots --threads 6 }
  "evaluate" { & $py -m xfire --out output --evaluate-harmonization; & $py -m xfire --out output --evaluate-forecast --forecast-tests 5 --hyperopt-evals 30 --threads 6 }
  "serve"    { & $py -m xfire --out output --serve --port 8000 }
  default    { throw "unknown step '$Step' (setup|test|build|evaluate|serve)" }
}
