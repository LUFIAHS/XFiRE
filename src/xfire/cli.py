import argparse
from pathlib import Path
from .pipeline import run
from .harmeval import HCfg


def _xcfg(a):
    try:
        from .xgbharm import XCfg
    except ImportError:
        return None
    return XCfg(rounds=a.xgb_rounds, device=a.device, recent_tau=a.xgb_recent_tau, threads=a.threads)


def main():
    ap = argparse.ArgumentParser(prog="xfire", description="XFiRE burning-activity calendar: MODIS + VIIRS harmonization (XGBoost vs statistical baselines) "
                                 "and LightGBM + Hyperopt six-month forecasting")
    ap.add_argument("--modis", nargs="+"); ap.add_argument("--viirs", nargs="+"); ap.add_argument("--out", default="output")
    ap.add_argument("--regions", default=None, help="JSON of analysis areas / spatial extent to keep")
    ap.add_argument("--cell", type=float, default=0.5, help="grid cell size in degrees (AOI resolution)")
    ap.add_argument("--global", dest="keep_all", action="store_true", help="retain all detections globally so any map-drawn area can be analysed")
    ap.add_argument("--serve", action="store_true", help="serve the app locally (hotspot CSV, forecasts for any drawn area)"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-hotspots", action="store_true", help="do not keep the raw hotspots (smaller output; CSV download then holds gridded counts)")
    q = ap.add_argument_group("Data quality (needs the FIRMS `type` / `satellite` columns; silently skipped when absent)")
    q.add_argument("--all-fire-types", action="store_true", help="keep volcanoes, gas flares/industrial and offshore detections (default: vegetation fires only)")
    q.add_argument("--satellite-policy", default="auto", choices=["auto", "off"], help="auto = one VIIRS satellite, MODIS only once Terra+Aqua both observe; off = keep everything")
    q.add_argument("--viirs-satellite", default=None, help="VIIRS satellite code to keep (default: the longest-running one in the files)")
    q.add_argument("--keep-terra-only-period", action="store_true", help="keep MODIS weeks before Aqua joined (2000-11 to 2002-07)")
    h = ap.add_argument_group("Harmonization (XGBoost vs statistical baselines)")
    h.add_argument("--harmonize", default="auto", choices=["auto", "ratio", "block_ratio", "xgboost", "learned"],
                   help="auto = the method that wins on held-out years (XGBoost only if it clearly beats both baselines); otherwise force one")
    h.add_argument("--no-xgboost", action="store_true", help="skip XGBoost (diagnostics and the statistical baselines still run and are evaluated)")
    h.add_argument("--retrain-harmonization", action="store_true", help="re-run ONLY the harmonization comparison / XGBoost (+ what depends on it) on an existing --out build")
    h.add_argument("--folds", type=int, default=3, help="requested forward-chaining folds (e.g. 8); the number actually possible is limited by the overlap years and reported")
    h.add_argument("--test-years", type=int, default=1, help="newest complete overlap years reserved as a final untouched test set (0 = off)")
    h.add_argument("--margin", type=float, default=0.02, help="relative improvement over the baseline required to replace it")
    h.add_argument("--xgb-rounds", type=int, default=400, help="maximum boosting rounds (early stopping decides)")
    h.add_argument("--xgb-recent-tau", type=float, default=0.0, help="recent-year weighting of XGBoost training weeks: half-life in years (0 = off)")
    h.add_argument("--recent-tau", default="auto", help="recent-year weighting of the seasonal ratio: auto (only if evidence supports it) | off | half-life in years")
    h.add_argument("--device", default="auto", help="XGBoost device: auto | cpu | cuda. auto/cuda only use the GPU after a test training run on it succeeded; otherwise the CPU is used")
    h.add_argument("--spatial-folds", type=int, default=4, help="spatial hold-out folds (8-degree blocks) for the newest cross-fit year; 0 = off (adds ~folds x methods trainings)")
    h.add_argument("--min-win", type=float, default=0.6, help="share of folds a method must win to replace its comparator")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads for XGBoost / LightGBM (0 = library default). On a laptop use cores-2 to keep it responsive")
    f = ap.add_argument_group("Forecast (LightGBM + Hyperopt)")
    f.add_argument("--hyperopt-evals", type=int, default=30, help="Hyperopt trials per area (tuned only on data before the backtest)")
    f.add_argument("--no-hyperopt", action="store_true", help="use the default LightGBM parameters")
    e = ap.add_argument_group("Evaluation")
    e.add_argument("--evaluate-harmonization", action="store_true", help="recompute the diagnostics and rewrite the harmonization reports for an existing --out build (no training)")
    e.add_argument("--evaluate-forecast", action="store_true", help="rolling-origin forecast evaluation vs baselines + interval calibration for an existing --out build")
    e.add_argument("--forecast-tests", type=int, default=6, help="rolling-origin validation years for --evaluate-forecast")
    a = ap.parse_args()
    xcfg = _xcfg(a); hcfg = HCfg(folds=a.folds, test_years=a.test_years, margin=a.margin, min_win=a.min_win, spatial_folds=a.spatial_folds); have = (Path(a.out) / "results.json").exists()
    force = None if a.harmonize == "auto" else a.harmonize
    from . import forecast as _fc; _fc.set_threads(a.threads)
    kw = dict(xgb_cfg=xcfg, use_xgb=not a.no_xgboost, harmonize_method=force, fc_evals=a.hyperopt_evals, tune_forecast=not a.no_hyperopt, keep_hotspots=not a.no_hotspots, hcfg=hcfg, recent_tau=a.recent_tau,
              veg_only=not a.all_fire_types, sat_policy=a.satellite_policy, viirs_satellite=a.viirs_satellite, trim_modis=not a.keep_terra_only_period)
    if a.retrain_harmonization or a.evaluate_harmonization or a.evaluate_forecast:
        if not have: ap.error("--retrain-harmonization / --evaluate-* need an existing build in --out")
        from . import pipeline
        if a.retrain_harmonization:
            from .xgbharm import backend_name
            if backend_name() != "xgboost": print(f"note: xgboost not installed - the learned method is {backend_name() or 'unavailable'} (reported under that name)")
            print("harmonization:", {k: v for k, v in pipeline.retrain_harmonization(a.out, xcfg, force, hcfg=hcfg).items() if k in ("status", "chosen", "why")})
        if a.evaluate_harmonization:
            print("harmonization diagnostics:", pipeline.evaluate_harmonization(a.out)["summary"])
        if a.evaluate_forecast:
            from . import forecast_eval
            print("forecast evaluation:", forecast_eval.run(a.out, tests=a.forecast_tests, evals=a.hyperopt_evals)["summary"])
    elif a.modis and a.viirs:
        run(a.modis, a.viirs, a.out, a.regions, a.cell, a.keep_all, **kw)
        print(f"XFiRE built -> {a.out}/dashboard.html, results.json, quality_report.json, harmonization_evaluation.md")
    elif not (a.serve and have):
        ap.error("give real --modis and --viirs FIRMS CSV files, or --serve with an existing --out")
    if a.serve:
        from .serve import make_server
        httpd, eng = make_server(a.out, a.port)
        print(f"XFiRE app: http://127.0.0.1:{a.port}  (Ctrl+C to stop)", flush=True)
        try: httpd.serve_forever()
        except KeyboardInterrupt: pass
