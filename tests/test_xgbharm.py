"""python tests/test_xgbharm.py - learned harmonizer on a SYNTHETIC build: features, mass calibration, no-year-feature, protocol plug-in, spatial mask and a forced build.
Runs on XGBoost when installed, otherwise on the scikit-learn stand-in (reported as hist_gbm, never as xgboost)."""
import sys, tempfile
from pathlib import Path
import numpy as np
from test_helpers import build_test_fixture
from xfire import xgbharm as X
from xfire.harmeval import HCfg, spatial_block_folds
from xfire.raster import Raster, predict_ratio
from xfire.harmonize import _wdates
with tempfile.TemporaryDirectory() as td:
    cfg = X.XCfg(rounds=60, max_rows=120000, device="cpu"); out = Path(td) / "o"
    res = build_test_fixture(out, xgb_cfg=cfg, hcfg=HCfg(folds=2, test_years=1), harmonize_method="learned")
    NM = X.backend_name(); assert NM in ("xgboost", "hist_gbm"), NM; assert X.pick_device("cpu") == "cpu" and X.pick_device("cuda") in ("cpu", "cuda") and (NM != "hist_gbm" or X.pick_device("auto") == "cpu")
    he = res["harm_eval"]; print(he["status"], he["chosen"], he["applied"], {k: v["mape"] for k, v in he["metrics"].items()})
    assert he["status"] == "ok" and set(he["metrics"]) == {"ratio", "block_ratio", NM} and he["applied"] == NM and (he["forced"] or he["chosen"] == NM)
    assert res["harm_method"] == "unet" and res["uc_method"] == NM and len(res["uc"]) == len(res["cells"])      # 'unet' = internal name of the cell-level estimate path
    R = Raster(res); w = R.overlap_weeks(); yr = _wdates(w).year.to_numpy(); tr, te = w[yr < 2022], w[yr == 2023]
    m = X.train_xgb(R, tr, cfg); assert m["rounds"] >= 1 and m["cal"].shape == (4,) and (m["cal"] > 0).all()
    t = int(te[10]); p = X.predict_xgb(R, m, t); V = R.img(t, "V"); assert p.shape == V.shape and (p >= 0).all() and (p[V == 0] == 0).all()   # zero where VIIRS saw nothing, like the baselines
    pm = sum(float(X.predict_xgb(R, m, int(x)).sum()) for x in te); mm = sum(float(R.Mw[int(x)]) for x in te); assert abs(pm / mm - 1) < .25, (pm, mm)  # mass-calibrated on TRAIN weeks only
    f = X.week_features(R, t, m["base"]); assert f is not None and f[2].shape[1] == 11 and np.isfinite(f[2]).all()
    # spatial mask: training on a subset of cells never uses the others; the model still predicts everywhere
    fid = spatial_block_folds(R, 4); mk = X.train_xgb(R, tr, cfg, cell_mask=(fid != 0)); assert mk["cell_mask"] is not None and mk["rounds"] >= 1
    pk = X.predict_xgb(R, mk, t); assert pk.shape == V.shape and (pk[V == 0] == 0).all()
    assert he["spatial_holdout"] and NM in he["spatial_holdout"]["metrics"]
    assert he["decision"] and any(d["a"] == NM for d in he["decision"])
    # recent-year weighting option is accepted and trains
    assert X.train_xgb(R, tr, X.XCfg(rounds=20, max_rows=40000, device="cpu", recent_tau=3.0))["rounds"] >= 1
print(f"learned harmonization test PASS (backend {NM})")
