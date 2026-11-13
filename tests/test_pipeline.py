"""python tests/test_pipeline.py (XGBoost kept small). SYNTHETIC build: calendar/map consistency inputs, learned-estimate plumbing, rule-based clusters, hotspot CSV, forecast + reports."""
import json, tempfile
from pathlib import Path
import numpy as np
from test_helpers import build_test_fixture
from xfire.raster import Raster, monthly_cube, predict_ratio, fit_ratio, to_uc
from xfire.harmonize import series_for, month_of_week
from xfire import events as ev, hotspots as hs
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "o"; res = build_test_fixture(out)
    assert res["modis_last_w"] < res["last_full_w"] and res["events"]["rule"] and res["hotspots"]["available"] and res["forecast"]
    for f in ("results.json", "quality_report.json", "dashboard.html", "harmonization_evaluation.md", "annual_ratio_diagnostics.csv", "sensor_comparison.csv", "harmonization_model_metrics.csv"): assert (out / f).exists(), f
    R = Raster(res); months, cube, nw = monthly_cube(R, "ratio")
    h = series_for(res, None, "ratio"); first, lfw = res["first_w"], res["last_full_w"]
    mo = month_of_week(np.arange(first, lfw + 1)); i = np.nonzero(months == mo[len(mo) // 2])[0][0]
    wk = np.arange(first, lfw + 1)[mo == months[i]]; v = h[wk].sum() * 30.4375 / (7 * len(wk))
    assert abs(cube[i].sum() - v) / v < .05, (cube[i].sum(), v)           # cube month ~ weekly series month (days-in-month differs slightly)
    # learned-estimate plumbing: a stand-in estimate goes through to_uc -> res['uc'] -> series_for / monthly_cube
    weeks = R.overlap_weeks(); g = fit_ratio(R, weeks)
    rest = [w for w in range(R.modis_last + 1, R.lfw + 1) if R.Vw[w] > 0]
    res["uc"] = to_uc(R, {w: predict_ratio(R, g, w) for w in list(weeks) + rest})
    hr, hu = series_for(res, None, "ratio"), series_for(res, None, "unet")
    assert abs(np.nansum(hu) / np.nansum(hr) - 1) < .1
    R2 = Raster(res); _, cube_u, _ = monthly_cube(R2, "unet"); assert abs(cube_u.sum() / cube.sum() - 1) < .1
    # harmonization uncertainty is separate from forecast uncertainty and brackets the estimate
    hb, lo, hi = series_for(res, None, "ratio", bounds=True); ok = np.isfinite(hb); assert (lo[ok] <= hb[ok] + 1e-6).all() and (hi[ok] >= hb[ok] - 1e-6).all() and (hi[ok] > lo[ok]).any()
    # rule-based clusters
    ok_ = nw >= 4; clim, has = ev.loyo_climatology(cube, months, ok_); th = ev.thresholds(cube, ok_)
    t = int(np.nonzero(ok_)[0][-1]); assert ev.rule_events(cube[t], clim[t], th)
    rows = "".join(hs.csv_chunks(hs.query(out / "hotspots", [-8, -2, -64, -56], "2019-08-01", "2019-08-31"))).splitlines()
    assert rows[0].startswith("latitude") and len(rows) > 100
    # forecast stored with intervals and measured coverage
    f = res["forecast"]["Whole extent"]; assert f["lo"] and f["skill"]["cover"] is not None
    print("pipeline test PASS", {"months": len(months), "csv_rows": len(rows) - 1})
