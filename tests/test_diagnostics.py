"""python tests/test_diagnostics.py - day/night, Terra/Aqua, annual ratio drift, recent-year weighting on SYNTHETIC data with known answers."""
import tempfile
from pathlib import Path
import numpy as np, pandas as pd
from xfire.pipeline import run
from xfire import diagnostics as dg


def make(start, end, sat, seed, ratio_fn, share_day=.5, kind="M"):
    rng = np.random.default_rng(seed); rows = []
    for wk in pd.date_range(start, end, freq="7D"):
        base = 120 * (1 + .8 * np.sin(2 * np.pi * (wk.dayofyear - 100) / 365))
        for dn, share in (("D", share_day), ("N", 1 - share_day)):
            lam = base * share * (1 if kind == "M" else ratio_fn(wk, dn)); n = rng.poisson(lam)
            rows.append(pd.DataFrame({"latitude": 23 + rng.normal(0, 1, n), "longitude": 90 + rng.normal(0, 1, n), "acq_date": wk + pd.to_timedelta(rng.integers(0, 7, n), unit="D"),
                                      "acq_time": rng.integers(0, 2359, n), "daynight": dn, "frp": 8.0, "confidence": 80, "satellite": sat, "type": 0}))
    return pd.concat(rows)


def build(td, ratio_fn):
    p = Path(td); m = pd.concat([make("2000-11-06", "2024-12-30", "Terra", 1, None), make("2002-07-08", "2024-12-30", "Aqua", 2, None)]); v = make("2012-02-06", "2024-12-30", "N", 3, ratio_fn, kind="V")
    m.to_csv(p / "m.csv", index=False); v.to_csv(p / "v.csv", index=False)
    return run([p / "m.csv"], [p / "v.csv"], p / "o", keep_all=True, use_xgb=False, log=lambda *a: None)


with tempfile.TemporaryDirectory() as td:
    res = build(td, lambda wk, dn: 3.0 if dn == "D" else 6.0)          # day ratio 3, night ratio 6, stable in time
    d = res["diagnostics"]; dn = d["day_night"]; print(dn["overall_ratio"], dn["separate_calibration_recommended"])
    assert abs(dn["overall_ratio"]["D"] / dn["overall_ratio"]["N"] - 0.5) < .1 and dn["separate_calibration_recommended"] is True
    assert d["ratio_drift"]["Whole extent"]["drift_detected"] is False
    ta = d["terra_aqua"]; assert ta["status"] == "ok" and abs(ta["ratio_second_over_first_overall"] - 1) < .1, ta
    assert abs(d["handover"].get("excess_log_change", 0)) < .2 or d["handover"]["status"] != "ok"
    assert res["recent_tau"] == 0.0, res["recent_tau"]                 # no evidence for weighting when the ratio is stable
with tempfile.TemporaryDirectory() as td:
    ramp = lambda wk, dn: 3.0 + 3.0 * (wk.year - 2012) / 12            # VIIRS/MODIS ratio drifts from 3 to 6
    res = build(td, ramp); d = res["diagnostics"]; print(d["ratio_drift"]["Whole extent"], d["recent_weighting"].get("recommended_tau"))
    assert d["ratio_drift"]["Whole extent"]["drift_detected"] is True
    assert res["recent_tau"] > 0, d["recent_weighting"]                  # weighting recent years helps when the ratio drifts
    import csv; rows = list(csv.DictReader(open(Path(td) / "o" / "annual_ratio_diagnostics.csv"))); assert len(rows) >= 10 and float(rows[-1]["ratio"]) > float(rows[0]["ratio"]) * 1.4
    sc = list(csv.DictReader(open(Path(td) / "o" / "sensor_comparison.csv"))); assert any(r["scope"] == "MODIS|Terra|D" for r in sc) and any(r["scope"].startswith("HARMONIZED|") for r in sc)
print("diagnostics test PASS")
