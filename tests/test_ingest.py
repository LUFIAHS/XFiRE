"""python tests/test_ingest.py - vegetation filter, satellite policy, per-stage filter counts, duplicates, real VIIRS start, missing weeks.
Synthetic FIRMS-like files: these tests check the CODE, they say nothing about real-data accuracy."""
import tempfile
from pathlib import Path
import numpy as np, pandas as pd
from xfire.pipeline import run
from xfire import hotspots as hs
from xfire.harmonize import series_for, viirs_week


def weekly(start, end, sat, rate, seed, typ_noise=0, skip=()):
    rng = np.random.default_rng(seed); rows = []
    for wk in pd.date_range(start, end, freq="7D"):
        if wk in skip: continue
        n = rng.poisson(rate * (1 + .8 * np.sin(2 * np.pi * wk.dayofyear / 365)))
        rows.append(pd.DataFrame({"latitude": 23 + rng.normal(0, 1, n), "longitude": 90 + rng.normal(0, 1, n),
                                  "acq_date": wk + pd.to_timedelta(rng.integers(0, 7, n), unit="D"), "acq_time": rng.integers(0, 2359, n), "daynight": rng.choice(["D", "N"], n), "frp": 5.0, "confidence": 80,
                                  "satellite": sat, "type": 0}))
        if typ_noise:                                            # a gas flare: fixed place, every week, type 2
            rows.append(pd.DataFrame({"latitude": [24.0] * typ_noise, "longitude": [91.0] * typ_noise, "acq_date": [wk] * typ_noise, "acq_time": 1200, "daynight": "D",
                                      "frp": 50.0, "confidence": 95, "satellite": sat, "type": 2}))
    return pd.concat(rows)


with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    m = pd.concat([weekly("2000-11-06", "2014-12-29", "Terra", 40, 1, 3), weekly("2002-07-08", "2014-12-29", "Aqua", 40, 2)])
    bad = m.iloc[:50].copy(); bad["latitude"] = 123.0; low = m.iloc[50:100].copy(); low["confidence"] = 5
    m = pd.concat([m, m.iloc[100:300], bad, low]); m.to_csv(p / "m.csv", index=False)       # 200 exact duplicates, 50 invalid coordinates, 50 low confidence
    v = pd.concat([weekly("2012-02-06", "2014-12-29", "N", 160, 3, 5), weekly("2013-06-03", "2014-12-29", "1", 160, 4)]); v.to_csv(p / "v.csv", index=False)
    res = run([p / "m.csv"], [p / "v.csv"], p / "o", keep_all=True, use_xgb=False, log=lambda *a: None)
    pol = res["sat_policy"]; print(pol)
    assert pol["viirs_kept"] == "N" and pol["viirs_dropped"] == ["1"], pol
    assert pol["modis_trim_date"] and pol["modis_trim_date"] >= "2002-07-01", pol
    f = res["quality"]["MODIS"]["filters"]; print(f)
    assert f["invalid_coordinates"] == 50 and f["low_confidence"] == 50 and f["duplicate"] >= 150 and f["non_vegetation"] > 0, f
    assert res["quality"]["MODIS"]["rows_read"] - res["quality"]["MODIS"]["rows_valid"] == sum(f.values())           # every removed row is accounted for
    assert res["first_w"] * 7 >= (pd.Timestamp("2002-07-01") - pd.Timestamp("2000-01-01")).days
    assert "2012-02-06" <= res["viirs_start"] <= "2012-02-13", res["viirs_start"]                                    # read from the data
    tot = lambda s: sum(c[2][s::3][i] for c in res["cells"] for i in range(len(c[2][s::3])))
    assert abs(tot(2) / (160 * 52 * 2.9) - 1) < .3, tot(2)           # only the 'N' satellite remains (about 2.9 years)
    rows = "".join(hs.csv_chunks(hs.query(p / "o" / "hotspots", None, "2001-01-01", "2002-01-01"))).splitlines()
    assert len(rows) == 1, len(rows)                                  # the Terra-only year is excluded from the download too
    assert (p / "o" / "diag_counts.csv.gz").exists()
    off = run([p / "m.csv"], [p / "v.csv"], p / "o2", keep_all=True, use_xgb=False, sat_policy="off", veg_only=False, log=lambda *a: None)
    assert off["first_w"] < res["first_w"] and off["quality"]["MODIS"]["non_vegetation_removed"] == 0
    # the transition is NOT a constant: VIIRS files that start in 2014 move the start
    v2 = weekly("2014-03-03", "2016-12-26", "N", 160, 5); m2 = pd.concat([weekly("2000-11-06", "2016-12-26", "Terra", 40, 1), weekly("2002-07-08", "2016-12-26", "Aqua", 40, 2)])
    m2.to_csv(p / "m2.csv", index=False); v2.to_csv(p / "v2.csv", index=False)
    late = run([p / "m2.csv"], [p / "v2.csv"], p / "o3", keep_all=True, use_xgb=False, log=lambda *a: None)
    assert "2014-03-03" <= late["viirs_start"] <= "2014-03-10", late["viirs_start"]; assert any("nominal" in w for w in late["data_warnings"])
    # missing observations are NaN, never zero: drop one week of BOTH sensors and one VIIRS-only week
    gap_m = pd.Timestamp("2015-05-04"); skip_m = {pd.Timestamp("2015-05-04") + pd.Timedelta(days=0)}
    m3 = pd.concat([weekly("2000-11-06", "2016-12-26", "Terra", 40, 1), weekly("2002-07-08", "2016-12-26", "Aqua", 40, 2)])
    v3 = weekly("2012-02-06", "2016-12-26", "N", 160, 3)
    wk = lambda d: (pd.Timestamp(d) - pd.Timestamp("2000-01-01")).days // 7
    m3 = m3[~m3.acq_date.between("2015-05-02", "2015-05-08")]; v3 = v3[~v3.acq_date.between("2015-05-02", "2015-05-22")]
    m3.to_csv(p / "m3.csv", index=False); v3.to_csv(p / "v3.csv", index=False)
    g = run([p / "m3.csv"], [p / "v3.csv"], p / "o4", keep_all=True, use_xgb=False, log=lambda *a: None)
    print(g["obs_gaps"]); assert wk("2015-05-02") in g["obs_gaps"]["MODIS"] and wk("2015-05-09") in g["obs_gaps"]["VIIRS"] and wk("2015-05-02") in g["obs_gaps"]["VIIRS"]
    h = series_for(g, None, "ratio")
    assert np.isnan(h[[w for w in range(len(h)) if w in set(g["obs_gaps"]["MODIS"]) & set(g["obs_gaps"]["VIIRS"])][0]]), "both missing -> NaN"
    onlyv = [w for w in g["obs_gaps"]["VIIRS"] if w not in g["obs_gaps"]["MODIS"]][0]; assert h[onlyv] > 0, "VIIRS missing -> MODIS alone is used, not zero"
    print("ingest test PASS")
