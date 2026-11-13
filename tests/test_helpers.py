"""Temporary generated FIRMS-like fixtures for isolated automated tests only."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from xfire.pipeline import run

TEST_CENTERS = ((23, 90, 100, 30), (-5, -60, 230, 45), (-25, 135, 300, 20))
TEST_HET = (1.5, 0.75, 1.0)
TEST_BURSTS = ((1, "2005-08-01", "2005-10-15", 2.6), (1, "2010-08-01", "2010-10-15", 3.0),
               (1, "2019-08-01", "2019-10-15", 3.2), (2, "2019-11-15", "2020-02-15", 3.8),
               (0, "2016-03-01", "2016-04-30", 2.6), (1, "2026-08-15", "2026-10-25", 2.4),
               (0, "2023-03-01", "2023-04-20", 2.2))

def write_test_observations(path, start, k, seed, centers=((23, 90, 100, 40),), end="2020-12-28", het=None, bursts=()):
    rng = np.random.default_rng(seed)
    rows = []
    for week in pd.date_range(start, end, freq="7D"):
        for ci, (lat, lon, phase, level) in enumerate(centers):
            truth = level * (1 + 0.8 * np.sin(2 * np.pi * (week.dayofyear - phase) / 365))
            for c, a, b, mult in bursts:
                if c == ci and pd.Timestamp(a) <= week <= pd.Timestamp(b):
                    truth *= mult
            n = rng.poisson(truth * k * (het[ci] if het else 1))
            rows.append(pd.DataFrame({"latitude": lat + rng.normal(0, 1.2, n),
                                      "longitude": lon + rng.normal(0, 1.5, n),
                                      "acq_date": week + pd.to_timedelta(rng.integers(0, 7, n), unit="D"),
                                      "frp": rng.lognormal(2, .5, n), "confidence": 80}))
    pd.concat(rows).to_csv(path, index=False)

def write_sensor_test_csvs(folder):
    folder = Path(folder)
    write_test_observations(folder / "m.csv", "2004-01-05", 1, 1)
    write_test_observations(folder / "v.csv", "2012-02-06", 4, 2)
    return folder / "m.csv", folder / "v.csv"

def build_test_fixture(out, **kw):
    kw.setdefault("fc_evals", 8)                                   # keep tests fast: few Hyperopt trials, small XGBoost
    if kw.get("use_xgb", True) and "xgb_cfg" not in kw:
        from xfire.xgbharm import XCfg; kw["xgb_cfg"] = XCfg(rounds=60, max_rows=100000, device="cpu")
    kw.setdefault("log", lambda *a: None)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        write_test_observations(p / "m.csv", "2001-01-01", 1, 1, TEST_CENTERS, "2026-06-22", None, TEST_BURSTS)
        write_test_observations(p / "v.csv", "2012-02-06", 4, 2, TEST_CENTERS, "2026-10-19", TEST_HET, TEST_BURSTS)
        return run([p / "m.csv"], [p / "v.csv"], out, keep_all=True, **kw)
