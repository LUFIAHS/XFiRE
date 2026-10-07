import numpy as np
import pandas as pd
from .config import VIIRS_START, SEASONS


def season_num(index):
    return ((index.month % 12) // 3).astype(int)


def _bootstrap_ratio(modis, viirs, n=1000, seed=42):
    if len(modis) == 0 or modis.sum() <= 0:
        return 1.0, 0.9, 1.1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(modis), size=(n, len(modis)))
    ratios = viirs.to_numpy()[idx].sum(axis=1) / np.maximum(modis.to_numpy()[idx].sum(axis=1), 1)
    return float(viirs.sum()/max(modis.sum(), 1)), *map(float, np.percentile(ratios, [5, 95]))


def harmonize_monthly(t, overlap_start=VIIRS_START):
    t = t.copy().sort_index()
    post = t.index >= pd.Timestamp(overlap_start)
    t["season_id"] = season_num(t.index)
    overlap = t.loc[post]
    if overlap.MODIS.sum() == 0 or overlap.VIIRS.sum() == 0:
        raise ValueError("Insufficient MODIS/VIIRS overlap for calibration in this region.")
    calib = {}
    for s in range(4):
        g = overlap.loc[overlap.season_id == s]
        if len(g) < 3 or g.MODIS.sum() < 20:
            k = overlap.VIIRS.sum()/max(overlap.MODIS.sum(), 1)
            calib[s] = (k, k*0.9, k*1.1, "fallback")
        else:
            k, lo, hi = _bootstrap_ratio(g.MODIS, g.VIIRS)
            calib[s] = (k, lo, hi, "seasonal")
    k = t.season_id.map(lambda s: calib[s][0])
    klo = t.season_id.map(lambda s: calib[s][1])
    khi = t.season_id.map(lambda s: calib[s][2])
    v = t.VIIRS.where(post, np.nan)
    # Single-sensor periods are preserved. During overlap, fuse MODIS and calibrated VIIRS.
    t["raw"] = t.MODIS.fillna(0) + t.VIIRS.fillna(0)
    both = post & t.MODIS.gt(0) & t.VIIRS.gt(0)
    only_v = post & t.MODIS.eq(0) & t.VIIRS.gt(0)
    t["harm"] = t.MODIS.astype(float)
    t.loc[both, "harm"] = 0.5 * (t.loc[both, "MODIS"] + v.loc[both] / k.loc[both])
    t.loc[only_v, "harm"] = v.loc[only_v] / k.loc[only_v]
    t["lo"] = t.harm.copy(); t["hi"] = t.harm.copy()
    t.loc[both, "lo"] = 0.5*(t.loc[both,"MODIS"] + v.loc[both]/khi.loc[both])
    t.loc[both, "hi"] = 0.5*(t.loc[both,"MODIS"] + v.loc[both]/klo.loc[both])
    t.loc[only_v, "lo"] = v.loc[only_v]/khi.loc[only_v]
    t.loc[only_v, "hi"] = v.loc[only_v]/klo.loc[only_v]
    valid = overlap.MODIS.gt(0)
    pred = overlap.loc[valid,"VIIRS"] / k.loc[overlap.index[valid]]
    obs = overlap.loc[valid,"MODIS"]
    mape = float((abs(pred-obs)/obs).mean()) if len(obs) else None
    def step(col):
        before = t.loc[(t.index >= "2010-01-01") & (t.index < pd.Timestamp(overlap_start)), col].mean()
        after = t.loc[(t.index >= pd.Timestamp(overlap_start)) & (t.index < pd.Timestamp(overlap_start)+pd.DateOffset(years=2)), col].mean()
        return float(after/before-1) if pd.notna(before) and before else None
    info = {
        "raw_transition": step("raw"), "harmonized_transition": step("harm"),
        "mape_overlap": mape,
        "calibration": {SEASONS[s]: {"ratio": calib[s][0], "lo": calib[s][1], "hi": calib[s][2], "method": calib[s][3]} for s in range(4)},
    }
    return t.drop(columns=["season_id"]), info
