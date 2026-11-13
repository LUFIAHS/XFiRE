"""Cross-sensor harmonization helpers (numpy/pandas only - no LightGBM / XGBoost needed).

Ways to turn VIIRS counts into MODIS-equivalent counts (all feed the same index):
  * "ratio": seasonal VIIRS/MODIS ratio k estimated on the overlap (the classic, transparent method; optional recent-year weighting)
  * "unet" : cell-level MODIS-equivalent estimates stored in results["uc"] (same cell order as results["cells"]). Despite the name it holds the
             estimate of whichever learned/regional method won the held-out comparison (results["uc_method"]: XGBoost or the regional block ratio).
The index: MODIS before VIIRS starts, the mean of MODIS and MODIS-equivalent VIIRS during the overlap, MODIS-equivalent VIIRS after MODIS ends.
Missing observations are NOT zeros: weeks in which a sensor delivered no data inside its active span (results["obs_gaps"]) are skipped for that
sensor, and weeks without any sensor stay NaN.
The VIIRS transition week is read from the processed data (results["viirs_start"]); VIIRS_W below is only the fallback for old builds.
"""
import math
import numpy as np
import pandas as pd
from .config import VIIRS_NOMINAL_START
from .grid import EPOCH

VIIRS_W = math.ceil((pd.Timestamp(VIIRS_NOMINAL_START) - EPOCH).days / 7)
_wdates = lambda w: EPOCH + pd.to_timedelta(np.asarray(w) * 7, unit="D")


def viirs_week(res):
    """First VIIRS week of the processed data (fallback: nominal start for builds that predate the field)."""
    s = (res or {}).get("viirs_start")
    return math.ceil((pd.Timestamp(s) - EPOCH).days / 7) if s else VIIRS_W


def gap_masks(res, n=None):
    """(gM, gV) boolean arrays: weeks inside a sensor's active span with NO observations at all (missing data, not zero fire)."""
    n = n or res["nweeks"]; g = res.get("obs_gaps") or {}; out = []
    for s in ("MODIS", "VIIRS"):
        a = np.zeros(n, bool); idx = [w for w in g.get(s, []) if 0 <= w < n]; a[idx] = True; out.append(a)
    return out[0], out[1]


def month_of_week(w):
    """Absolute month index (months since 2000-01) of a week, using the week's mid-point (start + 3 days).
    The browser (core.js) uses the same rule, so Python and JS calendars agree."""
    d = EPOCH + pd.to_timedelta(np.atleast_1d(np.asarray(w)) * 7 + 3, unit="D")
    return ((d.year - 2000) * 12 + d.month - 1).to_numpy()


def _in_box(lat, lon, box): return not box or (box[0] <= lat <= box[1] and box[2] <= lon <= box[3])


def box_series(cells, n, box=None):
    """Weekly (MODIS, VIIRS) sums over the cells inside `box`. v3.3: one concatenation + bincount instead of a per-cell np.add.at (about 7x faster on 30k cells, identical sums)."""
    parts = [np.asarray(d).reshape(-1, 3) for lat, lon, d in cells if _in_box(lat, lon, box) and len(d)]
    if not parts: return np.zeros(n), np.zeros(n)
    d = np.concatenate(parts); w = d[:, 0].astype(np.int64)
    return np.bincount(w, d[:, 1], n).astype(float), np.bincount(w, d[:, 2], n).astype(float)


def box_series_u(cells, uc, n, box=None):
    """Weekly sum of the learned MODIS-equivalent estimate inside a box. uc[i] = flat [w, u, w, u, ...] for cells[i]."""
    parts = [np.asarray(d, float).reshape(-1, 2) for (lat, lon, _), d in zip(cells, uc) if len(d) and _in_box(lat, lon, box)]
    if not parts: return np.zeros(n)
    d = np.concatenate(parts); return np.bincount(d[:, 0].astype(np.int64), d[:, 1], n).astype(float)


def _season_of(w): return (_wdates(w).month % 12) // 3


def _ratio_from_weeks(M, V, weeks, wt, fallback):
    """Per-season ratio from the given overlap weeks. wt = per-week weights (recent-year weighting) or None.
    Own ratio if >= 100 (unweighted) MODIS detections; else the pooled ratio (when >= 100 and no `fallback`); else fallback[q]."""
    s = _season_of(weeks); m, v = M[weeks], V[weeks]; ww = np.ones(len(weeks)) if wt is None else wt
    pm, pv, pmr = (m * ww).sum(), (v * ww).sum(), m.sum()
    pooled = pv / pm if pm > 0 and pv > 0 else None
    if fallback is not None and pmr < 100: pooled = None
    k = []
    for q in range(4):
        sel = s == q; mq, vq, mr = (m[sel] * ww[sel]).sum(), (v[sel] * ww[sel]).sum(), m[sel].sum()
        k.append(vq / mq if mr >= 100 and vq > 0 and mq > 0 else (pooled if pooled else (fallback[q] if fallback is not None else None)))
    return k


def _overlap(M, last, mlast, vw, gap):
    w = np.arange(len(M)); hi = last if mlast is None else min(last, mlast)
    ok = (w >= vw) & (w <= hi)
    if gap is not None: ok &= ~gap
    return w[ok]


def recent_weights(weeks, tau):
    """0.5 ** (age_in_years / tau): the newest overlap year has weight 1, an overlap year `tau` years older weighs 0.5. tau <= 0 -> None (unweighted)."""
    if not tau or tau <= 0 or len(weeks) == 0: return None
    yrs = _wdates(weeks).year.to_numpy(); return 0.5 ** ((yrs.max() - yrs) / float(tau))


def season_k(M, V, last, mlast=None, fallback=None, vw=None, gap=None, tau=0.0):
    """Seasonal VIIRS/MODIS ratio on the overlap, as in core.js. mlast = last week with MODIS data: later weeks have no MODIS (retired
    sensor), which is not "zero fires". gap = weeks with missing observations (excluded). tau = recent-year half-life in years (0 = unweighted)."""
    weeks = _overlap(M, last, mlast, VIIRS_W if vw is None else vw, gap)
    return _ratio_from_weeks(M, V, weeks, recent_weights(weeks, tau), fallback)


def ratio_ci(M, V, last, mlast=None, fallback=None, vw=None, gap=None, tau=0.0, reps=300, seed=0):
    """Year-block bootstrap of the seasonal ratio -> [(k, lo90, hi90)] per season (None entries when no ratio exists). This is the uncertainty of
    the SENSOR RATIO only - it is harmonization uncertainty, not forecast uncertainty."""
    weeks = _overlap(M, last, mlast, VIIRS_W if vw is None else vw, gap); k0 = _ratio_from_weeks(M, V, weeks, recent_weights(weeks, tau), fallback)
    yrs = _wdates(weeks).year.to_numpy(); uy = np.unique(yrs); rng = np.random.default_rng(seed); boot = [[] for _ in range(4)]
    if len(uy) >= 3:
        for _ in range(reps):
            pick = rng.choice(uy, len(uy)); idx = np.concatenate([weeks[yrs == y] for y in pick])
            for q, x in enumerate(_ratio_from_weeks(M, V, idx, recent_weights(idx, tau), fallback)):
                if x: boot[q].append(x)
    return [None if k is None else (k, float(np.quantile(b, .05)) if b else k, float(np.quantile(b, .95)) if b else k) for k, b in zip(k0, boot)]


def harmonize(M, V, first, last, k=None, mlast=None, vw=None, gM=None, gV=None, tau=0.0):
    """MODIS-equivalent index; a cell/area can reuse the extent-level ratios `k`.
    Before VIIRS: MODIS. Overlap: mean of MODIS and V/k. After MODIS ends (week > mlast): V/k alone.
    Weeks flagged missing for a sensor (gM / gV) use only the sensor that observed; both missing -> NaN (never zero)."""
    vw = VIIRS_W if vw is None else vw; gap = None if gM is None or gV is None else (gM | gV)
    k = k or season_k(M, V, last, mlast, vw=vw, gap=gap, tau=tau); s = _season_of(np.arange(len(M)))
    h = np.full(len(M), np.nan)
    for i in range(first, last + 1):
        kk = k[s[i]]; m_gap = bool(gM is not None and gM[i]); v_gap = bool(gV is not None and gV[i])
        if i < vw or not kk:
            h[i] = np.nan if m_gap else M[i]
        elif mlast is not None and i > mlast:
            h[i] = np.nan if v_gap else V[i] / kk
        elif m_gap and v_gap: h[i] = np.nan
        elif m_gap: h[i] = V[i] / kk
        elif v_gap: h[i] = M[i]
        else: h[i] = .5 * (M[i] + V[i] / kk)
    return h


def harmonize_u(M, U, first, last, mlast=None, vw=None, gM=None, gV=None):
    """Same index as `harmonize`, but with the learned MODIS-equivalent VIIRS estimate U instead of V / k."""
    vw = VIIRS_W if vw is None else vw; h = np.full(len(M), np.nan)
    for i in range(first, last + 1):
        m_gap = bool(gM is not None and gM[i]); v_gap = bool(gV is not None and gV[i])
        if i < vw: h[i] = np.nan if m_gap else M[i]
        elif mlast is not None and i > mlast: h[i] = np.nan if v_gap else U[i]
        elif m_gap and v_gap: h[i] = np.nan
        elif m_gap: h[i] = U[i]
        elif v_gap: h[i] = M[i]
        else: h[i] = .5 * (M[i] + U[i])
    return h


def modis_last(res):
    """Last week with MODIS detections (the sensor retires): stored by the pipeline, else derived from the data."""
    if res.get("modis_last_w") is not None: return int(res["modis_last_w"])
    M, _ = box_series(res["cells"], res["nweeks"]); nz = np.nonzero(M > 0)[0]
    return int(nz.max()) if len(nz) else -1


def series_for(res, box=None, method=None, bounds=False):
    """Harmonized weekly series of an area for the chosen method ('ratio' | 'unet'; default = res['harm_method']).
    bounds=True -> (h, lo, hi): lo/hi carry ONLY the harmonization uncertainty (ratio bootstrap, or the held-out log-error of the learned
    method) - they are not forecast intervals."""
    method = method or res.get("harm_method", "ratio")
    n, first, lfw = res["nweeks"], res["first_w"], res["last_full_w"]
    ml = modis_last(res); vw = viirs_week(res); gM, gV = gap_masks(res); tau = float(res.get("recent_tau") or 0.0)
    M, V = box_series(res["cells"], n, box)
    if method == "unet" and res.get("uc") is not None:
        U = box_series_u(res["cells"], res["uc"], n, box); h = harmonize_u(M, U, first, lfw, ml, vw, gM, gV)
        if not bounds: return h
        sd = ((res.get("harm_unc") or {}).get("log_sd_by_season") or [0, 0, 0, 0]); s = _season_of(np.arange(n)); z = 1.645 * np.asarray(sd)[s]
        lo, hi = harmonize_u(M, U * np.exp(-z), first, lfw, ml, vw, gM, gV), harmonize_u(M, U * np.exp(z), first, lfw, ml, vw, gM, gV); return h, lo, hi
    Mx, Vx = box_series(res["cells"], n); gap = gM | gV
    kx = season_k(Mx, Vx, lfw, ml, vw=vw, gap=gap, tau=tau); k = season_k(M, V, lfw, ml, kx, vw, gap, tau) if box else kx
    h = harmonize(M, V, first, lfw, k, ml, vw, gM, gV, tau)
    if not bounds: return h
    ci = ratio_ci(M, V, lfw, ml, kx if box else None, vw, gap, tau); s = _season_of(np.arange(n))
    klo = [c[2] if c else None for c in ci]; khi = [c[1] if c else None for c in ci]       # larger ratio -> fewer MODIS-equivalent detections
    return h, harmonize(M, V, first, lfw, klo, ml, vw, gM, gV, tau), harmonize(M, V, first, lfw, khi, ml, vw, gM, gV, tau)


def week_weights_ratio(res):
    """Per-week contribution of ONE detection to the harmonized index (ratio method, extent-level k):
    MODIS -> wM[w]; VIIRS -> wVk[w] (= weight / k). NaN for weeks outside the analysed range (so CSV rows can say so)."""
    n, first, lfw, ml = res["nweeks"], res["first_w"], res["last_full_w"], res.get("modis_last_w"); vw = viirs_week(res); gM, gV = gap_masks(res)
    M, V = box_series(res["cells"], n); k = season_k(M, V, lfw, ml, vw=vw, gap=gM | gV, tau=float(res.get("recent_tau") or 0.0)); s = _season_of(np.arange(n))
    wM, wVk = np.ones(n), np.zeros(n)
    for w in range(max(first, vw), lfw + 1):
        kk = k[s[w]]
        if not kk: continue
        if ml is not None and w > ml: wM[w], wVk[w] = 0.0, 1 / kk
        elif gM[w]: wM[w], wVk[w] = 0.0, 1 / kk
        elif gV[w]: wM[w], wVk[w] = 1.0, 0.0
        else: wM[w], wVk[w] = .5, .5 / kk
    wM[lfw + 1:] = np.nan; wVk[lfw + 1:] = np.nan
    return wM, wVk
