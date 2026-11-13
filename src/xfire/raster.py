"""Raster layer (numpy only): a dense, image-like view of results['cells'] used by the XGBoost stage, the statistical baselines and the rule-based cluster detector.

* Raster      weekly MODIS / VIIRS / learned-estimate images on the 0.5-degree grid + model input features
* baselines   global seasonal ratio (what the app always did) and a spatially varying block ratio (a stronger classical baseline)
* evaluate    cell-level Poisson deviance and area-level weekly error, out-of-fold - the yard-stick XGBoost must beat
* monthly_cube  harmonized monthly activity [T, H, W] (input of the event detectors)
Image row i = latitude cell index (south -> north); column j = longitude cell index (west -> east).
"""
import numpy as np
import pandas as pd
from .harmonize import viirs_week, gap_masks, _wdates, month_of_week, season_k


class Raster:
    def __init__(self, res):
        self.res, self.cell, self.n = res, float(res["cell"]), int(res["nweeks"])
        self.first, self.lfw = int(res["first_w"]), int(res["last_full_w"])
        cells = res["cells"]
        lat = np.array([c[0] for c in cells], float); lon = np.array([c[1] for c in cells], float)
        ci = np.rint((lat + 90) / self.cell - .5).astype(int); cj = np.rint((lon + 180) / self.cell - .5).astype(int)
        self.i0, self.j0 = int(ci.min()), int(cj.min())
        self.H, self.W = int(ci.max()) - self.i0 + 1, int(cj.max()) - self.j0 + 1
        lens = np.array([len(c[2]) // 3 for c in cells])
        flat = np.concatenate([np.asarray(c[2], dtype=np.int64).reshape(-1, 3) for c in cells])
        o = np.argsort(flat[:, 0], kind="stable")                         # entries sorted by week -> O(1) week slices
        self.w = flat[o, 0]; self.m = flat[o, 1].astype(np.float32); self.v = flat[o, 2].astype(np.float32)
        self.cid = np.repeat(np.arange(len(cells)), lens)[o]
        self.ri, self.rj = (ci - self.i0)[self.cid], (cj - self.j0)[self.cid]
        self.off = np.searchsorted(self.w, np.arange(self.n + 1))
        self.Mw = np.bincount(self.w, self.m, minlength=self.n); self.Vw = np.bincount(self.w, self.v, minlength=self.n)
        nz = np.nonzero(self.Mw > 0)[0]; self.modis_last = int(nz.max()) if len(nz) else -1
        self.vw = viirs_week(res); self.gM, self.gV = gap_masks(res, self.n)                      # real VIIRS start; weeks with missing observations
        self.cell_ri, self.cell_rj = ci - self.i0, cj - self.j0           # per cell (index = position in res['cells'])
        self.uw = self.uv = self.uri = self.urj = None
        if res.get("uc") is not None:                                     # learned estimates (XGBoost), parallel to cells
            ulen = np.array([len(d) // 2 for d in res["uc"]])
            if ulen.sum():
                uf = np.concatenate([np.asarray(d, float).reshape(-1, 2) for d in res["uc"] if len(d)])
                ucid = np.repeat(np.arange(len(cells)), ulen)
                self.uw, self.uv = uf[:, 0].astype(int), uf[:, 1].astype(np.float32)
                self.uri, self.urj = self.cell_ri[ucid], self.cell_rj[ucid]
        h = self.cell / 2
        self.lat_c = -90 + (self.i0 + np.arange(self.H) + .5) * self.cell; self.lon_c = -180 + (self.j0 + np.arange(self.W) + .5) * self.cell
        if res.get("keep_all", False) or not res.get("regions"):
            self.valid = np.ones((self.H, self.W), bool)
        else:
            self.valid = np.zeros((self.H, self.W), bool)
            for a, b, c, d in res["regions"].values():
                self.valid |= ((self.lat_c >= a - h) & (self.lat_c <= b + h))[:, None] & ((self.lon_c >= c - h) & (self.lon_c <= d + h))[None, :]
        wd = _wdates(np.arange(self.n)); self.season = ((wd.month % 12) // 3).to_numpy()
        doy = (wd + pd.Timedelta(days=3)).dayofyear.to_numpy(); self.sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
        self.cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
        self.latg = np.repeat((self.lat_c / 90)[:, None], self.W, 1).astype(np.float32)
        self.long = np.repeat((self.lon_c / 180)[None, :], self.H, 0).astype(np.float32)

    # ---- images -------------------------------------------------------------------------------
    def img(self, w, kind="V"):
        a = np.zeros((self.H, self.W), np.float32)
        if 0 <= w < self.n:
            s, e = self.off[w], self.off[w + 1]
            a[self.ri[s:e], self.rj[s:e]] = {"M": self.m, "V": self.v}[kind][s:e]
        return a

    def features(self, w):
        """Model input [7, H, W]: log1p VIIRS (this week, previous week), season sin/cos, lat, lon, valid-mask."""
        one = np.ones((self.H, self.W), np.float32)
        return np.stack([np.log1p(self.img(w, "V")), np.log1p(self.img(w - 1, "V")), self.sin[w] * one, self.cos[w] * one,
                         self.latg, self.long, self.valid.astype(np.float32)])

    def _img_crop(self, w, kind, i0, j0, size):
        a = np.zeros((size, size), np.float32)
        if 0 <= w < self.n:
            s, e = self.off[w], self.off[w + 1]; ri, rj = self.ri[s:e] - i0, self.rj[s:e] - j0
            m = (ri >= 0) & (ri < size) & (rj >= 0) & (rj < size)
            a[ri[m], rj[m]] = {"M": self.m, "V": self.v}[kind][s:e][m]
        return a

    def crop(self, w, i0, j0, size):
        """(features [7, size, size], MODIS target [size, size]) of a window whose corner is cell (i0, j0); same channels as
        features(); window parts outside the grid are zero with valid-mask 0. Cheap: never builds a full-extent image."""
        h, wd = min(size, self.H - i0), min(size, self.W - j0)

        def pad(a):
            o = np.zeros((size, size), np.float32); o[:h, :wd] = a[i0:i0 + h, j0:j0 + wd]; return o
        one = np.ones((size, size), np.float32)
        X = np.stack([np.log1p(self._img_crop(w, "V", i0, j0, size)), np.log1p(self._img_crop(w - 1, "V", i0, j0, size)),
                      self.sin[w] * one, self.cos[w] * one, pad(self.latg), pad(self.long), pad(self.valid.astype(np.float32))])
        return X, self._img_crop(w, "M", i0, j0, size)

    def overlap_weeks(self):
        """Weeks where BOTH sensors reported and neither has a missing-observation flag (the only weeks a MODIS target exists for)."""
        w = np.arange(self.n)
        return w[(w >= max(self.vw, self.first)) & (w <= min(self.lfw, self.modis_last)) & (self.Mw > 0) & (self.Vw > 0) & ~self.gM & ~self.gV]

    def box_mask(self, box):
        """(H, W) bool of the cells box_series would add up: lat_min <= centre <= lat_max etc."""
        return ((self.lat_c >= box[0]) & (self.lat_c <= box[1]))[:, None] & ((self.lon_c >= box[2]) & (self.lon_c <= box[3]))[None, :]


# ---- classical baselines ------------------------------------------------------------------------
def fit_ratio(R, train_w, block=None, lam=100.0, cell_mask=None):
    """VIIRS/MODIS ratio per season from the training weeks. block=None -> one ratio per season (the app's classical method).
    block=B  -> one per (B x B cells, season), shrunk towards the seasonal ratio with pseudo-count `lam` MODIS detections."""
    tw = np.zeros(R.n, bool); tw[np.asarray(train_w, int)] = True
    e = tw[R.w]
    if cell_mask is not None: e = e & cell_mask[R.ri, R.rj]          # spatial hold-out: cells outside the mask contribute nothing, so held-out blocks fall back to the seasonal ratio
    s = R.season[R.w][e]; M, V = R.m[e], R.v[e]
    k0 = []; pooled = V.sum() / max(M.sum(), 1e-9) if M.sum() > 0 and V.sum() > 0 else 1.0
    for q in range(4):
        m, v = M[s == q].sum(), V[s == q].sum(); k0.append(v / m if m >= 100 and v > 0 else pooled)
    if block is None:
        return {"k0": np.array(k0), "table": None, "block": None}
    nbi, nbj = -(-R.H // block), -(-R.W // block); bi, bj = R.ri[e] // block, R.rj[e] // block
    tab = np.zeros((4, nbi, nbj))
    for q in range(4):
        m = np.zeros((nbi, nbj)); v = np.zeros((nbi, nbj)); sel = s == q
        np.add.at(m, (bi[sel], bj[sel]), M[sel]); np.add.at(v, (bi[sel], bj[sel]), V[sel])
        tab[q] = (v + lam * k0[q]) / (m + lam)
    return {"k0": np.array(k0), "table": tab, "block": block}


def predict_ratio(R, fit, w):
    """MODIS-equivalent estimate image of week w = V / k."""
    V = R.img(w, "V"); q = R.season[w]
    if fit["table"] is None:
        return V / fit["k0"][q]
    B = fit["block"]; k = fit["table"][q][(np.arange(R.H) // B)[:, None], (np.arange(R.W) // B)[None, :]]
    return V / k


# ---- evaluation ---------------------------------------------------------------------------------
def poisson_dev(y, mu):
    mu = np.maximum(mu, 1e-3); t = np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0)
    return 2 * (t - (y - mu))


SEASONS = ("DJF", "MAM", "JJA", "SON")
LEVELS = (("low (5-50)", 5, 50), ("mid (50-500)", 50, 500), ("high (500+)", 500, 1e18))


class Evaluator:
    """Accumulates, week by week, how well each method's MODIS-equivalent estimate matches the real MODIS image. Feed only weeks a
    model did NOT train on. `boxes` = {name: [lat0, lat1, lon0, lon1] | None}. For every method it reports
         dev      mean Poisson deviance per active cell-week (lower is better)
         mape     mean over areas of the mean weekly |estimate - MODIS| / MODIS  (weeks with >= 5 MODIS detections)   [the selection metric]
         bias     signed error of the summed estimate vs summed MODIS over all area-weeks (+ = estimate too high)
         areas / by_season / by_year / by_level   the same error split by region, season, calendar year and MODIS activity level
    and `paired(a, b)` gives a block-bootstrap confidence interval of the error difference between two methods on the same weeks."""

    def __init__(self, R, boxes, classes=None, cell_mask=None):
        self.R, self.acc, self.classes, self.cm = R, {}, classes or {}, cell_mask
        base = R.valid if cell_mask is None else R.valid & cell_mask                 # cell_mask: evaluate only these cells (spatial hold-out blocks)
        self.valid = base; self.masks = {k: (base if b is None else base & R.box_mask(b)) for k, b in boxes.items()}

    def add(self, name, w, p):
        R = self.R; a = self.acc.setdefault(name, {"dev": 0.0, "cnt": 0, "weeks": 0, "rows": [], "sp": 0.0, "spn": 0})
        y, v = R.img(w, "M"), R.img(w, "V"); act = self.valid & ((y > 0) | (v > 0))
        a["dev"] += float(poisson_dev(y[act], p[act]).sum()); a["cnt"] += int(act.sum()); a["weeks"] += 1
        if act.sum() >= 20:                                                       # spatial agreement: correlation of log counts over the week's active cells
            ly, lp = np.log1p(y[act]), np.log1p(p[act])
            if ly.std() > 0 and lp.std() > 0: a["sp"] += float(np.corrcoef(ly, lp)[0, 1]); a["spn"] += 1
        for k, mk in self.masks.items():
            m = float(y[mk].sum())
            if m >= 5: a["rows"].append((int(w), k, m, float(p[mk].sum())))

    @staticmethod
    def _mape(rows, key=None):
        """mean over areas of mean APE; rows = (w, area, M, P)"""
        per = {}
        for w, k, m, p in rows: per.setdefault(k, []).append(abs(p - m) / m)
        got = {k: float(np.mean(x)) for k, x in per.items()}
        return (round(float(np.mean(list(got.values()))), 4) if got else None), {k: round(x, 4) for k, x in got.items()}

    def _split(self, rows, keyfn, order=None):
        groups = {}
        for r in rows: groups.setdefault(keyfn(r), []).append(r)
        keys = order if order is not None else sorted(groups)
        return {str(k): (self._mape(groups[k])[0] if k in groups else None) for k in keys}

    def result(self):
        out = {}
        for name, a in self.acc.items():
            rows = a["rows"]; mape, areas = self._mape(rows); wy = _wdates(np.array([r[0] for r in rows], int)).year.to_numpy() if rows else np.array([])
            ysum = {int(r[0]): int(y) for r, y in zip(rows, wy)}
            sm = sum(r[2] for r in rows); bias = (sum(r[3] for r in rows) - sm) / sm if sm > 0 else None
            whole = [r for r in rows if r[1] == "Whole extent"] or rows; ae = np.array([r[3] - r[2] for r in whole], float); lo, hi = self._boot(rows)
            wm = float(sum(r[2] for r in whole)); wape = None if wm <= 0 else round(float(sum(abs(r[3] - r[2]) for r in whole) / wm), 4)
            out[name] = {"dev": round(a["dev"] / max(a["cnt"], 1), 4), "mape": mape, "mape_lo90": lo, "mape_hi90": hi, "wape_whole": wape, "mae": None if not len(ae) else round(float(np.abs(ae).mean()), 1),
                         "rmse": None if not len(ae) else round(float(np.sqrt((ae ** 2).mean())), 1), "spatial_r": round(a["sp"] / a["spn"], 4) if a["spn"] else None,
                         "by_region_class": {c: self._mape([r for r in rows if self.classes.get(r[1]) == c])[0] for c in ("high", "mid", "low") if any(self.classes.get(r[1]) == c for r in rows)}, "areas": {k: (areas.get(k)) for k in self.masks},
                         "weeks": a["weeks"], "bias": None if bias is None else round(float(bias), 4),
                         "bias_whole": None if wm <= 0 else round(float((sum(r[3] for r in whole) - wm) / wm), 4),
                         "by_season": self._split(rows, lambda r: SEASONS[int(self.R.season[r[0]])], SEASONS),
                         "by_year": self._split(rows, lambda r: ysum[r[0]]),
                         "by_level": self._split(rows, lambda r: next(n for n, lo, hi in LEVELS if lo <= r[2] < hi), [n for n, _, _ in LEVELS])}
        return out

    @staticmethod
    def _area_year_matrix(areas, years, vals):
        """sum and count of `vals` per (area, year) -> (S, C) arrays; used so the bootstrap recomputes the SAME statistic as the headline (mean over areas of the area mean)."""
        ua, uy = sorted(set(areas)), np.unique(years); ai = {a: i for i, a in enumerate(ua)}; S = np.zeros((len(ua), len(uy))); C = np.zeros_like(S); yi = {y: i for i, y in enumerate(uy)}
        for a_, y_, v_ in zip(areas, years, vals): S[ai[a_], yi[y_]] += v_; C[ai[a_], yi[y_]] += 1
        return S, C

    @staticmethod
    def _boot_stat(S, C, reps, seed, level=.90):
        """Resample calendar years; statistic = mean over areas of (sum / count). Returns (point, lo, hi)."""
        ny = S.shape[1]; rng = np.random.default_rng(seed)
        def stat(idx):
            c = C[:, idx].sum(1); ok = c > 0
            return float((S[:, idx].sum(1)[ok] / c[ok]).mean()) if ok.any() else float("nan")
        point = stat(np.arange(ny)); b = np.array([stat(rng.integers(0, ny, ny)) for _ in range(reps)]); a = (1 - level) / 2
        return point, float(np.nanquantile(b, a)), float(np.nanquantile(b, 1 - a))

    def _boot(self, rows, reps=500, seed=0):
        """90% block-bootstrap (calendar years) interval of the SAME statistic as `mape` (mean over areas of the area's mean APE)."""
        if len(rows) < 20: return None, None
        w = np.array([r[0] for r in rows]); yrs = _wdates(w).year.to_numpy()
        if len(np.unique(yrs)) < 3: return None, None
        S, C = self._area_year_matrix([r[1] for r in rows], yrs, [abs(r[3] - r[2]) / r[2] for r in rows]); _, lo, hi = self._boot_stat(S, C, reps, seed)
        return round(lo, 4), round(hi, 4)

    def paired(self, a, b, reps=1000, seed=0):
        """Error difference mape(a) - mape(b) on the SAME area-weeks. delta is exactly the difference of the two headline `mape` values restricted to the shared area-weeks
        (v3.3: it used to be a pooled area-week mean, which could disagree with the headline). 90 % block-bootstrap interval (blocks = calendar years; 13-week blocks when
        there are fewer than 3 years). Negative = `a` is better. `win_rate` = share of area-weeks where a < b; `year_wins` = share of calendar years where a < b."""
        ra = {(r[0], r[1]): abs(r[3] - r[2]) / r[2] for r in self.acc.get(a, {}).get("rows", [])}
        rb = {(r[0], r[1]): abs(r[3] - r[2]) / r[2] for r in self.acc.get(b, {}).get("rows", [])}
        keys = sorted(set(ra) & set(rb))
        if len(keys) < 20: return None
        w = np.array([k[0] for k in keys]); d = np.array([ra[k] - rb[k] for k in keys]); yrs = _wdates(w).year.to_numpy(); blk = yrs if len(np.unique(yrs)) >= 3 else (w - w.min()) // 13
        S, C = self._area_year_matrix([k[1] for k in keys], blk, d); delta, lo, hi = self._boot_stat(S, C, reps, seed)
        ub = np.unique(yrs); yw = float(np.mean([d[yrs == u].mean() < 0 for u in ub]))
        return {"delta": round(delta, 4), "lo": round(lo, 4), "hi": round(hi, 4), "blocks": int(len(np.unique(blk))), "n": int(len(d)), "win_rate": round(float((d < 0).mean()), 3), "year_wins": round(yw, 3),
                "unit": "mean over areas of the area-mean absolute-percentage-error difference"}


def evaluate(R, preds, weeks, boxes):
    """preds: {method: {week: (H,W) estimate}} -> Evaluator.result() (convenience wrapper)."""
    ev = Evaluator(R, boxes)
    for name, P in preds.items():
        for w in weeks:
            if w in P: ev.add(name, w, P[w])
    return ev.result()


def folds_of(R, weeks, k=3, min_train_years=3, min_year_weeks=40):
    """Forward-chaining (chronological) folds in WHOLE calendar years: fold j tests year Y_j and trains only on overlap years before it, so no model ever
    sees the future or the year it predicts. At most k folds (the newest k complete years); fewer are returned - and reported by the caller - when
    the overlap is too short to leave `min_train_years` years of training data. A year is complete with >= min_year_weeks overlap weeks.
    Returns [(train_weeks, test_weeks), ...]."""
    weeks = np.asarray(weeks, int); yrs = _wdates(weeks).year.to_numpy(); uy = np.unique(yrs)
    full = [int(y) for y in uy if (yrs == y).sum() >= min_year_weeks]; cand = [y for y in full if (uy < y).sum() >= min_train_years]
    return [(weeks[yrs < Y], weeks[yrs == Y]) for Y in cand[-k:]] if cand and k > 0 else []


# ---- sparse <-> dense ---------------------------------------------------------------------------
class UcBuilder:
    """Dense weekly estimate images -> per-cell sparse lists [w, u, w, u, ...] parallel to results['cells'] (u rounded to
    2 decimals, values < thresh dropped). Memory stays small: images are added one at a time as compact numpy triples (v3.3: no per-cell Python loop)."""

    def __init__(self, R, thresh=0.05):
        self.R, self.t = R, thresh; self.n = len(R.res["cells"]); self.parts = []
        self.idx = -np.ones((R.H, R.W), int); self.idx[R.cell_ri, R.cell_rj] = np.arange(self.n)       # only cells that exist in results['cells']

    def add(self, w, img):
        img = np.where(self.R.valid, img, 0.0); sel = (img >= self.t) & (self.idx >= 0); c = self.idx[sel]
        if c.size: self.parts.append((c.astype(np.int32), np.full(c.size, int(w), np.int32), np.round(img[sel], 2).astype(np.float32)))

    def lists(self):
        out = [[] for _ in range(self.n)]
        if not self.parts: return out
        c = np.concatenate([p[0] for p in self.parts]); w = np.concatenate([p[1] for p in self.parts]); u = np.concatenate([p[2] for p in self.parts])
        o = np.lexsort((w, c)); c, w, u = c[o], w[o], u[o]; cuts = np.flatnonzero(np.diff(c)) + 1; starts = np.r_[0, cuts]; ends = np.r_[cuts, len(c)]
        for s_, e_ in zip(starts, ends): out[int(c[s_])] = [x for a, b in zip(w[s_:e_].tolist(), np.round(u[s_:e_].astype(float), 2).tolist()) for x in (a, b)]
        return out


def to_uc(R, preds, thresh=0.05):
    b = UcBuilder(R, thresh)
    for w in sorted(preds): b.add(w, preds[w])
    return b.lists()


# ---- monthly harmonized cube --------------------------------------------------------------------
def week_weights(R, method="ratio"):
    """Per-week weights (wM, wV) of the index: (1,0) before VIIRS or without a ratio, (.5,.5) in the overlap, (0,1) once MODIS has stopped.
    A week in which a sensor has NO observations (missing, not zero) gives its weight to the other sensor; both missing -> (0,0).
    Mirrors harmonize.harmonize / core.js."""
    tau = float(R.res.get("recent_tau") or 0.0)
    k = season_k(R.Mw, R.Vw, R.lfw, R.modis_last, vw=R.vw, gap=R.gM | R.gV, tau=tau) if method != "unet" else [1.0] * 4
    wM, wV = np.ones(R.n), np.zeros(R.n)
    for w in range(max(R.first, R.vw), R.lfw + 1):
        if not k[R.season[w]]: continue
        if w > R.modis_last or R.gM[w]: wM[w], wV[w] = 0.0, 1.0
        elif R.gV[w]: wM[w], wV[w] = 1.0, 0.0
        else: wM[w], wV[w] = .5, .5
    wM[R.gM & (np.arange(R.n) < R.vw)] = 0.0
    return wM, wV, k


def monthly_cube(R, method="ratio"):
    """-> (months, cube, nweeks): absolute month index (months since 2000-01), the harmonized index [T, H, W] of each month
    (sum of its complete weeks, rescaled by days_in_month / (7 * weeks) so 4- and 5-week months are comparable) and the number
    of weekly bins per month (< 4 = incomplete edge month). method 'unet' needs res['uc'] (else it silently uses the ratio)."""
    if method == "unet" and R.uw is None: method = "ratio"
    mabs = month_of_week(np.arange(R.n)); m0, m1 = int(mabs[R.first]), int(mabs[R.lfw])
    cube = np.zeros((m1 - m0 + 1, R.H, R.W), np.float32)
    wM, wV, k = week_weights(R, method)
    sel = (R.w >= R.first) & (R.w <= R.lfw)
    np.add.at(cube, (mabs[R.w[sel]] - m0, R.ri[sel], R.rj[sel]), (wM[R.w[sel]] * R.m[sel]))
    if method == "unet":
        s = (R.uw >= R.first) & (R.uw <= R.lfw)
        np.add.at(cube, (mabs[R.uw[s]] - m0, R.uri[s], R.urj[s]), wV[R.uw[s]] * R.uv[s])
    else:
        kk = np.array([x or 1.0 for x in k])[R.season[R.w[sel]]]
        np.add.at(cube, (mabs[R.w[sel]] - m0, R.ri[sel], R.rj[sel]), wV[R.w[sel]] * R.v[sel] / kk)
    months = np.arange(m0, m1 + 1); present = (wM + wV)[R.first:R.lfw + 1] > 0           # weeks with no observation at all do not count as weekly bins
    nw = np.bincount(mabs[R.first:R.lfw + 1][present] - m0, minlength=len(months))
    dim = np.array([pd.Timestamp(2000 + m // 12, m % 12 + 1, 1).days_in_month for m in months])
    cube *= (dim / (7.0 * np.maximum(nw, 1))).astype(np.float32)[:, None, None]
    return months, cube, nw


# ---- evaluation blocks ------------------------------------------------------------------------------------------------------------
def auto_blocks(R, weeks, size=30.0, top=12, min_det=500):
    """When no analysis regions are configured, evaluate on `size`-degree blocks that carry MODIS detections in the evaluation weeks, so errors can be
    split into high- and low-activity regions. -> ({name: box}, {name: 'high'|'mid'|'low'} by tercile of block MODIS totals)."""
    tw = np.zeros(R.n, bool); tw[np.asarray(weeks, int)] = True; e = tw[R.w]
    lat, lon = R.lat_c[R.ri[e]], R.lon_c[R.rj[e]]; bi, bj = np.floor((lat + 90) / size).astype(int), np.floor((lon + 180) / size).astype(int); tot = {}
    for a, b, m in zip(bi, bj, R.m[e]): tot[(a, b)] = tot.get((a, b), 0.0) + float(m)
    keep = sorted([(t, k) for k, t in tot.items() if t >= min_det], reverse=True)[:top]
    boxes, cls = {}, {}
    for r, (t, (a, b)) in enumerate(keep):
        nm = f"block lat {a * size - 90:g}..{a * size - 90 + size:g}, lon {b * size - 180:g}..{b * size - 180 + size:g}"
        boxes[nm] = [a * size - 90, a * size - 90 + size, b * size - 180, b * size - 180 + size]; cls[nm] = "high" if r < len(keep) / 3 else ("low" if r >= 2 * len(keep) / 3 else "mid")
    return boxes, cls
