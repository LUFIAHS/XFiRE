"""Stage 3: LightGBM forecasting of the harmonized weekly series (direct multi-horizon, 26 weeks).

Input is the harmonized MODIS-equivalent series (XGBoost / baseline harmonization stage). Hotspot activity = strong seasonal cycle (leakage-free
climatology) x a slowly decaying anomaly; the model learns only the anomaly.
v3.2 changes (see forecast_evaluation.md for the evidence on YOUR data):
- validation is rolling-origin by ORIGIN year: a fold's model is trained only on rows whose TARGET lies before the fold starts (v3.1 selected rows by
  target year, so training rows could contain outcomes that happened after the validation origin).
- candidates compared on identical rows: v3.1 log-scale blend, the same with a bias (smearing) correction learned on EARLIER folds only, and count objectives
  (Tweedie / Poisson with log-climatology offset). The winner is picked on the calibration folds and its held-out skill is reported honestly.
- prediction intervals at 50 / 80 / 90 %: conformal residual quantiles calibrated on earlier folds and scored on a later one (measured coverage, never assumed).
- baselines on the same rows: seasonal climatology, seasonal naive, forecast_basic (v2.1 fallback) and the v3.1 model.
Weekly residuals are autocorrelated, so coverage is approximate; the number of independent validation years is reported.
"""
import logging
import math
import numpy as np
import pandas as pd
from .harmonize import box_series, season_k, harmonize, VIIRS_W, _wdates      # single source of truth (re-exported here)
from .metrics import paired_block_bootstrap, blocks_of, interval_score
try:
    import hyperopt
    from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
except ImportError:
    hyperopt = None
try:
    import lightgbm as lgb
    BACKEND = "LightGBM"
except ImportError:                                  # same model family, scikit-learn implementation (slower, no GPU)
    lgb = None
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        BACKEND = "sklearn HistGradientBoosting"
    except ImportError:
        BACKEND = None

log = logging.getLogger("xfire.forecast")
N_JOBS = None                                    # CPU threads for LightGBM (None = library default); set with set_threads()


def set_threads(n):
    """Limit LightGBM threads (laptops: leave 2 cores free so the machine stays usable). n <= 0 / None = library default."""
    global N_JOBS
    N_JOBS = int(n) if n and int(n) > 0 else None

H = 26                                           # forecast horizon, weeks (approximately six months)
LOOKBACKS = (104, 52, 16)                        # tried in order, longest first


def _names(L):
    """Feature names for lookback L. Year-lag columns exist only if L is long enough."""
    n = ["h", "y0", "y1", "y2", "y3", "m4", "m13"]
    if L >= 52:
        n.append("ly1")
    if L >= 104:
        n.append("ly2")
    return n + ["sin", "cos"]


def _rows(y, doy, origins, horizons, L):
    """Build (X, T, target_week, origin_week). Always returns a 2-D X, even when empty."""
    X, T, tg, to = [], [], [], []
    for t in origins:
        for h in horizons:
            g = t + h; a = 2 * np.pi * doy[g] / 365.25
            row = [h, y[t], y[t-1], y[t-2], y[t-3], y[t-3:t+1].mean(), y[t-12:t+1].mean()]
            if L >= 52:
                row.append(y[g-52])
            if L >= 104:
                row.append(y[g-104])
            row += [np.sin(a), np.cos(a)]
            X.append(row)
            T.append(y[g] if g < len(y) else np.nan); tg.append(g); to.append(t)
    nf = len(_names(L))
    return (np.array(X, float).reshape(-1, nf), np.array(T, float),
            np.array(tg, int), np.array(to, int))


def _fit(X, T, mean_only=False):
    if mean_only:
        if lgb is not None: return {"m": lgb.LGBMRegressor(n_estimators=250, learning_rate=.05, num_leaves=15, min_child_samples=20, subsample=.8, subsample_freq=1, colsample_bytree=.9, verbose=-1, random_state=0).fit(X, T)}
        return {"m": HistGradientBoostingRegressor(max_iter=250, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0).fit(X, T)}
    if lgb is not None:
        p = dict(n_estimators=250, learning_rate=.05, num_leaves=15, min_child_samples=20, subsample=.8,
                 subsample_freq=1, colsample_bytree=.9, verbose=-1, random_state=0)
        return {"m": lgb.LGBMRegressor(**p).fit(X, T),
                "lo": lgb.LGBMRegressor(objective="quantile", alpha=.05, **p).fit(X, T),
                "hi": lgb.LGBMRegressor(objective="quantile", alpha=.95, **p).fit(X, T)}
    p = dict(max_iter=250, learning_rate=.05, max_leaf_nodes=15, min_samples_leaf=20, random_state=0)
    return {"m": HistGradientBoostingRegressor(**p).fit(X, T),
            "lo": HistGradientBoostingRegressor(loss="quantile", quantile=.05, **p).fit(X, T),
            "hi": HistGradientBoostingRegressor(loss="quantile", quantile=.95, **p).fit(X, T)}


def forecast_basic(h, first, last, tests=3, min_rows=300):
    """v2.1 direct multi-horizon model (kept as the fallback for histories too short for the v3 climatology model).
    h: harmonized weekly series (NaN outside data). Returns outlook + rolling-origin skill, or None (reason is logged)."""
    if BACKEND is None:
        log.warning("forecast skipped: neither lightgbm nor scikit-learn is installed.")
        return None
    h = np.asarray(h, float)
    try:
        first, last = int(first), int(last)
    except (TypeError, ValueError):
        log.warning("forecast skipped: first/last week are not integers (%r, %r)", first, last)
        return None
    if first < 0 or last < first or last >= len(h):
        log.warning("forecast skipped: invalid week range first=%s last=%s len(h)=%s", first, last, len(h))
        return None

    n = last + 1 + H
    y = np.full(n, np.nan)
    y[first:last + 1] = np.log1p(np.clip(h[first:last + 1], 0, None))
    doy = _wdates(np.arange(n)).dayofyear.to_numpy()
    span = last - first + 1

    # pick the longest lookback that leaves enough training rows
    L = None
    for cand in LOOKBACKS:
        n_org = max(0, (last - H) - (first + cand) + 1)
        if n_org * H >= min_rows:
            L = cand
            break
    if L is None:
        need = LOOKBACKS[-1] + H + math.ceil(min_rows / H)
        log.warning("forecast skipped: only %d weeks of data (~%.1f years); need at least ~%d weeks.",
                    span, span / 52.18, need)
        return None

    names = _names(L)
    ix = {k: i for i, k in enumerate(names)}
    org = np.arange(first + L, last - H + 1)
    X, T, tg, to = _rows(y, doy, org, range(1, H + 1), L)
    ok = np.isfinite(T) & np.isfinite(X).all(1)
    if ok.sum() < min_rows:
        log.warning("forecast skipped: only %d usable training rows (need %d).", int(ok.sum()), min_rows)
        return None
    if L < LOOKBACKS[0]:
        log.info("short history (%d weeks): using %d-week lookback%s.", span, L,
                 "" if L >= 52 else " without year-lag features")

    # validation folds: last `tests` calendar years, else one chronological hold-out
    years = _wdates(np.arange(n)).year.to_numpy()
    yrs = years[tg]
    ymax = yrs[ok].max()
    min_train = min(400, int(ok.sum() // 2))
    folds = []
    for Y in range(ymax - tests + 1, ymax + 1):
        start = np.searchsorted(years, Y)
        tr, te = ok & (tg < start), ok & (yrs == Y)
        if tr.sum() >= min_train and te.sum() >= 20:
            folds.append((tr, te))
    if not folds:
        cut = last - 26                                   # hold out the final half year
        tr, te = ok & (tg < cut), ok & (to >= cut)
        if tr.sum() >= 100 and te.sum() >= 20:
            folds.append((tr, te))
    if not folds:
        log.warning("forecast skipped: no valid validation fold (history too short).")
        return None

    skill = {"model": [], "naive": [], "clim": [], "cover": []}
    ycols = [ix[k] for k in ("ly1", "ly2") if k in ix]
    for tr, te in folds:
        m = _fit(X[tr], T[tr]); p = m["m"].predict(X[te]); obs = np.expm1(T[te]); Xt = X[te]
        naive = Xt[:, ix["ly1"]] if "ly1" in ix else Xt[:, ix["y0"]]      # seasonal naive, else persistence
        clim = Xt[:, ycols].mean(1) if ycols else Xt[:, ix["m13"]]
        lo, hi = m["lo"].predict(Xt), m["hi"].predict(Xt)
        skill["model"].append(np.abs(np.expm1(p) - obs).mean())
        skill["naive"].append(np.abs(np.expm1(naive) - obs).mean())
        skill["clim"].append(np.abs(np.expm1(clim) - obs).mean())
        skill["cover"].append(float(((T[te] >= np.minimum(lo, p)) & (T[te] <= np.maximum(hi, p))).mean()))

    m = _fit(X[ok], T[ok]); Xf, _, tgf, _ = _rows(y, doy, [last], range(1, H + 1), L)
    mean = m["m"].predict(Xf)
    lo = np.minimum(m["lo"].predict(Xf), mean)               # keep lo <= mean <= hi
    hi = np.maximum(m["hi"].predict(Xf), mean)
    r = lambda a: [round(float(v), 2) for v in np.expm1(a).clip(0)]
    return {"weeks": [int(w) for w in tgf], "mean": r(mean), "lo": r(lo), "hi": r(hi), "lookback": L, "backend": BACKEND,
            "skill": {k: round(float(np.mean(v)), 3) for k, v in skill.items()} | {"folds": len(folds)}}


# =====================================================================================================================
# v3 engine: climatology + anomaly model, honest skill, calibrated bands
# ---------------------------------------------------------------------------------------------------------------------
# Why: hotspot activity is a strong seasonal cycle (known years ahead) plus a slowly decaying ANOMALY (a dry year stays dry).
# v2 asked one gradient-boosting model to learn both from ~25 independent seasons, so it re-learned the season badly and was
# biased low. v3 hands the season to a leakage-free climatology and models only the anomaly:
#     log1p(activity[t+h]) = climatology(week of t+h)  +  anomaly(h)          anomaly = w * GBM + (1 - w) * damped-persistence ridge
# * the blend weight w and the skill numbers come from rolling-origin folds that never saw the validation year
# * the 90 % band is conformal: residual quantiles per horizon group x level tercile of the out-of-fold forecasts, so the stated
#   coverage is a measured coverage (calibrated on earlier years, scored on a later one)
# Falls back to forecast_basic (v2.1) when the history is too short for a climatology.
# =====================================================================================================================
YEAR_W = 365.25 / 7
CLIM_DECAY = 0.93                    # k-th previous year weighs 0.93**(k-1): all years count, recent ones slightly more (trends)
ORIGIN_SKIP = 56                     # first origin = first week + 56, so every origin has >= 1 earlier year of climatology
BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FNAMES = ("h", "an1", "an4", "an13", "an26", "y0", "m4", "m13", "cl", "dcl", "ly1d", "sin", "cos")
CORE = ("an1", "an4", "an13", "cl", "y0")          # rows need these to be finite
LIN = ("an1", "an4", "an13", "an26")


def _smooth3(y):
    f = np.isfinite(y); num = np.convolve(np.where(f, y, 0.0), np.ones(3), "same"); den = np.convolve(f.astype(float), np.ones(3), "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _climatology(y):
    """Seasonal climatology (log scale) of every week: weighted mean of the 3-week-smoothed series at the same week of ALL earlier
    years. cl[u] only uses weeks <= u - 51, so it can be computed once without leaking the future into a training row."""
    n = len(y); ys = _smooth3(y); num = np.zeros(n); den = np.zeros(n)
    for k in range(1, int(n / YEAR_W) + 1):
        r = int(round(YEAR_W * k))
        if r >= n: break
        w = CLIM_DECAY ** (k - 1); seg = ys[:n - r]; f = np.isfinite(seg)
        num[r:] += np.where(f, seg, 0.0) * w; den[r:] += f * w
    return ys, np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def _roll(a, L):
    """Trailing mean over L weeks (NaN-aware, needs >= L/2 finite values); element i covers weeks i-L+1 .. i."""
    f = np.isfinite(a); cs = np.concatenate([[0.0], np.cumsum(np.where(f, a, 0.0))]); cn = np.concatenate([[0.0], np.cumsum(f)])
    out = np.full(len(a), np.nan); i = np.arange(L - 1, len(a)); s = cs[i + 1] - cs[i + 1 - L]; c = cn[i + 1] - cn[i + 1 - L]
    out[i] = np.where(c >= max(1, L // 2), s / np.maximum(c, 1), np.nan)
    return out


def _series_feats(y, doy):
    ys, cl = _climatology(y); a = y - cl
    an26 = _roll(a, 26); an13 = _roll(a, 13)
    return {"y": y, "ys": ys, "cl": cl, "a": a, "an4": _roll(a, 4), "an13": an13, "an26": np.where(np.isfinite(an26), an26, an13),
            "m4": _roll(y, 4), "m13": _roll(y, 13), "doy": doy}


def _rows3(S, origins, horizons):
    """Vectorised feature matrix for (origin, horizon) pairs -> X [rows, len(FNAMES)], target log-activity, climatology at target,
    target week, origin week, horizon."""
    t, hh = np.meshgrid(np.asarray(origins, int), np.asarray(list(horizons), int), indexing="ij"); t, hh = t.ravel(), hh.ravel(); g = t + hh
    ang = 2 * np.pi * S["doy"][g] / 365.25; ly = g - 52
    ly1 = np.where(ly >= 0, S["ys"][np.maximum(ly, 0)], np.nan)
    cols = {"h": hh, "an1": S["a"][t], "an4": S["an4"][t], "an13": S["an13"][t], "an26": S["an26"][t], "y0": S["y"][t], "m4": S["m4"][t],
            "m13": S["m13"][t], "cl": S["cl"][g], "dcl": S["cl"][g] - S["cl"][t], "ly1d": ly1 - S["cl"][g], "sin": np.sin(ang), "cos": np.cos(ang)}
    X = np.column_stack([cols[k] for k in FNAMES]).astype(float)
    T = np.where(g < len(S["y"]), S["y"][np.minimum(g, len(S["y"]) - 1)], np.nan)
    return X, T, S["cl"][g], g, t, hh


_HGB_NAMES = {"n_estimators": "max_iter", "num_leaves": "max_leaf_nodes", "min_child_samples": "min_samples_leaf", "reg_lambda": "l2_regularization", "learning_rate": "learning_rate"}


def _gbm(**over):
    """Gradient-boosted trees with LightGBM parameter names. Without LightGBM the scikit-learn HistGradientBoostingRegressor is used and the
    parameter names are TRANSLATED (v3.3 fix: v3.3-pre passed LightGBM-only names such as n_estimators/num_leaves to scikit-learn and crashed;
    parameters that have no scikit-learn counterpart - subsample / colsample_bytree - are dropped)."""
    p = dict(n_estimators=120, learning_rate=.06, num_leaves=8, min_child_samples=60, reg_lambda=1.0, subsample=.8, subsample_freq=1, colsample_bytree=.9)
    p.update(over)
    if lgb is not None:
        p.update(verbose=-1, random_state=0, force_row_wise=True)
        if N_JOBS: p["n_jobs"] = N_JOBS
        return lgb.LGBMRegressor(**p)
    return HistGradientBoostingRegressor(random_state=0, **{_HGB_NAMES[k]: v for k, v in p.items() if k in _HGB_NAMES})


def _fit3(X, A):
    """Anomaly models on rows X with anomaly target A: gradient boosting (all horizons, h is a feature) + per-horizon ridge on the
    recent-anomaly features (damped persistence)."""
    ix = [FNAMES.index(k) for k in LIN]; hc = FNAMES.index("h"); lin = {}
    for h in range(1, H + 1):
        m = X[:, hc] == h
        if m.sum() < 20: continue
        B = np.column_stack([np.ones(m.sum()), X[m][:, ix]]); R = 3.0 * np.diag([0.0] + [1.0] * len(ix))
        lin[h] = np.linalg.solve(B.T @ B + R, B.T @ A[m])
    return {"gbm": _gbm().fit(X, A), "lin": lin, "ix": ix, "hc": hc}


def _pred3(m, X):
    B = X[:, m["ix"]]; hh = X[:, m["hc"]].astype(int); lin = np.zeros(len(X)); fb = np.mean([c[0] for c in m["lin"].values()]) if m["lin"] else 0.0
    for h in range(1, H + 1):
        s = hh == h
        if s.any(): c = m["lin"].get(h); lin[s] = (c[0] + B[s] @ c[1:]) if c is not None else fb
    return m["gbm"].predict(X), lin


def _qpair(r, alpha=.05):
    n = len(r)
    if n < 30: return None
    s = np.sort(r); return float(s[max(int(np.floor(alpha * (n + 1))) - 1, 0)]), float(s[min(int(np.ceil((1 - alpha) * (n + 1))) - 1, n - 1)])


class _Band:
    """Conformal band in log space: residual quantiles per (horizon group, level tercile) with fallbacks to coarser groups."""
    def __init__(self, p, r, hh, alpha=.05):
        self.edges = np.quantile(p, [1 / 3, 2 / 3]); self.q = {}
        g, b = (hh - 1) * 3 // H, np.searchsorted(self.edges, p)
        self.glob = _qpair(r, alpha) or (float(np.quantile(r, alpha)), float(np.quantile(r, 1 - alpha)))
        for gi in range(3):
            self.q[(gi, None)] = _qpair(r[g == gi], alpha) or self.glob
            for bi in range(3): self.q[(gi, bi)] = _qpair(r[(g == gi) & (b == bi)], alpha) or self.q[(gi, None)]

    def of(self, p, hh):
        g, b = (hh - 1) * 3 // H, np.searchsorted(self.edges, p)
        q = np.array([self.q[(int(a), int(c))] for a, c in zip(g, b)]); return q[:, 0], q[:, 1]


LEVELS = (0.5, 0.8, 0.9)


DEFAULT_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.3, n_estimators=120, learning_rate=.06, num_leaves=8, min_child_samples=60,
                      reg_lambda=1.0, subsample=.8, colsample_bytree=.9, clim_weight=0.0, blend_w=0.5)


def _ridge(X, A):
    """Per-horizon ridge on the recent-anomaly features (damped persistence of the log anomaly): the part of the forecast trees extrapolate badly."""
    ix = [FNAMES.index(k) for k in LIN]; hc = FNAMES.index("h"); lin = {}
    for h in range(1, H + 1):
        m = X[:, hc] == h
        if m.sum() < 20: continue
        B = np.column_stack([np.ones(m.sum()), X[m][:, ix]]); R = 3.0 * np.diag([0.0] + [1.0] * len(ix)); lin[h] = np.linalg.solve(B.T @ B + R, B.T @ A[m])
    return {"lin": lin, "ix": ix, "hc": hc}


def _ridge_pred(m, X):
    B = X[:, m["ix"]]; hh = X[:, m["hc"]].astype(int); out = np.zeros(len(X)); fb = np.mean([c[0] for c in m["lin"].values()]) if m["lin"] else 0.0
    for h in range(1, H + 1):
        s_ = hh == h
        if s_.any(): c = m["lin"].get(h); out[s_] = (c[0] + B[s_] @ c[1:]) if c is not None else fb
    return out


# Separate shrinkage for short-, medium- and long-range forecasts.
# Weights are estimated from earlier out-of-fold years only.
HORIZON_GROUPS = ((1, 5), (6, 13), (14, H))


def _fit_horizon_weights(pred, clim, obs, hh, mask):
    weights = {}
    grid = np.linspace(0.0, 1.0, 21)
    for lo, hi in HORIZON_GROUPS:
        m = mask & (hh >= lo) & (hh <= hi) & np.isfinite(pred) & np.isfinite(clim) & np.isfinite(obs)
        if m.sum() < 30:
            # No validation evidence is not evidence that the tree model is best.
            # Use the climatology fallback until this horizon group has enough rows.
            weights[f"{lo}-{hi}"] = 0.0
            continue
        weights[f"{lo}-{hi}"] = float(min(
            grid,
            key=lambda w: np.abs(w * pred[m] + (1.0 - w) * clim[m] - obs[m]).mean()
        ))
    return weights


def _apply_horizon_weights(pred, clim, hh, weights):
    out = np.asarray(pred, float).copy()
    for lo, hi in HORIZON_GROUPS:
        w = float(weights.get(f"{lo}-{hi}", 1.0))
        m = (hh >= lo) & (hh <= hi)
        out[m] = w * pred[m] + (1.0 - w) * clim[m]
    return np.clip(out, 0.0, None)


def _fit_params(X, T, cl, P):
    """LightGBM with parameters P on top of an OFFSET = log-climatology + ridge persistence. Count objectives (tweedie / poisson) model activity + 1 with that offset
    (prediction exp(offset + f(x)) - 1, correct mean); objective 'log' models the remaining log-anomaly; 'blend' is the v3.1 structure (w trees + (1-w) ridge on the log anomaly). The trees only learn what climatology and persistence miss."""
    kw = {k: v for k, v in P.items() if k not in ("objective", "clim_weight", "tweedie_variance_power", "blend_w")}; rd = _ridge(X, T - cl); off = cl + _ridge_pred(rd, X)
    if P["objective"] == "blend" or lgb is None: return ("blend", _gbm(**kw).fit(X, T - cl), rd)               # v3.1 structure: w * trees + (1 - w) * ridge, both on the log anomaly
    if P["objective"] == "log": return ("log", _gbm(**kw).fit(X, T - off), rd)
    obj = dict(objective=P["objective"]); obj.update({"tweedie_variance_power": P["tweedie_variance_power"]} if P["objective"] == "tweedie" else {})
    return (P["objective"], _gbm(**kw, **obj).fit(X, np.expm1(T) + 1.0, init_score=off), rd)


def _predict_params(model, X, cl, P):
    kind, m, rd = model; off = cl + _ridge_pred(rd, X)
    if kind == "blend": base = np.expm1(cl + float(P.get("blend_w", .5)) * m.predict(X) + (1 - float(P.get("blend_w", .5))) * _ridge_pred(rd, X))
    elif kind == "log": base = np.expm1(off + m.predict(X))
    else: base = np.exp(m.predict(X, raw_score=True) + off) - 1.0
    w = float(P.get("clim_weight", 0.0)); return (1 - w) * np.clip(base, 0, None) + w * np.expm1(cl).clip(0)      # optional shrink toward seasonal climatology


def _tune(X, T, cl, splits, max_evals=30, seed=0, min_gain=0.01):
    """Tune LightGBM parameters with chronological inner splits.

    Hyperopt's best candidate score is kept separate from the score of the
    parameters actually selected. The inner score is a tuning diagnostic,
    never an independent estimate of forecast performance.
    """
    def cast(q):
        q = dict(q)
        for k in ("n_estimators", "num_leaves", "min_child_samples"):
            q[k] = int(round(q[k]))
        return q

    fails = []

    def loss(q):
        q = cast(q)
        scores = []
        try:
            for tr, va in splits:
                pred = _predict_params(_fit_params(X[tr], T[tr], cl[tr], q), X[va], cl[va], q)
                scores.append(float(np.abs(pred - np.expm1(T[va])).mean()))
            score = float(np.mean(scores))
            return score if np.isfinite(score) else 1e18
        except Exception as e:
            fails.append(f"{type(e).__name__}: {str(e)[:120]}")
            return 1e18

    space = {"objective": hp.choice("objective", ["tweedie", "poisson", "log", "blend"]),
             "tweedie_variance_power": hp.uniform("tvp", 1.05, 1.8),
             "num_leaves": hp.quniform("num_leaves", 4, 15, 1),
             "learning_rate": hp.loguniform("learning_rate", np.log(.02), np.log(.12)),
             "n_estimators": hp.quniform("n_estimators", 40, 240, 20),
             "min_child_samples": hp.quniform("mcs", 40, 200, 20),
             "reg_lambda": hp.loguniform("reg_lambda", np.log(.5), np.log(30)),
             "subsample": hp.uniform("subsample", .6, 1.0),
             "colsample_bytree": hp.uniform("colsample", .5, 1.0),
             "clim_weight": hp.uniform("clim_weight", 0.0, .5),
             "blend_w": hp.uniform("blend_w", 0.0, 1.0)}

    base = loss(DEFAULT_PARAMS)
    if base >= 1e17:
        return dict(DEFAULT_PARAMS), {"status": f"tuning skipped: default parameters failed ({fails[0] if fails else 'unknown error'})",
                                     "evals": 0, "failed_trials": len(fails), "tuned_used": False,
                                     "default_mae": None, "tuned_mae": None, "selected_inner_validation_mae": None}

    trials = Trials()
    n_fail0 = len(fails)

    def objective(q):
        score = loss(q)
        return {"loss": score, "status": STATUS_FAIL if score >= 1e17 else STATUS_OK}

    try:
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
                    rstate=np.random.default_rng(seed), show_progressbar=False)
        if not best or not getattr(trials, "best_trial", None):
            raise RuntimeError("Hyperopt produced no successful trial")
        candidate = cast(space_eval(space, best))
        candidate_mae = loss(candidate)
    except Exception as e:
        failed = len(fails) - n_fail0
        log.warning("Hyperopt tuning failed; retaining defaults: %s", e)
        return dict(DEFAULT_PARAMS), {"status": f"Hyperopt failed; default parameters retained ({type(e).__name__})",
                                     "evals": len(trials.trials), "requested_evals": int(max_evals),
                                     "failed_trials": int(failed), "tuned_used": False,
                                     "default_mae": round(base, 3), "tuned_mae": None,
                                     "selected_inner_validation_mae": round(base, 3),
                                     "inner_validation_mae": round(base, 3)}

    candidate_valid = bool(np.isfinite(candidate_mae) and candidate_mae < 1e17)
    use_tuned = bool(candidate_valid and candidate_mae < base * (1 - min_gain))
    selected_mae = candidate_mae if use_tuned else base
    failed = len(fails) - n_fail0
    if failed:
        log.warning("hyperopt: %d of %d trials failed (first: %s)", failed, len(trials.trials), fails[n_fail0])
    if use_tuned:
        status = "tuned with Hyperopt (TPE); candidate beat default"
    elif not candidate_valid:
        status = "Hyperopt TPE produced no valid candidate; default parameters retained"
    else:
        status = "Hyperopt TPE completed; default parameters retained (candidate did not meet minimum gain)"
    candidate_score = round(candidate_mae, 3) if candidate_valid else None
    info = {"status": status, "evals": len(trials.trials), "requested_evals": int(max_evals),
            "inner_splits": len(splits), "default_mae": round(base, 3),
            "tuned_mae": candidate_score,
            "best_candidate_inner_mae": candidate_score,
            "selected_inner_validation_mae": round(selected_mae, 3),
            "inner_validation_mae": round(selected_mae, 3),
            "tuned_used": use_tuned, "failed_trials": int(failed), "minimum_required_gain": float(min_gain),
            "note": "inner validation scores are tuning diagnostics, not independent performance; use rolling-origin results for model evaluation"}
    return (candidate if use_tuned else dict(DEFAULT_PARAMS)), info


def _basic_oof(h, first, last, starts, to_all, hh_all):
    """forecast_basic (v2.1) out-of-fold predictions on the same origins: dict {(origin, horizon): count prediction}. Each fold trains only on rows whose
    target lies before the fold start."""
    out = {}
    if BACKEND is None: return out
    n = last + 1 + H; y = np.full(n, np.nan); y[first:last + 1] = np.log1p(np.clip(np.asarray(h, float)[first:last + 1], 0, None)); doy = _wdates(np.arange(n)).dayofyear.to_numpy()
    L = next((c for c in LOOKBACKS if max(0, (last - H) - (first + c) + 1) * H >= 300), None)
    if L is None: return out
    X, T, tg, to = _rows(y, doy, np.arange(first + L, last - H + 1), range(1, H + 1), L); ok = np.isfinite(T) & np.isfinite(X).all(1); bounds = list(starts) + [last + 2]
    for a, b in zip(bounds[:-1], bounds[1:]):
        tr, te = ok & (tg < a), ok & (to >= a) & (to < b)
        if tr.sum() < 100 or te.sum() == 0: continue
        p = np.expm1(_fit(X[tr], T[tr], True)["m"].predict(X[te])).clip(0)
        for o, g, v in zip(to[te], tg[te] - to[te], p): out[(int(o), int(g))] = float(v)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# v3.3 selection layer
# ---------------------------------------------------------------------------------------------------------------------
# Candidates, simplest first. Each has an out-of-fold prediction on every scored row (models trained only on targets before the fold's first origin):
#   climatology  expanding seasonal climatology (leak-free; log scale -> a median-like level)
#   ridge        climatology x exp(per-horizon ridge on recent log-anomalies): damped persistence, no trees
#   v31          climatology + w * trees + (1 - w) * ridge on the log anomaly (v3.1 structure, w cross-fitted)
#   model        LightGBM (objective + hyper-parameters from Hyperopt) on a climatology + ridge offset, shrunk toward climatology per horizon group
# The PUBLISHED forecast is chosen by NESTED cross-fitted selection: for validation fold f the candidate is picked from folds < f only, and a more complex
# candidate must beat the current best by SWITCH_MARGIN of MAE to displace it. Folds without MIN_CAL earlier folds are scored as climatology (nothing to
# select from). The skill reported for the published forecast is therefore out-of-sample for the selection as well. Non-selected candidates stay reported.
CAND_ORDER = ("climatology", "ridge", "v31", "model")
CAND_LABEL = {"climatology": "seasonal climatology", "ridge": "climatology + damped anomaly persistence (ridge)", "v31": "climatology + anomaly blend (trees + ridge, v3.1)",
              "model": "gradient-boosted anomaly model (shrunk toward climatology)"}


def _label(name, tuning):
    """Candidate label naming the backend that REALLY ran and whether Hyperopt really tuned it (never claim LightGBM / Hyperopt that did not run)."""
    if name != "model": return CAND_LABEL[name]
    tuned = isinstance(tuning, dict) and tuning.get("tuned_used") is True
    return f"{BACKEND or 'gradient-boosted'} anomaly model (shrunk toward climatology)" + (", Hyperopt-tuned" if tuned else ", default parameters")
MIN_CAL = 2                  # earlier folds required before a candidate other than climatology can be selected
SWITCH_MARGIN = 0.01         # relative MAE gain a more complex candidate needs over the current best
ORIGIN_BLOCK = 26            # origin weeks per bootstrap block: forecasts from origins < 26 weeks apart share target weeks, so they are not independent


def _pick(cands, obs, rows):
    """Simplest-first selection on `rows` (cross-fitted by the caller)."""
    best = CAND_ORDER[0]; bm = float(np.abs(cands[best][rows] - obs[rows]).mean())
    for nm in CAND_ORDER[1:]:
        a = cands[nm][rows]; ok = np.isfinite(a)
        if ok.sum() < rows.sum(): continue                       # candidate missing predictions on some rows: not comparable, never selected
        m = float(np.abs(a - obs[rows]).mean())
        if m < bm * (1 - SWITCH_MARGIN): best, bm = nm, m
    return best


def _verdict(pb, selected):
    """Plain-language statement about the published forecast vs climatology from the paired block bootstrap (positive = published forecast better)."""
    if pb is None: return "not validated: too little history for an out-of-sample comparison"
    if pb["lo"] is None: return "inconclusive: fewer than 3 independent validation blocks"
    if pb["lo"] > 0: return "beats seasonal climatology (90% block-bootstrap interval of the MAE gain is above 0)"
    if pb["hi"] < 0: return "WORSE than seasonal climatology (90% interval of the MAE gain is below 0)"
    return "no detectable improvement over seasonal climatology" + (" (the published forecast is the climatology)" if selected == "climatology" else "")


def forecast(h, first, last, tests=6, min_rows=300, params=None, tune=True, max_evals=30, detail=False):
    """h: harmonized weekly series (NaN outside data / for missing weeks). Returns the outlook + honest rolling-origin skill, or None (reason is logged).
    params: LightGBM parameters to reuse (e.g. tuned on the whole extent); tune=True runs Hyperopt on data that precede the backtest. detail=True adds
    out-of-fold arrays (used by forecast_eval.py)."""
    if BACKEND is None:
        log.warning("forecast skipped: neither lightgbm nor scikit-learn is installed."); return None
    h = np.asarray(h, float)
    try: first, last = int(first), int(last)
    except (TypeError, ValueError): return forecast_basic(h, first, last, 3, min_rows)
    if first < 0 or last < first or last >= len(h): return forecast_basic(h, first, last, 3, min_rows)
    n = last + 1 + H
    if ((last - H) - (first + ORIGIN_SKIP) + 1) * H < min_rows:
        log.info("history too short for the v3 climatology model (%d weeks); using the v2.1 model.", last - first + 1); return forecast_basic(h, first, last, 3, min_rows)
    y = np.full(n, np.nan); y[first:last + 1] = np.log1p(np.clip(h[first:last + 1], 0, None))
    wd = _wdates(np.arange(n)); years = wd.year.to_numpy(); S = _series_feats(y, wd.dayofyear.to_numpy())
    org = np.arange(first + ORIGIN_SKIP, last - H + 1); X, T, cl, tg, to, hh = _rows3(S, org, range(1, H + 1)); fi = {k: i for i, k in enumerate(FNAMES)}
    ok = np.isfinite(T) & np.isfinite(X[:, [fi[k] for k in CORE]]).all(1)
    if ok.sum() < min_rows: return forecast_basic(h, first, last, 3, min_rows)
    A = T - cl; yo = years[to]; ymax = yo[ok].max(); min_train = min(400, int(ok.sum() // 3)); folds, starts = [], []
    for Y in range(int(ymax) - tests + 1, int(ymax) + 1):
        st = int(np.searchsorted(years, Y)); tr, te = ok & (tg < st), ok & (yo == Y)          # train: targets known before the fold; validate: origins in year Y
        if tr.sum() >= min_train and te.sum() >= 20: folds.append((tr, te)); starts.append(st)
    if not folds: return forecast_basic(h, first, last, 3, min_rows)
    K = len(folds); fold_of = np.full(len(T), -1)
    # ---- hyper-parameters: Hyperopt on data that precede the first backtest fold (never on the backtest years)
    P = dict(DEFAULT_PARAMS); tuning = {"status": "default parameters"}
    if lgb is None: P["objective"] = "blend"                                         # without LightGBM only the tree + ridge blend exists (reported honestly below)
    if params: P.update(params); tuning = {"status": "parameters reused from the whole-extent tuning"}
    elif tune and hyperopt is not None and lgb is not None:
        pre = ok & (tg < starts[0]); splits = []
        if pre.sum() >= 1500:
            c1, c2 = np.quantile(to[pre], [.55, .78])
            for lo_, hi_ in ((c1, c2), (c2, np.inf)):
                trm, vam = pre & (tg < lo_), pre & (to >= lo_) & (to < hi_)
                if trm.sum() >= 500 and vam.sum() >= 300: splits.append((trm, vam))
        if splits: P, tuning = _tune(X, T, cl, splits, max_evals)
        else: tuning = {"status": "not tuned: too little history before the backtest"}
    elif tune: tuning = {"status": "not tuned: hyperopt / lightgbm not installed"}
    # ---- out-of-fold predictions of every candidate
    pred = np.full(len(T), np.nan); gb, ln = np.full(len(T), np.nan), np.full(len(T), np.nan)
    for f, (tr, te) in enumerate(folds):
        pred[te] = _predict_params(_fit_params(X[tr], T[tr], cl[tr], P), X[te], cl[te], P); fold_of[te] = f
        m3 = _fit3(X[tr], A[tr]); gb[te], ln[te] = _pred3(m3, X[te])
    sc = fold_of >= 0; obs = np.where(sc, np.expm1(np.where(np.isfinite(T), T, 0)), np.nan)
    clim = np.expm1(cl).clip(0); ridge_c = np.expm1(cl + ln).clip(0)
    pl = np.full(len(T), np.nan); w_fold = {}                                         # v3.1 blend weight: each fold uses EARLIER folds only
    for f in range(K):
        te_f, cal_f = sc & (fold_of == f), sc & (fold_of < f)
        w_f = min(BLEND_GRID, key=lambda w: np.abs(np.expm1(cl[cal_f] + w * gb[cal_f] + (1 - w) * ln[cal_f]) - obs[cal_f]).mean()) if cal_f.sum() >= 30 else 0.5
        pl[te_f] = np.expm1(cl[te_f] + w_f * gb[te_f] + (1 - w_f) * ln[te_f]).clip(0); w_fold[f] = w_f
    pred_m = np.full(len(T), np.nan)                                                  # model + horizon-group shrinkage, cross-fitted; no calibration evidence -> climatology
    for f in range(K):
        te_f, cal_f = sc & (fold_of == f), sc & (fold_of < f)
        # Without earlier-fold evidence, default to climatology rather than
        # assigning an unsupported weight of 1.0 to the learned model.
        wf = _fit_horizon_weights(pred, clim, obs, hh, cal_f) if cal_f.sum() >= 30 else {f"{lo}-{hi}": 0.0 for lo, hi in HORIZON_GROUPS}
        pred_m[te_f] = _apply_horizon_weights(pred[te_f], clim[te_f], hh[te_f], wf)
    final_horizon_weights = _fit_horizon_weights(pred, clim, obs, hh, sc)
    cands = {"climatology": clim, "ridge": ridge_c, "v31": pl, "model": pred_m}
    persist = np.expm1(X[:, fi["y0"]]).clip(0)
    nav = np.expm1(np.where(np.isfinite(X[:, fi["ly1d"]]), X[:, fi["ly1d"]] + cl, X[:, fi["y0"]])).clip(0)            # seasonal naive (last year's smoothed week), else persistence
    # ---- nested cross-fitted selection of the published forecast
    pred_adj = np.full(len(T), np.nan); sel_by_fold = {}
    for f in range(K):
        te_f, cal_f = sc & (fold_of == f), sc & (fold_of < f)
        nm = _pick(cands, obs, cal_f) if f >= MIN_CAL else CAND_ORDER[0]; sel_by_fold[f] = nm; pred_adj[te_f] = cands[nm][te_f]
    nested = sc & (fold_of >= MIN_CAL); cal_rows = sc & (fold_of >= 1) if K > 1 else sc
    final_name = _pick(cands, obs, sc) if K > MIN_CAL else CAND_ORDER[0]
    ref = nested if nested.any() else sc
    resid = np.where(sc, np.log1p(obs) - np.log1p(pred_adj), np.nan)
    basic = _basic_oof(h, first, last, starts, to, hh); bp = np.array([basic.get((int(o), int(g)), np.nan) for o, g in zip(to, hh)])
    # ---- intervals: conformal residual quantiles calibrated on EARLIER folds and scored on a LATER one. Raw nominal bands under-cover when the residual spread
    # changes between years, so the internal level is raised in 2-point steps (up to 99 %) until the earlier-year backtest reaches the target; the reported
    # coverage is then measured on the NEWEST held-out year, which that tuning never saw. Without >= 3 folds no tuning is done.
    bnd_rows = {}; in_rows = {lev: np.full(len(T), np.nan) for lev in LEVELS}; wid_rows = {lev: np.full(len(T), np.nan) for lev in LEVELS}
    def fw(q, folds_rng):
        ins, wid, rws, los, his = [], [], [], [], []
        for f in folds_rng:
            c, s_ = sc & (fold_of < f), sc & (fold_of == f)
            if c.sum() < 100 or s_.sum() < 20: continue
            ql, qh = _Band(np.log1p(pred_adj[c]), resid[c], hh[c], (1 - q) / 2).of(np.log1p(pred_adj[s_]), hh[s_]); r_ = resid[s_]; lp = np.log1p(pred_adj[s_])
            ins.append((r_ >= ql) & (r_ <= qh)); lo_, hi_ = np.expm1(lp + ql), np.expm1(lp + qh); wid.append(hi_ - lo_); rws.append(s_); los.append(lo_); his.append(hi_)
        return ins, wid, rws, los, his
    tune_f = range(1, K - 1) if K >= 3 else range(0); chosen_q, cover, cover_raw, cover_last, width, cover_n, last_n = {}, {}, {}, {}, {}, 0, 0
    for lev in LEVELS:
        i0 = fw(lev, range(1, K))[0]; cover_raw[lev] = float(np.concatenate(i0).mean()) if i0 else None; q = lev
        if len(tune_f):
            for cq in [round(float(x), 3) for x in np.arange(lev, .9901, .02)]:
                it = fw(cq, tune_f)[0]; q = cq
                if it and np.concatenate(it).mean() >= lev - 0.01: break
        chosen_q[lev] = q; ins, wid, rws, los, his = fw(q, range(1, K))
        if ins:
            lo_full, hi_full = np.full(len(T), np.nan), np.full(len(T), np.nan)
            for s_, l_, h_ in zip(rws, los, his): lo_full[s_] = l_; hi_full[s_] = h_
            bnd_rows[lev] = (lo_full, hi_full)
            cover[lev] = float(np.concatenate(ins).mean()); width[lev] = float(np.concatenate(wid).mean()); cover_n = int(sum(len(i) for i in ins))
            for s_, i_, w_ in zip(rws, ins, wid): in_rows[lev][s_] = i_; wid_rows[lev][s_] = w_
            if (rws[-1] == (fold_of == K - 1)).all(): cover_last[lev] = float(ins[-1].mean()); last_n = int(len(ins[-1]))
    # ---- final model on all usable rows, forecast from the last observed week ----
    Xf, _, clf, tgf, _, hf = _rows3(S, [last], range(1, H + 1))
    if not np.isfinite(Xf[:, [fi[k] for k in CORE]]).all():
        log.warning("v3 forecast skipped: the last observed weeks have gaps; using the v2.1 model."); return forecast_basic(h, first, last, 3, min_rows)
    if final_name == "climatology": mean = np.expm1(clf).clip(0)
    elif final_name == "ridge": mean = np.expm1(clf + _ridge_pred(_ridge(X[ok], A[ok]), Xf)).clip(0)
    elif final_name == "v31":
        wv = min(BLEND_GRID, key=lambda w: np.abs(np.expm1(cl[sc] + w * gb[sc] + (1 - w) * ln[sc]) - obs[sc]).mean()); gbf, lnf = _pred3(_fit3(X[ok], A[ok]), Xf)
        mean = np.expm1(clf + wv * gbf + (1 - wv) * lnf).clip(0)
    else: mean = _apply_horizon_weights(_predict_params(_fit_params(X[ok], T[ok], cl[ok], P), Xf, clf, P), np.expm1(clf).clip(0), hf, final_horizon_weights)
    pm = np.log1p(mean); bands = {}
    for lev in LEVELS:
        ql, qh = _Band(np.log1p(pred_adj[sc]), resid[sc], hh[sc], (1 - chosen_q[lev]) / 2).of(pm, hf); bands[lev] = (np.minimum(np.expm1(pm + ql).clip(0), mean), np.maximum(np.expm1(pm + qh).clip(0), mean))
    # ---- skill: every number below is computed on IDENTICAL rows for the models it compares
    r2 = lambda a: [round(float(x), 2) for x in a]; rd_ = lambda v, d=3: None if v is None or not np.isfinite(v) else round(float(v), d)
    mae = lambda a, m: float(np.abs(a[m] - obs[m]).mean()); rmse = lambda a, m: float(np.sqrt(((a[m] - obs[m]) ** 2).mean()))
    bias = lambda a, m: float((a[m].sum() - obs[m].sum()) / max(obs[m].sum(), 1e-9)); cv = lambda d, k: None if k not in d else round(d[k], 3)
    hm = lambda a, m: [rd_(np.abs(a[m & (hh == k)] - obs[m & (hh == k)]).mean()) if (m & (hh == k)).any() else None for k in range(1, H + 1)]
    tab = lambda rows, items: {nm: {"mae": rd_(mae(a, rows)), "rmse": rd_(rmse(a, rows)), "bias": rd_(bias(a, rows), 4)} for nm, a in items.items()}
    allc = {**cands, "seasonal_naive": nav, "persistence": persist}
    d_clim = np.abs(clim - obs) - np.abs(pred_adj - obs); d_naive = np.abs(nav - obs) - np.abs(pred_adj - obs); blk = blocks_of(to, ORIGIN_BLOCK)
    pb_clim = paired_block_bootstrap(d_clim[ref], blk[ref]); pb_naive = paired_block_bootstrap(d_naive[ref], blk[ref])
    yrs_ref = sorted({int(x) for x in years[tg][ref]}); per_year = {int(yv): rd_(mae(clim, ref & (years[tg] == yv)) - mae(pred_adj, ref & (years[tg] == yv)), 1) for yv in yrs_ref}
    last_fold = sc & (fold_of == K - 1); cov_pooled = {}
    for lev in LEVELS:                                                               # coverage pooled over ALL folds that had earlier folds to calibrate on, with a block-bootstrap interval
        ins = in_rows[lev]; m = sc & np.isfinite(ins)
        if m.sum() < 50: continue
        pb = paired_block_bootstrap(ins[m].astype(float), blk[m]); lof, hif = bnd_rows[lev]
        cov_pooled[str(lev)] = {"coverage": rd_(pb["mean"]), "lo": rd_(pb["lo"]), "hi": rd_(pb["hi"]), "n": pb["n"], "blocks": pb["blocks"], "mean_width": rd_(np.nanmean(wid_rows[lev][m]), 1),
                                "interval_score": rd_(interval_score(obs[m], lof[m], hif[m], lev), 1)}
    sk = {"model": rd_(mae(pred_adj, ref)), "raw_model": rd_(mae(pred, ref)), "naive": rd_(mae(nav, ref)), "clim": rd_(mae(clim, ref)), "v31": rd_(mae(pl, ref)), "ridge": rd_(mae(ridge_c, ref)), "persistence": rd_(mae(persist, ref)),
          "objective": P["objective"], "score": rd_(1 - mae(pred_adj, ref) / max(mae(clim, ref), 1e-9)), "score_vs_seasonal_naive": rd_(1 - mae(pred_adj, ref) / max(mae(nav, ref), 1e-9)),
          "bias": rd_(bias(pred_adj, ref)), "bias_v31": rd_(bias(pl, ref)), "rmse": rd_(rmse(pred_adj, ref)), "rmse_clim": rd_(rmse(clim, ref)),
          "selected": final_name, "selected_label": _label(final_name, tuning), "selected_by_fold": {str(k): v for k, v in sel_by_fold.items()},
          "verdict": _verdict(pb_clim, final_name), "paired_vs_climatology": pb_clim, "paired_vs_seasonal_naive": pb_naive, "gain_vs_climatology_by_year": per_year,
          "validated_rows": int(ref.sum()), "nested_selection": bool(nested.any()), "min_calibration_folds": MIN_CAL,
          "candidates": tab(cal_rows, allc), "candidate_rows": int(cal_rows.sum()),
          "last_fold_model": rd_(mae(pred_adj, last_fold)) if last_fold.any() else None, "last_fold_clim": rd_(mae(clim, last_fold)) if last_fold.any() else None,
          "last_fold_score": rd_(1 - mae(pred_adj, last_fold) / max(mae(clim, last_fold), 1e-9)) if last_fold.any() else None,
          "horizon_blend_weights": final_horizon_weights,
          "horizon_weighting_note": "Weights are fitted jointly within groups (1-5, 6-13, 14-26); a group weight optimizes pooled group MAE and does not imply that every individual week beats climatology.",
          "cover": cv(cover_last, .9) if .9 in cover_last else cv(cover, .9), "cover80": cv(cover_last, .8) if .8 in cover_last else cv(cover, .8), "cover50": cv(cover_last, .5) if .5 in cover_last else cv(cover, .5),
          "coverage_pooled": cov_pooled, "cover_n": (last_n or cover_n) or None, "cover_raw": cv(cover_raw, .9) if cover_raw.get(.9) is not None else None, "cover80_raw": cv(cover_raw, .8) if cover_raw.get(.8) is not None else None,
          "cover50_raw": cv(cover_raw, .5) if cover_raw.get(.5) is not None else None, "cover_all_folds": cv(cover, .9), "internal_levels": {str(k): v for k, v in chosen_q.items()}, "recalibrated": bool(len(tune_f)),
          "width": None if .9 not in width else round(width[.9], 1), "mae_h": hm(pred_adj, ref), "clim_h": hm(clim, ref), "folds": K, "scored_rows": int(sc.sum()), "validation_years": sorted({int(x) for x in years[tg][sc]}),
          "basic": rd_(mae(bp, sc & np.isfinite(bp))) if (sc & np.isfinite(bp)).sum() > 50 else None, "model_on_basic_rows": rd_(mae(pred_adj, sc & np.isfinite(bp))) if (sc & np.isfinite(bp)).sum() > 50 else None, "tuning": tuning}
    shown = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P.items()}
    out = {"weeks": [int(x) for x in tgf], "mean": r2(mean), "lo": r2(bands[0.9][0]), "hi": r2(bands[0.9][1]), "lo80": r2(bands[0.8][0]), "hi80": r2(bands[0.8][1]), "lo50": r2(bands[0.5][0]), "hi50": r2(bands[0.5][1]),
           "lookback": ORIGIN_SKIP, "backend": BACKEND, "model": _label(final_name, tuning), "params": shown, "skill": sk}
    if detail:
        out["oof"] = {"fold": fold_of, "scored": sc, "to": to, "tg": tg, "hh": hh, "obs": obs, "pred": pred_adj, "pred_raw": pred, "cl": clim, "naive": nav, "persistence": persist, "ridge": ridge_c, "model_shrunk": pred_m, "basic": bp, "v31": pl,
                      "resid": resid, "starts": starts, "yo": yo, "inside": in_rows, "width": wid_rows, "last_fold": fold_of == K - 1, "nested": nested, "ref": ref, "cal_rows": cal_rows,
                      "n_cal": {lev: int(((sc & (fold_of < K - 1))).sum()) for lev in LEVELS}, "tg_year": years[tg], "block": blk}
    return out
