"""XGBoost MODIS-equivalent harmonization (the learned harmonization method of XFiRE).

Target: the MODIS count of a grid cell in a week, for every cell-week in which VIIRS saw fire. Objective: Poisson (count data). The regional block ratio
(VIIRS / k_block) enters as the starting point (base margin), so the trees only learn the multiplicative CORRECTION of that transparent baseline.
Features of a cell-week: VIIRS count (this / previous / next week), VIIRS in the 8 neighbouring cells, the block-ratio estimate, latitude, longitude,
season (sin/cos). No year feature (no extrapolation in time). Cells with no VIIRS detection get 0, exactly like the ratio baselines; a seasonal mass factor (training weeks only) matches the MODIS total.
Training uses only the weeks it is given (chronological folds in harmeval.py); the last 15 % of those weeks (by time) stop the boosting early.
Runs on CPU, or on an NVIDIA GPU (device='cuda'), with modest memory (rows are sampled: XCfg.max_rows).

v3.3: (1) device='auto' only selects CUDA after a tiny training run on the GPU actually succeeded (nvidia-smi alone proves nothing about the installed XGBoost build);
(2) `cell_mask` restricts training to some cells (spatial hold-out); (3) when xgboost is not installed the SAME features / baseline offset / protocol run on
scikit-learn's HistGradientBoostingRegressor (Poisson loss, offset expressed as weight b and target y / b). That method is named `hist_gbm` everywhere, never `xgboost`."""
from dataclasses import dataclass
import numpy as np
from .harmeval import Method
from .harmonize import recent_weights
from .raster import fit_ratio, predict_ratio

import logging
log_ = logging.getLogger("xfire.xgbharm")
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError:
    HistGradientBoostingRegressor = None


@dataclass
class XCfg:
    rounds: int = 400           # max boosting rounds (early stopping decides)
    depth: int = 6
    eta: float = 0.08
    max_rows: int = 1_200_000   # training rows sampled over the training weeks
    device: str = "auto"        # auto | cpu | cuda
    seed: int = 0
    recent_tau: float = 0.0     # recent-year sample weights: half-life in years (0 = off)
    threads: int = 0            # CPU threads for XGBoost (0 = library default)


_DEV = {}


def _cuda_works():
    """True only if a 2-round XGBoost training on the GPU really runs (CUDA build + driver + runtime all present)."""
    if "cuda" in _DEV: return _DEV["cuda"]
    ok = False
    if xgb is not None:
        try:
            d = xgb.DMatrix(np.random.default_rng(0).normal(size=(64, 3)).astype(np.float32), np.arange(64) % 3 + 1.0)
            xgb.train({"objective": "count:poisson", "tree_method": "hist", "device": "cuda", "max_depth": 2}, d, 2); ok = True
        except Exception as e:
            log_.warning("CUDA requested/detected but XGBoost cannot use it (%s: %s); using the CPU.", type(e).__name__, str(e)[:100])
    _DEV["cuda"] = ok; return ok


def pick_device(name="auto"):
    """'cpu' | 'cuda' | 'auto' -> the device that will really be used. CUDA is never assumed: it is tested."""
    if name == "cpu" or xgb is None: return "cpu"
    if name == "cuda": return "cuda" if _cuda_works() else "cpu"
    try:
        import subprocess
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5).returncode != 0: return "cpu"
    except Exception:
        return "cpu"
    return "cuda" if _cuda_works() else "cpu"


def backend_name():
    return "xgboost" if xgb is not None else ("hist_gbm" if HistGradientBoostingRegressor is not None else None)


def _shift(a, di, dj):
    out = np.zeros_like(a); H, W = a.shape
    out[max(di, 0):H + min(di, 0), max(dj, 0):W + min(dj, 0)] = a[max(-di, 0):H + min(-di, 0), max(-dj, 0):W + min(-dj, 0)]; return out


def week_features(R, w, base, cell_mask=None):
    """Rows (cells with VIIRS > 0) of week w: (ri, rj, X[n, 11], baseline estimate). cell_mask: keep only cells inside it (training on a spatial subset)."""
    s, e = R.off[w], R.off[w + 1]; v = R.v[s:e]; ok = v > 0
    if cell_mask is not None: ok = ok & cell_mask[R.ri[s:e], R.rj[s:e]]
    if not ok.any(): return None
    V = R.img(w, "V"); nb = sum(_shift(V, di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0))
    ri, rj = R.ri[s:e][ok], R.rj[s:e][ok]; b = predict_ratio(R, base, w)[ri, rj]
    X = np.column_stack([np.log1p(V[ri, rj]), np.log1p(R.img(w - 1, "V")[ri, rj]), np.log1p(R.img(w + 1, "V")[ri, rj]), np.log1p(nb[ri, rj]), np.log1p(b),
                         R.lat_c[ri] / 90, R.lon_c[rj] / 180, R.sin[w] * np.ones(len(ri)), R.cos[w] * np.ones(len(ri)), np.abs(R.lat_c[ri]) / 90, np.log1p(V[ri, rj]) - np.log1p(b)]).astype(np.float32)
    return ri, rj, X, b


def _fit_booster(X, y, b, sw, nwk, nrows_weeks, cfg):
    """-> (predict_fn(X, b) -> estimate, rounds, device). Chronological early stopping on the last 15 % of the training weeks, for both backends."""
    bm = np.log(np.maximum(b, 0.05)).astype(np.float32)
    cut = max(1, min(nrows_weeks - 1, int(0.85 * nrows_weeks))) if nrows_weeks >= 2 else None
    if xgb is not None:
        prm = {"objective": "count:poisson", "tree_method": "hist", "device": pick_device(cfg.device), "max_depth": cfg.depth, "eta": cfg.eta, "min_child_weight": 20, "subsample": .8, "colsample_bytree": .9,
               "lambda": 5.0, "seed": cfg.seed, "eval_metric": "poisson-nloglik", "max_delta_step": 0.7}
        if cfg.threads: prm["nthread"] = cfg.threads
        if cut is not None:
            tr, va = nwk < cut, nwk >= cut
            bst = xgb.train(prm, xgb.DMatrix(X[tr], y[tr], weight=sw[tr], base_margin=bm[tr]), cfg.rounds, evals=[(xgb.DMatrix(X[va], y[va], base_margin=bm[va]), "val")], early_stopping_rounds=30, verbose_eval=False)
            rounds = int(bst.best_iteration + 1)
        else:
            bst = xgb.train(prm, xgb.DMatrix(X, y, weight=sw, base_margin=bm), cfg.rounds, verbose_eval=False); rounds = int(cfg.rounds)
        return (lambda Xn, bn: bst.predict(xgb.DMatrix(Xn, base_margin=np.log(np.maximum(bn, 0.05)).astype(np.float32)), iteration_range=(0, rounds))), rounds, prm["device"]
    if HistGradientBoostingRegressor is None: raise ImportError("neither xgboost nor scikit-learn is installed (pip install xgboost)")
    bb = np.maximum(b, 0.05).astype(np.float64); r_, w_ = y / bb, sw * bb                  # Poisson with exposure b  <=>  Poisson on y/b with weight b
    mk = lambda it: HistGradientBoostingRegressor(loss="poisson", max_iter=it, learning_rate=cfg.eta, max_depth=cfg.depth, min_samples_leaf=20, l2_regularization=5.0, early_stopping=False, random_state=cfg.seed)
    it_max = max(20, cfg.rounds // 2); rounds = it_max
    if cut is not None:
        tr, va = nwk < cut, nwk >= cut; m0 = mk(it_max).fit(X[tr], r_[tr], sample_weight=w_[tr]); best, bd = it_max, np.inf
        for i, pr in enumerate(m0.staged_predict(X[va]), 1):
            if i % 5 and i != it_max: continue
            mu = np.maximum(pr, 1e-6) * bb[va]; d = float(np.sum(sw[va] * (mu - y[va] * np.log(mu))))
            if d < bd: best, bd = i, d
        rounds = best
    m = mk(rounds).fit(X[nwk < cut] if cut is not None else X, (r_[nwk < cut] if cut is not None else r_), sample_weight=(w_[nwk < cut] if cut is not None else w_))
    return (lambda Xn, bn: np.maximum(m.predict(Xn), 0.0) * np.maximum(bn, 0.05)), int(rounds), "cpu"


def train_xgb(R, weeks, cfg, cell_mask=None):
    """Train the learned harmonizer on `weeks` (and, with cell_mask, only on those cells)."""
    weeks = np.sort(np.asarray(weeks, int)); base = fit_ratio(R, weeks, block=16, cell_mask=cell_mask); rng = np.random.default_rng(cfg.seed); per = max(200, cfg.max_rows // max(len(weeks), 1))
    Xs, Ys, Bs, Ws = [], [], [], []; pw = recent_weights(weeks, cfg.recent_tau)
    for k, w in enumerate(weeks):
        f = week_features(R, int(w), base, cell_mask)
        if f is None: continue
        ri, rj, X, b = f; m = R.img(int(w), "M")[ri, rj]
        if len(ri) > per: i = rng.choice(len(ri), per, replace=False); X, b, m = X[i], b[i], m[i]
        Xs.append(X); Ys.append(m); Bs.append(b); Ws.append(np.full(len(m), 1.0 if pw is None else pw[k], np.float32))
    if not Xs: raise ValueError("learned harmonization has no VIIRS-positive training cell-weeks.")
    X, y, b, sw = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Bs), np.concatenate(Ws)
    nwk = np.concatenate([np.full(len(a), k) for k, a in enumerate(Ys)])                # index of the week that produced each row (chronological)
    predict, rounds, device = _fit_booster(X, y, b, sw, nwk, len(Ys), cfg)
    model = {"predict": predict, "base": base, "rows": int(len(y)), "rounds": rounds, "device": device, "backend": backend_name(), "cal": np.ones(4), "cell_mask": cell_mask}
    # Mass calibration (training weeks, training cells only): the model predicts at cells where VIIRS saw fire, while MODIS also counts fires in cells VIIRS missed. A seasonal
    # factor makes the predicted total match the MODIS total on the training weeks - the same convention the ratio baselines have (k is fitted on all MODIS counts).
    P, Tm = np.zeros(4), np.zeros(4)
    for w in weeks[:: max(1, len(weeks) // 150)]:
        q = R.season[w]; P[q] += float(predict_xgb(R, model, int(w), calibrated=False).sum())
        Tm[q] += float(R.Mw[w]) if cell_mask is None else float(R.img(int(w), "M")[cell_mask & R.valid].sum())
    pooled = Tm.sum() / max(P.sum(), 1e-9); model["cal"] = np.array([Tm[q] / P[q] if P[q] > 0 and Tm[q] > 100 else pooled for q in range(4)])
    return model


def predict_xgb(R, model, w, calibrated=True):
    a = np.zeros((R.H, R.W), np.float32); f = week_features(R, int(w), model["base"])           # prediction is made for ALL cells (held-out cells included)
    if f is None: return a
    ri, rj, X, b = f; a[ri, rj] = model["predict"](X, b) * (float(model["cal"][R.season[w]]) if calibrated else 1.0); return a


def xgb_method(cfg, log=print):
    """The learned harmonizer as a method for harmeval.run_protocol (same folds, weeks and metrics as the baselines). Named after the backend that really runs."""
    nm = backend_name()
    if nm is None: raise ImportError("neither xgboost nor scikit-learn is installed")
    log(f"[{nm}] device {pick_device(cfg.device)}, up to {cfg.rounds} rounds, {cfg.max_rows} sampled training rows" + ("" if nm == "xgboost" else "  (xgboost not installed: scikit-learn Poisson gradient boosting is used and reported as hist_gbm)"))
    return Method(nm, lambda R, w, cell_mask=None: train_xgb(R, w, cfg, cell_mask), lambda m, R, ws: {int(w): predict_xgb(R, m, int(w)) for w in ws}, learned_cell_level=True)
