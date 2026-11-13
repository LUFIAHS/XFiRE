"""python tests/test_forecast_selection.py - v3.3 forecast selection layer on SYNTHETIC series with KNOWN structure.
Checks: sklearn fallback no longer crashes; all candidates are scored on identical rows; nested selection never sees the future; a series without
persistence does not get a skill claim; a persistent series gets the damped-anomaly model; intervals are calibrated pooled over years; the paired bootstrap works.
These tests say nothing about real FIRMS accuracy."""
import numpy as np
from xfire import forecast as F
from xfire.harmonize import _wdates
from xfire.metrics import paired_block_bootstrap

N, FIRST, LAST = 13 * 52 + 40, 20, 13 * 52 + 20


def make(seed, phi, lvl=2000, sigma=0.0):
    r = np.random.default_rng(seed); doy = _wdates(np.arange(N + 40)).dayofyear.to_numpy(); a = 2 * np.pi * (doy - 200) / 365.25; e = np.zeros(N + 40)
    for t in range(1, N + 40): e[t] = phi * e[t - 1] + r.normal(0, .12)
    y = r.poisson(lvl * np.exp(2.2 * (np.cos(a) - 1)) * np.exp(e) + 5).astype(float) * np.exp(r.normal(0, sigma, N + 40))
    return np.concatenate([np.full(FIRST, np.nan), y[FIRST:LAST + 1], np.full(N - LAST - 1, np.nan)])


# 1. scikit-learn fallback accepts LightGBM-style parameters (v3.3-pre crashed with TypeError)
m = F._gbm(n_estimators=40, num_leaves=6, min_child_samples=30, reg_lambda=2.0, subsample=.7, colsample_bytree=.8, learning_rate=.05)
X = np.random.default_rng(0).normal(size=(300, 3)); m.fit(X, X[:, 0] * 2); assert np.isfinite(m.predict(X)).all()

# 2. structure: candidates on identical rows, verdict text, published model label, schema-compatible legacy keys
h = make(11, .94); f = F.forecast(h, FIRST, LAST, detail=True, tune=False); s = f["skill"]; o = f["oof"]
assert set(s["candidates"]) >= {"climatology", "ridge", "v31", "model", "seasonal_naive", "persistence"}
assert f["model"] == s["selected_label"] and s["selected"] in F.CAND_ORDER and isinstance(s["verdict"], str) and s["folds"] >= 3
for k in ("model", "clim", "naive", "score", "bias", "cover", "cover80", "cover50", "mae_h", "clim_h", "tuning", "rmse", "persistence", "coverage_pooled"): assert k in s, k
assert len(s["mae_h"]) == len(s["clim_h"]) == F.H and all(x is not None for x in s["mae_h"])
ref = o["ref"]; assert ref.sum() == s["validated_rows"] and np.isfinite(o["obs"][ref]).all()
for nm in ("cl", "naive", "persistence", "ridge", "model_shrunk", "v31", "pred"): assert np.isfinite(o[nm][ref]).all(), nm            # every model has a value on every compared row
assert abs(s["clim"] - np.abs(o["cl"][ref] - o["obs"][ref]).mean()) < 1e-2
assert (s["selected_by_fold"]["0"], s["selected_by_fold"]["1"]) == ("climatology", "climatology")                                    # no earlier folds -> nothing to select from
sc = s["paired_vs_climatology"]; assert sc["blocks"] >= 3 and sc["lo"] <= sc["mean"] <= sc["hi"]

# 3. the nested selection and every out-of-fold prediction of earlier folds are unaffected by data of the newest fold year
st = o["starts"][-1]; h2 = h.copy(); h2[st:LAST + 1] = h2[st:LAST + 1] * 3 + 500
f2 = F.forecast(h2, FIRST, LAST, detail=True, tune=False); o2 = f2["oof"]; early = o["scored"] & (o["fold"] < len(o["starts"]) - 1)
assert (o2["fold"] == o["fold"]).all() and early.sum() > 1000
for nm in ("pred", "pred_raw", "cl", "ridge", "v31"): assert np.allclose(o[nm][early], o2[nm][early], equal_nan=True), f"future leaks into {nm}"
assert {k: v for k, v in s["selected_by_fold"].items() if int(k) < len(o["starts"]) - 1} == {k: v for k, v in f2["skill"]["selected_by_fold"].items() if int(k) < len(o["starts"]) - 1}

# 4. no persistence beyond climatology -> the published forecast must not claim skill
sel, score, beats = [], [], 0
for seed in range(1, 9):
    k = F.forecast(make(seed, 0.0), FIRST, LAST, tune=False)["skill"]; sel.append(k["selected"]); score.append(k["score"]); beats += k["verdict"].startswith("beats")
print("no-persistence series: published", {x: sel.count(x) for x in set(sel)}, "mean skill", round(float(np.mean(score)), 4), "false 'beats' verdicts", beats, "/ 8")
assert sel.count("climatology") >= 6 and np.mean(score) > -0.02 and beats <= 1

# 5. persistent anomalies -> damped-persistence candidate is used and gives real skill; intervals pooled over years are calibrated
sel, score, cov = [], [], {0.5: [], 0.8: [], 0.9: []}
for seed in range(21, 29):
    k = F.forecast(make(seed, .94), FIRST, LAST, tune=False)["skill"]; sel.append(k["selected"]); score.append(k["score"])
    for lev in cov: cov[lev].append(k["coverage_pooled"][str(lev)]["coverage"])
print("persistent series: published", {x: sel.count(x) for x in set(sel)}, "mean skill", round(float(np.mean(score)), 4), "pooled coverage", {l: round(float(np.mean(v)), 3) for l, v in cov.items()})
assert sum(x != "climatology" for x in sel) >= 5 and np.mean(score) > 0.03
for lev, v in cov.items(): assert abs(np.mean(v) - lev) < 0.07, (lev, np.mean(v))                                                  # pooled over validation years: calibrated within 7 points

# 6. paired block bootstrap: a real gain is detected, no gain is not
r = np.random.default_rng(0); b = np.repeat(np.arange(10), 40)
g = paired_block_bootstrap(r.normal(.5, 1, 400), b); z = paired_block_bootstrap(r.normal(0, 1, 400), b); assert g["lo"] > 0 and z["lo"] < 0 < z["hi"]
print("forecast selection test PASS")
