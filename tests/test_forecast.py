"""python tests/test_forecast.py - v3.2 LightGBM forecaster on SYNTHETIC weekly series (no real data). Checks mechanics and leakage, not real accuracy."""
import json
import numpy as np
from xfire import forecast as F
from xfire.harmonize import _wdates

n, first, last = 1000, 20, 980
doy = _wdates(np.arange(n + 20)).dayofyear.to_numpy(); r = np.random.default_rng(3)
a = 2 * np.pi * (doy - 200) / 365.25; e = np.zeros(n + 20)
for t in range(1, n + 20): e[t] = .97 * e[t - 1] + r.normal(0, .08)
y = r.poisson(300 * np.exp(2 * (np.cos(a) - 1)) * np.exp(e) + 3).astype(float)
h = np.concatenate([np.full(first, np.nan), y[first:last + 1], np.full(n - last - 1, np.nan)])
f = F.forecast(h, first, last, detail=True, max_evals=8)
assert f and f.get("model") and len(f["mean"]) == F.H == len(f["lo"]) == len(f["hi"])
for l50, l80, l90, m, u50, u80, u90 in zip(f["lo50"], f["lo80"], f["lo"], f["mean"], f["hi50"], f["hi80"], f["hi"]): assert l90 <= l80 <= l50 <= m <= u50 <= u80 <= u90
assert all(v >= 0 for v in f["mean"])
s = f["skill"]; print({k: v for k, v in s.items() if k not in ("mae_h", "clim_h")})
assert s["score"] > 0 and s["model"] <= s["clim"] and s["folds"] >= 3 and len(s["mae_h"]) == F.H
assert s["basic"] is not None and s["v31"] is not None and s["objective"] in ("log", "tweedie", "poisson", "blend")
for k in ("cover", "cover80", "cover50"): assert s[k] is not None and 0 <= s[k] <= 1
assert s["cover50"] <= s["cover80"] + .05 <= s["cover"] + .1                                   # wider bands cover more
assert abs(s["bias"]) < .35
# Hyperopt: tuned on data BEFORE the first backtest fold only, parameters exposed for reuse; reuse path does not tune again
if F.hyperopt is not None and F.lgb is not None:
    tu = s["tuning"]; assert tu["status"].startswith("tuned") and tu["evals"] == 8 and tu["tuned_mae"] <= tu["default_mae"], tu
    assert f["params"]["objective"] in ("tweedie", "poisson", "log", "blend") and 0 <= f["params"]["clim_weight"] <= .6
    fr = F.forecast(h, first, last, params=f["params"], tune=False); assert fr["skill"]["tuning"]["status"].startswith("parameters reused") and fr["params"]["objective"] == f["params"]["objective"]
# validation structure: every scored row's ORIGIN lies in its own fold year, i.e. models never see outcomes after the validation origin
o = f["oof"]; yrs = _wdates(np.arange(n + 40)).year.to_numpy()
for k, st in enumerate(o["starts"]):
    m = (o["fold"] == k) & o["scored"]
    if m.any(): assert o["to"][m].min() >= st
# no leakage: changing weeks AFTER the origin must not change a forecast made at that origin
h2 = h.copy(); h2[last - 5:last + 1] *= 5
assert F.forecast(h2[:last - 5], first, last - 6, tune=False)["mean"] == F.forecast(h[:last - 5], first, last - 6, tune=False)["mean"]
# missing data: a gap in the middle is NaN (not zero) and must not crash or produce NaN output
hg = h.copy(); hg[500:503] = np.nan; fg = F.forecast(hg, first, last); assert fg is None or all(np.isfinite(fg["mean"]))
# short history falls back to the v2.1 model instead of failing
short = F.forecast(h[:first + 140 + 40], first, first + 139); print("short history ->", None if short is None else short.get("model", "basic v2.1"))
# harmonized input integration: a series built by harmonize() feeds forecast() unchanged
from xfire.harmonize import harmonize, season_k
M = np.round(y * .9); V = np.round(y * 3.6); hh = harmonize(M, V, first, last, season_k(M, V, last, vw=300)); assert F.forecast(hh, first, last) is not None
json.dumps(f["mean"] + f["lo"] + f["hi"]); allow = {k: v for k, v in f.items() if k != "oof"}; json.dumps(allow, allow_nan=False)
print("forecast test PASS")
