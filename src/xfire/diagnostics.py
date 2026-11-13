"""Harmonization diagnostics (numpy/pandas only). Answers, from the real data, the questions that decide HOW to harmonize:
  availability / gaps     which weeks have no observations from a sensor (missing, NOT zero fire)
  annual ratio            VIIRS/MODIS per year, season and region + drift test           -> annual_ratio_diagnostics.csv
  sensor comparison       counts, mean/median/variance/percentiles per sensor, satellite, day/night, year and era -> sensor_comparison.csv
  Terra vs Aqua           counts, FRP and ratio by season/year (satellite identity is kept at ingest)
  day vs night            do day and night need their own VIIRS->MODIS factor? (forward-chaining test)
  recent-year weighting   does weighting recent overlap years improve the ratio? (forward-chaining on non-test years only)
  handover                MODIS-only -> overlap: does the harmonized series change level / variance more than MODIS alone does?
Nothing here invents data: every table is computed from the ingested detections (after the same quality filters used for the grid).
"""
import numpy as np
import pandas as pd
from .grid import EPOCH
from .harmonize import _wdates, _season_of, box_series, modis_last, viirs_week, gap_masks, _overlap, _ratio_from_weeks, recent_weights

SEAS = ("DJF", "MAM", "JJA", "SON")


class DiagAccumulator:
    """Counts + FRP sums per (sensor, satellite, day/night, week) over ALL detections that passed the quality filters (global, before the
    region filter and before the satellite-consistency policy, so Terra/Aqua and day/night stay visible)."""
    def __init__(self): self.parts = []

    def add(self, df):
        df = df[df.date >= EPOCH]
        if df.empty: return
        w = ((df.date - EPOCH).dt.days // 7).astype(int)
        self.parts.append(df.assign(w=w).groupby(["sensor", "sat", "dn", "w"]).agg(n=("frp", "size"), frp=("frp", "sum")))

    def table(self):
        if not self.parts: return pd.DataFrame(columns=["sensor", "sat", "dn", "w", "n", "frp"])
        return pd.concat(self.parts).groupby(level=[0, 1, 2, 3]).sum().reset_index()


def _arr(tab, n, sensor, sats=None, dn=None, col="n", wmin=0):
    t = tab[tab.sensor == sensor]
    if sats is not None: t = t[t.sat.isin(sats)]
    if dn is not None: t = t[t.dn == dn]
    t = t[(t.w >= wmin) & (t.w < n)]; a = np.zeros(n); np.add.at(a, t.w.to_numpy(int), t[col].to_numpy(float)); return a


def kept_sats(res):
    """Satellites that feed the grid after the consistency policy: {sensor: list of codes | None (= all)}."""
    p = res.get("sat_policy") or {}; return {"MODIS": None, "VIIRS": [p["viirs_kept"]] if p.get("viirs_kept") else None}, p.get("modis_trim_week") or 0


def observation_gaps(tab, res):
    """Weeks inside a sensor's active span in which a kept satellite delivered NO detections -> res['obs_gaps'] = {sensor: [weeks]}.
    Detections are filtered, so a whole week without any is a missing observation (outage), not a genuinely fire-free planet. Only meaningful for
    --global builds; for region-limited builds the global table is still complete (the accumulator sees every detection before the region filter)."""
    ks, trim = kept_sats(res); lfw = res["last_full_w"]; out = {"MODIS": set(), "VIIRS": set()}; info = {}
    for s in ("MODIS", "VIIRS"):
        t = tab[tab.sensor == s]
        if t.empty: continue
        sats = [c for c in t.sat.unique() if (ks[s] is None or c in ks[s])] or [""]
        for c in sats:
            tt = t[t.sat == c] if c in set(t.sat) else t
            w = np.unique(tt.w.to_numpy(int)); lo = max(int(w.min()), trim if s == "MODIS" else 0); hi = min(int(w.max()), lfw)
            if hi <= lo: continue
            miss = sorted(set(range(lo, hi + 1)) - set(w.tolist())); out[s] |= set(miss); info[f"{s}:{c or 'all'}"] = {"span": [lo, hi], "missing_weeks": len(miss)}
    return {k: sorted(v) for k, v in out.items()}, info


def _kendall(x, y, perms=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x)
    if n < 4: return None, None
    s = np.sign(x[:, None] - x[None, :]) * np.sign(y[:, None] - y[None, :]); tau = float(s[np.triu_indices(n, 1)].sum() / (n * (n - 1) / 2))
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(perms):
        yp = rng.permutation(y); sp = np.sign(x[:, None] - x[None, :]) * np.sign(yp[:, None] - yp[None, :]); cnt += abs(sp[np.triu_indices(n, 1)].sum() / (n * (n - 1) / 2)) >= abs(tau)
    return tau, float((cnt + 1) / (perms + 1))


def _theil_sen(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); i, j = np.triu_indices(len(x), 1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i]))) if len(x) >= 3 else None


def _block_boot_ratio(M, V, weeks, reps=200, seed=0, blk=4):
    if len(weeks) < 2 * blk or M[weeks].sum() <= 0: return None, None
    rng = np.random.default_rng(seed); nb = len(weeks) // blk; b = [weeks[i * blk:(i + 1) * blk] for i in range(nb)]; out = []
    for _ in range(reps):
        idx = np.concatenate([b[i] for i in rng.integers(0, nb, nb)]); m = M[idx].sum()
        if m > 0: out.append(V[idx].sum() / m)
    return (float(np.quantile(out, .05)), float(np.quantile(out, .95))) if out else (None, None)


def annual_ratio(res, boxes):
    """Rows of annual_ratio_diagnostics.csv + drift statistics. Overlap weeks only, weeks with missing observations excluded."""
    n, lfw, ml, vw = res["nweeks"], res["last_full_w"], modis_last(res), viirs_week(res); gM, gV = gap_masks(res); rows, drift = [], {}
    w = _overlap(np.zeros(n), lfw, ml, vw, gM | gV); yrs_all = _wdates(w).year.to_numpy()
    for name, box in boxes.items():
        M, V = box_series(res["cells"], n, box); keep = w[(M[w] > 0) & (V[w] > 0)] if len(w) else w
        if len(keep) == 0: continue
        yrs = _wdates(keep).year.to_numpy(); ys = np.unique(yrs); prev = None; lr = []
        for y in ys:
            wk = keep[yrs == y]; m, v = M[wk].sum(), V[wk].sum(); r = v / m if m > 0 else None; lo, hi = _block_boot_ratio(M, V, wk, seed=int(y))
            seas = []
            for q in range(4):
                ws = wk[_season_of(wk) == q]; ms, vs = M[ws].sum(), V[ws].sum(); seas.append(vs / ms if ms >= 100 and vs > 0 else None)
            rows.append({"region": name, "year": int(y), "weeks": int(len(wk)), "modis": int(m), "viirs": int(v), "ratio": r, "ratio_lo90": lo, "ratio_hi90": hi,
                         **{f"ratio_{SEAS[q]}": seas[q] for q in range(4)}, "change_vs_prev_year_pct": None if not prev or not r else 100 * (r / prev - 1),
                         "partial_year": bool(len(wk) < 40), "flag": ""})
            prev = r
            if r: lr.append((int(y), np.log(r)))
        if len(lr) >= 4:
            x, ly = np.array([a for a, _ in lr], float), np.array([b for _, b in lr]); med = np.median(ly); mad = max(1.4826 * np.median(np.abs(ly - med)), 0.02)
            for rr in [r for r in rows if r["region"] == name and r["ratio"]]:
                z = (np.log(rr["ratio"]) - med) / mad
                if abs(z) > 2.5: rr["flag"] = f"unusual (robust z={z:.1f})"
            tau, p = _kendall(x, ly); slope = _theil_sen(x, np.exp(ly)); rel = None if slope is None else 100 * slope / float(np.median(np.exp(ly)))
            drift[name] = {"years": int(len(lr)), "kendall_tau": tau, "p_value": p, "theil_sen_slope_per_year": slope, "relative_slope_pct_per_year": rel,
                           "drift_detected": bool(p is not None and p < 0.05 and rel is not None and abs(rel) >= 2.0),
                           "note": "ratio drift of >=2%/yr that is statistically clear would make a single fixed ratio unsafe; consider recent-year weighting"}
    return rows, drift


def eras(res):
    """Sensor eras from the processed data: dates, not constants."""
    first, lfw, ml, vw = res["first_w"], res["last_full_w"], modis_last(res), viirs_week(res); d = lambda w: str((EPOCH + pd.Timedelta(days=7 * int(w))).date()); out = []
    if vw > first: out.append({"name": "MODIS-only", "start": d(first), "end": d(min(vw - 1, lfw)), "w0": int(first), "w1": int(min(vw - 1, lfw))})
    if vw <= lfw and ml >= vw: out.append({"name": "MODIS + VIIRS overlap", "start": d(max(vw, first)), "end": d(min(ml, lfw)), "w0": int(max(vw, first)), "w1": int(min(ml, lfw))})
    if ml < lfw: out.append({"name": "VIIRS-dominated (MODIS record ended)", "start": d(max(ml + 1, first)), "end": d(lfw), "w0": int(max(ml + 1, first)), "w1": int(lfw)})
    return out


def _stats(a):
    a = a[np.isfinite(a)]
    if len(a) == 0: return dict(weekly_mean=None, weekly_median=None, weekly_variance=None, weekly_p10=None, weekly_p90=None)
    return dict(weekly_mean=float(a.mean()), weekly_median=float(np.median(a)), weekly_variance=float(a.var(ddof=1)) if len(a) > 1 else None, weekly_p10=float(np.quantile(a, .1)), weekly_p90=float(np.quantile(a, .9)))


def sensor_comparison(tab, res, h):
    """Rows of sensor_comparison.csv: raw per sensor / satellite / day-night (global, after quality filters) and the HARMONIZED weekly series per era."""
    n, lfw, first = res["nweeks"], res["last_full_w"], res["first_w"]; rows = []; yrs = _wdates(np.arange(n)).year.to_numpy(); seen = {}
    for (s, c, dn), g in tab.groupby(["sensor", "sat", "dn"]): seen[(s, c, dn)] = g
    scopes = [(s, None, None) for s in ("MODIS", "VIIRS")] + [(s, c, None) for (s, c) in sorted({(a, b) for a, b, _ in seen})] + sorted(seen)
    for s, c, dn in scopes:
        n_a = _arr(tab, n, s, None if c is None else [c], dn); f_a = _arr(tab, n, s, None if c is None else [c], dn, "frp"); lab = "|".join(str(x) for x in (s, c or "all", dn or "all"))
        w_ok = np.nonzero(n_a > 0)[0]
        if len(w_ok) == 0: continue
        for y in [None] + sorted(set(yrs[w_ok].tolist())):
            sel = np.arange(len(n_a)); sel = sel[(sel >= w_ok.min()) & (sel <= min(w_ok.max(), lfw))]
            if y is not None: sel = sel[yrs[sel] == y]
            if len(sel) == 0: continue
            tot = float(n_a[sel].sum()); rows.append({"scope": lab, "period": "ALL" if y is None else int(y), "detections": int(tot), "weeks": int(len(sel)), "weeks_with_data": int((n_a[sel] > 0).sum()),
                                                      **_stats(n_a[sel]), "frp_mean": float(f_a[sel].sum() / tot) if tot else None})
    for e in eras(res):
        sel = np.arange(e["w0"], e["w1"] + 1); a = h[sel]; rows.append({"scope": "HARMONIZED|" + e["name"], "period": f"{e['start']}..{e['end']}", "detections": int(np.nansum(a)), "weeks": int(len(sel)),
                                                                          "weeks_with_data": int(np.isfinite(a).sum()), **_stats(a), "frp_mean": None})
    return rows


def terra_aqua(tab, res, sensor="MODIS"):
    """Count ratio of the second satellite over the first inside weeks where both report (seasonal, yearly, day/night, drift test). sensor="VIIRS" compares the
    VIIRS platforms (S-NPP vs NOAA-20/21, which the consistency policy keeps apart); the harmonization uses only the kept platform."""
    t = tab[tab.sensor == sensor]; sats = sorted(c for c in t.sat.unique() if c not in ("", "nan"))
    if len(sats) < 2: return {"status": "unavailable", "why": f"the {sensor} files carry no `satellite` column or only one satellite"}
    a, b = sats[0], sats[1]; n, lfw = res["nweeks"], res["last_full_w"]; wa, wb = _arr(t, n, sensor, [a]), _arr(t, n, sensor, [b]); both = (wa > 0) & (wb > 0) & (np.arange(n) <= lfw)
    w = np.nonzero(both)[0]; out = {"status": "ok", "satellites": [a, b], "weeks_with_both": int(len(w)), "ratio_second_over_first_overall": float(wb[w].sum() / max(wa[w].sum(), 1)), "by_season": {}, "by_year": {}}
    fa, fb = _arr(t, n, sensor, [a], col="frp"), _arr(t, n, sensor, [b], col="frp")
    for q in range(4):
        s = w[_season_of(w) == q]
        out["by_season"][SEAS[q]] = {"count_ratio": float(wb[s].sum() / max(wa[s].sum(), 1)), f"frp_mean_{a}": float(fa[s].sum() / max(wa[s].sum(), 1)), f"frp_mean_{b}": float(fb[s].sum() / max(wb[s].sum(), 1))}
    yy = _wdates(w).year.to_numpy()
    for y in np.unique(yy): s = w[yy == y]; out["by_year"][int(y)] = float(wb[s].sum() / max(wa[s].sum(), 1))
    ys = sorted(out["by_year"]); tau, p = _kendall(ys, [out["by_year"][y] for y in ys]) if len(ys) >= 4 else (None, None)
    out["drift_kendall_tau"], out["drift_p_value"] = tau, p
    for dn in ("D", "N"):
        if (t.dn == dn).any(): xa, xb = _arr(t, n, sensor, [a], dn), _arr(t, n, sensor, [b], dn); out[f"ratio_{dn}"] = float(xb[w].sum() / max(xa[w].sum(), 1))
    out["note"] = ("Terra and Aqua are kept apart at ingest. The grid used for harmonization sums both (policy: MODIS used once both observe); orbit drift of either satellite changes overpass times and day/night mix." if sensor == "MODIS" else
                   "VIIRS platforms are kept apart at ingest; only the platform named in sat_policy feeds the grid so the constellation (and its overpass-time mix) stays constant. This table shows how different the dropped platforms are.")
    return out


def recent_weighting(res, boxes, reserved_years, taus=(0.0, 1.0, 2.0, 3.0, 5.0), margin=0.02, reps=1000):
    """Forward-chaining test (train on overlap years < Y, test year Y, years >= 3 of history) of recent-year weighting of the seasonal ratio.
    Reserved test years are excluded entirely. Error = mean over areas of mean weekly |V/k - MODIS| / MODIS (weeks with >= 5 MODIS detections).
    A tau is recommended only if it improves on unweighted by `margin`, wins in >= 2/3 of the test years and the bootstrap 90% interval over years excludes 0."""
    n, lfw, ml, vw = res["nweeks"], res["last_full_w"], modis_last(res), viirs_week(res); gM, gV = gap_masks(res); gap = gM | gV
    w = _overlap(np.zeros(n), lfw, ml, vw, gap); w = w[~np.isin(_wdates(w).year.to_numpy(), list(reserved_years))]; yrs = _wdates(w).year.to_numpy(); uy = np.unique(yrs)
    if len(uy) < 4: return {"status": "skipped", "why": f"only {len(uy)} usable overlap years (need >= 4)", "recommended_tau": 0.0}
    series = {k: box_series(res["cells"], n, b) for k, b in boxes.items()}; err = {t: {} for t in taus}
    for Y in uy[3:]:
        tr, te = w[yrs < Y], w[yrs == Y]
        for t in taus:
            per = []
            for k, (M, V) in series.items():
                kk = _ratio_from_weeks(M, V, tr, recent_weights(tr, t), None); s = _season_of(te); ok = M[te] >= 5
                pred = np.array([V[i] / kk[q] if kk[q] else np.nan for i, q in zip(te, s)])
                if ok.sum() >= 5 and np.isfinite(pred[ok]).all(): per.append(float(np.mean(np.abs(pred[ok] - M[te][ok]) / M[te][ok])))
            err[t][int(Y)] = float(np.mean(per)) if per else np.nan
    ys = [y for y in err[0.0] if np.isfinite(err[0.0][y])]; table = {}; best, rec = 0.0, None; rng = np.random.default_rng(0)
    for t in taus:
        d = np.array([err[t][y] - err[0.0][y] for y in ys]); base = np.mean([err[0.0][y] for y in ys]); m = float(np.mean([err[t][y] for y in ys]))
        boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
        table[str(t)] = {"mean_error": m, "delta_vs_unweighted": float(d.mean()), "lo90": float(np.quantile(boot, .05)), "hi90": float(np.quantile(boot, .95)), "years_improved": int((d < 0).sum()), "test_years": int(len(d))}
        if t > 0 and m < base * (1 - margin) and (d < 0).mean() >= 2 / 3 and table[str(t)]["hi90"] < 0 and (rec is None or m < rec): best, rec = t, m
    return {"status": "ok", "test_years": [int(y) for y in ys], "by_tau": table, "recommended_tau": float(best),
            "why": "weighting improves held-out error clearly and consistently" if best else "no weighting beats the unweighted ratio by the required margin with a clear, consistent gain"}


def day_night(tab, res, reserved_years, margin=0.02):
    """Does one VIIRS->MODIS factor serve both day and night? Forward-chaining: predict MODIS-day from VIIRS-day with (a) the shared seasonal ratio and
    (b) a day-specific ratio; same for night. Satellites follow the consistency policy of the build."""
    if not (tab.dn.isin(["D", "N"]).any()): return {"status": "unavailable", "why": "the FIRMS files have no `daynight` column"}
    n, lfw, ml, vw = res["nweeks"], res["last_full_w"], modis_last(res), viirs_week(res); ks, trim = kept_sats(res); gM, gV = gap_masks(res)
    arrs = {(s, d): _arr(tab, n, s, ks[s], d, wmin=trim if s == "MODIS" else 0) for s in ("MODIS", "VIIRS") for d in ("D", "N")}
    w = _overlap(np.zeros(n), lfw, ml, vw, gM | gV); w = w[~np.isin(_wdates(w).year.to_numpy(), list(reserved_years))]; yrs = _wdates(w).year.to_numpy(); uy = np.unique(yrs)
    Mt, Vt = arrs[("MODIS", "D")] + arrs[("MODIS", "N")], arrs[("VIIRS", "D")] + arrs[("VIIRS", "N")]
    out = {"status": "ok", "overall_ratio": {d: float(arrs[("VIIRS", d)][w].sum() / max(arrs[("MODIS", d)][w].sum(), 1)) for d in ("D", "N")}, "by_season": {}, "forward_test": {}}
    for q in range(4):
        s = w[_season_of(w) == q]; out["by_season"][SEAS[q]] = {d: float(arrs[("VIIRS", d)][s].sum() / max(arrs[("MODIS", d)][s].sum(), 1)) for d in ("D", "N")}
    if len(uy) < 4: out["forward_test"] = {"status": "skipped", "why": "fewer than 4 usable overlap years"}; out["separate_calibration_recommended"] = False; return out
    rng = np.random.default_rng(1); rec = False
    for d in ("D", "N"):
        Md, Vd = arrs[("MODIS", d)], arrs[("VIIRS", d)]; dif = []
        for Y in uy[3:]:
            tr, te = w[yrs < Y], w[yrs == Y]; ok = Md[te] >= 5
            if ok.sum() < 5: continue
            ks_ = _ratio_from_weeks(Mt, Vt, tr, None, None); kd = _ratio_from_weeks(Md, Vd, tr, None, None); s = _season_of(te)
            e = lambda k: np.mean(np.abs(np.array([Vd[i] / k[q] if k[q] else np.nan for i, q in zip(te, s)])[ok] - Md[te][ok]) / Md[te][ok])
            dif.append((e(kd), e(ks_)))
        if dif:
            a = np.array(dif); delta = a[:, 0] - a[:, 1]; boot = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(1000)]
            r = {"error_separate": float(a[:, 0].mean()), "error_shared": float(a[:, 1].mean()), "delta": float(delta.mean()), "lo90": float(np.quantile(boot, .05)), "hi90": float(np.quantile(boot, .95)), "test_years": int(len(delta))}
            r["separate_better"] = bool(r["error_separate"] < r["error_shared"] * (1 - margin) and r["hi90"] < 0); rec |= r["separate_better"]; out["forward_test"][d] = r
    out["separate_calibration_recommended"] = bool(rec)
    out["note"] = "The grid and the XGBoost harmonizer carry no day/night channel: day/night calibration is evaluated for the ratio method only and is NOT applied unless you decide to add it."
    return out


def handover(res, h, M, years=3):
    """Natural experiment around the VIIRS start: compare the `years` before with the `years` after, same weeks of the year. MODIS alone is observed in both
    windows, so any change of the HARMONIZED level beyond MODIS's own change is a handover artefact."""
    first, lfw, ml, vw = res["first_w"], res["last_full_w"], modis_last(res), viirs_week(res); L = int(round(years * 52.18))
    if vw - L < first or vw + L - 1 > min(lfw, ml): return {"status": "skipped", "why": f"needs {years} MODIS-only years before and {years} overlap years after the VIIRS start"}
    a, b = np.arange(vw - L, vw), np.arange(vw, vw + L); ok = lambda x, w_: x[w_][np.isfinite(x[w_])]
    def lvl(x): return float(np.nanmean(x[b]) / max(np.nanmean(x[a]), 1e-9))
    def sd(x, w_):
        v = np.log1p(x[w_]); c = ((_wdates(w_).dayofyear.to_numpy() - 1) // 7).clip(0, 52); m = np.array([np.nanmean(v[c == k]) if (c == k).any() else 0 for k in range(53)]); return float(np.nanstd(v - m[c]))
    lh, lm = lvl(h), lvl(M); rng = np.random.default_rng(0); ca, cb = (np.arange(L) // 4), None
    boot = []
    for _ in range(500):                                              # 4-week blocks resampled in both windows
        nb = L // 4; pick = rng.integers(0, nb, nb); ia = np.concatenate([a[p * 4:(p + 1) * 4] for p in pick]); ib = np.concatenate([b[p * 4:(p + 1) * 4] for p in pick])
        boot.append(np.log((np.nanmean(h[ib]) / max(np.nanmean(h[ia]), 1e-9)) / (np.nanmean(M[ib]) / max(np.nanmean(M[ia]), 1e-9))))
    d = float(np.log(lh / lm)); lo, hi = float(np.quantile(boot, .05)), float(np.quantile(boot, .95))
    return {"status": "ok", "viirs_start": res.get("viirs_start"), "window_years": years, "level_change_harmonized": lh - 1, "level_change_modis_alone": lm - 1,
            "excess_log_change": d, "excess_lo90": lo, "excess_hi90": hi, "artefact_suspected": bool(lo > 0 or hi < 0),
            "log_sd_ratio_harmonized": sd(h, b) / max(sd(h, a), 1e-9), "log_sd_ratio_modis_alone": sd(M, b) / max(sd(M, a), 1e-9),
            "interpretation": "excess_log_change = log[(harmonized after/before) / (MODIS-alone after/before)]; an interval excluding 0 means the harmonization itself shifts the level at the handover. "
                              "After MODIS ends there is no reference sensor, so that second transition cannot be tested this way."}


def compute(res, tab, boxes, reserved_years=(), method=None):
    """Full diagnostics dict (JSON-able) + rows for the CSVs."""
    from .harmonize import series_for
    n = res["nweeks"]; h = series_for(res, None, method or "ratio"); M, _ = box_series(res["cells"], n); rows, drift = annual_ratio(res, boxes); out = {}
    out["annual_ratio_rows"], out["ratio_drift"] = rows, drift
    out["unusual_ratio_years"] = sorted({r["year"] for r in rows if r["flag"] and r["region"] == "Whole extent" and not r["partial_year"]})
    out["sensor_rows"] = sensor_comparison(tab, res, h)
    out["terra_aqua"] = terra_aqua(tab, res); out["viirs_platforms"] = terra_aqua(tab, res, "VIIRS"); out["day_night"] = day_night(tab, res, reserved_years)
    out["recent_weighting"] = recent_weighting(res, boxes, reserved_years); out["handover"] = handover(res, h, M)
    # variance / distribution across eras of the harmonized series vs the raw MODIS series
    out["eras"] = eras(res); out["era_stats"] = {}
    for e in out["eras"]:
        sel = np.arange(e["w0"], e["w1"] + 1); mon = ((_wdates(sel).month.to_numpy() - 1)); hs = h[sel]
        mm = [np.nanmean(hs[mon == k]) for k in range(12) if (mon == k).any() and np.isfinite(hs[mon == k]).any()]
        out["era_stats"][e["name"]] = {**_stats(hs), "seasonal_peak_to_mean": float(np.max(mm) / max(np.mean(mm), 1e-9)) if mm else None}
    return out
