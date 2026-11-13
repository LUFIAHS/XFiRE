"""Forecast evaluation (python -m xfire --out OUT --evaluate-forecast).

Rolling-origin validation of the published 26-week forecast against seasonal climatology, seasonal-naive and persistence, together with the other
candidates (damped-anomaly ridge, v3.1 trees + ridge, LightGBM + Hyperopt model) on IDENTICAL rows, for the whole extent and for regions (configured
regions, otherwise automatic 30-degree blocks that carry most activity). Everything is computed from the build in OUT; nothing is hard-coded.

Writes forecast_evaluation.md/json, forecast_horizon_errors.csv, forecast_bias_diagnostics.csv, forecast_interval_diagnostics.csv.

Statistical honesty: weekly forecasts from neighbouring origins overlap, so rows are not independent. Differences are tested with a paired block
bootstrap (blocks of 26 origin weeks); coverage is reported pooled over all validation years with a block-bootstrap interval, because the coverage of a
single held-out year is too noisy to verify calibration (see tests/test_forecast.py::calibration).
Hyperopt's inner validation score is a tuning result, never reported here as performance.
"""
import csv, json
from pathlib import Path
import numpy as np
from .forecast import forecast, H, LEVELS, CAND_ORDER, CAND_LABEL
from .harmonize import series_for, box_series, _wdates
from .metrics import paired_block_bootstrap, interval_score

SEAS = ("DJF", "MAM", "JJA", "SON")
LVL = ("low", "mid", "high")


def _wcsv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ["empty"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: ("" if v is None else v) for k, v in r.items()} for r in rows])


def _r(x, d=4): return None if x is None or not np.isfinite(x) else round(float(x), d)


def auto_regions(res, n=6, size=30.0, min_det=20000):
    """Regions for the regional split when none are configured: the `n` most active size x size degree blocks (by raw detections)."""
    tot = {}
    for lat, lon, d in res["cells"]:
        a = np.asarray(d).reshape(-1, 3); key = (int(np.floor((lat + 90) / size)), int(np.floor((lon + 180) / size))); tot[key] = tot.get(key, 0.0) + float(a[:, 1].sum() + a[:, 2].sum())
    keep = sorted(((t, k) for k, t in tot.items() if t >= min_det), reverse=True)[:n]
    return {f"block lat {a * size - 90:g}..{a * size - 90 + size:g}, lon {b * size - 180:g}..{b * size - 180 + size:g}": [a * size - 90, a * size - 90 + size, b * size - 180, b * size - 180 + size] for _, (a, b) in keep}


def evaluate_area(name, h, first, last, tests, evals=30, params=None, tune=True):
    f = forecast(h, first, last, tests=tests, detail=True, tune=tune and params is None, max_evals=evals, params=params)
    if not f or "oof" not in f: return None, [], [], []
    o = f["oof"]; ref, sc = o["ref"], o["scored"]; obs, hh = o["obs"], o["hh"]; seas = (_wdates(o["tg"]).month.to_numpy() % 12) // 3; yr = o["tg_year"]; sk = f["skill"]
    lvl = np.full(len(obs), -1); q = np.quantile(o["cl"][ref], [1 / 3, 2 / 3]); lvl[ref] = np.searchsorted(q, o["cl"][ref])
    models = {"climatology": o["cl"], "seasonal_naive": o["naive"], "persistence": o["persistence"], "ridge": o["ridge"], "v31": o["v31"], "model": o["model_shrunk"], "selected": o["pred"]}
    mae = lambda a, m: _r(np.abs(a[m] - obs[m]).mean(), 3) if m.sum() else None; rmse = lambda a, m: _r(np.sqrt(((a[m] - obs[m]) ** 2).mean()), 3) if m.sum() else None
    rb = lambda a, m: _r((a[m].sum() - obs[m].sum()) / max(obs[m].sum(), 1e-9)) if m.sum() else None
    hr = []
    for k in range(1, H + 1):
        m = ref & (hh == k); row = {"area": name, "horizon_week": k, "n": int(m.sum())}
        for nm, a in models.items(): row[f"mae_{nm}"] = mae(a, m)
        row["rmse_selected"], row["rmse_climatology"] = rmse(models["selected"], m), rmse(models["climatology"], m)
        row["skill_vs_climatology"] = _r(1 - row["mae_selected"] / row["mae_climatology"]) if row["mae_selected"] is not None and row["mae_climatology"] else None
        row["skill_vs_seasonal_naive"] = _r(1 - row["mae_selected"] / row["mae_seasonal_naive"]) if row["mae_selected"] is not None and row["mae_seasonal_naive"] else None
        row["relative_bias_selected"] = rb(models["selected"], m); row["relative_bias_climatology"] = rb(models["climatology"], m); hr.append(row)
    br = []
    groups = [("overall", [("all", ref)]), ("year", [(int(y), ref & (yr == y)) for y in np.unique(yr[ref])]), ("season", [(SEAS[s], ref & (seas == s)) for s in range(4)]),
              ("horizon_block", [(f"{a}-{min(a + 5, H)}", ref & (hh >= a) & (hh <= a + 5)) for a in range(1, H + 1, 6)]), ("activity_level", [(LVL[i], ref & (lvl == i)) for i in range(3)])]
    for gt, gl in groups:
        for g, m in gl:
            if m.sum() == 0: continue
            br.append({"area": name, "group_type": gt, "group": g, "n": int(m.sum()), "mean_observed": _r(obs[m].mean(), 2),
                       **{f"mae_{nm}": mae(a, m) for nm, a in models.items()}, "rmse_selected": rmse(models["selected"], m), "rmse_climatology": rmse(models["climatology"], m),
                       "relative_bias_selected": rb(models["selected"], m), "relative_bias_climatology": rb(models["climatology"], m), "relative_bias_persistence": rb(models["persistence"], m)})
    ir = []
    for lev in LEVELS:
        ins, wid = o["inside"][lev], o["width"][lev]; m0 = sc & np.isfinite(ins)
        for gt, gl in (("pooled_all_folds", [("all", m0)]), ("newest_heldout_year", [("untuned", m0 & o["last_fold"])]), ("horizon_block", [(f"{a}-{min(a + 5, H)}", m0 & (hh >= a) & (hh <= a + 5)) for a in range(1, H + 1, 6)]),
                       ("activity_level", [(LVL[i], m0 & (lvl == i)) for i in range(3)]), ("season", [(SEAS[s], m0 & (seas == s)) for s in range(4)])):
            for g, m in gl:
                if m.sum() == 0: continue
                row = {"area": name, "nominal": lev, "group_type": gt, "group": g, "expected_coverage": lev, "actual_coverage": _r(ins[m].mean()), "mean_width": _r(wid[m].mean(), 2), "n_scored": int(m.sum()),
                       "n_calibration_max": o["n_cal"][lev], "validation_years": int(len(np.unique(yr[m])))}
                if gt == "pooled_all_folds":
                    pb = paired_block_bootstrap(ins[m].astype(float), o["block"][m]); row["coverage_lo90"], row["coverage_hi90"] = (None, None) if pb is None else (_r(pb["lo"]), _r(pb["hi"]))
                ir.append(row)
    summ = {"area": name, "params": f["params"], "tuning": sk.get("tuning"), "selected": sk["selected"], "selected_label": sk["selected_label"], "selected_by_fold": sk["selected_by_fold"], "verdict": sk["verdict"],
            "objective": sk["objective"], "folds": sk["folds"], "validation_years": sk["validation_years"], "validated_rows": sk["validated_rows"], "nested_selection": sk["nested_selection"],
            "candidates": sk["candidates"], "paired_vs_climatology": sk["paired_vs_climatology"], "paired_vs_seasonal_naive": sk["paired_vs_seasonal_naive"], "gain_vs_climatology_by_year": sk["gain_vs_climatology_by_year"],
            "mae": {"selected": sk["model"], "climatology": sk["clim"], "seasonal_naive": sk["naive"], "persistence": sk["persistence"], "ridge": sk["ridge"], "v31": sk["v31"], "model_unselected": sk["raw_model"]},
            "rmse": {"selected": sk["rmse"], "climatology": sk["rmse_clim"]}, "skill_vs_climatology": sk["score"], "skill_vs_seasonal_naive": sk["score_vs_seasonal_naive"], "bias": sk["bias"],
            "coverage_pooled": sk["coverage_pooled"], "cover_newest_year": {"0.5": sk["cover50"], "0.8": sk["cover80"], "0.9": sk["cover"], "n": sk["cover_n"]}, "cover_before_recalibration": {"0.5": sk["cover50_raw"], "0.8": sk["cover80_raw"], "0.9": sk["cover_raw"]},
            "recalibrated": sk["recalibrated"], "forecast_basic": sk["basic"], "selected_on_basic_rows": sk["model_on_basic_rows"]}
    return summ, hr, br, ir


def _md(S, tests, harm):
    L = ["# Six-month forecast: evaluation", "",
         f"Input: harmonized series (method `{harm or 'seasonal ratio'}`). Rolling-origin by origin year, {tests} validation years requested. Every number in one comparison is computed on the SAME rows.",
         "Harmonized values are MODIS-equivalent estimates, so skill describes forecasting of the estimated series, not of raw satellite counts. Forecasts are satellite-detected thermal-anomaly counts, not burned area or confirmed incidents.",
         "Known limitation: the harmonized history used as forecast input was built with ratios / final models fitted on the whole record (including years later than a given backtest origin); this transfers a sensor calibration constant, not fire outcomes, but it is not a fully pseudo-operational backtest.", ""]
    for s in S:
        c = s["candidates"]; pv = s["paired_vs_climatology"] or {}
        L += [f"## {s['area']}", "", f"**Verdict: {s['verdict']}.**", "",
              f"- Published forecast: **{s['selected_label']}** (chosen on all folds; per-fold choices used for scoring: {s['selected_by_fold']}).",
              f"- Scoring rows: {s['validated_rows']} (origin-horizon pairs), nested selection: {s['nested_selection']}; validation years {s['validation_years']}.",
              "", "| model | MAE | RMSE | rel. bias |", "|---|---:|---:|---:|"]
        for nm in ("climatology", "seasonal_naive", "persistence", "ridge", "v31", "model"):
            if nm in c: L.append(f"| {nm} (all calibrated folds) | {c[nm]['mae']} | {c[nm]['rmse']} | {c[nm]['bias']} |")
        L += [f"| **published (nested selection)** | {s['mae']['selected']} | {s['rmse']['selected']} | {s['bias']} |", f"| climatology, same rows | {s['mae']['climatology']} | {s['rmse']['climatology']} | |", "",
              f"- Skill vs climatology: {s['skill_vs_climatology']} (MAE gain {_r(pv.get('mean'), 1)} hotspots/week, 90% block-bootstrap interval {_r(pv.get('lo'), 1)} to {_r(pv.get('hi'), 1)}, {pv.get('blocks')} blocks, better in {pv.get('blocks_positive')} of them). Skill vs seasonal-naive: {s['skill_vs_seasonal_naive']}.",
              f"- MAE gain vs climatology by validation year: {s['gain_vs_climatology_by_year']}",
              f"- Learned candidate: objective `{s['objective']}`; tuning status: {(s['tuning'] or {}).get('status', s['tuning'])}" + (" (Hyperopt actually ran)" if (s['tuning'] or {}).get('tuned_used') else " (Hyperopt did NOT tune this model)") + ". NOTE: any Hyperopt score is an INNER validation (tuning) result, not independent performance; only the rolling-origin rows above are.", ""]
        cp = s["coverage_pooled"] or {}
        L += ["Interval coverage (expected -> measured, pooled over all validation years, 90% block-bootstrap interval):", ""]
        for lev in ("0.5", "0.8", "0.9"):
            if lev in cp: L.append(f"- {float(lev):.0%} -> {cp[lev]['coverage']} ({cp[lev]['lo']} to {cp[lev]['hi']}), mean width {cp[lev]['mean_width']}, interval score {cp[lev]['interval_score']}, n={cp[lev]['n']}, {cp[lev]['blocks']} blocks")
        n_ = s["cover_newest_year"]; L += [f"- Newest held-out year only (one year: noisy; do not use alone): 50% -> {n_['0.5']}, 80% -> {n_['0.8']}, 90% -> {n_['0.9']} (n={n_['n']}). Before recalibration: {s['cover_before_recalibration']}.", ""]
        if cp.get("0.9") and cp["0.9"]["hi"] is not None and cp["0.9"]["hi"] < 0.9: L += ["**The nominal 90% band covers less than 90% even at the upper end of its interval: do not describe these as 90% intervals.**", ""]
        if s["skill_vs_climatology"] is not None and s["skill_vs_climatology"] <= 0: L += ["**The published forecast is not better than seasonal climatology on this area; treat it as a climatology baseline.**", ""]
    L += ["Per-horizon errors, bias by year / season / horizon block / activity level and interval diagnostics are in the three CSV files.",
          "Uncertainty note: intervals cover forecast-model + residual uncertainty only. Harmonization uncertainty (sensor ratio) is estimated separately for the history and is not added."]
    return "\n".join(L) + "\n"


def run(out_dir, tests=6, evals=30, regions=None, auto=6):
    out = Path(out_dir); res = json.loads((out / "results.json").read_text("utf-8")); n, first, lfw = res["nweeks"], res["first_w"], res["last_full_w"]
    areas = {"Whole extent": None, **(res.get("regions") or (regions if regions is not None else auto_regions(res, auto)))}; S, HR, BR, IR = [], [], [], []; whole_params = None
    for name, box in areas.items():
        M, V = box_series(res["cells"], n, box)
        if M.sum() + V.sum() < 2000: continue
        s, hr, br, ir = evaluate_area(name, series_for(res, box), first, lfw, tests, evals, params=whole_params, tune=box is None)   # regions reuse the whole-extent parameters (no re-tuning)
        if s:
            S.append(s); HR += hr; BR += br; IR += ir
            if box is None: whole_params = {k: v for k, v in s["params"].items()}
    _wcsv(out / "forecast_horizon_errors.csv", HR); _wcsv(out / "forecast_bias_diagnostics.csv", BR); _wcsv(out / "forecast_interval_diagnostics.csv", IR)
    (out / "forecast_evaluation.json").write_text(json.dumps({"harmonization": res.get("harm_method"), "uc_method": res.get("uc_method"), "tests": tests, "areas": S}, indent=1, default=float))
    (out / "forecast_evaluation.md").write_text(_md(S, tests, res.get("uc_method")), encoding="utf-8")
    return {"summary": {"areas": [s["area"] for s in S], "verdicts": {s["area"]: s["verdict"] for s in S}, "files": ["forecast_evaluation.md/json", "forecast_horizon_errors.csv", "forecast_bias_diagnostics.csv", "forecast_interval_diagnostics.csv"]}}
