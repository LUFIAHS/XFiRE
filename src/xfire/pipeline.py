"""Build pipeline.  FIRMS CSVs -> weekly cell grid -> [XGBoost harmonization] -> LightGBM forecasts -> rule-based fire clusters -> dashboard.

The XGBoost stage is optional and can never break a build: if xgboost is missing, or the stage fails, the build falls back to the
transparent baseline (seasonal VIIRS/MODIS ratio) and says so in results.json.
"""
import json, math, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_regions, VIIRS_NOMINAL_START
from .ingest import iter_firms
from .grid import GridAccumulator, EPOCH
from .dashboard import write_dashboard
from .harmonize import box_series, series_for, modis_last, viirs_week, _wdates
from .forecast import forecast
from . import hotspots as hs, diagnostics as dg, harmeval as he

VERSION = "3.3.0"


# ---------------------------------------------------------------------------------------------- areas / forecasts
def _areas(res):
    """Harmonized weekly series (+ activity-weighted centroid) for the whole extent and configured analysis area,
    using the harmonization method the build chose (res['harm_method'])."""
    n, areas = res["nweeks"], {}
    for key, box in {"Whole extent": None, **res["regions"]}.items():
        M, V = box_series(res["cells"], n, box)
        if M.sum() + V.sum() < 2000: continue
        sel = [c for c in res["cells"] if not box or (box[0] <= c[0] <= box[1] and box[2] <= c[1] <= box[3])]
        wt = np.array([sum(c[2][1::3]) + sum(c[2][2::3]) for c in sel], float)
        areas[key] = (series_for(res, box), float(np.average([c[0] for c in sel], weights=wt)), float(np.average([c[1] for c in sel], weights=wt)))
    return areas


def stage_forecast(res, out, log=print):
    """LightGBM + Hyperopt 26-week outlook for the whole extent and configured analysis areas. A failure here never aborts the build.
    Hyperopt tunes on data that precede the backtest; the whole-extent parameters are stored (res['forecast_params']) and reused for drawn areas in the server."""
    res["forecast"] = {}; lfw = res["last_full_w"]; cfg = res.get("fc_cfg") or {"evals": 30, "tune": True}
    for key, (h, _, _) in _areas(res).items():
        f = forecast(h, res["first_w"], lfw, tune=cfg["tune"], max_evals=cfg["evals"])
        if f:
            res["forecast"][key] = f; t = f["skill"].get("tuning", {}); log(f"[forecast] {key}: {t.get('status')} -> objective {f['skill'].get('objective')}, skill vs climatology {f['skill'].get('score')}")
            if key == "Whole extent": res["forecast_params"] = f.get("params")
    return None


# ---------------------------------------------------------------------------------------------- harmonization (baselines + optional XGBoost)
def reserved_years(res, test_years=1):
    """Newest `test_years` COMPLETE overlap years (>= 40 weeks with both sensors observed): the reserved final test set. Same rule as harmeval.run_protocol."""
    n = res["nweeks"]; M, V = box_series(res["cells"], n); lfw, ml, vw = res["last_full_w"], modis_last(res), viirs_week(res)
    from .harmonize import gap_masks, _overlap
    gM, gV = gap_masks(res); w = _overlap(M, lfw, ml, vw, gM | gV); w = w[(M[w] > 0) & (V[w] > 0)]
    if len(w) == 0 or test_years <= 0: return []
    yrs = _wdates(w).year.to_numpy(); full = [int(y) for y in np.unique(yrs) if (yrs == y).sum() >= 40]
    return full[-test_years:] if len(full) >= test_years + 4 else []


def _boxes(res): return {"Whole extent": None, **{k: v for k, v in res["regions"].items()}}


def stage_diagnostics(res, out, tab, hcfg, recent_tau="auto", log=print):
    """Sensor diagnostics; also decides the recent-year weighting of the seasonal ratio (evidence on non-test years only)."""
    res["recent_tau"] = 0.0; ry = reserved_years(res, hcfg.test_years); boxes = _boxes(res)
    rw = dg.recent_weighting(res, boxes, ry)
    res["recent_tau"] = float(rw.get("recommended_tau", 0.0)) if recent_tau == "auto" else (0.0 if recent_tau in ("off", None) else float(recent_tau))
    diag = dg.compute(res, tab, boxes, ry, "ratio"); diag["recent_weighting"] = rw; diag["recent_tau_applied"] = res["recent_tau"]
    log(f"[diagnostics] recent-year weighting: {rw.get('status')} -> half-life applied to the ratio: {res['recent_tau']} y ({rw.get('why', '')})")
    return diag


def stage_harmonization(res, out, hcfg, xcfg=None, log=print, force=None, use_xgb=True):
    """Compare seasonal ratio / regional block ratio / XGBoost on identical held-out weeks and select the default.
    Fills res['uc'] (cell-level estimates of the selected non-ratio method), res['uc_method'], res['harm_method'] ('ratio' | 'unet' = cell-level estimate,
    the internal name of the learned/regional path), res['harm_eval'] (protocol report) and res['xgboost'] (availability)."""
    res.pop("uc", None); res["harm_method"] = "ratio"; res["uc_method"] = None; extra = []
    if not use_xgb: res["xgboost"] = {"status": "disabled"}
    else:
        try:
            from . import xgbharm
            xcfg = xcfg or xgbharm.XCfg(); extra = [xgbharm.xgb_method(xcfg, log)]; bk = xgbharm.backend_name()
            res["xgboost"] = {"status": "ok", "backend": bk, "cfg": vars(xcfg), "device": xgbharm.pick_device(xcfg.device) if bk == "xgboost" else "cpu",
                              **({} if bk == "xgboost" else {"note": "xgboost is not installed: the learned method is scikit-learn Poisson gradient boosting, reported as hist_gbm (pip install xgboost to use XGBoost)"})}
        except ImportError as e: res["xgboost"] = {"status": "unavailable", "why": f"{e}"}
    try:
        info, uc = he.run_protocol(res, out, extra, hcfg, force, log)
    except Exception as e:
        info, uc = {"status": "failed", "why": f"{type(e).__name__}: {str(e)[:300]}"}, None; log(f"[harmonization] failed: {info['why']}")
    res["harm_eval"] = info
    if info.get("status") == "ok":
        applied = info["applied"]; res["uc_method"] = applied if uc is not None else None
        if uc is not None: res["uc"] = uc; res["harm_method"] = "unet"; res["harm_unc"] = {"log_sd_by_season": info["log_sd_by_season"], "basis": "held-out log error of the selected method, whole extent"}
        log(f"[harmonization] recommended: {info['chosen']} ({info['why']})" + (f"; applied: {applied}" if info["forced"] else ""))
    else: log(f"[harmonization] {info.get('status')}: {info.get('why')}")
    return info


def evaluate_harmonization(out_dir, log=print):
    """Recompute diagnostics for an existing build and (re)write the harmonization reports. Does not retrain anything."""
    import pandas as pd
    out, res = _load(out_dir); tab = pd.read_csv(out / "diag_counts.csv.gz", keep_default_na=False)
    hcfg = he.HCfg(**(res.get("hcfg") or {})); diag = stage_diagnostics(res, out, tab, hcfg, res.get("recent_tau_mode", "auto"), log)
    diag2 = dg.compute(res, tab, _boxes(res), reserved_years(res, hcfg.test_years), res.get("harm_method") or "ratio"); diag2["recent_weighting"] = diag["recent_weighting"]
    he.write_outputs(out, res, res.get("harm_eval") or {"status": "not run"}, diag2); res["diagnostics"] = _slim(diag2); _write(res, out)
    return {"summary": {"files": ["harmonization_evaluation.md/json", "annual_ratio_diagnostics.csv", "sensor_comparison.csv", "harmonization_model_metrics.csv"], "recent_tau": res["recent_tau"]}}


def _slim(d): return {k: v for k, v in d.items() if k not in ("annual_ratio_rows", "sensor_rows")}


# ---------------------------------------------------------------------------------------------- fire clusters (rule-based)
def stage_events(res, out, log=print):
    """Fire clusters per complete month with the transparent rule detector (no learned model)."""
    from .raster import Raster, monthly_cube
    from . import events as ev
    R = Raster(res); method = res.get("harm_method", "ratio")
    months, cube, nw = monthly_cube(R, method)
    ok = nw >= 4
    if ok.sum() == 0:
        res["events"], res["events_report"] = {}, {"status": "skipped", "why": "no complete month"}; return
    clim, has = ev.loyo_climatology(cube, months, ok); th = ev.thresholds(cube, ok)
    rule = ev.detect_months(R, cube, months, nw, clim, has, th, ev.rule_events, np.nonzero(ok)[0])
    res["events"] = {"rule": rule}
    res["events_report"] = {"thresholds": {k: round(float(v), 2) for k, v in th.items()}, "months": int(ok.sum()), "method": method,
                            "rule": {"status": "ok", "clusters": int(sum(len(v) for v in rule.values())),
                                     "anomalous": int(sum(e["c"] for v in rule.values() for e in v))}}


# ---------------------------------------------------------------------------------------------- build
def _viirs_start(res, qc):
    """Real VIIRS start: first full week of the kept VIIRS satellite in the processed grid (never a constant)."""
    n, lfw = res["nweeks"], res["last_full_w"]; _, V = box_series(res["cells"], n); nz = np.nonzero(V[:lfw + 1] > 0)[0]
    if len(nz) == 0: return None
    kept = (res.get("sat_policy") or {}).get("viirs_kept"); sats = (qc.get("VIIRS") or {}).get("satellites") or {}
    d0 = (sats.get(kept) or {}).get("first") if kept in sats else (qc.get("VIIRS") or {}).get("first")
    w = max(int(nz.min()), math.ceil((pd.Timestamp(d0) - EPOCH).days / 7) if d0 else 0)
    return str((EPOCH + pd.Timedelta(days=7 * w)).date())


def run(modis_paths, viirs_paths, out_dir, regions_path=None, cell=0.5, keep_all=False,
        xgb_cfg=None, use_xgb=True, harmonize_method=None, keep_hotspots=True, log=print,
        veg_only=True, sat_policy="auto", viirs_satellite=None, trim_modis=True, hcfg=None, recent_tau="auto", fc_evals=30, tune_forecast=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); t0 = time.time(); hcfg = hcfg or he.HCfg()
    regions = load_regions(regions_path)
    acc = GridAccumulator(cell, None if keep_all else list(regions.values())); dacc = dg.DiagAccumulator()
    store = hs.HotspotStore(out / "hotspots") if keep_hotspots else None
    qc = {"MODIS": {}, "VIIRS": {}}
    for sensor, paths in (("MODIS", modis_paths), ("VIIRS", viirs_paths)):
        for chunk in iter_firms(paths, sensor, qc[sensor], veg_only=veg_only, dedupe_dir=out / "_dedupe" / sensor):
            dacc.add(chunk)                                                      # satellite / day-night / weekly availability, before region + satellite policy
            kept = acc.add(chunk)
            if store is not None: store.add(kept)
    grid = acc.export(sat_policy, viirs_satellite, trim_modis)                   # applies the satellite-consistency policy
    if store is not None: store.close(acc.policy_report)
    res = {"product": "XFiRE", "version": VERSION, "keep_all": bool(keep_all),
           "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
           "regions": {k: list(v) for k, v in regions.items()}, "quality": qc, "sat_policy": acc.policy_report, **grid}
    # last COMPLETE week: the newest week is partial unless data reach its final day
    lfw = res["last_w"] if (pd.Timestamp(res["asof"]) - (EPOCH + pd.Timedelta(days=7 * res["last_w"]))).days >= 6 else res["last_w"] - 1
    res["last_full_w"] = lfw
    M, _ = box_series(res["cells"], res["nweeks"]); nz = np.nonzero(M[:lfw + 1] > 0)[0]
    res["modis_last_w"] = int(nz.max()) if len(nz) else -1                       # the MODIS record ends here (sensor retired)
    res["viirs_start"] = _viirs_start(res, qc)                                   # real transition date, read from the data
    tab = dacc.table(); tab.to_csv(out / "diag_counts.csv.gz", index=False)
    res["obs_gaps"], res["obs_gap_info"] = dg.observation_gaps(tab, res)         # weeks without observations: MISSING, not zero
    res["eras"] = dg.eras(res); res["hcfg"] = vars(hcfg); res["recent_tau_mode"] = recent_tau
    res["data_warnings"] = data_warnings(res)
    for w_ in res["data_warnings"]: log(f"[data] WARNING: {w_}")
    res["hotspots"] = {"available": store is not None, "rows": int(store.total) if store is not None else 0}
    res["harm_method"] = "ratio"; res["forecast"] = {}; res["recent_tau"] = 0.0
    res["seconds"] = round(time.time() - t0, 1)
    _write(res, out); log(f"[build] baseline dashboard written: {out / 'dashboard.html'} (optional stages continue)")   # usable HTML even if a later stage is slow/fails
    diag = {}
    try:
        diag = stage_diagnostics(res, out, tab, hcfg, recent_tau, log)
    except Exception as e: log(f"[diagnostics] failed: {type(e).__name__}: {e}")
    info = stage_harmonization(res, out, hcfg, xgb_cfg, log, harmonize_method, use_xgb)
    try:
        diag2 = dg.compute(res, tab, _boxes(res), reserved_years(res, hcfg.test_years), res["harm_method"]); diag2["recent_weighting"] = diag.get("recent_weighting"); diag2["recent_tau_applied"] = res["recent_tau"]
        he.write_outputs(out, res, info, diag2); res["diagnostics"] = _slim(diag2)
        new = [w_ for w_ in data_warnings(res) if w_ not in res["data_warnings"]]; res["data_warnings"] += new
        for w_ in new: log(f"[data] WARNING: {w_}")
    except Exception as e: log(f"[diagnostics] report failed: {type(e).__name__}: {e}")
    try:
        res["fc_cfg"] = {"evals": fc_evals, "tune": tune_forecast}; stage_forecast(res, out, log)
    except Exception as e:                                                       # forecasts are an add-on too
        res["forecast"] = {}; log(f"[forecast] failed: {type(e).__name__}: {e}")
    try:
        stage_events(res, out, log)
    except Exception as e:                                                       # events are an add-on: never lose the build
        res["events"], res["events_report"] = {}, {"status": "failed", "why": f"{type(e).__name__}: {str(e)[:300]}"}; log(f"[events] failed: {e}")
    res["seconds"] = round(time.time() - t0, 1)
    _write(res, out)
    return res


def data_warnings(res):
    """Plain-language checks that explain a surprising calendar: a late VIIRS start, missing weeks, abrupt year-to-year level shifts of one sensor."""
    out, q = [], res.get("quality", {})
    v = res.get("viirs_start")
    if v and (pd.Timestamp(v) - pd.Timestamp(VIIRS_NOMINAL_START)).days > 56:
        out.append(f"VIIRS data in the files start {v}, much later than the nominal {VIIRS_NOMINAL_START}; the harmonization uses the real start ({v}), so the MODIS-only period is longer and the overlap shorter.")
    og = res.get("obs_gaps") or {}
    for s, w in og.items():
        if w: out.append(f"{s}: {len(w)} week(s) inside the record have no detections at all (first at {str((EPOCH + pd.Timedelta(days=7 * w[0])).date())}). They are treated as MISSING observations, not as zero fire activity.")
    for s, d in q.items():
        yrs = sorted((int(y), n) for y, n in (d.get("by_year") or {}).items())[1:-1]            # first / last year may be partial
        for (y0, n0), (y1, n1) in zip(yrs, yrs[1:]):
            if n0 > 1000 and not .45 < n1 / n0 < 2.2:
                out.append(f"{s}: detections change {n1 / n0:.1f}x between {y0} and {y1} - check for a satellite/product change in the files (see --satellite-policy).")
        fr = sorted((v[0], v[1], k) for k, v in (d.get("file_ranges") or {}).items())
        for (a0, a1, an), (b0, b1, bn) in zip(fr, fr[1:]):
            if b0 <= a1: out.append(f"{s}: files {an} ({a0}..{a1}) and {bn} ({b0}..{b1}) overlap in time; exact duplicates are removed, but overlapping downloads can still mix products (e.g. archive + NRT) - keep one file per period.")
        if d.get("duplicates_checked") is False: out.append(f"{s}: no acq_time column, so duplicate detections could not be checked.")
        if d.get("duplicate_check_complete") is False: out.append(f"{s}: duplicate check stopped at the memory cap; later duplicates may remain.")
    he_ = res.get("harm_eval") or {}
    if he_.get("status") == "ok" and he_.get("test"):
        f_, t_ = (he_["metrics"].get("ratio") or {}).get("mape"), (he_["test"].get("ratio") or {}).get("mape")
        if f_ and t_ and t_ > 2 * f_ and t_ > 0.5: out.append(f"The reserved test year {he_['test_years']} has {t_:.0%} error vs {f_:.0%} in the earlier folds for every method: that points to a data problem in that year (duplicated or mixed-product files, a missing satellite, a sensor change) rather than to the methods. Check annual_ratio_diagnostics.csv and sensor_comparison.csv for that year.")
    ur = (res.get("diagnostics") or {}).get("unusual_ratio_years") or []
    if ur: out.append(f"The VIIRS/MODIS ratio of the whole extent is unusual in {', '.join(map(str, ur))} (robust z > 2.5). Check those years for duplicated or mixed-product files, a missing satellite or a sensor change.")
    return out


def _write(res, out):
    out = Path(out)
    (out / "results.json").write_text(json.dumps(res, separators=(",", ":"), default=float), encoding="utf-8")
    (out / "quality_report.json").write_text(json.dumps({**res["quality"], "_satellite_policy": res.get("sat_policy"), "_warnings": res.get("data_warnings", []),
                                                         "_missing_observation_weeks": {k: len(v) for k, v in (res.get("obs_gaps") or {}).items()}}, indent=2), encoding="utf-8")
    write_dashboard(res, out / "dashboard.html")


# ---------------------------------------------------------------------------------------------- re-run single stages
def _load(out_dir):
    out = Path(out_dir); return out, json.loads((out / "results.json").read_text("utf-8"))


def retrain_harmonization(out_dir, cfg=None, force=None, log=print, hcfg=None):
    """Re-run ONLY the harmonization comparison / XGBoost, then everything that depends on it (LightGBM outlook, fire clusters, reports)."""
    import pandas as pd
    out, res = _load(out_dir); hcfg = hcfg or he.HCfg(**(res.get("hcfg") or {})); res["hcfg"] = vars(hcfg)
    tab = pd.read_csv(out / "diag_counts.csv.gz", keep_default_na=False)
    diag = stage_diagnostics(res, out, tab, hcfg, res.get("recent_tau_mode", "auto"), log)
    info = stage_harmonization(res, out, hcfg, cfg, log, force, True)
    diag2 = dg.compute(res, tab, _boxes(res), reserved_years(res, hcfg.test_years), res["harm_method"]); diag2["recent_weighting"] = diag["recent_weighting"]; he.write_outputs(out, res, info, diag2); res["diagnostics"] = _slim(diag2)
    stage_forecast(res, out, log)
    try:
        stage_events(res, out, log)
    except Exception as e:
        log(f"[events] failed: {e}")
    _write(res, out); return {**res["xgboost"], "chosen": info.get("chosen"), "why": info.get("why")}
