"""Harmonization evaluation protocol (numpy only - XGBoost plugs in as one more method, so the statistical baselines are evaluated even without it).

Methods compared on identical weeks: "ratio" (seasonal VIIRS/MODIS ratio, the app's classical method), "block_ratio" (regional block-ratio baseline) and,
when xgboost is installed, "xgboost" (learned MODIS-equivalent counts).
Protocol (all chronological, whole calendar years):
  1. the newest `test_years` COMPLETE overlap years are reserved. They never touch training, hyper-parameters, calibration or model selection.
  2. forward-chaining folds on the earlier years: fold j trains on years < Y_j and predicts year Y_j. The number of folds actually possible is
     reported next to the number requested (a fold needs >= 3 earlier years).
  3. the default method is chosen from step 2 ONLY (margin + paired block-bootstrap interval excluding 0 + wins in most folds; a learned model must also beat the
     best statistical baseline). The reserved years are scored once afterwards and reported as a confirmation - they cannot change the choice.
Spatial hold-out (v3.3): in addition, the newest cross-fit year is predicted by models trained WITHOUT 8-degree blocks of cells (K spatial folds), which measures how
much each method relies on having seen a location before. It is a diagnostic reported next to the decision; it does not change the choice (production always predicts
locations that were in training), but a learned method that loses to the block ratio on unseen blocks is flagged.
Decision table (v3.3): every criterion of the selection rule is computed once, from one estimator (mean over areas of the area-mean absolute percentage error), and printed
in full, so the rule, the paired table and the headline numbers can no longer disagree.
"""
import json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .raster import Raster, Evaluator, UcBuilder, fit_ratio, predict_ratio, folds_of, auto_blocks
from .harmonize import _wdates, _season_of

SEAS = ("DJF", "MAM", "JJA", "SON")


@dataclass
class HCfg:
    """Protocol settings (independent of XGBoost)."""
    folds: int = 3              # requested forward-chaining folds; the number actually possible is reported
    test_years: int = 1         # newest complete overlap years reserved as a final untouched test set (0 = off)
    margin: float = 0.02        # a method must beat the seasonal ratio by this relative margin on the held-out folds
    min_win: float = 0.6        # ... and win in at least this share of the folds
    spatial_folds: int = 4      # spatial hold-out folds for the newest cross-fit year (0 = off)


class Method:
    """fit(R, weeks) -> model ; predict(model, R, weeks) -> {week: (H, W) image}; on_final(model) is called with the model trained on all overlap weeks"""
    def __init__(self, name, fit, predict, learned_cell_level=False, on_final=None): self.name, self.fit, self.predict, self.cell, self.on_final = name, fit, predict, learned_cell_level, on_final
    # fit(R, weeks, cell_mask=None): cell_mask (H, W) bool restricts training to those cells (spatial hold-out)


def baseline_methods():
    f = lambda fn: (lambda R, w: fn(R, w))
    return [Method("ratio", lambda R, w, cell_mask=None: fit_ratio(R, w, cell_mask=cell_mask), lambda m, R, ws: {int(w): predict_ratio(R, m, int(w)) for w in ws}),
            Method("block_ratio", lambda R, w, cell_mask=None: fit_ratio(R, w, block=16, cell_mask=cell_mask), lambda m, R, ws: {int(w): predict_ratio(R, m, int(w)) for w in ws})]


BASELINES = ("ratio", "block_ratio")
LABEL = {"ratio": "seasonal ratio", "block_ratio": "regional block ratio", "xgboost": "XGBoost", "hist_gbm": "scikit-learn gradient boosting (hist_gbm)"}


def compare(a, b, metrics, paired, fold_mape, margin, min_win):
    """All criteria for 'a replaces b', from the SAME numbers that are printed in the report. -> dict (never None when both methods have a held-out error)."""
    ma, mb = (metrics.get(a) or {}).get("mape"), (metrics.get(b) or {}).get("mape")
    if ma is None or mb is None: return None
    pr = paired.get(f"{a}_vs_{b}"); fm = [(x, y) for x, y in zip(fold_mape.get(a, []), fold_mape.get(b, [])) if x is not None and y is not None]; wins = int(sum(x < y for x, y in fm))
    rel = (mb - ma) / mb if mb > 0 else 0.0
    c = {"a": a, "b": b, "mape_a": ma, "mape_b": mb, "relative_gain": round(rel, 4), "margin_required": margin, "pass_margin": bool(rel >= margin),
         "paired_delta": None if not pr else pr.get("delta"), "ci_lo": None if not pr else pr.get("lo"), "ci_hi": None if not pr else pr.get("hi"), "pass_ci": bool(pr is not None and pr.get("hi", 1) < 0),
         "fold_wins": wins, "folds": len(fm), "win_share_required": min_win, "pass_wins": bool(fm and wins / len(fm) >= min_win)}
    c["passes"] = bool(c["pass_margin"] and c["pass_ci"] and c["pass_wins"]); return c


def _why_not(c):
    if c is None: return "no held-out error"
    bad = []
    if not c["pass_margin"]: bad.append(f"relative gain {c['relative_gain']:.1%} < required {c['margin_required']:.0%}")
    if not c["pass_ci"]: bad.append("paired 90% interval " + ("unavailable" if c["ci_lo"] is None else f"[{c['ci_lo']}, {c['ci_hi']}] does not lie below 0"))
    if not c["pass_wins"]: bad.append(f"wins in only {c['fold_wins']} of {c['folds']} folds (needs >= {c['win_share_required']:.0%})")
    return "; ".join(bad) or "passes"


def choose(metrics, paired, fold_mape, margin=0.02, min_win=0.6):
    """-> (method, reason, decision_table). Uses cross-fitted held-out results ONLY (never the reserved test years). The regional block ratio must beat the seasonal ratio;
    a learned method must beat BOTH baselines. `decision_table` lists every comparison with each criterion, so the reason text and the paired table cannot disagree."""
    if (metrics.get("ratio") or {}).get("mape") is None: return "ratio", "no held-out error available", []
    learned = [n for n in metrics if n not in BASELINES]; table = []; cmp_ = lambda a, b: compare(a, b, metrics, paired, fold_mape, margin, min_win)
    cb = cmp_("block_ratio", "ratio") if "block_ratio" in metrics else None
    if cb: table.append(cb)
    good = []
    for n in learned:
        cr = cmp_(n, "ratio"); cbl = cmp_(n, "block_ratio") if "block_ratio" in metrics else None
        for c in (cr, cbl):
            if c: table.append(c)
        if cr and cr["passes"] and (cbl is None or cbl["passes"]): good.append(n)
    rule = f"margin {margin:.0%}, paired 90% block-bootstrap interval below 0, and wins in >= {min_win:.0%} of folds"
    if good:
        n = min(good, key=lambda k: metrics[k]["mape"])
        return n, f"{LABEL.get(n, n)} beats the seasonal ratio ({metrics['ratio']['mape']}) and the regional block ratio ({(metrics.get('block_ratio') or {}).get('mape')}) on every criterion ({rule}): held-out error {metrics[n]['mape']}", table
    notes = [f"{LABEL.get(n, n)} vs block ratio: {_why_not(cmp_(n, 'block_ratio') if 'block_ratio' in metrics else cmp_(n, 'ratio'))}" for n in learned]
    if cb and cb["passes"]:
        extra = (" The learned method does not clear the simpler method on every criterion (" + " | ".join(notes) + "), so the simpler method is kept.") if learned else ""
        return "block_ratio", f"the regional block ratio beats the seasonal ratio ({metrics['ratio']['mape']} -> {metrics['block_ratio']['mape']}) ({rule}).{extra}", table
    extra = (" Learned: " + " | ".join(notes) + ".") if learned else ""
    return "ratio", f"no alternative beats the seasonal ratio ({metrics['ratio']['mape']}) on every criterion ({rule}).{extra}", table


def _fold_mapes(ev, names, fold_weeks):
    out = {}
    for n in names:
        rows = ev.acc.get(n, {}).get("rows", []); out[n] = []
        for fw in fold_weeks:
            s = set(int(x) for x in fw); r = [x for x in rows if x[0] in s]; out[n].append(ev._mape(r)[0] if r else None)
    return out


def _log_sd(rows, season_of):
    """Spread of log((estimate+1)/(MODIS+1)) of the whole-extent weekly sums per season: the held-out harmonization uncertainty of the learned estimate."""
    sd = []
    for q in range(4):
        d = [np.log((p + 1) / (m + 1)) for w, k, m, p in rows if k == "Whole extent" and season_of[w] == q]
        sd.append(round(float(np.std(d)), 4) if len(d) >= 8 else None)
    ok = [x for x in sd if x is not None]; fb = float(np.mean(ok)) if ok else 0.0
    return [fb if x is None else x for x in sd]


def spatial_block_folds(R, k, block=16):
    """(H, W) int array: spatial fold id (0..k-1) per cell, assigned per `block` x `block` cell block (default 8 degrees = the block-ratio unit) by a fixed hash, so each
    fold is a scatter of whole blocks. -1 outside the grid cells."""
    bi, bj = (np.arange(R.H) // block)[:, None], (np.arange(R.W) // block)[None, :]
    return ((bi * 73856093) ^ (bj * 19349663)) % k


def spatial_holdout(R, methods, train_w, test_w, k, boxes, classes, log=print):
    """Strict check: the test weeks (a year the temporal folds also scored) are predicted by models that never saw the held-out 8-degree blocks. Models of the k spatial folds are
    stitched together (each predicts only its own blocks) and scored with the usual Evaluator, so numbers are comparable with the cross-fit table."""
    fid = spatial_block_folds(R, k); ev = Evaluator(R, boxes, classes)
    for m in methods:
        mods = [m.fit(R, train_w, cell_mask=(fid != j)) for j in range(k)]
        for w in test_w:
            img = np.zeros((R.H, R.W), np.float32)
            for j, mod in enumerate(mods): img = np.where(fid == j, m.predict(mod, R, [int(w)])[int(w)], img)
            ev.add(m.name, int(w), img)
    return ev.result()


def run_protocol(res, out_dir, extra_methods=(), cfg=None, force=None, log=print, build_uc=True):
    """Cross-fit + test + selection. extra_methods: [Method] added to the baselines (XGBoost). Returns (info, uc_lists | None)."""
    cfg = cfg or HCfg(); folds, test_years, margin = cfg.folds, cfg.test_years, cfg.margin
    t0 = time.time(); R = Raster(res); allw = R.overlap_weeks(); methods = baseline_methods() + list(extra_methods); names = [m.name for m in methods]
    if force in ("xgboost", "learned", "hist_gbm") and force not in names: force = next((n for n in names if n not in BASELINES), force)       # `xgboost` = the learned method that actually ran
    if len(allw) < 20: return {"status": "skipped", "why": f"only {len(allw)} weeks with both MODIS and VIIRS (need 20+)"}, None
    yrs = _wdates(allw).year.to_numpy(); uy = np.unique(yrs); full = [int(y) for y in uy if (yrs == y).sum() >= 40]; notes = []
    test_y = full[-test_years:] if test_years > 0 and len(full) >= test_years + 4 else []
    if test_years > 0 and not test_y: notes.append(f"No test year reserved: needs {test_years + 4}+ complete overlap years, found {len(full)}.")
    if test_y and yrs.max() > max(test_y): notes.append(f"Overlap weeks newer than the reserved test year {max(test_y)} ({int((yrs > max(test_y)).sum())} weeks) are not evaluated; only the final model uses them.")
    if test_y and (uy[-1] != max(test_y)) and uy[-1] not in full: notes.append(f"Newest overlap year {int(uy[-1])} is incomplete ({int((yrs == uy[-1]).sum())} weeks) and was not used as the test year.")
    pool = allw[yrs < min(test_y)] if test_y else allw[np.isin(yrs, full)]; test_w = allw[np.isin(yrs, test_y)] if test_y else np.array([], int)
    fl = folds_of(R, pool, folds)
    if not fl: return {"status": "skipped", "why": "not enough complete overlap years to hold data out (need >= 4 before the test year)", "notes": notes}, None
    if len(fl) < folds: notes.append(f"{folds} folds requested but only {len(fl)} are possible: each fold needs >= 3 earlier complete years. Using {len(fl)}.")
    boxes = {"Whole extent": None, **res.get("regions", {})}; classes = {}
    if not res.get("regions"):
        ab, classes = auto_blocks(R, pool); boxes.update(ab)
    ev = Evaluator(R, boxes, classes); builders = {m.name: UcBuilder(R) for m in methods if m.cell and build_uc}; seen = set(); sdev = R.season
    for f, (tr, te) in enumerate(fl):
        log(f"[harmonization] fold {f + 1}/{len(fl)}: train years {int(_wdates(tr).year.min())}-{int(_wdates(tr).year.max())} ({len(tr)} wk) -> test {int(_wdates(te).year[0])} ({len(te)} wk)")
        for m in methods:
            P = m.predict(m.fit(R, tr), R, te)
            for w, p in P.items():
                ev.add(m.name, w, p)
                if m.name in builders: builders[m.name].add(w, p)
        seen |= set(int(x) for x in te)
    metrics = ev.result(); fw = [te for _, te in fl]; fm = _fold_mapes(ev, names, fw); paired = pair_all(ev, names)
    chosen, why, decision = choose(metrics, paired, fm, margin, cfg.min_win)
    test = None; tpaired = {}; test_models = {}
    if len(test_w):
        log(f"[harmonization] reserved test year(s) {test_y}: scored once, after the choice was made from the folds")
        et = Evaluator(R, boxes, classes)
        for m in methods:
            mod = m.fit(R, pool); P = m.predict(mod, R, test_w); test_models[m.name] = mod
            for w, p in P.items():
                et.add(m.name, w, p)
                if m.name in builders: builders[m.name].add(w, p)
        seen |= set(int(x) for x in test_w); test = et.result(); tpaired = pair_all(et, names)
    spatial = None
    if cfg.spatial_folds and cfg.spatial_folds >= 2 and fl:
        tr_s, te_s = fl[-1]; log(f"[harmonization] spatial hold-out: {cfg.spatial_folds} folds of 8-degree blocks, year {int(_wdates(te_s).year[0])}")
        try:
            sm = spatial_holdout(R, methods, tr_s, te_s, cfg.spatial_folds, boxes, classes, log); yk = str(int(_wdates(te_s).year[0]))
            spatial = {"year": int(yk), "folds": cfg.spatial_folds, "block_degrees": 8, "metrics": sm, "temporal_only_same_year": {n: (metrics[n]["by_year"] or {}).get(yk) for n in names},
                       "increase_vs_temporal_only": {n: (None if sm[n]["mape"] is None or (metrics[n]["by_year"] or {}).get(yk) in (None, 0) else round(sm[n]["mape"] / metrics[n]["by_year"][yk] - 1, 3)) for n in names if n in sm}}
            sw = [n for n in names if n not in BASELINES and sm.get(n, {}).get("mape") is not None and sm.get("block_ratio", {}).get("mape") is not None and sm[n]["mape"] > sm["block_ratio"]["mape"]]
            spatial["learned_worse_than_block_ratio_on_unseen_blocks"] = sw
        except Exception as e:                                                         # diagnostic only: never lose the harmonization because of it
            log(f"[harmonization] spatial hold-out failed: {type(e).__name__}: {e}"); spatial = {"status": "failed", "why": f"{type(e).__name__}: {e}"}
    rows = ev.acc[chosen]["rows"] if chosen in ev.acc else []; sd = _log_sd(rows, sdev)
    uc = None; applied = force if force in names else chosen          # `force` (--harmonize) overrides the recommendation, and is recorded as such
    if build_uc and applied != "ratio":
        mc = next(m for m in methods if m.name == applied)
        if applied in builders: b = builders[applied]
        else:                                                                # cheap baselines: rebuild out-of-fold estimates for the chosen one
            b = UcBuilder(R)
            for tr, te in fl:
                for w, p in mc.predict(mc.fit(R, tr), R, te).items(): b.add(w, p)
            if len(test_w):
                for w, p in mc.predict(test_models[applied], R, test_w).items(): b.add(w, p)
        final = mc.fit(R, allw)
        if mc.on_final: mc.on_final(final)
        rest = [w for w in range(max(R.first, R.vw), R.lfw + 1) if R.Vw[w] > 0 and w not in seen]
        for w, p in mc.predict(final, R, rest).items(): b.add(w, p)
        uc = b.lists()
    info = {"status": "ok", "chosen": chosen, "why": why, "applied": applied, "forced": bool(force and force != chosen), "overlap_weeks": int(len(allw)), "margin": margin, "folds_requested": int(folds), "folds_used": len(fl),
            "fold_scheme": "forward-chaining whole calendar years", "test_years": test_y, "notes": notes,
            "periods": {"overlap_first": str(_wdates([allw.min()])[0].date()), "overlap_last": str(_wdates([allw.max()])[0].date()),
                        "folds": [{"fold": j + 1, "train_years": [int(_wdates(a).year.min()), int(_wdates(a).year.max())], "train_weeks": int(len(a)), "test_year": int(_wdates(b).year[0]), "test_weeks": int(len(b))} for j, (a, b) in enumerate(fl)],
                        "test": {"years": test_y, "weeks": int(len(test_w))}},
            "coverage": {"extent": [float(R.lat_c.min()), float(R.lat_c.max()), float(R.lon_c.min()), float(R.lon_c.max())], "cells": int(len(res["cells"])), "evaluation_areas": list(boxes)},
            "metrics": metrics, "fold_mape": fm, "paired": paired, "decision": decision, "spatial_holdout": spatial, "test": test, "test_paired": tpaired,
            "test_confirms_choice": test_confirms(chosen, test, tpaired), "test_ranking": None if test is None else sorted([n for n in names if test[n]["mape"] is not None], key=lambda n: test[n]["mape"]),
            "recent_tau_ratio": float(res.get("recent_tau") or 0.0),
            "log_sd_by_season": sd, "seconds": round(time.time() - t0, 1), "methods": names}
    return info, uc


# ------------------------------------------------------------------------------------------------ outputs
def pair_all(ev, names):
    """Every paired comparison the decision needs: each method vs the seasonal ratio, each learned method vs the block ratio, block ratio vs ratio (all on identical area-weeks)."""
    out = {}
    for a in names:
        if a != "ratio": out[f"{a}_vs_ratio"] = ev.paired(a, "ratio")
        if a not in BASELINES and "block_ratio" in names: out[f"{a}_vs_block_ratio"] = ev.paired(a, "block_ratio")
    return out


def test_confirms(chosen, test, tpaired):
    """Does the reserved test period support the choice? True when the chosen method has a lower test error than EVERY simpler baseline (ratio < block_ratio < learned), and for
    `ratio` when no alternative beats it with a paired interval below 0. (v3.3: this used to compare only with the seasonal ratio, which any method can beat.) None = no test."""
    if test is None or (test.get(chosen) or {}).get("mape") is None: return None
    simpler = {"ratio": [], "block_ratio": ["ratio"]}.get(chosen, ["ratio", "block_ratio"]); mm = lambda n: (test.get(n) or {}).get("mape")
    if chosen == "ratio": return not any((v and v["hi"] < 0) for k, v in tpaired.items() if k.endswith("_vs_ratio"))
    return all(mm(n) is None or test[chosen]["mape"] < mm(n) for n in simpler)


def metric_rows(info):
    """Flat rows of harmonization_model_metrics.csv (cross-fit and reserved test, overall + season/year/region/region-class/level)."""
    rows = []
    for split, src in (("cross_fit", info.get("metrics") or {}), ("test", info.get("test") or {})):
        for meth, m in src.items():
            rows.append({"method": meth, "split": split, "group_type": "overall", "group": "all", "mape": m["mape"], "mape_lo90": m.get("mape_lo90"), "mape_hi90": m.get("mape_hi90"), "bias": m["bias"],
                         "mae": m.get("mae"), "rmse": m.get("rmse"), "poisson_deviance": m["dev"], "spatial_r": m.get("spatial_r"), "weeks": m["weeks"]})
            for gt, key in (("season", "by_season"), ("year", "by_year"), ("region", "areas"), ("region_class", "by_region_class"), ("activity_level", "by_level")):
                for g, v in (m.get(key) or {}).items(): rows.append({"method": meth, "split": split, "group_type": gt, "group": g, "mape": v})
    return rows


def _md_table(headers, rows): return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join("n/a" if c is None else (f"{c:.3f}" if isinstance(c, float) else str(c)) for c in r) + " |" for r in rows])


def report_markdown(res, info, diag):
    L = ["# MODIS-VIIRS harmonization: evaluation", "",
         "Everything below was computed from the supplied FIRMS data. Harmonized values are MODIS-EQUIVALENT ESTIMATES, not satellite measurements. "
         "Error = mean over areas of the mean weekly |estimate - MODIS| / MODIS (weeks with >= 5 MODIS detections), on weeks the method never trained on.", ""]
    if info.get("status") != "ok":
        L += [f"**Evaluation not run**: {info.get('why')}", ""]
    else:
        p = info["periods"]; L += ["## Recommendation", "", f"**Use `{info['chosen']}` as the default harmonization.** {info['why']}.", ""]
        if info.get("forced"): L += [f"NOTE: the build was forced to `{info['applied']}` (--harmonize); this overrides the recommendation above.", ""]
        if info.get("test_confirms_choice") is not None:
            L += [f"Reserved test year(s) {info['test_years']}: {'confirms' if info['test_confirms_choice'] else 'does NOT confirm'} the choice (chosen method must beat every simpler baseline on the test years; they did not influence the choice). Test ranking, best first: {info.get('test_ranking')}.", ""]
        if info.get("decision"):
            L += ["## Decision table (cross-fitted folds only)", "", "Every criterion of the selection rule, computed from the same numbers as the tables below. A method replaces its comparator only if ALL three pass.", "",
                  _md_table(["comparison", "error a", "error b", "relative gain (>= margin)", "paired delta [90% CI] (hi < 0)", "fold wins (>= share)", "passes"],
                            [[f"{d['a']} vs {d['b']}", d["mape_a"], d["mape_b"], f"{d['relative_gain']:.1%} {'ok' if d['pass_margin'] else 'FAIL'} (>= {d['margin_required']:.0%})",
                              "n/a" if d["paired_delta"] is None else f"{d['paired_delta']} [{d['ci_lo']}, {d['ci_hi']}] {'ok' if d['pass_ci'] else 'FAIL'}", f"{d['fold_wins']}/{d['folds']} {'ok' if d['pass_wins'] else 'FAIL'} (>= {d['win_share_required']:.0%})", "yes" if d["passes"] else "no"] for d in info["decision"]]), ""]
        L += ["## Validation design", "", f"- Overlap {p['overlap_first']} to {p['overlap_last']}, {info['overlap_weeks']} weeks with both sensors observed (weeks with missing observations excluded).",
              f"- Folds: {info['folds_used']} used of {info['folds_requested']} requested ({info['fold_scheme']}).",
              *[f"  - fold {f['fold']}: train {f['train_years'][0]}-{f['train_years'][1]} ({f['train_weeks']} wk), test {f['test_year']} ({f['test_weeks']} wk)" for f in p["folds"]],
              f"- Reserved final test: {p['test']['years'] or 'none'} ({p['test']['weeks']} weeks), never used for training, tuning, calibration or selection.",
              f"- Geographic coverage: {info['coverage']['cells']} grid cells; extent lat {info['coverage']['extent'][0]:.1f}..{info['coverage']['extent'][1]:.1f}, lon {info['coverage']['extent'][2]:.1f}..{info['coverage']['extent'][3]:.1f}.",
              "- Spatial hold-out: " + ("see section below." if (info.get("spatial_holdout") or {}).get("metrics") else "not available for this run."), *[f"- NOTE: {n}" for n in info["notes"]], ""]
        for title, src in (("Cross-fitted held-out years", info["metrics"]), ("Reserved test year(s)", info.get("test") or {})):
            if src:
                L += [f"## {title}", "", _md_table(["method", "error", "90% CI", "bias", "whole-extent WAPE", "whole-extent bias", "MAE", "RMSE", "Poisson dev.", "spatial r"],
                      [[k, v["mape"], "n/a" if v.get("mape_lo90") is None else f"{v['mape_lo90']}-{v['mape_hi90']}", v["bias"], v.get("wape_whole"), v.get("bias_whole"), v.get("mae"), v.get("rmse"), v["dev"], v.get("spatial_r")] for k, v in src.items()]), ""]
                if title.startswith("Reserved") and info.get("test_paired"):
                    for k, v in info["test_paired"].items():
                        if v: L.append(f"- Test-year paired {k.replace('_vs_', ' vs ')} (negative = first better): {v['delta']} (90% CI {v['lo']} to {v['hi']}; {v['blocks']} blocks of 13 weeks, {v['n']} area-weeks). " + ("Interval excludes 0." if v["hi"] < 0 or v["lo"] > 0 else "Not distinguishable from 0."))
                    L += [""]
        m = info["metrics"]
        for title, key in (("By season", "by_season"), ("By year", "by_year"), ("By region / block", "areas"), ("High- vs low-activity regions", "by_region_class"), ("By MODIS activity level", "by_level")):
            names = list(m); keys = sorted({k for n in names for k in (m[n].get(key) or {})}, key=str)
            if keys: L += [f"## {title}", "", _md_table(["", *names], [[k, *[(m[n].get(key) or {}).get(k) for n in names]] for k in keys]), ""]
        for k, v in info["paired"].items():
            if v: L.append(f"- Paired {k.replace('_vs_', ' vs ')} (negative = first better): {v['delta']} (90% CI {v['lo']} to {v['hi']}; {v['blocks']} year-blocks, {v['n']} area-weeks; better in {v['win_rate']:.0%} of area-weeks and in {v.get('year_wins', float('nan')):.0%} of years). Unit: {v.get('unit', 'area-mean error difference')}.")
        L += [""]
        sp = info.get("spatial_holdout")
        if sp and sp.get("metrics"):
            L += [f"## Spatial hold-out (year {sp['year']}, {sp['folds']} folds of {sp['block_degrees']}-degree blocks never seen in training)", "",
                  _md_table(["method", "error on unseen blocks", "same year, seen locations", "relative increase", "bias", "spatial r"],
                            [[n, v["mape"], sp["temporal_only_same_year"].get(n), sp["increase_vs_temporal_only"].get(n), v["bias"], v.get("spatial_r")] for n, v in sp["metrics"].items()]), ""]
            if sp.get("learned_worse_than_block_ratio_on_unseen_blocks"): L += [f"**Caution: {', '.join(sp['learned_worse_than_block_ratio_on_unseen_blocks'])} is worse than the block ratio on unseen blocks**: its advantage depends on having seen the location. Production predicts locations present in training, so this does not change the choice, but do not transfer the model to new regions without re-validation.", ""]
        elif sp: L += [f"Spatial hold-out failed: {sp.get('why')}", ""]
    d = diag or {}
    L += ["## Sensor diagnostics", ""]
    rw = d.get("recent_weighting") or {}; L += [f"- Recent-year weighting: {rw.get('status')} - recommended half-life **{rw.get('recommended_tau', 0)} years** ({rw.get('why', '')}). " + ("Applied to the seasonal ratio." if rw.get("recommended_tau") else "Not applied.")]
    for k, v in (d.get("ratio_drift") or {}).items(): L.append(f"- Ratio drift ({k}): Kendall tau {v['kendall_tau']}, p = {v['p_value']}, {v['relative_slope_pct_per_year']}%/yr -> {'DRIFT DETECTED' if v['drift_detected'] else 'no clear drift'}")
    dn = d.get("day_night") or {}; L.append(f"- Day/night: {dn.get('status')}; separate calibration recommended: {dn.get('separate_calibration_recommended')}. {dn.get('note', dn.get('why', ''))}")
    ta = d.get("terra_aqua") or {}; L.append(f"- Terra/Aqua: {ta.get('status')} ({ta.get('satellites')}); drift p = {ta.get('drift_p_value')}. {ta.get('note', ta.get('why', ''))}")
    vp = d.get("viirs_platforms") or {}; L.append(f"- VIIRS platforms: {vp.get('status')} ({vp.get('satellites')}); ratio second/first = {vp.get('ratio_second_over_first_overall')}; drift p = {vp.get('drift_p_value')}. {vp.get('why', '')}")
    ho = d.get("handover") or {}; L.append(f"- Sensor handover: {ho.get('status')}. " + (f"harmonized level change {ho['level_change_harmonized']:+.1%} vs MODIS alone {ho['level_change_modis_alone']:+.1%}; artefact suspected: {ho['artefact_suspected']}; log-sd ratio {ho['log_sd_ratio_harmonized']:.2f} (harmonized) vs {ho['log_sd_ratio_modis_alone']:.2f} (MODIS alone)." if ho.get("status") == "ok" else ho.get("why", "")))
    L += ["- Era statistics (harmonized weekly series): " + json.dumps(d.get("era_stats")), "", "## Data integrity", ""]
    for s, q in (res.get("quality") or {}).items(): L.append(f"- {s}: read {q['rows_read']}, valid {q['rows_valid']}; removed by stage {q.get('filters')}; duplicate check: {q.get('duplicates_checked')} (complete: {q.get('duplicate_check_complete')})")
    og = res.get("obs_gaps") or {}; L += [f"- Weeks with missing observations (treated as missing, not zero): MODIS {len(og.get('MODIS', []))}, VIIRS {len(og.get('VIIRS', []))}", "",
          "## Limitations", "", "- Day/night and Terra/Aqua are diagnosed at the ingest level; the gridded harmonization (including XGBoost) works on total counts per cell and week.",
          "- After MODIS ends there is no reference sensor: the VIIRS-only era cannot be validated against MODIS and relies on the overlap relationship staying stable.",
          "- Hotspot counts are thermal anomalies, not burned area."]
    return "\n".join(L) + "\n"


def write_outputs(out_dir, res, info, diag):
    import csv
    out = Path(out_dir)
    def wcsv(name, rows):
        if not rows: (out / name).write_text(""); return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: ("" if v is None else v) for k, v in r.items()} for r in rows])
    wcsv("annual_ratio_diagnostics.csv", (diag or {}).get("annual_ratio_rows")); wcsv("sensor_comparison.csv", (diag or {}).get("sensor_rows")); wcsv("harmonization_model_metrics.csv", metric_rows(info))
    slim = {k: v for k, v in (diag or {}).items() if k not in ("annual_ratio_rows", "sensor_rows")}
    (out / "harmonization_evaluation.json").write_text(json.dumps({"protocol": info, "diagnostics": slim, "annual_ratio_rows": (diag or {}).get("annual_ratio_rows"),
                                                                    "quality": res.get("quality"), "obs_gaps": {k: len(v) for k, v in (res.get("obs_gaps") or {}).items()}}, indent=1, default=float))
    (out / "harmonization_evaluation.md").write_text(report_markdown(res, info, diag), encoding="utf-8")
