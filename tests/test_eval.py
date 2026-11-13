"""python tests/test_eval.py - harmonization protocol on SYNTHETIC data: forward-chaining folds, reserved test year integrity, selection rule, count regression,
fold reporting. A stand-in network (= a block ratio) plays XGBoost; this tests the PROTOCOL, not the network."""
import copy, tempfile
from pathlib import Path
import numpy as np
from test_helpers import build_test_fixture
from xfire import harmeval as he
from xfire.raster import Raster, fit_ratio, predict_ratio, folds_of, Evaluator
from xfire.harmonize import _wdates

with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "o"; res = build_test_fixture(out, use_xgb=False)
    R = Raster(res); weeks = R.overlap_weeks(); yrs = _wdates(weeks).year.to_numpy()
    # folds: forward chaining, whole years, never train on the future or on the test year; fewer folds than requested are reported, not faked
    fl = folds_of(R, weeks[yrs < 2025], 20); assert 1 <= len(fl) < 20
    for tr, te in fl:
        ty = _wdates(te).year.unique(); assert len(ty) == 1 and _wdates(tr).year.max() < ty[0] and len(np.unique(_wdates(tr).year)) >= 3
    # protocol with a stand-in learned method (a block ratio plays XGBoost: this tests the PROTOCOL)
    stand = he.Method("xgboost", lambda R_, tr, cell_mask=None: fit_ratio(R_, tr, block=12, cell_mask=cell_mask), lambda m, R_, ws: {int(w): predict_ratio(R_, m, int(w)) for w in ws}, learned_cell_level=True)
    info, uc = he.run_protocol(res, out, [stand], he.HCfg(folds=20, test_years=1), log=lambda *a: None)
    assert info["status"] == "ok" and info["folds_used"] < info["folds_requested"] == 20 and info["test_years"] == [2025], (info["folds_used"], info["test_years"])
    assert any("requested" in n for n in info["notes"]) and any("incomplete" in n for n in info["notes"])         # limitations are reported, not silent
    assert set(info["metrics"]) == {"ratio", "block_ratio", "xgboost"} and info["test"] and info["periods"]["test"]["years"] == [2025]
    assert info["periods"]["folds"][-1]["test_year"] < 2025 and all(f["train_years"][1] < f["test_year"] for f in info["periods"]["folds"])
    assert uc is None or len(uc) == len(res["cells"])
    for k, m in info["metrics"].items(): assert m["spatial_r"] is not None and m["mae"] is not None and m["rmse"] is not None, k
    # RESERVED TEST YEAR INTEGRITY: wrecking the MODIS data of the test year must change the test score but NOTHING that feeds training, folds or the choice
    bad = copy.deepcopy(res); test_w = set(int(w) for w in weeks[yrs == 2025])
    for c in bad["cells"]:
        d = c[2]
        for k in range(0, len(d), 3):
            if d[k] in test_w: d[k + 1] = d[k + 1] * 7 + 3
    info2, _ = he.run_protocol(bad, out, [stand], he.HCfg(folds=20, test_years=1), log=lambda *a: None, build_uc=False)
    assert info2["chosen"] == info["chosen"] and info2["why"] == info["why"], (info["chosen"], info2["chosen"])
    assert info2["metrics"] == info["metrics"] and info2["fold_mape"] == info["fold_mape"] and info2["paired"] == info["paired"]
    assert info2["test"]["ratio"]["mape"] != info["test"]["ratio"]["mape"]
    # a corrupted test year (e.g. duplicated rows) must raise a visible data warning
    from xfire.pipeline import data_warnings
    bad["harm_eval"] = info2; assert any("reserved test year" in w for w in data_warnings(bad)), data_warnings(bad)
    # selection rule: a method that is only better on the TEST year must not be chosen
    m_ = {"ratio": {"mape": .30}, "xgboost": {"mape": .29}}; assert he.choose(m_, {"xgboost_vs_ratio": {"hi": .01}}, {"ratio": [.3] * 3, "xgboost": [.29] * 3})[0] == "ratio"
    m_ = {"ratio": {"mape": .30}, "xgboost": {"mape": .20}, "block_ratio": {"mape": .21}}; pr = {"xgboost_vs_ratio": {"hi": -.05}, "block_ratio_vs_ratio": {"hi": -.05}, "xgboost_vs_block_ratio": {"hi": .01}}
    assert he.choose(m_, pr, {"ratio": [.3] * 3, "xgboost": [.2] * 3, "block_ratio": [.21] * 3})[0] == "block_ratio"
    pr["xgboost_vs_block_ratio"] = {"hi": -.01}; m_["xgboost"]["mape"] = .15; assert he.choose(m_, pr, {"ratio": [.3] * 3, "xgboost": [.15] * 3, "block_ratio": [.21] * 3})[0] == "xgboost"      # XGBoost must beat the regional block ratio too
    # v3.3 consistency: the paired delta IS the difference of the printed headline errors (it used to be a different, pooled estimator), and the decision table
    # carries every criterion of the rule, so reason text and tables cannot disagree
    assert info["decision"], "decision table missing"
    for d in info["decision"]:
        if d["paired_delta"] is not None: assert abs(d["paired_delta"] - (d["mape_a"] - d["mape_b"])) < 2.5e-4, d
        assert d["passes"] == (d["pass_margin"] and d["pass_ci"] and d["pass_wins"])
    ch = he.choose(m_, {**pr, "xgboost_vs_block_ratio": {"hi": .01, "lo": -.02, "delta": -.01}}, {"ratio": [.3] * 3, "xgboost": [.15] * 3, "block_ratio": [.21, .14, .16]})
    assert ch[0] == "block_ratio" and "does not lie below 0" in ch[1] and any(d["a"] == "xgboost" and d["b"] == "block_ratio" and not d["passes"] for d in ch[2])
    # test confirmation must compare with the strongest simpler baseline, not just the seasonal ratio (v3.3 fix)
    T_ = {"ratio": {"mape": .26}, "block_ratio": {"mape": .20}, "xgboost": {"mape": .23}}
    assert he.test_confirms("xgboost", T_, {}) is False and he.test_confirms("block_ratio", T_, {}) is True and he.test_confirms("ratio", T_, {"block_ratio_vs_ratio": {"hi": -.01}}) is False
    # spatial hold-out: the training-cell mask really removes cells; a held-out block falls back to the seasonal ratio; the diagnostic ran and is comparable
    fid = he.spatial_block_folds(R, 4); assert set(np.unique(fid)) <= {0, 1, 2, 3} and len(np.unique(fid)) >= 2
    fr = fit_ratio(R, weeks[yrs < 2024], block=16, cell_mask=(fid != 0)); held = np.argwhere(fid == 0)[0]; bi, bj = held[0] // 16, held[1] // 16
    assert np.allclose(fr["table"][:, bi, bj], fr["k0"]), "held-out block must use the seasonal ratio"
    sp = info["spatial_holdout"]; assert sp and sp.get("metrics") and set(sp["metrics"]) == {"ratio", "block_ratio", "xgboost"} and sp["folds"] == 4 and sp["year"] < 2025, sp
    assert all(sp["metrics"][n]["mape"] is not None for n in sp["metrics"])
    # outputs written by the build itself
    md = (out / "harmonization_evaluation.md").read_text()
    for t in ("Recommendation", "Validation design", "folds", "Reserved final test", "By season", "Limitations"): assert t in md, t
    for f in ("harmonization_evaluation.json", "annual_ratio_diagnostics.csv", "sensor_comparison.csv", "harmonization_model_metrics.csv"): assert (out / f).stat().st_size > 0, f
    print("eval test PASS")
