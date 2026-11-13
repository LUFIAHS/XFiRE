"""Shared evaluation statistics (numpy only).

Why this module exists: forecast and harmonization errors are strongly autocorrelated (neighbouring weeks, overlapping 26-week horizons, many
grid cells in one year). Treating rows as independent makes every difference look "significant". All uncertainty statements in XFiRE therefore
come from block bootstraps whose blocks are whole calendar years (harmonization) or 26-week origin blocks (forecasts), and every comparison is
paired (both methods scored on the same rows).
"""
import numpy as np


def mae(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); return float(np.abs(a - b).mean()) if a.size else float("nan")


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); return float(np.sqrt(((a - b) ** 2).mean())) if a.size else float("nan")


def rel_bias(pred, obs):
    """Signed relative error of the summed prediction (+ = too high)."""
    s = float(np.sum(obs)); return float((np.sum(pred) - s) / s) if s > 0 else float("nan")


def blocks_of(index, length):
    """Non-overlapping consecutive blocks of `length` index units (e.g. 26 forecast-origin weeks). Index may be any integer array."""
    index = np.asarray(index, int); return (index - index.min()) // max(int(length), 1)


def paired_block_bootstrap(d, blocks, reps=2000, seed=0, level=0.90):
    """Mean of the per-row difference `d` with a block-bootstrap interval. Blocks (not rows) are resampled, and the statistic is the
    row-weighted mean of the resample, i.e. exactly the statistic that is reported. Returns None when fewer than 3 blocks exist
    (an interval from 2 blocks would be meaningless)."""
    d = np.asarray(d, float); blocks = np.asarray(blocks); ok = np.isfinite(d); d, blocks = d[ok], blocks[ok]
    if d.size == 0: return None
    ub = np.unique(blocks)
    if len(ub) < 3: return {"mean": float(d.mean()), "lo": None, "hi": None, "blocks": int(len(ub)), "n": int(d.size), "blocks_positive": int(sum(d[blocks == u].mean() > 0 for u in ub))}
    s = np.array([d[blocks == u].sum() for u in ub]); c = np.array([(blocks == u).sum() for u in ub], float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ub), (reps, len(ub))); boot = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    a = (1 - level) / 2
    return {"mean": float(d.mean()), "lo": float(np.quantile(boot, a)), "hi": float(np.quantile(boot, 1 - a)), "blocks": int(len(ub)), "n": int(d.size),
            "blocks_positive": int(sum(d[blocks == u].mean() > 0 for u in ub))}


def interval_score(y, lo, hi, nominal):
    """Winkler / interval score (lower is better): width + (2/alpha) * distance outside the interval. Rewards narrow intervals that still cover."""
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi)); a = 1.0 - nominal
    return float((hi - lo + (2 / a) * np.maximum(lo - y, 0) + (2 / a) * np.maximum(y - hi, 0)).mean())
