"""Fire-cluster events (numpy only).

A *cluster* is a connected group of grid cells that burn clearly in one month. Class 0 = typical for that month of the year,
class 1 = anomalous (>= 2x the median of the same month in the other years). Detected by a transparent rule (thresholds + connected components) - no learned model involved.
Everything here works on image-like arrays [H, W] (row = latitude cell, south -> north) and returns boxes in cell indices
{"i0","i1","j0","j1"} (inclusive); `to_geo` turns them into [lat_min, lat_max, lon_min, lon_max].
"""
import numpy as np

CLASSES = ("cluster", "anomalous")


def month_key(m):
    return f"{2000 + int(m) // 12}-{int(m) % 12 + 1:02d}"


def month_index(key):
    y, m = key.split("-"); return (int(y) - 2000) * 12 + int(m) - 1


# ---- climatology & thresholds ---------------------------------------------------------------------
def loyo_climatology(cube, months, ok):
    """For every month: median of the same calendar month in the OTHER complete years -> (clim [T,H,W], has [T] bool).
    `has` is False when fewer than 3 other years exist (no meaningful anomaly)."""
    clim = np.zeros_like(cube); has = np.zeros(len(months), bool); moy = np.asarray(months) % 12
    for m in range(12):
        idx = np.nonzero((moy == m) & ok)[0]
        if len(idx) < 4: continue
        for t in idx:
            clim[t] = np.median(cube[idx[idx != t]], axis=0); has[t] = True
    return clim, has


def thresholds(cube, ok):
    """Data-adaptive cut-offs, computed once so labels are consistent across train / validation / inference."""
    pos = cube[ok][cube[ok] > 0]
    if pos.size == 0: return {"cmin": 5.0, "tsum": 25.0, "cap": 100.0}
    cmin = float(max(3.0, np.quantile(pos, 0.80)))
    return {"cmin": cmin, "tsum": 5 * cmin, "cap": float(max(np.quantile(pos, 0.999), 2 * cmin))}


# ---- rule-based detector --------------------------------------------------------------------------
def components(mask):
    """8-connected components of a boolean image -> list of (n, 2) arrays of (row, col)."""
    H, W = mask.shape; lab = -np.ones(mask.shape, np.int32); comps = []
    for i, j in np.argwhere(mask):
        if lab[i, j] >= 0: continue
        L = len(comps); lab[i, j] = L; stack = [(i, j)]; cells = []
        while stack:
            a, b = stack.pop(); cells.append((a, b))
            for na in (a - 1, a, a + 1):
                if na < 0 or na >= H: continue
                for nb in (b - 1, b, b + 1):
                    if 0 <= nb < W and mask[na, nb] and lab[na, nb] < 0:
                        lab[na, nb] = L; stack.append((na, nb))
        comps.append(np.array(cells))
    return comps


def rule_events(img, clim, th):
    """img: harmonized monthly activity [H, W]; clim: same-month median of other years [H, W] or None."""
    act = img >= th["cmin"]; d = act.copy()                         # 4-neighbour dilation: a one-cell gap does not split a cluster
    d[1:] |= act[:-1]; d[:-1] |= act[1:]; d[:, 1:] |= act[:, :-1]; d[:, :-1] |= act[:, 1:]
    out = []
    for cells in components(d):
        cc = cells[act[cells[:, 0], cells[:, 1]]]
        if len(cc) == 0: continue
        tot = float(img[cc[:, 0], cc[:, 1]].sum())
        if tot < th["tsum"]: continue
        cls = 0
        if clim is not None:
            cl = float(clim[cc[:, 0], cc[:, 1]].sum())
            cls = int(tot / max(cl, th["cmin"]) >= 2 and tot - cl >= th["tsum"])
        i0, j0 = cc.min(0); i1, j1 = cc.max(0)
        out.append({"i0": int(i0), "i1": int(i1), "j0": int(j0), "j1": int(j1), "cls": cls, "s": 1.0})
    return out


# ---- output ---------------------------------------------------------------------------------------
def to_geo(e, R, img):
    """cell-index box -> JSON event {"b": [lat0, lat1, lon0, lon1], "c": class, "s": score, "n": harmonized detections inside}."""
    c = R.cell; a = img[e["i0"]:e["i1"] + 1, e["j0"]:e["j1"] + 1]
    return {"b": [round(-90 + (R.i0 + e["i0"]) * c, 3), round(-90 + (R.i0 + e["i1"] + 1) * c, 3),
                  round(-180 + (R.j0 + e["j0"]) * c, 3), round(-180 + (R.j0 + e["j1"] + 1) * c, 3)],
            "c": int(e["cls"]), "s": round(float(e["s"]), 3), "n": round(float(a.sum()), 1)}


def detect_months(R, cube, months, nw, clim, has, th, detector, which):
    """Run `detector(img, clim_or_None, th) -> [cell boxes]` on the chosen month positions -> {"YYYY-MM": [events]}.
    Only complete months (>= 4 weekly bins) are analysed. Events are ranked by size."""
    out = {}
    for t in which:
        if nw[t] < 4: continue
        img = cube[t]; ev = detector(img, clim[t] if has[t] else None, th)
        out[month_key(months[t])] = sorted((to_geo(e, R, img) for e in ev), key=lambda x: -x["n"])
    return out
