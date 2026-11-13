"""Raw hotspot store -> downloadable CSV of the hotspots detected inside an area and date range.

Detections are written to compressed numpy parts while FIRMS is ingested (opt-in: --keep-hotspots), each part carrying its
date / bounding-box so a query only opens the parts that can match. Pure numpy/pandas; used by serve.py (GET /hotspots.csv).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .grid import EPOCH

HEADER = "latitude,longitude,acq_date,sensor,frp,index_weight\n"


class HotspotStore:
    def __init__(self, folder):
        self.dir = Path(folder); self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.glob("part-*.npz"): f.unlink()
        self.parts, self.total = [], 0

    def add(self, df):
        """df columns: lat, lon, date (datetime), frp, sensor ('MODIS' | 'VIIRS')."""
        if df is None or df.empty: return
        lat, lon = df.lat.to_numpy(np.float32), df.lon.to_numpy(np.float32)
        day = (df.date - EPOCH).dt.days.to_numpy(np.int32); s = (df.sensor == "VIIRS").to_numpy(np.uint8)
        name = f"part-{len(self.parts):05d}.npz"; q = df.q.to_numpy(np.uint8) if "q" in df.columns else np.zeros(len(df), np.uint8)
        np.savez_compressed(self.dir / name, lat=lat, lon=lon, day=day, s=s, frp=df.frp.to_numpy(np.float32), q=q)
        self.parts.append({"file": name, "n": int(len(df)), "d0": int(day.min()), "d1": int(day.max()),
                           "lat0": float(lat.min()), "lat1": float(lat.max()), "lon0": float(lon.min()), "lon1": float(lon.max())})
        self.total += len(df)

    def close(self, policy=None):
        """policy = the grid's satellite policy report: detections it dropped (other VIIRS satellites, MODIS before Aqua) are skipped at query time,
        so the CSV download holds exactly the detections that feed the calendar."""
        p = policy or {}
        pol = {"drop_q": p.get("drop_q") or [], "modis_trim_day": None if p.get("modis_trim_week") is None else int(p["modis_trim_week"]) * 7}
        (self.dir / "index.json").write_text(json.dumps({"total": self.total, "parts": self.parts, "policy": pol}))


def available(folder):
    return (Path(folder) / "index.json").exists()


def _day(s):
    return None if not s else int((pd.Timestamp(s) - EPOCH).days)


def query(folder, aoi=None, d0=None, d1=None, limit=5_000_000):
    """Yield dicts of numpy arrays (lat, lon, day, s, frp) for detections inside aoi=[lat0, lat1, lon0, lon1] and d0..d1 (ISO dates,
    inclusive). Stops after `limit` rows."""
    idx = json.loads((Path(folder) / "index.json").read_text()); a, b = _day(d0), _day(d1); left = limit; pol = idx.get("policy") or {}
    for p in idx["parts"]:
        if left <= 0: return
        if (a is not None and p["d1"] < a) or (b is not None and p["d0"] > b): continue
        if aoi and (p["lat1"] < aoi[0] or p["lat0"] > aoi[1] or p["lon1"] < aoi[2] or p["lon0"] > aoi[3]): continue
        z = np.load(Path(folder) / p["file"]); m = np.ones(p["n"], bool)
        if a is not None: m &= z["day"] >= a
        if b is not None: m &= z["day"] <= b
        if aoi: m &= (z["lat"] >= aoi[0]) & (z["lat"] <= aoi[1]) & (z["lon"] >= aoi[2]) & (z["lon"] <= aoi[3])
        if pol.get("drop_q") and "q" in z.files: m &= ~np.isin(z["q"], pol["drop_q"])
        if pol.get("modis_trim_day") is not None: m &= ~((z["s"] == 0) & (z["day"] < pol["modis_trim_day"]))
        if not m.any(): continue
        out = {k: z[k][m][:left] for k in ("lat", "lon", "day", "s", "frp")}; left -= len(out["day"]); yield out


def csv_chunks(chunks, weights=None):
    """Turn query() output into CSV text pieces (header first). weights=(wM, wVk) adds the per-detection contribution to the
    harmonized index, so the column sums to the harmonized count shown in the calendar (ratio method)."""
    yield HEADER
    for c in chunks:
        w = (c["day"] // 7).clip(0, None)
        iw = np.full(len(w), np.nan)
        if weights is not None:
            wM, wVk = weights; w = np.minimum(w, len(wM) - 1); iw = np.where(c["s"] == 1, wVk[w], wM[w])
        df = pd.DataFrame({"latitude": c["lat"].round(4), "longitude": c["lon"].round(4),
                           "acq_date": (EPOCH + pd.to_timedelta(c["day"], unit="D")).strftime("%Y-%m-%d"),
                           "sensor": np.where(c["s"] == 1, "VIIRS", "MODIS"), "frp": c["frp"].round(1), "index_weight": np.round(iw, 4)})
        yield df.to_csv(index=False, header=False)
