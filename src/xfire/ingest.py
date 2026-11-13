from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"latitude", "longitude", "acq_date", "frp", "confidence"}
OPTIONAL = {"type", "satellite", "daynight", "acq_time"}   # used when the archive has them: fire type (0 = presumed vegetation fire), satellite (Terra / Aqua / N / 1 ...), D/N flag, UTC time
DEDUPE_CAP = 50_000_000                                    # in-memory mode only (tests): max stored record hashes; the pipeline stores them on disk instead
FILTER_ORDER = ("invalid_coordinates", "invalid_date_or_frp", "low_confidence", "non_vegetation", "duplicate")


def _confidence_mask(s, sensor):
    c = s.astype(str).str.strip().str.lower()
    n = pd.to_numeric(c, errors="coerce")
    if sensor == "MODIS":
        return n.ge(30) | c.isin({"nominal", "high"})
    return n.ge(30) | c.isin({"n", "nominal", "h", "high"})


class _Dedupe:
    """Exact-duplicate detector over (lat, lon, date, time, satellite). Hash runs are kept per year as sorted arrays; with `store` (a directory) they live
    on DISK and are searched through memory maps, so RAM stays small however many records there are (about 8 bytes of disk per record, deleted afterwards).
    Without `store` (tests) they stay in memory, capped at DEDUPE_CAP (then complete=False)."""
    def __init__(self, store=None):
        self.store, self.runs, self.total, self.complete, self.n = (Path(store) if store else None), {}, 0, True, 0
        if self.store: self.store.mkdir(parents=True, exist_ok=True)

    def _load(self, r): return np.load(r, mmap_mode="r") if isinstance(r, Path) else r

    def mask(self, df, hashes):
        """True for rows that duplicate an earlier row (earlier chunk/file, or earlier in this chunk)."""
        dup = np.array(pd.Series(hashes, index=df.index).duplicated(), dtype=bool)
        yrs = np.array(df["date"].dt.year)
        for y in np.unique(yrs):
            idx = np.nonzero(yrs == y)[0]; h = hashes[idx]
            for r in self.runs.get(int(y), []):
                arr = self._load(r)
                if len(arr): pos = np.clip(np.searchsorted(arr, h), 0, len(arr) - 1); dup[idx[np.asarray(arr[pos]) == h]] = True
        for y in np.unique(yrs):
            idx = np.nonzero((yrs == y) & ~dup)[0]
            if len(idx) == 0: continue
            if self.store is None and self.total + len(idx) > DEDUPE_CAP: self.complete = False; continue
            run = np.sort(hashes[idx]); self.total += len(idx); self.n += 1
            if self.store is not None:
                path = self.store / f"{int(y)}_{self.n}.npy"; np.save(path, run); run = path
            self.runs.setdefault(int(y), []).append(run)
        return dup

    def cleanup(self):
        if self.store is not None:
            for rs in self.runs.values():
                for r in rs: Path(r).unlink(missing_ok=True)
            try: self.store.rmdir()
            except OSError: pass


def iter_firms(paths, sensor, qc, chunksize=1_000_000, veg_only=True, dedupe=True, dedupe_dir=None):
    """Stream validated detections chunk by chunk (FIRMS archives can be many GB). Timestamps: FIRMS acq_date / acq_time are UTC and are
    binned by UTC date (no timezone conversion).
    Every removal is counted per stage in qc['filters'] (stage order = FILTER_ORDER; a row is counted at the first stage that removes it):
      invalid_coordinates | invalid_date_or_frp | low_confidence | non_vegetation | duplicate
    veg_only: when the archive has a `type` column, keep only type 0 (presumed vegetation fire) and drop active volcanoes (1),
    static land sources such as gas flares and industrial heat (2) and offshore detections (3) - they are not burning and, being
    at fixed places, add the same counts every week (VIIRS 375 m sees many more of them than MODIS, which also biases the ratio).
    Confidence rule, identical in intent for both sensors: MODIS numeric >= 30 or nominal/high; VIIRS n/h (nominal/high) or numeric >= 30."""
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError(f"No {sensor} CSV files supplied.")
    qc.update(sensor=sensor, files=len(paths), rows_read=0, rows_valid=0, first=None, last=None, non_vegetation_removed=0,
              satellites={}, by_year={}, filters={k: 0 for k in FILTER_ORDER}, duplicates_checked=None, duplicate_check_complete=None, columns_seen=[])
    dd = _Dedupe(dedupe_dir) if dedupe else None; qc["file_ranges"] = {}
    for path in paths:
        for i, df in enumerate(pd.read_csv(path, usecols=lambda c: c in REQUIRED or c in OPTIONAL, chunksize=chunksize)):
            missing = REQUIRED - set(df.columns)
            if missing:
                raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
            qc["rows_read"] += len(df); qc["columns_seen"] = sorted(df.columns)
            out = pd.DataFrame({
                "lat": pd.to_numeric(df.latitude, errors="coerce"),
                "lon": pd.to_numeric(df.longitude, errors="coerce"),
                "date": pd.to_datetime(df.acq_date, errors="coerce"),
                "frp": pd.to_numeric(df.frp, errors="coerce"),
                "sensor": sensor,
                "sat": df["satellite"].astype(str).str.strip() if "satellite" in df.columns else "",
                "dn": df["daynight"].astype(str).str.strip().str.upper().str[:1] if "daynight" in df.columns else "",
            })
            alive = np.ones(len(out), bool)
            def stage(name, bad):
                nonlocal alive
                bad = np.asarray(bad) & alive; qc["filters"][name] += int(bad.sum()); alive &= ~bad
            stage("invalid_coordinates", ~(out.lat.between(-90, 90) & out.lon.between(-180, 180)).to_numpy())
            stage("invalid_date_or_frp", ~(out.date.notna() & out.frp.notna() & out.frp.ge(0)).to_numpy())
            stage("low_confidence", ~_confidence_mask(df["confidence"], sensor).to_numpy())
            if veg_only and "type" in df.columns:
                stage("non_vegetation", ~pd.to_numeric(df["type"], errors="coerce").fillna(0).eq(0).to_numpy())
                qc["non_vegetation_removed"] = qc["filters"]["non_vegetation"]
            if dd is not None and "acq_time" in df.columns:
                sub = out.loc[alive]
                if len(sub):
                    key = pd.DataFrame({"a": sub.lat.round(4), "b": sub.lon.round(4), "c": sub.date.dt.strftime("%Y%m%d"),
                                        "d": df.loc[sub.index, "acq_time"].astype(str), "e": sub.sat})
                    h = pd.util.hash_pandas_object(key, index=False).to_numpy()
                    d = dd.mask(sub, h); full = np.zeros(len(out), bool); full[np.nonzero(alive)[0]] = d; stage("duplicate", full)
                qc["duplicates_checked"] = True; qc["duplicate_check_complete"] = bool(dd.complete)
            elif dd is not None:
                qc["duplicates_checked"] = False        # no acq_time column -> a same-pixel, same-day, different-pass pair cannot be told from a duplicate
            out = out.loc[alive]
            if out.empty:
                continue
            qc["rows_valid"] += len(out)
            fr = qc["file_ranges"].setdefault(Path(path).name, [str(out.date.min().date()), str(out.date.max().date())]); fr[0] = min(fr[0], str(out.date.min().date())); fr[1] = max(fr[1], str(out.date.max().date()))
            for code, g in out.groupby("sat"):                             # per-satellite record, used to keep the constellation consistent
                d = qc["satellites"].setdefault(str(code), {"rows": 0, "first": None, "last": None})
                d["rows"] += len(g); a, b = str(g.date.min().date()), str(g.date.max().date())
                d["first"] = a if d["first"] is None else min(d["first"], a); d["last"] = b if d["last"] is None else max(d["last"], b)
            for y, c in out.date.dt.year.value_counts().items():
                qc["by_year"][str(int(y))] = qc["by_year"].get(str(int(y)), 0) + int(c)
            lo, hi = str(out.date.min().date()), str(out.date.max().date())
            qc["first"] = min(filter(None, [qc["first"], lo])); qc["last"] = max(filter(None, [qc["last"], hi]))
            yield out
    if dd is not None: dd.cleanup()
    qc["rows_removed"] = qc["rows_read"] - qc["rows_valid"]
