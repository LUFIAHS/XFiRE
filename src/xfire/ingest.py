from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {"latitude", "longitude", "acq_date", "frp", "confidence"}


def _confidence_mask(s, sensor):
    c = s.astype(str).str.strip().str.lower()
    n = pd.to_numeric(c, errors="coerce")
    if sensor == "MODIS":
        return n.ge(30) | c.isin({"nominal", "high"})
    return n.ge(30) | c.isin({"n", "nominal", "h", "high"})


def load_firms(paths, sensor):
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError(f"No {sensor} CSV files supplied.")
    frames, qc = [], {"sensor": sensor, "files": len(paths), "rows_read": 0, "rows_valid": 0, "rows_removed": 0}
    for path in paths:
        df = pd.read_csv(path)
        qc["rows_read"] += len(df)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{Path(path).name}: missing required columns: {sorted(missing)}")
        keep = _confidence_mask(df["confidence"], sensor)
        out = pd.DataFrame({
            "lat": pd.to_numeric(df.latitude, errors="coerce"),
            "lon": pd.to_numeric(df.longitude, errors="coerce"),
            "date": pd.to_datetime(df.acq_date, errors="coerce"),
            "frp": pd.to_numeric(df.frp, errors="coerce"),
            "sensor": sensor,
        })
        keep &= out.lat.between(-90, 90) & out.lon.between(-180, 180)
        keep &= out.date.notna() & out.frp.notna() & out.frp.ge(0)
        out = out.loc[keep].copy()
        frames.append(out)
    result = pd.concat(frames, ignore_index=True)
    qc["rows_valid"] = len(result)
    qc["rows_removed"] = qc["rows_read"] - qc["rows_valid"]
    return result, qc
