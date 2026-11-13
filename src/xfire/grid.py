"""Aggregate detections into (cell x week x sensor) counts so ANY area of interest
can be analysed later in the browser without keeping raw detections."""
import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("2000-01-01")
SENSORS = {"MODIS": 0, "VIIRS": 1}


class GridAccumulator:
    def __init__(self, cell=0.5, boxes=None):
        self.cell, self.boxes, self.parts, self.maxdate = cell, boxes, [], None
        self.sats = {}                                   # (sensor, satellite code) -> q, the satellite key of the aggregation

    def add(self, df):
        """Aggregate a chunk; returns the rows that were kept (inside the date range and regions) so callers can store them."""
        df = df[df.date >= EPOCH]
        if self.boxes:
            m = pd.Series(False, index=df.index)
            for a, b, c, d in self.boxes:
                m |= df.lat.between(a, b) & df.lon.between(c, d)
            df = df[m]
        if df.empty:
            return df
        i = np.floor((df.lat + 90) / self.cell).astype(int)
        j = np.floor((df.lon + 180) / self.cell).astype(int)
        w = ((df.date - EPOCH).dt.days // 7).astype(int)
        s = df.sensor.map(SENSORS).astype(int)
        sat = df["sat"] if "sat" in df.columns else pd.Series("", index=df.index)
        for code in sat.unique(): self.sats.setdefault((str(df.sensor.iloc[0]), str(code)), len(self.sats))
        q = sat.map({c: self.sats[(str(df.sensor.iloc[0]), str(c))] for c in sat.unique()}).astype(int)
        self.parts.append(df.assign(s=s, i=i, j=j, w=w, q=q).groupby(["s", "i", "j", "w", "q"]).size())
        mx = df.date.max()
        self.maxdate = mx if self.maxdate is None else max(self.maxdate, mx)
        return df.assign(q=q)                              # q lets the raw-hotspot store apply the same satellite policy as the grid

    def policy(self, ser, mode="auto", viirs_satellite=None, trim_modis=True):
        """Keep the sensor records consistent in time. Returns (filtered series, report).
        * VIIRS: more than one satellite in the files (S-NPP 2012, NOAA-20 2018, NOAA-21 2023 ...) means the counts jump every time a
          satellite joins, which looks like a fire trend and corrupts the MODIS/VIIRS ratio -> keep ONE satellite (the longest-running,
          or `viirs_satellite`).
        * MODIS: Terra alone before Aqua joined (2002-07) counts about half -> drop MODIS weeks before both satellites were observing.
        mode "off" keeps everything (the v3.0 behaviour)."""
        rep = {"mode": mode, "viirs_kept": None, "viirs_dropped": [], "modis_trim_week": None, "modis_trim_date": None, "drop_q": [], "rows_removed": 0}
        if mode == "off" or ser.empty: return ser, rep
        S_, W_, Q_ = (ser.index.get_level_values(k).to_numpy() for k in (0, 3, 4)); N_ = ser.to_numpy()
        info = {}
        for (sensor, code), q in self.sats.items():
            m = Q_ == q
            if m.any(): info[q] = {"sensor": sensor, "code": code, "first": int(W_[m].min()), "rows": int(N_[m].sum())}
        drop = set()
        v = {q: d for q, d in info.items() if d["sensor"] == "VIIRS" and d["code"] not in ("", "nan")}
        if v and (len(v) > 1 or viirs_satellite):
            pick = [q for q, d in v.items() if viirs_satellite and d["code"].lower() == str(viirs_satellite).lower()]
            keep = pick[0] if pick else min(v, key=lambda q: (v[q]["first"], -v[q]["rows"]))
            rep["viirs_kept"] = v[keep]["code"]; rep["viirs_dropped"] = [v[q]["code"] for q in v if q != keep]; drop |= {q for q in v if q != keep}
        mo = {q: d for q, d in info.items() if d["sensor"] == "MODIS" and d["code"] not in ("", "nan")}
        if trim_modis and len(mo) > 1:
            start, first = max(d["first"] for d in mo.values()), min(d["first"] for d in mo.values())
            if start - first >= 8:                                      # a real gap between the satellites, not a few days of overlap
                rep["modis_trim_week"] = int(start); rep["modis_trim_date"] = str((EPOCH + pd.Timedelta(days=7 * int(start))).date())
        keepm = ~np.isin(Q_, list(drop))
        if rep["modis_trim_week"] is not None: keepm &= ~((S_ == 0) & (W_ < rep["modis_trim_week"]))
        rep["rows_removed"] = int(N_[~keepm].sum()); rep["drop_q"] = sorted(int(q) for q in drop)
        return ser[keepm], rep

    def export(self, mode="auto", viirs_satellite=None, trim_modis=True):
        if not self.parts:
            raise ValueError("No valid detections fall inside the configured regions.")
        ser = pd.concat(self.parts).groupby(level=[0, 1, 2, 3, 4]).sum()
        ser, self.policy_report = self.policy(ser, mode, viirs_satellite, trim_modis)
        if ser.empty:
            raise ValueError("No detections left after the satellite-consistency policy.")
        t = ser.groupby(level=[0, 1, 2, 3]).sum().unstack(0, fill_value=0)
        t = t.reindex(columns=[0, 1], fill_value=0).astype(int)
        cells = []
        for (i, j), g in t.groupby(level=[0, 1]):
            w = g.index.get_level_values(2).to_numpy()
            flat = np.column_stack([w, g[0].to_numpy(), g[1].to_numpy()]).ravel().tolist()
            cells.append([round(-90 + (i + .5) * self.cell, 4), round(-180 + (j + .5) * self.cell, 4), flat])
        ws = t.index.get_level_values(2)
        return {"cell": self.cell, "cells": cells, "first_w": int(ws.min()), "last_w": int(ws.max()),
                "nweeks": int(ws.max()) + 1, "asof": str(self.maxdate.date())}
