import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def anomaly(series):
    if len(series) < 13: return None, "INSUFFICIENT_HISTORY"
    latest = series.index[-1]
    base = series[(series.index.month == latest.month) & (series.index.year != latest.year)]
    if len(base) < 3: return None, "INSUFFICIENT_BASELINE"
    sd = float(base.std()) or 1.0
    z = float((series.iloc[-1]-base.mean())/sd)
    label = "EXTREME" if z >= 2.5 else "HIGH" if z >= 1.5 else "ELEVATED" if z >= .5 else "NORMAL"
    return round(z,2), label


def cluster_events(df, days=90):
    r = df[df.date >= df.date.max()-pd.Timedelta(days=days)].copy()
    if r.empty: return []
    t0 = r.date.min()
    x = np.column_stack([r.lat*111.0, r.lon*111.0*np.cos(np.radians(r.lat)), (r.date-t0).dt.total_seconds()/86400*2.0])
    r["event_id"] = DBSCAN(eps=4, min_samples=2).fit_predict(x)
    out=[]
    for eid,g in r[r.event_id>=0].groupby("event_id"):
        out.append({"id":int(eid),"lat":round(g.lat.mean(),4),"lon":round(g.lon.mean(),4),"region":str(g.region.mode().iat[0]),"detections":len(g),"sensors":"+".join(sorted(g.sensor.unique())),"frp_total":round(float(g.frp.sum()),1),"frp_peak":round(float(g.frp.max()),1),"persistence_days":int(g.date.dt.date.nunique()),"first":str(g.date.min().date()),"last":str(g.date.max().date())})
    return sorted(out,key=lambda x:-x["frp_total"])[:500]
