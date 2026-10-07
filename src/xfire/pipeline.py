import json
from pathlib import Path
import pandas as pd
from .config import load_regions
from .ingest import load_firms
from .harmonize import harmonize_monthly
from .intelligence import anomaly, cluster_events
from .dashboard import write_dashboard


def assign_region(df, regions):
    out = pd.Series('Other', index=df.index, dtype='object')
    for name,(a,b,c,d) in regions.items():
        out.loc[df.lat.between(a,b)&df.lon.between(c,d)] = name
    return out


def run(modis_paths, viirs_paths, out_dir, regions_path=None):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    modis,qm=load_firms(modis_paths,'MODIS'); viirs,qv=load_firms(viirs_paths,'VIIRS')
    df=pd.concat([modis,viirs],ignore_index=True); regions=load_regions(regions_path)
    df['region']=assign_region(df,regions); df=df[df.region!='Other'].copy(); df['month']=df.date.dt.to_period('M').dt.to_timestamp()
    if df.empty: raise ValueError('No valid detections fall inside configured regions.')
    full=pd.date_range(df.month.min(),df.month.max(),freq='MS')
    tab=df.groupby(['region','month','sensor']).size().unstack('sensor',fill_value=0)
    for s in ['MODIS','VIIRS']:
        if s not in tab: tab[s]=0
    regs={r:tab.loc[r].reindex(full,fill_value=0)[['MODIS','VIIRS']] for r in tab.index.get_level_values(0).unique()}
    regs['All regions']=sum(regs.values())
    res={'product':'XFiRE','version':'1.0.0','asof':str(df.date.max().date()),'series':{},'info':{},'anomalies':[],'events':[],'regions':{k:list(v) for k,v in regions.items()},'quality':{'MODIS':qm,'VIIRS':qv,'rows_after_region_filter':len(df)}}
    for name,t in regs.items():
        h,info=harmonize_monthly(t); res['series'][name]={'months':[x.strftime('%Y-%m') for x in h.index],**{c:[round(float(v),3) for v in h[c]] for c in ['raw','harm','lo','hi']}}; res['info'][name]=info
        z,label=anomaly(h.harm); res['anomalies'].append({'region':name,'z':z,'label':label})
    res['events']=cluster_events(df)
    (out/'results.json').write_text(json.dumps(res,indent=2),encoding='utf-8'); (out/'quality_report.json').write_text(json.dumps(res['quality'],indent=2),encoding='utf-8'); write_dashboard(res,out/'dashboard.html')
    return res
