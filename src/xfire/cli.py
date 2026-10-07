import argparse, glob, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from .pipeline import run


def self_test():
    # Controlled execution test; deliberately labeled as non-scientific.
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); rows=[]
        rng=np.random.default_rng(7)
        for month in pd.date_range('2008-01-01','2020-12-01',freq='MS'):
            n=int(rng.poisson(30*(1+0.2*np.sin(month.month/12*6.28))))
            for i in range(n): rows.append({'latitude':23+rng.normal(0,1),'longitude':90+rng.normal(0,1),'acq_date':month+pd.Timedelta(days=int(rng.integers(0,27))),'frp':float(rng.lognormal(2,0.5)),'confidence':80})
        d=pd.DataFrame(rows); d.to_csv(p/'m.csv',index=False); d.to_csv(p/'v.csv',index=False)
        r=run([p/'m.csv'],[p/'v.csv'],p/'out')
        assert r['product']=='XFiRE' and Path(p/'out/dashboard.html').exists()
    print('XFiRE self-test: PASS')


def main():
    ap=argparse.ArgumentParser(prog='xfire',description='XFiRE cross-sensor fire intelligence pipeline')
    ap.add_argument('--modis',nargs='+'); ap.add_argument('--viirs',nargs='+'); ap.add_argument('--out',default='output'); ap.add_argument('--regions',default=None); ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test: self_test(); return
    if not a.modis or not a.viirs: ap.error('--modis and --viirs are required for real analysis')
    run(a.modis,a.viirs,a.out,a.regions)
    print(f'XFiRE complete → {a.out}/results.json, {a.out}/quality_report.json, {a.out}/dashboard.html')
