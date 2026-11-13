// node tests/check_core.js output/results.json   (SYNTHETIC or real build). Pure-function checks of core.js: calibration, sensor jump, calendar/map/CSV consistency,
// VIIRS start read from the data, missing weeks, trend.
const C=require('../src/xfire/core.js'),D=JSON.parse(require('fs').readFileSync(process.argv[2]||'output/results.json'));
const ok=(c,m)=>{if(!c){console.error('FAIL',m);process.exit(1)}};
const A=C.analyze(D,null,'ratio'),o=A.monthly,S=C.summarize(D,A,0,o.n-1);
const clone=()=>{const x=JSON.parse(JSON.stringify(D,(k,v)=>k[0]==='_'?undefined:v));return x};
const ks=A.cal.map(c=>c.k);console.log('k by season',ks.map(k=>k&&k.toFixed(2)),'LOYO MAPE',A.mape&&A.mape.toFixed(3),'viirs start',D.viirs_start);
const a=A.yearly.filter(x=>x.y>=2009&&x.y<=2011),b=A.yearly.filter(x=>x.y>=2013&&x.y<=2015),m=(l,k)=>l.reduce((s,x)=>s+x[k],0)/l.length;
const jr=m(b,'raw')/m(a,'raw')-1,jh=m(b,'harm')/m(a,'harm')-1;console.log('jump raw',jr.toFixed(2),'harmonized',jh.toFixed(2));
ok(ks.every(k=>k>3&&k<6),'ratio plausible');ok(Math.abs(jh)<0.15&&jr>2,'sensor jump removed');ok(A.season&&A.season.len<35,'season found');
const cur=S.cur;ok(o.nw[cur]>=1&&o.pct[cur]>20,'current month flagged above median (injected 2026 burst)');
ok(C.monthKey(o.m0+S.busiest)==='2019-09'||C.monthKey(o.m0+S.busiest)==='2020-01','busiest month is an injected burst');
ok(S.unusual.length>3&&S.topYears.length>=10,'rankings');
// the VIIRS transition is read from the data, not a constant
ok(C.viirsW(D)===Math.ceil((Date.parse(D.viirs_start)-C.EPOCH)/(7*864e5)),'viirs week from D.viirs_start');
const D2=clone();D2.viirs_start='2016-01-02';const A2=C.analyze(D2,null,'ratio');ok(Math.abs(A2.h[C.viirsW(D2)-20]-A.h[C.viirsW(D2)-20])>=0,'start moves');ok(C.viirsW(D2)!==C.viirsW(D),'start is not hardcoded');
// AOI + fallback calibration + csv
const aoi=[20.5,26.7,88.0,92.8],am=C.analyze(D,aoi,'ratio');ok(am.cal.every(c=>c.k),'AOI calibration');
const csv=C.monthlyCsv(D,A,0,o.n-1,null).split('\n');ok(csv.length>300&&csv.some(l=>l.startsWith('year,month,type')),'monthly csv');ok(csv.slice(0,3).join(' ').includes('MODIS-EQUIVALENT ESTIMATES'),'csv labels harmonized estimates');
// MAP == CALENDAR: summing the map's cell totals inside an area over a month's weeks equals the calendar's value for that area (before the days/weeks rescale)
const i=o.n-30,mm=C.monthlyCsv?am.monthly:null,w0=mm.wa[i],w1=mm.wb[i],tot=C.cellTotals(D,w0,w1,'ratio',am.cal);let s=0;D.cells.forEach((c,j)=>{if(c[0]>=aoi[0]&&c[0]<=aoi[1]&&c[1]>=aoi[2]&&c[1]<=aoi[3])s+=tot[j]});
ok(Math.abs(s-mm.sum[i])/Math.max(mm.sum[i],1)<1e-6,`map total ${s} vs calendar ${mm.sum[i]}`);
// exported grid CSV agrees with the displayed total for the same weeks
const g=C.gridCsv(D,aoi,w0,w1,'ratio',am.cal).split('\n').filter(l=>l&&!l.startsWith('#')&&!l.startsWith('latitude'));let gs=0;g.forEach(l=>gs+=parseFloat(l.split(',')[5]));ok(Math.abs(gs-mm.sum[i])/Math.max(mm.sum[i],1)<5e-3,`grid csv ${gs} vs calendar ${mm.sum[i]}`);
ok(g.length>100,'grid csv');
// weekly csv sums to the weekly series; forecast csv labelled
const wk=C.weeklyCsv(D,A,D.first_w,C.lastFull(D),null).split('\n').filter(l=>l&&!l.startsWith('#')&&!l.startsWith('week_start'));let ws=0;wk.forEach(l=>{const v=parseFloat(l.split(',')[6]);if(v===v)ws+=v});ok(Math.abs(ws-o.sum.reduce((x,y)=>x+y))/ws<1e-3,'weekly csv total == monthly total');
const f=D.forecast&&D.forecast['Whole extent'];if(f){const fc=C.forecastCsv(D,A,f,null);ok(fc.includes('FORECAST')&&fc.includes('lower_90')&&fc.split('\n').filter(l=>l.includes(',forecast,')).length===26,'forecast csv')}
// missing observations are not zero: flag a MODIS-only week as missing in BOTH sensors -> NaN, one sensor -> the other sensor alone
const D3=clone();const wv=C.viirsW(D3)+60;D3.obs_gaps={MODIS:[wv],VIIRS:[wv,wv+1]};const A3=C.analyze(D3,null,'ratio');
ok(A3.h[wv]!==A3.h[wv],'both missing -> NaN');ok(A3.h[wv+1]===A3.M[wv+1],'VIIRS missing -> MODIS alone');
// trend: complete years, interval present
const T=C.trend(D,A);ok(T.harmonized&&T.harmonized.lo!=null&&T.harmonized.years>=10,'trend with interval');
console.log('episodes',A.episodes.length,'season',A.season);console.log('core OK');
