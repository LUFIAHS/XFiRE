// XFiRE core v3: AOI -> weekly series -> harmonization (ratio | learned estimate: XGBoost / regional block ratio) -> weekly + MONTHLY calendar, anomalies vs the historical
// median, current month vs median, rankings, CSV builders. Pure functions (tested in node, run unchanged in the browser).
const EPOCH=Date.UTC(2000,0,1),DAY=864e5,WEEK=7*DAY,COLS=53;
// VIIRS start comes from the processed data (D.viirs_start); the 2012-02 constant is only a fallback for builds that predate the field.
const viirsW=D=>D._vw??(D._vw=Math.ceil(((D.viirs_start?Date.parse(D.viirs_start):Date.UTC(2012,1,1))-EPOCH)/WEEK));
// weeks inside a sensor's record with NO observations: missing data, never zero fire activity
const gaps=D=>D._gaps||(D._gaps={M:new Set((D.obs_gaps&&D.obs_gaps.MODIS)||[]),V:new Set((D.obs_gaps&&D.obs_gaps.VIIRS)||[])});
const tauOf=D=>+D.recent_tau||0;
const wDate=w=>new Date(EPOCH+w*WEEK);
const iso=d=>d.toISOString().slice(0,10);
const median=a=>{const s=a.filter(x=>x===x).sort((p,q)=>p-q),n=s.length;return n?(n%2?s[n>>1]:(s[n/2-1]+s[n/2])/2):NaN};
const pctl=(a,p)=>{const s=[...a].sort((x,y)=>x-y);return s[Math.min(s.length-1,Math.floor(p*s.length))]};
const rng=a=>()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
// a week belongs to the month that holds its middle day (same rule as harmonize.month_of_week in Python)
const monthOf=w=>{const d=new Date(EPOCH+w*WEEK+3*DAY);return(d.getUTCFullYear()-2000)*12+d.getUTCMonth()};
const daysIn=m=>new Date(Date.UTC(2000+Math.floor(m/12),m%12+1,0)).getUTCDate();
const monthKey=m=>`${2000+Math.floor(m/12)}-${String(m%12+1).padStart(2,'0')}`;
const weekOfDate=s=>Math.floor((Date.parse(s)-EPOCH)/WEEK);
const lastFull=D=>D.last_full_w!=null?D.last_full_w:D.last_w;

function weekInfo(D){
  if(!D._info)D._info=Array.from({length:D.nweeks},(_,w)=>{const d=wDate(w),y=d.getUTCFullYear();return{y,c:Math.floor((d-Date.UTC(y,0,1))/DAY/7),s:((d.getUTCMonth()+1)%12/3)|0}});
  return D._info;
}
function series(D,aoi){
  const n=D.nweeks,M=new Float64Array(n),V=new Float64Array(n),U=D.uc?new Float64Array(n):null;let cells=0;
  for(let i=0;i<D.cells.length;i++){
    const c=D.cells[i];if(aoi&&(c[0]<aoi[0]||c[0]>aoi[1]||c[1]<aoi[2]||c[1]>aoi[3]))continue;
    cells++;const d=c[2];for(let k=0;k<d.length;k+=3){M[d[k]]+=d[k+1];V[d[k]]+=d[k+2]}
    if(U){const u=D.uc[i];for(let k=0;k<u.length;k+=2)U[u[k]]+=u[k+1]}
  }
  return{M,V,U,cells};
}
const ext=D=>D._ext||(D._ext=series(D,null));
function modisLast(D){
  if(D.modis_last_w!=null)return D.modis_last_w;
  if(D._ml==null){const{M}=ext(D);let m=-1;for(let w=0;w<=lastFull(D);w++)if(M[w]>0)m=w;D._ml=m}
  return D._ml;
}

// ---- cross-sensor calibration: VIIRS/MODIS ratio per season on the overlap (only weeks where MODIS still exists) ----
function calibrate(D,M,V,fb){
  const info=weekInfo(D),hi=Math.min(lastFull(D),modisLast(D)),vw=viirsW(D),g=gaps(D),tau=tauOf(D),ys={};
  const ymax=info[Math.max(vw,hi)].y,wt=y=>tau>0?Math.pow(.5,(ymax-y)/tau):1;     // recent-year weighting (only when the build found evidence for it)
  for(let w=vw;w<=hi;w++){if(g.M.has(w)||g.V.has(w))continue;const{y,s}=info[w],q=(ys[y]??=Array.from({length:4},()=>[0,0,0]))[s];q[0]+=M[w]*wt(y);q[1]+=V[w]*wt(y);q[2]+=M[w]}
  const years=Object.keys(ys),ALL=[0,1,2,3];
  const R=(list,ss)=>{let m=0,v=0,r=0;for(const y of list)for(const s of ss){m+=ys[y][s][0];v+=ys[y][s][1];r+=ys[y][s][2]}return{m:r,k:m>0&&v>0?v/m:null}};
  const pooled=R(years,ALL),pooledOk=fb?(pooled.m>=100&&pooled.k):pooled.k;
  const cal=ALL.map(s=>{
    const own=R(years,[s]);let ss,k,m,method;
    if(own.m>=100&&own.k){ss=[s];k=own.k;m=own.m;method='seasonal'}
    else if(pooledOk){ss=ALL;k=pooled.k;m=pooled.m;method='pooled'}
    else if(fb&&fb[s].k)return{...fb[s],method:'extent'};
    else return{k:null,lo:null,hi:null,m:own.m,method:'none'};
    const bs=[],r=rng(s+1);
    if(years.length>=3)for(let b=0;b<300;b++){const x=R(years.map(()=>years[r()*years.length|0]),ss).k;if(x)bs.push(x)}
    return{k,lo:bs.length?pctl(bs,.05):k,hi:bs.length?pctl(bs,.95):k,m,method};
  });
  // out-of-sample overlap error: leave one year out, predict weekly MODIS from VIIRS
  const errs=[];
  for(const y of years){
    const rest=years.filter(q=>q!==y),ks=ALL.map(s=>{const x=R(rest,[s]);return x.m>=100&&x.k?x.k:R(rest,ALL).k});
    for(let w=vw;w<=hi;w++)if(info[w].y==y&&!g.M.has(w)&&!g.V.has(w)&&M[w]>=5&&ks[info[w].s])errs.push(Math.abs(V[w]/ks[info[w].s]-M[w])/M[w]);
  }
  return{cal,mape:errs.length?errs.reduce((a,b)=>a+b)/errs.length:null,tau};
}
const extCal=D=>D._extcal||(D._extcal=calibrate(D,ext(D).M,ext(D).V,null).cal);

function pickMethod(D,method){return method==='unet'&&D.uc?'unet':method==='ratio'?'ratio':(D.harm_method==='unet'&&D.uc?'unet':'ratio')}

function analyze(D,aoi,method){
  method=pickMethod(D,method);
  const n=D.nweeks,L=lastFull(D),ml=modisLast(D),{M,V,U,cells}=series(D,aoi),info=weekInfo(D);
  const{cal,mape}=calibrate(D,M,V,aoi?extCal(D):null);
  // harmonized MODIS-equivalent weekly index + band. The band is HARMONIZATION uncertainty only (sensor-ratio bootstrap, or the held-out log error of the learned
  // method) - it is not a forecast interval. Weeks without observations from a sensor use the other sensor; both missing -> NaN (never zero).
  const h=new Float64Array(n).fill(NaN),lo=new Float64Array(n).fill(NaN),hi=new Float64Array(n).fill(NaN),vw=viirsW(D),g=gaps(D),sd=(D.harm_unc&&D.harm_unc.log_sd_by_season)||[0,0,0,0];
  const comb=(w,m,v)=>w<vw?(g.M.has(w)?NaN:m):w>ml?(g.V.has(w)?NaN:v):g.M.has(w)&&g.V.has(w)?NaN:g.M.has(w)?v:g.V.has(w)?m:.5*(m+v);
  for(let w=D.first_w;w<=L;w++){
    const k=cal[info[w].s];
    if(method==='unet'){const z=1.645*(sd[info[w].s]||0);h[w]=comb(w,M[w],U[w]);lo[w]=comb(w,M[w],U[w]*Math.exp(-z));hi[w]=comb(w,M[w],U[w]*Math.exp(z))}
    else if(!k.k){h[w]=lo[w]=hi[w]=(w<vw||w<=ml)&&!g.M.has(w)?M[w]:NaN}
    else{h[w]=comb(w,M[w],V[w]/k.k);lo[w]=comb(w,M[w],V[w]/k.hi);hi[w]=comb(w,M[w],V[w]/k.lo)}
  }
  // weekly calendar matrix (year x week-of-year), robust leave-one-year-out anomaly
  const y0=info[D.first_w].y,y1=info[L].y,ny=y1-y0+1;
  const mat=Array.from({length:ny},()=>new Float64Array(COLS).fill(NaN)),wk=Array.from({length:ny},()=>new Int32Array(COLS).fill(-1));
  for(let w=D.first_w;w<=L;w++){const{y,c}=info[w];mat[y-y0][c]=h[w];wk[y-y0][c]=w}
  const z=Array.from({length:ny},()=>new Float64Array(COLS).fill(NaN)),zw=new Float64Array(n).fill(NaN),ex=new Float64Array(n);
  for(let c=0;c<COLS;c++)for(let i=0;i<ny;i++){
    const x=mat[i][c];if(x!==x)continue;
    const o=[];for(let j=0;j<ny;j++)if(j!==i&&mat[j][c]===mat[j][c])o.push(mat[j][c]);
    if(o.length<5)continue;
    const med=median(o),mad=median(o.map(v=>Math.abs(v-med))),zz=(x-med)/Math.max(1.4826*mad,Math.sqrt(med+1));
    z[i][c]=zz;zw[wk[i][c]]=zz;ex[wk[i][c]]=x-med;
  }
  // typical fire season = shortest circular window holding >=50% of climatological activity
  const prof=Array.from({length:COLS},(_,c)=>{const v=mat.map(r=>r[c]).filter(x=>x===x);return v.length?v.reduce((a,b)=>a+b)/v.length:0});
  const T=prof.reduce((a,b)=>a+b,0);let season=null;
  for(let l=1;l<=COLS&&!season;l++){let bs=-1,bt=0;for(let s0=0;s0<COLS;s0++){let t=0;for(let i=0;i<l;i++)t+=prof[(s0+i)%COLS];if(t>bt){bt=t;bs=s0}}if(T>0&&bt>=.5*T)season={start:bs,len:l,share:bt/T}}
  // critical periods = runs of weeks with robust z >= 2, ranked by excess detections
  const episodes=[];let cur=null;
  for(let w=D.first_w;w<=L+1;w++){
    if(w<=L&&zw[w]>=2){if(cur&&cur.end===w-1){cur.end=w;cur.peak=Math.max(cur.peak,zw[w]);cur.excess+=ex[w]}else{cur&&episodes.push(cur);cur={start:w,end:w,peak:zw[w],excess:ex[w]}}}
    else if(cur){episodes.push(cur);cur=null}
  }
  episodes.sort((a,b)=>b.excess-a.excess);
  const yearly=[];for(let y=y0+1;y<y1;y++){let r=0,hh=0;for(let w=D.first_w;w<=L;w++)if(info[w].y===y){r+=M[w]+V[w];hh+=h[w]}yearly.push({y,raw:r,harm:hh})}
  let total=0;for(let w=0;w<=L;w++)total+=M[w]+V[w];
  return{method,cal,mape,mat,z,wk,y0,prof,season,episodes:episodes.slice(0,12),yearly,cells,total,h,lo,hi,M,V,U,monthly:monthly(D,h,M,V)};
}

// ---- MONTHLY calendar ---------------------------------------------------------------------------------------------
// value of a month = harmonized detections of its weekly bins, rescaled by days_in_month / (7 * bins) so 4- and 5-week months and
// the running month compare fairly (identical to raster.monthly_cube in Python). Complete month = 4+ weekly bins.
function monthly(D,h,M,V){
  const L=lastFull(D),m0=monthOf(D.first_w),nm=monthOf(L)-m0+1,A=()=>new Float64Array(nm);
  const o={m0,n:nm,wa:new Int32Array(nm).fill(-1),wb:new Int32Array(nm).fill(-1),sum:A(),val:A().fill(NaN),rawM:A(),rawV:A(),nw:new Int32Array(nm),ok:new Uint8Array(nm),
           med:A().fill(NaN),mad:A().fill(NaN),z:A().fill(NaN),pct:A().fill(NaN),below:A().fill(NaN),rank:new Int32Array(nm),nOther:new Int32Array(nm)};
  for(let w=D.first_w;w<=L;w++){if(h[w]!==h[w])continue;const i=monthOf(w)-m0;if(o.wa[i]<0)o.wa[i]=w;o.wb[i]=w;o.nw[i]++;o.sum[i]+=h[w];o.rawM[i]+=M[w];o.rawV[i]+=V[w]}
  for(let i=0;i<nm;i++){if(o.nw[i]){o.val[i]=o.sum[i]*daysIn(m0+i)/(7*o.nw[i])}o.ok[i]=o.nw[i]>=4?1:0}
  for(let i=0;i<nm;i++){
    if(o.val[i]!==o.val[i])continue;
    const moy=(m0+i)%12,others=[];
    for(let j=(moy-m0%12+12)%12;j<nm;j+=12)if(j!==i&&o.ok[j])others.push(o.val[j]);
    o.nOther[i]=others.length;
    if(others.length<4)continue;
    const med=median(others),mad=median(others.map(v=>Math.abs(v-med))),x=o.val[i];
    o.med[i]=med;o.mad[i]=mad;o.z[i]=(x-med)/Math.max(1.4826*mad,Math.sqrt(med+1));o.pct[i]=med>0?(x-med)/med*100:NaN;
    o.below[i]=others.filter(v=>v<x).length/others.length;o.rank[i]=others.filter(v=>v>x).length+1;     // 1 = highest of all years for that month
  }
  return o;
}

// ---- summary for a month range (indices into monthly arrays) -------------------------------------------------------
function summarize(D,A,i0,i1){
  const o=A.monthly,m0=o.m0;i0=Math.max(0,i0);i1=Math.min(o.n-1,i1);
  let total=0,rawM=0,rawV=0,best=null;const months=[];
  for(let i=i0;i<=i1;i++){
    total+=o.sum[i];rawM+=o.rawM[i];rawV+=o.rawV[i];
    if(o.ok[i]){months.push(i);if(!best||o.val[i]>o.val[best])best=i}
  }
  const byYear={};
  for(let i=i0;i<=i1;i++){const y=2000+Math.floor((m0+i)/12);(byYear[y]??={y,n:0,total:0}).n+=o.ok[i]?1:0;byYear[y].total+=o.val[i]===o.val[i]?o.val[i]:0}
  const years=Object.values(byYear).filter(d=>d.n===12);
  years.sort((a,b)=>b.total-a.total);
  const topMonths=months.slice().sort((a,b)=>o.val[b]-o.val[a]).slice(0,10);
  const unusual=months.filter(i=>o.z[i]>=2).sort((a,b)=>o.z[b]-o.z[a]).slice(0,10);
  const last=o.n-1;
  return{total,rawM,rawV,busiest:best,topMonths,topYears:years.slice(0,10),unusual,cur:last,nCompleteYears:years.length};
}
// seasonal profile for the "current vs historical median" chart: per month of year, spread across complete months of ALL years
function seasonal(A,year){
  const o=A.monthly,out=[];
  for(let m=0;m<12;m++){
    const v=[];let cur=NaN,ci=-1;
    for(let i=0;i<o.n;i++)if((o.m0+i)%12===m){
      if(o.ok[i])v.push(o.val[i]);
      if(2000+Math.floor((o.m0+i)/12)===year){cur=o.val[i];ci=i}
    }
    out.push({m,n:v.length,med:median(v),p25:v.length?pctl(v,.25):NaN,p75:v.length?pctl(v,.75):NaN,min:v.length?Math.min(...v):NaN,max:v.length?Math.max(...v):NaN,cur,ci});
  }
  return out;
}

// ---- cell-level harmonized totals for the map (extent-level ratios: stable for any area) -------------------------
function weekWeights(D,method,cal){
  const key='_ww'+method+(cal?JSON.stringify(cal.map(c=>c.k)):'x');if(D[key])return D[key];
  const L=lastFull(D),ml=modisLast(D),info=weekInfo(D),kx=(cal||extCal(D)).map(c=>c.k),useU=method==='unet'&&!!D.uc,wM=new Float64Array(D.nweeks).fill(1),wV=new Float64Array(D.nweeks),vw=viirsW(D),g=gaps(D);
  for(let w=Math.max(D.first_w,vw);w<=L;w++){
    const k=kx[info[w].s];if(!useU&&!k)continue;
    const f=useU?1:1/k;
    if(w>ml||g.M.has(w)){wM[w]=0;wV[w]=f}else if(g.V.has(w)){wM[w]=1;wV[w]=0}else{wM[w]=.5;wV[w]=.5*f}
  }
  for(const w of g.M)if(w<vw&&w<D.nweeks)wM[w]=0;
  return D[key]={wM,wV,useU};
}
// map cell totals use the SAME ratios as the calendar of the current area (cal = analyze().cal), so a drawn area sums to its calendar value
function cellTotals(D,w0,w1,method,cal){
  method=pickMethod(D,method);const{wM,wV,useU}=weekWeights(D,method,cal),L=Math.min(w1,lastFull(D)),out=new Float64Array(D.cells.length);
  for(let i=0;i<D.cells.length;i++){
    const d=D.cells[i][2];let t=0;
    for(let k=0;k<d.length;k+=3){const w=d[k];if(w<w0||w>L)continue;t+=wM[w]*d[k+1]+(useU?0:wV[w]*d[k+2])}
    if(useU){const u=D.uc[i];for(let k=0;k<u.length;k+=2){const w=u[k];if(w>=w0&&w<=L)t+=wV[w]*u[k+1]}}
    out[i]=t;
  }
  return out;
}

// ---- long-term trend: Theil-Sen slope of complete-year totals with a bootstrap interval; computed for the harmonized index AND for raw MODIS alone
// (one sensor, no harmonization) so a sensor-driven trend can be told from a real one. Weeks without observations are excluded.
function trend(D,A){
  const info=weekInfo(D),L=lastFull(D),yrs={};
  for(let w=D.first_w;w<=L;w++){const y=info[w].y,q=(yrs[y]??={n:0,h:0,m:0});q.n++;q.h+=A.h[w]===A.h[w]?A.h[w]:0;q.m+=A.M[w]}
  const mlY=info[Math.min(modisLast(D),L)].y,ys=Object.keys(yrs).map(Number).filter(y=>yrs[y].n>=52&&yrs[y].n<=53);
  const ts=(x,v)=>{const s=[];for(let i=0;i<x.length;i++)for(let j=i+1;j<x.length;j++)if(x[j]!==x[i])s.push((v[j]-v[i])/(x[j]-x[i]));return s.length?median(s):NaN};
  const one=(key,list)=>{
    if(list.length<6)return null;const x=list,v=list.map(y=>yrs[y][key]),sl=ts(x,v),med=median(v),r=rng(7),b=[];
    for(let k=0;k<400;k++){const idx=x.map(()=>r()*x.length|0),xs=idx.map(i=>x[i]),vs=idx.map(i=>v[i]);if(new Set(xs).size<3)continue;const t=ts(xs,vs);if(t===t)b.push(t)}
    let c=0,d=0;for(let i=0;i<x.length;i++)for(let j=i+1;j<x.length;j++){const q=Math.sign(v[j]-v[i]);if(q>0)c++;else if(q<0)d++}
    const P=u=>100*u/Math.max(med,1e-9);
    return{years:x.length,from:x[0],to:x[x.length-1],pct_per_year:P(sl),lo:b.length?P(pctl(b,.05)):null,hi:b.length?P(pctl(b,.95)):null,tau:(c-d)/Math.max(c+d,1)};
  };
  return{harmonized:one('h',ys),modis_only:one('m',ys.filter(y=>y<=mlY)),note:'Complete years only; weeks without observations excluded. If the harmonized and MODIS-only trends disagree, suspect the sensor transition rather than fire activity.'};
}

// ---- CSV builders -------------------------------------------------------------------------------------------------------
const f2=x=>x===x&&x!=null?(Math.round(x*100)/100).toString():'';
const meta=(D,A,what)=>['# XFiRE '+what+' | source: NASA FIRMS (MODIS + VIIRS active fire) | data through '+D.asof+' | built '+D.built+' | XFiRE v'+D.version,
  '# harmonization: '+A.method+(A.method==='unet'?' ('+(D.uc_method||'learned estimate')+')':' (seasonal VIIRS/MODIS ratio)')+' | harmonized values are MODIS-EQUIVALENT ESTIMATES, not satellite measurements | units: hotspots (detections) per week or month',
  '# VIIRS start in the data: '+(D.viirs_start||'n/a')+' | MODIS record ends: '+(D.modis_last_w>=0?iso(wDate(D.modis_last_w+1)):'n/a')];
const eraOf=(D,w)=>{const e=(D.eras||[]).find(e=>w>=e.w0&&w<=e.w1);return e?e.name:''};
function monthlyCsv(D,A,i0,i1,aoi){
  const o=A.monthly,r=[...meta(D,A,'monthly burning-activity table'),'# weeks belong to the month holding their middle day; partial months are rescaled to a full month (complete=0 marks them); weeks without observations are excluded, not zero',
    '# area (lat_min,lat_max,lon_min,lon_max): '+(aoi?aoi.map(x=>x.toFixed(3)).join(','):'whole extent'),
    'year,month,type,weekly_bins,complete,modis_raw,viirs_raw,harmonized,median_same_month_other_years,anomaly_abs,anomaly_pct,robust_z,rank_among_years,flag_unusual'];
  for(let i=Math.max(0,i0);i<=Math.min(o.n-1,i1);i++){
    const m=o.m0+i;r.push([2000+Math.floor(m/12),m%12+1,'harmonized_estimate',o.nw[i],o.ok[i]?1:0,f2(o.rawM[i]),f2(o.rawV[i]),f2(o.val[i]),f2(o.med[i]),f2(o.val[i]-o.med[i]),f2(o.pct[i]),f2(o.z[i]),o.rank[i]&&o.med[i]===o.med[i]?o.rank[i]:'',o.z[i]>=2?1:0].join(','))
  }
  return r.join('\n')+'\n';
}
function weeklyCsv(D,A,w0,w1,aoi){
  const g=gaps(D),r=[...meta(D,A,'weekly harmonized series'),'# area (lat_min,lat_max,lon_min,lon_max): '+(aoi?aoi.map(x=>x.toFixed(3)).join(','):'whole extent'),
    '# harmonization_lo90/hi90 = HARMONIZATION uncertainty only (not a forecast interval)','week_start,sensor_era,type,modis_raw,viirs_raw,learned_modis_equivalent,harmonized,harmonization_lo90,harmonization_hi90,observation_status'];
  for(let w=Math.max(w0,D.first_w);w<=Math.min(w1,lastFull(D));w++)r.push([iso(wDate(w)),'"'+eraOf(D,w)+'"','harmonized_estimate',f2(A.M[w]),f2(A.V[w]),A.U?f2(A.U[w]):'',f2(A.h[w]),f2(A.lo[w]),f2(A.hi[w]),g.M.has(w)&&g.V.has(w)?'both sensors missing':g.M.has(w)?'MODIS missing':g.V.has(w)?'VIIRS missing':'observed'].join(','));
  return r.join('\n')+'\n';
}
function forecastCsv(D,A,f,aoi){
  const k=f.skill||{},r=[...meta(D,A,'26-week FORECAST (model output, not observed fire activity)'),'# area: '+(aoi?aoi.map(x=>x.toFixed(3)).join(','):'whole extent'),
    '# model: '+f.backend+' '+(f.model||'')+' objective='+(k.objective||'n/a')+' | intervals: conformal, forecast-model + residual uncertainty only (harmonization uncertainty is NOT included) | measured backtest coverage 50/80/90% = '+[k.cover50,k.cover80,k.cover].map(x=>x==null?'n/a':x).join('/'),
    'week_start,type,predicted_hotspots,lower_50,upper_50,lower_80,upper_80,lower_90,upper_90'];
  f.weeks.forEach((w,i)=>r.push([iso(wDate(w)),'forecast',f.mean[i],f.lo50?f.lo50[i]:'',f.hi50?f.hi50[i]:'',f.lo80?f.lo80[i]:'',f.hi80?f.hi80[i]:'',f.lo?f.lo[i]:'',f.hi?f.hi[i]:''].join(',')));
  return r.join('\n')+'\n';
}
function gridCsv(D,aoi,w0,w1,method,cal){   // static fallback when the raw hotspots are not served: counts per 0.5-degree cell and week
  const{wM,wV,useU}=weekWeights(D,pickMethod(D,method),cal),L=Math.min(w1,lastFull(D)),r=[...meta(D,{method:pickMethod(D,method)},'hotspot grid counts (0.5 degree cells)'),'latitude,longitude,week_start,modis_count,viirs_count,harmonized'];
  for(let i=0;i<D.cells.length;i++){
    const c=D.cells[i];if(aoi&&(c[0]<aoi[0]||c[0]>aoi[1]||c[1]<aoi[2]||c[1]>aoi[3]))continue;
    const d=c[2],u=useU?new Map():null;if(u){const q=D.uc[i];for(let k=0;k<q.length;k+=2)u.set(q[k],q[k+1])}
    for(let k=0;k<d.length;k+=3){const w=d[k];if(w<w0||w>L)continue;
      r.push([c[0],c[1],iso(wDate(w)),d[k+1],d[k+2],f2(wM[w]*d[k+1]+(useU?wV[w]*(u.get(w)||0):wV[w]*d[k+2]))].join(','))}
  }
  return r.join('\n')+'\n';
}
if(typeof module!=='undefined')module.exports={analyze,series,summarize,seasonal,cellTotals,monthlyCsv,weeklyCsv,gridCsv,forecastCsv,trend,wDate,monthOf,monthKey,daysIn,weekOfDate,viirsW,modisLast,lastFull,median,EPOCH};
