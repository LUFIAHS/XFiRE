import json
from pathlib import Path

TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>XFiRE | Cross-Sensor Fire Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4"></script><script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
:root{--bg:#06080c;--s1:#0d1219;--s2:#121a24;--line:#1c2733;--tx:#e8eef6;--mu:#8190a3;--ac:#ff6a2b;--ac2:#ffb347;--modis:#60a5fa;--viirs:#c084fc;--both:#34d399;--r:14px}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1200px 600px at 85% -10%,rgba(255,106,43,.13),transparent 60%),radial-gradient(900px 500px at -10% 10%,rgba(56,120,255,.08),transparent 60%),var(--bg);color:var(--tx);font:14px/1.5 Inter,system-ui,sans-serif;min-height:100vh}
header{position:sticky;top:0;z-index:1000;display:flex;align-items:center;gap:18px;padding:14px 32px;background:rgba(6,8,12,.78);backdrop-filter:blur(14px);border-bottom:1px solid var(--line);flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:12px}.mark{width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,var(--ac),var(--ac2));display:grid;place-items:center;font-size:20px;box-shadow:0 6px 24px rgba(255,106,43,.35)}
h1{margin:0;font-size:21px;font-weight:800;letter-spacing:.3px}h1 b{background:linear-gradient(90deg,var(--ac),var(--ac2));-webkit-background-clip:text;background-clip:text;color:transparent}.tag{font-size:11px;color:var(--mu);letter-spacing:.14em;text-transform:uppercase}
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border:1px solid var(--line);border-radius:99px;background:var(--s1);font-size:12px;color:var(--mu)}.pill b{color:var(--tx);font-family:'JetBrains Mono',monospace;font-weight:500}
.dot{width:8px;height:8px;border-radius:50%;background:var(--both);box-shadow:0 0 0 0 rgba(52,211,153,.6);animation:p 2s infinite}@keyframes p{70%{box-shadow:0 0 0 8px transparent}}
.sp{margin-left:auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
select,.btn{background:var(--s1);color:var(--tx);border:1px solid var(--line);padding:8px 12px;border-radius:9px;font:inherit;font-size:13px;cursor:pointer;transition:.15s}select:hover,.btn:hover{border-color:#34465a;background:var(--s2)}
main{max-width:1500px;margin:0 auto;padding:24px 32px 40px;display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
.hero{grid-column:1/-1;padding:20px 24px;border-radius:var(--r);border:1px solid rgba(255,106,43,.28);background:linear-gradient(110deg,rgba(255,106,43,.14),rgba(255,179,71,.04) 55%,transparent);display:flex;gap:16px;align-items:center}
.hero .ic{font-size:26px}.hero small{color:var(--ac2);letter-spacing:.14em;text-transform:uppercase;font-weight:700;font-size:11px}.hero p{margin:3px 0 0;font-size:16px;font-weight:500;line-height:1.5}.hero p b{color:var(--ac2)}
.kpis{grid-column:1/-1;display:grid;grid-template-columns:repeat(6,1fr);gap:12px}
.k{background:linear-gradient(180deg,var(--s2),var(--s1));border:1px solid var(--line);border-radius:var(--r);padding:16px 18px;position:relative;overflow:hidden}.k:before{content:'';position:absolute;inset:0 0 auto 0;height:2px;background:var(--c,var(--ac));opacity:.8}
.k label{color:var(--mu);font-size:11px;letter-spacing:.1em;text-transform:uppercase;font-weight:600}.k strong{display:block;font-size:28px;font-weight:800;margin:6px 0 2px;letter-spacing:-.5px}.k em{font-style:normal;color:var(--mu);font-size:12px}.up{color:var(--both)!important}
.card{background:linear-gradient(180deg,var(--s2),var(--s1));border:1px solid var(--line);border-radius:var(--r);padding:18px 20px;min-width:0}
.card h3{margin:0 0 2px;font-size:15px;font-weight:700}.card .sub{color:var(--mu);font-size:12px;margin-bottom:14px}.ch{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap}
.c8{grid-column:span 8}.c7{grid-column:span 7}.c5{grid-column:span 5}.c4{grid-column:span 4}.c6{grid-column:span 6}.c12{grid-column:1/-1}
.seg{display:inline-flex;background:var(--bg);border:1px solid var(--line);border-radius:9px;padding:2px}.seg button{background:none;border:0;color:var(--mu);padding:5px 11px;border-radius:7px;font:inherit;font-size:12px;cursor:pointer}.seg button.on{background:var(--s2);color:var(--tx);box-shadow:0 0 0 1px #2a3a4c}
.cv{position:relative;height:310px}#map{height:430px;border-radius:10px;background:#0a0e14;border:1px solid var(--line)}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:#0d1219;color:var(--tx);border:1px solid var(--line)}.leaflet-popup-content{margin:12px 14px;font-size:12.5px;line-height:1.6}.leaflet-container a{color:var(--ac2)}
.legend{display:flex;gap:14px;align-items:center;margin-top:10px;color:var(--mu);font-size:12px;flex-wrap:wrap}.grad{width:110px;height:8px;border-radius:5px;background:linear-gradient(90deg,#ffd166,#ff6a2b,#e5173f)}
table{width:100%;border-collapse:collapse;font-size:13px}th{color:var(--mu);font-size:11px;letter-spacing:.08em;text-transform:uppercase;text-align:left;font-weight:600;padding:8px 10px;border-bottom:1px solid var(--line)}td{padding:10px;border-bottom:1px solid #151e29}tr.r{cursor:pointer;transition:.12s}tr.r:hover td{background:rgba(255,255,255,.03)}tr:last-child td{border:0}.num{font-family:'JetBrains Mono',monospace;font-size:12.5px}
.chip{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;font-weight:700;letter-spacing:.04em;border:1px solid}
.EXTREME{color:#ff5d73;background:rgba(255,93,115,.1);border-color:rgba(255,93,115,.4)}.HIGH{color:#ff9f43;background:rgba(255,159,67,.1);border-color:rgba(255,159,67,.4)}.ELEVATED{color:#f5d04c;background:rgba(245,208,76,.1);border-color:rgba(245,208,76,.4)}.NORMAL{color:#4ade80;background:rgba(74,222,128,.1);border-color:rgba(74,222,128,.4)}.NA{color:var(--mu);border-color:var(--line)}
.MODIS{color:var(--modis);border-color:rgba(96,165,250,.4)}.VIIRS{color:var(--viirs);border-color:rgba(192,132,252,.4)}.BOTH{color:var(--both);background:rgba(52,211,153,.1);border-color:rgba(52,211,153,.45)}
.bar{height:6px;border-radius:4px;background:#17212c;position:relative;min-width:60px}.bar i{position:absolute;inset:0 auto 0 0;border-radius:4px;background:linear-gradient(90deg,var(--ac),var(--ac2))}
.zb{height:8px;border-radius:5px;background:#17212c;position:relative;min-width:110px}.zb:after{content:'';position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#3a4a5c}.zb i{position:absolute;top:0;bottom:0;border-radius:5px}
.cal{display:grid;grid-template-columns:44px 1fr 120px;gap:12px;align-items:center;margin:14px 0;font-size:12.5px}.rng{height:10px;background:#17212c;border-radius:6px;position:relative}.rng i{position:absolute;top:0;bottom:0;border-radius:6px;background:linear-gradient(90deg,rgba(255,106,43,.35),rgba(255,179,71,.6))}.rng u{position:absolute;top:-3px;bottom:-3px;width:3px;background:#fff;border-radius:2px}
.steps{display:grid;grid-template-columns:repeat(6,1fr);gap:10px}.st{padding:14px;border:1px solid var(--line);border-radius:12px;background:var(--bg);position:relative}.st b{display:block;font-size:12.5px}.st span{color:var(--mu);font-size:11.5px}.st n{display:grid;place-items:center;width:22px;height:22px;border-radius:7px;background:linear-gradient(135deg,var(--ac),var(--ac2));color:#1a0d05;font-weight:800;font-size:11px;margin-bottom:8px}
.qrow{display:flex;justify-content:space-between;font-size:12px;color:var(--mu);margin:12px 0 5px}.qrow b{color:var(--tx)}
.note{color:var(--mu);font-size:12px;margin-top:14px;padding:10px 12px;border-left:2px solid var(--ac);background:rgba(255,106,43,.05);border-radius:0 8px 8px 0}
footer{text-align:center;color:var(--mu);font-size:12px;padding:10px 0 34px}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}.c8,.c7,.c5,.c4,.c6{grid-column:1/-1}.steps{grid-template-columns:repeat(3,1fr)}}@media(max-width:640px){main,header{padding-left:16px;padding-right:16px}.kpis{grid-template-columns:repeat(2,1fr)}.steps{grid-template-columns:repeat(2,1fr)}}
@media print{header,.sp,.seg{display:none}body{background:#fff;color:#000}}
.inv{filter:invert(1) hue-rotate(180deg) brightness(.85) contrast(.9)}</style></head><body>
<header><div class="logo"><div class="mark">🔥</div><div><h1><b>XFiRE</b></h1><div class="tag">Cross-Sensor Fire Intelligence</div></div></div>
<span class="pill"><span class="dot"></span>As of <b id="asof"></b></span><span class="pill">MODIS <b>+</b> VIIRS · NASA FIRMS</span>
<div class="sp"><select id="region" aria-label="Region"></select><button class="btn" id="expCsv">⬇ Events CSV</button><button class="btn" id="expJson">⬇ JSON</button><button class="btn" onclick="print()">🖨 Report</button></div></header>
<main>
<section class="hero"><div class="ic">🛰️</div><div><small>Intelligence brief</small><p id="brief"></p></div></section>
<section class="kpis" id="kpis"></section>
<section class="card c8"><div class="ch"><div><h3>Harmonized fire activity</h3><div class="sub">MODIS-equivalent index with 90% uncertainty band · dotted line marks VIIRS launch</div></div>
<div style="display:flex;gap:8px;flex-wrap:wrap"><div class="seg" id="rng"><button data-r="0" class="on">All</button><button data-r="120">10y</button><button data-r="60">5y</button><button data-r="24">2y</button></div><div class="seg" id="tg"><button data-t="raw" class="on">Raw</button><button data-t="band" class="on">Band</button></div></div></div><div class="cv"><canvas id="chart"></canvas></div></section>
<section class="card c4"><h3>Seasonal context</h3><div class="sub">Last 12 months vs. historical monthly mean</div><div class="cv"><canvas id="clim"></canvas></div></section>
<section class="card c7"><div class="ch"><div><h3>Persistent fire events · 90 days</h3><div class="sub">Space-time clusters (DBSCAN) · click a marker or table row</div></div></div><div id="map"></div>
<div class="legend"><span>Event FRP</span><span class="grad"></span><span>low → high</span><span><svg width="14" height="14"><circle cx="7" cy="7" r="5" fill="none" stroke="#34d399" stroke-width="2"/></svg> confirmed by both sensors</span></div></section>
<section class="card c5"><div class="ch"><div><h3>Highest-impact events</h3><div class="sub">Ranked within selected region</div></div><div class="seg" id="srt"><button data-s="frp_total" class="on">FRP</button><button data-s="persistence_days">Persistence</button></div></div><table id="events"></table></section>
<section class="card c6"><h3>Regional anomaly monitor</h3><div class="sub">Latest month vs. same-calendar-month baseline (z-score) · click to focus</div><table id="anom"></table></section>
<section class="card c6"><h3>Calibration uncertainty</h3><div class="sub">VIIRS ÷ MODIS ratio by season · bar = 90% bootstrap interval, marker = estimate</div><div id="cal"></div><div class="note">XFiRE reports calibration error instead of assuming the sensors are identical.</div></section>
<section class="card c7"><h3>Does harmonization remove the sensor artifact?</h3><div class="sub">Change in activity across the VIIRS transition, per region</div><table id="quality"></table></section>
<section class="card c5"><h3>Data quality</h3><div class="sub" id="qsub"></div><div id="qc"></div></section>
<section class="card c12"><h3>Pipeline</h3><div class="sub">From raw satellite detections to uncertainty-aware intelligence</div><div class="steps" id="steps"></div></section>
</main><footer id="foot"></footer>
<script>
const D=__DATA__,$=s=>document.querySelector(s),sel=$('#region');
const A=Object.fromEntries(D.anomalies.map(a=>[a.region,a]));
const names=Object.keys(D.series).sort((a,b)=>a==='All regions'?-1:b==='All regions'?1:a.localeCompare(b));
names.forEach(n=>sel.add(new Option(n)));$('#asof').textContent=D.asof;
const pct=x=>x==null?'—':(x*100).toFixed(1)+'%',sp=x=>x==null?'—':(x>0?'+':'')+(x*100).toFixed(1)+'%',nf=x=>Math.round(x).toLocaleString();
const sens=s=>s.includes('+')?'BOTH':s,sensLbl=s=>s.includes('+')?'Cross-validated':s;
const lab=a=>a?(a.label.startsWith('INSUFF')?'NA':a.label):'NA',labTxt=a=>a?(a.label.startsWith('INSUFF')?'Insufficient data':a.label):'—';
let st={rng:0,raw:true,band:true,sort:'frp_total'},chart,clim,map,layer,marks={},evs=[];
Chart.defaults.color='#8190a3';Chart.defaults.borderColor='#1c2733';Chart.defaults.font.family='Inter,system-ui,sans-serif';
const launch={id:'launch',afterDraw(c){const i=c.data.labels.indexOf('2012-02');if(i<0)return;const x=c.scales.x.getPixelForValue(i),y=c.chartArea;const g=c.ctx;g.save();g.setLineDash([3,4]);g.strokeStyle='#c084fc';g.beginPath();g.moveTo(x,y.top);g.lineTo(x,y.bottom);g.stroke();g.fillStyle='#c084fc';g.font='11px Inter';g.fillText('VIIRS',x+5,y.top+12);g.restore()}};
const HAS_L=typeof L!=='undefined';
if(HAS_L){map=L.map('map',{zoomControl:true,minZoom:2,worldCopyJump:true}).setView([20,10],2);
const T=[['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',{attribution:'Tiles © Esri — Esri, HERE, Garmin, OpenStreetMap contributors',maxNativeZoom:16}],['https://tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap contributors',className:'inv'}]];
let ti=0,te=0,tl,lb;const useTiles=()=>{if(tl)map.removeLayer(tl);if(lb)map.removeLayer(lb);tl=L.tileLayer(T[ti][0],Object.assign({maxZoom:12},T[ti][1])).addTo(map);if(ti===0)lb=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',{maxZoom:12,maxNativeZoom:16,opacity:.8}).addTo(map);tl.on('tileerror',()=>{if(++te>=4&&ti<T.length-1){ti++;te=0;useTiles()}})};useTiles();
L.control.scale({imperial:false}).addTo(map);layer=L.layerGroup().addTo(map);addEventListener('load',()=>map.invalidateSize())}
else $('#map').innerHTML='<p style="padding:24px;color:#8190a3">Map library failed to load (check internet access to cdn.jsdelivr.net).</p>';
const col=t=>t<.5?`hsl(${45-t*40},100%,62%)`:`hsl(${25-(t-.5)*40},95%,${55-(t-.5)*14}%)`;
function slice(a,n){return n?a.slice(-n):a}
function kpis(r,i,ev){const conf=ev.filter(e=>e.sensors.includes('+')).length,a=A[r],red=i.raw_transition!=null&&i.harmonized_transition!=null?(Math.abs(i.raw_transition)-Math.abs(i.harmonized_transition))*100:null;
const K=[['Persistent events',nf(ev.length),'detected in last 90 days','#ff6a2b'],['Cross-validated',ev.length?Math.round(conf/ev.length*100)+'%':'—',conf+' seen by both sensors','#34d399'],['Total event FRP',nf(ev.reduce((s,e)=>s+e.frp_total,0)),'megawatts, summed','#ffb347'],
['Sensor artifact',sp(i.raw_transition)+' → '+sp(i.harmonized_transition),red==null?'':(red>=0?'▼ ':'▲ ')+Math.abs(red).toFixed(1)+' pp '+(red>=0?'removed':'added'),'#c084fc',red>0],['Overlap error (MAPE)',pct(i.mape_overlap),'VIIRS→MODIS calibration','#60a5fa'],['Current anomaly',labTxt(a),a&&a.z!=null?'z = '+a.z:'latest month','#f5d04c']];
$('#kpis').innerHTML=K.map(k=>`<div class="k" style="--c:${k[3]}"><label>${k[0]}</label><strong style="${k[1].length>9?'font-size:19px;margin:12px 0 8px':''}">${k[1]}</strong><em class="${k[4]?'up':''}">${k[2]}</em></div>`).join('');
$('#brief').innerHTML=`<b>${r}</b>: harmonization moved the apparent sensor-switch jump from <b>${sp(i.raw_transition)}</b> to <b>${sp(i.harmonized_transition)}</b>, with <b>${pct(i.mape_overlap)}</b> overlap error. Latest month is <b>${labTxt(a)}</b>${a&&a.z!=null?' (z '+a.z+')':''}; <b>${ev.length}</b> persistent events, <b>${conf}</b> confirmed by both sensors.`}
function charts(r){const s=D.series[r],L_=slice(s.months,st.rng||0),n=st.rng||0,
ds=[{label:'hi',data:slice(s.hi,n),borderWidth:0,pointRadius:0,hidden:!st.band,fill:false},{label:'90% interval',data:slice(s.lo,n),borderWidth:0,pointRadius:0,hidden:!st.band,fill:'-1',backgroundColor:'rgba(255,106,43,.16)'},
{label:'Raw detections',data:slice(s.raw,n),borderColor:'#5b6b7e',borderWidth:1.2,borderDash:[4,3],pointRadius:0,hidden:!st.raw},{label:'XFiRE harmonized',data:slice(s.harm,n),borderColor:'#ff8a4c',borderWidth:2.2,pointRadius:0,tension:.25}];
if(chart)chart.destroy();chart=new Chart($('#chart'),{type:'line',data:{labels:L_,datasets:ds},plugins:[launch],options:{maintainAspectRatio:false,animation:{duration:500},interaction:{mode:'index',intersect:false},plugins:{legend:{align:'end',labels:{usePointStyle:true,boxWidth:8,filter:i=>i.text!=='hi'}},tooltip:{backgroundColor:'#0d1219',borderColor:'#2a3a4c',borderWidth:1,padding:10,filter:i=>i.dataset.label!=='hi'}},scales:{x:{ticks:{maxTicksLimit:9},grid:{display:false}},y:{beginAtZero:true,ticks:{callback:v=>nf(v)}}}}});
const yrs=s.months.map(m=>+m.slice(0,4)),ly=yrs[yrs.length-1],base=Array(12).fill(0),cnt=Array(12).fill(0);s.months.forEach((m,k)=>{if(yrs[k]<ly){const j=+m.slice(5)-1;base[j]+=s.harm[k];cnt[j]++}});
const cur=Array(12).fill(null);s.months.slice(-12).forEach((m,k)=>cur[+m.slice(5)-1]=s.harm[s.harm.length-12+k]);
if(clim)clim.destroy();clim=new Chart($('#clim'),{data:{labels:['J','F','M','A','M','J','J','A','S','O','N','D'],datasets:[{type:'bar',label:'Historical mean',data:base.map((v,k)=>cnt[k]?v/cnt[k]:null),backgroundColor:'rgba(96,165,250,.35)',borderRadius:4},{type:'line',label:'Last 12 months',data:cur,borderColor:'#ff8a4c',backgroundColor:'#ff8a4c',pointRadius:3,borderWidth:2,tension:.3}]},options:{maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{usePointStyle:true,boxWidth:8}}},scales:{x:{grid:{display:false}},y:{ticks:{callback:v=>nf(v)}}}}})}
function mapdraw(r,ev){if(!HAS_L)return;layer.clearLayers();marks={};const mx=Math.max(1,...ev.map(e=>e.frp_total));
ev.forEach(e=>{const t=Math.sqrt(e.frp_total/mx),b=e.sensors.includes('+'),m=L.circleMarker([e.lat,e.lon],{radius:Math.min(20,4+t*16),color:b?'#34d399':col(t),weight:b?2:1,fillColor:col(t),fillOpacity:.55}).addTo(layer);
m.bindPopup(`<b>Event #${e.id}</b> <span class="chip ${sens(e.sensors)}">${sensLbl(e.sensors)}</span><br>${e.region}<br>Total FRP <b>${e.frp_total}</b> MW · peak <b>${e.frp_peak}</b> MW<br>${e.detections} detections · ${e.persistence_days} days<br><span style="color:#8190a3">${e.first} → ${e.last}</span>`);marks[e.id]=m});
const R=D.regions&&D.regions[r];if(r!=='All regions'&&R)map.flyToBounds([[R[0],R[2]],[R[1],R[3]]],{padding:[20,20],duration:.8});else if(r!=='All regions'&&ev.length)map.flyToBounds(ev.map(e=>[e.lat,e.lon]),{padding:[40,40],maxZoom:7});else map.flyTo([20,10],2,{duration:.8})}
function tables(r,ev){const top=[...ev].sort((a,b)=>b[st.sort]-a[st.sort]).slice(0,10),mx=Math.max(1,...top.map(e=>e[st.sort]));
$('#events').innerHTML='<tr><th>Region</th><th>FRP (MW)</th><th>Days</th><th>Sensors</th></tr>'+top.map(e=>`<tr class="r" data-id="${e.id}"><td>${e.region}</td><td><span class="num">${nf(e.frp_total)}</span><div class="bar"><i style="width:${e[st.sort]/mx*100}%"></i></div></td><td class="num">${e.persistence_days}</td><td><span class="chip ${sens(e.sensors)}">${sensLbl(e.sensors)}</span></td></tr>`).join('')||'<tr><td>No events</td></tr>';
document.querySelectorAll('#events .r').forEach(tr=>tr.onclick=()=>{const m=marks[tr.dataset.id];if(m){map.flyTo(m.getLatLng(),Math.max(map.getZoom(),6),{duration:.7});setTimeout(()=>m.openPopup(),700);$('#map').scrollIntoView({behavior:'smooth',block:'center'})}});
$('#anom').innerHTML='<tr><th>Region</th><th>Deviation</th><th>z</th><th>Status</th></tr>'+D.anomalies.filter(a=>a.region!=='All regions').map(a=>{const z=a.z==null?0:Math.max(-3,Math.min(3,a.z)),w=Math.abs(z)/3*50;return`<tr class="r" data-r="${a.region}"><td>${a.region}</td><td><div class="zb"><i style="${z>=0?'left:50%':'right:50%'};width:${w}%;background:${z>=1.5?'#ff5d73':z>=.5?'#f5d04c':'#4ade80'}"></i></div></td><td class="num">${a.z??'—'}</td><td><span class="chip ${lab(a)}">${labTxt(a)}</span></td></tr>`}).join('');
document.querySelectorAll('#anom .r').forEach(tr=>tr.onclick=()=>{sel.value=tr.dataset.r;draw()});
const c=(D.info[r]||{}).calibration||{},vals=Object.values(c),lo=Math.min(...vals.map(v=>v.lo)),hi=Math.max(...vals.map(v=>v.hi)),sc=x=>(x-lo)/Math.max(hi-lo,1e-9)*100;
$('#cal').innerHTML=Object.entries(c).map(([k,v])=>`<div class="cal"><b>${k}</b><div class="rng"><i style="left:${sc(v.lo)}%;width:${Math.max(2,sc(v.hi)-sc(v.lo))}%"></i><u style="left:${sc(v.ratio)}%"></u></div><span class="num">${v.ratio.toFixed(2)} <span style="color:#8190a3">[${v.lo.toFixed(2)}–${v.hi.toFixed(2)}]</span></span></div>`).join('')||'<span style="color:#8190a3">No calibration data</span>';
$('#quality').innerHTML='<tr><th>Region</th><th>Overlap MAPE</th><th>Raw jump</th><th>Harmonized jump</th><th>Artifact removed</th></tr>'+Object.entries(D.info).map(([n,x])=>{const d=x.raw_transition!=null&&x.harmonized_transition!=null?(Math.abs(x.raw_transition)-Math.abs(x.harmonized_transition))*100:null;return`<tr class="r" data-r="${n}"><td>${n}</td><td class="num">${pct(x.mape_overlap)}</td><td class="num">${sp(x.raw_transition)}</td><td class="num">${sp(x.harmonized_transition)}</td><td class="num ${d>0?'up':''}">${d==null?'—':(d>=0?'▼ ':'▲ ')+Math.abs(d).toFixed(1)+' pp'}</td></tr>`}).join('');
document.querySelectorAll('#quality .r').forEach(tr=>tr.onclick=()=>{sel.value=tr.dataset.r;draw();scrollTo({top:0,behavior:'smooth'})})}
function draw(){const r=sel.value,i=D.info[r]||{};evs=D.events.filter(e=>r==='All regions'||e.region===r);kpis(r,i,evs);charts(r);mapdraw(r,evs);tables(r,evs)}
function static_(){const q=D.quality||{};$('#qsub').textContent=nf(q.rows_after_region_filter||0)+' detections analysed inside configured regions';
$('#qc').innerHTML=['MODIS','VIIRS'].filter(s=>q[s]).map(s=>{const x=q[s],p=x.rows_read?x.rows_valid/x.rows_read*100:0;return`<div class="qrow"><span><span class="chip ${s}">${s}</span> ${x.files} file(s)</span><span><b>${nf(x.rows_valid)}</b> / ${nf(x.rows_read)} valid (${p.toFixed(1)}%)</span></div><div class="bar"><i style="width:${p}%"></i></div>`}).join('');
$('#steps').innerHTML=[['Ingest','MODIS & VIIRS FIRMS CSVs'],['Validate','Schema, dates, FRP, confidence'],['Harmonize','Seasonal VIIRS→MODIS calibration'],['Quantify','Bootstrap bounds & overlap error'],['Cluster','Persistent events in space + time'],['Detect','Seasonal z-score anomalies']].map((s,k)=>`<div class="st"><n>${k+1}</n><b>${s[0]}</b><span>${s[1]}</span></div>`).join('');
$('#foot').textContent='XFiRE v'+(D.version||'1.0')+' · harmonized analytical index — an empirical cross-sensor adjustment, not a physical replacement for either satellite product.'}
const dl=(n,t,m)=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([t],{type:m}));a.download=n;a.click()};
$('#expJson').onclick=()=>dl('xfire_results.json',JSON.stringify(D,null,1),'application/json');
$('#expCsv').onclick=()=>{const k=['id','region','lat','lon','frp_total','frp_peak','detections','persistence_days','sensors','first','last'];dl('xfire_events_'+sel.value.replace(/ /g,'_')+'.csv',k.join(',')+'\n'+evs.map(e=>k.map(c=>e[c]).join(',')).join('\n'),'text/csv')};
function seg(id,key,attr,fn){document.querySelectorAll(id+' button').forEach(b=>b.onclick=()=>{fn(b.dataset[attr]);document.querySelectorAll(id+' button').forEach(x=>x.classList.toggle('on',x===b||(id==='#tg'&&st[x.dataset.t])));draw()})}
seg('#rng','rng','r',v=>st.rng=+v);seg('#srt','sort','s',v=>st.sort=v);
document.querySelectorAll('#tg button').forEach(b=>b.onclick=()=>{st[b.dataset.t]=!st[b.dataset.t];b.classList.toggle('on',st[b.dataset.t]);charts(sel.value)});
sel.onchange=draw;static_();draw();
</script></body></html>'''


def write_dashboard(results, path):
    data = json.dumps(results, separators=(',', ':')).replace('</', '<\\/')
    Path(path).write_text(TEMPLATE.replace('__DATA__', data), encoding='utf-8')
