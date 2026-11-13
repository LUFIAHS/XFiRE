// node tests/check_dashboard.js output/dashboard.html   (needs: npm i jsdom). Loads the dashboard with a canvas stub, checks KPIs/forecast/eras render and every download works.
const {JSDOM}=require('jsdom');const fs=require('fs');
const html=fs.readFileSync(process.argv[2],'utf8');
const ctx=()=>new Proxy(function(){},{get:(t,p)=>p==='measureText'?()=>({width:10}):p==='canvas'?{}:()=>ctx(),set:()=>true,apply:()=>ctx()});
const dom=new JSDOM(html,{runScripts:'dangerously',pretendToBeVisual:true,url:'http://127.0.0.1:8000/',beforeParse(w){
 w.HTMLCanvasElement.prototype.getContext=()=>ctx();w.HTMLCanvasElement.prototype.toBlob=cb=>cb(new w.Blob(['x']));
 Object.defineProperty(w.HTMLElement.prototype,'clientWidth',{get:()=>900});w.fetch=()=>Promise.reject(new Error('nofetch'));w.URL.createObjectURL=()=>'blob:x';w.URL.revokeObjectURL=()=>{};
 w.addEventListener('error',e=>{console.log('PAGE ERROR',e.message);process.exitCode=1});}});
setTimeout(()=>{const d=dom.window.document;const t=s=>(d.querySelector(s)||{}).textContent||'';
 console.log('KPIS:',t('#kpis').slice(0,500));console.log('ERAS:',t('#eras').slice(0,400));console.log('UT:',t('#ut').slice(0,300));console.log('DG:',t('#dg').slice(0,400));console.log('FCT:',t('#fct').slice(0,500));
 for(const id of ['#dlReport','#dlForecast','#dlMonth','#dlWeek','#dlHot'])d.querySelector(id).click();
 for(const [id,must] of [['#kpis','Next 4 weeks'],['#eras','estimates'],['#fct','Forecast, not observation'],['#arch','LightGBM + Hyperopt'],['#arch','XGBoost'],['#ut','Seasonal ratio']])if(!t(id).includes(must)){console.log('FAIL',id);process.exitCode=1}
 console.log(process.exitCode?'dashboard FAIL':'dashboard OK');process.exit(process.exitCode||0)},800);
