#pragma once
#include <string_view>

namespace ninfer::serve {
// Completely static, bundled UI. No network dependencies besides this server's metrics APIs.
inline std::string_view monitor_dashboard_html() noexcept {
    return R"MONHTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NInfer Monitor</title>
<style>
:root{color-scheme:dark;font:14px/1.5 system-ui,Segoe UI,sans-serif;background:#0b1116;color:#e4ebee}
*{box-sizing:border-box}body{margin:0}main{max-width:1250px;margin:30px auto;padding:0 20px}
header,.row,.controls{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
h1{font-size:27px;margin:0}h2{font-size:15px;margin:0 0 13px}.muted,small{color:#94a8b4}
.controls{justify-content:flex-start;margin:18px 0 10px}label{font-size:12px;color:#a8b9c0;display:flex;gap:8px;align-items:center}
input{background:#111c23;border:1px solid #3a515e;color:#ecf3f6;border-radius:8px;padding:8px;font:inherit;max-width:160px}
button{background:#244e40;border:1px solid #349b72;color:#affad8;border-radius:8px;padding:9px 14px;cursor:pointer}
.pill{border:1px solid #3d5260;border-radius:18px;padding:4px 11px;color:#bdc9d2;font-size:12px}
.ok{color:#51dda8;border-color:#31916e}.bad{color:#ff9d92;border-color:#aa5047}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.card{background:#141d23;border:1px solid #293b44;border-radius:15px;padding:16px;min-width:0}
.label{font-size:12px;color:#94a8b4}.value{font-size:26px;font-weight:700;line-height:1.5;font-variant-numeric:tabular-nums}
.green{color:#4bdfae}.wide{grid-column:span 2}.section{margin-top:13px}
canvas{display:block;width:100%;height:190px}.legend{display:flex;gap:20px;color:#9eafbd;font-size:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px}
.track{margin:14px 0 10px;background:#293a44;border-radius:10px;overflow:hidden;height:12px}
.fill{height:100%;background:#44dbaa;transition:width .25s}
table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
th,td{padding:10px 7px;text-align:left;border-bottom:1px solid #2a3d48}th{color:#a2b7c2;font-weight:500}
.scroll{overflow:auto}#message{min-height:27px;color:#a8b9c3}.note{color:#91a6b2;font-size:12px}
@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:520px){.grid{grid-template-columns:1fr}.wide{grid-column:span 1}main{padding:0 12px}.value{font-size:23px}}
</style></head><body><main>
<header><div><h1>NInfer <span class="green">Monitor</span></h1><div class="muted" id="model">Connecting to model…</div></div>
<div><span class="pill" id="state">Connecting</span> <span class="pill">Read-only</span></div></header>
<div class="controls">
<label>API key <input type="password" id="key" placeholder="If enabled" autocomplete="off"></label>
<label>Model <input id="selectedModel" placeholder="Auto / model id"></label>
<label>GPU <input id="gpuIndex" type="number" min="0" max="7" value="0" style="width:58px"></label>
<button id="refresh">Refresh</button><small>Auto-refresh: 3 seconds</small>
</div><div id="message" role="status" aria-live="polite"></div>
<div class="grid">
<div class="card"><div class="label">Decode now</div><div class="value green" id="decode">–</div><small>tokens/s (last interval)</small></div>
<div class="card"><div class="label">Prefill now</div><div class="value" id="prefill">–</div><small>tokens/s (last interval)</small></div>
<div class="card"><div class="label">GPU utilization</div><div class="value" id="gpuLoad">–</div><small>NVIDIA NVML</small></div>
<div class="card"><div class="label">VRAM used</div><div class="value" id="vram">–</div><small id="vramTotal">NVIDIA NVML</small></div>
<div class="card"><div class="label">GPU temperature</div><div class="value" id="temp">–</div><small>°C</small></div>
<div class="card"><div class="label">GPU power</div><div class="value" id="power">–</div><small>W</small></div>
<div class="card"><div class="label">Running / waiting</div><div class="value" id="requests">–</div><small id="admitted">requests</small></div>
<div class="card"><div class="label">DFlash2 acceptance</div><div class="value" id="draft">–</div><small>last 50 requests · model drafts only</small></div>
</div>
<div class="grid section">
<section class="card wide"><h2>Live token throughput</h2><canvas id="rates" role="img" aria-label="Recent prefill and decode throughput"></canvas>
<div class="legend"><span><i class="dot" style="background:#49dcaa"></i>Decode</span><span><i class="dot" style="background:#59acfa"></i>Prefill</span></div></section>
<section class="card wide"><h2>GPU utilization / temperature</h2><canvas id="gpuChart" role="img" aria-label="Recent GPU utilization and temperature"></canvas>
<div class="legend"><span><i class="dot" style="background:#49dcaa"></i>Utilization %</span><span><i class="dot" style="background:#e7b95b"></i>Temperature °C</span></div></section>
<section class="card wide"><h2>Device KV occupancy</h2>
<div class="row"><strong id="kv">–</strong><span id="kvPct">–</span></div><div class="track"><div class="fill" id="kvFill" style="width:0"></div></div>
<p class="note">This is device KV usage, not the total input context. Configured maximum context: <span id="maxContext">–</span> tokens.</p></section>
<section class="card wide"><h2>Server statistics</h2>
<div class="row"><span class="muted">Completed requests</span><strong id="done">–</strong></div>
<div class="row"><span class="muted">Rejected / failed</span><strong id="errors">–</strong></div>
<div class="row"><span class="muted">Reused prompt tokens</span><strong id="reused">–</strong></div>
<div class="row"><span class="muted">Uptime</span><strong id="uptime">–</strong></div></section>
<section class="card" style="grid-column:1/-1"><h2>Recent requests (last 50)</h2><div class="scroll">
<table><thead><tr><th>Time</th><th>Status</th><th>Prompt</th><th>Cached</th><th>Output</th><th>Decode tok/s</th><th>Total duration</th></tr></thead>
<tbody id="recentRequests"><tr><td colspan="7">No requests recorded</td></tr></tbody></table></div>
<p class="note">Metadata only: no prompts or generated text are stored. History resets when the server restarts.</p>
</section>
<section class="card" style="grid-column:1/-1"><h2>Recent activity (aggregate samples)</h2><div class="scroll">
<table><thead><tr><th>Time</th><th>Completed Δ</th><th>Decode Δ</th><th>Prefill Δ</th><th>Running</th><th>Waiting</th></tr></thead>
<tbody id="activity"><tr><td colspan="6">Waiting for samples…</td></tr></tbody></table></div>
<p class="note">Data is stored only in this tab (120 samples). The optional Python sidecar tools/monitor/monitor.py provides full per-request JSONL logs, CPU/RAM and further sensor metrics.</p>
</section></div></main>
<script>
'use strict';
const $=id=>document.getElementById(id);
const samples=[];let prev=null,polling=false;
const n=v=>typeof v==='number'&&Number.isFinite(v)?v:null;
const format=v=>v===null||v===undefined?'–':Number(v).toLocaleString(undefined,{maximumFractionDigits:1});
const txt=(id,v)=>{$(id).textContent=v};
const count=(obj,key)=>n(obj?.[key])||0;
async function fetchApi(url){
 const headers={};const key=$('key').value;if(key)headers.Authorization='Bearer '+key;
 const response=await fetch(url,{headers,cache:'no-store'});
 if(!response.ok)throw Error('HTTP '+response.status+(response.status===401?' — API key required':''));
 return response;
}
function metricMap(text){
 const map={};for(const line of text.split(/\r?\n/)){
  if(!line||line[0]==='#')continue;
  const hit=line.match(/^([a-zA-Z_:][\w:]*)\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)(?:\s|$)/);
  if(hit)map[hit[1]]=Number(hit[2]);
 }return map;
}
function graph(id,a,b,ceiling){
 const cv=$(id),width=cv.getBoundingClientRect().width;
 if(!width)return;
 const dpr=window.devicePixelRatio||1,h=190;
 cv.width=Math.round(width*dpr);cv.height=Math.round(h*dpr);
 const c=cv.getContext('2d');c.scale(dpr,dpr);
 const p={left:42,right:9,top:9,bottom:24};
 const max=Math.max(ceiling||0,1,...samples.flatMap(s=>[n(s[a]),n(s[b])]).filter(v=>v!==null))*1.1;
 c.font='11px system-ui';c.lineWidth=1;
 for(let i=0;i<=4;i++){
  const y=p.top+(h-p.top-p.bottom)*i/4;
  c.strokeStyle='#2b414c';c.beginPath();c.moveTo(p.left,y);c.lineTo(width-p.right,y);c.stroke();
  c.fillStyle='#93a9b5';c.fillText(format(max*(4-i)/4),3,y+4);
 }
 const line=(field,color)=>{
  c.beginPath();c.strokeStyle=color;c.lineWidth=2;let started=false;
  samples.forEach((s,i)=>{
   const v=n(s[field]);if(v===null){started=false;return}
   const x=p.left+(width-p.left-p.right)*i/Math.max(1,samples.length-1);
   const y=h-p.bottom-Math.max(0,v)/max*(h-p.top-p.bottom);
   if(started)c.lineTo(x,y);else{c.moveTo(x,y);started=true}
  });c.stroke();
 };
 line(a,'#49dcaa');line(b,id==='rates'?'#59acfa':'#e7b95b');
 c.fillStyle='#93a9b5';
 if(samples.length){c.fillText(new Date(samples[0].time).toLocaleTimeString(),p.left,h-4);
 c.fillText(new Date(samples[samples.length-1].time).toLocaleTimeString(),Math.max(p.left,width-85),h-4)}
}
function draw(){graph('rates','decodeRate','prefillRate',0);graph('gpuChart','gpuUtil','gpuTemp',100)}
function requestRows(events){
 const body=$('recentRequests');body.replaceChildren();
 if(!Array.isArray(events)||!events.length){
  const tr=document.createElement('tr'),td=document.createElement('td');
  td.colSpan=7;td.textContent='No requests recorded';tr.appendChild(td);body.appendChild(tr);return;
 }
 for(const event of events){
  const tr=document.createElement('tr');
  const decodeRate=n(event.decode_seconds)>0&&n(event.output_tokens)>1?
   (event.output_tokens-1)/event.decode_seconds:null;
  const cells=[new Date(event.time_unix_ms).toLocaleTimeString(),event.status,
   event.prompt_tokens,event.cached_tokens,event.output_tokens,decodeRate,
   n(event.total_seconds)!==null?format(event.total_seconds)+' s':'–'];
  for(const v of cells){const td=document.createElement('td');
   td.textContent=typeof v==='string'?v:format(v);tr.appendChild(td)}
  body.appendChild(tr);
 }
}
function activity(){
 const body=$('activity');body.replaceChildren();
 for(const row of samples.slice(-15).reverse()){
  const tr=document.createElement('tr');
  const cells=[new Date(row.time).toLocaleTimeString(),row.completeDelta,row.decodeDelta,row.prefillDelta,row.running,row.waiting];
  for(const v of cells){const td=document.createElement('td');td.textContent=typeof v==='string'?v:format(v);tr.appendChild(td)}
  body.appendChild(tr);
 }
}
async function refresh(){
 if(polling||document.hidden)return;
 polling=true;
 try{
  const model=$('selectedModel').value.trim();
  const query=model?'?model='+encodeURIComponent(model):'';
  const results=await Promise.all([
    fetchApi('/stats'+query),fetchApi('/metrics'+query),fetchApi('/monitor/requests')]);
  const stats=await results[0].json(),metrics=metricMap(await results[1].text());
  const recent=await results[2].json();
  let gpu={available:false};
  try{const idx=Math.min(7,Math.max(0,Number($('gpuIndex').value)||0));
    gpu=await(await fetchApi('/monitor/gpu?index='+idx)).json();
  }catch(e){/* GPU telemetry is optional. */}
  const now=performance.now();
  const curr={time:Date.now(),uptime:stats.uptime_seconds,model:stats.model,
   prefill:count(stats.counters,'computed_prefill_tokens'),decode:count(stats.counters,'committed_decode_tokens'),
   completed:count(metrics,'ninfer:requests_total'),
   running:count(stats.requests,'running'),waiting:count(stats.requests,'waiting')};
  let decodeRate=null,prefillRate=null,decodeDelta=null,prefillDelta=null,completeDelta=null;
  const dt=prev?(now-prev.mono)/1000:0;
  if(prev&&dt>.1&&curr.uptime>=prev.uptime&&curr.model===prev.model&&curr.decode>=prev.decode&&curr.prefill>=prev.prefill){
    decodeDelta=curr.decode-prev.decode;prefillDelta=curr.prefill-prev.prefill;
    decodeRate=decodeDelta/dt;prefillRate=prefillDelta/dt;
    if(curr.completed>=prev.completed)completeDelta=curr.completed-prev.completed;
  }else if(prev){samples.length=0}
  prev={...curr,mono:now};
  txt('model',(stats.model||'Unknown model')+' · '+format(stats.capacity?.max_context)+' max context');
  txt('decode',format(decodeRate));txt('prefill',format(prefillRate));
  txt('requests',format(curr.running)+' / '+format(curr.waiting));
  txt('admitted',format(stats.requests?.admitted||0)+' admitted');
  const dflash=recent.filter(e=>typeof e.drafter==='string'&&e.drafter.toLowerCase()==='dflash2');
  const drafted=dflash.reduce((sum,e)=>sum+count(e,'drafted_tokens'),0);
  const accepted=dflash.reduce((sum,e)=>sum+count(e,'accepted_tokens'),0);
  txt('draft',drafted?format(accepted/drafted*100)+' %':'–');
  const capacity=count(stats.capacity,'kv_capacity_tokens'),used=count(stats.occupancy,'device_main_kv_tokens');
  const fill=capacity?Math.min(100,used/capacity*100):0;
  txt('kv',format(used)+' / '+format(capacity)+' tokens');txt('kvPct',capacity?format(fill)+' %':'–');
  $('kvFill').style.width=fill+'%';txt('maxContext',format(stats.capacity?.max_context));
  txt('done',format(curr.completed));
  txt('errors',format(count(metrics,'ninfer:requests_rejected_total'))+' / '+format(count(metrics,'ninfer:requests_failed_total')));
  txt('reused',format(count(stats.counters,'reused_prompt_tokens')));txt('uptime',format(curr.uptime/60)+' min');
  const available=gpu.available===true;
  txt('gpuLoad',available&&n(gpu.utilization_pct)!==null?format(gpu.utilization_pct)+' %':'–');
  txt('vram',available&&n(gpu.memory_used_bytes)!==null?format(gpu.memory_used_bytes/1073741824)+' GB':'–');
  txt('vramTotal',available&&n(gpu.memory_total_bytes)!==null?'of '+format(gpu.memory_total_bytes/1073741824)+' GB':'NVML unavailable');
  txt('temp',available&&n(gpu.temperature_c)!==null?format(gpu.temperature_c)+' °C':'–');
  txt('power',available&&n(gpu.power_w)!==null?format(gpu.power_w)+' W':'–');
  samples.push({time:curr.time,decodeRate,prefillRate,decodeDelta,prefillDelta,completeDelta,
   running:curr.running,waiting:curr.waiting,gpuUtil:n(gpu.utilization_pct),gpuTemp:n(gpu.temperature_c)});
  if(samples.length>120)samples.shift();
  draw();activity();requestRows(recent);
  $('state').className='pill ok';txt('state','Online');txt('message','');
 }catch(e){$('state').className='pill bad';txt('state','Offline');txt('message',e.message||String(e))}
 finally{polling=false}
}
$('refresh').addEventListener('click',()=>{prev=null;samples.length=0;refresh()});
for(const id of ['key','selectedModel','gpuIndex'])$(id).addEventListener('change',()=>{prev=null;samples.length=0;refresh()});
window.addEventListener('resize',draw);
document.addEventListener('visibilitychange',()=>{if(!document.hidden)refresh()});
refresh();setInterval(refresh,3000);
</script></body></html>)MONHTML";
}
} // namespace ninfer::serve
