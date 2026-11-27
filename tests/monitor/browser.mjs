import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
const html=readFileSync(process.argv[2],'utf8');
const browser=await chromium.launch();
try {
 const ctx=await browser.newContext({viewport:{width:390,height:844},acceptDownloads:true});
 const p=await ctx.newPage(); const errors=[];p.on('pageerror',e=>errors.push(e.message));
 let calls=0,chatPosts=0,keyObserved=false;
 const events=[
  {id:1,model:'fixture',status:'done',time_unix_ms:1,output_tokens:8,prompt_tokens:100,cached_tokens:20,ttft_seconds:.1,total_seconds:1,decode_seconds:.8,drafter:'dflash2',drafted_tokens:10,accepted_tokens:8},
  {id:2,model:'fixture',status:'done',time_unix_ms:2,output_tokens:8,prompt_tokens:100,cached_tokens:0,ttft_seconds:.4,total_seconds:3,decode_seconds:1,drafter:'mtp',drafted_tokens:10,accepted_tokens:6},
  {id:3,model:'fixture',status:'failed',time_unix_ms:3,ttft_seconds:99,total_seconds:99},
  {id:4,model:'fixture',status:'done',time_unix_ms:4,output_tokens:0,ttft_seconds:0,total_seconds:.5},
  {id:5,model:'different',status:'done',time_unix_ms:5,prompt_tokens:9999,cached_tokens:0,prefill_seconds:.1,output_tokens:80,decode_seconds:.1,ttft_seconds:20,total_seconds:20},
  {id:6,model:'fixture',status:'done',prompt_tokens:1024,cached_tokens:512,prefill_seconds:2,output_tokens:17,decode_seconds:2},
  {id:7,model:'fixture',status:'done',prompt_tokens:1024,cached_tokens:0,prefill_seconds:1,output_tokens:33,decode_seconds:1},
  {id:8,model:'fixture',status:'done',prompt_tokens:1024,cached_tokens:513,prefill_seconds:.1,output_tokens:15,decode_seconds:.1},
  {id:9,model:'fixture',status:'done',prompt_tokens:1024,cached_tokens:0,output_tokens:17,decode_seconds:0},
  {id:10,model:'fixture',status:'failed',prompt_tokens:9999,cached_tokens:0,prefill_seconds:.1,output_tokens:80,decode_seconds:.1}
 ];
 await ctx.route('http://127.0.0.1:8767/**',async route=>{
  const req=route.request(),url=new URL(req.url()),path=url.pathname;
  if(path==='/monitor')return route.fulfill({contentType:'text/html',body:html,headers:{'Content-Security-Policy':"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"}});
  if(req.headers().authorization!=='Bearer fixture-only')return route.fulfill({status:401,body:'Unauthorized'});
  keyObserved=true;
  if(path==='/stats')return route.fulfill({json:{model:'fixture',uptime_seconds:50+(++calls)*3,capacity:{max_context:32768,kv_capacity_tokens:65536},occupancy:{device_main_kv_tokens:100},counters:{computed_prefill_tokens:calls*300,committed_decode_tokens:calls*90,reused_prompt_tokens:0},requests:{running:1,waiting:0,admitted:1}}});
  if(path==='/metrics')return route.fulfill({body:'ninfer:requests_total 5\nllamacpp:predicted_tokens_seconds 192\nllamacpp:prompt_tokens_seconds 3100\n'});
  if(path==='/monitor/requests')return route.fulfill({json:events});
  if(path==='/monitor/system')return route.fulfill({json:{cpu_pct:12,ram_used_bytes:1073741824,ram_total_bytes:2147483648}});
  if(path==='/monitor/gpu')return route.fulfill({json:{available:true,graphics_clock_mhz:2400,memory_clock_mhz:10500,temperature_c:60}});
  if(path==='/v1/chat/completions'){
    assert.equal(req.method(),'POST');const body=req.postDataJSON();assert.equal(body.model,'fixture');assert.equal(body.stream,false);assert.equal(body.messages.at(-1).role,'user');
    chatPosts++;return route.fulfill({json:{choices:[{message:{content:'<img src=x onerror=alert(1)> fixture response'}}]}})
  }
  throw Error('Unexpected request '+req.url());
 });
 await p.goto('http://127.0.0.1:8767/monitor');
 await p.waitForFunction(()=>document.querySelector('#state').textContent==='Offline');
 await p.locator('#key').fill('fixture-only');await p.locator('#key').dispatchEvent('change');
 await p.locator('#selectedModel').fill('fixture');await p.locator('#selectedModel').dispatchEvent('change');
 await p.waitForFunction(()=>document.querySelector('#state').textContent==='Online'&&document.querySelector('#ttftN').textContent==='2');
 assert.equal(await p.locator('#ttft50').textContent(),'100');
 assert.equal(await p.locator('#ttft95').textContent(),'400');
 assert.equal(await p.locator('#latN').textContent(),'3');
 for(const [prefix,p50,upper] of [['prefillRate','256','1,024'],['decodeRate','8','32']]){
  assert.equal(await p.locator('#'+prefix+'N').textContent(),'2');
  assert.equal(await p.locator('#'+prefix+'50').textContent(),p50);
  assert.equal(await p.locator('#'+prefix+'95').textContent(),upper);
  assert.equal(await p.locator('#'+prefix+'99').textContent(),upper);
 }
 await p.locator('#selectedModel').fill('missing');await p.locator('#selectedModel').dispatchEvent('change');
 await p.waitForFunction(()=>document.querySelector('#prefillRateN').textContent==='0'&&document.querySelector('#decodeRate50').textContent==='–');
 await p.locator('#selectedModel').fill('fixture');await p.locator('#selectedModel').dispatchEvent('change');
 await p.waitForFunction(()=>document.querySelector('#prefillRateN').textContent==='2');
 assert.equal(await p.locator('#mtp').textContent(),'60 %');
 assert.equal(await p.locator('#draft').textContent(),'80 %');
 assert.equal(await p.locator('#avgDecode').textContent(),'192');
 assert.equal(await p.locator('#cpu').textContent(),'12 %');
 assert.equal(await p.locator('#gpuClock').textContent(),'2,400');
 assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
 await p.locator('#lang').selectOption('de');assert.equal(await p.locator('html').getAttribute('lang'),'de');
 await p.locator('#persist').check();
 await p.waitForFunction(()=>JSON.parse(localStorage.getItem('ninfer-monitor-history-v1')||'{}').samples?.length>0);
 const stored=await p.evaluate(()=>localStorage.getItem('ninfer-monitor-history-v1'));
 assert.ok(!stored.includes('fixture'));assert.ok(!stored.includes('Bearer'));
 const json=p.waitForEvent('download');await p.locator('#jsonExport').click();
 const saved=JSON.parse(readFileSync(await (await json).path(),'utf8'));
 assert.ok(saved.samples.every(row=>Object.values(row).every(v=>v===null||typeof v==='number')));
 const csv=p.waitForEvent('download');await p.locator('#csvExport').click();assert.match((await csv).suggestedFilename(),/\.csv$/);
 await p.locator('#chatInput').fill('synthetic chat');assert.equal(chatPosts,0);
 await p.locator('#chatSend').click();await p.waitForFunction(()=>document.querySelector('#chatLog').textContent.includes('fixture response'));
 assert.equal(chatPosts,1);assert.equal(await p.locator('#chatLog img').count(),0);
 await p.reload();assert.equal(await p.locator('#key').inputValue(),'');
 assert.equal(await p.locator('#chatLog').textContent(),'');assert.equal(await p.locator('#persist').isChecked(),true);
 await p.locator('#clearHistory').click();assert.equal(await p.evaluate(()=>localStorage.getItem('ninfer-monitor-history-v1')),null);
 assert.ok(keyObserved);assert.deepEqual(errors,[]);
 console.log('PASS: native HTML, authorization, real request status, model filtering, TTFT/percentiles, MTP/DFlash2, clocks, localization, history/export, explicit safe chat');
} finally {await browser.close()}
