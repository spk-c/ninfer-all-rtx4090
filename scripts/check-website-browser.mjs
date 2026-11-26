import { chromium } from "playwright";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const axe = require.resolve("axe-core/axe.min.js");
const count = [...readFileSync("site/sitemap.xml","utf8").matchAll(/<loc>/g)].length;

const server = spawn("python3", ["-m", "http.server", "8765", "--bind", "127.0.0.1", "--directory", "site"], {stdio:"ignore"});
const base = "http://127.0.0.1:8765/";
let browser;
try {
  for(let i=0;i<50;i++){
    try {if((await fetch(base)).ok)break;}catch{}
    await new Promise(r=>setTimeout(r,100));
  }
  browser=await chromium.launch({headless:true});
  const context=await browser.newContext({viewport:{width:390,height:844}, permissions:["clipboard-read","clipboard-write"]});
  await context.route("https://api.github.com/**", route=>route.fulfill({json:route.request().url().includes("/releases")?[
    {tag_name:"rtx4090-v9.0.0",prerelease:false,draft:false,published_at:"2026-11-26",assets:[]}
  ]:{stargazers_count:42}}));
  // The sitemap contains production canonical URLs. Serve those same files from the test server.
  await context.route(base+"sitemap.xml", async route=>{
    const res=await route.fetch();
    await route.fulfill({response:res,body:(await res.text()).replaceAll("https://spk-c.github.io/ninfer-all-rtx4090/",base)});
  });
  const p=await context.newPage();
  const errors=[];
  p.on("pageerror",e=>errors.push(e.message));
  await p.goto(base+"configurator.html");
  await p.locator("#start-command").waitFor();
  assert.match(await p.locator("#start-command").textContent(),/--spec dflash2 --draft-tokens 5/);
  assert.match(await p.locator("#download-command").textContent(),/hf download SPK1337/);
  assert.match(await p.locator("#kv-estimate").textContent(),/0\.55 GiB/);
  await p.locator("#context").fill("32769");
  assert.match(await p.locator("#kv-estimate").textContent(),/32,832 rounded tokens/);
  await p.locator("#context").fill("32768");
  await p.locator("#concurrency").fill("2");
  assert.match(await p.locator("#kv-estimate").textContent(),/1\.09 GiB/);
  assert.match(await p.locator("#start-command").textContent(),/--kv-capacity 65536/);
  await p.locator("#concurrency").fill("1");
  await p.locator("#model").selectOption("ptq");
  assert.equal(await p.locator("#spec").inputValue(),"mtp");
  assert.equal(await p.locator('#spec option[value="dflash2"]').count(),0);
  assert.match(await p.locator("#download-command").textContent(),/WaveCut\/Ternary-Bonsai-2.5-27B-PTQ1_0/);
  await p.locator("#context").fill("300000");
  assert.equal(await p.locator("#start-command").textContent(),"");
  await p.locator("#yarn").check();
  assert.match(await p.locator("#start-command").textContent(),/--rope-yarn/);
  await p.locator("#other").fill("24");
  assert.match(await p.locator("#memory-estimate").textContent(),/over budget/);
  await p.locator("#spec").selectOption("none");
  assert.equal(await p.locator("#draft").isDisabled(),true);
  assert.doesNotMatch(await p.locator("#start-command").textContent(),/--draft-tokens/);
  await p.locator(".copy-btn").last().click();
  await p.waitForFunction(()=>[...document.querySelectorAll(".copy-btn")].some(b=>b.textContent==="Copied"));
  assert.match(await p.evaluate(()=>navigator.clipboard.readText()),/--spec none/);
  await p.locator(".theme-btn").click();
  assert.equal(await p.locator("html").getAttribute("data-theme"),"light");
  await p.goto(base+"configurator-de.html");
  assert.match(await p.locator("#kv-estimate").textContent(),/Geschätzter Haupt-KV/);
  assert.equal(await p.locator(".theme-btn").textContent(),"Design wechseln");
  await p.goto(base+"models.html");
  await p.locator("#model-filter").fill("Bonsai");
  assert.equal(await p.locator("main tbody tr:visible").count(),1);
  assert.equal(await p.locator("main article.panel:visible").count(),0);
  await p.locator("#model-filter").fill("");
  assert.ok(await p.locator("main tbody tr:visible").count()>1);
  await p.goto(base+"search.html");
  await p.locator("#site-query").fill("DFlash2");
  await p.locator("#search-submit").click();
  await p.waitForFunction(()=>document.querySelectorAll("#search-results article").length>0);
  assert.match(await p.locator("#search-status").textContent(),new RegExp(count+" loaded pages"));
  await p.locator("#site-query").fill("zzzz-nonexistent-9999");
  await p.locator("#search-submit").click();
  await p.waitForFunction(()=>document.querySelector("#search-status").textContent.startsWith("0 matches"));
  await p.goto(base);
  await p.waitForFunction(()=>document.querySelector("#star-count").textContent.includes("42 stars"));
  assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,"mobile horizontal overflow");
  await p.goto(base+"changelog.html");
  await p.waitForFunction(()=>document.querySelector("#release-list").textContent.includes("rtx4090-v9.0.0"));
  for(const path of ["monitor.html","monitor-de.html"]){
    await p.goto(base+path);
    assert.equal(await p.locator("main h1").textContent(),"NInfer Monitor");
    assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,"monitor mobile overflow");
    for(const src of ["monitor-desktop.png","monitor-mobile.png"]){
      const shot=p.locator('main img[src="'+src+'"]');
      // Wait for lazy images to actually load before decoding (Chromium can
      // reject decode() when the lazy request has not been initiated yet).
      await shot.scrollIntoViewIfNeeded();
      await p.waitForFunction(src=>{
        const img=document.querySelector('main img[src="'+src+'"]');
        return img?.complete && img.naturalWidth>0;
      },src);
      assert.ok(await shot.evaluate(async img=>{await img.decode();return img.naturalWidth>0}),src);
    }
    assert.ok(await p.locator('main a[href="#open-monitor"]').count());
    assert.match(await p.locator("#open-monitor").textContent(),/127\.0\.0\.1:8099\/monitor/);
  }
  for(const path of ["","configurator.html","configurator-de.html","models.html","search.html","benchmarks.html","monitor.html","monitor-de.html"]){
    await p.goto(base+path);
    await p.addScriptTag({path:axe});
    const result=await p.evaluate(async()=>await window.axe.run({runOnly:{type:"tag",values:["wcag2a","wcag2aa","wcag21a","wcag21aa"]}}));
    const serious=result.violations.filter(v=>["critical","serious"].includes(v.impact));
    console.log("axe",path||"/",JSON.stringify(result.violations.map(v=>({id:v.id,impact:v.impact,targets:v.nodes.map(n=>n.target)}))));
    assert.deepEqual(serious.map(v=>v.id),[],`accessibility: ${path}`);
  }
  for(const path of ["sm89-native.html","sm89-native-de.html","sm89-speculation.html","sm89-speculation-de.html"]){
    await p.goto(base+path);
    assert.ok((await p.locator("main h1").textContent()).includes("SM89"));
    assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,path+" overflow");
  }
  for(const [path,lang] of [["monitor-demo.html","en"],["monitor-demo-de.html","de"]]){
    const requests=[];
    const capture=req=>requests.push(req.url());
    p.on("request",capture);
    await p.goto(base+path);
    await p.waitForFunction(()=>document.querySelector("#state").textContent==="Online");
    assert.equal(await p.locator("html").getAttribute("lang"),lang);
    assert.equal(await p.locator("#key").isDisabled(),true);
    assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,path+" overflow");
    assert.match(await p.locator("#ttftN").textContent(),/8/);
    await p.locator("#demoModel").selectOption("mtp");
    await p.waitForFunction(()=>document.querySelector("#mtp").textContent.includes("%"));
    assert.equal(await p.locator("#draft").textContent(),"–");
    await p.locator("#lang").selectOption(lang==="en"?"de":"en");
    assert.equal(await p.locator("html").getAttribute("lang"),lang==="en"?"de":"en");
    await p.locator("#persist").check();
    await p.waitForFunction(()=>JSON.parse(localStorage.getItem("ninfer-monitor-public-demo-history-v1")||"{}").samples?.length>0);
    const persisted=await p.evaluate(()=>JSON.parse(localStorage.getItem("ninfer-monitor-public-demo-history-v1")));
    assert.ok(persisted.samples.every(s=>Object.values(s).every(v=>v===null||typeof v==="number")));
    const dl= p.waitForEvent("download");await p.locator("#jsonExport").click();assert.match((await dl).suggestedFilename(),/\.json$/);
    const csv=p.waitForEvent("download");await p.locator("#csvExport").click();assert.match((await csv).suggestedFilename(),/\.csv$/);
    await p.locator("#chatInput").fill("demo-test-only");
    await p.locator("#chatSend").click();
    await p.waitForFunction(()=>document.querySelector("#chatLog").textContent.includes("demo-test-only"));
    assert.match(await p.locator("#chatLog").textContent(),/simulat|Simulat/i);
    await p.locator("#chatReset").click();assert.equal(await p.locator("#chatLog").textContent(),"");
    await p.locator("#clearHistory").click();assert.equal(await p.evaluate(()=>localStorage.getItem("ninfer-monitor-public-demo-history-v1")),null);
    p.off("request",capture);
    assert.ok(requests.every(url=>url===base+path),JSON.stringify(requests));
  }
  assert.deepEqual(errors,[]);
  console.log("PASS: configurator, incompatible backend, invalid input, memory, clipboard, theme, German locale, model filter, full-text search, stars, releases and mobile width");
} finally {
  if(browser)await browser.close();
  server.kill();
}
