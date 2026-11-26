import { chromium } from "playwright";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";

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
  assert.match(await p.locator("#search-status").textContent(),/24 loaded pages/);
  await p.locator("#site-query").fill("zzzz-nonexistent-9999");
  await p.locator("#search-submit").click();
  await p.waitForFunction(()=>document.querySelector("#search-status").textContent.startsWith("0 matches"));
  await p.goto(base);
  await p.waitForFunction(()=>document.querySelector("#star-count").textContent.includes("42 stars"));
  assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,"mobile horizontal overflow");
  await p.goto(base+"changelog.html");
  await p.waitForFunction(()=>document.querySelector("#release-list").textContent.includes("rtx4090-v9.0.0"));
  assert.deepEqual(errors,[]);
  console.log("PASS: configurator, incompatible backend, invalid input, memory, clipboard, theme, German locale, model filter, full-text search, stars, releases and mobile width");
} finally {
  if(browser)await browser.close();
  server.kill();
}
