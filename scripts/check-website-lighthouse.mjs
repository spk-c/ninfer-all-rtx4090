import { chromium } from "playwright";
import lighthouse from "lighthouse";
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
const base="http://127.0.0.1:8766/";
const server=spawn("python3",["-m","http.server","8766","--bind","127.0.0.1","--directory","site"],{stdio:"ignore"});
let chrome;
try {
  for(let i=0;i<50;i++){try{if((await fetch(base)).ok)break}catch{};await new Promise(r=>setTimeout(r,100))}
  const port=9223;
  chrome=spawn(chromium.executablePath(),["--headless","--no-sandbox","--disable-dev-shm-usage","--remote-debugging-port="+port,"--remote-debugging-address=127.0.0.1","--disable-gpu","--user-data-dir=/tmp/ninfer-lighthouse-"+process.pid],{stdio:"ignore"});
  for(let i=0;i<50;i++){try{if((await fetch("http://127.0.0.1:"+port+"/json/version")).ok)break}catch{};await new Promise(r=>setTimeout(r,100))}
  mkdirSync("lighthouse-results",{recursive:true});
  for(const device of ["mobile","desktop"]){
    for(const [name,path] of [["home",""],["configurator","configurator.html"]]){
      const desktop=device==="desktop";
      const flags={port,output:"json",logLevel:"error",onlyCategories:["performance","accessibility","best-practices","seo"],formFactor:device,screenEmulation:desktop?{mobile:false,width:1350,height:940,deviceScaleFactor:1,disabled:false}:{mobile:true,width:412,height:823,deviceScaleFactor:1.75,disabled:false},throttlingMethod:desktop?"provided":"simulate",skipAudits:["uses-http2","is-on-https"]};
      const result=await lighthouse(base+path,flags);
      const scores=Object.fromEntries(Object.entries(result.lhr.categories).map(([k,v])=>[k,v.score]));
      console.log(device,name,JSON.stringify(scores));
      writeFileSync("lighthouse-results/"+device+"-"+name+".json",JSON.stringify(result.lhr,null,2));
    }
  }
} finally {if(chrome)chrome.kill();server.kill()}
