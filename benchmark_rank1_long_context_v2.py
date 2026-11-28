"""Local synthetic long-context retrieval and real prefill qualification.
Not a general long-context or coding benchmark. Never runs generated code.
"""
from pathlib import Path
import subprocess,json,time,urllib.request,urllib.error,random,re,os,argparse
root=Path(__file__).resolve().parent
script=root.parent/"Start-Inkling2-GSQ-RCO-Rank1-RTX4090-Experimental.ps1"
logdir=root/"atomicchat-adapter"
port=8082
url=f"http://127.0.0.1:{port}"
model="inkling2-gsq-rco-rank1-experimental"
parser=argparse.ArgumentParser()
parser.add_argument("--context",type=int,choices=[131072,262144],default=262144)
parser.add_argument("--variant",choices=["rank1","original"],default="rank1")
args=parser.parse_args()
variant=args.variant
apps=root/"build-rank1-sm89"/"apps"
filename={"rank1":"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer","original":"Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer"}[variant]
artifact=root.parent/"models"/filename
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
results={}
def save(): (logdir/f"long-context-{variant}-{args.context}.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
def call(path,data=None,timeout=60):
    r=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=timeout) as f:return json.load(f)
try:call("/v1/models",timeout=1)
except (OSError,urllib.error.URLError):pass
else:raise RuntimeError("Port already occupied")
for context,target in [(args.context,125000 if args.context==131072 else 250000)]:
    record={"status":"starting","context_limit":context}
    results[str(context)]=record;save()
    with (logdir/f"long-context-server-{variant}-{context}.log").open("wb") as log:
        # Launch the exact executable as the owned child; no PowerShell child-tree race.
        p=subprocess.Popen([str(apps/"ninfer-serve.exe"),str(artifact),"--model-id",model,"--host","127.0.0.1","--port",str(port),"--max-context",str(context),"--kv-dtype","rk8v4","--expert-residency","device","--max-concurrency","1","--prefill-chunk","2048","--spec","dflash2","--draft-tokens","7","--draft-min-p","0.5","--ngram-draft-tokens","0","--default-thinking-budget","16384"],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            for i in range(90):
                if p.poll() is not None:raise RuntimeError("Server exited "+str(p.returncode))
                try:call("/v1/models",timeout=2);break
                except (OSError,urllib.error.URLError):time.sleep(1)
            else:raise RuntimeError("Server startup timeout")
            rng=random.Random(context)
            lines=[f"Record {i:06d}: inventory {rng.randrange(1000,9999)}; queue {rng.randrange(10000,99999)}; region {rng.choice(['north','south','east','west'])}; status {rng.choice(['ready','pending','closed'])}." for i in range(30000)]
            needles={"ALPHA":"K7Q4Z9","BETA":"M3X8R2","GAMMA":"P6V1N5"}
            def prompt(n):
                data=lines[:n]
                for pos,(label,value) in zip([max(1,n//20),n//2,n-n//20],needles.items()):
                    data.insert(pos,f"IMPORTANT: lookup key {label} has exact code {value}.")
                return "Read the inventory records. Remember exact codes for keys ALPHA, BETA and GAMMA.\n"+"\n".join(data)+"\nReturn only the three exact codes in order ALPHA, BETA, GAMMA, separated by commas."
            def count(s):return len(call("/tokenize",{"content":s,"add_special":False},timeout=120)["tokens"])
            n=1000
            c=count(prompt(n))
            n=int(n*target/c)
            for _ in range(5):
                s=prompt(n);c=count(s)
                if abs(c-target)<500:break
                n=max(1,min(30000,int(n*target/c)))
            s=prompt(n);c=count(s)
            record.update(status="generating",raw_prompt_tokens=c,records=n,expected=list(needles.values()));save()
            print("RUN",context,c,flush=True)
            result=call("/v1/chat/completions",{"model":model,"temperature":0,"max_tokens":512,"stream":False,"messages":[{"role":"user","content":s}]},timeout=1500)
            content=result["choices"][0]["message"].get("content") or ""
            matched=[code for code in needles.values() if code in content]
            record.update(status="completed",usage=result.get("usage"),timings=result.get("timings"),finish=result["choices"][0].get("finish_reason"),answer=content,matched=matched,all_retrieved=len(matched)==3)
            print(json.dumps(record),flush=True);save()
        except BaseException as e:
            record.update(status="failed",error=repr(e));save();print(json.dumps(record),flush=True)
        finally:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=20)
            if record["status"]!="completed":raise RuntimeError(record.get("error","test failed"))
