"""Matched chat-adapted HumanEval generation only; no generated code is executed.
Official dataset pinned to a Git commit. Evaluation must use an isolated sandbox.
"""
from pathlib import Path
import subprocess,os,json,time,urllib.request,urllib.error,gzip,hashlib,argparse
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
models=root.parent/"models"
out=root/"atomicchat-adapter"/"humaneval"
out.mkdir(exist_ok=True)
revision="b9453e8f1865df0a0057a1383767caa400acfbcf"
source=f"https://raw.githubusercontent.com/openai/human-eval/{revision}/data/HumanEval.jsonl.gz"
dataset=out/"HumanEval.jsonl.gz"
if not dataset.exists():
    with urllib.request.urlopen(source,timeout=60) as f:raw=f.read(4*1024*1024)
    dataset.write_bytes(raw)
raw=dataset.read_bytes()
problems=[json.loads(line) for line in gzip.decompress(raw).splitlines() if line.strip()]
assert len(problems)==164 and len({p["task_id"] for p in problems})==164
problems.sort(key=lambda p:int(p["task_id"].split("/")[-1]))
parser=argparse.ArgumentParser()
parser.add_argument("--prepare",action="store_true")
parser.add_argument("--count",type=int,default=164,choices=range(1,165))
args=parser.parse_args()
# Evenly spread subsets if explicitly requested; default is the entire official set.
tasks=problems if args.count==164 else [problems[i*164//args.count] for i in range(args.count)]
variants={"original":"Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer","rank1":"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"}
settings={"context":8192,"kv":"rk8v4","drafter":"dflash2","draft_tokens":7,"draft_min_p":0.5,"ngram":0,"max_tokens":4096,"default_thinking_budget":2048,"temperature":0,"prompt_protocol":"chat full standalone function; not canonical completion prompt"}
manifest={"source":source,"revision":revision,"sha256":hashlib.sha256(raw).hexdigest(),"task_ids":[p["task_id"] for p in tasks],"settings":settings,"status":"prepared","completed":{}}
manifest_path=out/"manifest.json"
def save():manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
if args.prepare:
    print(json.dumps(manifest),flush=True);save();raise SystemExit(0)
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
port=8082
url=f"http://127.0.0.1:{port}"
def call(path,data=None,timeout=60):
    r=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=timeout) as f:return json.load(f)
def ensure_idle():
    probe=subprocess.run(["tasklist","/FI","IMAGENAME eq ninfer-serve.exe","/NH"],capture_output=True,timeout=10,check=True)
    if b"ninfer-serve.exe" in probe.stdout.lower():raise RuntimeError("NInfer server already running; refusing to interfere")
    probe=subprocess.run(["tasklist","/FI","IMAGENAME eq ninfer.exe","/NH"],capture_output=True,timeout=10,check=True)
    if b"ninfer.exe" in probe.stdout.lower():raise RuntimeError("NInfer CLI already running")
    try:call("/v1/models",timeout=1)
    except (OSError,urllib.error.URLError):pass
    else:raise RuntimeError("Port already occupied")
for variant in variants:
    if (out/(variant+".jsonl")).exists():raise FileExistsError("Refusing to overwrite previous candidates")
manifest["status"]="running";save()
try:
    for variant,filename in variants.items():
        ensure_idle()
        model="inkling2-humaneval-"+variant
        with (out/(variant+".server.log")).open("wb") as log:
            cmd=[str(apps/"ninfer-serve.exe"),str(models/filename),"--model-id",model,"--host","127.0.0.1","--port",str(port),"--max-context","8192","--kv-dtype","rk8v4","--expert-residency","device","--max-concurrency","1","--spec","dflash2","--draft-tokens","7","--draft-min-p","0.5","--ngram-draft-tokens","0","--default-thinking-budget","2048","--default-max-tokens","4096"]
            p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                for i in range(90):
                    if p.poll() is not None:raise RuntimeError("server exited "+str(p.returncode))
                    try:call("/v1/models",timeout=2);break
                    except (OSError,urllib.error.URLError):time.sleep(1)
                else:raise RuntimeError("startup timeout")
                with (out/(variant+".jsonl")).open("x",encoding="utf-8") as f:
                    for n,t in enumerate(tasks,1):
                        prompt="Complete the Python programming task below. Return the complete standalone Python function, including any needed imports and the specified function signature. Return only Python code, no markdown or explanation.\n\n"+t["prompt"]
                        request={"model":model,"temperature":0,"max_tokens":4096,"stream":False,"messages":[{"role":"user","content":prompt}]}
                        try:
                            result=call("/v1/chat/completions",request,timeout=120)
                            choice=result["choices"][0]
                            row={"task_id":t["task_id"],"entry_point":t["entry_point"],"candidate":choice["message"].get("content") or "","finish":choice.get("finish_reason"),"usage":result.get("usage"),"timings":result.get("timings")}
                        except (OSError,urllib.error.URLError) as exc:
                            row={"task_id":t["task_id"],"entry_point":t["entry_point"],"error":repr(exc),"candidate":""}
                        f.write(json.dumps(row)+"\n");f.flush()
                        manifest["completed"][variant]=n;save()
                        print(variant,n,"/",len(tasks),t["task_id"],row.get("finish",row.get("error")),flush=True)
                        if p.poll() is not None:raise RuntimeError("server died")
            finally:
                if p.poll() is None:
                    p.terminate()
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=20)
    manifest["status"]="generation_completed_not_evaluated";save()
except BaseException as e:
    manifest.update(status="failed",error=repr(e));save();raise
