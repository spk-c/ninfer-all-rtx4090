from pathlib import Path
import subprocess,os,json,re,hashlib,time
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
models=root.parent/"models"
adapt=models/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"
base=models/"Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer"
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
cases=[("bf16_reference","ninfer-rank1-bf16-reference.exe",adapt,False),("q8_no_graph","ninfer.exe",adapt,False),("q8_graph","ninfer.exe",adapt,True),("original_graph","ninfer.exe",base,True)]
results={}
for name,exe,model,graph in cases:
    cmd=[str(apps/exe),str(model),"--max-context","8192","--kv-dtype","rk8v4","--expert-residency","device","--spec","none","--greedy","--max-new","512","--prompt","Write a Python function that adds two integers and a unit test for it."]
    if not graph:cmd+=["--no-cuda-graph"]
    p=subprocess.run(cmd,cwd=apps,env=env,capture_output=True,timeout=100)
    for channel in ("stdout","stderr"):
        (root/"atomicchat-adapter"/(name+"."+channel+".log")).write_bytes(getattr(p,channel))
    log=(p.stderr+p.stdout).decode("utf-8",errors="replace")
    if p.returncode:raise RuntimeError(name+":"+log[-3000:])
    def metric(key):
        m=re.search(r"summary\s+"+key+r"\s+([^\r\n]+)",log)
        return m.group(1).strip() if m else None
    results[name]={"decode":metric("decode speed"),"prefill":metric("prefill speed"),"generated":metric("generated tokens"),"graph_used":metric("CUDA Graph used"),"finish":metric("finish reason"),"stdout_sha256":hashlib.sha256(p.stdout).hexdigest(),"stdout_bytes":len(p.stdout)}
    print(name,results[name],flush=True)
results["graph_output_identical"]=results["q8_no_graph"]["stdout_sha256"]==results["q8_graph"]["stdout_sha256"]
(root/"atomicchat-adapter"/"q8-ab.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
print(json.dumps(results,indent=2),flush=True)
