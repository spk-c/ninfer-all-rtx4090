from pathlib import Path
import subprocess,os,json,re,hashlib
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
model=root.parent/"models"/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
cases=[("none",[]),("dspark3",["--draft-tokens","3","--draft-min-p","0.5"]),("dflash2_7",["--draft-tokens","7","--draft-min-p","0.5"])]
results={}
for name,args in cases:
    spec={"none":"none","dspark3":"dspark","dflash2_7":"dflash2"}[name]
    cmd=[str(apps/"ninfer.exe"),str(model),"--max-context","8192","--kv-dtype","rk8v4","--expert-residency","device","--spec",spec,"--greedy","--max-new","512","--ngram-draft-tokens","0","--prompt","Write a Python function that adds two integers and a unit test for it."]+args
    p=subprocess.run(cmd,cwd=apps,env=env,capture_output=True,timeout=100)
    for channel in ("stdout","stderr"):
        (root/"atomicchat-adapter"/("drafter-"+name+"."+channel+".log")).write_bytes(getattr(p,channel))
    log=(p.stderr+p.stdout).decode("utf-8",errors="replace")
    def metric(key):
        m=re.search(r"summary\s+"+key+r"\s+([^\r\n]+)",log)
        return m.group(1).strip() if m else None
    results[name]={"exit_code":p.returncode,"decode":metric("decode speed"),"prefill":metric("prefill speed"),"generated":metric("generated tokens"),"finish":metric("finish reason"),"stdout_sha256":hashlib.sha256(p.stdout).hexdigest(),"draft_metrics":[s.strip() for s in log.splitlines() if s.startswith("summary") and any(k in s.lower() for k in ("draft","accept","speculat"))]}
    if p.returncode:results[name]["error"]=log[-2500:]
    results[name]["matches_no_drafter"]=results[name]["stdout_sha256"]==results["none"]["stdout_sha256"] and p.returncode==0
    print(name,json.dumps(results[name]),flush=True)
    (root/"atomicchat-adapter"/"drafter-ab.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
