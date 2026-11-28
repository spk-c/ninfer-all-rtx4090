from pathlib import Path
import subprocess,os,json,re
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
model=root.parent/"models"/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
results={}
for context in (131072,196608,262144):
    cmd=[str(apps/"ninfer.exe"),str(model),"--max-context",str(context),"--kv-dtype","rk8v4","--expert-residency","device","--spec","dflash2","--draft-tokens","7","--draft-min-p","0.5","--greedy","--max-new","32","--prompt","Return the number 42."]
    p=subprocess.run(cmd,cwd=apps,env=env,capture_output=True,timeout=100)
    log=(p.stderr+p.stdout).decode("utf-8",errors="replace")
    (root/"atomicchat-adapter"/f"capacity-{context}.log").write_text(log,encoding="utf-8")
    def metric(key):
        m=re.search(r"summary\s+"+key+r"\s+([^\r\n]+)",log)
        return m.group(1).strip() if m else None
    results[str(context)]={"exit_code":p.returncode,"max_context":metric("max context"),"kv_capacity":metric("KV capacity"),"prompt":metric("prompt tokens"),"generated":metric("generated tokens"),"free_after_startup":metric("free after startup"),"planned_total":metric("planned device total"),"kv_payload":metric("kv cache payload")}
    if p.returncode:results[str(context)]["error"]=log[-1800:]
    print(context,json.dumps(results[str(context)]),flush=True)
    (root/"atomicchat-adapter"/"capacity.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
