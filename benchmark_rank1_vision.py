from pathlib import Path
import subprocess,os,json,re
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
models=root.parent/"models"
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
cases={"original":"Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer","rank1":"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"}
results={}
for name,model in cases.items():
    cmd=[str(apps/"ninfer.exe"),str(models/model),"--max-context","8192","--kv-dtype","rk8v4","--expert-residency","device","--spec","dflash2","--draft-tokens","7","--draft-min-p","0.5","--greedy","--no-thinking","--max-new","512","--vision","--messages","examples/cli/messages/image_natural.json"]
    p=subprocess.run(cmd,cwd=root,env=env,capture_output=True,timeout=100)
    out=p.stdout.decode("utf-8",errors="replace")
    err=p.stderr.decode("utf-8",errors="replace")
    (root/"atomicchat-adapter"/("vision-"+name+".log")).write_text(err+"\n"+out,encoding="utf-8")
    results[name]={"exit_code":p.returncode,"output":out,"summary":[x.strip() for x in err.splitlines() if x.startswith("summary") or x.startswith("generate    vision")]}
    if p.returncode:results[name]["error"]=err[-1800:]
    print(name,json.dumps(results[name],ensure_ascii=True),flush=True)
(root/"atomicchat-adapter"/"vision-ab.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
