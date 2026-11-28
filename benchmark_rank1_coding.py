from pathlib import Path
import subprocess,os,json,re
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
models=root.parent/"models"
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
cases={
"unique":"Implement Python function stable_unique(xs) returning the first occurrence of each hashable item in input order. Return only Python code, no prose.",
"brackets":"Implement Python function balanced(s) returning True only if a string contains a balanced arrangement of brackets (), [], {}. Ignore all other characters. Return only Python code, no prose.",
"merge":"Implement Python function merge_intervals(xs) taking a list of integer pairs [start,end] with start <= end. Return a new sorted list of lists, merging overlapping or touching intervals, without mutating input. Return only Python code, no prose."
}
results={}
for variant,model in [("original","Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer"),("rank1","Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer")]:
    for task,prompt in cases.items():
        cmd=[str(apps/"ninfer.exe"),str(models/model),"--max-context","8192","--kv-dtype","rk8v4","--expert-residency","device","--spec","dflash2","--draft-tokens","7","--draft-min-p","0.5","--greedy","--no-thinking","--max-new","1024","--prompt",prompt]
        p=subprocess.run(cmd,cwd=apps,env=env,capture_output=True,timeout=100)
        out=p.stdout.decode("utf-8",errors="replace")
        err=p.stderr.decode("utf-8",errors="replace")
        def metric(key):
            m=re.search(r"summary\s+"+key+r"\s+([^\r\n]+)",err)
            return m.group(1).strip() if m else None
        results[variant+"-"+task]={"exit_code":p.returncode,"output":out,"finish":metric("finish reason"),"decode":metric("decode speed"),"generated":metric("generated tokens")}
        if p.returncode:results[variant+"-"+task]["error"]=err[-1200:]
        (root/"atomicchat-adapter"/"coding-ab.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
        print(variant,task,json.dumps(results[variant+"-"+task]),flush=True)
