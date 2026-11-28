from pathlib import Path
import subprocess,os,json,hashlib
root=Path(__file__).resolve().parent
apps=root/"build-rank1-sm89"/"apps"
models=root.parent/"models"
guarded=models/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-v1-EXPERIMENTAL.ninfer"
old=models/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"
base=models/"Inkling-2-Mini-GSQ-RCO-IQ3S-Vision-DFlash2-DSpark-ninfer-v3.ninfer"
env=dict(os.environ)
env["PATH"]=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin;C:\coding\ninfer-codex-deps\vcpkg\x64-windows\bin;"+env["PATH"]
cases=[("legacy_reject_guarded","ninfer-rank1-unguarded-reference.exe",guarded,False),("new_reject_unguarded","ninfer.exe",old,False),("new_accept_base","ninfer.exe",base,True),("new_accept_guarded","ninfer.exe",guarded,True),("legacy_accept_unguarded","ninfer-rank1-unguarded-reference.exe",old,True)]
results={}
for name,exe,model,ok in cases:
    cmd=[str(apps/exe),str(model),"--max-context","8192","--kv-dtype","rk8v4","--expert-residency","device","--spec","dflash2","--draft-tokens","7","--draft-min-p","0.5","--ngram-draft-tokens","0","--greedy","--no-thinking","--max-new","512","--prompt","Write a Python function that adds two integers and a unit test for it."]
    p=subprocess.run(cmd,cwd=apps,env=env,capture_output=True,timeout=90)
    err=p.stderr.decode("utf-8",errors="replace")
    out=p.stdout.decode("utf-8",errors="replace")
    (root/"atomicchat-adapter"/("guard-"+name+".log")).write_text(err+"\n"+out,encoding="utf-8")
    reason=next((x for x in err.splitlines() if "mismatch" in x.lower() or "error" in x.lower()),"")
    passed=(p.returncode==0)==ok and (ok or ("mismatch" in err.lower() and "inkling2" in err.lower()))
    results[name]={"exit_code":p.returncode,"passed":passed,"reason":reason,"sha256":hashlib.sha256(p.stdout).hexdigest(),"output":out[:1500]}
    print(name,json.dumps(results[name]),flush=True)
results["guarded_matches_previous"]=results["new_accept_guarded"]["sha256"]==results["legacy_accept_unguarded"]["sha256"]
(root/"atomicchat-adapter"/"guard-runtime-smoke.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
assert all(results[n]["passed"] for n,*_ in cases) and results["guarded_matches_previous"]
