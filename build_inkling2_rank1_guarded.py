"""Build the experimental artifact. Requires the matching patched NInfer runtime."""
from pathlib import Path
import json, subprocess, sys
from huggingface_hub import hf_hub_download
from tools.reference.inkling2_rank1_adapter import sha256
root = Path(__file__).resolve().parent
inputs = root / "inkling2-inputs"
out = root.parent/"models"/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-v1-EXPERIMENTAL.ninfer"
adapter = root/"atomicchat-adapter"/"Inkling-2-Mini-Uncensored-LoRA.gguf"
base = inputs/"text"/"Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf"
state_path = root/"atomicchat-adapter"/"artifact-guarded-build.json"
state = {"status":"preflight","output":str(out),"runtime":"requires inkling2_rank1_v1 parser and matching runtime; not yet GPU validated"}
def save(): state_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
save()
try:
    if sha256(base) != "8a3515374c3cd9b65a1677b17e805bb1e32df8e98756e5f0d2fe1837d62aa4cb":
        raise RuntimeError("Base hash mismatch")
    if sha256(adapter) != "42914b85c874e7fe249e93f0ac6e8abf08dc0bdf32aa1ec46c72714e64808f12":
        raise RuntimeError("Adapter hash mismatch")
    manifest = json.loads((inputs/"build-manifest.json").read_text())
    state["sources"] = manifest["sources"]
    state["status"] = "downloading_components"
    save()
    for role in ("vision","dspark","dflash2"):
        repo,revision,files = manifest["sources"][role]
        for name in files:
            print("DOWNLOADING",role,name,flush=True)
            hf_hub_download(repo,name,revision=revision,local_dir=inputs/role)
    if out.exists(): raise FileExistsError(out)
    temp=out.with_suffix(".partial.ninfer")
    if temp.exists(): raise FileExistsError(temp)
    cmd = [sys.executable,"-m","tools.convert","--model",str(inputs/"config"),"--recipe","inkling2_gguf",
           "--source","gguf="+str(base),
           "--source","rank1="+str(adapter),
           "--source","vision="+str(inputs/"vision"/manifest["sources"]["vision"][2][0]),
           "--source","dspark="+str(inputs/"dspark"/manifest["sources"]["dspark"][2][0]),
           "--source","dflash2="+str(inputs/"dflash2"),
           "--components","text,vision,dspark,dflash2","--name","Inkling2-GSQ-RCO-IQ3S-Rank1-v1-EXPERIMENTAL",
           "--out",str(temp),"--device","cpu"]
    state.update(status="converting",command=cmd)
    save()
    subprocess.run(cmd,cwd=root,check=True)
    report=Path(str(temp)+".conversion.json")
    if not temp.is_file() or temp.stat().st_size < 1_000_000_000:
        raise RuntimeError("Invalid artifact")
    if not report.is_file(): raise RuntimeError("Missing conversion report")
    rep=json.loads(report.read_text(encoding="utf-8"))
    report_out=Path(str(out)+".conversion.json")
    temp.rename(out)
    report.rename(report_out)
    state.update(status="converted_not_runtime_validated",bytes=out.stat().st_size,sha256=sha256(out),report=str(report_out))
    save()
    print("CONVERTED",out,state["bytes"],state["sha256"],flush=True)
except BaseException as e:
    state.update(status="failed",error=repr(e))
    save()
    raise
