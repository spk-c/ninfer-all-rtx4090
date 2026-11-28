"""Resume the original GSQ-RCO source download and test the actual base rows."""
from pathlib import Path
import json, subprocess, sys
from huggingface_hub import hf_hub_download
from tools.reference.inkling2_rank1_adapter import sha256
root = Path(__file__).resolve().parent
status = root / "atomicchat-adapter" / "base-validation.json"
repo = "ISTA-DASLab/Inkling-2-Mini-GSQ-RCO-GGUF"
revision = "f0da1097633dd7c2f5733e4077cff861be122245"
name = "Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf"
expected = "8a3515374c3cd9b65a1677b17e805bb1e32df8e98756e5f0d2fe1837d62aa4cb"
state = {"repo":repo,"revision":revision,"name":name,"status":"downloading"}
def save():
    status.write_text(json.dumps(state,indent=2),encoding="utf-8")
save()
try:
    p = Path(hf_hub_download(repo,name,revision=revision,local_dir=root/"inkling2-inputs"/"text"))
    digest = sha256(p)
    if digest != expected:
        raise ValueError(f"Unexpected base digest: {digest}")
    state.update(status="validating",sha256=digest,bytes=p.stat().st_size)
    save()
    subprocess.run([sys.executable,str(root/"test_atomic_rank1_reference.py"),"--base",str(p)],cwd=root,check=True)
    state["status"] = "row_checks_passed_not_runtime_validated"
    save()
except BaseException as e:
    state.update(status="failed",error=repr(e))
    save()
    raise
