"""End-to-end FP64 comparison using a fixed benign coding prompt."""
from pathlib import Path
import json, time
import numpy as np
from tools.convert.sources.gguf import GGUFFile
from tools.reference.inkling2 import forward, target_nll
from tools.reference.inkling2_rank1_adapter import Rank1Adapter
root=Path(__file__).resolve().parent
result_path=root/"atomicchat-adapter"/"forward-validation.json"
gguf_path=root/"inkling2-inputs"/"text"/"Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf"
adapter=Rank1Adapter(root/"atomicchat-adapter"/"Inkling-2-Mini-Uncensored-LoRA.gguf")
tokenizer = json.loads((root/"inkling2-inputs"/"config"/"tokenizer.json").read_text(encoding="utf-8"))
vocab = tokenizer["model"]["vocab"]
# Fixed valid IDs for numerical qualification, not a tokenizer-quality test.
ids = [vocab[token] for token in ("def", "(", "a", ")")]
assert len(ids) == 4 and len(set(ids)) == 4
result={"tokens":ids,"status":"running","runs":[]}
def save():
    result_path.write_text(json.dumps(result,indent=2),encoding="utf-8")
save()
try:
    outputs={}
    for name, a, scale in (("original",None,0.),("scale_zero",adapter,0.),("scale_one",adapter,1.)):
        start=time.perf_counter()
        print("RUN",name,flush=True)
        with GGUFFile(gguf_path) as g:
            f=forward(g,ids,adapter=a,adapter_scale=scale)
        logits=f["logits"]
        assert np.isfinite(logits).all()
        outputs[name]=f
        result["runs"].append({"name":name,"elapsed_seconds":time.perf_counter()-start,"mean_nll":float(target_nll(logits,ids).mean()),"argmax":logits.argmax(-1).tolist(),"visits":len([k for k in f if k.startswith("visit-")])})
        save()
    original, disabled, modified=(outputs[n] for n in ("original","scale_zero","scale_one"))
    assert set(original)==set(disabled)==set(modified)
    for k in original:
        np.testing.assert_array_equal(original[k],disabled[k],err_msg=k)
    assert not np.array_equal(original["logits"],modified["logits"])
    result.update(status="passed",scale_zero_bit_identical=True,max_logit_delta=float(np.max(np.abs(original["logits"]-modified["logits"]))),mean_abs_logit_delta=float(np.mean(np.abs(original["logits"]-modified["logits"]))))
    save()
    print(json.dumps(result,indent=2),flush=True)
except BaseException as e:
    result.update(status="failed",error=repr(e))
    save()
    raise
