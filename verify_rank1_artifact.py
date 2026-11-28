"""Verify the stored quantized text blocks and the adapter factors."""
from pathlib import Path
import hashlib,json,numpy as np
from tools.artifact.reader import Artifact
from tools.convert.sources.gguf import GGUFFile
from tools.reference.inkling2_rank1_adapter import Rank1Adapter
root=Path(__file__).resolve().parent
path=root.parent/"models"/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"
report=json.loads(Path(str(path)+".conversion.json").read_text(encoding="utf-8"))
base=root/"inkling2-inputs"/"text"/"Inkling-2-Mini-GSQ-RCO-IQ3_S.gguf"
adapter=Rank1Adapter(root/"atomicchat-adapter"/"Inkling-2-Mini-Uncensored-LoRA.gguf")
pairs={}
for name,(a,b) in adapter.pairs.items():
    if name=="token_embd.weight": prefix="text/rank1/embedding"
    elif name.endswith(".attn_output.weight"): prefix="text/layers/"+name.split(".")[1]+"/rank1/attention"
    else: prefix="text/layers/"+name.split(".")[1]+"/rank1/expert"
    pairs[prefix+"_a"]=a
    pairs[prefix+"_b"]=b
stats={"quantized_objects":0,"quantized_bytes":0,"adapter_objects":0,"adapter_bytes":0,"embedding_bytes":0}
with Artifact(path) as art, GGUFFile(base) as gguf:
    assert set(art.directory.components)=={"text","vision","dspark","dflash2"}
    assert len(art.objects)==1352
    for m in report["methods"]:
        params=m["parameters"]
        if len(params)==1 and params[0] in pairs:
            data=art.read_object(m["object"])
            expected=np.ascontiguousarray(pairs.pop(params[0]),dtype="<f4").tobytes()
            assert data==expected, params
            stats["adapter_objects"]+=1
            stats["adapter_bytes"]+=len(data)
        elif all(n.startswith("text/") for n in params) and (m["method"]=="import_encoded" or params==["text/token_embedding"]):
            sources=[s.split("[",1)[0] for s in m["sources"]]
            expected_hash=hashlib.sha256()
            count=0
            for name in sources:
                n=gguf.info(name).nbytes
                for off in range(0,n,8*1024*1024):
                    expected_hash.update(gguf.tensor_bytes(name,off,min(n,off+8*1024*1024)))
                count+=n
            obj=art.object(m["object"])
            assert obj.bytes==count,(params,obj.bytes,count)
            got=hashlib.sha256()
            for chunk in art.iter_object(obj.id,chunk_bytes=8*1024*1024):got.update(chunk)
            assert got.digest()==expected_hash.digest(),params
            if params==["text/token_embedding"]: stats["embedding_bytes"]+=count
            else:
                stats["quantized_objects"]+=1
                stats["quantized_bytes"]+=count
    assert not pairs,pairs.keys()
    assert stats["adapter_objects"]==98
    assert stats["quantized_objects"]>100
    stats["status"]="passed"
(root/"atomicchat-adapter"/"artifact-verification.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
print(json.dumps(stats,indent=2),flush=True)
