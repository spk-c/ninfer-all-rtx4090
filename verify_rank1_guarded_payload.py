"""Confirm that guarded v3 metadata is the only semantic change."""
from pathlib import Path
import struct,json,hashlib
root=Path(__file__).resolve().parent
models=root.parent/"models"
old=models/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"
new=models/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-v1-EXPERIMENTAL.ninfer"
def read_dir(f):
    magic,size,uid=struct.unpack("<8sQ16s",f.read(32))
    assert magic==b"NINFER\0\3"
    d=json.loads(f.read(size))
    return d,(32+size+4095)//4096*4096
with old.open("rb") as a,new.open("rb") as b:
    x,ax=read_dir(a); y,by=read_dir(b)
    assert x["components"]["text"]["config"]["model_type"]=="inkling2"
    assert y["components"]["text"]["config"]["model_type"]=="inkling2_rank1_v1"
    y["components"]["text"]["config"]["model_type"]="inkling2"
    for key in ("components","objects","bindings","uses","files"):
        assert x.get(key)==y.get(key),key
    a.seek(ax);b.seek(by)
    n=0;digest=hashlib.sha256()
    while True:
        u=a.read(8*1024*1024);v=b.read(8*1024*1024)
        assert u==v,f"payload differs at {n}"
        if not u:break
        digest.update(u);n+=len(u)
report={"status":"passed","objects":len(x["objects"]),"payload_bytes":n,"payload_sha256":digest.hexdigest(),"model_type":"inkling2_rank1_v1"}
(root/"atomicchat-adapter"/"guarded-payload-verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report),flush=True)
