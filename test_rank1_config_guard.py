"""Compile the old and new config parsers independently, without using the GPU."""
from pathlib import Path
import subprocess, json, struct
root=Path(__file__).resolve().parent
scratch=root/"atomicchat-adapter"/"guard-test"
scratch.mkdir(exist_ok=True)
model=root.parent/"models"/"Inkling-2-Mini-GSQ-RCO-IQ3S-Rank1-EXPERIMENTAL.ninfer"
with model.open("rb") as f:
    magic,n,uid=struct.unpack("<8sQ16s",f.read(32))
    assert magic==b"NINFER\x00\x03"
    directory=json.loads(f.read(n))
cfg=directory["components"]["text"]["config"]
assert cfg["model_type"]=="inkling2"
(scratch/"config.json").write_text(json.dumps(cfg),encoding="utf-8")
old=subprocess.check_output(["git","show","466f027ea5c8397da33ccdf2a9826c82dbb85fcc:src/models/inkling2/config.cpp"],cwd=root)
(scratch/"legacy.cpp").write_bytes(old)
print("Fixture extracted from real v3 model; old parser extracted from base commit.")
