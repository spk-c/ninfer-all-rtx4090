from pathlib import Path
import json
from huggingface_hub import HfApi, hf_hub_download
root=Path(__file__).resolve().parent
sources=json.loads((root/"atomicchat-adapter"/"artifact-build.json").read_text())["sources"]
out=root/"atomicchat-adapter"/"licenses"
out.mkdir(exist_ok=True)
api=HfApi()
report={}
for role,(repo,rev,_) in sources.items():
    info=api.model_info(repo,revision=rev)
    card=info.card_data.to_dict() if info.card_data else {}
    names=[f.rfilename for f in info.siblings]
    license_files=[n for n in names if "/" not in n and (n.lower().startswith(("license","notice","copying")) or n.lower()=="readme.md")]
    saved=[]
    for n in license_files:
        f=hf_hub_download(repo,n,revision=rev,local_dir=out/role)
        saved.append(str(f))
    report[role]={"repo":repo,"revision":rev,"license":card.get("license"),"license_name":card.get("license_name"),"license_link":card.get("license_link"),"files":license_files}
    print(role,json.dumps(report[role]),flush=True)
(out/"audit.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
