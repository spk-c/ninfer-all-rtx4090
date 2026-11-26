"""Add website backlinks to the two owned model cards. Uses normal authenticated HF session."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import HfApi, hf_hub_download

IDS = [
    "SPK1337/Huihui-Qwen3.8-27B-Uncensored-GSQ-RCO-IQ3S-Vision-DFlash2-MTP-NInfer",
    "SPK1337/Huihui-Qwen3.8-27B-Swift1.5-Abliterated-IQ3S-DFlash2-NInfer",
]
BASE = "https://spk-c.github.io/ninfer-all-rtx4090/"
BLOCK = ("**Project portal:** [RTX 4090 local LLM guide]("+BASE+") · "
         "[VRAM and launch configurator]("+BASE+"configurator.html) · "
         "[Reproducible benchmarks]("+BASE+"benchmarks.html).\n\n")
p=argparse.ArgumentParser()
p.add_argument("--apply",action="store_true")
args=p.parse_args()
api=HfApi()
if args.apply and api.whoami().get("name","").casefold()!="spk1337":
    raise SystemExit("Authenticate as SPK1337 using hf auth login.")
for repo in IDS:
    revision=api.model_info(repo).sha
    original=Path(hf_hub_download(repo,"README.md",revision=revision)).read_text(encoding="utf-8")
    if BASE in original:
        print(repo,"website already linked");continue
    marker="## NInfer-all RTX 4090 — Windows CUDA inference engine"
    if marker not in original:
        raise SystemExit("Unexpected model card; not modifying "+repo)
    updated=original.replace(marker+"\n\n",marker+"\n\n"+BLOCK,1)
    if updated==original:raise SystemExit("Could not locate insertion point")
    if not args.apply:
        print(repo,"would add",BLOCK);continue
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/"README.md";path.write_text(updated,encoding="utf-8")
        commit=api.upload_file(path_or_fileobj=path,path_in_repo="README.md",repo_id=repo,repo_type="model",parent_commit=revision,commit_message="docs: link NInfer RTX 4090 project portal and configurator")
        saved=Path(hf_hub_download(repo,"README.md",revision=commit.oid)).read_text(encoding="utf-8")
        if saved!=updated:raise SystemExit("Read-back verification failed")
        print(repo,"published and verified",commit.oid)
