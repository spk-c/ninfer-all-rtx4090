"""Cross-link two owned model cards; preserves all other bytes and files."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import HfApi, hf_hub_download

IDS = ["SPK1337/Huihui-Qwen3.8-27B-Uncensored-GSQ-RCO-IQ3S-Vision-DFlash2-MTP-NInfer","SPK1337/Huihui-Qwen3.8-27B-Swift1.5-Abliterated-IQ3S-DFlash2-NInfer"]
MARKER = b"**Project portal:**"
p=argparse.ArgumentParser()
p.add_argument("--apply",action="store_true")
args=p.parse_args()
api=HfApi()
if args.apply and api.whoami().get("name","").casefold()!="spk1337":
    raise SystemExit("Authenticate as SPK1337 (hf auth login).")
for index, repo in enumerate(IDS):
    other=IDS[1-index]
    revision=api.model_info(repo).sha
    original=Path(hf_hub_download(repo,"README.md",revision=revision)).read_bytes()
    url="https://huggingface.co/"+other
    if url.encode("utf-8") in original:
        print(repo,"already linked",url);continue
    if original.count(MARKER)!=1:
        raise SystemExit("Missing or ambiguous project portal insertion point: "+repo)
    start=original.index(MARKER)
    end=original.find(b"\n",start)
    if end<0:raise SystemExit("Unexpected README formatting: "+repo)
    nl=b"\r\n" if original[end-1:end]==b"\r" else b"\n"
    if index==0:
        note=("**Related model:** [Huihui Qwen3.8 27B Swift 1.5 IQ3_S — NInfer]("+url+"). "
              "A separate Swift 1.5 conversion, also packaging DFlash2, MTP and optional vision. "
              "Its historical HTTP benchmark includes sustained 2,048-output-token tests at up to 160K input.")
    else:
        note=("**Related model:** [Huihui Qwen3.8 27B IQ3_S (without Swift 1.5) — NInfer]("+url+"). "
              "A separate conversion with DFlash2, MTP and optional vision; the historical serving report covers 1,831 completed requests.")
    note+=" Both artifacts are approximately 13.99 GiB on disk. Benchmark workloads and builds differ, so the results do not establish a direct model speed or quality ranking. Choose the appropriate artifact and measure on your own workload."
    updated=original[:end+1]+nl+note.encode("utf-8")+nl+original[end+1:]
    if not args.apply:
        print(repo,"would add:",note);continue
    with TemporaryDirectory() as tmp:
        path=Path(tmp)/"README.md"
        path.write_bytes(updated)
        result=api.upload_file(path_or_fileobj=path,path_in_repo="README.md",repo_id=repo,repo_type="model",parent_commit=revision,commit_message="docs: cross-link companion Huihui NInfer model cards")
        saved=Path(hf_hub_download(repo,"README.md",revision=result.oid)).read_bytes()
        if saved!=updated:raise SystemExit("Read-back verification failed: "+repo)
        print(repo,"published and verified",result.oid)
