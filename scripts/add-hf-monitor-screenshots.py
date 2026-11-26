"""Add an actual, demo-data Monitor screenshot and links to both owned cards.

Preview by default. Apply changes README.md only; no model files are modified.
"""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from huggingface_hub import HfApi, hf_hub_download

IDS = ["SPK1337/Huihui-Qwen3.8-27B-Uncensored-GSQ-RCO-IQ3S-Vision-DFlash2-MTP-NInfer","SPK1337/Huihui-Qwen3.8-27B-Swift1.5-Abliterated-IQ3S-DFlash2-NInfer"]
MARKER = b"**Project portal:**"
BASE = "https://spk-c.github.io/ninfer-all-rtx4090/"
IMAGE = BASE + "monitor-desktop.png"
p = argparse.ArgumentParser()
p.add_argument("--apply", action="store_true")
args = p.parse_args()
api = HfApi()
if args.apply and api.whoami().get("name", "").casefold() != "spk1337":
    raise SystemExit("Authenticate as SPK1337 using hf auth login.")
for repo in IDS:
    revision = api.model_info(repo).sha
    original = Path(hf_hub_download(repo, "README.md", revision=revision)).read_bytes()
    if IMAGE.encode() in original:
        print(repo, "already includes Monitor screenshot")
        continue
    if original.count(MARKER) != 1:
        raise SystemExit("Missing or ambiguous project portal insertion point: " + repo)
    start = original.index(MARKER)
    end = original.find(b"\n", start)
    if end < 0:
        raise SystemExit("Unexpected README formatting: " + repo)
    nl = b"\r\n" if original[end-1:end] == b"\r" else b"\n"
    section = (
        "**Built-in NInfer Monitor** — open the local server's /monitor page to inspect "
        "GPU utilization, VRAM, temperature, token throughput, DFlash2 acceptance, KV occupancy "
        "and recent request metadata. No extra Python process is required.\n\n"
        f"[![NInfer Monitor v0.1.4 screenshot with synthetic demonstration data]({IMAGE})]({BASE}monitor.html)\n\n"
        "*Actual v0.1.4 dashboard rendered with synthetic demonstration data; these values are not benchmarks.* "
        f"[Monitor guide]({BASE}monitor.html) · [Mobile screenshot]({BASE}monitor-mobile.png) · "
        f"[Interactive development preview (simulated)]({BASE}monitor-demo.html)."
    ).encode("utf-8").replace(b"\n", nl)
    updated = original[:end+1] + nl + section + nl + original[end+1:]
    if not args.apply:
        print(repo, "would add Monitor screenshot and three links; no model changes")
        continue
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "README.md"
        path.write_bytes(updated)
        result = api.upload_file(
            path_or_fileobj=path, path_in_repo="README.md", repo_id=repo, repo_type="model",
            parent_commit=revision, commit_message="docs: showcase built-in NInfer Monitor",
        )
        saved = Path(hf_hub_download(repo, "README.md", revision=result.oid)).read_bytes()
        if saved != updated:
            raise SystemExit("Read-back verification failed: " + repo)
        print(repo, "published and verified", result.oid)
