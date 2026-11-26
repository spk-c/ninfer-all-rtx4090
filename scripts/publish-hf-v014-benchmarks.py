"""Add the verified v0.1.4 benchmark section to the matching HF model card.

Preview is read-only. Use --apply after authenticating as SPK1337 via the
normal Hugging Face CLI. Never reads or writes model weights.
"""
from pathlib import Path
import argparse
import hashlib

REPO = "SPK1337/Huihui-Qwen3.8-27B-Uncensored-GSQ-RCO-IQ3S-Vision-DFlash2-MTP-NInfer"
MODEL = "Huihui-Qwen3.8-27B-GSQ-RCO-IQ3S-Vision-DFlash2.ninfer"
DIGEST = "a410396311ccb27048008a7ccb8a38f91099f6f6a8e10d10bd8baed7248c046a"
MARKER = b"## NInfer v0.1.4 - controlled long-context and coding benchmarks"
INSERT_BEFORE = b"## Model at a glance"
REPORT = Path(__file__).resolve().parents[1] / "docs/performance/rtx4090-v014-huihui-model-card-addendum.md"

def updated_card(original, addendum):
    if MARKER in original:
        return original
    if original.count(INSERT_BEFORE) != 1:
        raise ValueError("Missing or ambiguous insertion point")
    if not addendum.startswith(b"# "):
        raise ValueError("Unexpected report heading")
    nl = b"\r\n" if b"\r\n" in original else b"\n"
    body = addendum.replace(b"\r\n", b"\n").split(b"\n", 1)[1].strip()
    section = MARKER + b"\n\n" + body + b"\n\n"
    return original.replace(INSERT_BEFORE, section.replace(b"\n", nl) + INSERT_BEFORE, 1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if args.apply and api.whoami().get("name", "").casefold() != "spk1337":
        raise SystemExit("Authenticate as SPK1337 with hf auth login first.")
    revision = api.model_info(REPO).sha
    def read(name, rev=revision):
        return Path(hf_hub_download(REPO, name, revision=rev)).read_bytes()
    checksums = read("SHA256SUMS").decode().splitlines()
    if DIGEST + "  " + MODEL not in checksums:
        raise SystemExit("Model checksum differs; refusing to attach benchmark")
    original = read("README.md")
    addendum = REPORT.read_bytes()
    changed = updated_card(original, addendum)
    if changed == original:
        print("Already published; no change.")
        return
    print("Repository:", REPO, "revision:", revision)
    print("Original README SHA-256:", hashlib.sha256(original).hexdigest())
    print("Updated README SHA-256:", hashlib.sha256(changed).hexdigest())
    if not args.apply:
        print("Preview only; rerun with --apply when authenticated.")
        print(changed.decode())
        return
    result = api.upload_file(
        repo_id=REPO, repo_type="model", path_in_repo="README.md",
        path_or_fileobj=changed, parent_commit=revision,
        commit_message="docs: add verified NInfer v0.1.4 RTX 4090 benchmark results",
    )
    if read("README.md", result.oid) != changed:
        raise SystemExit("Read-back verification failed.")
    print("Published and verified:", result.oid)

if __name__ == "__main__":
    main()
