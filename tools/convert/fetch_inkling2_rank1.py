"""Download the original AtomicChat adapter, pinned and SHA256-verified.
The upstream adapter remains hosted by its author; do not vendor binary weights.
"""
from pathlib import Path
import argparse
import hashlib
from huggingface_hub import hf_hub_download

REPO = "AtomicChat/Inkling-2-Mini-Uncensored-GGUF"
REVISION = "c5667da36fe35da026fbee394d00547a921e98ab"
FILENAME = "Inkling-2-Mini-Uncensored-LoRA.gguf"
SHA256 = "42914b85c874e7fe249e93f0ac6e8abf08dc0bdf32aa1ec46c72714e64808f12"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    path = Path(hf_hub_download(REPO, FILENAME, revision=REVISION, local_dir=args.out_dir))
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Adapter checksum mismatch: {digest}")
    print(path.resolve())

if __name__ == "__main__":
    main()
