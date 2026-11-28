#!/usr/bin/env python3
"""Select the hosted CI matrix; release workflows retain their own matrices."""
import json
import os

CUDA_ARCHITECTURES = {
    "12.9.2": ("70", "75", "80", "86", "89"),
    "13.4.3": ("75", "80", "86", "89", "120a"),
}


def full_coverage(event, ref):
    return event in ("workflow_dispatch", "schedule") or (
        event == "push" and ref.startswith("refs/heads/ci/full/")
    )


def select_matrix(event, ref):
    full = full_coverage(event, ref)
    rows = [
        {"cuda": cuda, "archs": arch}
        for cuda, architectures in CUDA_ARCHITECTURES.items()
        for arch in architectures
        if full or arch == "89"
    ]
    shards = [1, 2, 3] if full else [1]
    return {"include": rows}, shards


if __name__ == "__main__":
    matrix, shards = select_matrix(os.getenv("GITHUB_EVENT_NAME", ""),
                                   os.getenv("GITHUB_REF", ""))
    print("cuda_matrix=" + json.dumps(matrix, separators=(",", ":")))
    print("windows_shards=" + json.dumps(shards))
    print("windows_shard_count=" + str(len(shards)))
