"""Immutable, verified ccache artifact transfer between GitHub-hosted CUDA shards.

Each shard owns a disjoint set of Ninja object targets. The final job restores all
compiler-cache files, verifies the *whole* expected object set was covered, and
fails rather than silently performing an expensive cold CUDA rebuild.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

SCHEMA = 1
FAMILY = "rtx4090-sm89-shard-cache"
TARGETS = ("ninfer", "ninfer-serve", "ninfer_bench", "ninfer-calibrate",
           "ninfer-perplexity", "ninfer-ngram-profile")


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(name):
    part = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or part.is_absolute()
            or any(x in ("", ".", "..") for x in part.parts)
            or part.as_posix() != name):
        raise ValueError(f"Unsafe cache entry: {name!r}")
    return part


def build_objects(build):
    out = subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "compdb-targets", *TARGETS],
        text=True)
    spec = importlib.util.spec_from_file_location(
        "release_shard_selector", Path(__file__).with_name("build-object-shard.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return json.loads(out), module.select_objects


def pack(source, objects_json, output, sha, shard, count):
    if count < 2 or not 0 <= shard < count or len(sha) != 40:
        raise ValueError("Invalid shard parameters")
    objects = json.loads(objects_json.read_text(encoding="utf-8"))
    if not isinstance(objects, list) or not objects or len(objects) != len(set(objects)):
        raise ValueError("Invalid compiled-object list")
    commands, selector = build_objects(objects_json.parent)
    if objects != selector(commands, shard, count):
        raise ValueError("Shard objects do not match the configured build graph")
    if not source.is_dir():
        raise ValueError("ccache source directory missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = []
    seen_bytes = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symlinks forbidden in compiler cache: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            if (path.name in ("lock", "stats") or path.name.endswith((".lock", ".tmp"))
                    or any(part == "tmp" for part in path.relative_to(source).parts)):
                continue
            safe_name(relative)
            size = path.stat().st_size
            if size == 0:
                continue
            seen_bytes += size
            if seen_bytes > 2 * 1024 ** 3:
                raise ValueError("Shard cache exceeds 2 GiB: reduce CCACHE_MAXSIZE")
            files.append({"path": relative, "size": size, "sha256": hash_file(path)})
            archive.write(path, "cache/" + relative)
        if not files:
            raise ValueError("Refusing to export an empty compiler cache")
        manifest = {"schema": SCHEMA, "family": FAMILY, "revision": sha,
                    "shard": shard, "count": count, "objects": objects, "files": files}
        archive.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
    print(f"Shard {shard}/{count}: {len(objects)} objects, {len(files)} cache files,"
          f" {output.stat().st_size} archive bytes", flush=True)


def restore(input_root, destination, build, sha, count):
    if count < 2 or len(sha) != 40:
        raise ValueError("Invalid restore parameters")
    commands, selector = build_objects(build)
    expected = [selector(commands, i, count) for i in range(count)]
    all_objects = set()
    all_files = {}
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        archive_file = input_root / f"rtx4090-objects-{index}" / "cache.zip"
        if not archive_file.is_file():
            raise ValueError(f"Missing shard artifact: {archive_file}")
        with zipfile.ZipFile(archive_file) as archive:
            metadata = json.loads(archive.read("manifest.json"))
            if any((metadata.get("schema") != SCHEMA,
                    metadata.get("family") != FAMILY,
                    metadata.get("revision") != sha,
                    metadata.get("shard") != index,
                    metadata.get("count") != count)):
                raise ValueError(f"Wrong source/toolchain identity in shard {index}")
            if metadata.get("objects") != expected[index]:
                raise ValueError(f"Stale/different Ninja object graph in shard {index}")
            if all_objects.intersection(metadata["objects"]):
                raise ValueError("Object ownership overlaps between shards")
            all_objects.update(metadata["objects"])
            file_entries = metadata.get("files")
            if not isinstance(file_entries, list) or not file_entries:
                raise ValueError("Empty shard cache")
            permitted = {"manifest.json"}
            for item in file_entries:
                name = safe_name(item["path"]).as_posix()
                if len(item["sha256"]) != 64 or item["size"] < 1:
                    raise ValueError("Malformed cache manifest entry")
                member = "cache/" + name
                if member in permitted:
                    raise ValueError("Duplicate archive member")
                permitted.add(member)
                target = destination / Path(*PurePosixPath(name).parts)
                if name in all_files:
                    if all_files[name] != item["sha256"]:
                        raise ValueError(f"ccache hash collision with different data: {name}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if target.stat().st_size != item["size"] or hash_file(target) != item["sha256"]:
                    target.unlink(missing_ok=True)
                    raise ValueError(f"Corrupted artifact: {name}")
                all_files[name] = item["sha256"]
            if set(archive.namelist()) != permitted:
                raise ValueError("Undeclared file in cache artifact")
        print(f"Validated shard {index + 1}/{count}: {len(expected[index])} Ninja objects", flush=True)
    wanted = set().union(*map(set, expected))
    if all_objects != wanted or not wanted:
        raise ValueError("Incomplete shard coverage")
    print(f"Validated {len(wanted)} unique compile outputs and "
          f"{len(all_files)} content-verified compiler cache files", flush=True)


def check_hits(minimum, min_ratio=0.6):
    sys.path.insert(0, str(Path(__file__).parent))
    from build_telemetry import cache_stats
    stats = cache_stats()
    hits = stats.get("direct_cache_hit", 0) + stats.get("preprocessed_cache_hit", 0)
    misses = stats.get("cache_miss", 0)
    print(f"Final compilation: {hits} cache hits, {misses} cache misses", flush=True)
    ratio = hits / (hits + misses) if hits + misses else 0
    print(f"Compiler-cache hit ratio: {ratio:.1%} (required {min_ratio:.1%})", flush=True)
    if hits < minimum or ratio < min_ratio:
        raise ValueError("Shard cache reuse below safe minimum; refusing cold CUDA rebuild")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--objects", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    r = sub.add_parser("restore")
    r.add_argument("--input", type=Path, required=True)
    r.add_argument("--dest", type=Path, required=True)
    r.add_argument("--build", type=Path, required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--count", type=int, required=True)
    q = sub.add_parser("check-hits")
    q.add_argument("--minimum", type=int, default=40)
    q.add_argument("--min-ratio", type=float, default=0.6)
    args = parser.parse_args()
    if args.action == "pack":
        pack(args.source, args.objects, args.output, args.revision, args.shard, args.count)
    elif args.action == "restore":
        restore(args.input, args.dest, args.build, args.revision, args.count)
    else:
        check_hits(args.minimum, args.min_ratio)


if __name__ == "__main__":
    main()
