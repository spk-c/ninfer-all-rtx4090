"""Opt-in, fail-closed Nsight Systems/Compute capture for fixed SM89 A/B builds.

Dry planning (default) does not run a GPU and must never be called a profile.
--execute requires a dedicated, idle RTX 4090, two pinned source trees and
separately built benchmark binaries. It saves raw tool output and .nsys/.ncu
artifacts without modifying any production kernel or release configuration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Sequence

from tools.bench.plan_rtx4090_sm89 import BASE_SHA, nsight_commands, sha256_file
from tools.bench.run_rtx4090_sm89_evidence import (
    commit_sha, gpu_snapshot, source_is_clean,
)


def profile_plan(master: Path, candidate: Path, output: Path,
                 mode: str = "both") -> list[dict]:
    if mode not in ("nsys", "ncu", "both"):
        raise ValueError("mode must be nsys, ncu or both")
    if master.resolve() == candidate.resolve():
        raise ValueError("master/candidate benchmark binaries must be in separate build trees")
    runs = []
    for arm, bench in (("A", master), ("B", candidate)):
        entries = nsight_commands(bench, output / arm)
        for entry in entries:
            for tool in ("nsys", "ncu"):
                if mode != "both" and tool != mode:
                    continue
                runs.append({
                    "arm": arm, "kernel_family": entry["name"],
                    "tool": tool, "argv": entry[tool],
                    "required_metrics": entry["required_metrics"],
                    "status": "NOT_RUN",
                })
    return runs


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, errors="replace",
                          timeout=timeout)


def validate_machine(manifest: dict, a: Path, b: Path,
                     source_a: Path, source_b: Path,
                     idle_limit: float, probe=gpu_snapshot,
                     runner=_run) -> dict:
    if manifest.get("schema") != "ninfer_rtx4090_sm89_ab_plan_v2":
        raise ValueError("strict plan schema v2 required")
    if manifest.get("baseline_commit_sha") != BASE_SHA:
        raise ValueError("plan has the wrong master baseline")
    if commit_sha(source_a) != BASE_SHA:
        raise ValueError("master source checkout must match frozen baseline")
    if commit_sha(source_b) != manifest.get("candidate_commit_sha"):
        raise ValueError("candidate source checkout does not match plan commit")
    if not source_is_clean(source_a) or not source_is_clean(source_b):
        raise ValueError("tracked source modifications present")
    for path in (a, b):
        if not path.is_file():
            raise ValueError("missing independently built bench binary: " + str(path))
        if path.suffix.lower() != ".exe" or path.name.lower() != "ninfer_benches.exe":
            raise ValueError("expected Windows ninfer_benches.exe for native SM89")
        if path.parent.name.lower() not in ("bench", "release"):
            raise ValueError("bench executable should remain in its own build/bench tree")
    gpu = probe(0)
    if gpu.get("gpu_index") != 0 or gpu.get("compute_cap") != "8.9" or "RTX 4090" not in gpu.get("gpu_name", ""):
        raise ValueError("an actual GPU 0 RTX 4090 / Ada SM89 is required")
    if gpu.get("used_mib", 999999) > idle_limit or gpu.get("util_pct", 999) > 10:
        raise ValueError("GPU is not idle; do not disrupt existing model server")
    names = {
        "ninfer_q4_linear_swiglu_schedule_bench",
        "ninfer_q5_linear_add_schedule_bench",
        "ninfer_a8_prefill_schedule_bench",
    }
    for bench in (a, b):
        test = runner([str(bench), "--list"], 40)
        if test.returncode != 0 or not all(name in test.stdout + test.stderr for name in names):
            raise ValueError("bench binary failed --list or expected SM89 programs are missing: " + str(bench))
    return gpu


def execute_plan(runs: list[dict], timeout: int,
                 runner=_run) -> list[dict]:
    for item in runs:
        # -o is the export path prefix; create its parent ahead of time.
        argv = item["argv"]
        if "-o" not in argv:
            raise ValueError("Nsight command missing export location")
        prefix = Path(argv[argv.index("-o") + 1])
        prefix.parent.mkdir(parents=True, exist_ok=True)
        name = f"{item['arm']}-{item['kernel_family']}-{item['tool']}"
        try:
            proc = runner(argv, timeout)
            code, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            code, stdout, stderr = -1, "", str(exc)
        (prefix.parent / (name + ".stdout.log")).write_text(stdout, encoding="utf-8")
        (prefix.parent / (name + ".stderr.log")).write_text(stderr, encoding="utf-8")
        expected_ext = ".nsys-rep" if item["tool"] == "nsys" else ".ncu-rep"
        artifact = Path(str(prefix) + expected_ext)
        valid_report = code == 0 and artifact.is_file() and artifact.stat().st_size > 0
        item["exit_code"] = code
        item["status"] = (
            "captured" if valid_report else
            "MISSING_REPORT" if code == 0 else "FAILED"
        )
        item["report_path"] = str(artifact) if valid_report else None
        item["report_sha256"] = sha256_file(artifact) if valid_report else None
        item["error"] = (
            stderr[-1000:] if code != 0 else
            "Profiler returned success but no nonempty Nsight report was found"
            if not valid_report else None
        )
    return runs


def capture(out: Path, manifest_path: Path, a: Path, b: Path,
            source_a: Path, source_b: Path, mode: str, execute: bool,
            idle_limit: float, timeout: int, probe=gpu_snapshot,
            runner=_run) -> dict:
    if out.exists():
        raise ValueError("output already exists; never overwrite profiler evidence")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = profile_plan(a, b, out, mode)
    meta = {
        "schema": "ninfer_rtx4090_nsight_capture_v1",
        "fixed_master": BASE_SHA,
        "candidate": manifest.get("candidate_commit_sha"),
        "baseline_bench": str(a), "candidate_bench": str(b),
        "mode": mode, "status": "NOT_RUN",
        "tool_versions": {},
        "gpu_verified": False,
        "profiles": runs,
        "note": "A dry plan or a zero-exit Nsight tool does not establish kernel gain",
    }
    if execute:
        gpu = validate_machine(manifest, a, b, source_a, source_b,
                               idle_limit, probe, runner)
        versions = {}
        for tool in (("nsys", "ncu") if mode == "both" else (mode,)):
            result = runner([tool, "--version"], 30)
            if result.returncode != 0:
                raise ValueError(tool + " unavailable: " + (result.stderr or ""))
            versions[tool] = (result.stdout + result.stderr).strip()
        meta["gpu_verified"] = True
        meta["gpu"] = gpu
        meta["tool_versions"] = versions
        meta["bench_sha256"] = {"A": sha256_file(a), "B": sha256_file(b)}
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if execute:
        runs = execute_plan(runs, timeout, runner)
        meta["profiles"] = runs
        meta["status"] = ("CAPTURED_ALL" if all(x["status"] == "captured" for x in runs)
                          else "PARTIAL_OR_FAILED")
        # One changed benchmark binary invalidates provenance of all traces.
        if sha256_file(a) != meta["bench_sha256"]["A"] or sha256_file(b) != meta["bench_sha256"]["B"]:
            meta["status"] = "INVALID_BINARY_CHANGED"
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-manifest", required=True, type=Path)
    p.add_argument("--master-bench", required=True, type=Path)
    p.add_argument("--candidate-bench", required=True, type=Path)
    p.add_argument("--master-source", required=True, type=Path)
    p.add_argument("--candidate-source", required=True, type=Path)
    p.add_argument("--mode", choices=("nsys", "ncu", "both"), default="both")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--execute", action="store_true",
                   help="ONLY when independent pinned GPU builds and an idle RTX 4090 exist")
    p.add_argument("--max-idle-vram-mib", type=float, default=2048.0)
    p.add_argument("--timeout", type=int, default=7200)
    args = p.parse_args(argv)
    if args.timeout <= 0 or args.max_idle_vram_mib < 0:
        p.error("invalid timeout or GPU idle limit")
    try:
        report = capture(
            args.out, args.plan_manifest, args.master_bench, args.candidate_bench,
            args.master_source, args.candidate_source, args.mode, args.execute,
            args.max_idle_vram_mib, args.timeout)
    except (OSError, ValueError, subprocess.SubprocessError) as err:
        p.error(str(err))
    print(json.dumps({"status": report["status"], "profiles": len(report["profiles"]),
                      "gpu_verified": report["gpu_verified"]}, indent=2))
    return 0 if report["status"] in ("NOT_RUN", "CAPTURED_ALL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
