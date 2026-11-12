"""Emit the pinned RTX 4090 SM89 A/B plan (never run GPU tests here)."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

BASE_SHA = "f879a2db06c15133260551a104feaf03886cc620"
DRAFT_K = (3, 5, 7, 9)
PREFILL_CHUNKS = (1024, 2048, 4096)
CONTEXTS = (32768, 65536, 131072)
KV_TYPES = ("rk4v4", "rk8v4")
SEEDS = tuple(range(1, 6))
PROMPTS = ("coding-short", "coding-long", "coding-edit")
DECODE_METRIC = r"decode speed\s+([\d.]+) tok/s"
PREFILL_METRIC = r"prefill speed\s+([\d.]+) tok/s"


@dataclass(frozen=True)
class Case:
    family: str
    name: str
    fixture: str
    metric: str
    max_context: int
    flags: tuple[str, ...]
    max_new: int = 256
    minimum_prompt_tokens: int = 0


def build_cases() -> list[Case]:
    rows: list[Case] = []
    for prompt in PROMPTS:
        for k in DRAFT_K:
            rows.append(Case(
                "dflash2", f"dflash2-{prompt}-k{k}", f"{prompt}.json",
                DECODE_METRIC, 32768,
                ("--kv-dtype", "rk4v4", "--prefill-chunk", "1024",
                 "--spec", "dflash2", "--draft-tokens", str(k))))
    for prompt in ("coding-short", "coding-long"):
        for chunk in PREFILL_CHUNKS:
            for cublas in (False, True):
                rows.append(Case(
                    "prefill", f"prefill-{prompt}-{chunk}-cublas-{int(cublas)}",
                    f"{prompt}.json", PREFILL_METRIC, 32768,
                    ("--kv-dtype", "rk4v4", "--spec", "none",
                     "--prefill-chunk", str(chunk),
                     "--prefill-cublas" if cublas else "--no-prefill-cublas"),
                    max_new=128))
    for ctx in CONTEXTS:
        for kv in KV_TYPES:
            for seed in SEEDS:
                rows.append(Case(
                    "kv", f"kv-{ctx}-{kv}-seed{seed}",
                    f"kv-{ctx}-seed{seed}.json", DECODE_METRIC, ctx,
                    ("--kv-dtype", kv, "--spec", "none",
                     "--prefill-chunk", "1024"),
                    max_new=128, minimum_prompt_tokens=int(ctx * 0.90)))
    assert len(rows) == len({case.name for case in rows})
    return rows


def quote(arg: str) -> str:
    """Quotation compatible with run_interleaved_ab.py's Windows parser."""
    if '"' in arg or "\n" in arg or "\r" in arg:
        raise ValueError("cannot safely round-trip quotes/newlines in command")
    return '"' + arg + '"' if not arg or any(c.isspace() for c in arg) else arg


def command_tokens(case: Case, model: Path, fixtures: Path) -> list[str]:
    common = [
        "{exe}", str(model), "--messages", str(fixtures / case.fixture),
        "--device", "0", "--greedy", "--no-thinking", "--gdn-state-fp16",
        "--ngram-draft-tokens", "0", "--max-context", str(case.max_context),
        "--max-new", str(case.max_new),
    ]
    return common + list(case.flags)


def ab_command(case: Case, master: Path, candidate: Path, model: Path,
               fixtures: Path, repetitions: int,
               runner: Path = Path("tools/bench/run_interleaved_ab.py")) -> list[str]:
    spec = case.name + ":" + " ".join(
        quote(value) for value in command_tokens(case, model, fixtures)
    )
    return ["python", str(runner), "--arm-a", str(master),
            "--arm-b", str(candidate), "--allow-separate-dirs",
            "--reps", str(repetitions), "--timeout", "1800",
            "--metric-regex", case.metric, "--config", spec]


def nsight_commands(bench: Path, out: Path) -> list[dict]:
    settings = (
        ("q4-small-t", "ninfer_q4_linear_swiglu_schedule_bench",
         ("--tokens", "1,2,4,8,16,32", "--repeat", "9", "--warmup", "3", "--spread")),
        ("q5-small-t", "ninfer_q5_linear_add_schedule_bench",
         ("--tokens", "1,2,4,8,16,32", "--repeat", "9", "--warmup", "3", "--spread")),
        ("w4a8", "ninfer_a8_prefill_schedule_bench",
         ("--tokens", "128,512,1024", "--repeat", "9", "--warmup", "3", "--panel")),
    )
    return [{
        "name": name, "target": [str(bench), target, *args],
        "nsys": ["nsys", "profile", "--trace=cuda,nvtx",
                 "--cuda-graph-trace=node", "--force-overwrite=true",
                 "-o", str(out / f"{name}-timeline"),
                 str(bench), target, *args],
        "ncu": ["ncu", "--target-processes", "all", "--set", "full",
                "-o", str(out / f"{name}-ncu"), str(bench), target, *args],
        "required_metrics": [
            "registers_per_thread", "achieved_and_theoretical_occupancy",
            "shared_memory", "local_memory_spills", "dram_bandwidth",
            "L2_traffic", "tensor_core_utilization", "kernel_duration",
        ],
        "note": "Filter the hottest real SM89 kernel/shape after inspecting nsys; "
                "check available ncu metric names on installed version.",
    } for name, target, args in settings]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_plan(master: Path, candidate: Path, model: Path, fixtures: Path,
                candidate_sha: str, repetitions: int, out: Path,
                strict: bool = False) -> dict:
    if not re.fullmatch(r"[a-fA-F0-9]{40}", candidate_sha):
        raise ValueError("candidate SHA must be the full 40-character commit hash")
    if candidate_sha.lower() == BASE_SHA:
        raise ValueError("cannot benchmark master against the same commit")
    if not 5 <= repetitions <= 30:
        raise ValueError("at least five paired A/B repetitions required")
    if master == candidate:
        raise ValueError("baseline and candidate paths must differ")
    if out.exists():
        raise ValueError("output exists; refusing to overwrite plan")
    rows = build_cases()
    fixture_names = sorted({case.fixture for case in rows})
    if strict:
        for path in (master, candidate, model):
            if not path.is_file():
                raise ValueError(f"required model or executable missing: {path}")
        if master.resolve() == candidate.resolve():
            raise ValueError("baseline and candidate resolve to the same binary")
        if master.parent.name.casefold() != "apps" or candidate.parent.name.casefold() != "apps":
            raise ValueError("keep each executable in its own apps build tree")
        missing = [str(fixtures / name) for name in fixture_names
                   if not (fixtures / name).is_file()]
        if missing:
            raise ValueError("missing coding/retrieval fixtures: " + ", ".join(missing))
    out.mkdir(parents=True)
    model_hash = sha256_file(model) if model.is_file() else None
    manifest = {
        "schema": "ninfer_rtx4090_sm89_ab_plan_v1",
        "baseline_commit_sha": BASE_SHA,
        "candidate_commit_sha": candidate_sha.lower(),
        "master_executable": str(master), "candidate_executable": str(candidate),
        "model": str(model), "model_sha256": model_hash,
        "model_hash_verified": model_hash is not None,
        "fixtures_dir": str(fixtures), "repetitions": repetitions,
        "gpu": "RTX 4090 / sm_89 required; NOT verified by plan",
        "toolchain": "native Windows CUDA 13.4.3 required; NOT verified by plan",
        "evidence_status": "NOT_RUN", "case_count": len(rows),
        "expected_arm_runs": len(rows) * repetitions * 2,
        "quality_validation": "PENDING: exact prompt tokens, near-limit retrieval and perplexity",
        "release_contract": "Preserve existing CUDA-13.4.3-SM89 sharded release workflow",
        "kernel_defaults": "All unqualified kernel optimizations must remain OFF",
    }
    payloads = {
        "manifest.json": manifest,
        "cases.json": [asdict(case) for case in rows],
        "fixture-requirements.json": [{
            "file": name,
            "minimum_actual_prompt_tokens": max(
                case.minimum_prompt_tokens for case in rows if case.fixture == name),
            "actual_prompt_tokens_verified": False,
            "reference_answer_required": name.startswith("kv-"),
        } for name in fixture_names],
        "ab-commands.json": [{
            "name": case.name, "family": case.family, "metric": case.metric,
            "argv": ab_command(case, master, candidate, model, fixtures, repetitions),
            "status": "NOT_RUN",
        } for case in rows],
        "nsight-commands.json": nsight_commands(
            Path("build-rtx4090/bench/ninfer_benches.exe"), out / "profiles"),
    }
    for name, data in payloads.items():
        (out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-exe", required=True, type=Path)
    parser.add_argument("--candidate-exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixtures-dir", required=True, type=Path)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    opts = parser.parse_args(argv)
    try:
        plan = create_plan(opts.master_exe, opts.candidate_exe, opts.model,
                           opts.fixtures_dir, opts.candidate_sha,
                           opts.repetitions, opts.out, opts.strict)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "case_count": plan["case_count"],
        "expected_arm_runs": plan["expected_arm_runs"],
        "evidence_status": "NOT_RUN",
        "result_directory": str(opts.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
