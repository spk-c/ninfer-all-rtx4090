"""Collect auditable SM89 paired evidence from an existing 54-case plan.

Unlike the plan generator this executes binaries. No test is counted when the GPU,
commit identity, input lengths or metrics are unqualified. Output is append-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Sequence

from tools.bench.plan_rtx4090_sm89 import (
    BASE_SHA, sha256_file, build_cases, binary_fingerprint,
)
from tools.bench.run_interleaved_ab import preflight

ANSI = re.compile(r"\x1b\[[0-9;]*m")
RATE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+tok/s", re.I)
NUMBER = re.compile(r"^(\d+)")
FIELDS = (
    "case", "family", "rep", "arm", "status", "error", "exit_code", "elapsed_s",
    "prompt_tokens", "generated_tokens", "prefill_tok_s", "decode_tok_s",
    "acceptance_pct", "tokens_per_round", "workspace_peak",
    "gpu_initial_used_mib", "gpu_peak_used_mib", "gpu_peak_util_pct",
    "gpu_peak_temp_c", "gpu_peak_power_w",
    "gpu_min_sm_clock_mhz", "gpu_max_sm_clock_mhz",
    "gpu_min_mem_clock_mhz", "gpu_max_mem_clock_mhz",
    "answer_sha256", "gold_answer_pass",
    "kv_dtype_reported", "max_context_reported", "finish_reason",
)


def extract_summary(stderr: str) -> dict[str, str]:
    """NInfer prints one stderr 'summary LABEL   VALUE' line per counter."""
    summary = {}
    labels = ("dflash2 acceptance length", "dflash2 acceptance rate",
              "gpu workspace peak", "kv cache dtype", "max context",
              "finish reason", "prompt tokens", "generated tokens",
              "prefill speed", "decode speed")
    for line in ANSI.sub("", stderr).splitlines():
        for label in labels:
            match = re.match(r"^\s*summary\s+" + re.escape(label)
                             + r"\s+(.+?)\s*$", line)
            if match:
                summary[label] = match.group(1).strip()
                break
    return summary


def parse_metrics(stderr: str) -> dict:
    raw = extract_summary(stderr)

    def rate(label: str):
        found = RATE.search(raw.get(label, ""))
        return float(found.group(1)) if found else None

    def integer(label: str):
        found = NUMBER.search(raw.get(label, ""))
        return int(found.group(1)) if found else None

    accepted = raw.get("dflash2 acceptance rate", "")
    length = raw.get("dflash2 acceptance length", "")
    p = re.search(r"(\d+(?:\.\d+)?)\s*%", accepted)
    t = re.search(r"(\d+(?:\.\d+)?)\s*tok/round", length)
    return {
        "prompt_tokens": integer("prompt tokens"),
        "generated_tokens": integer("generated tokens"),
        "prefill_tok_s": rate("prefill speed"),
        "decode_tok_s": rate("decode speed"),
        "acceptance_pct": float(p.group(1)) if p else None,
        "tokens_per_round": float(t.group(1)) if t else None,
        "workspace_peak": raw.get("gpu workspace peak"),
        "kv_dtype_reported": raw.get("kv cache dtype"),
        "max_context_reported": integer("max context"),
        "finish_reason": raw.get("finish reason"),
    }


SMI_FIELDS = (
    "index,name,driver_version,memory.total,memory.used,utilization.gpu,"
    "temperature.gpu,compute_cap,power.draw,clocks.sm,clocks.mem"
)


def gpu_snapshot(device: int) -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + SMI_FIELDS,
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi unavailable; no qualified GPU evidence") from exc
    for line in result.stdout.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != 11:
            continue
        if int(cols[0]) == device:
            def number(i: int) -> float:
                return float(cols[i])

            def optional_number(i: int) -> float | None:
                try:
                    value = float(cols[i])
                    return value if math.isfinite(value) else None
                except ValueError:
                    return None

            return {
                "gpu_index": device, "gpu_name": cols[1], "driver": cols[2],
                "total_mib": number(3), "used_mib": number(4),
                "util_pct": number(5), "temp_c": number(6),
                "compute_cap": cols[7],
                "power_w": optional_number(8),
                "sm_clock_mhz": optional_number(9),
                "mem_clock_mhz": optional_number(10),
            }
    raise RuntimeError(f"nvidia-smi device index {device} not found")


def commit_sha(repo: Path) -> str:
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True, timeout=15)
    return result.stdout.strip().lower()


def source_is_clean(root: Path) -> bool:
    """Only clean checked-out source is attributable to a pinned commit."""
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    return not result.stdout.strip()


def rotate_cases(cases: list[dict], repetition: int) -> list[dict]:
    """Counterbalance ascending/descending workload order each measured round."""
    if not cases:
        return []
    offset = (repetition - 1) % len(cases)
    shifted = cases[offset:] + cases[:offset]
    return list(reversed(shifted)) if repetition % 2 == 0 else shifted


def command_for_case(case: dict, arm: str, manifest: dict, binaries: dict) -> list[str]:
    return [
        str(binaries[arm]), manifest["model"],
        "--messages", str(Path(manifest["fixtures_dir"]) / case["fixture"]),
        "--device", "0", "--seed", "4090", "--greedy", "--no-thinking",
        "--gdn-state-fp16", "--ngram-draft-tokens", "0",
        "--max-context", str(case["max_context"]), "--max-new", str(case["max_new"]),
        *case["flags"],
    ]


def verify_binary_fingerprints(manifest: dict) -> dict:
    expected = manifest.get("binary_fingerprints")
    if not isinstance(expected, dict) or set(expected) != {"A", "B"}:
        raise ValueError("strict plan is missing both executable/runtime SHA-256 fingerprints")
    paths = {"A": Path(manifest["master_executable"]),
             "B": Path(manifest["candidate_executable"])}
    current = {arm: binary_fingerprint(path) for arm, path in paths.items()}
    for arm in ("A", "B"):
        if current[arm] != expected[arm]:
            raise ValueError(
                f"{arm} executable or adjacent runtime DLLs changed after plan generation; "
                "rebuild, rerun --strict plan and repeat all measurements")
    return current


def validate_plan(directory: Path, source_a: Path, source_b: Path,
                  max_idle_used_mib: float, gpu: dict, gold: dict) -> tuple[dict, list, dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    cases = json.loads((directory / "cases.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "ninfer_rtx4090_sm89_ab_plan_v2":
        raise ValueError("strict binary fingerprint schema v2 required; regenerate plan")
    if manifest.get("baseline_commit_sha") != BASE_SHA:
        raise ValueError("baseline is not the required fixed master SHA")
    if commit_sha(source_a) != BASE_SHA:
        raise ValueError("baseline source tree is not the fixed master SHA")
    if commit_sha(source_b) != manifest.get("candidate_commit_sha"):
        raise ValueError("candidate source tree HEAD differs from plan SHA; regenerate plan")
    if not source_is_clean(source_a) or not source_is_clean(source_b):
        raise ValueError("uncommitted tracked source changes; use clean build checkouts")
    if not ("RTX 4090" in gpu["gpu_name"] and gpu["gpu_index"] == 0):
        raise ValueError("GPU 0 must be the dedicated GeForce RTX 4090")
    if gpu.get("compute_cap") != "8.9":
        raise ValueError("GPU compute capability must be Ada sm_89 (8.9)")
    if gpu["used_mib"] > max_idle_used_mib or gpu["util_pct"] > 10.0:
        raise ValueError("GPU not idle or VRAM already occupied; stop other workloads first")
    if not isinstance(gold, dict):
        raise ValueError("gold answers must be a JSON object")
    if len(cases) != 54 or len({c["name"] for c in cases}) != 54:
        raise ValueError("unexpected matrix; exactly 54 distinct cases required")
    if not manifest.get("inputs_verified_at_plan"):
        raise ValueError("plan must be generated with --strict and frozen artifact hashes")
    if manifest.get("model_sha256") is None:
        raise ValueError("plan lacks model SHA256; recreate with the artifact present")
    if sha256_file(Path(manifest["model"])) != manifest["model_sha256"]:
        raise ValueError("model changed after plan generation")
    fixtures = Path(manifest["fixtures_dir"])
    for name, digest in manifest.get("fixture_sha256", {}).items():
        if sha256_file(fixtures / name) != digest:
            raise ValueError("fixture was modified after plan generation: " + name)
    if len(manifest.get("fixture_sha256", {})) != 18:
        raise ValueError("plan has fewer than 18 immutable coding/context fixtures")
    if (sha256_file(fixtures / "gold-answers.json") !=
            manifest.get("gold_answers_sha256")):
        raise ValueError("gold answers changed after plan generation")
    a, b = (Path(manifest[k]) for k in ("master_executable", "candidate_executable"))
    verify_binary_fingerprints(manifest)
    preflight(a, b, allow_separate_dirs=True)
    # Gold answers must be independently curated; only a verified expected token
    # can qualify long-context retrieval. The benchmark never guesses one.
    for c in cases:
        if c["family"] == "kv":
            fixture = c["fixture"]
            entry = gold.get(fixture)
            if not isinstance(entry, dict) or not isinstance(entry.get("expected"), str):
                raise ValueError("missing independently authored gold answer: " + fixture)
            if not entry["expected"].strip():
                raise ValueError("blank gold answer: " + fixture)
    expected = {c.name: c for c in build_cases()}
    if set(expected) != {c["name"] for c in cases}:
        raise ValueError("case names differ from checked-in test matrix")
    for c in cases:
        e = expected[c["name"]]
        if (c["family"], c["fixture"], c["max_context"], c["flags"],
            c["minimum_prompt_tokens"], c["max_new"], c["metric"]) != (
            e.family, e.fixture, e.max_context, list(e.flags),
            e.minimum_prompt_tokens, e.max_new, e.metric):
            raise ValueError("case configuration differs from reviewed matrix")
        fixture_path = Path(manifest["fixtures_dir"]) / c["fixture"]
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("expected nonempty JSON chat array: " + str(fixture_path))
        if any(not isinstance(msg, dict) or "role" not in msg or "content" not in msg
               for msg in payload):
            raise ValueError("invalid chat JSON fixture: " + str(fixture_path))
    return manifest, cases, {"A": a, "B": b}


def call_one(argv: list[str], timeout_s: int, device: int,
             gpu_probe=gpu_snapshot) -> tuple[int, str, str, float, dict, str]:
    """Sample whole-card memory while subprocess runs; NOT per-process CUDA allocation."""
    snapshots: list[dict] = []
    stopped = threading.Event()

    def poll() -> None:
        while not stopped.is_set():
            try:
                snapshots.append(gpu_probe(device))
            except (RuntimeError, ValueError):
                pass
            stopped.wait(0.5)

    start = time.monotonic()
    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        run = subprocess.run(argv, capture_output=True, text=True,
                             errors="replace", timeout=timeout_s)
        code, stdout, stderr, error = run.returncode, run.stdout, run.stderr, ""
    except subprocess.TimeoutExpired as exc:
        code, stdout, stderr, error = -1, str(exc.stdout or ""), str(exc.stderr or ""), "timeout"
    except OSError as exc:
        code, stdout, stderr, error = -2, "", str(exc), "launch error"
    finally:
        stopped.set()
        thread.join(timeout=2)
    elapsed = round(time.monotonic() - start, 3)
    initial = snapshots[0] if snapshots else None

    def aggregate(key: str, operation=max) -> float | None:
        values = [x.get(key) for x in snapshots]
        numeric = [v for v in values if isinstance(v, (int, float))
                   and math.isfinite(v)]
        return operation(numeric) if numeric else None

    telemetry = {
        "gpu_initial_used_mib": initial["used_mib"] if initial else None,
        "gpu_peak_used_mib": max((x["used_mib"] for x in snapshots), default=None),
        "gpu_peak_util_pct": max((x["util_pct"] for x in snapshots), default=None),
        "gpu_peak_temp_c": aggregate("temp_c"),
        "gpu_peak_power_w": aggregate("power_w"),
        "gpu_min_sm_clock_mhz": aggregate("sm_clock_mhz", min),
        "gpu_max_sm_clock_mhz": aggregate("sm_clock_mhz"),
        "gpu_min_mem_clock_mhz": aggregate("mem_clock_mhz", min),
        "gpu_max_mem_clock_mhz": aggregate("mem_clock_mhz"),
        "sampling_note": "nvidia-smi reports WHOLE GPU usage, including other processes",
    }
    return code, stdout, stderr, elapsed, telemetry, error


def canonical_gold_answer(text: str) -> str:
    """Remove presentation wrappers, never permit substring-only retrieval matches."""
    result = text.strip()
    if result.startswith("```") and result.endswith("```"):
        lines = result.splitlines()
        if len(lines) >= 3:
            result = "\n".join(lines[1:-1]).strip()
    if len(result) >= 2 and result[0] == result[-1] and result[0] in ("\"", "'"):
        result = result[1:-1].strip()
    return result.casefold()


def classify(case: dict, metrics: dict, stdout: str, code: int,
             error: str, gold: dict, stderr: str = "") -> tuple[str, str, bool | None]:
    if code != 0:
        diagnostic = (error + "\n" + stderr).lower()
        if re.search(r"out.of.memory|cudamalloc|cudaerrormemoryallocation|insufficient memory", diagnostic):
            return "OOM", "GPU memory allocation failure", None
        if "unsupported" in diagnostic and "context" in diagnostic:
            return "unsupported", "context unsupported by artifact/runtime", None
        return ("timeout" if error == "timeout" else "failed", error or f"exit {code}", None)
    metric_key = "prefill_tok_s" if case["family"] == "prefill" else "decode_tok_s"
    speed = metrics.get(metric_key)
    if speed is None or not math.isfinite(speed) or speed <= 0:
        return "failed", "missing or invalid measured throughput", None
    tokens = metrics.get("prompt_tokens")
    if tokens is None:
        return "failed", "CLI omitted actual prompt token count", None
    if metrics.get("max_context_reported") != case["max_context"]:
        return "failed", "CLI context differs from requested context", None
    flags = list(case["flags"])
    expected_kv = flags[flags.index("--kv-dtype") + 1]
    if metrics.get("kv_dtype_reported") != expected_kv:
        return "failed", "CLI KV dtype differs from requested configuration", None
    if (case["family"] == "kv" and
            (tokens < case["minimum_prompt_tokens"] or
             tokens + case["max_new"] > case["max_context"])):
        return "insufficient_context", "actual prompt is too short or lacks decode headroom", None
    if not metrics.get("generated_tokens"):
        return "failed", "zero generated output tokens", None
    if case["family"] == "dflash2" and metrics.get("acceptance_pct") is None:
        return "failed", "DFlash2 acceptance unavailable", None
    if case["family"] == "kv":
        answer = gold[case["fixture"]]["expected"]
        ok = canonical_gold_answer(stdout) == answer.strip().casefold()
        return ("passed", "", True) if ok else (
            "quality_regression", "generated answer differs from exact gold", False)
    return "passed", "", None


def run_campaign(plan_dir: Path, out: Path, source_a: Path, source_b: Path,
                 gold_path: Path, reps: int, max_cases: int, idle_limit: float,
                 timeout: int, gpu_probe=gpu_snapshot, executor=call_one) -> int:
    if out.exists():
        raise ValueError("output directory already exists; never overwrite raw evidence")
    if reps < 5:
        raise ValueError("a qualified run requires at least five AB/BA pairs")
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    initial = gpu_probe(0)
    manifest, cases, binaries = validate_plan(
        plan_dir, source_a, source_b, idle_limit, initial, gold)
    canonical_gold = Path(manifest["fixtures_dir"]) / "gold-answers.json"
    if gold_path.resolve() != canonical_gold.resolve():
        raise ValueError("gold-answer file must be the exact hashed fixture manifest input")
    if sha256_file(gold_path) != manifest["gold_answers_sha256"]:
        raise ValueError("supplied gold-answer file changed after plan generation")
    if reps != manifest["repetitions"]:
        raise ValueError("repetitions must equal the plan; regenerate instead of changing runs")
    selected = cases[:max_cases] if max_cases else cases
    out.mkdir(parents=True)
    raw = out / "raw"
    raw.mkdir()
    metadata = {
        "plan_manifest": manifest, "gpu_start": initial,
        "exe_sha256": {arm: sha256_file(path) for arm, path in binaries.items()},
        "binary_fingerprints": manifest["binary_fingerprints"],
        "gold_answers_sha256": sha256_file(gold_path),
        "status": "IN_PROGRESS", "selected_cases": len(selected),
        "warmup_count": 0, "warmup_failures": 0,
        "full_matrix": len(selected) == len(cases),
        "note": "CLI process startup included in wall time. CLI prefill is NOT HTTP TTFT.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    all_rows = []
    with (out / "samples.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        csv_file.flush()
        warmup_dir = raw / "warmup"
        warmup_dir.mkdir()
        for rep in range(1, reps + 1):
            for case in rotate_cases(selected, rep):
                order = ("A", "B") if rep % 2 else ("B", "A")
                if rep == 1:
                    # Warm-up is never included in the measured CSV; failure
                    # blocks campaign qualification, but raw evidence survives.
                    for arm in ("A", "B"):
                        warm_argv = command_for_case(case, arm, manifest, binaries)
                        wc, wo, we, wt, wgpu, werror = executor(warm_argv, timeout, 0)
                        wstatus, wreason, _ = classify(
                            case, parse_metrics(we), wo, wc, werror, gold, we)
                        if wstatus == "passed" and wgpu.get("gpu_peak_used_mib") is None:
                            wstatus, wreason = "failed", "missing warm-up GPU telemetry"
                        prefix = f"{case['name']}-{arm}"
                        (warmup_dir / (prefix + ".stdout.txt")).write_text(
                            wo, encoding="utf-8")
                        (warmup_dir / (prefix + ".stderr.txt")).write_text(
                            we, encoding="utf-8")
                        metadata["warmup_count"] += 1
                        if wstatus != "passed":
                            metadata["warmup_failures"] += 1
                            print(f"WARMUP {prefix}: {wstatus}: {wreason}", flush=True)
                for arm in order:
                    argv = command_for_case(case, arm, manifest, binaries)
                    code, stdout, stderr, elapsed, telemetry, error = executor(
                        argv, timeout, 0)
                    metrics = parse_metrics(stderr)
                    status, reason, gold_pass = classify(
                        case, metrics, stdout, code, error, gold, stderr)
                    if status == "passed" and telemetry["gpu_peak_used_mib"] is None:
                        status, reason = "failed", "missing sampled GPU/VRAM evidence"
                    stem = f"{case['name']}-rep{rep:02}-{arm}"
                    (raw / (stem + ".stdout.txt")).write_text(stdout, encoding="utf-8")
                    (raw / (stem + ".stderr.txt")).write_text(stderr, encoding="utf-8")
                    row = {key: None for key in FIELDS}
                    row.update({
                        "case": case["name"], "family": case["family"],
                        "rep": rep, "arm": arm, "status": status,
                        "error": reason, "exit_code": code, "elapsed_s": elapsed,
                        "answer_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
                        "gold_answer_pass": gold_pass,
                    })
                    row.update({k: metrics.get(k) for k in (
                        "prompt_tokens", "generated_tokens", "prefill_tok_s", "decode_tok_s",
                        "acceptance_pct", "tokens_per_round", "workspace_peak",
                        "kv_dtype_reported", "max_context_reported", "finish_reason")})
                    row.update({k: telemetry.get(k) for k in (
                        "gpu_initial_used_mib", "gpu_peak_used_mib",
                        "gpu_peak_util_pct", "gpu_peak_temp_c",
                        "gpu_peak_power_w", "gpu_min_sm_clock_mhz",
                        "gpu_max_sm_clock_mhz", "gpu_min_mem_clock_mhz",
                        "gpu_max_mem_clock_mhz")})
                    writer.writerow(row)
                    csv_file.flush()
                    all_rows.append(row)
                    print(f"{stem}: {status}; {row['prefill_tok_s']} prefill / "
                          f"{row['decode_tok_s']} decode tok/s", flush=True)
    # Detect replaced EXE/DLL files during a long campaign. Preserve all raw
    # evidence but disqualify the result if the runtime's bytes changed.
    try:
        verify_binary_fingerprints(manifest)
        binary_integrity = "UNCHANGED"
    except ValueError as exc:
        binary_integrity = str(exc)
    metadata["binary_integrity"] = binary_integrity
    metadata["status"] = (
        "COMPLETE" if binary_integrity == "UNCHANGED"
        and len(selected) == len(cases)
        and metadata["warmup_failures"] == 0
        and metadata["warmup_count"] == len(selected) * 2 else
        "FAILED_WARMUP" if metadata["warmup_failures"] else "PARTIAL"
    )
    metadata["successful_samples"] = sum(r["status"] == "passed" for r in all_rows)
    metadata["total_samples"] = len(all_rows)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0 if metadata["status"] == "COMPLETE" and all(
        r["status"] == "passed" for r in all_rows) else 2


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--master-source", type=Path, required=True)
    p.add_argument("--candidate-source", type=Path, required=True)
    p.add_argument("--gold-answers", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--max-cases", type=int, default=0,
                   help="smoke subset (always PARTIAL, never qualifies release)")
    p.add_argument("--max-idle-vram-mib", type=float, default=2048.0)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args(argv)
    if args.max_cases < 0 or args.timeout < 1:
        p.error("invalid max-cases or timeout")
    try:
        return run_campaign(
            args.plan_dir, args.out, args.master_source, args.candidate_source,
            args.gold_answers, args.repetitions, args.max_cases,
            args.max_idle_vram_mib, args.timeout)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
