"""RTX 4090 real-text DFlash2, prefill, KV and copy-drafting sweep.

An audit tool, not an automatic tuning switch. Benchmark-corpus repetition can
inflate speculative acceptance; use native ninfer CLI responses instead.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
from typing import Sequence


@dataclass(frozen=True)
class Case:
    group: str
    label: str
    flags: tuple[str, ...]
    context: int = 32768
    prompt: str = "new_code"
    output_tokens: int = 192


PROMPTS = {
    "new_code": "Write a full Python implementation of a bounded thread-safe LRU cache with types, edge cases, tests and a concise explanation.",
    "reasoning": "Explain paged versus contiguous KV for concurrent LLM requests. Analyze fragmentation, context reuse and latency. Recommend an implementation.",
    "edit_code": "Refactor this Python module for typing. Return the full revised module and tests.\n\n" + "\n".join(
        f"def calculate_{n}(values, factor=1):\n    return round(sum(float(v) for v in values) * factor, 3)\n" for n in range(20)),
    "long_prefill": "Review this architecture description for concurrency and memory bugs:\n" + "\n".join(
        f"Section {n}: Request ownership covers paged KV, GDN snapshots, device events, resource reservation, CPU worker cancellation, and context eviction." for n in range(170)),
}


def cases(groups: Sequence[str]) -> list[Case]:
    result: list[Case] = []
    if "draft" in groups:
        for prompt in ("new_code", "reasoning", "edit_code"):
            for k in (3, 5, 7, 9):
                result.append(Case("draft", f"{prompt}-k{k}", (
                    "--spec", "dflash2", "--draft-tokens", str(k),
                    "--ngram-draft-tokens", "0"), prompt=prompt))
    if "prefill" in groups:
        for chunk, cublas in ((1024, False), (2048, False), (2048, True),
                              (4096, False), (4096, True)):
            flags = ("--prefill-chunk", str(chunk))
            if cublas:
                flags += ("--prefill-cublas",)
            result.append(Case("prefill", f"chunk{chunk}-cublas{int(cublas)}",
                               flags, prompt="long_prefill", output_tokens=96))
    if "kv" in groups:
        for capacity in (32768, 65536, 131072):
            for kv in ("rk4v4", "rk8v4"):
                result.append(Case("kv", f"{kv}-cap{capacity}",
                                   ("--kv-dtype", kv), context=capacity))
    if "copy" in groups:
        for ngram in (0, 15):
            result.append(Case("copy", f"ngram{ngram}", (
                "--spec", "dflash2", "--draft-tokens", "5",
                "--ngram-draft-tokens", str(ngram),
                "--ngram-min-match", "12"), prompt="edit_code"))
    if not result or len(set(x.label for x in result)) != len(result):
        raise ValueError("empty or duplicate benchmark matrix")
    return result


def parse_measurements(stderr: str) -> dict:
    def floating(pattern: str) -> float | None:
        m = re.search(pattern, stderr, re.IGNORECASE)
        if m is None:
            return None
        value = float(m.group(1))
        return value if math.isfinite(value) and value > 0 else None

    def integer(pattern: str) -> int | None:
        m = re.search(pattern, stderr, re.IGNORECASE)
        return int(m.group(1)) if m else None

    return {
        "decode_tok_s": floating(r"decode speed\s+([0-9]+(?:\.[0-9]+)?)\s+tok/s"),
        "prefill_tok_s": floating(r"prefill speed\s+([0-9]+(?:\.[0-9]+)?)\s+tok/s"),
        "generated_tokens": integer(r"generated tokens\s+(\d+)"),
        "acceptance_pct": floating(r"acceptance rate\s+([0-9]+(?:\.[0-9]+)?)\s*%"),
        "tokens_per_round": floating(r"acceptance length\s+([0-9]+(?:\.[0-9]+)?)"),
    }


def command(cli: Path, model: Path, case: Case, messages: Path, device: int) -> list[str]:
    args = [str(cli), str(model), "--messages", str(messages),
            "--device", str(device), "--max-new", str(case.output_tokens),
            "--max-context", str(case.context), "--greedy", "--no-thinking",
            "--gdn-state-fp16"]
    if "--kv-dtype" not in case.flags:
        args += ["--kv-dtype", "rk4v4"]
    if "--spec" not in case.flags:
        args += ["--ngram-draft-tokens", "0"]
    return args + list(case.flags)


def order_for_round(matrix: Sequence[Case], index: int) -> list[Case]:
    start = (index // 2) % len(matrix)
    ordered = list(matrix[start:]) + list(matrix[:start])
    if index % 2:
        ordered.reverse()
    return ordered


def summarize(samples: Sequence[dict], matrix: Sequence[Case], rounds: int) -> dict:
    result = {}
    for case in matrix:
        values = [s for s in samples if s["case"] == case.label
                  and s["status"] == "ok"]
        metric = "prefill_tok_s" if case.group == "prefill" else "decode_tok_s"
        valid = len(values) == rounds and all(
            isinstance(s.get(metric), (int, float)) and s[metric] > 0
            and isinstance(s.get("generated_tokens"), int)
            and s["generated_tokens"] >= min(32, case.output_tokens // 2)
            for s in values)
        result[case.label] = {
            "group": case.group, "valid": valid, "samples": len(values),
            "metric": metric,
            "median": statistics.median(s[metric] for s in values) if valid else None,
            "generated_token_counts": [s.get("generated_tokens") for s in values],
            "output_sha256": [s["output_sha256"] for s in values],
            "warning": "Different text or token counts are not equivalent workloads.",
        }
    return result


def sha256_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_metadata(index: int) -> dict:
    try:
        probe = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,"
                                "power.limit,temperature.gpu,clocks.gr,memory.total",
                                "--format=csv,noheader"], capture_output=True,
                               text=True, check=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("nvidia-smi unavailable; GPU not qualified") from exc
    matched = [row.strip() for row in probe.stdout.splitlines()
               if row.split(",", 1)[0].strip() == str(index)]
    if len(matched) != 1 or "RTX 4090" not in matched[0]:
        raise ValueError(f"index {index} is not a confirmed RTX 4090: {matched}")
    return {"gpu": matched[0], "warning": "CUDA and nvidia-smi indices may differ"}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cli", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--groups", nargs="+", choices=("draft", "prefill", "kv", "copy"),
                   default=["draft", "prefill", "kv", "copy"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if not 3 <= a.rounds <= 20 or a.timeout <= 0 or a.device < 0:
        p.error("rounds must be 3..20; timeout positive; device nonnegative")
    if a.out.exists():
        p.error(f"refusing to overwrite results: {a.out}")
    matrix = cases(a.groups)
    if not a.dry_run and (not a.cli.is_file() or not a.model.is_file()):
        p.error("ninfer CLI and model must both exist")
    try:
        hardware = {"qualified": False} if a.dry_run else gpu_metadata(a.device)
    except ValueError as exc:
        p.error(str(exc))
    a.out.mkdir(parents=True)
    message_files = {}
    for name, prompt in PROMPTS.items():
        path = a.out / f"prompt-{name}.json"
        path.write_text(json.dumps([{"role": "user", "content": prompt}],
                                   ensure_ascii=False) + "\n", encoding="utf-8")
        message_files[name] = path
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = None
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": sha or None, "model": str(a.model), "cli": str(a.cli),
        "model_sha256": None if a.dry_run else sha256_stream(a.model),
        "gpu": hardware, "rounds": a.rounds, "groups": a.groups,
        "cases": [asdict(x) for x in matrix],
        "limitations": [
            "dry-run provides no hardware evidence",
            "configured context capacity does not establish long-context recall",
            "draft width or KV can change output text and token counts",
            "no model, production setting, kernel path, or GPU clock is modified",
        ],
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    samples: list[dict] = []
    columns = ("round", "case", "group", "status", "decode_tok_s",
               "prefill_tok_s", "generated_tokens", "acceptance_pct",
               "tokens_per_round", "output_sha256", "error")
    with (a.out / "samples.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for i in range(a.rounds):
            for case in order_for_round(matrix, i):
                label = f"r{i+1}-{case.label}"
                cmd = command(a.cli, a.model, case, message_files[case.prompt], a.device)
                (a.out / f"{label}.command.json").write_text(json.dumps(cmd, indent=2) + "\n",
                                                               encoding="utf-8")
                item = {"round": i+1, "case": case.label, "group": case.group,
                        "status": "planned" if a.dry_run else "failed"}
                if not a.dry_run:
                    try:
                        run = subprocess.run(cmd, capture_output=True, text=True,
                                             errors="replace", timeout=a.timeout)
                        (a.out / f"{label}.stderr.log").write_text(run.stderr, encoding="utf-8")
                        (a.out / f"{label}.stdout.txt").write_text(run.stdout, encoding="utf-8")
                        if run.returncode:
                            raise RuntimeError(f"CLI exit {run.returncode}")
                        measures = parse_measurements(run.stderr)
                        required = "prefill_tok_s" if case.group == "prefill" else "decode_tok_s"
                        if measures[required] is None or measures["generated_tokens"] is None:
                            raise ValueError(f"missing valid {required} or token count")
                        item.update(measures)
                        item["output_sha256"] = hashlib.sha256(
                            run.stdout.encode("utf-8")).hexdigest()
                        item["status"] = "ok"
                    except (OSError, ValueError, RuntimeError,
                            subprocess.TimeoutExpired) as exc:
                        item["error"] = str(exc)
                writer.writerow({key: item.get(key) for key in columns})
                fp.flush()
                samples.append(item)
                print(f"{label}: {item['status']}", flush=True)
    (a.out / "summary.json").write_text(json.dumps(
        summarize(samples, matrix, a.rounds), indent=2) + "\n", encoding="utf-8")
    if a.dry_run:
        print(f"planned {len(samples)} runs; no model/GPU used")
        return 0
    return 1 if any(s["status"] != "ok" for s in samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
