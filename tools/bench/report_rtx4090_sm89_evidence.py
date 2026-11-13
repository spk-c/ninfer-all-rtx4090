"""Summarize SM89 raw paired evidence without promoting unverified kernel changes.

Use paired per-repetition B/A ratios. A confidence interval does not replace
independent coding-quality, long-context and Nsight sign-off.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Sequence

from tools.bench.analyze_rtx4090_sm89_sweeps import summarize_sweeps


def paired_ratios(rows: list[dict], metric: str) -> tuple[list[float], int]:
    by_rep = defaultdict(dict)
    for row in rows:
        if row["status"] != "passed":
            continue
        try:
            value = float(row[metric])
            rep = int(row["rep"])
        except (ValueError, TypeError, KeyError):
            continue
        if math.isfinite(value) and value > 0:
            by_rep[rep][row["arm"]] = value
    pairs = [d["B"] / d["A"] for _, d in sorted(by_rep.items())
             if "A" in d and "B" in d]
    return pairs, sum(r <= 1.0 for r in pairs)


def median_ci(ratios: list[float], trials: int = 4000) -> tuple[float, float] | None:
    if len(ratios) < 5:
        return None
    rng = random.Random(4090)
    n = len(ratios)
    samples = sorted(statistics.median(ratios[rng.randrange(n)]
                                       for _ in range(n)) for _ in range(trials))
    return samples[int(trials * 0.025)], samples[int(trials * 0.975)]


def number_or_none(raw) -> float | None:
    try:
        n = float(raw)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def eligible_paired_samples(rows: list[dict], repetitions: int) -> tuple[list[dict], dict]:
    """Only comparable prompt AND generated-token lengths enter rate ratios.

    Missing, duplicate, failed and uneven-length pairs remain in the report as
    disqualifications; they are never quietly dropped and called complete.
    """
    by_rep: dict[int, dict[str, dict]] = defaultdict(dict)
    duplicates = 0
    for row in rows:
        try:
            rep = int(row["rep"])
            arm = row["arm"]
        except (ValueError, TypeError, KeyError):
            duplicates += 1
            continue
        if not 1 <= rep <= repetitions or arm not in ("A", "B"):
            duplicates += 1
            continue
        if arm in by_rep[rep]:
            duplicates += 1
        by_rep[rep][arm] = row

    pair_rows = []
    reasons: dict[str, int] = defaultdict(int)
    missing_hash = 0
    for rep in range(1, repetitions + 1):
        pair = by_rep[rep]
        if set(pair) != {"A", "B"}:
            reasons["missing_pair"] += 1
            continue
        arm_a, arm_b = pair["A"], pair["B"]
        if arm_a["status"] != "passed" or arm_b["status"] != "passed":
            reasons["failed_arm"] += 1
            continue
        try:
            prompt_a, prompt_b = int(arm_a["prompt_tokens"]), int(arm_b["prompt_tokens"])
            output_a, output_b = int(arm_a["generated_tokens"]), int(arm_b["generated_tokens"])
        except (KeyError, TypeError, ValueError):
            reasons["missing_token_counts"] += 1
            continue
        if min(prompt_a, prompt_b, output_a, output_b) <= 0:
            reasons["missing_token_counts"] += 1
            continue
        if prompt_a != prompt_b:
            reasons["different_prompt_tokens"] += 1
            continue
        if output_a != output_b:
            reasons["different_generated_tokens"] += 1
            continue
        if not arm_a.get("answer_sha256") or not arm_b.get("answer_sha256"):
            missing_hash += 1
            reasons["missing_answer_hash"] += 1
            continue
        pair_rows.extend((arm_a, arm_b))
    return pair_rows, {
        "duplicate_rows": duplicates,
        "disqualified_pairs": dict(reasons),
        "missing_hash_pairs": missing_hash,
    }


def summarize(rows: list[dict], metadata: dict, expected_cases: list[dict]) -> dict:
    by_name = defaultdict(list)
    for row in rows:
        by_name[row["case"]].append(row)
    reps = metadata["plan_manifest"]["repetitions"]
    evidence = []
    for case in expected_cases:
        name = case["name"]
        samples = by_name.get(name, [])
        family = case["family"]
        metric = "prefill_tok_s" if family == "prefill" else "decode_tok_s"
        eligible, flags = eligible_paired_samples(samples, reps)
        ratios, negative = paired_ratios(eligible, metric)
        ci = median_ci(ratios)
        valid = (len(samples) == 2 * reps and len(ratios) == reps
                 and flags["duplicate_rows"] == 0
                 and not flags["disqualified_pairs"]
                 and all(r["status"] == "passed" for r in samples))
        hashes = defaultdict(dict)
        for row in samples:
            hashes[row["rep"]][row["arm"]] = row.get("answer_sha256")
        diverged = sum(
            d.get("A") != d.get("B") for d in hashes.values()
            if d.get("A") and d.get("B"))
        median = statistics.median(ratios) if ratios else None
        nominal_win = (valid and ci is not None and median is not None
                       and median >= 1.03 and ci[0] > 1.0
                       and diverged == 0)
        gpu_mib = []
        for row in samples:
            value = row.get("gpu_peak_used_mib")
            try:
                if value not in (None, "", "None"):
                    n = float(value)
                    if math.isfinite(n) and n >= 0:
                        gpu_mib.append(n)
            except (ValueError, TypeError):
                pass
        evidence.append({
            "case": name, "family": family, "metric": metric, "valid": valid,
            "pairs": len(ratios), "expected_pairs": reps,
            "median_paired_gain_pct": round((median - 1) * 100, 3)
                                      if median is not None else None,
            "bootstrap_95_ci_gain_pct": [
                round((x - 1) * 100, 3) for x in ci] if ci else None,
            "non_positive_pairs": negative,
            "answer_sha_divergent_pairs": diverged,
            "nominal_performance_win": bool(nominal_win),
            "pair_disqualifications": flags["disqualified_pairs"],
            "duplicate_sample_rows": flags["duplicate_rows"],
            "quality_passes": sum(r.get("gold_answer_pass") in (True, "True")
                                  for r in samples),
            "failed_samples": [
                {"rep": r["rep"], "arm": r["arm"],
                 "status": r["status"], "reason": r.get("error", "")}
                for r in samples if r["status"] != "passed"],
            "max_observed_whole_gpu_vram_mib": max(gpu_mib, default=None),
            "max_observed_gpu_power_w": max(
                (n for row in samples
                 if (n := number_or_none(row.get("gpu_peak_power_w"))) is not None),
                default=None),
            "min_observed_sm_clock_mhz": min(
                (n for row in samples
                 if (n := number_or_none(row.get("gpu_min_sm_clock_mhz"))) is not None),
                default=None),
            "max_observed_sm_clock_mhz": max(
                (n for row in samples
                 if (n := number_or_none(row.get("gpu_max_sm_clock_mhz"))) is not None),
                default=None),
        })
    complete = (metadata.get("status") == "COMPLETE"
                and metadata.get("warmup_count") == 2 * len(expected_cases)
                and metadata.get("warmup_failures") == 0
                and len(rows) == len(expected_cases) * 2 * reps
                and all(x["valid"] for x in evidence))
    regression = [x for x in evidence if x["valid"]
                  and x["median_paired_gain_pct"] is not None
                  and x["median_paired_gain_pct"] < -2]
    return {
        "schema": "ninfer_rtx4090_sm89_evidence_summary_v2",
        "baseline_commit": metadata["plan_manifest"]["baseline_commit_sha"],
        "candidate_commit": metadata["plan_manifest"]["candidate_commit_sha"],
        "complete": complete, "recorded_samples": len(rows),
        "required_samples": len(expected_cases) * 2 * reps,
        "quality_signoff": "PENDING_INDEPENDENT_REVIEW",
        "ttft": "NOT_MEASURED (CLI process elapsed is not server TTFT)",
        "nsight_qualification": "PENDING",
        "autopromote": False,
        "regression_cases": regression,
        "parameter_sweeps": summarize_sweeps(rows, reps),
        "cases": evidence,
    }


def write_report(directory: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("report directory already exists; refusing to overwrite")
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    plan_dir = Path(meta["plan_manifest"]["fixtures_dir"])
    # Case source belongs to the benchmark input, not to an altered model checkpoint.
    # The checked-in matrix yields exactly 54 named cases.
    from tools.bench.plan_rtx4090_sm89 import build_cases
    expected = [dict(name=c.name, family=c.family) for c in build_cases()]
    with (directory / "samples.csv").open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    report = summarize(rows, meta, expected)
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n",
                                         encoding="utf-8")
    lines = [
        "# RTX 4090 / SM89 paired campaign evidence",
        "",
        f"Baseline: {report['baseline_commit']}",
        f"Candidate: {report['candidate_commit']}",
        f"Recorded: {report['recorded_samples']} / {report['required_samples']} samples",
        f"Complete paired matrix: **{report['complete']}**",
        "",
        "This is **not** a release or kernel-promotion decision: quality sign-off "
        "and Nsight profiling are outstanding. CUDA CLI elapsed is not HTTP TTFT.",
        "",
        "| Case | Pairs | Median B/A | 95% bootstrap interval | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for c in report["cases"]:
        gain = f"{c['median_paired_gain_pct']:+.2f}%" if c["median_paired_gain_pct"] is not None else "—"
        ci = (" / ".join(f"{x:+.2f}%" for x in c["bootstrap_95_ci_gain_pct"])
              if c["bootstrap_95_ci_gain_pct"] else "—")
        status = "complete" if c["valid"] else "INCOMPLETE / FAILED"
        lines.append(f"| {c['case']} | {c['pairs']}/{c['expected_pairs']} "
                     f"| {gain} | {ci} | {status} |")
    lines += [
        "",
        "## Configuration sweeps (exploratory; NOT a promotion decision)",
        "",
        "Rates are descriptive per configuration. Cross-configuration B/A ratios "
        "are only reported when actual prompt/output counts and response hashes match.",
        "",
        "### DFlash2 — K5 reference",
        "",
        "| Arm | Fixture | K | Decode median tok/s | Acceptance | K vs K5 (equal output) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    sweep = report["parameter_sweeps"]
    for entry in sweep["dflash2_k"]:
        x = entry["descriptive"]
        ratio = entry["against_k5"]
        gain = (ratio["median_same_output_gain_pct"] if ratio else None)
        lines.append(
            f"| {entry['arm']} | {entry['fixture']} | {entry['draft_tokens']} "
            f"| {x['median_decode_tok_s'] if x['median_decode_tok_s'] is not None else '—'} "
            f"| {x['median_acceptance_pct'] if x['median_acceptance_pct'] is not None else '—'} "
            f"| {f'{gain:+.2f}%' if gain is not None else '—'} |"
        )
    lines += [
        "",
        "### Prefill — cuBLAS on/off by chunk",
        "",
        "| Arm | Fixture | Chunk | Off median tok/s | On median tok/s | On vs off (equal output) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entry in sweep["prefill_cublas"]:
        ratio = entry["on_vs_off"]["median_same_output_gain_pct"]
        off = entry["off"]["median_prefill_tok_s"]
        on = entry["on"]["median_prefill_tok_s"]
        lines.append(
            f"| {entry['arm']} | {entry['fixture']} | {entry['chunk']} "
            f"| {off if off is not None else '—'} | {on if on is not None else '—'} "
            f"| {f'{ratio:+.2f}%' if ratio is not None else '—'} |"
        )
    lines += ["", "### KV format / 32K–128K", "",
              "Per-seed exact-gold passes, median throughput and observed whole-GPU VRAM "
              "for rk4v4 and rk8v4 are in report.json > parameter_sweeps > kv_dtype. "
              "Missing or OOM contexts remain explicitly unqualified.",
              "", "## Promotion status",
              "",
              "Not eligible: independent coding-quality, long-context retrieval, "
              "precision/perplexity and Nsight sign-off are not attached.",
              "No experimental kernel is enabled by this report.",
              ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        summary = write_report(args.evidence_dir, args.out)
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    print(json.dumps({"complete": summary["complete"],
                      "samples": summary["recorded_samples"],
                      "autopromote": False}, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
