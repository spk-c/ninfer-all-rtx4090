"""Exploratory WITHIN-BUILD K, cuBLAS and KV sweeps from SM89 raw evidence.

Unlike master-vs-candidate paired rates, these modes can generate different
answers. Never rank different-output throughput as a proven optimization.
The 54-case dataset remains the source of truth; no benchmark is executed.
"""
from __future__ import annotations

from collections import defaultdict
import math
import statistics

from tools.bench.plan_rtx4090_sm89 import CONTEXTS, DRAFT_K, KV_TYPES, PREFILL_CHUNKS, SEEDS, PROMPTS


def number(raw):
    try:
        n = float(raw)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def median_positive(rows: list[dict], key: str) -> float | None:
    data = [value for row in rows if row.get("status") == "passed"
            if (value := number(row.get(key))) is not None and value > 0]
    return round(statistics.median(data), 4) if data else None


def valid_rows(index: dict, case: str, arm: str) -> list[dict]:
    return [r for r in index.get((case, arm), {}).values()
            if r.get("status") == "passed"]


def descriptive(index: dict, name: str, arm: str, family: str) -> dict:
    rows = valid_rows(index, name, arm)
    metric = "prefill_tok_s" if family == "prefill" else "decode_tok_s"
    return {
        "case": name,
        "arm": arm,
        "successful_runs": len(rows),
        "median_" + metric: median_positive(rows, metric),
        "median_acceptance_pct": median_positive(rows, "acceptance_pct"),
        "median_tokens_per_round": median_positive(rows, "tokens_per_round"),
        "median_whole_gpu_vram_mib": median_positive(rows, "gpu_peak_used_mib"),
        "gold_answer_pass_count": sum(
            r.get("gold_answer_pass") in (True, "True") for r in rows),
    }


def pair_exploratory(index: dict, baseline: str, variant: str,
                     arm: str, metric: str, repetitions: int) -> dict:
    """Only a byte-identical output can support matched-rate comparison."""
    a = index.get((baseline, arm), {})
    b = index.get((variant, arm), {})
    comparable = []
    different_answers = 0
    mismatched_counts = 0
    not_measured = 0
    for rep in range(1, repetitions + 1):
        old, new = a.get(rep), b.get(rep)
        if not old or not new or old.get("status") != "passed" or new.get("status") != "passed":
            not_measured += 1
            continue
        try:
            p_a, p_b = int(old["prompt_tokens"]), int(new["prompt_tokens"])
            g_a, g_b = int(old["generated_tokens"]), int(new["generated_tokens"])
        except (TypeError, ValueError, KeyError):
            mismatched_counts += 1
            continue
        if p_a <= 0 or g_a <= 0 or p_a != p_b or g_a != g_b:
            mismatched_counts += 1
            continue
        if not old.get("answer_sha256") or old["answer_sha256"] != new.get("answer_sha256"):
            different_answers += 1
            continue
        first, second = number(old.get(metric)), number(new.get(metric))
        if first is None or second is None or first <= 0 or second <= 0:
            not_measured += 1
            continue
        comparable.append(second / first)
    return {
        "arm": arm, "baseline": baseline, "variant": variant,
        "metric": metric, "identical_output_pairs": len(comparable),
        "expected_pairs": repetitions,
        "different_answer_pairs": different_answers,
        "token_count_mismatch_pairs": mismatched_counts,
        "missing_or_failed_pairs": not_measured,
        "median_same_output_gain_pct": (
            round(100 * (statistics.median(comparable) - 1), 3)
            if len(comparable) == repetitions else None),
        "comparison_status": (
            "EXPLORATORY_MATCHED_OUTPUT" if len(comparable) == repetitions
            else "UNQUALIFIED_DIFFERENT_OUTPUT_OR_MISSING_EVIDENCE"),
        "promotion_allowed": False,
    }


def summarize_sweeps(rows: list[dict], repetitions: int) -> dict:
    index: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    duplicates = 0
    for row in rows:
        try:
            rep, arm = int(row["rep"]), row["arm"]
            if arm not in ("A", "B") or not (1 <= rep <= repetitions):
                duplicates += 1
                continue
            key = (row["case"], arm)
        except (TypeError, ValueError, KeyError):
            duplicates += 1
            continue
        if rep in index[key]:
            duplicates += 1
        index[key][rep] = row

    draft, prefill, kv = [], [], []
    for arm in ("A", "B"):
        for prompt in PROMPTS:
            baseline = f"dflash2-{prompt}-k5"
            for k in DRAFT_K:
                candidate = f"dflash2-{prompt}-k{k}"
                draft.append({
                    "arm": arm, "fixture": prompt, "draft_tokens": k,
                    "descriptive": descriptive(index, candidate, arm, "dflash2"),
                    "against_k5": (
                        pair_exploratory(index, baseline, candidate, arm,
                                         "decode_tok_s", repetitions)
                        if k != 5 else None),
                })
        for prompt in ("coding-short", "coding-long"):
            for chunk in PREFILL_CHUNKS:
                baseline = f"prefill-{prompt}-{chunk}-cublas-0"
                variant = f"prefill-{prompt}-{chunk}-cublas-1"
                prefill.append({
                    "arm": arm, "fixture": prompt, "chunk": chunk,
                    "off": descriptive(index, baseline, arm, "prefill"),
                    "on": descriptive(index, variant, arm, "prefill"),
                    "on_vs_off": pair_exploratory(
                        index, baseline, variant, arm, "prefill_tok_s", repetitions),
                })
        for ctx in CONTEXTS:
            for seed in SEEDS:
                reference = f"kv-{ctx}-rk4v4-seed{seed}"
                variant = f"kv-{ctx}-rk8v4-seed{seed}"
                kv.append({
                    "arm": arm, "context_target": ctx, "seed": seed,
                    "rk4v4": descriptive(index, reference, arm, "kv"),
                    "rk8v4": descriptive(index, variant, arm, "kv"),
                    "rk8v4_vs_rk4v4": pair_exploratory(
                        index, reference, variant, arm, "decode_tok_s", repetitions),
                })
    return {
        "schema": "ninfer_rtx4090_parameter_sweeps_v1",
        "note": ("Within-arm case order can differ and outputs may diverge. "
                 "Matched same-output ratios are EXPLORATORY, not promotion evidence; "
                 "separate quality, Nsight and hardware checks are required."),
        "duplicate_sample_rows": duplicates,
        "dflash2_k": draft,
        "prefill_cublas": prefill,
        "kv_dtype": kv,
        "promote_any_default": False,
    }
