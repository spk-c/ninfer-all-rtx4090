"""CPU-only tests for exploratory K5/cuBLAS/KV comparisons."""
from tools.bench.analyze_rtx4090_sm89_sweeps import summarize_sweeps


def record(case: str, arm: str, rep: int, rate: float,
           *, metric="decode_tok_s", tokens=128, answer="SAME",
           prompt=1900, status="passed", vram=12000, accepted=None,
           gold=None):
    return {
        "case": case, "arm": arm, "rep": str(rep),
        "status": status, "prompt_tokens": str(prompt),
        "generated_tokens": str(tokens), metric: str(rate),
        "answer_sha256": answer, "gpu_peak_used_mib": str(vram),
        "acceptance_pct": accepted, "gold_answer_pass": gold,
    }


def test_dflash2_k5_reference_and_answer_drift():
    rows = []
    for rep in range(1, 6):
        rows.append(record("dflash2-coding-short-k5", "A", rep, 100, accepted=60))
        rows.append(record("dflash2-coding-short-k3", "A", rep, 110, accepted=40))
    result = summarize_sweeps(rows, 5)
    k3 = next(x for x in result["dflash2_k"]
              if x["arm"] == "A" and x["fixture"] == "coding-short"
              and x["draft_tokens"] == 3)
    assert k3["against_k5"]["identical_output_pairs"] == 5
    assert k3["against_k5"]["median_same_output_gain_pct"] == 10.0
    assert k3["descriptive"]["median_decode_tok_s"] == 110
    assert k3["descriptive"]["median_acceptance_pct"] == 40
    assert result["promote_any_default"] is False
    rows[-1]["answer_sha256"] = "DIFFERENT"
    result = summarize_sweeps(rows, 5)
    k3 = next(x for x in result["dflash2_k"]
              if x["arm"] == "A" and x["fixture"] == "coding-short"
              and x["draft_tokens"] == 3)
    assert k3["against_k5"]["identical_output_pairs"] == 4
    assert k3["against_k5"]["median_same_output_gain_pct"] is None
    assert k3["against_k5"]["different_answer_pairs"] == 1


def test_cublas_and_kv_early_missing_and_memory():
    rows = []
    for rep in range(1, 6):
        rows.append(record(
            "prefill-coding-long-2048-cublas-0", "B", rep, 150,
            metric="prefill_tok_s", tokens=128))
        rows.append(record(
            "prefill-coding-long-2048-cublas-1", "B", rep, 180,
            metric="prefill_tok_s", tokens=128))
    rows.append(record("kv-65536-rk4v4-seed2", "A", 1, 45,
                       gold=True, vram=13000))
    rows.append(record("kv-65536-rk8v4-seed2", "A", 1, 40,
                       gold=True, vram=17000))
    s = summarize_sweeps(rows, 5)
    prefill = next(x for x in s["prefill_cublas"] if x["arm"] == "B"
                   and x["fixture"] == "coding-long" and x["chunk"] == 2048)
    assert prefill["on_vs_off"]["median_same_output_gain_pct"] == 20.0
    assert prefill["off"]["median_prefill_tok_s"] == 150.0
    assert prefill["on"]["median_prefill_tok_s"] == 180.0
    kv = next(x for x in s["kv_dtype"] if x["arm"] == "A"
              and x["context_target"] == 65536 and x["seed"] == 2)
    assert kv["rk4v4"]["gold_answer_pass_count"] == 1
    assert kv["rk8v4"]["median_whole_gpu_vram_mib"] == 17000
    assert kv["rk8v4_vs_rk4v4"]["identical_output_pairs"] == 1
    assert kv["rk8v4_vs_rk4v4"]["median_same_output_gain_pct"] is None


def test_empty_matrix_does_not_invent_speed_or_accuracy():
    s = summarize_sweeps([], 5)
    assert len(s["dflash2_k"]) == 24
    assert len(s["prefill_cublas"]) == 12
    assert len(s["kv_dtype"]) == 30
    assert s["promote_any_default"] is False
    assert all(x["descriptive"]["median_decode_tok_s"] is None
               for x in s["dflash2_k"])


def test_zero_dflash_acceptance_is_measured_not_missing():
    rows = [record("dflash2-coding-short-k3", "A", 1, 30, accepted="0")]
    s = summarize_sweeps(rows, 5)
    k3 = next(x for x in s["dflash2_k"] if x["arm"] == "A"
              and x["fixture"] == "coding-short" and x["draft_tokens"] == 3)
    assert k3["descriptive"]["median_acceptance_pct"] == 0.0
