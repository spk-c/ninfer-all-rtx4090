"""CPU-only correctness tests for the paired SM89 measurement and report path."""
import csv
import json
from pathlib import Path

import pytest

from tools.bench.plan_rtx4090_sm89 import BASE_SHA, build_cases
from tools.bench.run_rtx4090_sm89_evidence import (
    classify, extract_summary, parse_metrics, run_campaign, validate_plan,
)
from tools.bench.report_rtx4090_sm89_evidence import (
    paired_ratios, median_ci, summarize, write_report,
)


CLI_LOG = """
summary     prompt tokens             31042
summary     generated tokens          128
summary     prefill speed             270.50 tok/s
summary     decode speed              84.55 tok/s
summary     max context               32768
summary     kv cache dtype            rk4v4
summary     gpu workspace peak        200 MiB / 500 MiB
summary     dflash2 acceptance rate   71.02%
summary     dflash2 acceptance length 2.71 tok/round
"""


def test_actual_cli_summary_and_long_label():
    fields = extract_summary(CLI_LOG)
    assert fields["prompt tokens"] == "31042"
    assert fields["dflash2 acceptance length"] == "2.71 tok/round"
    values = parse_metrics(CLI_LOG)
    assert values["prompt_tokens"] == 31042
    assert values["prefill_tok_s"] == 270.5
    assert values["decode_tok_s"] == 84.55
    assert values["acceptance_pct"] == 71.02
    assert values["tokens_per_round"] == 2.71
    assert values["max_context_reported"] == 32768
    assert values["workspace_peak"].startswith("200 MiB")


def test_fail_closed_for_missing_metrics_and_actual_kv_tokens():
    draft = next(c for c in build_cases() if c.family == "dflash2")
    kv = next(c for c in build_cases() if c.family == "kv")
    draft_row = vars(draft).copy()
    kv_row = vars(kv).copy()
    status, error, _ = classify(draft_row, parse_metrics(CLI_LOG),
                                "answer", 0, "", {})
    assert status == "passed"
    without = parse_metrics(CLI_LOG)
    without["acceptance_pct"] = None
    assert classify(draft_row, without, "answer", 0, "", {})[0] == "failed"
    kv_metrics = parse_metrics(CLI_LOG)
    kv_metrics["prompt_tokens"] = 1000
    kv_metrics["max_context_reported"] = kv.max_context
    assert classify(kv_row, kv_metrics, "code", 0, "",
                    {kv.fixture: {"expected": "code"}})[0] == "insufficient_context"


def test_gold_answer_and_oom_classification():
    kv = next(c for c in build_cases() if c.family == "kv")
    case = vars(kv).copy()
    values = parse_metrics(CLI_LOG)
    values["prompt_tokens"] = kv.minimum_prompt_tokens + 10
    values["max_context_reported"] = kv.max_context
    expected = {kv.fixture: {"expected": "C0DE-REFERENCE-123"}}
    assert classify(case, values, "C0DE-REFERENCE-123", 0, "", expected)[0] == "passed"
    assert classify(case, values, "wrong output", 0, "", expected)[0] == "quality_regression"
    assert classify(case, values, "", 1, "", expected,
                    "CUDA out of memory")[0] == "OOM"


def test_paired_statistics_use_per_rep_ratios_not_ratio_of_medians():
    rows = []
    for rep, a, b in [(1, 10, 12), (2, 100, 108), (3, 30, 33),
                      (4, 50, 56), (5, 18, 21)]:
        rows += [
            {"arm": "A", "rep": str(rep), "status": "passed", "decode_tok_s": str(a)},
            {"arm": "B", "rep": str(rep), "status": "passed", "decode_tok_s": str(b)},
        ]
    ratios, nonpositive = paired_ratios(rows, "decode_tok_s")
    assert len(ratios) == 5 and nonpositive == 0
    ci = median_ci(ratios, 500)
    assert ci and ci[0] > 1 and ci[1] >= ci[0]
    rows[3]["status"] = "OOM"
    ratios, _ = paired_ratios(rows, "decode_tok_s")
    assert len(ratios) == 4
    assert median_ci(ratios) is None


def test_incomplete_matrix_cannot_auto_promote():
    row = {"case": "dflash2-coding-short-k3", "rep": "1", "arm": "A",
           "status": "passed", "decode_tok_s": "100",
           "answer_sha256": "aaa", "gpu_peak_used_mib": "12000"}
    other = dict(row, arm="B", decode_tok_s="150", answer_sha256="bbb")
    meta = {"status": "PARTIAL", "plan_manifest": {
        "repetitions": 5, "baseline_commit_sha": BASE_SHA,
        "candidate_commit_sha": "f" * 40}}
    summary = summarize([row, other], meta, [
        {"name": "dflash2-coding-short-k3", "family": "dflash2"}])
    assert summary["complete"] is False
    assert summary["cases"][0]["answer_sha_divergent_pairs"] == 1
    assert summary["cases"][0]["pairs"] == 1
    assert summary["autopromote"] is False


def test_report_generates_real_artifacts_without_simulated_speed(tmp_path):
    evidence = tmp_path / "measure"
    evidence.mkdir()
    out = tmp_path / "report"
    metadata = {"status": "PARTIAL", "plan_manifest": {
        "fixtures_dir": str(tmp_path), "repetitions": 5,
        "baseline_commit_sha": BASE_SHA, "candidate_commit_sha": "f" * 40}}
    (evidence / "metadata.json").write_text(json.dumps(metadata))
    with (evidence / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case", "rep", "arm", "status", "decode_tok_s",
            "prefill_tok_s", "answer_sha256", "gold_answer_pass",
            "gpu_peak_used_mib",
        ])
        writer.writeheader()
    result = write_report(evidence, out)
    assert result["complete"] is False
    assert result["recorded_samples"] == 0
    assert result["autopromote"] is False
    assert (out / "summary.md").is_file()
    with pytest.raises(ValueError, match="overwrite"):
        write_report(evidence, out)


def test_invalid_run_parameters_fail_before_any_hardware_call(tmp_path):
    with pytest.raises(ValueError, match="five AB/BA"):
        run_campaign(tmp_path, tmp_path / "out", tmp_path, tmp_path,
                     tmp_path / "gold.json", 2, 0, 2048, 30)
