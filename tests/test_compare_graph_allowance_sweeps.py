"""GPU-free comparison tests for graph-memory reports."""
import pytest

from tools.bench.compare_graph_allowance_sweeps import compare_reports


def sample(capacity, observed, code=0):
    return {"capacity": capacity, "observed_bytes": observed,
            "returncode": code}


def sweep(values, **overrides):
    config = {"model": "C:/models/qwen.ninfer", "spec": "dflash2",
              "draft_tokens": 5, "kv_dtype": "rk4v4", "prefill_chunk": 1024}
    config.update(overrides)
    return {"config": config, "runs": values}


def test_nonoverlapping_observations_are_only_described_as_memory_deltas():
    first = sweep([sample(8192, 90), sample(8192, 100), sample(8192, 105)])
    second = sweep([sample(8192, 60), sample(8192, 65), sample(8192, 70)])
    result = compare_reports(first, second)
    row = result["rows"][0]
    assert row["baseline_median_bytes"] == 100
    assert row["candidate_median_bytes"] == 65
    assert row["candidate_minus_baseline_bytes"] == -35
    assert row["verdict"].startswith("lower observed VRAM delta")
    assert result["scope"] == "CUDA Graph startup-memory observations, not throughput"


def test_device_wide_noise_and_zero_deltas_do_not_become_a_speedup():
    first = sweep([sample(8192, 0), sample(8192, 80), sample(8192, 120)])
    second = sweep([sample(8192, 100), sample(8192, 110), sample(8192, 130)])
    row = compare_reports(first, second)["rows"][0]
    assert row["baseline_successes"] == 2
    assert row["verdict"].startswith("inconclusive")


def test_startup_failures_are_not_counted_as_valid_calibration_samples():
    first = sweep([sample(4096, 90), sample(4096, 100)])
    second = sweep([sample(4096, 10, code=1), sample(4096, 40)])
    row = compare_reports(first, second)["rows"][0]
    assert row["candidate_successes"] == 1
    assert row["verdict"] == "insufficient samples"


def test_mixed_backend_is_not_compared():
    with pytest.raises(ValueError, match="spec"):
        compare_reports(sweep([sample(4096, 60)]),
                        sweep([sample(4096, 50)], spec="mtp"))


def test_nonmatching_capacities_are_rejected():
    with pytest.raises(ValueError, match="no matching"):
        compare_reports(sweep([sample(4096, 60)]), sweep([sample(8192, 50)]))


def test_mixed_gdn_precision_is_not_compared():
    with pytest.raises(ValueError, match="gdn_state_fp16"):
        compare_reports(sweep([sample(8192, 60)], gdn_state_fp16=False),
                        sweep([sample(8192, 50)], gdn_state_fp16=True))


def test_copy_proposals_change_graph_topology_and_are_not_compared():
    with pytest.raises(ValueError, match="ngram_draft_tokens"):
        compare_reports(sweep([sample(8192, 60)], ngram_draft_tokens=0),
                        sweep([sample(8192, 50)], ngram_draft_tokens=15))


def budget_sample(observed, allowance=100, code=0, **extra):
    return dict(sample(8192, observed, code), allowance_bytes=allowance, **extra)


def test_partial_or_over_budget_sweep_cannot_pass_startup_screening():
    baseline = sweep([budget_sample(70), budget_sample(80)])
    for third in (budget_sample(0), budget_sample(20, code=1),
                  budget_sample(110), budget_sample(100),
                  budget_sample(20, graph_blocked=True),
                  budget_sample(float("inf")), budget_sample(float("nan")),
                  budget_sample(20, allowance=None)):
        candidate = sweep([budget_sample(30), budget_sample(40), third])
        result = compare_reports(baseline, candidate)
        assert not result["rows"][0]["startup_screening_complete"]
        assert not result["default_reduction_qualified"]


def test_complete_startup_screen_is_not_default_reduction_qualification():
    result = compare_reports(
        sweep([budget_sample(70), budget_sample(80)]),
        sweep([budget_sample(30), budget_sample(40)]))
    assert result["rows"][0]["startup_screening_complete"]
    assert result["all_capacities_matched"]
    assert not result["default_reduction_qualified"]


@pytest.mark.parametrize("second_hash", [None, "b" * 64])
def test_model_path_alone_cannot_override_known_hash(second_hash):
    rows = [budget_sample(30), budget_sample(40)]
    with pytest.raises(ValueError, match="model_sha256"):
        compare_reports(sweep(rows, model_sha256="a" * 64),
                        sweep(rows, model_sha256=second_hash))


def test_unmatched_capacity_is_reported():
    a = sweep([budget_sample(30), budget_sample(40), sample(16384, 80)])
    b = sweep([budget_sample(30), budget_sample(40)])
    assert not compare_reports(a, b)["all_capacities_matched"]
