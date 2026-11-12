"""Tests for the read-only cache/ReplaySSM metrics monitor."""
import pytest
from tools.bench.rtx4090_context_store_monitor import (
    parse_metrics, selected_series, differences, main,
)


def test_prometheus_parsing_and_histogram_labels():
    entries = parse_metrics(
        "# HELP ...\n"
        "ninfer:prefix_cache_hit_tokens_total 12\n"
        'ninfer:context_transfer_bytes_total{object="state",direction="d2h"} 4.25e3\n'
        'ninfer_time_to_first_token_seconds_bucket{le="1"} 2\n'
        "llamacpp:prompt_tokens_total 20\n"
        "uninvolved_stat 9\n")
    assert entries["ninfer:prefix_cache_hit_tokens_total"] == 12
    selected = selected_series(entries)
    assert "uninvolved_stat" not in selected
    assert "llamacpp:prompt_tokens_total" in selected
    assert len(selected) == 4


def test_metric_rejects_bad_input_and_duplicates():
    with pytest.raises(ValueError, match="unexpected"):
        parse_metrics("not valid\n")
    with pytest.raises(ValueError, match="duplicate"):
        parse_metrics("ninfer:host_cache_used_bytes 1\nninfer:host_cache_used_bytes 2\n")
    assert parse_metrics("ninfer:host_cache_used_bytes NaN\n") == {}


def test_difference_detects_resets_and_does_not_infer_speedup():
    result = differences({"ninfer:cache_total": 20, "ninfer:kv_pages": 4},
                         {"ninfer:cache_total": 1, "ninfer:kv_pages": 8})
    assert result["negative_changes_possible_reset"] == ["ninfer:cache_total"]
    assert result["observed_changes"]["ninfer:kv_pages"] == 4
    assert "ninfer:cache_total" not in result["observed_changes"]


def test_remote_monitor_rejected_without_network(tmp_path):
    with pytest.raises(SystemExit):
        main(["--url", "http://example.com", "--output", str(tmp_path / "a.json")])
