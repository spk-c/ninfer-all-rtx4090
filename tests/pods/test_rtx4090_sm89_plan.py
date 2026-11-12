"""CPU-only contract tests of the RTX 4090 master-vs-branch plan."""
import json
from pathlib import Path

import pytest

from tools.bench.plan_rtx4090_sm89 import (
    BASE_SHA, CONTEXTS, DECODE_METRIC, DRAFT_K, KV_TYPES, PREFILL_CHUNKS,
    PREFILL_METRIC, SEEDS, ab_command, build_cases, command_tokens,
    create_plan, nsight_commands,
)


def test_exact_selected_matrix():
    cases = build_cases()
    assert len(cases) == 54
    assert len(set(c.name for c in cases)) == 54
    assert {c.family: sum(x.family == c.family for x in cases)
            for c in cases} == {"dflash2": 12, "prefill": 12, "kv": 30}
    assert DRAFT_K == (3, 5, 7, 9)
    assert PREFILL_CHUNKS == (1024, 2048, 4096)
    assert CONTEXTS == (32768, 65536, 131072)
    assert KV_TYPES == ("rk4v4", "rk8v4")
    assert len(SEEDS) == 5


def test_same_weights_sampler_and_explicit_isolated_config():
    for case in build_cases():
        tokens = command_tokens(case, Path("model.ninfer"), Path("fixtures"))
        assert tokens[:2] == ["{exe}", "model.ninfer"]
        assert tokens.count("--messages") == 1
        assert tokens.count("--kv-dtype") == 1
        assert tokens.count("--ngram-draft-tokens") == 1
        assert tokens[tokens.index("--ngram-draft-tokens") + 1] == "0"
        assert "--greedy" in tokens and "--no-thinking" in tokens
        assert tokens[tokens.index("--seed") + 1] == "4090"
        assert "--gdn-state-fp16" in tokens
        assert tokens[tokens.index("--max-context") + 1] == str(case.max_context)
        if case.family == "dflash2":
            assert tokens[tokens.index("--spec") + 1] == "dflash2"
        else:
            assert tokens[tokens.index("--spec") + 1] == "none"


def test_family_metrics_and_off_on_prefill():
    cases = build_cases()
    for case in cases:
        assert case.metric == (PREFILL_METRIC if case.family == "prefill"
                               else DECODE_METRIC)
    switches = {flag for case in cases if case.family == "prefill"
                for flag in case.flags if "cublas" in flag}
    assert switches == {"--prefill-cublas", "--no-prefill-cublas"}


def test_argv_keeps_distinct_build_trees_and_quotes(tmp_path):
    case = build_cases()[0]
    command = ab_command(case, Path("master/apps/ninfer.exe"),
                         Path("candidate/apps/ninfer.exe"), Path("model.ninfer"),
                         tmp_path / "coding prompts", 5)
    assert "--allow-separate-dirs" in command
    assert command[command.index("--arm-a") + 1] != command[command.index("--arm-b") + 1]
    config = command[command.index("--config") + 1]
    assert config.count("{exe}") == 1
    assert '"' + str(tmp_path / "coding prompts" / case.fixture) + '"' in config
    assert command[command.index("--reps") + 1] == "5"


def test_nsys_and_ncu_cover_all_three_sm89_kernel_families(tmp_path):
    profiles = nsight_commands(Path("build/bench/ninfer_benches.exe"), tmp_path)
    assert {c["name"] for c in profiles} == {"q4-small-t", "q5-small-t", "w4a8"}
    assert all("--trace=cuda,nvtx" in c["nsys"] for c in profiles)
    assert all("--set" in c["ncu"] for c in profiles)
    assert all("registers_per_thread" in c["required_metrics"] for c in profiles)


def test_plan_is_evidence_free_and_never_overwrites(tmp_path):
    out = tmp_path / "report"
    params = (Path("master/apps/ninfer.exe"), Path("candidate/apps/ninfer.exe"),
              Path("nonexistent-model.ninfer"), tmp_path, "c" * 40, 5, out)
    plan = create_plan(*params)
    assert plan["baseline_commit_sha"] == BASE_SHA
    assert plan["case_count"] == 54
    assert plan["expected_arm_runs"] == 540
    assert plan["model_sha256"] is None
    assert plan["evidence_status"] == "NOT_RUN"
    assert len(json.loads((out / "fixture-requirements.json").read_text())) == 18
    assert len(json.loads((out / "ab-commands.json").read_text())) == 54
    with pytest.raises(ValueError, match="overwrite"):
        create_plan(*params)


def test_preflight_rejects_unqualified_runs(tmp_path):
    with pytest.raises(ValueError, match="same commit"):
        create_plan(Path("a"), Path("b"), Path("c"), tmp_path,
                    BASE_SHA, 5, tmp_path / "same")
    with pytest.raises(ValueError, match="five paired"):
        create_plan(Path("a"), Path("b"), Path("c"), tmp_path,
                    "d" * 40, 2, tmp_path / "few")
    with pytest.raises(ValueError, match="required model or executable"):
        create_plan(Path("a"), Path("b"), Path("c"), tmp_path,
                    "d" * 40, 5, tmp_path / "strict", strict=True)
