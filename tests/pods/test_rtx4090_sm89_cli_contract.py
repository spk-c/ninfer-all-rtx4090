"""Static source contract for the unmodified NInfer CLI used by SM89 tooling.

These guards check that future CLI changes don't silently turn benchmark flags
or KV-dtype parsing into rejected commands. They do not execute a CUDA binary.
"""
from pathlib import Path

from tools.bench.plan_rtx4090_sm89 import build_cases, command_tokens
from tools.bench.run_rtx4090_sm89_evidence import KV_SUMMARY_DTYPE


ROOT = Path(__file__).resolve().parents[2]
OPTIONS = (ROOT / "apps/cli/options.cpp").read_text(encoding="utf-8")
MAIN = (ROOT / "apps/cli/main.cpp").read_text(encoding="utf-8")
SPEC = (ROOT / "src/product/speculative_options.h").read_text(encoding="utf-8")


def test_all_planned_cli_switches_exist_in_options_parser():
    known = {
        "--messages", "--device", "--seed", "--greedy", "--no-thinking",
        "--gdn-state-fp16", "--ngram-draft-tokens", "--max-context",
        "--max-new", "--kv-dtype", "--prefill-chunk", "--spec",
        "--draft-tokens", "--prefill-cublas", "--no-prefill-cublas",
    }
    actual = set()
    for case in build_cases():
        args = command_tokens(case, Path("model.ninfer"), Path("fixtures"))
        actual.update(value for value in args if value.startswith("--"))
    assert actual == known
    # Positive/negative boolean switches are implemented by
    # product::switch_option, not literal flag comparisons in the parser.
    boolean_names = {"prefill-cublas", "thinking", "greedy", "gdn-state-fp16"}
    for name in boolean_names:
        assert f'switch_option(arg, "{name}"' in OPTIONS
    for flag in actual:
        if flag.removeprefix("--").removeprefix("no-") not in boolean_names:
            assert f'"{flag}"' in OPTIONS


def test_spec_none_and_dflash2_and_rotated_kv_formatting_are_real():
    assert 'if (value == "none") { return SpeculativeBackend::None; }' in SPEC
    assert 'if (value == "dflash2") { return SpeculativeBackend::DFlash2; }' in SPEC
    assert 'if (text == "rk4v4") { return KvCacheStorage::RotatedLloyd4KeyInt4Value; }' in OPTIONS
    assert 'if (text == "rk8v4") { return KvCacheStorage::RotatedInt8KeyInt4ValueGroup64; }' in OPTIONS
    assert 'return "rotated-lloyd4g64-v4g32";' in MAIN
    assert 'return "rotated-k8g64-v4g32";' in MAIN
    assert KV_SUMMARY_DTYPE == {
        "rk4v4": "rotated-lloyd4g64-v4g32",
        "rk8v4": "rotated-k8g64-v4g32",
    }
