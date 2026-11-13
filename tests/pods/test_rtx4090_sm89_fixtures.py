"""CPU-only fixture reproducibility and gold-answer checks."""
import json
from pathlib import Path

import pytest

from tools.bench.make_rtx4090_sm89_fixtures import (
    build, canary, choose_fragments, make_kv_fixture, source_fragments,
)


def project_source(tmp_path):
    root = tmp_path / "master"
    (root / "src").mkdir(parents=True)
    for i in range(8):
        content = (
            f"// unique project file {i}\n"
            + "".join(f"int unique_symbol_{i}_{n}(int a) {{ return a + {n}; }}\n"
                      for n in range(120))
        )
        (root / "src" / f"source_{i}.cpp").write_text(content, encoding="utf-8")
    return root


def test_fixture_generation_is_reproducible_and_non_tiled(tmp_path):
    root = project_source(tmp_path)
    out1, out2 = tmp_path / "output1", tmp_path / "output2"
    a = build(root, out1, contexts=(1024,), seeds=(1, 2),
              chars_per_token=3.0)
    b = build(root, out2, contexts=(1024,), seeds=(1, 2),
              chars_per_token=3.0)
    assert a["files_sha256"] == b["files_sha256"]
    assert a["actual_tokenizer_counts_verified"] is False
    assert len(a["files_sha256"]) == 5
    gold = json.loads((out1 / "gold-answers.json").read_text(encoding="utf-8"))
    assert set(gold) == {"kv-1024-seed1.json", "kv-1024-seed2.json"}
    for name, entry in gold.items():
        messages = json.loads((out1 / name).read_text(encoding="utf-8"))
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert entry["expected"] in messages[1]["content"]
        assert entry["symbol"] in messages[1]["content"]
        assert entry["real_model_prompt_tokens_verified"] is False
    with pytest.raises(ValueError, match="overwrite"):
        build(root, out1, contexts=(1024,), seeds=(1,))


def test_long_context_must_have_enough_distinct_source(tmp_path):
    root = project_source(tmp_path)
    with pytest.raises(ValueError, match="Refusing to repeat"):
        build(root, tmp_path / "huge", contexts=(131072,), seeds=(1,),
              chars_per_token=3.0)


def test_canary_positions_and_selected_gold_are_deterministic(tmp_path):
    root = project_source(tmp_path)
    fragments = source_fragments(root)
    for seed in (1, 2, 3):
        messages, expected = make_kv_fixture(fragments, 1024, seed, 3.0)
        content = messages[1]["content"]
        assert expected["symbol"] in content
        assert expected["expected"] in content
        assert content.count("constexpr auto kSm89Gold_") == 3
    assert canary(1024, 1, 0) == canary(1024, 1, 0)


def test_choose_fragments_does_not_repeat_source():
    with pytest.raises(ValueError, match="Refusing to repeat"):
        choose_fragments(["one small distinct source"], seed=1, target=8000)
