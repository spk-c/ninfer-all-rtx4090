"""CPU-only functional checks for the RTX 4090 qualification campaign."""
import json
from tools.bench.run_rtx4090_optimization_campaign import (
    Case, cases, command, main, order_for_round, parse_measurements, summarize,
)


def test_matrix_all_modes_and_unique_names():
    matrix = cases(("draft", "prefill", "kv", "copy"))
    assert len(matrix) == 25
    assert len({x.label for x in matrix}) == len(matrix)
    assert {x.group for x in matrix} == {"draft", "prefill", "kv", "copy"}


def test_argument_building_and_ngram_isolation(tmp_path):
    kv = Case("kv", "rk8v4-cap65536", ("--kv-dtype", "rk8v4"), context=65536)
    args = command(tmp_path / "cli", tmp_path / "model", kv, tmp_path / "msg", 0)
    assert args.count("--kv-dtype") == 1
    assert args[args.index("--kv-dtype")+1] == "rk8v4"
    assert args[args.index("--max-context")+1] == "65536"
    assert "--greedy" in args and "--no-thinking" in args
    k3 = command(tmp_path / "cli", tmp_path / "model",
                 cases(("draft",))[0], tmp_path / "msg", 0)
    assert k3[k3.index("--ngram-draft-tokens")+1] == "0"


def test_parser_and_missing_fields():
    values = parse_measurements("prefill speed 3000.25 tok/s\n"
                                "decode speed 132.50 tok/s\n"
                                "generated tokens 192\n"
                                "dflash2 acceptance rate 76.2%\n"
                                "dflash2 acceptance length 5.4\n")
    assert values["decode_tok_s"] == 132.5
    assert values["prefill_tok_s"] == 3000.25
    assert values["acceptance_pct"] == 76.2
    assert values["tokens_per_round"] == 5.4
    assert parse_measurements("decode speed NaN tok/s")["decode_tok_s"] is None


def test_rotation_and_full_sample_requirement():
    c = cases(("copy",))
    assert order_for_round(c, 0) != order_for_round(c, 1)
    good = {"case": "ngram0", "status": "ok",
            "decode_tok_s": 150.0, "generated_tokens": 192,
            "output_sha256": "abcd"}
    only = [c[0]]
    assert summarize([good] * 3, only, 3)["ngram0"]["median"] == 150
    assert not summarize([good] * 2, only, 3)["ngram0"]["valid"]
    short = [{**good, "generated_tokens": 1}] * 3
    assert not summarize(short, only, 3)["ngram0"]["valid"]


def test_dry_run_uses_no_model_or_gpu(tmp_path):
    out = tmp_path / "run"
    assert main(["--model", "missing.ninfer", "--cli", "missing.exe",
                 "--out", str(out), "--groups", "copy", "--rounds", "3",
                 "--dry-run"]) == 0
    report = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert report["gpu"] == {"qualified": False}
    assert len(list(out.glob("*.command.json"))) == 6
    assert (out / "summary.json").exists()
