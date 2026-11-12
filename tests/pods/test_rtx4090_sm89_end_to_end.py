"""CPU-only simulated end-to-end rehearsal of all SM89 campaign case flows.

All inference and GPU telemetry in this test are synthetic fixtures. This proves
bookkeeping and CLI contract only, never validates throughput on an RTX 4090.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from tools.bench.plan_rtx4090_sm89 import (
    build_cases, command_tokens, sha256_file, binary_fingerprint, BASE_SHA,
)
from tools.bench import run_rtx4090_sm89_evidence as collector
from tools.bench import report_rtx4090_sm89_evidence as reporter


def make_fake_strict_manifest(tmp_path):
    base = tmp_path / "master" / "apps" / "Release" / "ninfer.exe"
    cand = tmp_path / "candidate" / "apps" / "Release" / "ninfer.exe"
    for path, contents in ((base, b"frozen binary A"), (cand, b"frozen binary B")):
        path.parent.mkdir(parents=True)
        path.write_bytes(contents)
        (path.parent / "ninfer-kernels.dll").write_bytes(b"same DLL version")
    model = tmp_path / "weights.ninfer"
    model.write_bytes(b"fake-model-bytes-not-a-real-model")
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    cases = [asdict(c) for c in build_cases()]
    names = sorted({c["fixture"] for c in cases})
    for name in names:
        (fixtures / name).write_text(
            json.dumps([{"role": "user", "content": "synthetic smoke only"}]),
            encoding="utf-8")
    gold = {
        c["fixture"]: {"expected": "SIMULATED_" + c["fixture"]}
        for c in cases if c["family"] == "kv"
    }
    gold_file = fixtures / "gold-answers.json"
    gold_file.write_text(json.dumps(gold), encoding="utf-8")
    manifest = {
        "schema": "ninfer_rtx4090_sm89_ab_plan_v2",
        "baseline_commit_sha": BASE_SHA,
        "candidate_commit_sha": "c" * 40,
        "master_executable": str(base),
        "candidate_executable": str(cand),
        "model": str(model),
        "model_sha256": sha256_file(model),
        "binary_fingerprints": {"A": binary_fingerprint(base),
                                "B": binary_fingerprint(cand)},
        "fixture_sha256": {name: sha256_file(fixtures / name) for name in names},
        "gold_answers_sha256": sha256_file(gold_file),
        "fixtures_dir": str(fixtures),
        "repetitions": 5,
        "inputs_verified_at_plan": True,
    }
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (plan_dir / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    return plan_dir, manifest, cases, base, cand, gold_file


def fake_executor(argv, timeout, device):
    assert argv[argv.index("--seed") + 1] == "4090"
    assert argv[argv.index("--device") + 1] == "0"
    assert "--greedy" in argv and "--no-thinking" in argv
    assert "--ngram-draft-tokens" in argv
    kv = argv[argv.index("--kv-dtype") + 1]
    dtype = {
        "rk4v4": "rotated-lloyd4g64-v4g32",
        "rk8v4": "rotated-k8g64-v4g32",
    }[kv]
    ctx = int(argv[argv.index("--max-context") + 1])
    max_new = int(argv[argv.index("--max-new") + 1])
    file = Path(argv[argv.index("--messages") + 1]).name
    is_kv = file.startswith("kv-")
    is_spec = argv[argv.index("--spec") + 1] == "dflash2"
    arm = "B" if "candidate" in argv[0] else "A"
    prompt_tokens = int(ctx * 0.92) if is_kv else 1900
    if prompt_tokens + max_new > ctx:
        raise AssertionError("synthetic corpus accidentally exceeds context")
    gold = "SIMULATED_" + file if is_kv else "synthetic non-KV code response"
    speed = 111.0 if arm == "A" else 115.0
    stderr = (
        f"summary     prompt tokens             {prompt_tokens}\n"
        f"summary     generated tokens          {max_new}\n"
        f"summary     prefill speed             {speed:.2f} tok/s\n"
        f"summary     decode speed              {speed:.2f} tok/s\n"
        f"summary     finish reason             output-limit\n"
        f"summary     max context               {ctx}\n"
        f"summary     kv cache dtype            {dtype}\n"
        f"summary     gpu workspace peak        200 MiB / 500 MiB\n"
    )
    if is_spec:
        stderr += (
            "summary     dflash2 acceptance rate   65.0%\n"
            "summary     dflash2 acceptance length 2.6 tok/round\n"
        )
    telemetry = {
        "gpu_initial_used_mib": 800.0,
        "gpu_peak_used_mib": 12000.0,
        "gpu_peak_util_pct": 95.0,
        "gpu_peak_temp_c": 65.0,
        "gpu_peak_power_w": 280.0,
        "gpu_min_sm_clock_mhz": 2100.0,
        "gpu_max_sm_clock_mhz": 2700.0,
        "gpu_min_mem_clock_mhz": 10500.0,
        "gpu_max_mem_clock_mhz": 10500.0,
    }
    return 0, gold, stderr, 0.1, telemetry, ""


def test_collector_cli_matches_generator_for_all_54_cases(tmp_path):
    model, fixtures = Path("model.ninfer"), tmp_path / "fixtures"
    arms = {"A": Path("master/apps/ninfer.exe"), "B": Path("candidate/apps/ninfer.exe")}
    manifest = {"model": str(model), "fixtures_dir": str(fixtures)}
    for case in build_cases():
        for arm in ("A", "B"):
            actual = collector.command_for_case(asdict(case), arm, manifest, arms)
            expected = command_tokens(case, model, fixtures)
            assert actual == [str(arms[arm]), *expected[1:]]


def test_all_54_cases_complete_540_measurements_and_108_warmups(tmp_path, monkeypatch):
    plan_dir, manifest, cases, master, candidate, gold = make_fake_strict_manifest(tmp_path)
    def check_plan(*args):
        assert args[0] == plan_dir
        return manifest, cases, {"A": master, "B": candidate}
    monkeypatch.setattr(collector, "validate_plan", check_plan)
    out = tmp_path / "synthetic-execution"
    status = collector.run_campaign(
        plan_dir, out, tmp_path / "master", tmp_path / "candidate", gold,
        reps=5, max_cases=0, idle_limit=2048, timeout=3,
        gpu_probe=lambda index: {"gpu_index": index, "used_mib": 0, "util_pct": 0},
        executor=fake_executor,
    )
    assert status == 0
    saved = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert saved["status"] == "COMPLETE"
    assert saved["warmup_count"] == 108
    assert saved["warmup_failures"] == 0
    assert saved["total_samples"] == 540
    assert saved["successful_samples"] == 540
    assert saved["artifact_integrity"] == "UNCHANGED"
    assert len(list((out / "raw" / "warmup").glob("*.stdout.txt"))) == 108
    assert len(list((out / "raw").glob("*.stdout.txt"))) == 540
    report_dir = tmp_path / "synthetic-report"
    report = reporter.write_report(out, report_dir)
    assert report["complete"]
    assert report["recorded_samples"] == 540
    assert report["required_samples"] == 540
    assert len(report["cases"]) == 54
    assert all(c["valid"] and c["pairs"] == 5 for c in report["cases"])
    assert all(c["quality_passes"] == 10 for c in report["cases"]
               if c["family"] == "kv")
    # No GPU hardware was used or characterized by this rehearsal.
    assert report["autopromote"] is False
    assert report["quality_signoff"] == "PENDING_INDEPENDENT_REVIEW"
    assert report["nsight_qualification"] == "PENDING"


def test_mid_campaign_mutation_is_not_labeled_partial(tmp_path, monkeypatch):
    plan_dir, manifest, cases, master, candidate, gold = make_fake_strict_manifest(tmp_path)
    monkeypatch.setattr(
        collector, "validate_plan",
        lambda *args: (manifest, cases, {"A": master, "B": candidate}),
    )
    changed = False
    def mutate_during_measurement(argv, timeout, device):
        nonlocal changed
        result = fake_executor(argv, timeout, device)
        if not changed:
            (Path(manifest["fixtures_dir"]) / "coding-short.json").write_text("tampered")
            changed = True
        return result
    out = tmp_path / "tampered-execution"
    result = collector.run_campaign(
        plan_dir, out, tmp_path / "master", tmp_path / "candidate", gold,
        reps=5, max_cases=1, idle_limit=2048, timeout=3,
        gpu_probe=lambda index: {"gpu_index": index, "used_mib": 0, "util_pct": 0},
        executor=mutate_during_measurement,
    )
    meta = json.loads((out / "metadata.json").read_text())
    assert result == 2
    assert meta["status"] == "INVALID_ARTIFACT"
    assert "fixture changed" in meta["artifact_integrity"]
    assert meta["total_samples"] == 10
    assert meta["warmup_count"] == 2
    report = reporter.write_report(out, tmp_path / "unqualified-report")
    assert not report["complete"]
    assert not report["autopromote"]
