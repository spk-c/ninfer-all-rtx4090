"""CPU-only tests: Nsight plans never count as captures or kernel wins."""
import json
from pathlib import Path
import subprocess

import pytest

from tools.bench.plan_rtx4090_sm89 import BASE_SHA
from tools.bench import capture_rtx4090_sm89_nsight as profiler


def _sample_plan(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema": "ninfer_rtx4090_sm89_ab_plan_v2",
        "baseline_commit_sha": BASE_SHA,
        "candidate_commit_sha": "d" * 40,
    }), encoding="utf-8")
    return plan


def _paths(tmp_path):
    a = tmp_path / "master" / "bench" / "ninfer_benches.exe"
    b = tmp_path / "candidate" / "bench" / "ninfer_benches.exe"
    for path in (a, b):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test-bench")
    return a, b


def test_dry_run_creates_12_unexecuted_commands_without_gpu(tmp_path):
    plan = _sample_plan(tmp_path)
    a, b = _paths(tmp_path)
    out = tmp_path / "profiles"
    def no_gpu(*args, **kwargs):
        raise AssertionError("dry plan must not touch GPU")
    result = profiler.capture(
        out, plan, a, b, tmp_path / "master", tmp_path / "candidate",
        "both", False, 2048, 30, probe=no_gpu, runner=no_gpu)
    assert result["status"] == "NOT_RUN"
    assert result["gpu_verified"] is False
    assert len(result["profiles"]) == 12
    assert {x["tool"] for x in result["profiles"]} == {"nsys", "ncu"}
    assert {x["kernel_family"] for x in result["profiles"]} == {
        "q4-small-t", "q5-small-t", "w4a8"}
    assert all(x["status"] == "NOT_RUN" for x in result["profiles"])
    assert json.loads((out / "manifest.json").read_text())["status"] == "NOT_RUN"
    with pytest.raises(ValueError, match="never overwrite"):
        profiler.capture(out, plan, a, b, tmp_path, tmp_path, "both",
                         False, 2048, 30)


def test_ncu_and_nsys_commands_use_distinct_original_builds(tmp_path):
    a, b = _paths(tmp_path)
    runs = profiler.profile_plan(a, b, tmp_path / "out", mode="both")
    assert len(runs) == 12
    for row in runs:
        exe = a if row["arm"] == "A" else b
        assert str(exe) in row["argv"]
        assert "-o" in row["argv"]
        assert "--set" in row["argv"] if row["tool"] == "ncu" else "--trace=cuda,nvtx" in row["argv"]
    with pytest.raises(ValueError, match="separate"):
        profiler.profile_plan(a, a, tmp_path, "nsys")
    with pytest.raises(ValueError, match="mode"):
        profiler.profile_plan(a, b, tmp_path, "unknown")


def test_capture_must_have_real_nonempty_export(tmp_path):
    a, b = _paths(tmp_path)
    paths = profiler.profile_plan(a, b, tmp_path / "prof", "both")[:2]
    def mock_runner(argv, timeout):
        prefix = Path(argv[argv.index("-o") + 1])
        ext = ".nsys-rep" if argv[0] == "nsys" else ".ncu-rep"
        Path(str(prefix) + ext).write_bytes(b"realistic-rep-payload")
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
    out = profiler.execute_plan(paths, 30, runner=mock_runner)
    assert all(x["status"] == "captured" for x in out)
    assert all(x["report_sha256"] for x in out)
    assert all(Path(x["report_path"]).is_file() for x in out)
    missing = profiler.profile_plan(a, b, tmp_path / "missing", "ncu")[:1]
    def no_report(argv, timeout):
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")
    result = profiler.execute_plan(missing, 30, runner=no_report)
    assert result[0]["status"] == "MISSING_REPORT"
    assert not result[0]["report_sha256"]


def test_missing_gpu_is_a_fatal_preflight_without_created_output(tmp_path):
    plan = _sample_plan(tmp_path)
    a, b = _paths(tmp_path)
    monkey = lambda source: BASE_SHA if source == tmp_path / "master" else "d" * 40
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(profiler, "commit_sha", monkey)
        patch.setattr(profiler, "source_is_clean", lambda source: True)
        with pytest.raises(ValueError, match="GPU"):
            profiler.capture(
                tmp_path / "result", plan, a, b, tmp_path / "master",
                tmp_path / "candidate", "ncu", True, 2048, 30,
                probe=lambda device: {
                    "gpu_index": 0, "compute_cap": "8.6",
                    "gpu_name": "NVIDIA GeForce RTX 3090",
                    "used_mib": 0, "util_pct": 0,
                },
            )
    assert not (tmp_path / "result").exists()
