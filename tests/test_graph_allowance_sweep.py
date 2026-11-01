"""GPU-free contract tests for the RTX graph allowance measurement runner."""
import json
from types import SimpleNamespace

from tools.bench import run_graph_allowance_sweep as sweep


def test_graph_diagnostic_reads_stdout_and_keeps_dflash_graph_count_unknown(monkeypatch, tmp_path):
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="CUDA Graph used 11.50 MiB\nCUDA Graph allowance 24 MiB\n", stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    measured = sweep.measure(tmp_path / "ninfer.exe", tmp_path / "model.ninfer",
                             8192, "dflash2", 7, "rk4v4", 1024)
    assert measured["started"] and not measured["graph_blocked"]
    assert measured["observed_bytes"] == int(11.5 * sweep.MIB)
    assert measured["allowance_bytes"] == 24 * sweep.MIB
    assert measured["predicted_graphs"] is None
    assert commands[0][-7:] == ["--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft",
                                "--ngram-draft-tokens", "0"]


def test_exact_failure_bytes_take_precedence_over_rounded_graph_line(monkeypatch, tmp_path):
    def run(*args, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stderr=("CUDA Graph used 12.00 MiB\nCUDA Graph allowance 2.00 MiB\n"
                    "CUDA Graph preparation consumed 12582913 bytes, "
                    "exceeding the planned allowance of 2097152 bytes"),
            stdout="",
        )

    monkeypatch.setattr(sweep.subprocess, "run", run)
    measured = sweep.measure(tmp_path / "ninfer.exe", tmp_path / "model.ninfer",
                             2048, None, 3, "int8", 640)
    assert measured["graph_blocked"]
    assert measured["observed_bytes"] == 12582913
    assert measured["allowance_bytes"] == 2097152
    assert measured["predicted_graphs"] == sweep.ordinary_graph_count(2048)


def test_cli_emits_reproducible_sweep_and_handles_speculative_counts(monkeypatch, tmp_path):
    exe = tmp_path / "ninfer.exe"
    model = tmp_path / "qwen.ninfer"
    exe.write_bytes(b"test")
    model.write_bytes(b"test")
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="",
                               stderr="CUDA Graph used 0 B\nCUDA Graph allowance 64 MiB")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    code = sweep.main([
        "--cli", str(exe), "--model", str(model),
        "--spec", "dflash2", "--draft-tokens", "5",
        "--kv-dtype", "rk4v4", "--prefill-chunk", "1024",
        "--capacities", "2048", "8192", "--repeats", "2",
        "--output", str(tmp_path / "results"),
    ])
    assert code == 0
    assert len(calls) == 4
    report = json.loads((tmp_path / "results" /
                         "graph-allowance-dflash2-rk4v4.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == 2
    assert report["config"]["capacities"] == [2048, 8192]
    assert all(r["predicted_graphs"] is None for r in report["runs"])
    assert all(r["observed_bytes"] == 0 for r in report["runs"])


def test_missing_graph_diagnostics_is_incomplete_not_zero(monkeypatch, tmp_path):
    exe = tmp_path / "ninfer.exe"
    model = tmp_path / "qwen.ninfer"
    exe.write_bytes(b"x")
    model.write_bytes(b"x")
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="running", stderr=""))
    result = sweep.main([
        "--cli", str(exe), "--model", str(model),
        "--capacities", "4096", "--repeats", "2", "--output", str(tmp_path / "out"),
    ])
    assert result == 2


def test_failed_startup_with_memory_diagnostic_still_fails(monkeypatch, tmp_path):
    exe, model = tmp_path / "ninfer.exe", tmp_path / "qwen.ninfer"
    exe.write_bytes(b"x")
    model.write_bytes(b"x")
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout="",
                                        stderr="CUDA Graph used 12 MiB\nCUDA Graph allowance 64 MiB\nlater failure"))
    assert sweep.main([
        "--cli", str(exe), "--model", str(model), "--capacities", "8192",
        "--repeats", "2", "--output", str(tmp_path / "out"),
    ]) == 2
    report = json.loads((tmp_path / "out" / "graph-allowance-none-int8.json").read_text())
    assert all(r["observed_bytes"] == 12 * sweep.MIB for r in report["runs"])
    assert all(not r["started"] for r in report["runs"])


def test_gdn_fp16_is_used_and_recorded(monkeypatch, tmp_path):
    exe, model = tmp_path / "ninfer.exe", tmp_path / "qwen.ninfer"
    exe.write_bytes(b"x")
    model.write_bytes(b"x")
    commands = []

    def run(cmd, **kwargs):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="CUDA Graph used 1 MiB\nCUDA Graph allowance 64 MiB", stderr="")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    assert sweep.main([
        "--cli", str(exe), "--model", str(model), "--capacities", "8192",
        "--repeats", "2", "--gdn-state-fp16", "--output", str(tmp_path / "out"),
    ]) == 0
    assert all("--gdn-state-fp16" in cmd for cmd in commands)
    report = json.loads((tmp_path / "out" / "graph-allowance-none-int8.json").read_text())
    assert report["config"]["gdn_state_fp16"] is True


def test_unmeasured_graph_usage_is_not_substituted_with_allowance(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="", stderr=(
                            "summary     CUDA Graph allowance      64 MiB\n"
                            "summary     CUDA Graph used           unmeasured (no free device memory left)\n")))
    result = sweep.measure(tmp_path / "ninfer.exe", tmp_path / "qwen.ninfer",
                           8192, "dflash2", 5, "rk4v4", 1024, True)
    assert result["observed_bytes"] is None
    assert "unmeasured" in result["unparsed_output"]
