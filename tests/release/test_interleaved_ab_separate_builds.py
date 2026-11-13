"""Guard the safe separate-build option for interleaved NInfer benchmarks."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "tools/bench/run_interleaved_ab.py"
SPEC = importlib.util.spec_from_file_location("interleaved_ab_under_test", SOURCE)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class SeparateBuildPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.upstream = base / "upstream" / "apps" / "ninfer.exe"
        self.fork = base / "fork" / "apps" / "ninfer.exe"
        for path in (self.upstream, self.fork):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"dummy executable")

    def test_different_directories_rejected_by_default(self):
        with patch.object(RUNNER.subprocess, "run") as run:
            with self.assertRaises(SystemExit):
                RUNNER.preflight(self.upstream, self.fork)
            run.assert_not_called()

    def test_separate_builds_are_probed_in_their_own_apps_directories(self):
        completed = subprocess.CompletedProcess([], 0, stdout="ninfer version", stderr="")
        with patch.object(RUNNER.subprocess, "run", return_value=completed) as run:
            RUNNER.preflight(self.upstream, self.fork, allow_separate_dirs=True)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0], [str(self.upstream.resolve()), "--version"])
        self.assertEqual(run.call_args_list[1].args[0], [str(self.fork.resolve()), "--version"])

    def test_separate_msvc_release_trees_are_supported(self):
        base = Path(self.temp.name)
        master = base / "master-msvc" / "apps" / "Release" / "ninfer.exe"
        candidate = base / "candidate-msvc" / "apps" / "Release" / "ninfer.exe"
        for path in (master, candidate):
            path.parent.mkdir(parents=True)
            path.write_bytes(b"dummy executable")
        completed = subprocess.CompletedProcess([], 0, stdout="ninfer version", stderr="")
        with patch.object(RUNNER.subprocess, "run", return_value=completed) as run:
            RUNNER.preflight(master, candidate, allow_separate_dirs=True)
        self.assertEqual(run.call_count, 2)

    def test_arbitrary_nested_build_subdirectory_is_rejected(self):
        base = Path(self.temp.name)
        invalid = base / "foreign" / "apps" / "other" / "ninfer.exe"
        invalid.parent.mkdir(parents=True)
        invalid.write_bytes(b"dummy executable")
        with patch.object(RUNNER.subprocess, "run") as run:
            with self.assertRaisesRegex(SystemExit, "apps directory"):
                RUNNER.preflight(invalid, self.fork, allow_separate_dirs=True)
            run.assert_not_called()

    def test_missing_or_broken_dependencies_abort_comparison(self):
        completed = subprocess.CompletedProcess([], 127, stdout="", stderr="DLL not found")
        with patch.object(RUNNER.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(SystemExit, "repair that build"):
                RUNNER.preflight(self.upstream, self.fork, allow_separate_dirs=True)

    def test_non_apps_directory_is_not_allowed(self):
        outsider = Path(self.temp.name) / "outside" / "ninfer.exe"
        outsider.parent.mkdir()
        outsider.write_bytes(b"dummy")
        with patch.object(RUNNER.subprocess, "run") as run:
            with self.assertRaisesRegex(SystemExit, "apps directory"):
                RUNNER.preflight(outsider, self.fork, allow_separate_dirs=True)
            run.assert_not_called()

    def test_identical_executables_refused_even_with_override(self):
        with self.assertRaisesRegex(SystemExit, "same file"):
            RUNNER.preflight(self.upstream, self.upstream, allow_separate_dirs=True)


if __name__ == "__main__":
    unittest.main()
