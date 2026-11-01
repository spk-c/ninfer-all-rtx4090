"""Exercise CI's immutable baseline selection without GHCR, network or credentials."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/build-inputs.sh"
SCOPES = ROOT / ".github/scripts/change-scopes.py"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash") and shutil.which("git"),
                     "Requires Bash and Git on Linux")
class BuildInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "checkout"
        self.repo.mkdir()
        self.remote = self.root / "origin.git"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.git("init", "--bare", str(self.remote), cwd=self.root)
        self.git("init", "-b", "master")
        self.git("config", "user.name", "CI Tester")
        self.git("config", "user.email", "ci@example.invalid")
        self.git("remote", "add", "origin", str(self.remote))
        scripts = self.repo / ".github/scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(SCRIPT, scripts / SCRIPT.name)
        shutil.copy2(SCOPES, scripts / SCOPES.name)
        (self.repo / "README.md").write_text("initial\n", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-m", "Initial")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()
        (self.bin / "gh").write_text(
            "#!/bin/sh\n"
            'if [ "$FAKE_GH_FAIL" = 1 ]; then exit 1; fi\n'
            'printf "%s\\n" "$FAKE_GH_TAG"\n', encoding="utf-8")
        (self.bin / "gh").chmod(0o755)

    def git(self, *args, cwd=None):
        return subprocess.run(["git", *args], cwd=cwd or self.repo,
                              text=True, capture_output=True, check=True)

    def run_detector(self, fail=False, event="push", ref="refs/heads/master", native=False):
        env = dict(os.environ, GITHUB_REPOSITORY="spk-c/ninfer-all-rtx4090",
                   GITHUB_EVENT_NAME=event, GITHUB_REF=ref,
                   FAKE_GH_FAIL="1" if fail else "0", FAKE_GH_TAG="rtx4090-v1.0.0",
                   PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        command = ["bash", ".github/scripts/build-inputs.sh"]
        if native:
            command.append("--native")
        result = subprocess.run(command, cwd=self.repo, env=env,
                                text=True, capture_output=True, check=True)
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def test_missing_published_release_forces_all_checks_not_failure(self):
        changed = self.run_detector(fail=True)
        self.assertEqual(changed["native"], "true")
        self.assertEqual(changed["changed"], "true")
        self.assertEqual(changed["release_tools"], "true")

    def test_latest_release_tag_detects_changes_since_published_sha(self):
        self.git("tag", "rtx4090-v1.0.0", self.base)
        self.git("push", "origin", "refs/tags/rtx4090-v1.0.0")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "engine.cpp").write_text("// input changed\n")
        self.git("add", ".")
        self.git("commit", "-m", "Native code")
        changed = self.run_detector()
        self.assertEqual(changed["changed"], "true")
        self.assertEqual(changed["native"], "true")
        self.assertEqual(changed["calculator"], "false")

    def test_annotated_release_tag_is_dereferenced(self):
        self.git("tag", "-a", "-m", "Release", "rtx4090-v1.0.0", self.base)
        self.git("push", "origin", "refs/tags/rtx4090-v1.0.0")
        changed = self.run_detector()
        self.assertEqual(changed["native"], "false")
        self.assertEqual(changed["changed"], "false")

    def test_missing_remote_tag_forces_all_checks(self):
        changed = self.run_detector()
        self.assertEqual(changed["changed"], "true")
        self.assertEqual(changed["native"], "true")

    def test_pull_request_uses_first_parent_without_release_lookup(self):
        (self.repo / "src").mkdir()
        (self.repo / "src" / "state.cpp").write_text("int state = 1;\n")
        self.git("add", ".")
        self.git("commit", "-m", "PR patch")
        changed = self.run_detector(fail=True, event="pull_request", ref="refs/pull/4/merge")
        self.assertEqual(changed["native"], "true")
        self.assertEqual(changed["changed"], "true")


if __name__ == "__main__":
    unittest.main()
