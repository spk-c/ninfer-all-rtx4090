"""Offline coverage for the routine/full hosted CI allocation policy."""
import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "ci_matrix", ROOT / ".github/scripts/ci-matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


class CiMatrixTests(unittest.TestCase):
    def test_routine_checks_both_toolkits_for_sm89(self):
        for event, ref in [
            ("pull_request", "refs/pull/42/merge"),
            ("push", "refs/heads/master"),
            ("push", "refs/heads/ci/rtx4090-shard-test/example"),
            ("push", "refs/heads/ci/fullness/example"),
        ]:
            with self.subTest(event=event, ref=ref):
                matrix, shards = MATRIX.select_matrix(event, ref)
                self.assertEqual(matrix["include"], [
                    {"cuda": "12.9.2", "archs": "89"},
                    {"cuda": "13.4.3", "archs": "89"},
                ])
                self.assertEqual(shards, [1])

    def test_full_runs_keep_all_ten_cuda_pairs_and_three_windows_shards(self):
        expected = {
            ("12.9.2", arch) for arch in ("70", "75", "80", "86", "89")
        } | {
            ("13.4.3", arch) for arch in ("75", "80", "86", "89", "120a")
        }
        for event, ref in [
            ("workflow_dispatch", "refs/heads/master"),
            ("schedule", "refs/heads/master"),
            ("push", "refs/heads/ci/full/check"),
        ]:
            with self.subTest(event=event, ref=ref):
                matrix, shards = MATRIX.select_matrix(event, ref)
                rows = matrix["include"]
                self.assertEqual(len(rows), 10)
                self.assertEqual({(r["cuda"], r["archs"]) for r in rows}, expected)
                self.assertEqual(shards, [1, 2, 3])

    def test_workflow_uses_matrix_and_shard_count(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn("fromJSON(needs.build-inputs.outputs.cuda_matrix)", workflow)
        self.assertIn("fromJSON(needs.build-inputs.outputs.windows_shards)", workflow)
        self.assertIn("matrix.shard }}/${{ needs.build-inputs.outputs.windows_shard_count }}",
                      workflow)
        self.assertIn("max-parallel: 4", workflow)
        self.assertIn("max-parallel: 2", workflow)
        self.assertIn("schedule:", workflow)
        self.assertIn("refs/heads/ci/full/", workflow)
        self.assertIn("github.event_name == 'schedule'", workflow)
