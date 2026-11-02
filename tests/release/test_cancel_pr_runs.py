"""Guard PR cancellation against cancelling unrelated push/tag workflow runs."""
from pathlib import Path
import unittest

WORKFLOW = (Path(__file__).resolve().parents[2] /
            ".github/workflows/cancel-superseded.yml")


class CancelPRWorkflowTests(unittest.TestCase):
    def test_only_incomplete_pull_request_runs_are_selected(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('select(.event == "pull_request" and .status != "completed")', text)
        self.assertNotIn('select(.status != "completed")', text)

    def test_no_pull_request_checkout_or_broad_branch_cancellation(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        cancel = text.split("  cancel:", 1)[1]
        self.assertNotIn("actions/checkout@", cancel)
        self.assertIn('head_sha=$SHA', cancel)


if __name__ == "__main__":
    unittest.main()
