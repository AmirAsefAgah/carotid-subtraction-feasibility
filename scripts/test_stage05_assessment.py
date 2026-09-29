"""Check that assessment cannot create or promote a review decision."""

import json
import tempfile
import unittest
from pathlib import Path

from stage05_assessment import load_review


class AssessmentReviewTests(unittest.TestCase):
    def test_missing_review_is_not_created(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            with self.assertRaises(FileNotFoundError):
                load_review(path)
            self.assertFalse(path.exists())

    def test_incomplete_or_rejected_review_blocks_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            for decision in [None, "", "pending", "rejected"]:
                with self.subTest(decision=decision):
                    path.write_text(
                        json.dumps({"decision": decision}), encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        load_review(path)

    def test_recorded_review_is_preserved(self):
        review = {
            "decision": "acceptable qualitative visual review",
            "scope": "sampled signal",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps(review), encoding="utf-8")
            before = path.read_bytes()
            self.assertEqual(load_review(path), review)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
