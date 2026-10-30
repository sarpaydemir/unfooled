from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_factory.death_census_batch import (
    BatchError,
    FROZEN_CHUNK_SIZE,
    REUSED_V4_TASK_SHA256,
    TOTAL_CANDIDATES,
    run_batches,
)


class DeathCensusBatchTest(unittest.TestCase):
    def candidate_csv(self, root: Path) -> Path:
        path = root / "candidates.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["article_code"])
            writer.writeheader()
            for index in range(TOTAL_CANDIDATES):
                writer.writerow({"article_code": f"{index:03d}"})
        return path

    @staticmethod
    def summary(outcome: str, count: int, *, finish_reason: str = "stop"):
        passed = outcome == "batch_passed"
        return {
            "outcome": outcome,
            "provider_finish_reason": finish_reason,
            "usage": {"usd": 0.01},
            "calibration_audit": {
                "event_count": count,
                "normalized_evidence_pass_count": count if passed else 0,
                "acceptance_valid_count": 6 if passed else 0,
                "semantic_error_count": 0,
                **(
                    {"provider_output_error": "missing"}
                    if finish_reason == "length"
                    else {}
                ),
            },
        }

    def test_all_offsets_run_with_frozen_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []

            def classifier(*_args, **kwargs):
                calls.append(kwargs)
                return self.summary("batch_passed", kwargs["article_limit"])

            accepted = run_batches(
                self.candidate_csv(root),
                root / "corpus",
                root / "output",
                root / "usage.jsonl",
                model="model",
                api_key="key",
                classifier=classifier,
                emit=lambda *_args, **_kwargs: None,
            )
            self.assertEqual(len(accepted), 32)
            self.assertEqual(len(calls), 32)
            self.assertTrue(all(call["article_limit"] == 5 for call in calls[:-1]))
            self.assertEqual(calls[-1]["article_limit"], 4)
            self.assertEqual(FROZEN_CHUNK_SIZE, 5)

    def test_length_increases_tokens_and_quality_retries_only_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outcomes = [
                self.summary("batch_quality_failed", 5, finish_reason="length"),
                self.summary("batch_quality_failed", 5),
                self.summary("batch_passed", 5),
            ]
            calls = []

            def classifier(*_args, **kwargs):
                calls.append(kwargs)
                if outcomes:
                    return outcomes.pop(0)
                return self.summary("batch_passed", kwargs["article_limit"])

            run_batches(
                self.candidate_csv(root),
                root / "corpus",
                root / "output",
                root / "usage.jsonl",
                model="model",
                api_key="key",
                classifier=classifier,
                emit=lambda *_args, **_kwargs: None,
            )
            self.assertEqual([call["max_tokens"] for call in calls[:3]], [16000, 32000, 32000])
            self.assertEqual([call["retry_generation"] for call in calls[:3]], [0, 1, 2])

    def test_reuses_authorized_first_90_and_resumes_offset_90_with_new_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            for offset in range(0, 90, 5):
                path = output / f"chunk-{offset:03d}" / "attempt-0-max-16000"
                path.mkdir(parents=True)
                (path / "batch-summary.json").write_text(
                    json.dumps(
                        {
                            "outcome": "batch_passed",
                            "task_sha256": REUSED_V4_TASK_SHA256,
                        }
                    ),
                    encoding="utf-8",
                )
            for generation in (0, 1):
                (output / "chunk-090" / f"attempt-{generation}-max-16000").mkdir(
                    parents=True
                )
            calls = []

            def classifier(*_args, **kwargs):
                calls.append(kwargs)
                return self.summary("batch_passed", kwargs["article_limit"])

            accepted = run_batches(
                self.candidate_csv(root),
                root / "corpus",
                output,
                root / "usage.jsonl",
                model="model",
                api_key="key",
                classifier=classifier,
                emit=lambda *_args, **_kwargs: None,
            )
            self.assertEqual(len(accepted), 32)
            self.assertEqual(calls[0]["article_offset"], 90)
            self.assertEqual(calls[0]["retry_generation"], 2)
            self.assertEqual(len(calls), 14)

    def test_semantic_conflict_stops(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(BatchError, "semantic conflict"):
                run_batches(
                    self.candidate_csv(root),
                    root / "corpus",
                    root / "output",
                    root / "usage.jsonl",
                    model="model",
                    api_key="key",
                    classifier=lambda *_args, **_kwargs: self.summary(
                        "batch_semantic_conflict", 5
                    ),
                    emit=lambda *_args, **_kwargs: None,
                )


if __name__ == "__main__":
    unittest.main()
