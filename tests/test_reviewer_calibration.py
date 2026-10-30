from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from research_factory.models import canonical_json
from research_factory.providers import AIResponse, StaticProvider
from research_factory.reviewer_benchmark_cases import benchmark_cases, benchmark_suite
from research_factory.reviewer_calibration import run_suite, summarize_results
from research_factory.storage import ResearchStore


class HashAwareProvider(StaticProvider):
    def __init__(self) -> None:
        super().__init__({}, model="z-ai/glm-5.2", provider="static")
        self.temperature = 0.0

    def generate(self, prompt: str, schema: dict) -> AIResponse:
        marker = "Candidate code SHA256: "
        code_hash = prompt.split(marker, 1)[1].splitlines()[0]
        payload = {
            "reviewed_code_sha256": code_hash,
            "hypothesis_code_consistent": True,
            "indicator_semantics_verified": True,
            "timeframe_semantics_verified": True,
            "entry_timing_consistent": True,
            "materiality_basis": "pass",
            "review_notes": "Synthetic passing review.",
            "findings": [],
        }
        return AIResponse(payload, "static", self.model, 10, 5, 0.0, "request-1")


class ReviewerCalibrationTests(unittest.TestCase):
    def test_frozen_suite_has_ten_matches_and_ten_material_mismatches(self) -> None:
        cases = benchmark_cases()
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(case["expected"] == "pass" for case in cases), 10)
        self.assertEqual(sum(case["expected"] == "reject" for case in cases), 10)
        self.assertIn("P10-RIGHT-LABEL-GAPS", {case["case_id"] for case in cases})
        suite = benchmark_suite()
        material = dict(suite)
        expected_hash = material.pop("suite_sha256")
        self.assertEqual(expected_hash, sha256(canonical_json(material).encode()).hexdigest())

    def test_summary_distinguishes_false_reject_from_false_pass(self) -> None:
        rows = [
            {"case_id": "P", "expected": "pass", "predicted_pass": False},
            {"case_id": "N", "expected": "reject", "predicted_pass": True},
        ]
        summary = summarize_results(rows)
        self.assertEqual(summary["false_reject_cases"], ["P"])
        self.assertEqual(summary["false_pass_cases"], ["N"])
        self.assertFalse(summary["accepted_for_hard_gate"])

    def test_runner_is_idempotent_and_does_not_create_candidates_or_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ResearchStore(root / "research.db", writable=True)
            full = benchmark_suite()
            small = {
                **full,
                "cases": [full["cases"][0]],
            }
            small.pop("suite_sha256")
            small["suite_sha256"] = sha256(canonical_json(small).encode()).hexdigest()
            provider = HashAwareProvider()
            first = run_suite(small, store, root / "output", provider)
            second = run_suite(small, store, root / "output", provider)
            self.assertEqual(first, second)
            self.assertEqual(store.candidate_count(), 0)
            self.assertEqual(store.monthly_ai_cost(), 0.0)
            self.assertEqual(store.list_jobs()[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
