from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from research_factory.causality import prefix_causality_metrics
from research_factory.gates import GateEvaluator


CAUSAL = """
def signal(dataframe):
    prior_mean = dataframe['close'].rolling(12).mean().shift(1)
    return (dataframe['close'] > prior_mean).fillna(False)
"""

FUTURE_DEPENDENT = """
def signal(dataframe):
    full_frame_mean = dataframe['close'].mean()
    return (dataframe['close'] > full_frame_mean).fillna(False)
"""


class CausalityTests(unittest.TestCase):
    def metrics(self, source: str) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.py"
            manifest = Path(directory) / "manifest.json"
            path.write_text(source, encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "candidate_code_sha256": sha256(source.encode()).hexdigest(),
                        "hypothesis": {
                            "hypothesis_id": "causality-fixture",
                            "mechanism_ids": ["fixture"],
                            "title": "Causality fixture",
                            "rationale": "Synthetic prefix-only validation.",
                            "expected_regimes": ["fixture"],
                            "entry_logic": ["Synthetic signal."],
                            "exit_neutral_horizons": [48],
                            "controls": ["No market data."],
                            "failure_conditions": ["Any prefix mismatch."],
                            "required_timeframes": ["1h"],
                            "schema_version": 2,
                            "indicator_contracts": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            return prefix_causality_metrics(path, manifest)

    def test_causal_signal_is_prefix_invariant(self) -> None:
        metrics = self.metrics(CAUSAL)
        self.assertEqual(metrics["prefix_checks"], 12)
        self.assertEqual(metrics["mismatched_prefixes"], 0)
        self.assertEqual(metrics["execution_errors"], 0)

    def test_full_frame_mean_is_detected_as_future_dependent(self) -> None:
        metrics = self.metrics(FUTURE_DEPENDENT)
        self.assertEqual(metrics["prefix_checks"], 12)
        self.assertGreater(metrics["mismatched_prefixes"], 0)
        self.assertGreater(metrics["mismatched_rows"], 0)

    def test_gate_binds_probe_to_candidate_hash(self) -> None:
        metrics = self.metrics(CAUSAL)
        self.assertTrue(
            GateEvaluator(expected_code_sha256=metrics["candidate_code_sha256"])
            .evaluate("causality_probe", metrics)
            .passed
        )
        result = GateEvaluator(expected_code_sha256="0" * 64).evaluate(
            "causality_probe", metrics
        )
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
