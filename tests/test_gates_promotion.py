import unittest

from research_factory.gates import GateArtifactError, GateEvaluator


class GateTests(unittest.TestCase):
    def test_specification_review_requires_every_semantic_check(self) -> None:
        code_hash = "a" * 64
        metrics = {
            "reviewed_code_sha256": code_hash,
            "hypothesis_code_consistent": True,
            "indicator_semantics_verified": False,
            "timeframe_semantics_verified": True,
            "entry_timing_consistent": True,
            "materiality_basis": "confirmed_signal_mismatch",
            "review_notes": "Aroon direction is inverted.",
        }
        result = GateEvaluator(expected_code_sha256=code_hash).evaluate(
            "specification_review", metrics
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.reasons, ("indicator_semantics_verified is false",))
        metrics["indicator_semantics_verified"] = True
        metrics["materiality_basis"] = "pass"
        self.assertTrue(
            GateEvaluator(expected_code_sha256=code_hash)
            .evaluate("specification_review", metrics)
            .passed
        )

    def test_specification_review_is_bound_to_candidate_code(self) -> None:
        metrics = {
            "reviewed_code_sha256": "a" * 64,
            "hypothesis_code_consistent": True,
            "indicator_semantics_verified": True,
            "timeframe_semantics_verified": True,
            "entry_timing_consistent": True,
            "materiality_basis": "pass",
            "review_notes": "Independent semantic review passed.",
        }
        result = GateEvaluator(expected_code_sha256="b" * 64).evaluate(
            "specification_review", metrics
        )
        self.assertFalse(result.passed)
        self.assertIn("does not match", result.reasons[0])

    def test_specification_review_rejects_missing_or_empty_evidence(self) -> None:
        with self.assertRaises(GateArtifactError):
            GateEvaluator().evaluate(
                "specification_review",
                {
                    "hypothesis_code_consistent": True,
                    "reviewed_code_sha256": "a" * 64,
                    "indicator_semantics_verified": True,
                    "timeframe_semantics_verified": True,
                    "entry_timing_consistent": True,
                    "materiality_basis": "pass",
                    "review_notes": "",
                },
            )

    def test_specification_review_rejects_contradictory_materiality_basis(self) -> None:
        metrics = {
            "reviewed_code_sha256": "a" * 64,
            "hypothesis_code_consistent": True,
            "indicator_semantics_verified": True,
            "timeframe_semantics_verified": True,
            "entry_timing_consistent": True,
            "materiality_basis": "confirmed_signal_mismatch",
            "review_notes": "No semantic check is false.",
        }
        result = GateEvaluator(expected_code_sha256="a" * 64).evaluate(
            "specification_review", metrics
        )
        self.assertFalse(result.passed)
        self.assertIn("must be pass", result.reasons[0])

    def test_event_study_gate_explains_failure(self) -> None:
        result = GateEvaluator().evaluate(
            "event_study",
            {
                "effective_events": 59,
                "median_return_pct": -0.1,
                "mean_return_pct": 0.2,
                "positive_pair_fraction": 0.7,
                "positive_half_fraction": 1.0,
                "mfe_mae_ratio": 1.3,
            },
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.reasons), 2)

    def test_final_gate_enforces_sample_and_fee_stress(self) -> None:
        result = GateEvaluator().evaluate(
            "final_locked",
            {
                "trades": 100,
                "total_profit_pct": 10,
                "median_profit_pct": 0.2,
                "profit_factor": 1.3,
                "max_drawdown_pct": 5,
                "positive_pair_fraction": 0.8,
                "fee_stress_total_profit_pct": -0.1,
                "fee_stress_profit_factor": 1.04,
                "positive_neighborhood_fraction": 0.8,
                "combined_cluster_ci_lower_pct": 0.01,
                "combined_effective_samples": 299,
            },
        )
        self.assertFalse(result.passed)
        self.assertEqual(len(result.reasons), 3)

    def test_missing_metric_cannot_pass(self) -> None:
        with self.assertRaises(GateArtifactError):
            GateEvaluator().evaluate("ablation", {})

    def test_statistics_uses_preregistered_campaign_threshold(self) -> None:
        metrics = {
            "cluster_ci_lower_pct": 0.1,
            "family_adjusted_p_value": 0.002,
            "effective_clusters": 100,
        }
        self.assertFalse(GateEvaluator(alpha_threshold=0.05 / 180).evaluate("statistics", metrics).passed)
        metrics["family_adjusted_p_value"] = 0.0002
        self.assertTrue(GateEvaluator(alpha_threshold=0.05 / 180).evaluate("statistics", metrics).passed)


if __name__ == "__main__":
    unittest.main()
