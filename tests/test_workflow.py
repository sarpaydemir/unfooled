from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_factory.models import (
    CandidateManifest,
    CandidateStatus,
    GateResult,
    Hypothesis,
    canonical_json,
)
from research_factory.providers import AIResponse, StaticProvider
from research_factory.safety import CandidateSafetyValidator
from research_factory.static_validation import CompositeStaticValidator
from research_factory.storage import ResearchStore
from research_factory.talib_metadata import (
    EXPECTED_C_LIBRARY_VERSION,
    EXPECTED_METADATA_SHA256,
    EXPECTED_PYTHON_WRAPPER_VERSION,
    PINNED_WORKER_IMAGE,
)
from research_factory.validation import (
    PayloadValidationError,
    hypothesis_from_stored_payload,
    parse_candidate_payload,
)
from research_factory.workflow import REQUIRED_PROMOTION_GATES, ResearchWorkflow


SOURCE = """
def entry_reclaim_1h(dataframe):
    return (dataframe['willr'] < -80) & (dataframe['cmf'] > 0)
"""


class CountingProvider(StaticProvider):
    def __init__(self, payload: dict) -> None:
        super().__init__(payload)
        self.calls = 0

    def generate(self, prompt: str, schema: dict) -> AIResponse:
        self.calls += 1
        return super().generate(prompt, schema)


class WorkflowTests(unittest.TestCase):
    def setup_workflow(self, directory: str) -> tuple[ResearchWorkflow, Path]:
        root = Path(directory)
        source = root / "source.py"
        source.write_text(SOURCE)
        store = ResearchStore(root / "state.db", writable=True)
        return ResearchWorkflow(store, root / "artifacts"), source

    def write_fixture_attestation(
        self,
        workflow: ResearchWorkflow,
        key: str = "test-generation",
    ) -> Path:
        path = workflow.artifact_root.parent / "preflight" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "idempotency_key": key,
            "worker_image_digest": PINNED_WORKER_IMAGE,
            "metadata_sha256": EXPECTED_METADATA_SHA256,
            "runtime_metadata_sha256": EXPECTED_METADATA_SHA256,
            "python_wrapper_version": EXPECTED_PYTHON_WRAPPER_VERSION,
            "c_library_version": EXPECTED_C_LIBRARY_VERSION,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(canonical_json(payload), encoding="utf-8")
        return path

    def generate(
        self,
        workflow: ResearchWorkflow,
        mechanisms: list,
        provider: StaticProvider,
        key: str = "test-generation",
    ) -> list[CandidateManifest]:
        return workflow.generate_candidates(
            mechanisms,
            provider,
            self.write_fixture_attestation(workflow, key),
            key,
        )

    def payload(self, mechanism_id: str, code: str = "def signal(dataframe):\n    return dataframe['close'] > 0\n") -> dict:
        return {
            "hypothesis": {
                "hypothesis_id": "hyp-clean-reclaim",
                "mechanism_ids": [mechanism_id],
                "title": "Causal reclaim",
                "rationale": "A clean-room reclaim after exhaustion.",
                "expected_regimes": ["neutral|range|normal"],
                "entry_logic": ["closed candle reclaim"],
                "exit_neutral_horizons": [12, 24, 48],
                "controls": ["same regime without reclaim"],
                "failure_conditions": ["median <= 0"],
                "required_timeframes": ["1h", "4h"],
                "schema_version": 2,
                "indicator_contracts": [],
            },
            "generated_code": code,
        }

    def v2_payload(self, mechanism_id: str) -> dict:
        payload = self.payload(mechanism_id)
        payload["hypothesis"]["indicator_contracts"] = [
            {
                "function": "AROON",
                "keyword_parameters": [{"name": "timeperiod", "value": 14}],
                "output_bindings": [
                    {"output": "aroondown", "target": "aroon_down"}
                ],
            }
        ]
        return payload

    def test_schema_v2_payload_parses_indicator_contracts(self) -> None:
        hypothesis, _ = parse_candidate_payload(self.v2_payload("m-1"))
        self.assertEqual(hypothesis.schema_version, 2)
        self.assertEqual(hypothesis.indicator_contracts[0].function, "AROON")

    def test_schema_v2_payload_requires_indicator_contracts(self) -> None:
        payload = self.v2_payload("m-1")
        del payload["hypothesis"]["indicator_contracts"]
        with self.assertRaises(PayloadValidationError):
            parse_candidate_payload(payload)

    def test_explicit_schema_v1_stored_payload_remains_readable(self) -> None:
        raw = asdict(
            Hypothesis(
                hypothesis_id="legacy",
                mechanism_ids=("m-1",),
                title="Legacy hypothesis",
                rationale="Historical read-only record.",
                expected_regimes=("neutral|range|normal",),
                entry_logic=("legacy entry",),
                exit_neutral_horizons=(12,),
                controls=("legacy control",),
                failure_conditions=("legacy failure",),
                required_timeframes=("1h",),
            )
        )
        parsed = hypothesis_from_stored_payload(raw)
        self.assertEqual(parsed.schema_version, 1)
        self.assertEqual(parsed.indicator_contracts, ())

    def test_invalid_attestation_blocks_provider_before_usage_or_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            mechanisms = workflow.scan(
                source,
                "repo",
                "commit",
                sha256(SOURCE.encode()).hexdigest(),
            )
            provider = CountingProvider(self.payload(mechanisms[0].mechanism_id))
            with self.assertRaises(RuntimeError):
                workflow.generate_candidates(
                    mechanisms,
                    provider,
                    Path(directory) / "missing.json",
                    "EXP-075",
                )
            self.assertEqual(provider.calls, 0)
            self.assertEqual(workflow.store.candidate_count(), 0)
            self.assertEqual(workflow.store.monthly_ai_cost(), 0.0)

    def test_static_registry_error_blocks_provider_and_candidate_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            mechanisms = workflow.scan(
                source,
                "repo",
                "commit",
                sha256(SOURCE.encode()).hexdigest(),
            )
            provider = CountingProvider(self.payload(mechanisms[0].mechanism_id))
            with patch(
                "research_factory.workflow.CompositeStaticValidator",
                side_effect=RuntimeError("registry unavailable"),
            ):
                with self.assertRaisesRegex(RuntimeError, "registry unavailable"):
                    self.generate(workflow, mechanisms, provider)
            self.assertEqual(provider.calls, 0)
            self.assertEqual(workflow.store.candidate_count(), 0)
            self.assertEqual(workflow.store.monthly_ai_cost(), 0.0)

    def test_composite_rejects_n07_even_when_code_safety_passes(self) -> None:
        source = Path(
            Path(__file__).parent / "fixtures/EXP-075/N07/candidate.py"
        ).read_text(encoding="utf-8")
        manifest = json.loads(
            Path(Path(__file__).parent / "fixtures/EXP-075/N07/manifest.json").read_text(
                encoding="utf-8"
            )
        )
        hypothesis = hypothesis_from_stored_payload(manifest["hypothesis"])
        self.assertTrue(CandidateSafetyValidator().validate(source).passed)
        report = CompositeStaticValidator().validate(hypothesis, source)
        self.assertFalse(report.passed)

    def test_v1_candidate_cannot_reaudit_or_materialize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, _ = self.setup_workflow(directory)

            def insert_v1(suffix: str, status: CandidateStatus) -> str:
                hypothesis = Hypothesis(
                    hypothesis_id=f"legacy-{suffix}",
                    mechanism_ids=(f"legacy-{suffix}",),
                    title="Legacy candidate",
                    rationale="Historical read-only fixture.",
                    expected_regimes=("legacy",),
                    entry_logic=(f"legacy entry {suffix}",),
                    exit_neutral_horizons=(48,),
                    controls=("legacy control",),
                    failure_conditions=("legacy failure",),
                    required_timeframes=("1h",),
                )
                candidate = CandidateManifest.create(
                    hypothesis,
                    "def signal(dataframe):\n"
                    "    return dataframe['close'].gt(0).fillna(False)\n",
                    "static",
                    "legacy-v1",
                    "legacy",
                )
                self.assertTrue(workflow.store.add_candidate(candidate))
                workflow.store.set_candidate_status(candidate.candidate_id, status)
                if status is CandidateStatus.REJECTED:
                    workflow.store.add_gate_result(
                        candidate.candidate_id,
                        GateResult("static_safety", False, {"violations": 1}),
                    )
                return candidate.candidate_id

            rejected = insert_v1("rejected", CandidateStatus.REJECTED)
            with self.assertRaisesRegex(RuntimeError, "schema v1 is historical"):
                workflow.reaudit_static(rejected)

            draft = insert_v1("draft", CandidateStatus.DRAFT)
            with self.assertRaisesRegex(RuntimeError, "schema v1 is historical"):
                workflow.materialize_draft(draft)

            testing = insert_v1("testing", CandidateStatus.TESTING)
            with self.assertRaisesRegex(RuntimeError, "schema v1 is historical"):
                workflow.record_gate(
                    testing,
                    GateResult("causality_probe", True, {}),
                )

            review_hold = insert_v1("review-hold", CandidateStatus.REVIEW_HOLD)
            with self.assertRaisesRegex(RuntimeError, "schema v1 is historical"):
                workflow.resolve_review_hold(review_hold, {}, {})

            passed = insert_v1("passed", CandidateStatus.PASSED)
            with self.assertRaisesRegex(RuntimeError, "schema v1 is historical"):
                workflow.approve(passed)

            rejected_by_user = insert_v1(
                "user-reject",
                CandidateStatus.TESTING,
            )
            with self.assertRaisesRegex(RuntimeError, "schema v1 is historical"):
                workflow.reject(rejected_by_user)

    def test_n07_generation_stops_at_static_gate_without_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source_path = self.setup_workflow(directory)
            mechanisms = workflow.scan(
                source_path,
                "repo",
                "commit",
                sha256(SOURCE.encode()).hexdigest(),
            )
            payload = self.v2_payload(mechanisms[0].mechanism_id)
            payload["hypothesis"]["indicator_contracts"][0]["output_bindings"] = [
                {"output": "aroondown", "target": "aroondown"}
            ]
            payload["generated_code"] = Path(
                Path(__file__).parent / "fixtures/EXP-075/N07/candidate.py"
            ).read_text(encoding="utf-8")
            candidate = self.generate(
                workflow,
                mechanisms,
                StaticProvider(payload),
            )[0]
            self.assertEqual(
                workflow.store.candidate_status(candidate.candidate_id),
                CandidateStatus.REJECTED,
            )
            gates = workflow.store.gate_results(candidate.candidate_id)
            self.assertEqual([item["gate"] for item in gates], ["static_safety"])
            self.assertFalse(gates[0]["passed"])
            with self.assertRaises(KeyError):
                workflow.store.alpha_slot(candidate.candidate_id)

    def passing_metrics(self, gate: str, code: str) -> dict:
        metrics = {
            "causality_probe": {
                "candidate_code_sha256": sha256(code.encode()).hexdigest(),
                "fixture_version": "synthetic-ohlcv-v1",
                "prefix_checks": 12,
                "mismatched_prefixes": 0,
                "mismatched_rows": 0,
                "execution_errors": 0,
            },
            "specification_review": {
                "reviewed_code_sha256": sha256(code.encode()).hexdigest(),
                "hypothesis_code_consistent": True,
                "indicator_semantics_verified": True,
                "timeframe_semantics_verified": True,
                "entry_timing_consistent": True,
                "materiality_basis": "pass",
                "review_notes": "Independent semantic review passed.",
            },
            "event_study": {
                "effective_events": 60,
                "median_return_pct": 0.1,
                "mean_return_pct": 0.1,
                "positive_pair_fraction": 0.7,
                "positive_half_fraction": 1.0,
                "mfe_mae_ratio": 1.2,
            },
            "ablation": {
                "median_lift_pct": 0.1,
                "mean_lift_pct": 0.1,
                "positive_pair_fraction": 0.7,
            },
            "statistics": {
                "cluster_ci_lower_pct": 0.1,
                "family_adjusted_p_value": 0.0002,
                "effective_clusters": 60,
            },
            "specialist_backtest": {
                "trades": 60,
                "total_profit_pct": 1.0,
                "median_profit_pct": 0.1,
                "profit_factor": 1.2,
                "max_drawdown_pct": 5.0,
                "positive_pair_fraction": 0.7,
            },
            "multi_seed_hyperopt": {
                "seed_count": 3,
                "positive_seed_fraction": 1.0,
                "positive_neighborhood_fraction": 0.8,
                "stable_plateau": True,
            },
            "walk_forward": {
                "fold_count": 3,
                "positive_fold_fraction": 0.7,
                "median_profit_pct": 0.1,
                "profit_factor": 1.2,
                "max_drawdown_pct": 5.0,
            },
            "final_locked": {
                "trades": 60,
                "total_profit_pct": 1.0,
                "median_profit_pct": 0.1,
                "profit_factor": 1.2,
                "max_drawdown_pct": 5.0,
                "positive_pair_fraction": 0.7,
                "fee_stress_total_profit_pct": 0.1,
                "fee_stress_profit_factor": 1.1,
                "positive_neighborhood_fraction": 0.8,
                "combined_cluster_ci_lower_pct": 0.1,
                "combined_effective_samples": 300,
            },
        }
        return metrics[gate]

    def review_payload(self, code: str, passed: bool = True) -> dict:
        return {
            "reviewed_code_sha256": sha256(code.encode()).hexdigest(),
            "hypothesis_code_consistent": passed,
            "indicator_semantics_verified": passed,
            "timeframe_semantics_verified": passed,
            "entry_timing_consistent": passed,
            "materiality_basis": "pass" if passed else "confirmed_signal_mismatch",
            "review_notes": "Independent adversarial semantic review completed.",
            "findings": [] if passed else ["Indicator direction contradicts the manifest."],
        }

    def test_safe_candidate_requires_all_gates_and_human_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            provider = StaticProvider(self.payload(mechanisms[0].mechanism_id))
            candidate = self.generate(workflow, mechanisms, provider)[0]
            with self.assertRaises(KeyError):
                workflow.store.alpha_slot(candidate.candidate_id)
            self.assertEqual(workflow.next_gate(candidate.candidate_id), "causality_probe")
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            self.assertEqual(workflow.next_gate(candidate.candidate_id), "specification_review")
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "specification_review",
                    True,
                    self.passing_metrics("specification_review", candidate.generated_code),
                ),
            )
            alpha = workflow.store.alpha_slot(candidate.candidate_id)
            self.assertEqual(alpha["slot_number"], 1)
            self.assertAlmostEqual(alpha["alpha_threshold"], 0.05 / 180)
            self.assertEqual(workflow.next_gate(candidate.candidate_id), "event_study")
            with self.assertRaises(RuntimeError):
                workflow.approve(candidate.candidate_id)
            for gate in REQUIRED_PROMOTION_GATES[3:]:
                workflow.record_gate(
                    candidate.candidate_id,
                    GateResult(gate, True, self.passing_metrics(gate, candidate.generated_code)),
                )
            workflow.approve(candidate.candidate_id)
            self.assertEqual(workflow.store.candidate_status(candidate.candidate_id).value, "approved")

    def test_unsafe_candidate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            provider = StaticProvider(
                self.payload(mechanisms[0].mechanism_id, "import os\nos.system('bad')\n")
            )
            candidate = self.generate(workflow, mechanisms, provider)[0]
            self.assertEqual(workflow.store.candidate_status(candidate.candidate_id).value, "rejected")
            self.assertIsNone(workflow.next_gate(candidate.candidate_id))

    def test_technical_static_reaudit_preserves_audit_without_alpha_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            provider = StaticProvider(
                self.payload(mechanisms[0].mechanism_id, "def signal(dataframe):\n    return open('x')\n")
            )
            candidate = self.generate(workflow, mechanisms, provider)[0]
            payload = workflow.store.candidate_payload(candidate.candidate_id)
            payload["generated_code"] = "def signal(dataframe):\n    return dataframe['close'] > 0\n"
            with workflow.store.connect() as connection:
                from research_factory.models import canonical_json
                connection.execute(
                    "UPDATE candidates SET payload = ? WHERE candidate_id = ?",
                    (canonical_json(payload), candidate.candidate_id),
                )
            result = workflow.reaudit_static(candidate.candidate_id)
            self.assertTrue(result.passed)
            self.assertEqual(len(workflow.store.gate_results(candidate.candidate_id)), 2)
            with self.assertRaises(KeyError):
                workflow.store.alpha_slot(candidate.candidate_id)
            self.assertEqual(workflow.next_gate(candidate.candidate_id), "causality_probe")

    def test_gate_order_cannot_be_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            with self.assertRaises(RuntimeError):
                workflow.record_gate(candidate.candidate_id, GateResult("walk_forward", True, {}))

    def test_workflow_recomputes_forged_specification_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            metrics = self.passing_metrics("specification_review", candidate.generated_code)
            metrics["indicator_semantics_verified"] = False
            status = workflow.record_gate(
                candidate.candidate_id,
                GateResult("specification_review", True, metrics),
            )
            self.assertEqual(status.value, "review_hold")
            self.assertFalse(workflow.store.gate_results(candidate.candidate_id)[-1]["passed"])
            with self.assertRaises(KeyError):
                workflow.store.alpha_slot(candidate.candidate_id)

    def test_semantic_rejection_cannot_be_reopened_by_static_reaudit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            metrics = self.passing_metrics("specification_review", candidate.generated_code)
            metrics["hypothesis_code_consistent"] = False
            workflow.record_gate(
                candidate.candidate_id,
                GateResult("specification_review", False, metrics),
            )
            with self.assertRaises(RuntimeError):
                workflow.reaudit_static(candidate.candidate_id)

    def test_independent_reviewer_passes_and_reserves_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            status = workflow.review_specification(
                candidate.candidate_id,
                StaticProvider(
                    self.review_payload(candidate.generated_code),
                    model="reviewer-v1",
                ),
            )
            self.assertEqual(status.value, "testing")
            self.assertEqual(workflow.next_gate(candidate.candidate_id), "event_study")
            self.assertEqual(workflow.store.alpha_slot(candidate.candidate_id)["slot_number"], 1)

    def test_reviewer_failure_holds_without_alpha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            status = workflow.review_specification(
                candidate.candidate_id,
                StaticProvider(
                    self.review_payload(candidate.generated_code, passed=False),
                    model="reviewer-v1",
                ),
            )
            self.assertEqual(status.value, "review_hold")
            self.assertIsNone(workflow.next_gate(candidate.candidate_id))
            with self.assertRaises(KeyError):
                workflow.store.alpha_slot(candidate.candidate_id)

    def test_review_hold_machine_mismatch_becomes_permanent_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            workflow.review_specification(
                candidate.candidate_id,
                StaticProvider(
                    self.review_payload(candidate.generated_code, passed=False),
                    model="reviewer-v1",
                ),
            )
            code_hash = sha256(candidate.generated_code.encode()).hexdigest()
            review_path = (
                Path(directory)
                / "artifacts"
                / "candidates"
                / candidate.candidate_id
                / "specification-review.json"
            )
            evidence = {
                "candidate_code_sha256": code_hash,
                "verifier": "runtime_signal_diff",
                "checks": 12,
                "failures": 3,
            }
            resolution = {
                "candidate_code_sha256": code_hash,
                "review_artifact_sha256": sha256(review_path.read_bytes()).hexdigest(),
                "decision": "confirmed_signal_mismatch",
                "verifier": "runtime_signal_diff",
                "evidence_sha256": sha256(canonical_json(evidence).encode()).hexdigest(),
                "notes": "Reference signal differs on three rows.",
            }
            status = workflow.resolve_review_hold(candidate.candidate_id, resolution, evidence)
            self.assertEqual(status.value, "rejected")
            self.assertEqual(
                workflow.store.gate_results(candidate.candidate_id)[-1]["gate"],
                "specification_adjudication",
            )
            with self.assertRaises(KeyError):
                workflow.store.alpha_slot(candidate.candidate_id)

    def test_review_hold_false_positive_can_proceed_only_with_clean_machine_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            workflow.review_specification(
                candidate.candidate_id,
                StaticProvider(
                    self.review_payload(candidate.generated_code, passed=False),
                    model="reviewer-v1",
                ),
            )
            code_hash = sha256(candidate.generated_code.encode()).hexdigest()
            review_path = (
                Path(directory)
                / "artifacts"
                / "candidates"
                / candidate.candidate_id
                / "specification-review.json"
            )
            evidence = {
                "candidate_code_sha256": code_hash,
                "verifier": "timeframe_anchor",
                "checks": 6,
                "failures": 0,
            }
            resolution = {
                "candidate_code_sha256": code_hash,
                "review_artifact_sha256": sha256(review_path.read_bytes()).hexdigest(),
                "decision": "reviewer_false_positive",
                "verifier": "timeframe_anchor",
                "evidence_sha256": sha256(canonical_json(evidence).encode()).hexdigest(),
                "notes": "Golden availability anchors all match.",
            }
            status = workflow.resolve_review_hold(candidate.candidate_id, resolution, evidence)
            self.assertEqual(status.value, "testing")
            self.assertEqual(workflow.next_gate(candidate.candidate_id), "event_study")
            self.assertEqual(workflow.store.alpha_slot(candidate.candidate_id)["slot_number"], 1)

    def test_review_hold_cannot_be_released_by_failing_machine_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            workflow.review_specification(
                candidate.candidate_id,
                StaticProvider(
                    self.review_payload(candidate.generated_code, passed=False),
                    model="reviewer-v1",
                ),
            )
            code_hash = sha256(candidate.generated_code.encode()).hexdigest()
            review_path = (
                Path(directory)
                / "artifacts"
                / "candidates"
                / candidate.candidate_id
                / "specification-review.json"
            )
            evidence = {
                "candidate_code_sha256": code_hash,
                "verifier": "timeframe_anchor",
                "checks": 6,
                "failures": 1,
            }
            resolution = {
                "candidate_code_sha256": code_hash,
                "review_artifact_sha256": sha256(review_path.read_bytes()).hexdigest(),
                "decision": "reviewer_false_positive",
                "verifier": "timeframe_anchor",
                "evidence_sha256": sha256(canonical_json(evidence).encode()).hexdigest(),
                "notes": "One anchor still fails.",
            }
            with self.assertRaises(ValueError):
                workflow.resolve_review_hold(candidate.candidate_id, resolution, evidence)
            self.assertEqual(
                workflow.store.candidate_status(candidate.candidate_id).value,
                "review_hold",
            )

    def test_generator_model_family_cannot_review_its_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            candidate = self.generate(
                workflow,
                mechanisms, StaticProvider(self.payload(mechanisms[0].mechanism_id))
            )[0]
            workflow.record_gate(
                candidate.candidate_id,
                GateResult(
                    "causality_probe",
                    True,
                    self.passing_metrics("causality_probe", candidate.generated_code),
                ),
            )
            with self.assertRaises(RuntimeError):
                workflow.review_specification(
                    candidate.candidate_id,
                    StaticProvider(
                        self.review_payload(candidate.generated_code),
                        model="static-v2",
                    ),
                )

    def test_used_mechanism_is_not_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            checksum = sha256(SOURCE.encode()).hexdigest()
            mechanisms = workflow.scan(source, "repo", "commit", checksum)
            provider = StaticProvider(self.payload(mechanisms[0].mechanism_id))
            self.assertEqual(len(self.generate(workflow, mechanisms, provider)), 1)
            self.assertEqual(self.generate(workflow, mechanisms, provider), [])

    def test_generation_prompt_requires_canonical_and_closed_htf_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, source = self.setup_workflow(directory)
            mechanisms = workflow.scan(
                source,
                "repo",
                "commit",
                sha256(SOURCE.encode()).hexdigest(),
            )
            prompt = workflow._prompt(mechanisms[0])
            self.assertIn("canonical published definition", prompt)
            self.assertIn("same scale", prompt)
            self.assertIn("exact variant/formula", prompt)
            self.assertIn("availability timestamp explicitly", prompt)
            self.assertIn("delay by one completed higher-timeframe bar", prompt)
            self.assertIn("canonical means TA-Lib", prompt)
            self.assertIn("population/sample convention", prompt)
            self.assertIn("every semantically relevant keyword exactly", prompt)
            self.assertIn("which returned output enters the setup", prompt)
            self.assertIn("STOCHRSI(14,14,3,3) is forbidden", prompt)
            self.assertIn("contains only open, high, low, close, and volume", prompt)
            self.assertIn("derive every needed feature causally inside signal", prompt)
            self.assertIn("boolean pandas Series aligned to the full input index", prompt)
            self.assertIn("never return a scalar bool", prompt)
            self.assertIn("distinct completed aggregate OHLCV series", prompt)
            self.assertIn("Never forward-fill aggregate OHLCV onto 1h", prompt)
            self.assertIn("first appears at T1 + D*P", prompt)
            self.assertIn("[00:00,04:00) first appears at 08:00", prompt)
            self.assertIn("Jan 1 daily bar first appears at Jan 3 00:00 UTC", prompt)
            self.assertIn("schema_version=2", prompt)
            self.assertIn("indicator_contracts", prompt)
            self.assertIn("output_bindings", prompt)
            self.assertIn("direct `import talib`", prompt)
            self.assertIn("tuple destructuring inside signal()", prompt)
            self.assertIn("single final return", prompt)
            self.assertIn("straight-line", prompt)

    def test_review_prompt_uses_talib_not_platform_defaults(self) -> None:
        prompt = ResearchWorkflow._review_prompt(
            {"entry_logic": ["canonical Bollinger Bands"]},
            "def signal(dataframe):\n    return dataframe['close'] > 0\n",
            "a" * 64,
        )
        self.assertIn("TA-Lib semantics", prompt)
        self.assertIn("not an unspecified industry, TradingView, pandas", prompt)
        self.assertIn("name the TA-Lib function", prompt)
        self.assertIn("unverified convention claims", prompt)
        self.assertIn("materially changes the returned setup", prompt)
        self.assertIn("Unused outputs, style, redundancy", prompt)
        self.assertIn("compact positional notation", prompt)
        self.assertIn("every semantically relevant TA-Lib keyword value", prompt)
        self.assertIn("output consumed by the returned setup", prompt)
        self.assertIn("aggregate distinct completed higher-timeframe", prompt)
        self.assertIn("only then align/forward-fill", prompt)
        self.assertIn("first appears at T1 + D*P", prompt)
        self.assertIn("right label names T1 and does not itself increment D", prompt)
        self.assertIn("Jan 1 daily bar first appears at Jan 3 00:00 UTC", prompt)
        self.assertIn("TA-Lib consumes ordered numeric observations", prompt)
        self.assertIn("discard incomplete aggregate bars", prompt)
        self.assertIn("not by itself a semantic mismatch", prompt)
        self.assertIn("calendar-contiguous grid", prompt)
        self.assertIn("hypothetical missing-bar effect is not", prompt)

    def test_draft_can_be_materialized_without_new_ai_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow, _ = self.setup_workflow(directory)
            candidate = CandidateManifest.create(
                hypothesis=Hypothesis(
                    hypothesis_id="hyp-recovered",
                    mechanism_ids=("m-recovered",),
                    title="Recovered draft",
                    rationale="Recover an already billed candidate without a new provider call.",
                    expected_regimes=("neutral",),
                    entry_logic=("close > 0",),
                    exit_neutral_horizons=(12, 24, 48),
                    controls=("same regime without signal",),
                    failure_conditions=("median <= 0",),
                    required_timeframes=("1h",),
                    schema_version=2,
                    indicator_contracts=(),
                ),
                generated_code="def signal(dataframe):\n    return dataframe['close'] > 0\n",
                provider="static",
                model="static",
                prompt="prompt",
            )
            workflow.store.add_candidate(candidate)
            result = workflow.materialize_draft(candidate.candidate_id)
            self.assertTrue(result.passed)
            self.assertEqual(workflow.store.candidate_status(candidate.candidate_id).value, "testing")
            self.assertTrue(
                (workflow.artifact_root / "candidates" / candidate.candidate_id / "candidate.py").is_file()
            )


if __name__ == "__main__":
    unittest.main()
