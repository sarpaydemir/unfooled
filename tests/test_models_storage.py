from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import unittest

from research_factory.models import (
    CandidateManifest,
    CandidateStatus,
    GateResult,
    Hypothesis,
    IndicatorContract,
    IndicatorOutputBinding,
    IndicatorParameter,
)
from research_factory.experiment_alpha import ExperimentAlphaStore
from research_factory.policy import BudgetPolicy
from research_factory.storage import ResearchStore


def hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id="hyp-1",
        mechanism_ids=("m-1",),
        title="Causal reclaim",
        rationale="Price reclaims a causal range after exhaustion.",
        expected_regimes=("neutral|range|normal",),
        entry_logic=("close > prior_range_high",),
        exit_neutral_horizons=(12, 24, 48),
        controls=("same regime without reclaim",),
        failure_conditions=("median <= 0",),
        required_timeframes=("1h", "4h"),
    )


def hypothesis_with_contract(
    target: str,
    value: int | float = 14,
    reverse: bool = False,
) -> Hypothesis:
    parameters = [IndicatorParameter("timeperiod", value)]
    bindings = [IndicatorOutputBinding("aroondown", target)]
    if reverse:
        parameters.reverse()
        bindings.reverse()
    return Hypothesis(
        hypothesis_id="hyp-aroon",
        mechanism_ids=("m-aroon",),
        title="Canonical AROON down",
        rationale="Use the canonical TA-Lib AROON down output.",
        expected_regimes=("neutral",),
        entry_logic=("TA-Lib AROON down is above 70.",),
        exit_neutral_horizons=(48,),
        controls=("same regime without AROON condition",),
        failure_conditions=("median <= 0",),
        required_timeframes=("1h",),
        schema_version=2,
        indicator_contracts=(
            IndicatorContract("AROON", tuple(parameters), tuple(bindings)),
        ),
    )


class ModelStorageTests(unittest.TestCase):
    def test_factory_store_does_not_create_experiment_alpha_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research.db"
            ResearchStore(path, writable=True)
            before = path.read_bytes()
            read_only = ResearchStore(path)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(sqlite3.OperationalError):
                with read_only.connect() as connection:
                    connection.execute("CREATE TABLE forbidden_write(value TEXT)")
            with sqlite3.connect(path) as connection:
                names = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_schema WHERE type = 'table'"
                    )
                }
            self.assertNotIn("experiment_alpha_campaigns", names)
            self.assertNotIn("experiment_alpha_slots", names)

    def test_non_candidate_experiment_alpha_slot_is_atomic_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentAlphaStore(Path(directory) / "experiment-alpha.db")
            store.ensure_campaign(
                "overlay", "open-development", 1, "exploratory"
            )
            first = store.reserve_slot(
                "overlay", "EXP-095", task_sha256="a" * 64,
                manifest_sha256="b" * 64,
            )
            second = store.reserve_slot(
                "overlay", "EXP-095", task_sha256="a" * 64,
                manifest_sha256="b" * 64,
            )
            self.assertEqual(first, second)
            self.assertEqual(first["slot_number"], 1)
            self.assertEqual(
                store.campaign_status("overlay")["used_hypotheses"],
                1,
            )
            with self.assertRaises(RuntimeError):
                store.reserve_slot(
                    "overlay", "EXP-096", task_sha256="c" * 64,
                    manifest_sha256="d" * 64,
                )
            with self.assertRaises(RuntimeError):
                store.reserve_slot(
                    "overlay", "EXP-095", task_sha256="e" * 64,
                    manifest_sha256="b" * 64,
                )

    def test_candidate_fingerprint_is_order_stable(self) -> None:
        first = CandidateManifest.create(hypothesis(), "x = 1", "mock", "mock", "prompt")
        second = CandidateManifest.create(hypothesis(), "x = 2", "mock", "mock", "prompt")
        self.assertEqual(first.candidate_id, second.candidate_id)

    def test_target_variable_name_does_not_change_semantic_identity(self) -> None:
        first = CandidateManifest.create(
            hypothesis_with_contract("aroon_down"),
            "x = 1",
            "mock",
            "mock",
            "prompt",
        )
        second = CandidateManifest.create(
            hypothesis_with_contract("down_4h"),
            "x = 2",
            "mock",
            "mock",
            "prompt",
        )
        self.assertEqual(first.candidate_id, second.candidate_id)

    def test_contract_numeric_representation_is_canonical(self) -> None:
        first = CandidateManifest.create(
            hypothesis_with_contract("aroon_down", value=14),
            "x = 1",
            "mock",
            "mock",
            "prompt",
        )
        second = CandidateManifest.create(
            hypothesis_with_contract("aroon_down", value=14.0, reverse=True),
            "x = 2",
            "mock",
            "mock",
            "prompt",
        )
        self.assertEqual(first.candidate_id, second.candidate_id)

    def test_store_rejects_semantic_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            first = CandidateManifest.create(hypothesis(), "x = 1", "mock", "mock", "prompt")
            second = CandidateManifest.create(hypothesis(), "x = 2", "mock", "mock", "prompt")
            self.assertTrue(store.add_candidate(first))
            self.assertFalse(store.add_candidate(second))
            store.set_candidate_status(first.candidate_id, CandidateStatus.TESTING)
            store.add_gate_result(first.candidate_id, GateResult("static", True, {}))
            self.assertEqual(store.list_candidates()[0]["status"], "testing")

    def test_budget_stops_new_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            store.record_usage("openrouter", "model", 1, 1, 10.0)
            with self.assertRaises(RuntimeError):
                BudgetPolicy(monthly_ai_budget_usd=10).assert_ai_budget(store)

    def test_generation_pause_is_fail_closed(self) -> None:
        with self.assertRaises(RuntimeError):
            BudgetPolicy(generation_enabled=False).assert_generation_enabled()

    def test_manual_generation_job_is_claimed_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            payload = {"source": "pinned.py", "checksum": "a" * 64}
            first = store.begin_idempotent_job(
                "manual-generation-1", "EXP-069", "manual_generation", payload
            )
            second = store.begin_idempotent_job(
                "manual-generation-1", "EXP-069", "manual_generation", payload
            )
            self.assertTrue(first["acquired"])
            self.assertFalse(second["acquired"])
            self.assertEqual(second["status"], "running")

    def test_idempotency_key_cannot_be_reused_with_different_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            store.begin_idempotent_job(
                "manual-generation-1",
                "EXP-069",
                "manual_generation",
                {"checksum": "a" * 64},
            )
            with self.assertRaises(RuntimeError):
                store.begin_idempotent_job(
                    "manual-generation-1",
                    "EXP-069",
                    "manual_generation",
                    {"checksum": "b" * 64},
                )

    def test_queued_job_is_claimed_by_only_one_concurrent_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "research.db"
            store = ResearchStore(path, writable=True)
            self.assertTrue(
                store.enqueue_job(
                    "cycle-1",
                    "cycle-1",
                    "research_cycle",
                    {"effective_events": 60},
                )
            )

            def claim() -> dict | None:
                return ResearchStore(path, writable=True).claim_job("research_cycle")

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(lambda _: claim(), range(2)))

            claimed = [result for result in results if result is not None]
            self.assertEqual(len(claimed), 1)
            self.assertEqual(claimed[0]["job_id"], "cycle-1")
            self.assertEqual(store.list_jobs()[0]["status"], "running")

    def test_stale_running_job_requires_explicit_fail_closed_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            store.begin_idempotent_job(
                "manual-generation-1",
                "EXP-073-stale",
                "manual_generation",
                {"checksum": "a" * 64},
            )
            with store.connect() as connection:
                connection.execute(
                    "UPDATE jobs SET updated_at = '2020-01-01 00:00:00' "
                    "WHERE job_id = 'manual-generation-1'"
                )
            self.assertEqual(
                [row["job_id"] for row in store.stale_running_jobs(120)],
                ["manual-generation-1"],
            )
            with self.assertRaises(ValueError):
                store.resolve_stale_job(
                    "manual-generation-1", "completed", "cannot infer completion", 120
                )
            store.resolve_stale_job(
                "manual-generation-1",
                "blocked",
                "manual reconciliation required; automatic retry forbidden",
                120,
            )
            self.assertEqual(store.list_jobs()[0]["status"], "blocked")
            self.assertEqual(store.stale_running_jobs(120), [])

    def test_fresh_running_job_cannot_be_resolved_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            store.begin_idempotent_job(
                "manual-generation-1",
                "EXP-073-fresh",
                "manual_generation",
                {"checksum": "a" * 64},
            )
            with self.assertRaises(RuntimeError):
                store.resolve_stale_job(
                    "manual-generation-1",
                    "failed",
                    "must not close an active job",
                    120,
                )

    def test_gate_transition_rolls_back_every_write_on_alpha_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            candidate = CandidateManifest.create(
                hypothesis(), "x = 1", "mock", "mock", "prompt"
            )
            self.assertTrue(store.add_candidate(candidate))
            result = GateResult("specification_review", True, {"evidence": "passed"})
            with self.assertRaises(KeyError):
                store.apply_gate_transition(
                    candidate.candidate_id,
                    result,
                    CandidateStatus.TESTING,
                    alpha_campaign_id="missing-campaign",
                )
            self.assertEqual(
                store.candidate_status(candidate.candidate_id),
                CandidateStatus.DRAFT,
            )
            self.assertEqual(store.gate_results(candidate.candidate_id), [])
            with self.assertRaises(KeyError):
                store.alpha_slot(candidate.candidate_id)

    def test_alpha_reservation_remains_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "research.db", writable=True)
            candidate = CandidateManifest.create(
                hypothesis(), "x = 1", "mock", "mock", "prompt"
            )
            self.assertTrue(store.add_candidate(candidate))
            store.ensure_alpha_campaign("campaign", "dataset", 2, 3, 0.05, "test")
            first = store.reserve_alpha_slot("campaign", candidate.candidate_id)
            second = store.reserve_alpha_slot("campaign", candidate.candidate_id)
            self.assertEqual(first, second)
            self.assertEqual(first["slot_number"], 1)


if __name__ == "__main__":
    unittest.main()
