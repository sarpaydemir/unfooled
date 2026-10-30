from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .miner import MechanismMiner, write_catalog
from .gates import GateEvaluator
from .models import (
    CandidateManifest,
    CandidateStatus,
    GateResult,
    Hypothesis,
    Mechanism,
    canonical_json,
)
from .policy import BudgetPolicy
from .providers import AIProvider
from .schemas import HYPOTHESIS_RESPONSE_SCHEMA, SPECIFICATION_REVIEW_SCHEMA
from .static_validation import CompositeStaticValidator
from .storage import ResearchStore
from .talib_metadata import verify_runtime_attestation
from .validation import hypothesis_from_stored_payload, parse_candidate_payload


REQUIRED_PROMOTION_GATES = (
    "static_safety",
    "causality_probe",
    "specification_review",
    "event_study",
    "ablation",
    "statistics",
    "specialist_backtest",
    "multi_seed_hyperopt",
    "walk_forward",
    "final_locked",
)
DEFAULT_ALPHA_CAMPAIGN = "nfi-cleanroom-v1"
DEFAULT_ALPHA_DATASET = "development-2019-2025-source-contaminated"
REVIEW_SYSTEM_PROMPT = (
    "You independently audit quantitative code against its manifest. "
    "Do not estimate performance. Return only the required structured output."
)


class ResearchWorkflow:
    def __init__(
        self,
        store: ResearchStore,
        artifact_root: Path,
        policy: BudgetPolicy | None = None,
    ) -> None:
        self.store = store
        self.artifact_root = artifact_root
        self.policy = policy or BudgetPolicy()
        if self.store.writable:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
            self.store.ensure_alpha_campaign(
                DEFAULT_ALPHA_CAMPAIGN,
                DEFAULT_ALPHA_DATASET,
                max_hypotheses=30,
                prior_hypotheses=150,
                familywise_alpha=0.05,
                evidence_label="exploratory_source_contaminated",
            )
        else:
            # Read paths validate existing state but never initialize it.
            self.store.alpha_campaign_status(DEFAULT_ALPHA_CAMPAIGN)

    def scan(
        self,
        source_path: Path,
        repository: str,
        commit_sha: str,
        checksum: str,
    ) -> list[Mechanism]:
        mechanisms = MechanismMiner(repository, commit_sha, checksum).mine(source_path)
        for item in mechanisms:
            fingerprint = self._mechanism_fingerprint(item)
            self.store.add_mechanism(item, fingerprint)
        write_catalog(self.artifact_root / "catalogs" / f"{commit_sha}.json", mechanisms)
        return mechanisms

    def generate_candidates(
        self,
        mechanisms: list[Mechanism],
        provider: AIProvider,
        attestation_path: Path,
        idempotency_key: str,
    ) -> list[CandidateManifest]:
        self.policy.assert_generation_enabled()
        verify_runtime_attestation(attestation_path, idempotency_key)
        static_validator = CompositeStaticValidator()
        self.policy.assert_ai_budget(self.store)
        used_mechanisms = self.store.used_mechanism_ids()
        eligible = [
            item for item in mechanisms
            if (
                item.rejected_reason is None
                and item.intent == "entry"
                and item.mechanism_id not in used_mechanisms
            )
        ]
        selected = eligible[: self.policy.max_candidates_per_cycle]
        candidates: list[CandidateManifest] = []
        for mechanism in selected:
            prompt = self._prompt(mechanism)
            response = provider.generate(prompt, HYPOTHESIS_RESPONSE_SCHEMA)
            self.store.record_usage(
                response.provider,
                response.model,
                response.prompt_tokens,
                response.completion_tokens,
                response.cost_usd,
            )
            hypothesis, code = parse_candidate_payload(response.payload)
            if set(hypothesis.mechanism_ids) != {mechanism.mechanism_id}:
                raise ValueError("AI response changed or omitted the selected mechanism lineage")
            candidate = CandidateManifest.create(
                hypothesis, code, response.provider, response.model, prompt
            )
            if not self.store.add_candidate(candidate):
                continue
            self._write_candidate(candidate)
            report = static_validator.validate(hypothesis, code)
            gate = GateResult(
                gate="static_safety",
                passed=report.passed,
                metrics={"violations": len(report.violations)},
                reasons=report.violations,
            )
            self.store.apply_gate_transition(
                candidate.candidate_id,
                gate,
                CandidateStatus.TESTING if report.passed else CandidateStatus.REJECTED,
            )
            candidates.append(candidate)
        return candidates

    def record_gate(self, candidate_id: str, result: GateResult) -> CandidateStatus:
        self._require_v2_candidate(candidate_id)
        current = self.store.candidate_status(candidate_id)
        if current in {CandidateStatus.APPROVED, CandidateStatus.ARCHIVED}:
            raise RuntimeError(f"candidate is immutable in status {current.value}")
        expected = self.next_gate(candidate_id)
        if result.gate != expected:
            raise RuntimeError(f"expected gate '{expected}', received '{result.gate}'")
        alpha_threshold = None
        if result.gate == "statistics":
            alpha_threshold = float(self.store.alpha_slot(candidate_id)["alpha_threshold"])
        expected_code_sha256 = None
        if result.gate in {"causality_probe", "specification_review"}:
            code = self.store.candidate_payload(candidate_id)["generated_code"]
            expected_code_sha256 = sha256(code.encode()).hexdigest()
        evaluated = GateEvaluator(
            alpha_threshold=alpha_threshold,
            expected_code_sha256=expected_code_sha256,
        ).evaluate(result.gate, result.metrics)
        status = CandidateStatus.TESTING if evaluated.passed else CandidateStatus.REJECTED
        if evaluated.gate == "specification_review" and not evaluated.passed:
            status = CandidateStatus.REVIEW_HOLD
        if evaluated.passed and evaluated.gate == REQUIRED_PROMOTION_GATES[-1]:
            status = CandidateStatus.PASSED
        alpha_campaign_id = (
            DEFAULT_ALPHA_CAMPAIGN
            if evaluated.gate == "specification_review" and evaluated.passed
            else None
        )
        self.store.apply_gate_transition(
            candidate_id,
            evaluated,
            status,
            alpha_campaign_id=alpha_campaign_id,
        )
        return status

    def next_gate(self, candidate_id: str) -> str | None:
        if self._candidate_hypothesis(candidate_id).schema_version != 2:
            return None
        if self.store.candidate_status(candidate_id) in {
            CandidateStatus.REJECTED,
            CandidateStatus.REVIEW_HOLD,
            CandidateStatus.APPROVED,
            CandidateStatus.ARCHIVED,
        }:
            return None
        results = self.store.gate_results(candidate_id)
        latest = {item["gate"]: item for item in results}
        passed = {gate for gate, item in latest.items() if item["passed"]}
        failed = {gate for gate, item in latest.items() if not item["passed"]}
        if failed:
            return None
        return next((gate for gate in REQUIRED_PROMOTION_GATES if gate not in passed), None)

    def reaudit_static(self, candidate_id: str) -> GateResult:
        if self.store.candidate_status(candidate_id) is not CandidateStatus.REJECTED:
            raise RuntimeError("static reaudit is only allowed for a rejected candidate")
        hypothesis = self._require_v2_candidate(candidate_id)
        static_results = [
            item for item in self.store.gate_results(candidate_id)
            if item["gate"] == "static_safety"
        ]
        if not static_results or static_results[-1]["passed"]:
            raise RuntimeError("static reaudit requires a failed latest static_safety result")
        payload = self.store.candidate_payload(candidate_id)
        code = payload["generated_code"]
        report = CompositeStaticValidator().validate(hypothesis, code)
        gate = GateResult(
            gate="static_safety",
            passed=report.passed,
            metrics={"violations": len(report.violations), "technical_reaudit": True},
            reasons=report.violations,
        )
        self.store.apply_gate_transition(
            candidate_id,
            gate,
            CandidateStatus.TESTING if report.passed else CandidateStatus.REJECTED,
        )
        return gate

    def materialize_draft(self, candidate_id: str) -> GateResult:
        if self.store.candidate_status(candidate_id) is not CandidateStatus.DRAFT:
            raise RuntimeError("materialization is only allowed for a draft candidate")
        hypothesis = self._require_v2_candidate(candidate_id)
        payload = self.store.candidate_payload(candidate_id)
        code = payload["generated_code"]
        report = CompositeStaticValidator().validate(hypothesis, code)
        directory = self.artifact_root / "candidates" / candidate_id
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "candidate.py").write_text(code, encoding="utf-8")
            (directory / "manifest.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        gate = GateResult(
            gate="static_safety",
            passed=report.passed,
            metrics={"violations": len(report.violations), "recovered_draft": True},
            reasons=report.violations,
        )
        self.store.apply_gate_transition(
            candidate_id,
            gate,
            CandidateStatus.TESTING if report.passed else CandidateStatus.REJECTED,
        )
        return gate

    def review_specification(
        self,
        candidate_id: str,
        provider: AIProvider,
    ) -> CandidateStatus:
        self._require_v2_candidate(candidate_id)
        if self.next_gate(candidate_id) != "specification_review":
            raise RuntimeError("candidate is not ready for specification review")
        candidate = self.store.candidate_payload(candidate_id)
        generator_model = str(candidate["model"])
        declared_reviewer_model = getattr(provider, "model", None)
        if declared_reviewer_model and self._model_family(str(declared_reviewer_model)) == (
            self._model_family(generator_model)
        ):
            raise RuntimeError("reviewer model family must differ from generator model family")
        code = candidate["generated_code"]
        code_hash = sha256(code.encode()).hexdigest()
        prompt = self._review_prompt(candidate["hypothesis"], code, code_hash)
        response = provider.generate(prompt, SPECIFICATION_REVIEW_SCHEMA)
        self.store.record_usage(
            response.provider,
            response.model,
            response.prompt_tokens,
            response.completion_tokens,
            response.cost_usd,
        )
        if self._model_family(response.model) == self._model_family(generator_model):
            raise RuntimeError("review response used the generator model family")
        metrics = self._validate_review_payload(response.payload)
        directory = self.artifact_root / "candidates" / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "specification-review.json").write_text(
            json.dumps(
                {
                    "reviewer_provider": response.provider,
                    "reviewer_model": response.model,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.record_gate(
            candidate_id,
            GateResult("specification_review", False, metrics),
        )

    def resolve_review_hold(
        self,
        candidate_id: str,
        resolution: dict[str, Any],
        evidence: dict[str, Any],
    ) -> CandidateStatus:
        self._require_v2_candidate(candidate_id)
        if self.store.candidate_status(candidate_id) is not CandidateStatus.REVIEW_HOLD:
            raise RuntimeError("candidate is not in review_hold")
        required = {
            "candidate_code_sha256",
            "review_artifact_sha256",
            "decision",
            "verifier",
            "evidence_sha256",
            "notes",
        }
        if not isinstance(resolution, dict) or set(resolution) != required:
            raise ValueError("review resolution fields do not match the contract")
        if resolution["decision"] not in {
            "confirmed_signal_mismatch",
            "reviewer_false_positive",
        }:
            raise ValueError("review resolution decision is not approved")
        if resolution["verifier"] not in {
            "runtime_signal_diff",
            "canonical_reference_probe",
            "timeframe_anchor",
            "manifest_code_exact_match",
        }:
            raise ValueError("review resolution verifier is not approved")
        if not isinstance(resolution["notes"], str) or not resolution["notes"].strip():
            raise ValueError("review resolution notes must be non-empty")
        candidate = self.store.candidate_payload(candidate_id)
        code_hash = sha256(candidate["generated_code"].encode()).hexdigest()
        if resolution["candidate_code_sha256"] != code_hash:
            raise ValueError("review resolution candidate code SHA256 mismatch")
        review_path = self.artifact_root / "candidates" / candidate_id / "specification-review.json"
        if not review_path.is_file():
            raise RuntimeError("specification review artifact is missing")
        if resolution["review_artifact_sha256"] != sha256(review_path.read_bytes()).hexdigest():
            raise ValueError("review resolution artifact SHA256 mismatch")
        if not isinstance(evidence, dict):
            raise ValueError("machine evidence must be an object")
        if resolution["evidence_sha256"] != sha256(canonical_json(evidence).encode()).hexdigest():
            raise ValueError("review resolution machine evidence SHA256 mismatch")
        if evidence.get("candidate_code_sha256") != code_hash:
            raise ValueError("machine evidence candidate code SHA256 mismatch")
        if evidence.get("verifier") != resolution["verifier"]:
            raise ValueError("machine evidence verifier mismatch")
        checks = evidence.get("checks")
        failures = evidence.get("failures")
        if not isinstance(checks, int) or checks <= 0:
            raise ValueError("machine evidence checks must be a positive integer")
        if not isinstance(failures, int) or failures < 0 or failures > checks:
            raise ValueError("machine evidence failures are invalid")

        if resolution["decision"] == "confirmed_signal_mismatch":
            if failures == 0:
                raise ValueError("confirmed mismatch requires machine evidence failures")
            result = GateResult(
                "specification_adjudication",
                False,
                {
                    **resolution,
                    "checks": checks,
                    "failures": failures,
                },
                ("machine-confirmed material semantic mismatch",),
            )
            self.store.apply_gate_transition(candidate_id, result, CandidateStatus.REJECTED)
            return CandidateStatus.REJECTED

        if failures != 0:
            raise ValueError("reviewer false positive resolution requires zero evidence failures")
        review_metrics = {
            "reviewed_code_sha256": code_hash,
            "hypothesis_code_consistent": True,
            "indicator_semantics_verified": True,
            "timeframe_semantics_verified": True,
            "entry_timing_consistent": True,
            "materiality_basis": "pass",
            "review_notes": (
                "Machine-bound adjudication established that the reviewer finding was "
                f"a false positive: {resolution['notes']}"
            ),
            "adjudicated": True,
            "adjudication_evidence_sha256": resolution["evidence_sha256"],
        }
        evaluated = GateEvaluator(
            expected_code_sha256=code_hash,
        ).evaluate("specification_review", review_metrics)
        self.store.apply_gate_transition(
            candidate_id,
            evaluated,
            CandidateStatus.TESTING,
            alpha_campaign_id=DEFAULT_ALPHA_CAMPAIGN,
        )
        return CandidateStatus.TESTING

    def approve(self, candidate_id: str) -> None:
        self._require_v2_candidate(candidate_id)
        if self.store.candidate_status(candidate_id) is not CandidateStatus.PASSED:
            raise RuntimeError("only a candidate that passed every locked gate can be approved")
        self.store.set_candidate_status(candidate_id, CandidateStatus.APPROVED)

    def reject(self, candidate_id: str) -> None:
        self._require_v2_candidate(candidate_id)
        if self.store.candidate_status(candidate_id) is CandidateStatus.APPROVED:
            raise RuntimeError("approved candidate must use the rollback workflow")
        self.store.set_candidate_status(candidate_id, CandidateStatus.REJECTED)

    def _candidate_hypothesis(self, candidate_id: str) -> Hypothesis:
        payload = self.store.candidate_payload(candidate_id)
        return hypothesis_from_stored_payload(payload["hypothesis"])

    def _require_v2_candidate(self, candidate_id: str) -> Hypothesis:
        hypothesis = self._candidate_hypothesis(candidate_id)
        if hypothesis.schema_version != 2:
            raise RuntimeError(
                "schema v1 is historical and cannot enter progression or execution"
            )
        return hypothesis

    def _write_candidate(self, candidate: CandidateManifest) -> None:
        directory = self.artifact_root / "candidates" / candidate.candidate_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "candidate.py").write_text(candidate.generated_code, encoding="utf-8")
        (directory / "manifest.json").write_text(
            json.dumps(asdict(candidate), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _mechanism_fingerprint(mechanism: Mechanism) -> str:
        material = {
            "intent": mechanism.intent,
            "indicators": sorted(mechanism.indicators),
            "conditions": sorted(mechanism.conditions),
        }
        from hashlib import sha256
        return sha256(canonical_json(material).encode()).hexdigest()

    @staticmethod
    def _model_family(model: str) -> str:
        normalized = model.casefold().strip()
        return normalized.split("/", 1)[0].split("-", 1)[0]

    @staticmethod
    def _validate_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
        required = {
            "reviewed_code_sha256",
            "hypothesis_code_consistent",
            "indicator_semantics_verified",
            "timeframe_semantics_verified",
            "entry_timing_consistent",
            "materiality_basis",
            "review_notes",
            "findings",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("specification review fields do not match the contract")
        for key in (
            "hypothesis_code_consistent",
            "indicator_semantics_verified",
            "timeframe_semantics_verified",
            "entry_timing_consistent",
        ):
            if not isinstance(payload[key], bool):
                raise ValueError(f"specification review '{key}' must be boolean")
        if not isinstance(payload["reviewed_code_sha256"], str):
            raise ValueError("reviewed_code_sha256 must be text")
        if not isinstance(payload["materiality_basis"], str) or payload["materiality_basis"] not in {
            "pass",
            "confirmed_signal_mismatch",
            "unresolved_material_ambiguity",
        }:
            raise ValueError("materiality_basis is not approved")
        if not isinstance(payload["review_notes"], str) or not payload["review_notes"].strip():
            raise ValueError("review_notes must be non-empty text")
        findings = payload["findings"]
        if not isinstance(findings, list) or not all(
            isinstance(item, str) and item.strip() for item in findings
        ):
            raise ValueError("findings must be a list of non-empty text")
        return payload

    @staticmethod
    def _review_prompt(hypothesis: dict[str, Any], code: str, code_hash: str) -> str:
        return (
            "Act as an adversarial independent quantitative code reviewer. "
            "Do not estimate performance and do not suggest threshold changes. "
            "Fail any boolean when evidence is missing or ambiguous. Verify that every "
            "manifest entry condition is implemented with the stated direction, canonical "
            "indicator meaning, completed-timeframe availability, and same-row setup output "
            "for evaluator-controlled next-open entry. Canonical indicator meaning in this "
            "project is TA-Lib semantics, not an unspecified industry, TradingView, pandas, or "
            "other-library default. Check TA-Lib lookback/window, variance convention, scale, "
            "tie handling, and warm-up behavior. Treat compact positional notation such as "
            "STOCHRSI(14,14,3,3) as insufficient evidence: verify that the manifest explicitly "
            "names every semantically relevant TA-Lib keyword value and the output consumed by "
            "the returned setup, and that these exactly match generated_code. For every 4h/1d "
            "feature, verify the strict order: aggregate distinct completed higher-timeframe "
            "OHLCV bars, calculate the indicator on that aggregate-frequency series, apply the "
            "declared availability delay, and only then align/forward-fill the indicator result "
            "to 1h. Availability is defined numerically: for a bar [T0,T1), its result first "
            "appears at T1 + D*P, where P is the HTF period and this project requires D=1. "
            "A right label names T1 and does not itself increment D. Therefore [00:00,04:00) "
            "first appears at 08:00, and the Jan 1 daily bar first appears at Jan 3 00:00 UTC. "
            "The candidate input index is UTC-aware. Calculating an indicator after "
            "forward-filling raw higher-timeframe OHLCV "
            "onto 1h is a material timeframe/indicator mismatch. If claiming a "
            "mismatch, name the TA-Lib "
            "function and concrete semantic difference; keep unverified convention claims "
            "explicitly ambiguous rather than presenting them as fact. TA-Lib consumes ordered "
            "numeric observations and is not passed timestamp spacing. If the manifest "
            "explicitly says to discard incomplete aggregate bars, passing the remaining "
            "chronologically ordered observations to TA-Lib is not by itself a semantic "
            "mismatch. Reject gap handling only when the manifest explicitly requires a "
            "calendar-contiguous grid or when a concrete signal difference from the stated "
            "manifest behavior is demonstrated; a hypothetical missing-bar effect is not a "
            "confirmed signal mismatch. Set a semantic boolean "
            "false only when the issue materially changes the returned setup or an explicitly "
            "required manifest behavior. Unused outputs, style, redundancy, or a convention "
            "difference with no demonstrated signal effect cannot cause rejection. Use "
            "materiality_basis=confirmed_signal_mismatch for a verified material difference, "
            "unresolved_material_ambiguity only when missing evidence directly prevents signal "
            "or timeframe correctness, and pass only when every semantic boolean is true. "
            "Return the supplied "
            "SHA256 unchanged. "
            f"Candidate code SHA256: {code_hash}\n"
            f"Hypothesis: {canonical_json(hypothesis)}\n"
            f"Generated code:\n{code}"
        )

    @staticmethod
    def _prompt(mechanism: Mechanism) -> str:
        source = {
            "mechanism_id": mechanism.mechanism_id,
            "name": mechanism.name,
            "intent": mechanism.intent,
            "original_timeframe": mechanism.timeframe,
            "indicators": mechanism.indicators,
            "conditions": mechanism.conditions,
            "source_commit": mechanism.source.commit_sha,
            "source_lines": [mechanism.source.start_line, mechanism.source.end_line],
        }
        return (
            "Create one clean-room, causal Binance spot long-only research hypothesis from "
            "the supplied mechanism. Execution must be 1h with optional closed 4h/1d data. "
            "Do not copy source code, use DCA, shorts, leverage, pair-specific tuning, future "
            "data, ROI/trailing optimization, or locked/blind data. generated_code must expose "
            "a pure function signal(dataframe) and may only calculate causal features. "
            "The dataframe input contract contains only open, high, low, close, and volume. "
            "Do not require precomputed framework, regime, protection, indicator, or condition "
            "columns; derive every needed feature causally inside signal from raw OHLCV. "
            "signal(dataframe) must always return a boolean pandas Series aligned to the full "
            "input index, including early/no-history paths; never return a scalar bool, "
            "return bool(...), or select only the last row. It must return the boolean setup "
            "on that same fully closed row; "
            "it must never shift/delay the returned setup because the evaluator alone enters "
            "at the next 1h candle open. Feature calculations may use positive historical shifts. "
            "Every named technical indicator must use its canonical published definition and "
            "the threshold must use the same scale. In this project, canonical means TA-Lib "
            "function semantics, including its lookback, population/sample convention, tie "
            "handling, output scale, and warm-up behavior. An equivalent custom implementation "
            "must match TA-Lib. Return hypothesis schema_version=2 and an "
            "indicator_contracts list; use an empty list only when no multi-output TA-Lib "
            "function is called. Every multi-output contract must name the function, every "
            "semantic keyword/value, and output_bindings from each canonical output name to "
            "the exact local target used by generated_code. generated_code must use direct "
            "`import talib`; multi-output calls must use tuple destructuring inside signal(), "
            "with optional parameters written as keywords. A signal() that uses such an output "
            "must be straight-line local-variable dataflow with a single final return; do not "
            "pass the output through helpers, dataframe columns, branches, loops, aliases, or "
            "side-effect calls. If a non-canonical variant is necessary, "
            "the hypothesis title and entry_logic must name the exact variant/formula instead "
            "of using the canonical indicator name alone. In entry_logic, spell every TA-Lib "
            "call with the function name and every semantically relevant keyword exactly as "
            "used in generated_code, and explicitly name which returned output enters the "
            "setup. Compact or positional notation such as STOCHRSI(14,14,3,3) is forbidden "
            "because parameter meanings are ambiguous. For any manually derived 4h/1d "
            "feature, first build the distinct completed aggregate OHLCV series, calculate each "
            "indicator on that aggregate-frequency series, apply the availability delay, and "
            "only then align or forward-fill the indicator result onto 1h. Never forward-fill "
            "aggregate OHLCV onto 1h before calculating its indicator. The input index is "
            "UTC-aware. Define availability as: a bar [T0,T1) first appears at T1 + D*P, "
            "where P is the HTF period and D=1. label='right' names T1 but does not count as "
            "a delay. Explicit anchors: [00:00,04:00) first appears at 08:00; the Jan 1 daily "
            "bar first appears at Jan 3 00:00 UTC. Define its availability "
            "timestamp explicitly and use only aggregates "
            "completed before the evaluated 1h row. Do not assume simultaneous timeframe closes "
            "are safely available; when availability is ambiguous, delay by one completed "
            "higher-timeframe bar. A reviewer will reject semantics it cannot verify. "
            f"Selected mechanism: {canonical_json(source)}"
        )
