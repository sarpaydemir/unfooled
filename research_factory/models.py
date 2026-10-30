from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class CandidateStatus(StrEnum):
    DRAFT = "draft"
    REJECTED = "rejected"
    REVIEW_HOLD = "review_hold"
    TESTING = "testing"
    PASSED = "passed"
    APPROVED = "approved"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class SourceRef:
    repository: str
    commit_sha: str
    checksum: str
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class Mechanism:
    mechanism_id: str
    name: str
    intent: str
    timeframe: str
    indicators: tuple[str, ...]
    conditions: tuple[str, ...]
    source: SourceRef
    portability_score: float
    novelty_score: float
    sample_capacity_score: float
    safety_score: float
    rejected_reason: str | None = None

    @property
    def score(self) -> float:
        return round(
            0.25 * self.portability_score
            + 0.30 * self.novelty_score
            + 0.20 * self.sample_capacity_score
            + 0.25 * self.safety_score,
            4,
        )


@dataclass(frozen=True)
class IndicatorParameter:
    name: str
    value: int | float


@dataclass(frozen=True)
class IndicatorOutputBinding:
    output: str
    target: str


@dataclass(frozen=True)
class IndicatorContract:
    function: str
    keyword_parameters: tuple[IndicatorParameter, ...]
    output_bindings: tuple[IndicatorOutputBinding, ...]


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    mechanism_ids: tuple[str, ...]
    title: str
    rationale: str
    expected_regimes: tuple[str, ...]
    entry_logic: tuple[str, ...]
    exit_neutral_horizons: tuple[int, ...]
    controls: tuple[str, ...]
    failure_conditions: tuple[str, ...]
    required_timeframes: tuple[str, ...]
    schema_version: int = 1
    indicator_contracts: tuple[IndicatorContract, ...] = ()


@dataclass
class CandidateManifest:
    candidate_id: str
    hypothesis: Hypothesis
    generated_code: str
    provider: str
    model: str
    prompt_checksum: str
    semantic_fingerprint: str
    status: CandidateStatus = CandidateStatus.DRAFT
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        hypothesis: Hypothesis,
        generated_code: str,
        provider: str,
        model: str,
        prompt: str,
    ) -> "CandidateManifest":
        indicator_contracts = []
        for contract in hypothesis.indicator_contracts:
            parameters = [
                [item.name, _canonical_number(item.value)]
                for item in sorted(contract.keyword_parameters, key=lambda item: item.name)
            ]
            outputs = sorted(item.output for item in contract.output_bindings)
            indicator_contracts.append(
                {
                    "function": contract.function,
                    "keyword_parameters": parameters,
                    "outputs": outputs,
                }
            )
        indicator_contracts.sort(
            key=lambda item: canonical_json(item)
        )
        semantic_material = {
            "mechanisms": sorted(hypothesis.mechanism_ids),
            "entry_logic": sorted(x.casefold().strip() for x in hypothesis.entry_logic),
            "timeframes": sorted(hypothesis.required_timeframes),
            "indicator_contracts": indicator_contracts,
        }
        fingerprint = sha256(canonical_json(semantic_material).encode()).hexdigest()
        candidate_id = f"cand-{fingerprint[:12]}"
        return cls(
            candidate_id=candidate_id,
            hypothesis=hypothesis,
            generated_code=generated_code,
            provider=provider,
            model=model,
            prompt_checksum=sha256(prompt.encode()).hexdigest(),
            semantic_fingerprint=fingerprint,
        )


def _canonical_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    metrics: dict[str, float | int | str | bool]
    reasons: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class PromotionCard:
    candidate_id: str
    recommendation: str
    trades: int
    profit_factor: float
    max_drawdown_pct: float
    median_profit_pct: float
    fee_stress_positive: bool
    confidence_lower_pct: float
    champion_delta_pct: float
    regimes: tuple[str, ...]
    rollback_version: str


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
