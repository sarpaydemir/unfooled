from __future__ import annotations

from typing import Any

from .models import GateResult


class GateArtifactError(ValueError):
    pass


def _number(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise GateArtifactError(f"metric '{key}' must be numeric")
    return float(value)


def _boolean(metrics: dict[str, Any], key: str) -> bool:
    value = metrics.get(key)
    if not isinstance(value, bool):
        raise GateArtifactError(f"metric '{key}' must be boolean")
    return value


def _text(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if not isinstance(value, str) or not value.strip():
        raise GateArtifactError(f"metric '{key}' must be non-empty text")
    return value


def _minimum(metrics: dict[str, Any], key: str, threshold: float, reasons: list[str]) -> None:
    value = _number(metrics, key)
    if value < threshold:
        reasons.append(f"{key}={value:g} < {threshold:g}")


def _maximum(metrics: dict[str, Any], key: str, threshold: float, reasons: list[str]) -> None:
    value = _number(metrics, key)
    if value > threshold:
        reasons.append(f"{key}={value:g} > {threshold:g}")


def _positive(metrics: dict[str, Any], key: str, reasons: list[str]) -> None:
    value = _number(metrics, key)
    if value <= 0:
        reasons.append(f"{key}={value:g} is not positive")


class GateEvaluator:
    def __init__(
        self,
        alpha_threshold: float | None = None,
        expected_code_sha256: str | None = None,
    ) -> None:
        self.alpha_threshold = alpha_threshold
        self.expected_code_sha256 = expected_code_sha256

    def evaluate(self, gate: str, metrics: dict[str, Any]) -> GateResult:
        evaluator = getattr(self, f"_evaluate_{gate}", None)
        if evaluator is None:
            raise GateArtifactError(f"unsupported deterministic gate: {gate}")
        reasons = evaluator(metrics)
        return GateResult(gate=gate, passed=not reasons, metrics=metrics, reasons=tuple(reasons))

    def _evaluate_specification_review(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        reviewed_hash = _text(metrics, "reviewed_code_sha256")
        if (
            len(reviewed_hash) != 64
            or any(character not in "0123456789abcdef" for character in reviewed_hash)
        ):
            raise GateArtifactError("metric 'reviewed_code_sha256' must be lowercase SHA256")
        if self.expected_code_sha256 is None:
            raise GateArtifactError("specification review requires expected candidate code SHA256")
        if reviewed_hash != self.expected_code_sha256:
            reasons.append("reviewed_code_sha256 does not match candidate code")
        semantic_keys = (
            "hypothesis_code_consistent",
            "indicator_semantics_verified",
            "timeframe_semantics_verified",
            "entry_timing_consistent",
        )
        for key in semantic_keys:
            if not _boolean(metrics, key):
                reasons.append(f"{key} is false")
        basis = _text(metrics, "materiality_basis")
        allowed_basis = {
            "pass",
            "confirmed_signal_mismatch",
            "unresolved_material_ambiguity",
        }
        if basis not in allowed_basis:
            raise GateArtifactError("metric 'materiality_basis' is not approved")
        all_consistent = all(bool(metrics[key]) for key in semantic_keys)
        if all_consistent and basis != "pass":
            reasons.append("materiality_basis must be pass when all semantic checks are true")
        if not all_consistent and basis == "pass":
            reasons.append("materiality_basis cannot be pass when a semantic check is false")
        _text(metrics, "review_notes")
        return reasons

    def _evaluate_causality_probe(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        code_hash = _text(metrics, "candidate_code_sha256")
        if (
            len(code_hash) != 64
            or any(character not in "0123456789abcdef" for character in code_hash)
        ):
            raise GateArtifactError("metric 'candidate_code_sha256' must be lowercase SHA256")
        if self.expected_code_sha256 is None:
            raise GateArtifactError("causality probe requires expected candidate code SHA256")
        if code_hash != self.expected_code_sha256:
            reasons.append("candidate_code_sha256 does not match candidate code")
        if _text(metrics, "fixture_version") != "synthetic-ohlcv-v1":
            reasons.append("fixture_version is not approved")
        _minimum(metrics, "prefix_checks", 12, reasons)
        _maximum(metrics, "mismatched_prefixes", 0, reasons)
        _maximum(metrics, "mismatched_rows", 0, reasons)
        _maximum(metrics, "execution_errors", 0, reasons)
        return reasons

    def _evaluate_event_study(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        _minimum(metrics, "effective_events", 60, reasons)
        _positive(metrics, "median_return_pct", reasons)
        _positive(metrics, "mean_return_pct", reasons)
        _minimum(metrics, "positive_pair_fraction", 2 / 3, reasons)
        _minimum(metrics, "positive_half_fraction", 1.0, reasons)
        _minimum(metrics, "mfe_mae_ratio", 1.2, reasons)
        return reasons

    def _evaluate_ablation(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        _positive(metrics, "median_lift_pct", reasons)
        _positive(metrics, "mean_lift_pct", reasons)
        _minimum(metrics, "positive_pair_fraction", 2 / 3, reasons)
        return reasons

    def _evaluate_statistics(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        _positive(metrics, "cluster_ci_lower_pct", reasons)
        if self.alpha_threshold is None:
            raise GateArtifactError("statistics gate requires a preregistered alpha threshold")
        _maximum(metrics, "family_adjusted_p_value", self.alpha_threshold, reasons)
        _minimum(metrics, "effective_clusters", 60, reasons)
        return reasons

    def _evaluate_specialist_backtest(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        _minimum(metrics, "trades", 60, reasons)
        _positive(metrics, "total_profit_pct", reasons)
        _positive(metrics, "median_profit_pct", reasons)
        _minimum(metrics, "profit_factor", 1.10, reasons)
        _maximum(metrics, "max_drawdown_pct", 15.0, reasons)
        _minimum(metrics, "positive_pair_fraction", 2 / 3, reasons)
        return reasons

    def _evaluate_multi_seed_hyperopt(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        _minimum(metrics, "seed_count", 3, reasons)
        _minimum(metrics, "positive_seed_fraction", 1.0, reasons)
        _minimum(metrics, "positive_neighborhood_fraction", 0.70, reasons)
        _boolean(metrics, "stable_plateau")
        if not metrics["stable_plateau"]:
            reasons.append("stable_plateau is false")
        return reasons

    def _evaluate_walk_forward(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        _minimum(metrics, "fold_count", 3, reasons)
        _minimum(metrics, "positive_fold_fraction", 2 / 3, reasons)
        _positive(metrics, "median_profit_pct", reasons)
        _minimum(metrics, "profit_factor", 1.20, reasons)
        _maximum(metrics, "max_drawdown_pct", 15.0, reasons)
        return reasons

    def _evaluate_final_locked(self, metrics: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        _minimum(metrics, "trades", 60, reasons)
        _positive(metrics, "total_profit_pct", reasons)
        _positive(metrics, "median_profit_pct", reasons)
        _minimum(metrics, "profit_factor", 1.20, reasons)
        _maximum(metrics, "max_drawdown_pct", 15.0, reasons)
        _minimum(metrics, "positive_pair_fraction", 2 / 3, reasons)
        _positive(metrics, "fee_stress_total_profit_pct", reasons)
        _minimum(metrics, "fee_stress_profit_factor", 1.05, reasons)
        _minimum(metrics, "positive_neighborhood_fraction", 0.70, reasons)
        _positive(metrics, "combined_cluster_ci_lower_pct", reasons)
        _minimum(metrics, "combined_effective_samples", 300, reasons)
        return reasons
