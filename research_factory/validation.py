from __future__ import annotations

import math
from typing import Any

from .models import (
    Hypothesis,
    IndicatorContract,
    IndicatorOutputBinding,
    IndicatorParameter,
)


class PayloadValidationError(ValueError):
    pass


def parse_candidate_payload(payload: dict[str, Any]) -> tuple[Hypothesis, str]:
    if set(payload) != {"hypothesis", "generated_code"}:
        raise PayloadValidationError("candidate payload must contain hypothesis and generated_code")
    raw = payload["hypothesis"]
    required = {
        "hypothesis_id", "mechanism_ids", "title", "rationale", "expected_regimes",
        "entry_logic", "exit_neutral_horizons", "controls", "failure_conditions",
        "required_timeframes", "schema_version", "indicator_contracts",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise PayloadValidationError("hypothesis fields do not match the contract")
    if raw["schema_version"] != 2:
        raise PayloadValidationError("new candidate hypothesis schema_version must be 2")
    hypothesis = _parse_hypothesis(raw, schema_version=2)
    code = payload["generated_code"]
    if not isinstance(code, str) or not code.strip():
        raise PayloadValidationError("generated_code must be non-empty")
    return hypothesis, code


def hypothesis_from_stored_payload(raw: dict[str, Any]) -> Hypothesis:
    if not isinstance(raw, dict):
        raise PayloadValidationError("stored hypothesis must be an object")
    schema_version = raw.get("schema_version", 1)
    base_required = {
        "hypothesis_id", "mechanism_ids", "title", "rationale", "expected_regimes",
        "entry_logic", "exit_neutral_horizons", "controls", "failure_conditions",
        "required_timeframes",
    }
    versioned_required = base_required | {"schema_version", "indicator_contracts"}
    valid_fields = (
        set(raw) == base_required
        or (
            schema_version == 1
            and set(raw) == versioned_required
            and raw["indicator_contracts"] in ([], ())
        )
        or (schema_version == 2 and set(raw) == versioned_required)
    )
    if schema_version not in {1, 2} or not valid_fields:
        raise PayloadValidationError("stored hypothesis fields do not match the contract")
    return _parse_hypothesis(raw, schema_version=schema_version)


def _parse_hypothesis(raw: dict[str, Any], schema_version: int) -> Hypothesis:
    horizons = tuple(int(item) for item in raw["exit_neutral_horizons"])
    if not horizons or not set(horizons).issubset({12, 24, 48}):
        raise PayloadValidationError("exit horizons must be a non-empty subset of 12/24/48")
    timeframes = tuple(str(item) for item in raw["required_timeframes"])
    if not timeframes or not set(timeframes).issubset({"1h", "4h", "1d"}):
        raise PayloadValidationError("timeframes must be a non-empty subset of 1h/4h/1d")
    contracts = (
        _parse_indicator_contracts(raw["indicator_contracts"])
        if schema_version == 2
        else ()
    )
    hypothesis = Hypothesis(
        hypothesis_id=str(raw["hypothesis_id"]),
        mechanism_ids=tuple(str(item) for item in raw["mechanism_ids"]),
        title=str(raw["title"]),
        rationale=str(raw["rationale"]),
        expected_regimes=tuple(str(item) for item in raw["expected_regimes"]),
        entry_logic=tuple(str(item) for item in raw["entry_logic"]),
        exit_neutral_horizons=horizons,
        controls=tuple(str(item) for item in raw["controls"]),
        failure_conditions=tuple(str(item) for item in raw["failure_conditions"]),
        required_timeframes=timeframes,
        schema_version=schema_version,
        indicator_contracts=contracts,
    )
    if not hypothesis.mechanism_ids:
        raise PayloadValidationError("at least one source mechanism is required")
    return hypothesis


def _parse_indicator_contracts(raw_contracts: Any) -> tuple[IndicatorContract, ...]:
    if not isinstance(raw_contracts, list):
        raise PayloadValidationError("indicator_contracts must be a list")
    contracts: list[IndicatorContract] = []
    identities: set[tuple] = set()
    for raw in raw_contracts:
        if not isinstance(raw, dict) or set(raw) != {
            "function", "keyword_parameters", "output_bindings"
        }:
            raise PayloadValidationError("indicator contract fields do not match the contract")
        function = raw["function"]
        if (
            not isinstance(function, str)
            or function != function.upper()
            or not function.isidentifier()
        ):
            raise PayloadValidationError("indicator function must be an uppercase identifier")
        parameters = _parse_parameters(raw["keyword_parameters"])
        bindings = _parse_bindings(raw["output_bindings"])
        identity = (
            function,
            tuple((item.name, _canonical_number(item.value)) for item in parameters),
            tuple((item.output, item.target) for item in bindings),
        )
        if identity in identities:
            raise PayloadValidationError("duplicate indicator contract")
        identities.add(identity)
        contracts.append(IndicatorContract(function, parameters, bindings))
    return tuple(
        sorted(
            contracts,
            key=lambda item: (
                item.function,
                tuple((p.name, _canonical_number(p.value)) for p in item.keyword_parameters),
                tuple((b.output, b.target) for b in item.output_bindings),
            ),
        )
    )


def _parse_parameters(raw_parameters: Any) -> tuple[IndicatorParameter, ...]:
    if not isinstance(raw_parameters, list):
        raise PayloadValidationError("keyword_parameters must be a list")
    parameters: list[IndicatorParameter] = []
    names: set[str] = set()
    for raw in raw_parameters:
        if not isinstance(raw, dict) or set(raw) != {"name", "value"}:
            raise PayloadValidationError("indicator parameter fields do not match the contract")
        name = raw["name"]
        value = raw["value"]
        if not isinstance(name, str) or not name.isidentifier():
            raise PayloadValidationError("indicator parameter name must be an identifier")
        if name in names:
            raise PayloadValidationError("duplicate indicator parameter name")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise PayloadValidationError("indicator parameter value must be a finite number")
        names.add(name)
        parameters.append(IndicatorParameter(name, _canonical_number(value)))
    return tuple(sorted(parameters, key=lambda item: item.name))


def _parse_bindings(raw_bindings: Any) -> tuple[IndicatorOutputBinding, ...]:
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise PayloadValidationError("output_bindings must be a non-empty list")
    bindings: list[IndicatorOutputBinding] = []
    outputs: set[str] = set()
    targets: set[str] = set()
    for raw in raw_bindings:
        if not isinstance(raw, dict) or set(raw) != {"output", "target"}:
            raise PayloadValidationError("indicator binding fields do not match the contract")
        output = raw["output"]
        target = raw["target"]
        if not isinstance(output, str) or not output.isidentifier():
            raise PayloadValidationError("indicator output must be an identifier")
        if (
            not isinstance(target, str)
            or not target.isidentifier()
            or target == "_"
        ):
            raise PayloadValidationError("indicator target must be a non-placeholder identifier")
        if output in outputs or target in targets:
            raise PayloadValidationError("duplicate indicator output or target")
        outputs.add(output)
        targets.add(target)
        bindings.append(IndicatorOutputBinding(output, target))
    return tuple(sorted(bindings, key=lambda item: item.output))


def _canonical_number(value: int | float) -> int | float:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value
