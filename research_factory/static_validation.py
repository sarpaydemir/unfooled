from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType
from typing import Callable

from .indicator_contract import IndicatorContractValidator
from .models import Hypothesis
from .safety import CandidateSafetyValidator, SafetyReport
from .talib_metadata import load_pinned_metadata
from .validation import PayloadValidationError, hypothesis_from_stored_payload


class CompositeStaticValidator:
    def __init__(self) -> None:
        self.indicator_validator = IndicatorContractValidator(
            load_pinned_metadata()
        )

    def validate(self, hypothesis: Hypothesis, source: str) -> SafetyReport:
        code_report = CandidateSafetyValidator().validate(source)
        contract_report = self.indicator_validator.validate(hypothesis, source)
        violations = tuple(
            sorted(set(code_report.violations + contract_report.violations))
        )
        return SafetyReport(not violations, violations)


def load_v2_candidate(
    candidate_path: Path,
    manifest_path: Path,
) -> Callable[..., object]:
    try:
        source = candidate_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"candidate code cannot be read: {candidate_path}") from error
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"candidate manifest cannot be read: {manifest_path}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("hypothesis"), dict):
        raise RuntimeError("candidate manifest is missing its hypothesis")
    try:
        hypothesis = hypothesis_from_stored_payload(manifest["hypothesis"])
    except PayloadValidationError as error:
        raise RuntimeError(f"candidate manifest is invalid: {error}") from error
    if hypothesis.schema_version != 2:
        raise RuntimeError("schema v1 is historical and cannot be executed")

    code_digest = sha256(source.encode()).hexdigest()
    bindings = 0
    if "candidate_code_sha256" in manifest:
        bindings += 1
        if manifest["candidate_code_sha256"] != code_digest:
            raise RuntimeError("candidate manifest code SHA does not match candidate.py")
    if "generated_code" in manifest:
        bindings += 1
        generated_code = manifest["generated_code"]
        if not isinstance(generated_code, str) or sha256(
            generated_code.encode()
        ).hexdigest() != code_digest:
            raise RuntimeError("candidate manifest generated_code does not match candidate.py")
    if bindings == 0:
        raise RuntimeError("candidate manifest has no code SHA binding")

    report = CompositeStaticValidator().validate(hypothesis, source)
    if not report.passed:
        raise RuntimeError("unsafe candidate: " + "; ".join(report.violations))
    try:
        compiled = compile(source, str(candidate_path), "exec")
    except SyntaxError as error:
        raise RuntimeError(f"candidate cannot compile: {error}") from error
    module = ModuleType("research_candidate")
    try:
        exec(compiled, module.__dict__)
    except Exception as error:
        raise RuntimeError("candidate module initialization failed") from error
    signal = getattr(module, "signal", None)
    if not callable(signal):
        raise RuntimeError("candidate must define callable signal(dataframe)")
    return signal
