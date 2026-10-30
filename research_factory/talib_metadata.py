from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .models import canonical_json


EXPECTED_METADATA_SHA256 = (
    "2637c70b8ebffde3399f42a8d13f49f9b33079c200b24ae8a265a440ee8c3ec1"
)
EXPECTED_PYTHON_WRAPPER_VERSION = "0.6.8"
EXPECTED_C_LIBRARY_VERSION = "0.6.4 (Oct 20 2025 20:23:51)"
PINNED_WORKER_IMAGE = (
    "freqtradeorg/freqtrade@"
    "sha256:87aa5c6d65359b34e9d99a0bb260a38c0efe0315253811e6f48c2afe8f278a6a"
)
ATTESTATION_MAX_AGE_SECONDS = 600
_METADATA_PATH = Path(__file__).with_name("talib_multi_output_v1.json")
_METADATA_FIELDS = {
    "schema_version",
    "python_wrapper_version",
    "c_library_version",
    "source",
    "multi_output_functions",
}
_ATTESTATION_FIELDS = {
    "schema_version",
    "idempotency_key",
    "worker_image_digest",
    "metadata_sha256",
    "runtime_metadata_sha256",
    "python_wrapper_version",
    "c_library_version",
    "verified_at",
}


@dataclass(frozen=True)
class TalibMetadata:
    schema_version: int
    python_wrapper_version: str
    c_library_version: str
    source: str
    outputs: Mapping[str, tuple[str, ...]]
    sha256: str


def load_pinned_metadata() -> TalibMetadata:
    try:
        raw = json.loads(_METADATA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("TA-Lib metadata artifact cannot be read") from error
    if not isinstance(raw, dict) or set(raw) != _METADATA_FIELDS:
        raise RuntimeError("TA-Lib metadata fields do not match schema v1")
    if (
        raw["schema_version"] != 1
        or raw["python_wrapper_version"] != EXPECTED_PYTHON_WRAPPER_VERSION
        or raw["c_library_version"] != EXPECTED_C_LIBRARY_VERSION
        or raw["source"] != "talib.abstract.Function.output_names"
        or not isinstance(raw["multi_output_functions"], dict)
    ):
        raise RuntimeError("TA-Lib metadata identity does not match the pinned runtime")
    digest = sha256(canonical_json(raw).encode()).hexdigest()
    if digest != EXPECTED_METADATA_SHA256:
        raise RuntimeError("TA-Lib metadata checksum mismatch")
    outputs: dict[str, tuple[str, ...]] = {}
    for function, names in raw["multi_output_functions"].items():
        if (
            not isinstance(function, str)
            or not isinstance(names, list)
            or len(names) < 2
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise RuntimeError("TA-Lib output metadata is malformed")
        outputs[function] = tuple(names)
    return TalibMetadata(
        schema_version=1,
        python_wrapper_version=raw["python_wrapper_version"],
        c_library_version=raw["c_library_version"],
        source=raw["source"],
        outputs=MappingProxyType(dict(sorted(outputs.items()))),
        sha256=digest,
    )


def write_runtime_attestation(idempotency_key: str, output: Path) -> dict:
    if not idempotency_key.strip():
        raise RuntimeError("idempotency key must be non-empty")
    metadata = load_pinned_metadata()
    try:
        import talib
        from talib import abstract
    except ImportError as error:
        raise RuntimeError("TA-Lib is unavailable in the research worker") from error

    wrapper_version = str(talib.__version__)
    c_version_raw = talib.__ta_version__
    c_library_version = (
        c_version_raw.decode()
        if isinstance(c_version_raw, bytes)
        else str(c_version_raw)
    )
    runtime_outputs = {
        function: list(abstract.Function(function).output_names)
        for function in metadata.outputs
    }
    runtime_metadata = {
        "schema_version": metadata.schema_version,
        "python_wrapper_version": wrapper_version,
        "c_library_version": c_library_version,
        "source": metadata.source,
        "multi_output_functions": runtime_outputs,
    }
    runtime_digest = sha256(canonical_json(runtime_metadata).encode()).hexdigest()
    if (
        wrapper_version != metadata.python_wrapper_version
        or c_library_version != metadata.c_library_version
        or runtime_digest != metadata.sha256
    ):
        raise RuntimeError("TA-Lib worker runtime drifted from pinned metadata")

    payload = {
        "schema_version": 1,
        "idempotency_key": idempotency_key,
        "worker_image_digest": PINNED_WORKER_IMAGE,
        "metadata_sha256": metadata.sha256,
        "runtime_metadata_sha256": runtime_digest,
        "python_wrapper_version": wrapper_version,
        "c_library_version": c_library_version,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(canonical_json(payload), encoding="utf-8")
    temporary.replace(output)
    return payload


def verify_runtime_attestation(
    path: Path,
    idempotency_key: str,
    now: datetime | None = None,
) -> dict:
    metadata = load_pinned_metadata()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("TA-Lib runtime attestation cannot be read") from error
    if not isinstance(payload, dict) or set(payload) != _ATTESTATION_FIELDS:
        raise RuntimeError("TA-Lib runtime attestation fields are invalid")
    expected = {
        "schema_version": 1,
        "idempotency_key": idempotency_key,
        "worker_image_digest": PINNED_WORKER_IMAGE,
        "metadata_sha256": metadata.sha256,
        "runtime_metadata_sha256": metadata.sha256,
        "python_wrapper_version": metadata.python_wrapper_version,
        "c_library_version": metadata.c_library_version,
    }
    for field, value in expected.items():
        if payload[field] != value:
            raise RuntimeError(f"TA-Lib runtime attestation mismatch: {field}")
    try:
        verified_at = datetime.fromisoformat(payload["verified_at"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("TA-Lib attestation timestamp is invalid") from error
    if verified_at.tzinfo is None or verified_at.utcoffset() != timezone.utc.utcoffset(None):
        raise RuntimeError("TA-Lib attestation timestamp must be UTC-aware")
    checked_at = now or datetime.now(timezone.utc)
    if checked_at.tzinfo is None:
        raise RuntimeError("attestation verification time must be timezone-aware")
    age = (checked_at.astimezone(timezone.utc) - verified_at.astimezone(timezone.utc)).total_seconds()
    if age < 0 or age > ATTESTATION_MAX_AGE_SECONDS:
        raise RuntimeError("TA-Lib runtime attestation is future-dated or expired")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m research_factory.talib_metadata")
    commands = parser.add_subparsers(dest="command", required=True)
    attest = commands.add_parser("attest")
    attest.add_argument("--idempotency-key", required=True)
    attest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_runtime_attestation(args.idempotency_key, args.output)
    print(canonical_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
