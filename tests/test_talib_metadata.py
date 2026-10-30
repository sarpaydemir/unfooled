from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from research_factory.models import canonical_json
from research_factory.talib_metadata import (
    EXPECTED_C_LIBRARY_VERSION,
    EXPECTED_METADATA_SHA256,
    EXPECTED_PYTHON_WRAPPER_VERSION,
    PINNED_WORKER_IMAGE,
    load_pinned_metadata,
    verify_runtime_attestation,
)


NOW = datetime(2026, 7, 30, 9, 0, tzinfo=timezone.utc)


class TalibMetadataTests(unittest.TestCase):
    def write_fixture_attestation(
        self,
        directory: str,
        key: str = "EXP-075",
        verified_at: datetime = NOW,
    ) -> Path:
        metadata = load_pinned_metadata()
        payload = {
            "schema_version": 1,
            "idempotency_key": key,
            "worker_image_digest": PINNED_WORKER_IMAGE,
            "metadata_sha256": EXPECTED_METADATA_SHA256,
            "runtime_metadata_sha256": metadata.sha256,
            "python_wrapper_version": EXPECTED_PYTHON_WRAPPER_VERSION,
            "c_library_version": EXPECTED_C_LIBRARY_VERSION,
            "verified_at": verified_at.isoformat(),
        }
        path = Path(directory) / f"{key}.json"
        path.write_text(canonical_json(payload), encoding="utf-8")
        return path

    def mutated_attestation(self, directory: str, field: str) -> Path:
        path = self.write_fixture_attestation(directory)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[field] = "invalid"
        path.write_text(canonical_json(payload), encoding="utf-8")
        return path

    def test_pinned_registry_matches_external_digest(self) -> None:
        metadata = load_pinned_metadata()
        self.assertEqual(metadata.sha256, EXPECTED_METADATA_SHA256)
        self.assertEqual(metadata.outputs["AROON"], ("aroondown", "aroonup"))

    def test_attestation_rejects_wrong_key_and_expiry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_fixture_attestation(directory)
            with self.assertRaises(RuntimeError):
                verify_runtime_attestation(path, "EXP-other", now=NOW)
            with self.assertRaises(RuntimeError):
                verify_runtime_attestation(
                    path,
                    "EXP-075",
                    now=NOW + timedelta(minutes=11),
                )

    def test_attestation_rejects_runtime_or_image_drift(self) -> None:
        fields = (
            "worker_image_digest",
            "metadata_sha256",
            "runtime_metadata_sha256",
            "python_wrapper_version",
            "c_library_version",
        )
        with tempfile.TemporaryDirectory() as directory:
            for field in fields:
                with self.subTest(field=field):
                    path = self.mutated_attestation(directory, field)
                    with self.assertRaises(RuntimeError):
                        verify_runtime_attestation(path, "EXP-075", now=NOW)

    def test_attestation_rejects_future_or_naive_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            future = self.write_fixture_attestation(
                directory,
                verified_at=NOW + timedelta(seconds=1),
            )
            with self.assertRaises(RuntimeError):
                verify_runtime_attestation(future, "EXP-075", now=NOW)
            naive = self.write_fixture_attestation(
                directory,
                verified_at=NOW.replace(tzinfo=None),
            )
            with self.assertRaises(RuntimeError):
                verify_runtime_attestation(naive, "EXP-075", now=NOW)

