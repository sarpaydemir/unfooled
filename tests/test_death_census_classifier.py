from __future__ import annotations

import csv
from io import BytesIO
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError

from research_factory.death_census_classifier import (
    ClassificationError,
    IDENTITY_EVENT_FIXTURES,
    MAX_COMPLETION_TOKENS,
    build_prompt,
    build_request_envelope,
    calibration_audit,
    classification_schema,
    load_candidates,
    materialize_payload,
    partition_candidates,
    request_call_hash,
    retry_call_hash,
    run_classifier,
    slice_candidates,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class DeathCensusClassifierTest(unittest.TestCase):
    def fixture(self, root: Path):
        corpus = root / "corpus"
        articles = corpus / "articles"
        articles.mkdir(parents=True)
        article = {
            "code": "article-1",
            "title": "OLD Renamed as NEW",
            "body": "OLD will be renamed as NEW.",
        }
        (articles / "article-1.json").write_text(
            json.dumps(article, separators=(",", ":")), encoding="utf-8"
        )
        candidate = root / "candidates.csv"
        fields = [
            "release_date_utc",
            "article_code",
            "catalog_ids",
            "event_kinds",
            "matched_bases",
            "match_reasons",
            "title",
            "evidence_snippets",
        ]
        with candidate.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "release_date_utc": "2024-01-01T00:00:00+00:00",
                    "article_code": "article-1",
                    "catalog_ids": "49",
                    "event_kinds": "rename",
                    "matched_bases": "OLD,NEW",
                    "match_reasons": "OLD:direct_identity",
                    "title": article["title"],
                    "evidence_snippets": "[]",
                }
            )
        return corpus, candidate

    def payload(self):
        return {
            "classifications": [
                {
                    "article_id": "article-1",
                    "events": [
                        {
                            "event_type": "rename",
                            "subject_symbol": "OLD",
                            "successor_ticker": "NEW",
                            "evidence_quote": "OLD will be renamed as NEW.",
                        }
                    ],
                    "no_event_reason": None,
                }
            ],
            "acceptance_controls": [
                {
                    "event": fixture["event"],
                    "event_type": "redenomination"
                    if fixture["event"] == "BTT_REDENOMINATION"
                    else "rename",
                    "evidence_quote": fixture["text"],
                }
                for fixture in IDENTITY_EVENT_FIXTURES
            ],
        }

    def response(self):
        return {
            "id": "request-1",
            "model": "z-ai/glm-5.2",
            "choices": [{"message": {"content": json.dumps(self.payload())}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01},
        }

    def test_schema_and_hash_are_deterministic(self):
        schema = classification_schema(1)
        self.assertEqual(schema["properties"]["classifications"]["minItems"], 1)
        self.assertEqual(
            request_call_hash("model", "prompt", schema),
            request_call_hash("model", "prompt", schema),
        )
        self.assertEqual(
            build_request_envelope("model", "prompt", schema)["max_tokens"],
            16_000,
        )
        self.assertNotIn(
            "max_completion_tokens", build_request_envelope("model", "prompt", schema)
        )
        self.assertEqual(MAX_COMPLETION_TOKENS, 16_000)
        classification = schema["properties"]["classifications"]["items"]
        self.assertEqual(
            classification["required"], ["article_id", "events", "no_event_reason"]
        )
        event_required = classification["properties"]["events"]["items"]["required"]
        self.assertNotIn("real_identity_event", event_required)
        self.assertNotIn("in_scan_set", event_required)
        self.assertNotIn("is_identity_event", event_required)
        event_types = classification["properties"]["events"]["items"]["properties"][
            "event_type"
        ]["enum"]
        self.assertNotIn("not_identity", event_types)
        self.assertIn("delisting_only", event_types)
        control_events = schema["properties"]["acceptance_controls"]["items"][
            "properties"
        ]["event"]["enum"]
        self.assertEqual(
            control_events,
            [fixture["event"] for fixture in IDENTITY_EVENT_FIXTURES],
        )
        self.assertNotEqual(
            retry_call_hash(
                "model", "prompt", schema, max_tokens=32_000, retry_generation=1
            ),
            request_call_hash("model", "prompt", schema),
        )

    def test_out_of_scan_identity_event_is_semantically_valid(self):
        candidate = {
            "article_id": "article-1",
            "matched_bases": ("BTC",),
            "title": "TOMO renamed to VIC",
            "body": "TOMO will be renamed to VIC.",
        }
        payload = self.payload()
        payload["classifications"][0]["events"][0].update(
            {
                "subject_symbol": "TOMO",
                "event_type": "rename",
                "successor_ticker": "VIC",
                "evidence_quote": "TOMO will be renamed to VIC.",
            }
        )
        audit = calibration_audit(payload, [candidate])
        self.assertEqual(audit["semantic_error_count"], 0)
        self.assertTrue(audit["calibration_pass"])
        materialized = materialize_payload(payload, [candidate])
        event = materialized["classifications"][0]["events"][0]
        self.assertFalse(event["in_scan_set"])
        self.assertFalse(event["is_identity_event"])
        self.assertFalse(event["needs_manual_review"])

    def test_multiple_events_are_preserved_and_materialized_independently(self):
        candidate = {
            "article_id": "article-1",
            "matched_bases": ("OLD",),
            "title": "OLD and ALT token swaps",
            "body": "OLD will swap to NEW. ALT will swap to NEXT.",
        }
        payload = self.payload()
        payload["classifications"][0]["events"] = [
            {
                "event_type": "token_swap",
                "subject_symbol": "OLD",
                "successor_ticker": "NEW",
                "evidence_quote": "OLD will swap to NEW.",
            },
            {
                "event_type": "token_swap",
                "subject_symbol": "ALT",
                "successor_ticker": "NEXT",
                "evidence_quote": "ALT will swap to NEXT.",
            },
        ]
        audit = calibration_audit(payload, [candidate])
        self.assertEqual(audit["event_count"], 2)
        self.assertEqual(audit["normalized_evidence_pass_count"], 2)
        materialized = materialize_payload(payload, [candidate])["classifications"][0]
        self.assertEqual(len(materialized["events"]), 2)
        self.assertTrue(materialized["events"][0]["is_identity_event"])
        self.assertFalse(materialized["events"][1]["is_identity_event"])

    def test_empty_event_array_requires_reason(self):
        candidate = {
            "article_id": "article-1",
            "matched_bases": ("OLD",),
            "title": "Listing notice",
            "body": "OLD is now available in Convert.",
        }
        payload = self.payload()
        payload["classifications"][0] = {
            "article_id": "article-1",
            "events": [],
            "no_event_reason": "Convert availability is not an identity event.",
        }
        audit = calibration_audit(payload, [candidate])
        self.assertEqual(audit["event_count"], 0)
        self.assertTrue(audit["calibration_pass"])
        payload["classifications"][0]["no_event_reason"] = None
        self.assertEqual(calibration_audit(payload, [candidate])["semantic_error_count"], 1)

    def test_null_identity_subject_is_retained_for_manual_review(self):
        candidate = {
            "article_id": "article-1",
            "matched_bases": ("OLD",),
            "title": "A token will be renamed",
            "body": "The project will be renamed to NEW.",
        }
        payload = self.payload()
        payload["classifications"][0]["events"][0].update(
            {
                "subject_symbol": None,
                "successor_ticker": "NEW",
                "evidence_quote": "The project will be renamed to NEW.",
            }
        )
        event = materialize_payload(payload, [candidate])["classifications"][0][
            "events"
        ][0]
        self.assertTrue(event["is_identity_change"])
        self.assertTrue(event["needs_manual_review"])
        self.assertFalse(event["is_identity_event"])

    def test_successor_must_occur_and_duplicate_events_are_rejected(self):
        candidate = {
            "article_id": "article-1",
            "matched_bases": ("OLD",),
            "title": "OLD Renamed as NEW",
            "body": "OLD will be renamed as NEW.",
        }
        absent = self.payload()
        absent["classifications"][0]["events"][0]["successor_ticker"] = "MISSING"
        self.assertEqual(calibration_audit(absent, [candidate])["semantic_error_count"], 1)
        duplicate = self.payload()
        duplicate["classifications"][0]["events"].append(
            dict(duplicate["classifications"][0]["events"][0])
        )
        self.assertEqual(
            calibration_audit(duplicate, [candidate])["semantic_error_count"], 1
        )

    def test_two_chunks_are_balanced_sorted_and_cover_once(self):
        candidates = [{"article_id": value} for value in ("9", "1", "7", "3", "5")]
        first = partition_candidates(candidates, 1, 2)
        second = partition_candidates(candidates, 2, 2)
        self.assertEqual([row["article_id"] for row in first], ["1", "3", "5"])
        self.assertEqual([row["article_id"] for row in second], ["7", "9"])
        self.assertEqual(
            {row["article_id"] for row in first + second},
            {row["article_id"] for row in candidates},
        )

    def test_candidate_slice_is_sorted_and_bounded(self):
        candidates = [{"article_id": value} for value in ("9", "1", "7", "3", "5")]
        self.assertEqual(
            [row["article_id"] for row in slice_candidates(candidates, 1, 2)],
            ["3", "5"],
        )

    def test_calibration_reports_raw_and_normalized_evidence_separately(self):
        candidate = {
            "article_id": "article-1",
            "matched_bases": ("OLD", "NEW"),
            "title": "OLD Renamed as NEW",
            "body": "OLD will be \u201crenamed\u201d as NEW.",
        }
        payload = self.payload()
        payload["classifications"][0]["events"][0]["evidence_quote"] = (
            'OLD will be "renamed" as NEW.'
        )
        audit = calibration_audit(payload, [candidate])
        self.assertEqual(audit["raw_evidence_failure_count"], 1)
        self.assertEqual(audit["normalized_evidence_failure_count"], 0)
        self.assertTrue(audit["calibration_pass"])

    def test_control_evidence_may_come_from_any_bound_article_but_not_elsewhere(self):
        candidate = {
            "article_id": "article-1",
            "matched_bases": ("OLD", "NEW"),
            "title": "OLD Renamed as NEW",
            "body": (
                "OLD will be renamed as NEW. "
                "LUNA (old) was renamed as LUNC while LUNA (new) assumes the ticker LUNA."
            ),
        }
        payload = self.payload()
        payload["acceptance_controls"][0]["evidence_quote"] = (
            "LUNA (old) was renamed as LUNC while LUNA (new) assumes the ticker LUNA."
        )
        audit = calibration_audit(payload, [candidate])
        self.assertEqual(audit["acceptance_valid_count"], 6)
        materialize_payload(payload, [candidate])

        payload["acceptance_controls"][0]["evidence_quote"] = "invented control quote"
        audit = calibration_audit(payload, [candidate])
        self.assertEqual(audit["acceptance_valid_count"], 5)
        with self.assertRaisesRegex(ClassificationError, "acceptance evidence"):
            materialize_payload(payload, [candidate])

    def test_failed_calibration_is_persisted_without_accepted_classification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            response = self.response()
            payload = self.payload()
            payload["classifications"][0]["events"][0]["evidence_quote"] = "invented"
            response["choices"][0]["message"]["content"] = json.dumps(payload)
            summary = run_classifier(
                candidate,
                corpus,
                root / "calibration",
                root / "usage.jsonl",
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: FakeResponse(response),
                article_offset=0,
                article_limit=1,
                calibration=True,
            )
            self.assertEqual(summary["outcome"], "calibration_failed")
            self.assertFalse((root / "calibration" / "classification.json").exists())
            self.assertEqual(
                json.loads((root / "usage.jsonl").read_text())["idempotency_status"],
                "calibration_failed",
            )
            replay = run_classifier(
                candidate,
                corpus,
                root / "calibration",
                root / "usage.jsonl",
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: self.fail(
                    "failed calibration replay called provider"
                ),
                article_offset=0,
                article_limit=1,
                calibration=True,
            )
            self.assertEqual(replay, summary)
            self.assertEqual(len((root / "usage.jsonl").read_text().splitlines()), 1)

    def test_completion_exhaustion_is_a_logged_calibration_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            response = {
                "id": "request-length",
                "model": "z-ai/glm-5.2",
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": None, "reasoning": "unfinished"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 16_000,
                    "cost": 0.04,
                },
            }
            summary = run_classifier(
                candidate,
                corpus,
                root / "calibration",
                root / "usage.jsonl",
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: FakeResponse(response),
                article_offset=0,
                article_limit=1,
                calibration=True,
            )
            self.assertEqual(summary["outcome"], "calibration_failed")
            self.assertEqual(
                summary["calibration_audit"]["provider_output_error"],
                "provider structured output is not an object",
            )
            self.assertFalse((root / "calibration" / "classification.json").exists())
            ledger = json.loads((root / "usage.jsonl").read_text())
            self.assertEqual(ledger["usd"], 0.04)
            self.assertEqual(ledger["idempotency_status"], "calibration_failed")

    def test_batch_mode_separates_pass_from_semantic_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            passed = run_classifier(
                candidate,
                corpus,
                root / "passed",
                root / "usage.jsonl",
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: FakeResponse(self.response()),
                article_offset=0,
                article_limit=1,
                batch=True,
                retry_generation=1,
            )
            self.assertEqual(passed["outcome"], "batch_passed")
            self.assertTrue((root / "passed" / "classification.json").is_file())

            response = self.response()
            payload = self.payload()
            payload["classifications"][0]["events"][0]["subject_symbol"] = "MISSING"
            response["choices"][0]["message"]["content"] = json.dumps(payload)
            conflict = run_classifier(
                candidate,
                corpus,
                root / "conflict",
                root / "usage.jsonl",
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: FakeResponse(response),
                article_offset=0,
                article_limit=1,
                batch=True,
            )
            self.assertEqual(conflict["outcome"], "batch_semantic_conflict")
            self.assertFalse((root / "conflict" / "classification.json").exists())

    def test_single_call_persists_separate_usage_and_replays_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            calls = []

            def opener(_request, timeout):
                calls.append(timeout)
                return FakeResponse(self.response())

            output = root / "output"
            ledger = root / "usage.jsonl"
            first = run_classifier(
                candidate,
                corpus,
                output,
                ledger,
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=opener,
            )
            second = run_classifier(
                candidate,
                corpus,
                output,
                ledger,
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: self.fail("idempotent replay called provider"),
            )
            self.assertEqual(calls, [600])
            self.assertEqual(first["call_hash"], second["call_hash"])
            self.assertEqual(
                first["usage_ledger_row_sha256"], second["usage_ledger_row_sha256"]
            )
            self.assertEqual(len(ledger.read_text().splitlines()), 1)
            self.assertEqual(json.loads(ledger.read_text())["usd"], 0.01)

    def test_only_transport_error_is_retried_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            calls = 0

            def opener(_request, timeout):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise URLError("temporary")
                return FakeResponse(self.response())

            run_classifier(
                candidate,
                corpus,
                root / "output",
                root / "usage.jsonl",
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=opener,
                sleep=lambda _seconds: None,
            )
            self.assertEqual(calls, 2)

    def test_shared_append_only_ledger_does_not_break_earlier_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            ledger = root / "usage.jsonl"
            run_classifier(
                candidate,
                corpus,
                root / "chunk-1",
                ledger,
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: FakeResponse(self.response()),
            )
            run_classifier(
                candidate,
                corpus,
                root / "chunk-2",
                ledger,
                model="z-ai/glm-5.2-alt",
                api_key="secret",
                opener=lambda *_args, **_kwargs: FakeResponse(self.response()),
            )
            replay = run_classifier(
                candidate,
                corpus,
                root / "chunk-1",
                ledger,
                model="z-ai/glm-5.2",
                api_key="secret",
                opener=lambda *_args, **_kwargs: self.fail("earlier replay called provider"),
            )
            self.assertEqual(len(ledger.read_text().splitlines()), 2)
            self.assertEqual(replay["chunk_index"], 1)

    def test_non_verbatim_evidence_fails_without_a_second_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            response = self.response()
            payload = self.payload()
            payload["classifications"][0]["events"][0]["evidence_quote"] = "invented"
            response["choices"][0]["message"]["content"] = json.dumps(payload)
            calls = 0

            def opener(_request, timeout):
                nonlocal calls
                calls += 1
                return FakeResponse(response)

            with self.assertRaisesRegex(ClassificationError, "not verbatim"):
                run_classifier(
                    candidate,
                    corpus,
                    root / "output",
                    root / "usage.jsonl",
                    model="z-ai/glm-5.2",
                    api_key="secret",
                    opener=opener,
                )
            self.assertEqual(calls, 1)
            with self.assertRaisesRegex(ClassificationError, "not verbatim"):
                run_classifier(
                    candidate,
                    corpus,
                    root / "output",
                    root / "usage.jsonl",
                    model="z-ai/glm-5.2",
                    api_key="secret",
                    opener=lambda *_args, **_kwargs: self.fail("invalid replay called provider"),
                )
            self.assertEqual(calls, 1)

    def test_non_retryable_http_error_body_is_persisted_and_secret_redacted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            output = root / "output"
            calls = 0

            def opener(_request, timeout):
                nonlocal calls
                calls += 1
                raise HTTPError(
                    "https://openrouter.ai/api/v1/chat/completions",
                    404,
                    "Not Found",
                    {},
                    BytesIO(b'{"error":"no endpoint for secret-value"}'),
                )

            with self.assertRaisesRegex(ClassificationError, "HTTPError"):
                run_classifier(
                    candidate,
                    corpus,
                    output,
                    root / "usage.jsonl",
                    model="z-ai/glm-5.2",
                    api_key="secret-value",
                    opener=opener,
                )
            error = json.loads(
                (output / "http-error-attempt-1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(calls, 1)
            self.assertEqual(error["http_status"], 404)
            self.assertIn("[REDACTED]", error["body"])
            self.assertNotIn("secret-value", error["body"])
            with self.assertRaisesRegex(ClassificationError, "persisted non-retryable"):
                run_classifier(
                    candidate,
                    corpus,
                    root / "output",
                    root / "usage.jsonl",
                    model="z-ai/glm-5.2",
                    api_key="secret-value",
                    opener=lambda *_args, **_kwargs: self.fail("HTTP 404 replay called provider"),
                )
            self.assertEqual(calls, 1)

    def test_prompt_contains_full_bound_article(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, candidate = self.fixture(root)
            candidates = load_candidates(candidate, corpus)
            prompt = build_prompt(candidates)
            self.assertIn(candidates[0]["raw_article"], prompt)
            self.assertIn("ARTICLE article-1", prompt)
            self.assertNotIn("MATCHED_BASES", prompt)
            self.assertNotIn("real_identity_event", prompt)
            self.assertEqual(prompt.count("ACCEPTANCE_CONTROLS_BEGIN"), 1)


if __name__ == "__main__":
    unittest.main()
