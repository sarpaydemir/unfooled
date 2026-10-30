from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

EXPERIMENT_ID = "EXP-101-DEATH-CENSUS"
TASK_SHA256 = "929ddd71eeebcb9c52ad0de52a0f36235f4397bc322bb6e991b7226850e0810d"
MODEL_ENV = "RESEARCH_REVIEW_MODEL"
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS = 16_000
PROVIDER_CONTEXT_LIMIT_TOKENS = 1_024_000
USABLE_INPUT_LIMIT_TOKENS = PROVIDER_CONTEXT_LIMIT_TOKENS - MAX_COMPLETION_TOKENS
MAX_ATTEMPTS = 2
API_URL = "https://openrouter.ai/api/v1/chat/completions"
IDENTITY_CHANGE_TYPES = (
    "rename",
    "token_swap",
    "redenomination",
    "relisting_different_asset",
)
EVENT_TYPES = (
    *IDENTITY_CHANGE_TYPES,
    "delisting_only",
)
IDENTITY_EVENT_FIXTURES: tuple[Mapping[str, str], ...] = (
    {"event": "LUNA_TO_LUNC", "text": "LUNA (old) Renamed as LUNC"},
    {
        "event": "MATIC_TO_POL",
        "text": "Binance Will Support the Polygon (MATIC) Token Swap to Polygon (POL)",
    },
    {
        "event": "FTM_TO_S",
        "text": "Binance Will Support the Fantom (FTM) Token Swap and Rebranding to Sonic (S)",
    },
    {
        "event": "BTT_REDENOMINATION",
        "text": "Binance Will Support the BitTorrent (BTT) Token Migration, Redenomination and Renaming Plan",
    },
    {
        "event": "PAX_TO_USDP",
        "text": "Paxos Standard (PAX) Rebrand and Rename to Pax Dollar (USDP)",
    },
    {
        "event": "BCHABC_TO_BCH",
        "text": "Binance Will Rename BCHABC to BCH and Apply the Ticker Change",
    },
)
SYSTEM_PROMPT = (
    "Classify official exchange announcements for asset identity changes. "
    "Use only the supplied article, copy one exact evidence sentence, and "
    "return only the required JSON-schema object."
)


class ClassificationError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _plain_announcement_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def classification_schema(candidate_count: int) -> dict[str, Any]:
    event = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event_type": {"type": "string", "enum": list(EVENT_TYPES)},
            "subject_symbol": {
                "type": ["string", "null"],
                "description": "The old ticker that is the subject of the article event, or null.",
            },
            "successor_ticker": {"type": ["string", "null"]},
            "evidence_quote": {"type": "string", "minLength": 1},
        },
        "required": [
            "event_type",
            "subject_symbol",
            "successor_ticker",
            "evidence_quote",
        ],
    }
    classification = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "article_id": {"type": "string"},
            "events": {"type": "array", "items": event},
            "no_event_reason": {"type": ["string", "null"]},
        },
        "required": ["article_id", "events", "no_event_reason"],
    }
    control = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "event": {
                "type": "string",
                "enum": [fixture["event"] for fixture in IDENTITY_EVENT_FIXTURES],
            },
            "event_type": {
                "type": "string",
                "enum": list(IDENTITY_CHANGE_TYPES),
            },
            "evidence_quote": {"type": "string", "minLength": 1},
        },
        "required": ["event", "event_type", "evidence_quote"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "classifications": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": classification,
            },
            "acceptance_controls": {
                "type": "array",
                "minItems": len(IDENTITY_EVENT_FIXTURES),
                "maxItems": len(IDENTITY_EVENT_FIXTURES),
                "items": control,
            },
        },
        "required": ["classifications", "acceptance_controls"],
    }


def load_candidates(candidate_csv: Path, corpus_dir: Path) -> list[dict[str, Any]]:
    with candidate_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["article_code"] for row in rows}) != len(rows):
        raise ClassificationError("candidate CSV is empty or contains duplicate article ids")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        article_id = str(row["article_code"])
        path = corpus_dir / "articles" / f"{article_id}.json"
        if path.is_symlink() or not path.is_file():
            raise ClassificationError(f"candidate article missing or symlinked: {article_id}")
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if str(payload.get("code")) != article_id or str(payload.get("title")) != row["title"]:
            raise ClassificationError(f"candidate article identity mismatch: {article_id}")
        candidates.append(
            {
                "article_id": article_id,
                "matched_bases": tuple(filter(None, str(row["matched_bases"]).split(","))),
                "raw_article": raw,
                "title": str(payload["title"]),
                "body": str(payload["body"]),
                "source_sha256": file_sha256(path),
            }
        )
    return candidates


def partition_candidates(
    candidates: Sequence[Mapping[str, Any]], chunk_index: int, chunk_count: int
) -> list[Mapping[str, Any]]:
    if chunk_count < 1 or not 1 <= chunk_index <= chunk_count:
        raise ClassificationError("invalid chunk index/count")
    ordered = sorted(candidates, key=lambda row: str(row["article_id"]))
    quotient, remainder = divmod(len(ordered), chunk_count)
    start = (chunk_index - 1) * quotient + min(chunk_index - 1, remainder)
    size = quotient + (1 if chunk_index <= remainder else 0)
    chunk = ordered[start : start + size]
    if not chunk:
        raise ClassificationError("candidate chunk is empty")
    return chunk


def slice_candidates(
    candidates: Sequence[Mapping[str, Any]], offset: int, limit: int
) -> list[Mapping[str, Any]]:
    if offset < 0 or limit < 1:
        raise ClassificationError("invalid candidate slice")
    ordered = sorted(candidates, key=lambda row: str(row["article_id"]))
    selected = ordered[offset : offset + limit]
    if not selected:
        raise ClassificationError("candidate slice is empty")
    return selected


def build_prompt(candidates: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "Return one article record for every ARTICLE and six acceptance controls.",
        "events must contain every distinct rename, token swap, redenomination, relisting as a different asset, or plain delisting described by that article.",
        "event_type is the single canonical decision. An ordinary first/additional listing, Binance Convert addition, or network integration is not relisting_different_asset and produces no event unless another qualifying event is explicitly described.",
        "Use delisting_only only for a plain asset delisting without an identity change. subject_symbol is the old ticker and successor_ticker is the new ticker when stated.",
        "Every event needs one verbatim evidence sentence. If there are no qualifying events, return an empty events array and a non-empty no_event_reason; otherwise no_event_reason must be null.",
        "ARTICLES_BEGIN",
    ]
    for candidate in candidates:
        lines.append(f"ARTICLE {candidate['article_id']}")
        lines.append(str(candidate["raw_article"]))
        lines.append("ARTICLE_END")
    lines.append("ACCEPTANCE_CONTROLS_BEGIN")
    for fixture in IDENTITY_EVENT_FIXTURES:
        lines.append(
            json.dumps(
                {"event": fixture["event"], "text": fixture["text"]},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    lines.append("ACCEPTANCE_CONTROLS_END")
    return "\n".join(lines)


def request_call_hash(model: str, prompt: str, schema: Mapping[str, Any]) -> str:
    schema_bytes = json.dumps(
        schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(model.encode("utf-8") + prompt.encode("utf-8") + schema_bytes).hexdigest()


def retry_call_hash(
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    *,
    max_tokens: int,
    retry_generation: int,
) -> str:
    base = request_call_hash(model, prompt, schema)
    if retry_generation == 0 and max_tokens == MAX_COMPLETION_TOKENS:
        return base
    suffix = f"\0max_tokens={max_tokens}\0retry_generation={retry_generation}".encode()
    return hashlib.sha256(bytes.fromhex(base) + suffix).hexdigest()


def build_request_envelope(
    model: str,
    prompt: str,
    schema: Mapping[str, Any],
    *,
    max_tokens: int = MAX_COMPLETION_TOKENS,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "exp_101_identity_classification",
                "strict": True,
                "schema": schema,
            },
        },
        "provider": {"require_parameters": True},
        "usage": {"include": True},
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "stream": False,
    }


_EVIDENCE_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
    }
)


def normalize_evidence_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.translate(_EVIDENCE_TRANSLATION)).strip()


def _evidence_presence(quote: str, candidate: Mapping[str, Any]) -> tuple[bool, bool]:
    readable = _plain_announcement_text(f"{candidate['title']}\n{candidate['body']}")
    raw_present = quote.strip() in readable
    normalized_present = normalize_evidence_text(quote) in normalize_evidence_text(readable)
    return raw_present, normalized_present


def _exact_evidence_present(quote: str, candidate: Mapping[str, Any]) -> bool:
    return _evidence_presence(quote, candidate)[1]


def _ticker_present(symbol: str, candidate: Mapping[str, Any]) -> bool:
    if not symbol:
        return False
    readable = _plain_announcement_text(f"{candidate['title']}\n{candidate['body']}")
    pattern = rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])"
    return re.search(pattern, readable, flags=re.IGNORECASE) is not None


def _control_evidence_present(
    quote: str, candidates: Sequence[Mapping[str, Any]]
) -> bool:
    evidence = normalize_evidence_text(quote)
    if not evidence:
        return False
    sources = [str(fixture["text"]) for fixture in IDENTITY_EVENT_FIXTURES]
    sources.extend(
        _plain_announcement_text(f"{candidate['title']}\n{candidate['body']}")
        for candidate in candidates
    )
    return any(evidence in normalize_evidence_text(source) for source in sources)


def calibration_audit(
    payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if set(payload) != {"classifications", "acceptance_controls"}:
        raise ClassificationError("classification payload top-level schema mismatch")
    rows = payload.get("classifications")
    controls = payload.get("acceptance_controls")
    if not isinstance(rows, list) or not isinstance(controls, list):
        raise ClassificationError("classification arrays missing")
    by_id = {str(candidate["article_id"]): candidate for candidate in candidates}
    if len(rows) != len(by_id) or {str(row.get("article_id")) for row in rows} != set(by_id):
        raise ClassificationError("classification article coverage mismatch")

    required = {"article_id", "events", "no_event_reason"}
    event_required = {
        "event_type",
        "subject_symbol",
        "successor_ticker",
        "evidence_quote",
    }
    raw_pass = 0
    normalized_pass = 0
    event_count = 0
    semantic_errors = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            semantic_errors += 1
            continue
        article_id = str(row["article_id"])
        candidate = by_id[article_id]
        events = row["events"]
        no_event_reason = row["no_event_reason"]
        valid = isinstance(events, list)
        if valid:
            valid = (bool(events) and no_event_reason is None) or (
                not events
                and isinstance(no_event_reason, str)
                and bool(no_event_reason.strip())
            )
        seen_events: set[tuple[Any, Any, Any]] = set()
        for event in events if isinstance(events, list) else ():
            event_count += 1
            if not isinstance(event, dict) or set(event) != event_required:
                valid = False
                continue
            event_type = event["event_type"]
            subject_symbol = event["subject_symbol"]
            successor_ticker = event["successor_ticker"]
            valid = valid and event_type in EVENT_TYPES
            valid = valid and (subject_symbol is None or isinstance(subject_symbol, str))
            valid = valid and (
                successor_ticker is None or isinstance(successor_ticker, str)
            )
            if subject_symbol is not None:
                valid = valid and _ticker_present(subject_symbol, candidate)
            if successor_ticker is not None:
                valid = valid and _ticker_present(successor_ticker, candidate)
            identity = (event_type, subject_symbol, successor_ticker)
            if identity in seen_events:
                valid = False
            seen_events.add(identity)
            quote = event["evidence_quote"]
            if isinstance(quote, str) and quote.strip():
                raw_present, normalized_present = _evidence_presence(quote, candidate)
                raw_pass += int(raw_present)
                normalized_pass += int(normalized_present)
        if not valid:
            semantic_errors += 1

    expected_controls = {str(fixture["event"]): fixture for fixture in IDENTITY_EVENT_FIXTURES}
    control_required = {"event", "event_type", "evidence_quote"}
    valid_controls = 0
    seen_controls: set[str] = set()
    for row in controls:
        if not isinstance(row, dict) or set(row) != control_required:
            continue
        event = str(row["event"])
        fixture = expected_controls.get(event)
        if fixture is None or event in seen_controls:
            continue
        seen_controls.add(event)
        if (
            row["event_type"] in IDENTITY_CHANGE_TYPES
            and _control_evidence_present(str(row["evidence_quote"]), candidates)
        ):
            valid_controls += 1

    total = len(rows)
    raw_failures = event_count - raw_pass
    normalized_failures = event_count - normalized_pass
    rescued = normalized_pass - raw_pass
    rescue_fraction = rescued / raw_failures if raw_failures else 0.0
    result = {
        "classification_count": total,
        "event_count": event_count,
        "raw_evidence_pass_count": raw_pass,
        "raw_evidence_failure_count": raw_failures,
        "normalized_evidence_pass_count": normalized_pass,
        "normalized_evidence_failure_count": normalized_failures,
        "normalized_evidence_rate": normalized_pass / event_count if event_count else 1.0,
        "normalization_rescued_count": rescued,
        "normalization_rescue_fraction_of_raw_failures": rescue_fraction,
        "normalization_rescue_headline": rescue_fraction > 0.20,
        "semantic_error_count": semantic_errors,
        "acceptance_control_count": len(controls),
        "acceptance_valid_count": valid_controls,
    }
    result["calibration_pass"] = (
        result["normalized_evidence_rate"] >= 0.95
        and valid_controls == len(IDENTITY_EVENT_FIXTURES)
        and len(controls) == len(IDENTITY_EVENT_FIXTURES)
    )
    return result


def validate_payload(
    payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> None:
    if set(payload) != {"classifications", "acceptance_controls"}:
        raise ClassificationError("classification payload top-level schema mismatch")
    rows = payload.get("classifications")
    controls = payload.get("acceptance_controls")
    if not isinstance(rows, list) or not isinstance(controls, list):
        raise ClassificationError("classification arrays missing")
    by_id = {str(candidate["article_id"]): candidate for candidate in candidates}
    if len(rows) != len(by_id) or {str(row.get("article_id")) for row in rows} != set(by_id):
        raise ClassificationError("classification article coverage mismatch")
    required = {"article_id", "events", "no_event_reason"}
    event_required = {
        "event_type",
        "subject_symbol",
        "successor_ticker",
        "evidence_quote",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required:
            raise ClassificationError("classification row schema mismatch")
        article_id = str(row["article_id"])
        candidate = by_id[article_id]
        events = row["events"]
        reason = row["no_event_reason"]
        if not isinstance(events, list):
            raise ClassificationError(f"events is not an array: {article_id}")
        if events and reason is not None:
            raise ClassificationError(f"eventful article has no-event reason: {article_id}")
        if not events and (not isinstance(reason, str) or not reason.strip()):
            raise ClassificationError(f"empty article lacks no-event reason: {article_id}")
        seen_events: set[tuple[Any, Any, Any]] = set()
        for event in events:
            if not isinstance(event, dict) or set(event) != event_required:
                raise ClassificationError(f"event schema mismatch: {article_id}")
            event_type = event["event_type"]
            if event_type not in EVENT_TYPES:
                raise ClassificationError(f"event type mismatch: {article_id}")
            subject_symbol = event["subject_symbol"]
            successor_ticker = event["successor_ticker"]
            if subject_symbol is not None and not isinstance(subject_symbol, str):
                raise ClassificationError(f"subject symbol type mismatch: {article_id}")
            if successor_ticker is not None and not isinstance(successor_ticker, str):
                raise ClassificationError(f"successor ticker type mismatch: {article_id}")
            if subject_symbol is not None and not _ticker_present(subject_symbol, candidate):
                raise ClassificationError(f"subject symbol absent from article: {article_id}")
            if successor_ticker is not None and not _ticker_present(
                successor_ticker, candidate
            ):
                raise ClassificationError(f"successor ticker absent from article: {article_id}")
            identity = (event_type, subject_symbol, successor_ticker)
            if identity in seen_events:
                raise ClassificationError(f"duplicate event: {article_id}")
            seen_events.add(identity)
            quote = event["evidence_quote"]
            if not isinstance(quote, str) or not _exact_evidence_present(quote, candidate):
                raise ClassificationError(f"evidence quote is not verbatim: {article_id}")

    expected_controls = {str(fixture["event"]): fixture for fixture in IDENTITY_EVENT_FIXTURES}
    if len(controls) != len(expected_controls) or {
        str(row.get("event")) for row in controls
    } != set(expected_controls):
        raise ClassificationError("acceptance control coverage mismatch")
    control_required = {"event", "event_type", "evidence_quote"}
    for row in controls:
        if not isinstance(row, dict) or set(row) != control_required:
            raise ClassificationError("acceptance control schema mismatch")
        event = str(row["event"])
        if row["event_type"] not in IDENTITY_CHANGE_TYPES:
            raise ClassificationError(f"acceptance control rejected: {event}")
        if not _control_evidence_present(str(row["evidence_quote"]), candidates):
            raise ClassificationError(f"acceptance evidence is not verbatim: {event}")


def materialize_payload(
    payload: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    validate_payload(payload, candidates)
    by_id = {str(candidate["article_id"]): candidate for candidate in candidates}
    rows: list[dict[str, Any]] = []
    for source_row in payload["classifications"]:
        row = dict(source_row)
        candidate = by_id[str(row["article_id"])]
        events: list[dict[str, Any]] = []
        for source_event in row["events"]:
            event = dict(source_event)
            event_type = event["event_type"]
            subject_symbol = event["subject_symbol"]
            is_identity_change = event_type in IDENTITY_CHANGE_TYPES
            in_scan_set = (
                subject_symbol is not None
                and subject_symbol.upper() in candidate["matched_bases"]
            )
            event["is_identity_change"] = is_identity_change
            event["is_asset_exit_signal"] = event_type == "delisting_only"
            event["in_scan_set"] = in_scan_set
            event["is_identity_event"] = bool(is_identity_change and in_scan_set)
            event["needs_manual_review"] = bool(
                is_identity_change and subject_symbol is None
            )
            events.append(event)
        row["events"] = events
        rows.append(row)
    return {
        "classifications": rows,
        "acceptance_controls": [dict(row) for row in payload["acceptance_controls"]],
    }


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ClassificationError(f"append-only ledger may not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _transport_retryable(error: Exception) -> bool:
    if isinstance(error, HTTPError):
        return error.code in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(error, (URLError, TimeoutError))


def _persist_http_error(
    path: Path, error: HTTPError, *, api_key: str, call_hash: str, attempt: int
) -> None:
    try:
        raw_body = error.read(65_536)
    finally:
        error.close()
    body = raw_body.decode("utf-8", errors="replace")
    if api_key:
        body = body.replace(api_key, "[REDACTED]")
    _atomic_write_bytes(
        path,
        _canonical_bytes(
            {
                "call_hash": call_hash,
                "attempt": attempt,
                "http_status": error.code,
                "reason": str(error.reason),
                "body": body,
                "body_truncated": len(raw_body) == 65_536,
            }
        ),
    )


def _decode_provider_usage(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        usage = raw["usage"]
        return {
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "usd": float(usage["cost"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ClassificationError("provider usage is incomplete") from error


def _decode_provider_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    try:
        content = raw["choices"][0]["message"]["content"]
        payload = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ClassificationError("provider response content is incomplete") from error
    if not isinstance(payload, dict):
        raise ClassificationError("provider structured output is not an object")
    return payload


def _decode_provider_response(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return _decode_provider_payload(raw), _decode_provider_usage(raw)


def failed_calibration_audit(expected_count: int, error: str) -> dict[str, Any]:
    return {
        "classification_count": 0,
        "event_count": 0,
        "expected_classification_count": expected_count,
        "raw_evidence_pass_count": 0,
        "raw_evidence_failure_count": expected_count,
        "normalized_evidence_pass_count": 0,
        "normalized_evidence_failure_count": expected_count,
        "normalized_evidence_rate": 0.0,
        "normalization_rescued_count": 0,
        "normalization_rescue_fraction_of_raw_failures": 0.0,
        "normalization_rescue_headline": False,
        "semantic_error_count": expected_count,
        "acceptance_control_count": 0,
        "acceptance_valid_count": 0,
        "provider_output_error": error,
        "calibration_pass": False,
    }


def run_classifier(
    candidate_csv: Path,
    corpus_dir: Path,
    output_dir: Path,
    usage_ledger: Path,
    *,
    model: str,
    api_key: str,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
    chunk_index: int = 1,
    chunk_count: int = 1,
    article_offset: int | None = None,
    article_limit: int | None = None,
    calibration: bool = False,
    batch: bool = False,
    max_tokens: int = MAX_COMPLETION_TOKENS,
    retry_generation: int = 0,
) -> dict[str, Any]:
    if calibration and batch:
        raise ClassificationError("calibration and batch modes are mutually exclusive")
    if max_tokens < 1 or retry_generation < 0:
        raise ClassificationError("invalid max tokens or retry generation")
    if not model:
        raise ClassificationError(f"{MODEL_ENV} must be explicitly configured")
    if not api_key:
        raise ClassificationError("OPENROUTER_API_KEY is not configured")
    all_candidates = load_candidates(candidate_csv, corpus_dir)
    if article_offset is None and article_limit is None:
        candidates = partition_candidates(all_candidates, chunk_index, chunk_count)
    elif article_offset is not None and article_limit is not None:
        candidates = slice_candidates(all_candidates, article_offset, article_limit)
    else:
        raise ClassificationError("article offset and limit must be provided together")
    schema = classification_schema(len(candidates))
    prompt = build_prompt(candidates)
    prompt_estimate = math.ceil(len(prompt.encode("utf-8")) / 4)
    if prompt_estimate > USABLE_INPUT_LIMIT_TOKENS:
        raise ClassificationError(
            f"paid-call input limit exceeded: {prompt_estimate}>{USABLE_INPUT_LIMIT_TOKENS}"
        )
    call_hash = retry_call_hash(
        model,
        prompt,
        schema,
        max_tokens=max_tokens,
        retry_generation=retry_generation,
    )
    envelope = build_request_envelope(model, prompt, schema, max_tokens=max_tokens)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request.json"
    response_path = output_dir / "response.json"
    result_path = output_dir / "classification.json"
    summary_name = (
        "calibration-summary.json"
        if calibration
        else ("batch-summary.json" if batch else "classification-summary.json")
    )
    summary_path = output_dir / summary_name
    phase = (
        "paid_identity_calibration"
        if calibration
        else ("paid_identity_batch" if batch else "paid_identity_classification")
    )
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "task_sha256": TASK_SHA256,
        "call_hash": call_hash,
        "model": model,
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "candidate_count": len(candidates),
        "candidate_total_count": len(all_candidates),
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "article_offset": article_offset,
        "article_limit": article_limit,
        "calibration": calibration,
        "batch": batch,
        "retry_generation": retry_generation,
        "candidate_csv_sha256": file_sha256(candidate_csv),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": canonical_json_sha256(schema),
        "prompt_input_estimate_bytes_div_4": prompt_estimate,
        "request": envelope,
    }
    if request_path.exists():
        existing = json.loads(request_path.read_text(encoding="utf-8"))
        if existing.get("call_hash") != call_hash or existing != meta:
            raise ClassificationError("existing request claim does not match current call hash")
    else:
        _atomic_write_bytes(request_path, _canonical_bytes(meta))

    raw_response: dict[str, Any]
    if response_path.exists():
        raw_response = json.loads(response_path.read_text(encoding="utf-8"))
    else:
        existing_attempts = sorted(output_dir.glob("attempt-*.json"))
        existing_http_errors = sorted(output_dir.glob("http-error-attempt-*.json"))
        for error_path in existing_http_errors:
            persisted_error = json.loads(error_path.read_text(encoding="utf-8"))
            if int(persisted_error.get("http_status", 0)) not in {
                408,
                425,
                429,
                500,
                502,
                503,
                504,
            }:
                raise ClassificationError(
                    "persisted non-retryable HTTP error forbids another provider call"
                )
        if len(existing_attempts) >= MAX_ATTEMPTS:
            raise ClassificationError("paid-call attempt cap exhausted without a persisted response")
        raw_response = {}
        for attempt in range(len(existing_attempts) + 1, MAX_ATTEMPTS + 1):
            attempt_path = output_dir / f"attempt-{attempt}.json"
            _atomic_write_bytes(
                attempt_path,
                _canonical_bytes(
                    {
                        "call_hash": call_hash,
                        "attempt": attempt,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                ),
            )
            request = Request(
                API_URL,
                data=json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Sarp Local Research Factory",
                },
            )
            try:
                with opener(request, timeout=600) as response:
                    raw_response = json.loads(response.read())
                _atomic_write_bytes(response_path, _canonical_bytes(raw_response))
                break
            except Exception as error:
                if isinstance(error, HTTPError):
                    _persist_http_error(
                        output_dir / f"http-error-attempt-{attempt}.json",
                        error,
                        api_key=api_key,
                        call_hash=call_hash,
                        attempt=attempt,
                    )
                if not _transport_retryable(error) or attempt >= MAX_ATTEMPTS:
                    raise ClassificationError(
                        f"OpenRouter request failed without an allowed retry: {type(error).__name__}"
                    ) from error
                sleep(2 ** (attempt - 1))
        if not response_path.exists():
            raise ClassificationError("provider returned no persistable response")

    usage = _decode_provider_usage(raw_response)
    payload: dict[str, Any] | None = None
    payload_error: ClassificationError | None = None
    try:
        payload = _decode_provider_payload(raw_response)
    except ClassificationError as error:
        payload_error = error
    audit = (
        calibration_audit(payload, candidates)
        if payload is not None
        else failed_calibration_audit(len(candidates), str(payload_error))
    )
    validation_error: ClassificationError | None = None
    quality_pass = bool(audit["calibration_pass"])
    semantic_pass = payload is not None and audit["semantic_error_count"] == 0
    if calibration:
        outcome = "calibration_passed" if quality_pass else "calibration_failed"
    elif batch:
        if not quality_pass:
            outcome = "batch_quality_failed"
        elif not semantic_pass:
            outcome = "batch_semantic_conflict"
        else:
            outcome = "batch_passed"
    elif payload_error is not None:
        outcome = "response_invalid"
        validation_error = payload_error
    else:
        outcome = "identity_classification_complete"
        try:
            assert payload is not None
            validate_payload(payload, candidates)
        except ClassificationError as error:
            outcome = "response_invalid"
            validation_error = error

    ledger_rows = []
    if usage_ledger.exists():
        for line in usage_ledger.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ledger_rows.append(json.loads(line))
    matching = [row for row in ledger_rows if row.get("call_hash") == call_hash]
    if not matching:
        ledger_row = {
            "call_hash": call_hash,
            "model": str(raw_response.get("model") or model),
            **usage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "idempotency_status": (
                outcome
                if calibration or batch
                else ("response_validated" if validation_error is None else "response_invalid")
            ),
            "request_id": raw_response.get("id"),
        }
        _append_jsonl(
            usage_ledger,
            ledger_row,
        )
    elif len(matching) != 1:
        raise ClassificationError("usage ledger contains duplicate call hash rows")
    else:
        ledger_row = matching[0]
    if validation_error is not None:
        raise validation_error

    accepted_payload = payload is not None and (
        (calibration and quality_pass and semantic_pass)
        or (batch and outcome == "batch_passed")
        or (not calibration and not batch and validation_error is None)
    )
    if accepted_payload:
        if result_path.exists():
            existing_payload = json.loads(result_path.read_text(encoding="utf-8"))
            if existing_payload != payload:
                raise ClassificationError("persisted classification payload mismatch")
        else:
            _atomic_write_bytes(result_path, _canonical_bytes(payload))
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "task_sha256": TASK_SHA256,
        "outcome": outcome,
        "call_hash": call_hash,
        "model": str(raw_response.get("model") or model),
        "provider_request_id": raw_response.get("id"),
        "candidate_count": len(candidates),
        "candidate_total_count": len(all_candidates),
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "article_offset": article_offset,
        "article_limit": article_limit,
        "calibration": calibration,
        "batch": batch,
        "max_tokens": max_tokens,
        "retry_generation": retry_generation,
        "provider_finish_reason": (
            raw_response.get("choices", [{}])[0].get("finish_reason")
            if raw_response.get("choices")
            else None
        ),
        "classified_event_count": sum(
            len(row.get("events", []))
            for row in (payload or {}).get("classifications", [])
            if isinstance(row, dict)
        ),
        "acceptance_control_count": len(
            (payload or {}).get("acceptance_controls", [])
        ),
        "acceptance_pass_count": sum(
            row.get("event_type") in IDENTITY_CHANGE_TYPES
            for row in (payload or {}).get("acceptance_controls", [])
            if isinstance(row, dict)
        ),
        "calibration_audit": audit,
        "prompt_input_estimate_bytes_div_4": prompt_estimate,
        "model_called": True,
        "provider_response_persisted": True,
        "usage": usage,
        "usage_ledger_row_sha256": canonical_json_sha256(ledger_row),
        "protected_rows_read": False,
        "performance_or_death_metrics_computed": False,
        "main_research_db_written": False,
        "artifacts": {
            request_path.name: file_sha256(request_path),
            response_path.name: file_sha256(response_path),
        },
    }
    if result_path.exists():
        summary["artifacts"][result_path.name] = file_sha256(result_path)
    summary["summary_sha256"] = canonical_json_sha256(summary)
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing_summary != summary:
            raise ClassificationError("persisted classification summary mismatch")
    else:
        _atomic_write_bytes(summary_path, _canonical_bytes(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-101 paid identity classifier")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--usage-ledger", type=Path, required=True)
    parser.add_argument("--chunk-index", type=int, default=1)
    parser.add_argument("--chunk-count", type=int, default=1)
    parser.add_argument("--article-offset", type=int)
    parser.add_argument("--article-limit", type=int)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument("--retry-generation", type=int, default=0)
    args = parser.parse_args()
    result = run_classifier(
        args.candidate_csv,
        args.corpus_dir,
        args.output_dir,
        args.usage_ledger,
        model=os.getenv(MODEL_ENV, "").strip(),
        api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        chunk_index=args.chunk_index,
        chunk_count=args.chunk_count,
        article_offset=args.article_offset,
        article_limit=args.article_limit,
        calibration=args.calibration,
        batch=args.batch,
        max_tokens=args.max_tokens,
        retry_generation=args.retry_generation,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
