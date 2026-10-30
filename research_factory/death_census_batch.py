from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

from research_factory.death_census_classifier import (
    MAX_COMPLETION_TOKENS,
    _atomic_write_bytes,
    _canonical_bytes,
    canonical_json_sha256,
    file_sha256,
    load_candidates,
    materialize_payload,
    run_classifier,
    slice_candidates,
)


EXPERIMENT_ID = "EXP-101-DEATH-CENSUS"
TASK_SHA256 = "929ddd71eeebcb9c52ad0de52a0f36235f4397bc322bb6e991b7226850e0810d"
REUSED_V4_TASK_SHA256 = "0c402b737c2aaf63ab6d195f425fca7f25e16c96a523b438583ee3536b66d643"
REUSED_V4_LAST_OFFSET = 85
FROZEN_CHUNK_SIZE = 5
TOTAL_CANDIDATES = 159


class BatchError(RuntimeError):
    pass


def _candidate_count(candidate_csv: Path) -> int:
    with candidate_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != TOTAL_CANDIDATES or len({row["article_code"] for row in rows}) != len(rows):
        raise BatchError("batch requires the frozen 159 unique candidates")
    return len(rows)


def run_batches(
    candidate_csv: Path,
    corpus_dir: Path,
    output_root: Path,
    usage_ledger: Path,
    *,
    model: str,
    api_key: str,
    classifier: Callable[..., dict[str, Any]] = run_classifier,
    emit: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    total = _candidate_count(candidate_csv)
    accepted: list[dict[str, Any]] = []
    for offset in range(0, total, FROZEN_CHUNK_SIZE):
        limit = min(FROZEN_CHUNK_SIZE, total - offset)
        chunk_dir = output_root / f"chunk-{offset:03d}"
        passed_paths = sorted(chunk_dir.glob("attempt-*/batch-summary.json"))
        passed_summaries = []
        for path in passed_paths:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("outcome") == "batch_passed":
                passed_summaries.append(summary)
        if passed_summaries:
            if len(passed_summaries) != 1:
                raise BatchError(f"multiple accepted attempts at offset {offset}")
            summary = passed_summaries[0]
            allowed_reuse = (
                summary.get("task_sha256") == TASK_SHA256
                or (
                    offset <= REUSED_V4_LAST_OFFSET
                    and summary.get("task_sha256") == REUSED_V4_TASK_SHA256
                )
            )
            if not allowed_reuse:
                raise BatchError(f"accepted attempt has unauthorized task at offset {offset}")
            accepted.append(summary)
            emit(f"offset={offset} count={limit} outcome=reused_batch_passed", flush=True)
            continue

        existing_generations = []
        for path in chunk_dir.glob("attempt-*-max-*"):
            match = re.fullmatch(r"attempt-(\d+)-max-(\d+)", path.name)
            if match:
                existing_generations.append((int(match.group(1)), int(match.group(2))))
        retry_generation = (
            max(generation for generation, _tokens in existing_generations) + 1
            if existing_generations
            else 0
        )
        max_tokens = MAX_COMPLETION_TOKENS
        quality_retries = 0
        while True:
            attempt_dir = chunk_dir / f"attempt-{retry_generation}-max-{max_tokens}"
            summary = classifier(
                candidate_csv,
                corpus_dir,
                attempt_dir,
                usage_ledger,
                model=model,
                api_key=api_key,
                article_offset=offset,
                article_limit=limit,
                batch=True,
                max_tokens=max_tokens,
                retry_generation=retry_generation,
            )
            audit = summary["calibration_audit"]
            emit(
                " ".join(
                    (
                        f"offset={offset}",
                        f"count={limit}",
                        f"outcome={summary['outcome']}",
                        f"evidence={audit['normalized_evidence_pass_count']}/{audit['event_count']}",
                        f"controls={audit['acceptance_valid_count']}/6",
                        f"semantic_errors={audit['semantic_error_count']}",
                        f"max_tokens={max_tokens}",
                        f"usd={summary['usage']['usd']:.8f}",
                    )
                ),
                flush=True,
            )
            if summary["outcome"] == "batch_passed":
                accepted.append(summary)
                break
            if summary["outcome"] == "batch_semantic_conflict":
                raise BatchError(f"research-content semantic conflict at offset {offset}")
            if summary["outcome"] != "batch_quality_failed":
                raise BatchError(f"unexpected batch outcome at offset {offset}")

            provider_output_error = audit.get("provider_output_error")
            if provider_output_error and summary.get("provider_finish_reason") == "length":
                retry_generation += 1
                max_tokens *= 2
                continue
            if quality_retries >= 1:
                raise BatchError(f"quality gate failed twice at offset {offset}")
            quality_retries += 1
            retry_generation += 1
    return accepted


def aggregate_batches(
    candidate_csv: Path,
    corpus_dir: Path,
    batch_root: Path,
    usage_ledger: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        summary_path = output_dir / "combined-summary.json"
        combined_path = output_dir / "combined-classification.json"
        if not summary_path.is_file() or not combined_path.is_file():
            raise BatchError("partial combined output exists; refusing overwrite")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stored = summary.pop("summary_sha256", None)
        if stored != canonical_json_sha256(summary):
            raise BatchError("existing combined summary checksum mismatch")
        summary["summary_sha256"] = stored
        if (
            summary.get("task_sha256") != TASK_SHA256
            or summary.get("candidate_count") != TOTAL_CANDIDATES
            or summary.get("combined_classification_sha256") != file_sha256(combined_path)
        ):
            raise BatchError("existing combined output invariant mismatch")
        return summary
    all_candidates = load_candidates(candidate_csv, corpus_dir)
    if len(all_candidates) != TOTAL_CANDIDATES:
        raise BatchError("combined output requires 159 candidates")
    ledger_rows = [
        json.loads(line)
        for line in usage_ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    classifications: list[dict[str, Any]] = []
    accepted_chunks: list[dict[str, Any]] = []
    call_hashes: set[str] = set()
    run_usd = 0.0
    raw_evidence_pass_count = 0
    normalized_evidence_pass_count = 0
    evidence_event_count = 0
    for offset in range(0, TOTAL_CANDIDATES, FROZEN_CHUNK_SIZE):
        limit = min(FROZEN_CHUNK_SIZE, TOTAL_CANDIDATES - offset)
        summaries: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted((batch_root / f"chunk-{offset:03d}").glob("attempt-*/batch-summary.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            if summary.get("outcome") == "batch_passed":
                summaries.append((path, summary))
        if len(summaries) != 1:
            raise BatchError(f"offset {offset} has {len(summaries)} accepted attempts")
        summary_path, summary = summaries[0]
        stored = summary.pop("summary_sha256", None)
        if stored != canonical_json_sha256(summary):
            raise BatchError(f"batch summary checksum mismatch at offset {offset}")
        summary["summary_sha256"] = stored
        if (
            summary["task_sha256"]
            not in (
                {TASK_SHA256, REUSED_V4_TASK_SHA256}
                if offset <= REUSED_V4_LAST_OFFSET
                else {TASK_SHA256}
            )
            or summary["article_offset"] != offset
            or summary["candidate_count"] != limit
            or summary["calibration_audit"]["semantic_error_count"] != 0
            or summary["calibration_audit"]["normalized_evidence_rate"] < 0.95
            or summary["calibration_audit"]["acceptance_valid_count"] != 6
        ):
            raise BatchError(f"batch invariant mismatch at offset {offset}")
        payload_path = summary_path.parent / "classification.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        candidates = slice_candidates(all_candidates, offset, limit)
        materialized = materialize_payload(payload, candidates)
        expected_ids = sorted(str(row["article_id"]) for row in candidates)
        actual_ids = sorted(str(row["article_id"]) for row in payload["classifications"])
        if actual_ids != expected_ids:
            raise BatchError(f"classification order/coverage mismatch at offset {offset}")
        call_hash = str(summary["call_hash"])
        matching = [row for row in ledger_rows if row.get("call_hash") == call_hash]
        if len(matching) != 1 or matching[0].get("idempotency_status") != "batch_passed":
            raise BatchError(f"usage ledger mismatch at offset {offset}")
        if canonical_json_sha256(matching[0]) != summary["usage_ledger_row_sha256"]:
            raise BatchError(f"usage ledger checksum mismatch at offset {offset}")
        if call_hash in call_hashes:
            raise BatchError("accepted call hash reused across chunks")
        call_hashes.add(call_hash)
        classifications.extend(materialized["classifications"])
        run_usd += float(summary["usage"]["usd"])
        audit = summary["calibration_audit"]
        raw_evidence_pass_count += int(audit["raw_evidence_pass_count"])
        normalized_evidence_pass_count += int(audit["normalized_evidence_pass_count"])
        evidence_event_count += int(audit["event_count"])
        accepted_chunks.append(
            {
                "article_offset": offset,
                "candidate_count": limit,
                "call_hash": call_hash,
                "source_task_sha256": summary["task_sha256"],
                "summary_sha256": stored,
                "classification_sha256": file_sha256(payload_path),
            }
        )

    expected_all_ids = sorted(str(row["article_id"]) for row in all_candidates)
    actual_all_ids = sorted(str(row["article_id"]) for row in classifications)
    if len(classifications) != TOTAL_CANDIDATES or actual_all_ids != expected_all_ids:
        raise BatchError("combined classification coverage mismatch")

    combined = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "combined_paid_identity_classification",
        "task_sha256": TASK_SHA256,
        "classifications": classifications,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    combined_path = output_dir / "combined-classification.json"
    _atomic_write_bytes(combined_path, _canonical_bytes(combined))
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "combined_paid_identity_classification",
        "task_sha256": TASK_SHA256,
        "outcome": "identity_classification_complete",
        "candidate_count": len(classifications),
        "chunk_size": FROZEN_CHUNK_SIZE,
        "accepted_chunk_count": len(accepted_chunks),
        "classified_event_count": sum(
            len(row["events"]) for row in classifications
        ),
        "identity_event_count": sum(
            bool(event["is_identity_event"])
            for row in classifications
            for event in row["events"]
        ),
        "asset_exit_signal_count": sum(
            bool(event["is_asset_exit_signal"])
            for row in classifications
            for event in row["events"]
        ),
        "needs_manual_review_count": sum(
            bool(event["needs_manual_review"])
            for row in classifications
            for event in row["events"]
        ),
        "out_of_scan_identity_article_count": sum(
            any(
                event["is_identity_change"]
                and not event["in_scan_set"]
                and not event["needs_manual_review"]
                for event in row["events"]
            )
            for row in classifications
        ),
        "raw_evidence_pass_count": raw_evidence_pass_count,
        "normalized_evidence_pass_count": normalized_evidence_pass_count,
        "evidence_event_count": evidence_event_count,
        "normalization_rescued_count": (
            normalized_evidence_pass_count - raw_evidence_pass_count
        ),
        "normalization_rescue_headline": (
            normalized_evidence_pass_count > raw_evidence_pass_count
            and (
                normalized_evidence_pass_count - raw_evidence_pass_count
            )
            / (evidence_event_count - raw_evidence_pass_count)
            > 0.20
        ),
        "run_usd": run_usd,
        "accepted_chunks": accepted_chunks,
        "combined_classification_sha256": file_sha256(combined_path),
        "protected_rows_read": False,
        "performance_or_death_metrics_computed": False,
        "main_research_db_written": False,
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _atomic_write_bytes(output_dir / "combined-summary.json", _canonical_bytes(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run and combine EXP-101 five-article batches")
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--usage-ledger", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    args = parser.parse_args()
    model = os.getenv("RESEARCH_REVIEW_MODEL", "").strip()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    run_batches(
        args.candidate_csv,
        args.corpus_dir,
        args.batch_root,
        args.usage_ledger,
        model=model,
        api_key=api_key,
    )
    result = aggregate_batches(
        args.candidate_csv,
        args.corpus_dir,
        args.batch_root,
        args.usage_ledger,
        args.combined_output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
