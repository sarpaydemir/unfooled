from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_factory.death_census import (
    CUTOFF,
    EXPERIMENT_ID,
    SYSTEMATIC_ANNOUNCEMENT_START,
    _atomic_write_bytes,
    _load_announcement_corpus_for_parser,
    detect_announcement_event_kinds,
    file_sha256,
)
from research_factory.hashing import canonical_json_sha256


AUDIT_TASK_SHA256 = "9d5176e679c5882fd1fde079e59f95e0bad6ac496df12b671931008129211919"
EXPECTED_BROAD_PATTERN_COUNT = 869

KNOWN_REAL_EVENT_ARTICLES: tuple[Mapping[str, str], ...] = (
    {
        "event": "LUNA_TO_LUNC",
        "old_symbol": "LUNA",
        "new_symbol": "LUNC",
        "article_code": "d044a6742e484b77a170111460b0eed3",
        "title": "Binance Will List Terra 2.0 (LUNA) in the Innovation Zone *LUNA (old) Renamed as LUNC",
    },
    {
        "event": "MATIC_TO_POL",
        "old_symbol": "MATIC",
        "new_symbol": "POL",
        "article_code": "6a6de383727f4659a3050f7982e1620f",
        "title": "Binance Will Support the Polygon (MATIC) Token Swap to Polygon (POL)",
    },
    {
        "event": "FTM_TO_S",
        "old_symbol": "FTM",
        "new_symbol": "S",
        "article_code": "aec6fcbc84b749eeab6690e6bcac2f3d",
        "title": "Binance Will Support the Fantom (FTM) Token Swap and Rebranding to Sonic (S)",
    },
    {
        "event": "BTT_REDENOMINATION",
        "old_symbol": "BTT",
        "new_symbol": "BTTC",
        "article_code": "2722e6da4f5141dd9b2fb07b5b1f3f75",
        "title": "Binance Will Support the BitTorrent (BTT) Token Migration, Redenomination and Renaming Plan",
    },
    {
        "event": "PAX_TO_USDP",
        "old_symbol": "PAX",
        "new_symbol": "USDP",
        "article_code": "c313480177e4457bb1e1b848c88bbf78",
        "title": "Binance Will Support the Paxos Standard (PAX) Rebrand to Pax Dollar (USDP)",
    },
    {
        "event": "BCHABC_TO_BCH",
        "old_symbol": "BCHABC",
        "new_symbol": "BCH",
        "article_code": "360037219511",
        "title": "Binance Will Update the Ticker of BCHABC to BCH",
    },
)

ALTERNATE_REAL_EVENT_ARTICLES: Mapping[str, tuple[Mapping[str, str], ...]] = {
    "LUNA_TO_LUNC": (
        {
            "article_code": "c52fa3c686be4b2b9d5df50de15847ec",
            "title": "Binance Will Support the Terra Network Airdrop",
        },
        {
            "article_code": "9c8abd90df5d4890822e6988f7dc8f0c",
            "title": "Binance Staking Completes First Phase of Terra 2.0 (LUNA) Airdrop Distribution",
        },
    )
}

RECALL_AUDIT_FIELDS = (
    "event",
    "old_symbol",
    "new_symbol",
    "article_code",
    "release_date_utc",
    "title",
    "corpus_present",
    "systematic_window",
    "canonical_broad_pattern_match",
    "canonical_narrowed_candidate",
    "event_evidence_article_codes",
    "event_candidate_article_codes",
    "event_narrowed_candidate",
    "canonical_diagnostic",
    "status",
    "exclusion_stage",
    "exclusion_reason",
)


class RecallAuditError(RuntimeError):
    pass


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECALL_AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_recall_status(
    *,
    systematic_window: bool,
    broad_pattern_match: bool,
    narrowed_candidate: bool,
    old_symbol: str,
    new_symbol: str,
    title: str,
) -> tuple[str, str, str]:
    if not systematic_window:
        return (
            "outside_systematic_window",
            "coverage",
            "official_article_predates_2022_02_systematic_window",
        )
    if not broad_pattern_match:
        return (
            "blocked",
            "broad_pattern",
            "real_identity_article_missing_from_869_pattern_surface",
        )
    if narrowed_candidate:
        return ("covered", "none", "none")
    old = re.escape(old_symbol)
    new = re.escape(new_symbol)
    uses_as = bool(
        re.search(
            rf"(?<![A-Za-z0-9]){old}(?![A-Za-z0-9]).*?\bas\b.*?"
            rf"(?<![A-Za-z0-9]){new}(?![A-Za-z0-9])",
            title,
            flags=re.IGNORECASE,
        )
    )
    reason = (
        "D1_D3e_direction_marker_as_not_allowed"
        if uses_as
        else "D1_D2_D3_D4_narrowed_surface_exclusion"
    )
    return ("blocked", "narrowed_D1_D4", reason)


def build_recall_rows(
    index: Sequence[Mapping[str, Any]],
    broad_codes: set[str],
    candidate_codes: set[str],
) -> list[dict[str, Any]]:
    by_code = {str(row["code"]): row for row in index}
    rows: list[dict[str, Any]] = []
    for frozen in KNOWN_REAL_EVENT_ARTICLES:
        code = str(frozen["article_code"])
        evidence = (frozen, *ALTERNATE_REAL_EVENT_ARTICLES.get(str(frozen["event"]), ()))
        for expected in evidence:
            evidence_article = by_code.get(str(expected["article_code"]))
            if evidence_article is None or str(evidence_article.get("title")) != expected["title"]:
                raise RecallAuditError(
                    f"known real announcement identity mismatch: {frozen['event']}"
                )
        article = by_code[code]
        release_ms = int(article["release_date_ms"])
        systematic = release_ms >= int(SYSTEMATIC_ANNOUNCEMENT_START.timestamp() * 1000)
        evidence_codes = tuple(str(item["article_code"]) for item in evidence)
        event_broad_codes = tuple(sorted(set(evidence_codes) & broad_codes))
        event_candidate_codes = tuple(sorted(set(evidence_codes) & candidate_codes))
        broad = bool(event_broad_codes)
        candidate = bool(event_candidate_codes)
        status, stage, reason = classify_recall_status(
            systematic_window=systematic,
            broad_pattern_match=broad,
            narrowed_candidate=candidate,
            old_symbol=str(frozen["old_symbol"]),
            new_symbol=str(frozen["new_symbol"]),
            title=str(frozen["title"]),
        )
        rows.append(
            {
                **frozen,
                "release_date_utc": datetime.fromtimestamp(
                    release_ms / 1000,
                    tz=timezone.utc,
                ).isoformat(),
                "corpus_present": True,
                "systematic_window": systematic,
                "canonical_broad_pattern_match": code in broad_codes,
                "canonical_narrowed_candidate": code in candidate_codes,
                "event_evidence_article_codes": ",".join(evidence_codes),
                "event_candidate_article_codes": ",".join(event_candidate_codes),
                "event_narrowed_candidate": candidate,
                "canonical_diagnostic": (
                    "canonical_article_covered"
                    if code in candidate_codes
                    else "canonical_article_missed_event_covered"
                    if candidate
                    else "canonical_article_and_event_missed"
                ),
                "status": status,
                "exclusion_stage": stage,
                "exclusion_reason": reason,
            }
        )
    return rows


def run_recall_audit(
    corpus_dir: Path,
    candidate_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise RecallAuditError("recall audit output already exists; refusing overwrite")
    corpus_manifest, index = _load_announcement_corpus_for_parser(corpus_dir)
    summary_path = candidate_dir / "parser-summary.json"
    candidate_path = candidate_dir / "announcement-candidates.csv"
    if any(path.is_symlink() or not path.is_file() for path in (summary_path, candidate_path)):
        raise RecallAuditError("narrowed candidate inputs missing or symlinked")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stored = summary.pop("summary_sha256", None)
    if stored != canonical_json_sha256(summary):
        raise RecallAuditError("narrowed parser summary checksum mismatch")
    summary["summary_sha256"] = stored
    expected_candidate_count = summary.get("selected_symbol_intersection_count")
    expected_candidate_sha = summary.get("artifacts", {}).get(candidate_path.name)
    if (
        summary.get("experiment_id") != EXPERIMENT_ID
        or summary.get("task_sha256") != AUDIT_TASK_SHA256
        or summary.get("phase") != "announcement_candidate_parser"
        or summary.get("pattern_match_count") != EXPECTED_BROAD_PATTERN_COUNT
        or summary.get("model_called") is not False
        or summary.get("model_articles_read") != 0
        or not isinstance(expected_candidate_count, int)
        or expected_candidate_sha != file_sha256(candidate_path)
    ):
        raise RecallAuditError("narrowed parser binding mismatch")
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        candidate_rows = list(csv.DictReader(handle))
    candidate_codes = {str(row["article_code"]) for row in candidate_rows}
    if len(candidate_rows) != expected_candidate_count or len(candidate_codes) != len(candidate_rows):
        raise RecallAuditError("narrowed candidate identity invariant failed")

    start_ms = int(SYSTEMATIC_ANNOUNCEMENT_START.timestamp() * 1000)
    broad_codes: set[str] = set()
    details = corpus_dir / "articles"
    for article in index:
        if int(article["release_date_ms"]) < start_ms:
            continue
        code = str(article["code"])
        detail = json.loads((details / f"{code}.json").read_text(encoding="utf-8"))
        combined = f"{article['title']}\n{detail.get('title') or ''}\n{detail['body']}"
        if detect_announcement_event_kinds(combined):
            broad_codes.add(code)
    if len(broad_codes) != EXPECTED_BROAD_PATTERN_COUNT:
        raise RecallAuditError("broad 869 pattern surface replay mismatch")

    rows = build_recall_rows(index, broad_codes, candidate_codes)
    blocked = [row for row in rows if row["status"] == "blocked"]
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "known-event-recall-audit.csv"
    _write_csv(csv_path, rows)
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "known_real_event_recall_audit",
        "outcome": "real_corpus_recall_blocked" if blocked else "real_corpus_recall_passed",
        "task_sha256": AUDIT_TASK_SHA256,
        "cutoff_end_exclusive": CUTOFF.isoformat(),
        "systematic_start_inclusive": SYSTEMATIC_ANNOUNCEMENT_START.isoformat(),
        "corpus_manifest_sha256": corpus_manifest["manifest_sha256"],
        "parser_summary_sha256": stored,
        "candidate_csv_sha256": file_sha256(candidate_path),
        "event_recall_unit": "known_event_any_bound_official_article",
        "known_event_count": len(rows),
        "covered_count": sum(row["status"] == "covered" for row in rows),
        "outside_systematic_window_count": sum(
            row["status"] == "outside_systematic_window" for row in rows
        ),
        "blocked_count": len(blocked),
        "blocked_events": [row["event"] for row in blocked],
        "systematic_window_event_count": sum(bool(row["systematic_window"]) for row in rows),
        "systematic_window_covered_count": sum(
            bool(row["systematic_window"]) and row["status"] == "covered" for row in rows
        ),
        "systematic_window_canonical_covered_count": sum(
            bool(row["systematic_window"]) and bool(row["canonical_narrowed_candidate"])
            for row in rows
        ),
        "broad_pattern_replay_count": len(broad_codes),
        "narrowed_candidate_count": len(candidate_codes),
        "paid_model_call_allowed": not blocked,
        "model_called": False,
        "model_articles_read": 0,
        "performance_or_death_metrics_computed": False,
        "protected_rows_read": False,
        "artifacts": {csv_path.name: file_sha256(csv_path)},
    }
    result["summary_sha256"] = canonical_json_sha256(result)
    _atomic_write_bytes(output_dir / "recall-audit-summary.json", _canonical_bytes(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="EXP-101 known-real-event recall audit")
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_recall_audit(args.corpus_dir, args.candidate_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
