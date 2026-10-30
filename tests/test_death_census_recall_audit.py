from __future__ import annotations

import unittest

import pandas as pd

from research_factory.death_census_recall_audit import (
    ALTERNATE_REAL_EVENT_ARTICLES,
    KNOWN_REAL_EVENT_ARTICLES,
    RecallAuditError,
    build_recall_rows,
    classify_recall_status,
)


class DeathCensusRecallAuditTest(unittest.TestCase):
    def test_status_distinguishes_coverage_broad_and_narrow_failures(self) -> None:
        self.assertEqual(
            classify_recall_status(
                systematic_window=False,
                broad_pattern_match=False,
                narrowed_candidate=False,
                old_symbol="PAX",
                new_symbol="USDP",
                title="PAX Rebrand to USDP",
            ),
            (
                "outside_systematic_window",
                "coverage",
                "official_article_predates_2022_02_systematic_window",
            ),
        )
        self.assertEqual(
            classify_recall_status(
                systematic_window=True,
                broad_pattern_match=False,
                narrowed_candidate=False,
                old_symbol="OLD",
                new_symbol="NEW",
                title="OLD Renamed to NEW",
            )[1],
            "broad_pattern",
        )
        self.assertEqual(
            classify_recall_status(
                systematic_window=True,
                broad_pattern_match=True,
                narrowed_candidate=False,
                old_symbol="LUNA",
                new_symbol="LUNC",
                title="LUNA (old) Renamed as LUNC",
            ),
            (
                "blocked",
                "narrowed_D1_D4",
                "D1_D3e_direction_marker_as_not_allowed",
            ),
        )
        self.assertEqual(
            classify_recall_status(
                systematic_window=True,
                broad_pattern_match=True,
                narrowed_candidate=True,
                old_symbol="MATIC",
                new_symbol="POL",
                title="MATIC Token Swap to POL",
            ),
            ("covered", "none", "none"),
        )

    def test_event_level_recall_accepts_bound_alternate_article(self) -> None:
        start_ms = int(pd.Timestamp("2022-02-01T00:00:00Z").timestamp() * 1000)
        index = []
        for position, frozen in enumerate(KNOWN_REAL_EVENT_ARTICLES):
            before_window = frozen["event"] in {
                "BTT_REDENOMINATION",
                "PAX_TO_USDP",
                "BCHABC_TO_BCH",
            }
            index.append(
                {
                    "code": frozen["article_code"],
                    "title": frozen["title"],
                    "release_date_ms": start_ms - 1 if before_window else start_ms + position,
                }
            )
        for alternatives in ALTERNATE_REAL_EVENT_ARTICLES.values():
            for position, article in enumerate(alternatives, start=20):
                index.append(
                    {
                        "code": article["article_code"],
                        "title": article["title"],
                        "release_date_ms": start_ms + position,
                    }
                )
        broad_codes = {
            frozen["article_code"]
            for frozen in KNOWN_REAL_EVENT_ARTICLES
            if frozen["event"] in {"LUNA_TO_LUNC", "MATIC_TO_POL", "FTM_TO_S"}
        }
        candidate_codes = {
            frozen["article_code"]
            for frozen in KNOWN_REAL_EVENT_ARTICLES
            if frozen["event"] in {"MATIC_TO_POL", "FTM_TO_S"}
        }
        candidate_codes.add(
            str(ALTERNATE_REAL_EVENT_ARTICLES["LUNA_TO_LUNC"][0]["article_code"])
        )
        rows = build_recall_rows(index, broad_codes, candidate_codes)
        by_event = {row["event"]: row for row in rows}
        self.assertEqual(by_event["LUNA_TO_LUNC"]["status"], "covered")
        self.assertFalse(by_event["LUNA_TO_LUNC"]["canonical_narrowed_candidate"])
        self.assertEqual(
            by_event["LUNA_TO_LUNC"]["canonical_diagnostic"],
            "canonical_article_missed_event_covered",
        )
        self.assertEqual(by_event["MATIC_TO_POL"]["status"], "covered")
        self.assertEqual(by_event["FTM_TO_S"]["status"], "covered")
        self.assertEqual(
            sum(row["status"] == "outside_systematic_window" for row in rows), 3
        )

    def test_frozen_article_title_mismatch_fails_closed(self) -> None:
        frozen = KNOWN_REAL_EVENT_ARTICLES[0]
        with self.assertRaisesRegex(RecallAuditError, "identity mismatch"):
            build_recall_rows(
                [
                    {
                        "code": frozen["article_code"],
                        "title": "tampered",
                        "release_date_ms": int(
                            pd.Timestamp("2022-05-01T00:00:00Z").timestamp() * 1000
                        ),
                    }
                ],
                {str(frozen["article_code"])},
                set(),
            )

    def test_all_systematic_event_evidence_missing_still_blocks(self) -> None:
        start_ms = int(pd.Timestamp("2022-02-01T00:00:00Z").timestamp() * 1000)
        index = []
        for position, frozen in enumerate(KNOWN_REAL_EVENT_ARTICLES):
            before_window = frozen["event"] in {
                "BTT_REDENOMINATION",
                "PAX_TO_USDP",
                "BCHABC_TO_BCH",
            }
            index.append(
                {
                    "code": frozen["article_code"],
                    "title": frozen["title"],
                    "release_date_ms": start_ms - 1 if before_window else start_ms + position,
                }
            )
        for alternatives in ALTERNATE_REAL_EVENT_ARTICLES.values():
            for position, article in enumerate(alternatives, start=20):
                index.append(
                    {
                        "code": article["article_code"],
                        "title": article["title"],
                        "release_date_ms": start_ms + position,
                    }
                )
        broad_codes = {str(row["article_code"]) for row in KNOWN_REAL_EVENT_ARTICLES}
        rows = build_recall_rows(index, broad_codes, set())
        by_event = {row["event"]: row for row in rows}
        self.assertEqual(by_event["LUNA_TO_LUNC"]["status"], "blocked")
        self.assertFalse(by_event["LUNA_TO_LUNC"]["event_narrowed_candidate"])


if __name__ == "__main__":
    unittest.main()
