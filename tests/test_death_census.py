from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pandas as pd
import research_factory.death_census as death_census

from research_factory.death_census import (
    ANNOUNCEMENT_CATALOGS,
    CUTOFF,
    IDENTITY_EVENT_FIXTURES,
    KLINE_FIELDS,
    STABLE_FIAT_BASES,
    DeathCensusError,
    Selection,
    _announcement_index,
    _catalog_article_index,
    _load_catalog_checkpoint,
    _selection_dates,
    _write_catalog_checkpoint,
    build_selection_scan_universe,
    build_rename_successor_rows,
    candidate_evidence_snippets,
    detect_announcement_event_kinds,
    detect_identity_event_kinds,
    download_announcement_details,
    fetch_cms_json,
    match_selected_bases,
    parse_catalog_metadata,
    parse_catalog_article_page,
    parse_archive_symbol_page,
    parse_full_asset_delist_title,
    select_at,
    scan_announcement_candidate_rows,
    selected_rows_from_available,
    validate_identity_event_acceptance,
    validate_symbol_csv,
)


def frame(
    start: str,
    periods: int,
    *,
    quote_volume: float,
    missing: tuple[int, ...] = (),
) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    dates = dates.delete(list(missing)) if missing else dates
    return pd.DataFrame(
        {
            "open": 1.0,
            "high": 1.1,
            "low": 0.9,
            "close": 1.0,
            "volume": 10.0,
            "quote_volume": quote_volume,
            "trade_count": 5,
        },
        index=dates,
    )


class DeathCensusTest(unittest.TestCase):
    def test_announcement_event_detector_covers_frozen_surfaces(self) -> None:
        self.assertEqual(
            detect_announcement_event_kinds(
                "Binance Will List Alpha (AAA) and Open Trading for AAA/USDT"
            ),
            ("listing",),
        )
        self.assertEqual(
            detect_announcement_event_kinds(
                "Notice of Removal of Spot Trading Pairs - AAA/USDT"
            ),
            ("removal",),
        )
        self.assertEqual(
            detect_announcement_event_kinds("Ethereum Network Upgrade and Hard Fork"),
            (),
        )

    def test_selected_base_match_requires_usdt_pair_or_structured_event(self) -> None:
        kinds = detect_announcement_event_kinds(
            "Removal of Spot Trading Pairs: BTC/EUR and ONE/BTC"
        )
        matched, reasons = match_selected_bases(
            "Removal of Spot Trading Pairs",
            "Removal of Spot Trading Pairs: BTC/EUR and ONE/BTC",
            ("BTC", "ONE", "ID"),
            kinds,
        )
        self.assertEqual(matched, ())
        self.assertEqual(reasons, ())

        matched, reasons = match_selected_bases(
            "Removal of Spot Trading Pairs",
            "Removal of Spot Trading Pairs: ABC/USDT",
            ("ABC", "BTC"),
            ("removal",),
        )
        self.assertEqual(matched, ("ABC",))
        self.assertEqual(reasons, ("ABC:removal",))

        matched, reasons = match_selected_bases(
            "LUNA Renaming to LUNC",
            "LUNA Renaming to LUNC",
            ("LUNA", "LUNC", "ONE", "ID"),
            ("rename",),
        )
        self.assertEqual(matched, ("LUNA", "LUNC"))
        self.assertEqual(
            reasons,
            ("LUNA:direct_identity", "LUNC:direct_identity"),
        )

        matched, reasons = match_selected_bases(
            "LUNA (old) Renamed as LUNC",
            "LUNA (old) Renamed as LUNC",
            ("LUNA", "LUNC", "ONE", "ID"),
            ("rename",),
        )
        self.assertEqual(matched, ("LUNA", "LUNC"))
        self.assertEqual(
            reasons,
            ("LUNA:direct_identity", "LUNC:direct_identity"),
        )

        matched, reasons = match_selected_bases(
            "LUNA and LUNC are available as trading assets",
            "LUNA and LUNC are available as trading assets",
            ("LUNA", "LUNC"),
            (),
        )
        self.assertEqual(matched, ())
        self.assertEqual(reasons, ())

    def test_candidate_scan_orders_pattern_then_selected_intersection(self) -> None:
        start_ms = int(pd.Timestamp("2022-02-01T00:00:00Z").timestamp() * 1000)
        index = [
            {"code": "old", "title": "OLD Token Swap to NEW", "release_date_ms": start_ms - 1, "catalog_ids": [49]},
            {"code": "hit", "title": "OLD Token Swap to NEW", "release_date_ms": start_ms, "catalog_ids": [49]},
            {"code": "other", "title": "XYZ Token Swap to QQQ", "release_date_ms": start_ms + 1, "catalog_ids": [49]},
            {"code": "noise", "title": "OLD Weekly Update", "release_date_ms": start_ms + 2, "catalog_ids": [49]},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            details = Path(temporary)
            for row in index:
                (details / f"{row['code']}.json").write_text(
                    json.dumps({"code": row["code"], "title": row["title"], "body": row["title"]}),
                    encoding="utf-8",
                )
            rows, counts = scan_announcement_candidate_rows(
                index,
                details,
                ("OLD",),
                ("OLD", "NEW", "XYZ", "QQQ"),
            )
        self.assertEqual([row["article_code"] for row in rows], ["hit"])
        self.assertEqual(counts["corpus_article_count"], 4)
        self.assertEqual(counts["systematic_window_article_count"], 3)
        self.assertEqual(counts["pattern_match_count"], 2)
        self.assertEqual(counts["selected_symbol_intersection_count"], 1)
        self.assertIn(
            "OLD Token Swap",
            candidate_evidence_snippets("OLD Token Swap to NEW", ("OLD",))[0],
        )

    def test_candidate_union_adds_temporal_removal_then_later_listing(self) -> None:
        start_ms = int(pd.Timestamp("2022-02-01T00:00:00Z").timestamp() * 1000)
        index = [
            {"code": "early-list", "title": "Binance Will List Alpha (AAA)", "release_date_ms": start_ms, "catalog_ids": [48]},
            {"code": "remove", "title": "Removal of Spot Trading Pairs", "release_date_ms": start_ms + 1, "catalog_ids": [161]},
            {"code": "context", "title": "The Rename of Alpha (AAA)", "release_date_ms": start_ms + 2, "catalog_ids": [49]},
            {"code": "later-list", "title": "Binance Will Re-list Alpha (AAA)", "release_date_ms": start_ms + 3, "catalog_ids": [48]},
        ]
        bodies = {
            "early-list": "AAA/USDT",
            "remove": "Removal of Spot Trading Pairs: AAA/USDT",
            "context": "The Rename of Alpha (AAA) is under review.",
            "later-list": "Trading opens for AAA/USDT.",
        }
        with tempfile.TemporaryDirectory() as temporary:
            details = Path(temporary)
            for row in index:
                (details / f"{row['code']}.json").write_text(
                    json.dumps(
                        {
                            "code": row["code"],
                            "title": row["title"],
                            "body": bodies[row["code"]],
                        }
                    ),
                    encoding="utf-8",
                )
            rows, counts = scan_announcement_candidate_rows(
                index, details, ("AAA",), ("AAA",)
            )
        by_code = {row["article_code"]: row for row in rows}
        self.assertEqual(set(by_code), {"remove", "context", "later-list"})
        self.assertIn("AAA:temporal_removal_t1", by_code["remove"]["match_reasons"])
        self.assertIn("AAA:temporal_listing_t2", by_code["later-list"]["match_reasons"])
        self.assertIn("AAA:temporal_identity_context", by_code["context"]["match_reasons"])
        self.assertEqual(counts["temporal_symbol_count"], 1)

    def test_detail_download_is_sequential_resumable_and_rejects_extras(self) -> None:
        index = [
            {"code": "a", "title": "A", "release_date_ms": 1, "catalog_ids": [48]},
            {"code": "b", "title": "B", "release_date_ms": 2, "catalog_ids": [48]},
        ]
        calls: list[str] = []

        def fetcher(url: str) -> dict[str, object]:
            code = url.split("articleCode=")[1]
            calls.append(code)
            return {"data": {"title": code.upper(), "body": f"body-{code}"}}

        with tempfile.TemporaryDirectory() as temporary:
            details = Path(temporary)
            (details / "a.json").write_text(
                json.dumps({"code": "a", "title": "A", "body": "body-a"}),
                encoding="utf-8",
            )
            self.assertEqual(download_announcement_details(index, details, fetcher), 1)
            self.assertEqual(calls, ["b"])
            self.assertEqual(download_announcement_details(index, details, fetcher), 0)
            self.assertEqual(calls, ["b"])
            (details / "extra.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(DeathCensusError, "unindexed"):
                download_announcement_details(index, details, fetcher)

    def test_catalog_checkpoint_is_sha_and_cutoff_bound(self) -> None:
        rows = [{"code": "past", "title": "Past", "release_date_ms": 1}]
        summary = {
            "catalog_id": 48,
            "catalog_name": ANNOUNCEMENT_CATALOGS[48],
            "article_total_current": 2,
            "article_rows_seen": 2,
            "article_count_before_cutoff": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog-48.json"
            _write_catalog_checkpoint(path, rows, summary)
            loaded_rows, loaded_summary = _load_catalog_checkpoint(
                path, 48, ANNOUNCEMENT_CATALOGS[48]
            )
            self.assertEqual(loaded_rows, rows)
            self.assertEqual(loaded_summary, summary)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["articles"][0]["title"] = "tampered"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(DeathCensusError, "checksum"):
                _load_catalog_checkpoint(path, 48, ANNOUNCEMENT_CATALOGS[48])

    def test_cms_fetch_retries_rate_limit_under_scoped_policy(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self) -> bytes:
                return b'{"ok":true}'

        calls = 0

        def urlopen(_request, timeout):
            nonlocal calls
            self.assertEqual(timeout, 60)
            calls += 1
            if calls == 1:
                raise HTTPError("https://example.invalid", 429, "rate", {"Retry-After": "0"}, None)
            return Response()

        death_census._CMS_NEXT_REQUEST_AT = 0.0
        try:
            with patch.object(death_census.urllib.request, "urlopen", side_effect=urlopen), patch.object(
                death_census.time, "sleep"
            ):
                self.assertEqual(fetch_cms_json("https://example.invalid"), {"ok": True})
        finally:
            death_census._CMS_NEXT_REQUEST_AT = 0.0
        self.assertEqual(calls, 2)

    def test_identity_detector_accepts_all_six_frozen_known_events(self) -> None:
        rows = validate_identity_event_acceptance()
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {row["event"] for row in rows},
            {fixture["event"] for fixture in IDENTITY_EVENT_FIXTURES},
        )
        self.assertTrue(all(row["passed"] and row["detected_kinds"] for row in rows))

    def test_identity_detector_context_guards_and_candidate_only_scope(self) -> None:
        for negative in (
            "Binance Supports the Ethereum Network Upgrade and Hard Fork",
            "New Futures and Margin Promotion with Reduced Fees",
            "Network Migration Is Complete",
        ):
            self.assertEqual(detect_identity_event_kinds(negative), ())
        self.assertEqual(
            detect_identity_event_kinds("ABC Token Migration to the New Chain"),
            ("migration",),
        )
        self.assertIn(
            "conversion",
            detect_identity_event_kinds("Binance Will Auto-Convert OLD to NEW"),
        )

    def test_catalog_article_page_requires_frozen_identity(self) -> None:
        payload = {
            "data": {
                "catalogs": [{
                    "catalogId": 48,
                    "catalogName": ANNOUNCEMENT_CATALOGS[48],
                    "total": 1,
                    "articles": [{"code": "a", "title": "A", "releaseDate": 1}],
                }]
            }
        }
        total, rows = parse_catalog_article_page(payload, 48, ANNOUNCEMENT_CATALOGS[48])
        self.assertEqual(total, 1)
        self.assertEqual(rows[0]["code"], "a")
        with self.assertRaisesRegex(DeathCensusError, "identity"):
            parse_catalog_article_page(payload, 49, ANNOUNCEMENT_CATALOGS[49])

    def test_multi_catalog_index_counts_post_cutoff_rows_without_persisting_them(self) -> None:
        calls: list[int] = []

        def fetcher(url: str) -> dict[str, object]:
            self.assertIn("pageSize=50", url)
            page = int(url.split("pageNo=")[1].split("&")[0])
            calls.append(page)
            rows = {
                1: [{"code": "future", "title": "future", "releaseDate": int(CUTOFF.timestamp() * 1000)}],
                2: [{"code": "past", "title": "past", "releaseDate": int(CUTOFF.timestamp() * 1000) - 1}],
            }[page]
            return {
                "data": {
                    "catalogs": [{
                        "catalogId": 48,
                        "catalogName": ANNOUNCEMENT_CATALOGS[48],
                        "total": 2,
                        "articles": rows,
                    }]
                }
            }

        rows, summary = _catalog_article_index(48, ANNOUNCEMENT_CATALOGS[48], fetcher)
        self.assertEqual(calls, [1, 2])
        self.assertEqual([row["code"] for row in rows], ["past"])
        self.assertEqual(summary["article_rows_seen"], 2)
        self.assertEqual(summary["article_count_before_cutoff"], 1)

    def test_catalog_inventory_parser_uses_metadata_not_articles(self) -> None:
        payload = {
            "data": {
                "catalogs": [
                    {
                        "catalogId": 48,
                        "parentCatalogId": None,
                        "catalogName": "New Cryptocurrency Listing",
                        "description": None,
                        "catalogType": 1,
                        "total": 2225,
                        "articles": [{"title": "SYMBOL_RESULT_MUST_NOT_BE_PERSISTED"}],
                    }
                ]
            }
        }
        row = parse_catalog_metadata(payload, 48)
        self.assertEqual(row["catalog_name"], "New Cryptocurrency Listing")
        self.assertNotIn("articles", row)
        self.assertNotIn("SYMBOL", json.dumps(row))

    def test_catalog_inventory_parser_rejects_wrong_identity(self) -> None:
        payload = {
            "data": {
                "catalogs": [
                    {
                        "catalogId": 49,
                        "catalogName": "Mismatch",
                        "catalogType": 1,
                        "total": 1,
                    }
                ]
            }
        }
        with self.assertRaisesRegex(DeathCensusError, "identity"):
            parse_catalog_metadata(payload, 48)

    def test_unavailable_partial_basket_never_enters_exit_candidates(self) -> None:
        def rows(count: int, prefix: str) -> tuple[dict[str, object], ...]:
            return tuple(
                {
                    "symbol": f"{prefix}{index}USDT",
                    "base_asset": f"{prefix}{index}",
                    "mean_quote_volume_180d": float(count - index),
                    "observed_returns_before_t": 180,
                    "leveraged_token": False,
                    "quote_like_asset": False,
                }
                for index in range(count)
            )

        unavailable = Selection(pd.Timestamp("2019-05-01T00:00:00Z"), 19, rows(19, "OLD"))
        available = Selection(pd.Timestamp("2019-06-01T00:00:00Z"), 20, rows(20, "NEW"))
        frozen, counts = selected_rows_from_available((unavailable, available))
        self.assertEqual(len(frozen), 20)
        self.assertEqual(set(counts), {f"NEW{index}USDT" for index in range(20)})

    def test_selection_calendar_is_utc_and_cutoff_safe(self) -> None:
        dates = _selection_dates()
        self.assertEqual(dates[0], pd.Timestamp("2018-01-01T00:00:00Z"))
        self.assertEqual(dates[-1], pd.Timestamp("2025-01-01T00:00:00Z"))
        self.assertTrue(all(stamp.tz is not None and stamp < CUTOFF for stamp in dates))

    def test_announcement_pagination_counts_filtered_future_rows(self) -> None:
        calls: list[int] = []

        def fetcher(url: str) -> dict[str, object]:
            page = int(url.split("pageNo=")[1].split("&")[0])
            calls.append(page)
            rows = {
                1: [
                    {
                        "code": "future",
                        "title": "future row",
                        "releaseDate": int(CUTOFF.timestamp() * 1000) + 1,
                    }
                ],
                2: [
                    {
                        "code": "past",
                        "title": "past row",
                        "releaseDate": int(CUTOFF.timestamp() * 1000) - 1,
                    }
                ],
            }[page]
            return {"data": {"catalogs": [{"total": 2, "articles": rows}]}}

        result = _announcement_index(fetcher)
        self.assertEqual(calls, [1, 2])
        self.assertEqual([row["code"] for row in result], ["past"])

    def test_archive_page_parses_symbols_and_continuation(self) -> None:
        payload = b"""<?xml version='1.0'?>
        <ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>
          <IsTruncated>true</IsTruncated>
          <NextContinuationToken>a+b/c==</NextContinuationToken>
          <CommonPrefixes><Prefix>data/spot/monthly/klines/AIONUSDT/</Prefix></CommonPrefixes>
          <CommonPrefixes><Prefix>data/spot/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
        </ListBucketResult>"""
        symbols, token = parse_archive_symbol_page(payload)
        self.assertEqual(symbols, ("AIONUSDT", "BTCUSDT"))
        self.assertEqual(token, "a+b/c==")

    def test_truncated_archive_page_without_token_fails_closed(self) -> None:
        payload = b"""<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'>
        <IsTruncated>true</IsTruncated></ListBucketResult>"""
        with self.assertRaisesRegex(DeathCensusError, "continuation"):
            parse_archive_symbol_page(payload)

    def test_full_asset_delist_title_is_narrow(self) -> None:
        parsed = parse_full_asset_delist_title(
            "Binance Will Delist AION, MIR, ANC on 2023-02-27"
        )
        self.assertEqual(parsed, (("AION", "MIR", "ANC"), pd.Timestamp("2023-02-27").date()))
        self.assertIsNone(
            parse_full_asset_delist_title(
                "Binance Will Delist FTT Margin Pairs, USD-M Futures Contracts on 2022-11-14"
            )
        )
        self.assertIsNone(
            parse_full_asset_delist_title(
                "Notice of Removal of Spot Trading Pairs - 2024-01-05"
            )
        )

    def test_selection_requires_observed_returns_and_is_causal(self) -> None:
        selection_date = pd.Timestamp("2020-07-01T00:00:00Z")
        frames = {
            "AAAUSDT": frame("2019-12-01", 214, quote_volume=100.0),
            "BBBUSDT": frame("2019-12-01", 214, quote_volume=200.0),
            "CCCUSDT": frame("2020-02-01", 151, quote_volume=1_000.0),
        }
        selected = select_at(frames, selection_date)
        self.assertEqual(selected.pool_size, 2)
        self.assertEqual(tuple(row["symbol"] for row in selected.rows), ("BBBUSDT", "AAAUSDT"))

        mutated = {symbol: value.copy() for symbol, value in frames.items()}
        future_date = selection_date + pd.Timedelta(days=1)
        mutated["AAAUSDT"].loc[future_date] = mutated["AAAUSDT"].iloc[-1]
        mutated["AAAUSDT"].loc[future_date, "quote_volume"] = 10**12
        replay = select_at(mutated, selection_date)
        self.assertEqual(replay.rows, selected.rows)

    def test_observed_returns_require_consecutive_days(self) -> None:
        selection_date = pd.Timestamp("2020-07-01T00:00:00Z")
        broken = frame("2019-11-01", 244, quote_volume=100.0, missing=tuple(range(1, 70, 2)))
        selected = select_at({"BROKENUSDT": broken}, selection_date)
        self.assertEqual(selected.pool_size, 0)

    def test_selection_flags_non_coin_product_types_without_excluding(self) -> None:
        selection_date = pd.Timestamp("2020-07-01T00:00:00Z")
        selected = select_at(
            {
                "BTCUPUSDT": frame("2019-12-01", 214, quote_volume=200.0),
                "BUSDUSDT": frame("2019-12-01", 214, quote_volume=100.0),
            },
            selection_date,
        )
        self.assertTrue(selected.rows[0]["leveraged_token"])
        self.assertTrue(selected.rows[1]["quote_like_asset"])

    def test_stable_fiat_rule_applies_before_ranking(self) -> None:
        selection_date = pd.Timestamp("2020-07-01T00:00:00Z")
        frames = {
            "BUSDUSDT": frame("2019-12-01", 214, quote_volume=1_000.0),
            "BTCUSDT": frame("2019-12-01", 214, quote_volume=100.0),
        }
        literal = select_at(frames, selection_date)
        coin_only = select_at(frames, selection_date, exclude_stable_fiat=True)
        self.assertEqual(tuple(row["symbol"] for row in literal.rows), ("BUSDUSDT", "BTCUSDT"))
        self.assertEqual(tuple(row["symbol"] for row in coin_only.rows), ("BTCUSDT",))
        self.assertIn("UST", STABLE_FIAT_BASES)
        self.assertIn("EUR", STABLE_FIAT_BASES)
        self.assertNotIn("PAXG", STABLE_FIAT_BASES)

    def test_scan_prepass_is_stable_fiat_excluded_and_keeps_known_superset(self) -> None:
        selection_date = pd.Timestamp("2020-07-01T00:00:00Z")
        frames = {
            f"C{index:02d}USDT": frame(
                "2019-12-01", 214, quote_volume=float(100 - index)
            )
            for index in range(21)
        }
        frames["BUSDUSDT"] = frame("2019-12-01", 214, quote_volume=10_000.0)
        with patch.object(death_census, "_selection_dates", return_value=(selection_date,)):
            scan_rows, known_rows = build_selection_scan_universe(frames)
        self.assertEqual(len(scan_rows), 20)
        self.assertNotIn("BUSD", {row["base_asset"] for row in scan_rows})
        self.assertIn("BUSD", {row["base_asset"] for row in known_rows})

    def test_successor_audit_applies_redenomination_multiplier(self) -> None:
        frames = {}
        specifications = {
            "BCHABC": ("2019-11-27", 3, 4.0),
            "BCH": ("2019-11-28", 3, 4.0),
            "BTT": ("2022-01-15", 3, 0.002),
            "BTTC": ("2022-01-25", 3, 0.000002),
            "ERD": ("2020-08-29", 3, 20.0),
            "EGLD": ("2020-09-03", 3, 20_000.0),
            "PAX": ("2021-09-04", 3, 1.0),
            "USDP": ("2021-09-10", 3, 1.0),
            "MATIC": ("2024-09-08", 3, 0.4),
            "POL": ("2024-09-13", 3, 0.4),
            "RNDR": ("2024-07-20", 3, 7.0),
            "RENDER": ("2024-07-26", 3, 7.0),
            "FTM": ("2025-01-11", 3, 0.7),
            "S": ("2025-01-16", 3, 0.7),
            "LUNA": ("2022-05-11", 3, 100.0),
            "LUNC": ("2022-09-09", 3, 0.0001),
        }
        for base, (start, periods, close) in specifications.items():
            frames[f"{base}USDT"] = frame(start, periods, quote_volume=1.0).assign(close=close)
        rows = {row["predecessor"]: row for row in build_rename_successor_rows(frames)}
        self.assertAlmostEqual(rows["BTT"]["transition_continuity_ratio"], 1.0)
        self.assertAlmostEqual(rows["ERD"]["transition_continuity_ratio"], 1.0)
        self.assertAlmostEqual(rows["RNDR"]["transition_continuity_ratio"], 1.0)
        self.assertLess(rows["LUNA"]["successor_to_predecessor_peak_ratio"], 0.01)

    def test_validate_symbol_csv_rejects_cutoff_and_bad_price(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "BADUSDT.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(KLINE_FIELDS)
                writer.writerow(
                    [CUTOFF.date().isoformat(), 1, 1, 1, 1, 1, 1, 1]
                )
            with self.assertRaisesRegex(DeathCensusError, "date invariant"):
                validate_symbol_csv(path)

            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(KLINE_FIELDS)
                writer.writerow(["2024-01-01", 1, 1, 1, 0, 1, 1, 1])
            with self.assertRaisesRegex(DeathCensusError, "non-positive"):
                validate_symbol_csv(path)

    def test_validate_symbol_csv_accepts_exact_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "OKUSDT.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(KLINE_FIELDS)
                writer.writerow(["2024-01-01", 1, 2, 0.5, 1.5, 10, 15, 3])
            validated = validate_symbol_csv(path)
            self.assertEqual(len(validated), 1)
            self.assertEqual(validated.index.tz, pd.Timestamp.now(tz="UTC").tz)


if __name__ == "__main__":
    unittest.main()
