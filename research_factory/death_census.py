from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from research_factory.hashing import canonical_json_sha256


EXPERIMENT_ID = "EXP-101-DEATH-CENSUS"
TASK_SHA256 = "9d5176e679c5882fd1fde079e59f95e0bad6ac496df12b671931008129211919"
ANNOUNCEMENT_CORPUS_TASK_SHA256 = "51f44b4401f881d87d36eea2bb6a2de54fbf6e8bdb582305e8539b73da828108"
SOURCE_TASK_SHA256 = "ac2a810ee4b731a90d84dc8dc559ce12693cf399ddc080cd54cee245e5d2e184"
CUTOFF = pd.Timestamp("2025-01-29T00:00:00Z")
SYSTEMATIC_ANNOUNCEMENT_START = pd.Timestamp("2022-02-01T00:00:00Z")
DOWNLOAD_START = pd.Timestamp("2017-01-01T00:00:00Z")
TARGET_SIZE = 20
MIN_OBSERVED_RETURNS = 180
VOLUME_LOOKBACK_DAYS = 180
KLINE_INTERVAL_MS = 86_400_000
KLINE_LIMIT = 1000
DEFAULT_WORKERS = 8
CMS_MIN_REQUEST_INTERVAL_SECONDS = 0.75
CMS_MAX_ATTEMPTS = 12
CMS_ARTICLE_PAGE_SIZE = 50
IDENTITY_CANDIDATE_COST_BRAKE = 100

MARKET_API = "https://data-api.binance.vision"
ARCHIVE_BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ANNOUNCEMENT_API = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"
)
ANNOUNCEMENT_DETAIL_API = (
    "https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query"
)
DELIST_CATALOG_ID = 161
CATALOG_SCAN_START = 1
CATALOG_SCAN_END = 300
ANNOUNCEMENT_CATALOGS: Mapping[int, str] = {
    48: "New Cryptocurrency Listing",
    49: "Latest Binance News",
    51: "API Updates",
    128: "Crypto Airdrop",
    157: "Maintenance Updates",
    161: "Delisting",
}
EXCLUDED_ANNOUNCEMENT_CATALOGS: Mapping[int, str] = {
    50: "New Fiat Listings",
    93: "Latest Activities",
}

_CMS_RATE_LOCK = threading.Lock()
_CMS_NEXT_REQUEST_AT = 0.0

KLINE_FIELDS = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
)
PREFLIGHT_FIELDS = (
    "selection_date",
    "pool_size",
    "selected_count",
    "selection_available",
    "leveraged_count",
    "quote_like_count",
    "selected_symbols",
)
SELECTED_FIELDS = (
    "selection_date",
    "rank",
    "symbol",
    "base_asset",
    "mean_quote_volume_180d",
    "observed_returns_before_t",
    "leveraged_token",
    "quote_like_asset",
)
EXIT_CANDIDATE_FIELDS = (
    "symbol",
    "base_asset",
    "first_observation",
    "last_observation",
    "selected_start_count",
    "official_full_delist_date",
    "official_announcement_code",
    "classification",
)
DIAGNOSTIC_FIELDS = (
    "selection_date",
    "pool_size",
    "selected_count",
    "overlap_with_current_20",
    "selected_symbols",
)
STABLE_FIAT_FIELDS = ("base_asset", "symbols", "rule")
COLLAPSE_AUDIT_FIELDS = (
    "base_asset",
    "event_date",
    "selected_start_count",
    "selected_before_event",
    "selected_on_or_after_event",
    "first_selected",
    "last_selected",
    "exit_candidate_classification",
    "absence_reason",
    "phase_b_blocker",
)
RENAME_AUDIT_FIELDS = (
    "predecessor",
    "successor",
    "multiplier",
    "manager_status",
    "predecessor_last_date",
    "predecessor_last_close",
    "predecessor_peak_close",
    "successor_first_date",
    "successor_first_close",
    "adjusted_successor_value_per_predecessor",
    "transition_continuity_ratio",
    "successor_to_predecessor_peak_ratio",
    "unobserved_gap_days",
)
CATALOG_INVENTORY_FIELDS = (
    "catalog_id",
    "parent_catalog_id",
    "catalog_name",
    "description",
    "catalog_type",
    "article_total_current",
)
ANNOUNCEMENT_CATALOG_SUMMARY_FIELDS = (
    "catalog_id",
    "catalog_name",
    "article_total_current",
    "article_rows_seen",
    "article_count_before_cutoff",
)
ANNOUNCEMENT_CANDIDATE_FIELDS = (
    "release_date_utc",
    "article_code",
    "catalog_ids",
    "event_kinds",
    "matched_bases",
    "match_reasons",
    "title",
    "evidence_snippets",
)
SCAN_PREPASS_FIELDS = (
    "symbol",
    "base_asset",
    "selected_start_count",
    "first_selected",
    "last_selected",
)
KNOWN_BASE_FIELDS = ("symbol", "base_asset")

IDENTITY_EVENT_FIXTURES: tuple[Mapping[str, str], ...] = (
    {
        "event": "LUNA_TO_LUNC",
        "old_symbol": "LUNA",
        "new_symbol": "LUNC",
        "text": "LUNA (old) Renamed as LUNC",
    },
    {
        "event": "MATIC_TO_POL",
        "old_symbol": "MATIC",
        "new_symbol": "POL",
        "text": "Binance Will Support the Polygon (MATIC) Token Swap to Polygon (POL)",
    },
    {
        "event": "FTM_TO_S",
        "old_symbol": "FTM",
        "new_symbol": "S",
        "text": "Binance Will Support the Fantom (FTM) Token Swap and Rebranding to Sonic (S)",
    },
    {
        "event": "BTT_REDENOMINATION",
        "old_symbol": "BTT",
        "new_symbol": "BTTC",
        "text": "Binance Will Support the BitTorrent (BTT) Token Migration, Redenomination and Renaming Plan",
    },
    {
        "event": "PAX_TO_USDP",
        "old_symbol": "PAX",
        "new_symbol": "USDP",
        "text": "Paxos Standard (PAX) Rebrand and Rename to Pax Dollar (USDP)",
    },
    {
        "event": "BCHABC_TO_BCH",
        "old_symbol": "BCHABC",
        "new_symbol": "BCH",
        "text": "Binance Will Rename BCHABC to BCH and Apply the Ticker Change",
    },
)

LEVERAGED_BASE_RE = re.compile(r"(?:UP|DOWN|BULL|BEAR)$")
STABLE_FIAT_BASES = frozenset(
    {
        "AEUR",
        "AUD",
        "BFUSD",
        "BUSD",
        "DAI",
        "EUR",
        "EURI",
        "FDUSD",
        "GBP",
        "PAX",
        "RLUSD",
        "SUSD",
        "TUSD",
        "USD1",
        "USDC",
        "USDE",
        "USDP",
        "USDS",
        "USDSB",
        "USDSOLD",
        "UST",
        "USTC",
        "XUSD",
    }
)
QUOTE_LIKE_BASES = STABLE_FIAT_BASES

KNOWN_COLLAPSE_EVENTS: Mapping[str, str] = {
    "LUNA": "2022-05-14",
    "UST": "2022-05-14",
    "CEL": "2022-06-13",
    "FTT": "2022-11-08",
    "SRM": "2022-11-08",
    "ANC": "2022-05-12",
    "WAVES": "2024-06-17",
    "OMG": "2024-06-17",
    "HNT": "2023-03-24",
}

RENAME_SUCCESSORS: Mapping[str, Mapping[str, Any]] = {
    "BCHABC": {"successor": "BCH", "multiplier": 1.0, "predecessor_end": "2019-11-28", "status": "sensitivity_only"},
    "BTT": {"successor": "BTTC", "multiplier": 1000.0, "predecessor_end": "2022-01-17", "status": "value_preserving"},
    "ERD": {"successor": "EGLD", "multiplier": 0.001, "predecessor_end": "2020-08-31", "status": "value_preserving"},
    "PAX": {"successor": "USDP", "multiplier": 1.0, "predecessor_end": "2021-09-06", "status": "value_preserving"},
    "MATIC": {"successor": "POL", "multiplier": 1.0, "predecessor_end": "2024-09-10", "status": "value_preserving"},
    "RNDR": {"successor": "RENDER", "multiplier": 1.0, "predecessor_end": "2024-07-22", "status": "value_preserving"},
    "FTM": {"successor": "S", "multiplier": 1.0, "predecessor_end": "2025-01-13", "status": "value_preserving"},
    "LUNA": {"successor": "LUNC", "multiplier": 1.0, "predecessor_end": "2022-05-13", "status": "hidden_economic_death"},
}
CURRENT_PRODUCT_BASES = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "ADA",
    "XRP",
    "ZEC",
    "UNI",
    "AAVE",
    "FET",
    "XLM",
    "INJ",
    "BCH",
    "ARB",
    "HBAR",
    "LDO",
    "ICP",
    "OP",
    "ATOM",
    "CRV",
)


class DeathCensusError(RuntimeError):
    pass


@dataclass(frozen=True)
class Selection:
    selection_date: pd.Timestamp
    pool_size: int
    rows: tuple[dict[str, Any], ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _url(path: str, params: Mapping[str, Any]) -> str:
    return f"{path}?{urllib.parse.urlencode(params)}"


def fetch_bytes(url: str, *, attempts: int = 5) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "exp-101-death-census/1"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {418, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise DeathCensusError(f"HTTP failure for {url}: {exc.code}") from exc
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 15)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 == attempts:
                raise DeathCensusError(f"network failure for {url}: {exc}") from exc
            time.sleep(min(2**attempt, 15))
    raise AssertionError("unreachable")


def fetch_json(url: str) -> Any:
    try:
        return json.loads(fetch_bytes(url))
    except json.JSONDecodeError as exc:
        raise DeathCensusError(f"invalid JSON from {url}") from exc


def fetch_cms_json(url: str) -> Any:
    """Fetch CMS JSON under one process-wide rate and backoff budget."""
    global _CMS_NEXT_REQUEST_AT
    request = urllib.request.Request(
        url, headers={"User-Agent": "exp-101-announcement-corpus/1"}
    )
    for attempt in range(CMS_MAX_ATTEMPTS):
        with _CMS_RATE_LOCK:
            delay = max(0.0, _CMS_NEXT_REQUEST_AT - time.monotonic())
            if delay:
                time.sleep(delay)
            _CMS_NEXT_REQUEST_AT = time.monotonic() + CMS_MIN_REQUEST_INTERVAL_SECONDS
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in {418, 429, 500, 502, 503, 504} or attempt + 1 == CMS_MAX_ATTEMPTS:
                exc.close()
                raise DeathCensusError(f"CMS HTTP failure for {url}: {exc.code}") from exc
            retry_after = exc.headers.get("Retry-After")
            status_code = exc.code
            exc.close()
            try:
                backoff = float(retry_after) if retry_after else min(10 * (attempt + 1), 60)
            except ValueError:
                backoff = min(10 * (attempt + 1), 60)
            with _CMS_RATE_LOCK:
                _CMS_NEXT_REQUEST_AT = max(
                    _CMS_NEXT_REQUEST_AT, time.monotonic() + backoff
                )
            print(
                f"cms_backoff status={status_code} seconds={backoff:g} attempt={attempt + 1}",
                flush=True,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt + 1 == CMS_MAX_ATTEMPTS:
                raise DeathCensusError(f"CMS network/schema failure for {url}: {exc}") from exc
            with _CMS_RATE_LOCK:
                _CMS_NEXT_REQUEST_AT = max(
                    _CMS_NEXT_REQUEST_AT, time.monotonic() + min(2 ** attempt, 15)
                )
    raise AssertionError("unreachable")


def parse_archive_symbol_page(payload: bytes) -> tuple[tuple[str, ...], str | None]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise DeathCensusError("invalid Binance archive XML") from exc
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    symbols: list[str] = []
    prefix_root = "data/spot/monthly/klines/"
    for node in root.findall("s3:CommonPrefixes/s3:Prefix", namespace):
        value = (node.text or "").strip()
        if value.startswith(prefix_root) and value.endswith("/"):
            symbols.append(value[len(prefix_root) : -1])
    truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=namespace) or "false")
    token = root.findtext("s3:NextContinuationToken", default=None, namespaces=namespace)
    if truncated == "true" and not token:
        raise DeathCensusError("truncated Binance archive page has no continuation token")
    return tuple(symbols), token


def list_archive_symbols(fetcher: Callable[[str], bytes] = fetch_bytes) -> tuple[str, ...]:
    symbols: set[str] = set()
    token: str | None = None
    while True:
        params: dict[str, Any] = {
            "list-type": "2",
            "prefix": "data/spot/monthly/klines/",
            "delimiter": "/",
        }
        if token:
            params["continuation-token"] = token
        page_symbols, token = parse_archive_symbol_page(fetcher(_url(ARCHIVE_BUCKET, params)))
        symbols.update(page_symbols)
        if not token:
            break
    return tuple(sorted(symbols))


def _api_usdt_symbols(fetcher: Callable[[str], Any] = fetch_json) -> tuple[str, ...]:
    payload = fetcher(_url(f"{MARKET_API}/api/v3/exchangeInfo", {}))
    rows = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise DeathCensusError("exchangeInfo symbol list missing")
    symbols = {
        str(row["symbol"])
        for row in rows
        if isinstance(row, dict) and row.get("quoteAsset") == "USDT" and row.get("symbol")
    }
    return tuple(sorted(symbols))


def _announcement_index(fetcher: Callable[[str], Any] = fetch_json) -> list[dict[str, Any]]:
    articles: dict[str, dict[str, Any]] = {}
    seen_codes: set[str] = set()
    page = 1
    expected_total: int | None = None
    while expected_total is None or len(seen_codes) < expected_total:
        payload = fetcher(
            _url(
                ANNOUNCEMENT_API,
                {
                    "type": 1,
                    "catalogId": DELIST_CATALOG_ID,
                    "pageNo": page,
                    "pageSize": 20,
                },
            )
        )
        try:
            catalog = payload["data"]["catalogs"][0]
            expected_total = int(catalog["total"])
            page_rows = catalog["articles"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeathCensusError("Binance announcement index schema mismatch") from exc
        if not page_rows:
            break
        for row in page_rows:
            code = str(row["code"])
            seen_codes.add(code)
            release_ms = int(row["releaseDate"])
            if release_ms < int(CUTOFF.timestamp() * 1000):
                articles[code] = {
                    "code": code,
                    "title": str(row["title"]),
                    "release_date_ms": release_ms,
                }
        page += 1
        if page > 100:
            raise DeathCensusError("Binance announcement pagination exceeded guard")
    return sorted(articles.values(), key=lambda row: (row["release_date_ms"], row["code"]))


def _announcement_detail(code: str, fetcher: Callable[[str], Any] = fetch_json) -> dict[str, Any]:
    payload = fetcher(_url(ANNOUNCEMENT_DETAIL_API, {"articleCode": code}))
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("body"), str):
        raise DeathCensusError(f"Binance announcement detail schema mismatch: {code}")
    return {"code": code, "title": str(data.get("title") or ""), "body": data["body"]}


def _plain_announcement_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def detect_identity_event_kinds(text: str) -> tuple[str, ...]:
    """Return high-recall identity-event labels; never an asset classification."""
    normalized = _plain_announcement_text(text).casefold()
    kinds: list[str] = []
    direct_patterns = (
        ("token_swap", r"\btoken[ -]?swap\b"),
        ("redenomination", r"\bredenominat(?:e|ed|es|ing|ion)\b"),
        ("rebrand", r"\brebrand(?:ed|ing)?\b"),
        ("ticker_change", r"\bticker\s+(?:symbol\s+)?chang(?:e|ed|es|ing)\b"),
        ("rename", r"\brenam(?:e|ed|es|ing)\b"),
        ("conversion", r"\b(?:auto[ -]?)?convert(?:ed|s|ing)?\b|\b(?:auto[ -]?)?conversion\b"),
    )
    for label, pattern in direct_patterns:
        if re.search(pattern, normalized):
            kinds.append(label)
    if re.search(r"\bmigrat(?:e|ed|es|ing|ion)\b", normalized) and re.search(
        r"\b(?:token|ticker|swap)\b", normalized
    ):
        kinds.append("migration")
    return tuple(kinds)


def detect_announcement_event_kinds(text: str) -> tuple[str, ...]:
    normalized = _plain_announcement_text(text).casefold()
    kinds: list[str] = []
    if re.search(
        r"\bwill\s+list\b|\bre-?list(?:ed|ing|s)?\b|"
        r"\bopen(?:s|ed|ing)?\s+trading\b|"
        r"\badds?\b.{0,120}\bspot\s+trading\s+pairs?\b",
        normalized,
    ):
        kinds.append("listing")
    if re.search(
        r"\bdelist(?:ed|ing|s)?\b|"
        r"\bremoval\s+of\s+(?:selected\s+)?spot\s+trading\s+pairs?\b|"
        r"\bremove(?:d|s|ing)?\b.{0,120}\bspot\s+trading\s+pairs?\b",
        normalized,
    ):
        kinds.append("removal")
    kinds.extend(detect_identity_event_kinds(text))
    return tuple(dict.fromkeys(kinds))


def _announcement_semantic_units(title: str, body: str) -> tuple[str, ...]:
    """Return the title and body sentences without treating the whole article as one unit."""
    units: list[str] = []
    plain_title = _plain_announcement_text(title)
    if plain_title:
        units.append(plain_title)
    marked = re.sub(
        r"</(?:p|div|li|td|tr|h[1-6])\s*>|<br\s*/?>",
        ". ",
        html.unescape(body),
        flags=re.IGNORECASE,
    )
    plain_body = _plain_announcement_text(marked)
    for unit in re.split(r"(?<=[.!?])\s+|[\r\n]+", plain_body):
        normalized = unit.strip(" .")
        if normalized and normalized not in units:
            units.append(normalized)
    return tuple(units)


def _structured_list_values(text: str) -> tuple[str, ...]:
    values: list[str] = []
    item_pattern = re.compile(
        r"<(?P<tag>li|td)\b[^>]*>(?P<value>.*?)</(?P=tag)\s*>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in item_pattern.finditer(text):
        value = _plain_announcement_text(match.group("value"))
        if value:
            values.append(value)
    stripped = text.strip()
    if stripped.startswith(("[", "{")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None

        def collect(value: Any, *, in_array: bool = False) -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item, in_array=True)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item, in_array=False)
            elif in_array and isinstance(value, str) and value.strip():
                values.append(value.strip())

        collect(payload)
    return tuple(dict.fromkeys(values))


def _uppercase_base_occurrences(
    unit: str,
    known_bases: Iterable[str],
) -> tuple[tuple[str, int, int], ...]:
    known = {str(base).upper() for base in known_bases}
    occurrences = []
    for match in re.finditer(r"(?<![A-Za-z0-9])[A-Z0-9]{1,20}(?![A-Za-z0-9])", unit):
        token = match.group(0)
        if token in known:
            occurrences.append((token, match.start(), match.end()))
    return tuple(occurrences)


def _directional_identity_pairs(
    unit: str,
    known_bases: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    occurrences = _uppercase_base_occurrences(unit, known_bases)
    identity_verb = re.compile(
        r"\b(?:token[ -]?swap|migrat(?:e|ed|es|ing|ion)|rebrand(?:ed|ing)?|"
        r"renam(?:e|ed|es|ing)|ticker\s+(?:symbol\s+)?chang(?:e|ed|es|ing)|"
        r"(?:auto[ -]?)?convert(?:ed|s|ing)?|(?:auto[ -]?)?conversion)\b",
        flags=re.IGNORECASE,
    )
    directed = re.compile(r"(?:\b(?:to|into|for|as)\b|→|->)", flags=re.IGNORECASE)
    pairs: list[tuple[str, str]] = []
    if identity_verb.search(unit) is None:
        return ()
    for old, _old_start, old_end in occurrences:
        for new, new_start, _new_end in occurrences:
            if old == new or new_start <= old_end:
                continue
            between = unit[old_end:new_start]
            if directed.search(between) is None:
                continue
            pair = (old, new)
            if pair not in pairs:
                pairs.append(pair)
    return tuple(pairs)


def _structured_bases(
    unit: str,
    known_bases: Iterable[str],
    *,
    list_values: Iterable[str] = (),
) -> tuple[str, ...]:
    known = {str(base).upper() for base in known_bases}
    matched: set[str] = set()
    for base, _start, _end in _uppercase_base_occurrences(unit, known):
        escaped = re.escape(base)
        if re.search(rf"\(\s*{escaped}\s*\)", unit) or re.search(
            rf"(?<![A-Z0-9]){escaped}\s*/\s*USDT(?![A-Z0-9])|"
            rf"(?<![A-Z0-9]){escaped}USDT(?![A-Z0-9])",
            unit,
        ):
            matched.add(base)
    for raw in list_values:
        value = str(raw).strip().upper()
        base = value
        if value.endswith("/USDT"):
            base = value[: -len("/USDT")]
        elif value.endswith("USDT"):
            base = value[: -len("USDT")]
        if base in known and value in {base, f"{base}/USDT", f"{base}USDT"}:
            matched.add(base)
    return tuple(sorted(matched))


def _article_candidate_channels(
    title: str,
    body: str,
    scan_bases: Iterable[str],
    known_bases: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    scan = {str(base).upper() for base in scan_bases}
    known = {str(base).upper() for base in known_bases}
    list_values = _structured_list_values(body)
    direct: set[str] = set()
    identity_support: set[str] = set()
    listing: set[str] = set()
    removal: set[str] = set()
    for index, unit in enumerate(_announcement_semantic_units(title, body)):
        kinds = set(detect_announcement_event_kinds(unit))
        if not kinds:
            continue
        structured = set(
            _structured_bases(
                unit,
                known,
                list_values=list_values if index == 0 else (),
            )
        )
        relations = _directional_identity_pairs(unit, known)
        relation_bases = {base for pair in relations for base in pair}
        identity_kinds = kinds - {"listing", "removal"}
        if identity_kinds:
            identity_support.update((structured | relation_bases) & scan)
            if "redenomination" in identity_kinds:
                direct.update(structured & scan)
            if identity_kinds - {"redenomination"}:
                direct.update(relation_bases & scan)
        if "listing" in kinds:
            listing.update(structured & scan)
        if "removal" in kinds:
            removal.update(structured & scan)
    return {
        "direct_identity": tuple(sorted(direct)),
        "identity_support": tuple(sorted(identity_support)),
        "listing": tuple(sorted(listing)),
        "removal": tuple(sorted(removal)),
    }


def match_selected_bases(
    title: str,
    text: str,
    selected_bases: Iterable[str],
    event_kinds: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    channels = _article_candidate_channels(title, text, selected_bases, selected_bases)
    kinds = set(event_kinds)
    matched: set[str] = set()
    reasons: list[str] = []
    channel_names = ["direct_identity"]
    if "listing" in kinds:
        channel_names.append("listing")
    if "removal" in kinds:
        channel_names.append("removal")
    for channel in channel_names:
        for base in channels[channel]:
            matched.add(base)
            reasons.append(f"{base}:{channel}")
    return tuple(sorted(matched)), tuple(sorted(reasons))


def candidate_evidence_snippets(
    text: str,
    matched_bases: Sequence[str],
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    plain = _plain_announcement_text(text)
    positions: list[int] = []
    trigger = re.compile(
        r"token[ -]?swap|redenominat\w*|rebrand\w*|ticker\s+(?:symbol\s+)?chang\w*|"
        r"renam\w*|(?:auto[ -]?)?convert\w*|(?:auto[ -]?)?conversion|"
        r"migrat\w*|re-?list\w*|delist\w*|removal\s+of\s+(?:selected\s+)?spot\s+trading\s+pairs?",
        flags=re.IGNORECASE,
    )
    positions.extend(match.start() for match in trigger.finditer(plain))
    for base in matched_bases:
        token = re.compile(rf"(?<![A-Z0-9]){re.escape(base)}(?![A-Z0-9])")
        positions.extend(match.start() for match in token.finditer(plain))
    snippets: list[str] = []
    for position in sorted(set(positions)):
        snippet = plain[max(0, position - 100) : min(len(plain), position + 220)].strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) == limit:
            break
    return tuple(snippets)


def validate_identity_event_acceptance() -> tuple[dict[str, Any], ...]:
    known_bases = {
        str(fixture[key]).upper()
        for fixture in IDENTITY_EVENT_FIXTURES
        for key in ("old_symbol", "new_symbol")
    }
    rows: list[dict[str, Any]] = []
    for fixture in IDENTITY_EVENT_FIXTURES:
        kinds = detect_identity_event_kinds(fixture["text"])
        channels = _article_candidate_channels(
            fixture["text"], fixture["text"], known_bases, known_bases
        )
        matched = set(channels["direct_identity"])
        expected = {
            base
            for base, _start, _end in _uppercase_base_occurrences(
                fixture["text"],
                (fixture["old_symbol"], fixture["new_symbol"]),
            )
        }
        if not kinds or matched != expected:
            raise DeathCensusError(
                f"identity detector acceptance miss: {fixture['event']}"
            )
        rows.append(
            {
                **fixture,
                "detected_kinds": list(kinds),
                "matched_bases": list(channels["direct_identity"]),
                "passed": True,
            }
        )
    return tuple(rows)


def parse_catalog_article_page(
    payload: Any,
    expected_catalog_id: int,
    expected_catalog_name: str,
) -> tuple[int, tuple[dict[str, Any], ...]]:
    try:
        catalog = payload["data"]["catalogs"][0]
        catalog_id = int(catalog["catalogId"])
        catalog_name = str(catalog["catalogName"])
        total = int(catalog["total"])
        page_rows = catalog["articles"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DeathCensusError("Binance catalog article schema mismatch") from exc
    if catalog_id != expected_catalog_id or catalog_name != expected_catalog_name:
        raise DeathCensusError(
            f"Binance catalog article identity mismatch: {expected_catalog_id}"
        )
    if total < 0 or not isinstance(page_rows, list):
        raise DeathCensusError("Binance catalog article count/schema mismatch")
    rows: list[dict[str, Any]] = []
    for row in page_rows:
        if not isinstance(row, dict):
            raise DeathCensusError("Binance catalog article row is not an object")
        try:
            code = str(row["code"]).strip()
            title = str(row["title"])
            release_ms = int(row["releaseDate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DeathCensusError("Binance catalog article row mismatch") from exc
        if not code or release_ms <= 0:
            raise DeathCensusError("Binance catalog article identity/date missing")
        rows.append(
            {"code": code, "title": title, "release_date_ms": release_ms}
        )
    return total, tuple(rows)


def _catalog_article_index(
    catalog_id: int,
    catalog_name: str,
    fetcher: Callable[[str], Any] = fetch_json,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: dict[str, dict[str, Any]] = {}
    page = 1
    expected_total: int | None = None
    while expected_total is None or len(all_rows) < expected_total:
        payload = fetcher(
            _url(
                ANNOUNCEMENT_API,
                {
                    "type": 1,
                    "catalogId": catalog_id,
                    "pageNo": page,
                    "pageSize": CMS_ARTICLE_PAGE_SIZE,
                },
            )
        )
        total, page_rows = parse_catalog_article_page(
            payload, catalog_id, catalog_name
        )
        if expected_total is not None and total != expected_total:
            raise DeathCensusError(
                f"catalog total changed during pagination: {catalog_id}"
            )
        expected_total = total
        if not page_rows:
            break
        for row in page_rows:
            existing = all_rows.get(row["code"])
            if existing is not None and existing != row:
                raise DeathCensusError(
                    f"catalog article changed during pagination: {row['code']}"
                )
            all_rows[row["code"]] = row
        page += 1
        if page > max(10, (expected_total + CMS_ARTICLE_PAGE_SIZE - 1) // CMS_ARTICLE_PAGE_SIZE + 5):
            raise DeathCensusError(
                f"Binance catalog pagination exceeded guard: {catalog_id}"
            )
    if expected_total is None or len(all_rows) != expected_total:
        raise DeathCensusError(
            f"Binance catalog pagination incomplete: {catalog_id} "
            f"seen={len(all_rows)} expected={expected_total}"
        )
    cutoff_ms = int(CUTOFF.timestamp() * 1000)
    before_cutoff = [row for row in all_rows.values() if row["release_date_ms"] < cutoff_ms]
    before_cutoff.sort(key=lambda row: (row["release_date_ms"], row["code"]))
    summary = {
        "catalog_id": catalog_id,
        "catalog_name": catalog_name,
        "article_total_current": expected_total,
        "article_rows_seen": len(all_rows),
        "article_count_before_cutoff": len(before_cutoff),
    }
    return before_cutoff, summary


def _load_frozen_article_index(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise DeathCensusError("announcement corpus index is not a list")
    cutoff_ms = int(CUTOFF.timestamp() * 1000)
    seen: set[str] = set()
    for row in payload:
        if not isinstance(row, dict) or set(row) != {
            "catalog_ids", "code", "release_date_ms", "title"
        }:
            raise DeathCensusError("announcement corpus index row mismatch")
        code = str(row["code"])
        if code in seen or int(row["release_date_ms"]) >= cutoff_ms:
            raise DeathCensusError("announcement corpus index identity/cutoff violation")
        catalog_ids = row["catalog_ids"]
        if not isinstance(catalog_ids, list) or not catalog_ids or any(
            int(value) not in ANNOUNCEMENT_CATALOGS for value in catalog_ids
        ):
            raise DeathCensusError("announcement corpus catalog binding mismatch")
        seen.add(code)
    return payload


def _load_catalog_checkpoint(
    path: Path,
    catalog_id: int,
    catalog_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DeathCensusError("announcement catalog checkpoint is not an object")
    stored = payload.pop("checkpoint_sha256", None)
    if stored != canonical_json_sha256(payload):
        raise DeathCensusError("announcement catalog checkpoint checksum mismatch")
    if payload.get("task_sha256") != ANNOUNCEMENT_CORPUS_TASK_SHA256 or payload.get("cutoff_end_exclusive") != CUTOFF.isoformat():
        raise DeathCensusError("announcement catalog checkpoint binding mismatch")
    summary = payload.get("catalog_summary")
    rows = payload.get("articles")
    if not isinstance(summary, dict) or not isinstance(rows, list):
        raise DeathCensusError("announcement catalog checkpoint schema mismatch")
    if int(summary.get("catalog_id", -1)) != catalog_id or summary.get("catalog_name") != catalog_name:
        raise DeathCensusError("announcement catalog checkpoint identity mismatch")
    cutoff_ms = int(CUTOFF.timestamp() * 1000)
    if len(rows) != int(summary.get("article_count_before_cutoff", -1)):
        raise DeathCensusError("announcement catalog checkpoint count mismatch")
    if any(
        not isinstance(row, dict)
        or set(row) != {"code", "release_date_ms", "title"}
        or int(row["release_date_ms"]) >= cutoff_ms
        for row in rows
    ):
        raise DeathCensusError("announcement catalog checkpoint cutoff/schema violation")
    return rows, summary


def _write_catalog_checkpoint(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "task_sha256": ANNOUNCEMENT_CORPUS_TASK_SHA256,
        "cutoff_end_exclusive": CUTOFF.isoformat(),
        "catalog_summary": dict(summary),
        "articles": [dict(row) for row in rows],
    }
    payload["checkpoint_sha256"] = canonical_json_sha256(payload)
    _atomic_write_bytes(path, _canonical_json_bytes(payload))


def download_announcement_details(
    index: Sequence[Mapping[str, Any]],
    details_dir: Path,
    fetcher: Callable[[str], Any] = fetch_cms_json,
) -> int:
    details_dir.mkdir(parents=True, exist_ok=True)
    indexed_codes = {str(row["code"]) for row in index}
    extra = {path.stem for path in details_dir.glob("*.json")} - indexed_codes
    if extra:
        raise DeathCensusError(
            f"unindexed announcement detail files present: {sorted(extra)[:3]}"
        )
    pending: list[str] = []
    for row in index:
        code = str(row["code"])
        path = details_dir / f"{code}.json"
        if path.exists():
            detail = json.loads(path.read_text(encoding="utf-8"))
            if (
                not isinstance(detail, dict)
                or detail.get("code") != code
                or not isinstance(detail.get("body"), str)
            ):
                raise DeathCensusError(f"invalid resumed announcement detail: {code}")
        else:
            pending.append(code)
    for completed, code in enumerate(pending, start=1):
        detail = _announcement_detail(code, fetcher)
        _atomic_write_bytes(
            details_dir / f"{detail['code']}.json", _canonical_json_bytes(detail)
        )
        if completed % 100 == 0 or completed == len(pending):
            print(f"announcement_details={completed}/{len(pending)}", flush=True)
    return len(pending)


def run_announcement_corpus(
    output_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if workers < 1 or workers > 16:
        raise DeathCensusError("workers must be in [1, 16]")
    acceptance = validate_identity_event_acceptance()
    manifest_path = output_dir / "corpus-manifest.json"
    if manifest_path.exists():
        raise DeathCensusError("announcement corpus manifest already exists; refusing overwrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(
        output_dir / "identity-detector-acceptance.json",
        _canonical_json_bytes(acceptance),
    )
    index_path = output_dir / "article-index.json"
    catalog_summary_path = output_dir / "catalog-summary.csv"
    if index_path.exists() != catalog_summary_path.exists():
        raise DeathCensusError("partial announcement index metadata is inconsistent")

    if index_path.exists():
        index = _load_frozen_article_index(index_path)
        with catalog_summary_path.open(encoding="utf-8", newline="") as handle:
            summary_rows = list(csv.DictReader(handle))
        if {int(row["catalog_id"]) for row in summary_rows} != set(ANNOUNCEMENT_CATALOGS):
            raise DeathCensusError("frozen announcement catalog summary mismatch")
        print(f"announcement_index=resume articles={len(index)}", flush=True)
    else:
        by_code: dict[str, dict[str, Any]] = {}
        summary_rows: list[dict[str, Any]] = []
        checkpoint_dir = output_dir / "catalog-checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for catalog_id, catalog_name in ANNOUNCEMENT_CATALOGS.items():
            checkpoint_path = checkpoint_dir / f"catalog-{catalog_id}.json"
            if checkpoint_path.exists():
                rows, catalog_summary = _load_catalog_checkpoint(
                    checkpoint_path, catalog_id, catalog_name
                )
                print(
                    f"catalog_resume={catalog_id} "
                    f"before_cutoff={catalog_summary['article_count_before_cutoff']}",
                    flush=True,
                )
            else:
                rows, catalog_summary = _catalog_article_index(
                    catalog_id, catalog_name, fetch_cms_json
                )
                _write_catalog_checkpoint(checkpoint_path, rows, catalog_summary)
            summary_rows.append(catalog_summary)
            for row in rows:
                existing = by_code.get(row["code"])
                if existing is None:
                    by_code[row["code"]] = {**row, "catalog_ids": [catalog_summary["catalog_id"]]}
                else:
                    if (
                        existing["title"] != row["title"]
                        or existing["release_date_ms"] != row["release_date_ms"]
                    ):
                        raise DeathCensusError(
                            f"cross-catalog article metadata mismatch: {row['code']}"
                        )
                    existing["catalog_ids"].append(catalog_summary["catalog_id"])
            print(
                f"catalog_complete={catalog_summary['catalog_id']} "
                f"before_cutoff={catalog_summary['article_count_before_cutoff']}",
                flush=True,
            )
        index = sorted(by_code.values(), key=lambda row: (row["release_date_ms"], row["code"]))
        for row in index:
            row["catalog_ids"] = sorted(set(row["catalog_ids"]))
        summary_rows.sort(key=lambda row: int(row["catalog_id"]))
        _atomic_write_bytes(index_path, _canonical_json_bytes(index))
        _write_csv(
            catalog_summary_path,
            ANNOUNCEMENT_CATALOG_SUMMARY_FIELDS,
            summary_rows,
        )

    details_dir = output_dir / "articles"
    download_announcement_details(index, details_dir, fetch_cms_json)

    files = {
        str(path.relative_to(output_dir)): file_sha256(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != manifest_path.name
    }
    if len(tuple(details_dir.glob("*.json"))) != len(index):
        raise DeathCensusError("announcement corpus detail count mismatch")
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "announcement_corpus",
        "task_sha256": ANNOUNCEMENT_CORPUS_TASK_SHA256,
        "cutoff_end_exclusive": CUTOFF.isoformat(),
        "included_catalogs": {str(key): value for key, value in ANNOUNCEMENT_CATALOGS.items()},
        "excluded_catalogs": {str(key): value for key, value in EXCLUDED_ANNOUNCEMENT_CATALOGS.items()},
        "acceptance_fixture_count": len(acceptance),
        "acceptance_pass_count": sum(bool(row["passed"]) for row in acceptance),
        "article_count_before_cutoff": len(index),
        "post_cutoff_articles_persisted": False,
        "parser_auto_segments_assets": False,
        "protected_rows_read": False,
        "performance_or_death_metrics_computed": False,
        "files": files,
    }
    summary["manifest_sha256"] = canonical_json_sha256(summary)
    _atomic_write_bytes(manifest_path, _canonical_json_bytes(summary))
    return summary


def _load_announcement_corpus_for_parser(
    corpus_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = corpus_dir / "corpus-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise DeathCensusError("announcement corpus manifest missing or symlinked")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored = manifest.pop("manifest_sha256", None)
    if stored != canonical_json_sha256(manifest):
        raise DeathCensusError("announcement corpus canonical checksum mismatch")
    manifest["manifest_sha256"] = stored
    if (
        manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("phase") != "announcement_corpus"
        or manifest.get("task_sha256") != ANNOUNCEMENT_CORPUS_TASK_SHA256
        or manifest.get("cutoff_end_exclusive") != CUTOFF.isoformat()
        or manifest.get("acceptance_pass_count") != 6
        or manifest.get("post_cutoff_articles_persisted") is not False
        or manifest.get("parser_auto_segments_assets") is not False
    ):
        raise DeathCensusError("announcement corpus contract mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DeathCensusError("announcement corpus file manifest missing")
    for relative, expected in files.items():
        path = corpus_dir / str(relative)
        if path.is_symlink() or not path.is_file() or file_sha256(path) != expected:
            raise DeathCensusError(f"announcement corpus file checksum mismatch: {relative}")
    index = _load_frozen_article_index(corpus_dir / "article-index.json")
    if len(index) != int(manifest.get("article_count_before_cutoff", -1)):
        raise DeathCensusError("announcement corpus article count mismatch")
    return manifest, index


def _load_selected_preflight_bases(
    preflight_dir: Path,
) -> tuple[tuple[str, ...], str, str]:
    summary_path = preflight_dir / "preflight-summary.json"
    selected_path = preflight_dir / "selected-symbols.csv"
    if (
        summary_path.is_symlink()
        or selected_path.is_symlink()
        or not summary_path.is_file()
        or not selected_path.is_file()
    ):
        raise DeathCensusError("selected preflight inputs missing or symlinked")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stored = summary.pop("summary_sha256", None)
    if stored != canonical_json_sha256(summary):
        raise DeathCensusError("selected preflight canonical checksum mismatch")
    summary["summary_sha256"] = stored
    expected_selected_sha = summary.get("artifacts", {}).get("selected-symbols.csv")
    selected_sha = file_sha256(selected_path)
    if (
        summary.get("experiment_id") != EXPERIMENT_ID
        or summary.get("phase") != "classification_preflight"
        or summary.get("task_sha256") != SOURCE_TASK_SHA256
        or summary.get("selected_unique_symbol_count") != 84
        or expected_selected_sha != selected_sha
    ):
        raise DeathCensusError("selected preflight binding mismatch")
    frame = pd.read_csv(selected_path)
    required = {"symbol", "base_asset", "selection_date"}
    if not required.issubset(frame.columns):
        raise DeathCensusError("selected preflight schema mismatch")
    pairs = frame[["symbol", "base_asset"]].drop_duplicates()
    if len(pairs) != 84 or pairs["symbol"].duplicated().any() or pairs["base_asset"].duplicated().any():
        raise DeathCensusError("selected preflight 84-symbol invariant failed")
    for row in pairs.itertuples(index=False):
        if str(row.symbol) != f"{str(row.base_asset)}USDT":
            raise DeathCensusError("selected preflight base/symbol identity mismatch")
    return tuple(sorted(pairs["base_asset"].astype(str))), stored, selected_sha


def build_selection_scan_universe(
    frames: Mapping[str, pd.DataFrame],
    known_symbols: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    selections = [
        select_at(frames, stamp, exclude_stable_fiat=True)
        for stamp in _selection_dates()
    ]
    selected_rows, _counts = selected_rows_from_available(selections)
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        by_symbol.setdefault(str(row["symbol"]), []).append(row)
    scan_rows = [
        {
            "symbol": symbol,
            "base_asset": str(rows[0]["base_asset"]),
            "selected_start_count": len(rows),
            "first_selected": min(str(row["selection_date"]) for row in rows),
            "last_selected": max(str(row["selection_date"]) for row in rows),
        }
        for symbol, rows in sorted(by_symbol.items())
    ]
    frozen_symbols = tuple(sorted(set(known_symbols or frames)))
    if any(not symbol.endswith("USDT") or len(symbol) <= len("USDT") for symbol in frozen_symbols):
        raise DeathCensusError("selection scan prepass known-symbol identity mismatch")
    known_rows = [
        {"symbol": symbol, "base_asset": symbol[: -len("USDT")]}
        for symbol in frozen_symbols
    ]
    return scan_rows, known_rows


def run_selection_scan_prepass(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise DeathCensusError("selection scan prepass output already exists; refusing overwrite")
    manifest = _load_source_manifest(source_dir)
    frames = _load_frames(source_dir)
    known_symbols = tuple(
        line.strip()
        for line in (source_dir / "usdt-symbols.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    scan_rows, known_rows = build_selection_scan_universe(frames, known_symbols)
    if not scan_rows or not known_rows:
        raise DeathCensusError("selection scan prepass produced an empty universe")
    if any(row["base_asset"] in STABLE_FIAT_BASES for row in scan_rows):
        raise DeathCensusError("selection scan prepass retained stable/fiat base")
    output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = output_dir / "scan-selected-symbols.csv"
    known_path = output_dir / "known-base-symbols.csv"
    _write_csv(selected_path, SCAN_PREPASS_FIELDS, scan_rows)
    _write_csv(known_path, KNOWN_BASE_FIELDS, known_rows)
    artifacts = {
        selected_path.name: file_sha256(selected_path),
        known_path.name: file_sha256(known_path),
    }
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "selection_scan_prepass",
        "task_sha256": TASK_SHA256,
        "cutoff_end_exclusive": CUTOFF.isoformat(),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "selection_rule": "monthly_top20_trailing180d_mean_quote_volume",
        "stable_fiat_excluded": True,
        "lifecycle_segmentation_applied": False,
        "death_hazard_or_performance_computed": False,
        "protected_rows_read": False,
        "selected_unique_symbol_count": len(scan_rows),
        "known_base_symbol_count": len(known_rows),
        "artifacts": artifacts,
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _atomic_write_bytes(
        output_dir / "prepass-summary.json", _canonical_json_bytes(summary)
    )
    return summary


def _load_selection_scan_prepass(
    prepass_dir: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], str, str, str]:
    summary_path = prepass_dir / "prepass-summary.json"
    selected_path = prepass_dir / "scan-selected-symbols.csv"
    known_path = prepass_dir / "known-base-symbols.csv"
    for path in (summary_path, selected_path, known_path):
        if path.is_symlink() or not path.is_file():
            raise DeathCensusError("selection scan prepass input missing or symlinked")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stored = summary.pop("summary_sha256", None)
    if stored != canonical_json_sha256(summary):
        raise DeathCensusError("selection scan prepass canonical checksum mismatch")
    summary["summary_sha256"] = stored
    selected_sha = file_sha256(selected_path)
    known_sha = file_sha256(known_path)
    if (
        summary.get("experiment_id") != EXPERIMENT_ID
        or summary.get("phase") != "selection_scan_prepass"
        or summary.get("task_sha256") != TASK_SHA256
        or summary.get("cutoff_end_exclusive") != CUTOFF.isoformat()
        or summary.get("stable_fiat_excluded") is not True
        or summary.get("lifecycle_segmentation_applied") is not False
        or summary.get("death_hazard_or_performance_computed") is not False
        or summary.get("protected_rows_read") is not False
        or summary.get("artifacts", {}).get(selected_path.name) != selected_sha
        or summary.get("artifacts", {}).get(known_path.name) != known_sha
    ):
        raise DeathCensusError("selection scan prepass binding mismatch")
    selected = pd.read_csv(selected_path)
    known = pd.read_csv(known_path)
    if tuple(selected.columns) != SCAN_PREPASS_FIELDS or tuple(known.columns) != KNOWN_BASE_FIELDS:
        raise DeathCensusError("selection scan prepass schema mismatch")
    if (
        len(selected) != int(summary["selected_unique_symbol_count"])
        or len(known) != int(summary["known_base_symbol_count"])
        or selected["base_asset"].duplicated().any()
        or known["base_asset"].duplicated().any()
        or bool(set(selected["base_asset"].astype(str)) & set(STABLE_FIAT_BASES))
        or any(str(row.symbol) != f"{row.base_asset}USDT" for row in selected.itertuples())
        or any(str(row.symbol) != f"{row.base_asset}USDT" for row in known.itertuples())
    ):
        raise DeathCensusError("selection scan prepass universe invariant failed")
    return (
        tuple(sorted(selected["base_asset"].astype(str))),
        tuple(sorted(known["base_asset"].astype(str))),
        str(stored),
        selected_sha,
        known_sha,
    )


def scan_announcement_candidate_rows(
    index: Sequence[Mapping[str, Any]],
    details_dir: Path,
    scan_bases: Sequence[str],
    known_bases: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    known_bases = tuple(known_bases or scan_bases)
    start_ms = int(SYSTEMATIC_ANNOUNCEMENT_START.timestamp() * 1000)
    pattern_match_count = 0
    systematic_article_count = 0
    records: list[dict[str, Any]] = []
    for article in index:
        release_ms = int(article["release_date_ms"])
        if release_ms < start_ms:
            continue
        systematic_article_count += 1
        code = str(article["code"])
        detail_path = details_dir / f"{code}.json"
        if detail_path.is_symlink() or not detail_path.is_file():
            raise DeathCensusError(f"announcement detail missing or symlinked: {code}")
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if detail.get("code") != code or not isinstance(detail.get("body"), str):
            raise DeathCensusError(f"announcement detail identity/schema mismatch: {code}")
        title = str(article["title"] or detail.get("title") or "")
        body = f"{detail.get('title') or ''}. {detail['body']}"
        combined = f"{title}\n{body}"
        kinds = detect_announcement_event_kinds(combined)
        if not kinds:
            continue
        pattern_match_count += 1
        channels = _article_candidate_channels(title, body, scan_bases, known_bases)
        if not any(channels.values()):
            continue
        release = datetime.fromtimestamp(release_ms / 1000, tz=timezone.utc).isoformat()
        records.append(
            {
                "release_date_utc": release,
                "release_date_ms": release_ms,
                "article_code": code,
                "catalog_ids": ",".join(str(value) for value in article["catalog_ids"]),
                "event_kinds": ",".join(kinds),
                "title": title,
                "combined": combined,
                "channels": channels,
            }
        )
    reasons_by_code: dict[str, set[str]] = {}
    bases_by_code: dict[str, set[str]] = {}

    def mark(record: Mapping[str, Any], base: str, reason: str) -> None:
        code = str(record["article_code"])
        bases_by_code.setdefault(code, set()).add(base)
        reasons_by_code.setdefault(code, set()).add(f"{base}:{reason}")

    for record in records:
        for base in record["channels"]["direct_identity"]:
            mark(record, base, "direct_identity")

    temporal_symbols: set[str] = set()
    for base in sorted(set(scan_bases)):
        removals = [record for record in records if base in record["channels"]["removal"]]
        listings = [record for record in records if base in record["channels"]["listing"]]
        for removal in removals:
            later = [
                listing
                for listing in listings
                if int(listing["release_date_ms"]) > int(removal["release_date_ms"])
            ]
            if not later:
                continue
            temporal_symbols.add(base)
            mark(removal, base, "temporal_removal_t1")
            for listing in later:
                mark(listing, base, "temporal_listing_t2")
    for record in records:
        for base in set(record["channels"]["identity_support"]) & temporal_symbols:
            mark(record, base, "temporal_identity_context")

    candidates: list[dict[str, Any]] = []
    for record in records:
        code = str(record["article_code"])
        matched_bases = tuple(sorted(bases_by_code.get(code, ())))
        if not matched_bases:
            continue
        snippets = candidate_evidence_snippets(str(record["combined"]), matched_bases)
        candidates.append(
            {
                "release_date_utc": record["release_date_utc"],
                "article_code": code,
                "catalog_ids": record["catalog_ids"],
                "event_kinds": record["event_kinds"],
                "matched_bases": ",".join(matched_bases),
                "match_reasons": ",".join(sorted(reasons_by_code[code])),
                "title": record["title"],
                "evidence_snippets": json.dumps(
                    snippets, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )
    candidates.sort(key=lambda row: (row["release_date_utc"], row["article_code"]))
    return candidates, {
        "corpus_article_count": len(index),
        "systematic_window_article_count": systematic_article_count,
        "pattern_match_count": pattern_match_count,
        "selected_symbol_intersection_count": len(candidates),
        "direct_identity_article_count": sum(
            bool(record["channels"]["direct_identity"]) for record in records
        ),
        "temporal_symbol_count": len(temporal_symbols),
        "temporal_candidate_article_count": sum(
            any(reason.endswith(("temporal_removal_t1", "temporal_listing_t2")) for reason in reasons)
            for reasons in reasons_by_code.values()
        ),
    }


def _candidate_review_packet(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# EXP-101 — Announcement candidate review packet",
        "",
        "Deterministic code produced these candidates. Nothing below is an automatic",
        "asset classification or lifecycle segmentation.",
        "",
        f"Candidate count: `{len(rows)}`.",
        "",
    ]
    for row in rows:
        safe_title = str(row["title"]).replace("\n", " ").strip()
        lines.extend(
            [
                f"## {row['release_date_utc']} · `{row['article_code']}`",
                "",
                f"- Catalogs: `{row['catalog_ids']}`",
                f"- Event kinds: `{row['event_kinds']}`",
                f"- Selected bases: `{row['matched_bases']}`",
                f"- Match reasons: `{row['match_reasons']}`",
                f"- Title: {safe_title}",
            ]
        )
        for snippet in json.loads(str(row["evidence_snippets"])):
            lines.append(f"- Evidence: {snippet}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_announcement_candidate_parser(
    corpus_dir: Path,
    literal_preflight_dir: Path,
    selection_prepass_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise DeathCensusError("announcement candidate output already exists; refusing overwrite")
    acceptance = validate_identity_event_acceptance()
    corpus_manifest, index = _load_announcement_corpus_for_parser(corpus_dir)
    literal_bases, literal_summary_sha, literal_selected_sha = _load_selected_preflight_bases(
        literal_preflight_dir
    )
    (
        prepass_bases,
        known_bases,
        selection_prepass_summary_sha,
        prepass_selected_sha,
        known_bases_sha,
    ) = _load_selection_scan_prepass(selection_prepass_dir)
    scan_bases = tuple(sorted(set(literal_bases) | set(prepass_bases)))
    rows, counts = scan_announcement_candidate_rows(
        index, corpus_dir / "articles", scan_bases, known_bases
    )
    brake_triggered = len(rows) > IDENTITY_CANDIDATE_COST_BRAKE
    outcome = (
        "candidate_cost_brake_triggered"
        if brake_triggered
        else "candidate_surface_ready_for_model_review"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    candidate_path = output_dir / "announcement-candidates.csv"
    _write_csv(candidate_path, ANNOUNCEMENT_CANDIDATE_FIELDS, rows)
    artifacts = {candidate_path.name: file_sha256(candidate_path)}
    if not brake_triggered:
        packet_path = output_dir / "candidate-review-packet.md"
        _atomic_write_bytes(packet_path, _candidate_review_packet(rows).encode("utf-8"))
        artifacts[packet_path.name] = file_sha256(packet_path)
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "announcement_candidate_parser",
        "outcome": outcome,
        "task_sha256": TASK_SHA256,
        "corpus_manifest_sha256": corpus_manifest["manifest_sha256"],
        "literal_preflight_summary_sha256": literal_summary_sha,
        "literal_selected_symbols_sha256": literal_selected_sha,
        "selection_prepass_summary_sha256": selection_prepass_summary_sha,
        "prepass_selected_symbols_sha256": prepass_selected_sha,
        "known_base_symbols_sha256": known_bases_sha,
        "systematic_start_inclusive": SYSTEMATIC_ANNOUNCEMENT_START.isoformat(),
        "cutoff_end_exclusive": CUTOFF.isoformat(),
        "literal_base_count": len(literal_bases),
        "prepass_base_count": len(prepass_bases),
        "known_base_count": len(known_bases),
        "scan_base_count": len(scan_bases),
        "scan_universe_rule": "literal_preflight_union_stable_fiat_excluded_selection_prepass",
        "direct_identity_rule": "same_unit_two_known_tickers_verb_direction_or_redenomination",
        "direct_identity_direction_markers": ["to", "into", "for", "as", "→", "->"],
        "candidate_union_rule": "direct_identity_same_unit_union_temporal_removal_then_listing",
        **counts,
        "cost_brake_max_inclusive": IDENTITY_CANDIDATE_COST_BRAKE,
        "cost_brake_triggered": brake_triggered,
        "acceptance_fixture_count": len(acceptance),
        "acceptance_pass_count": sum(bool(row["passed"]) for row in acceptance),
        "model_called": False,
        "model_articles_read": 0,
        "parser_auto_segments_assets": False,
        "performance_or_death_metrics_computed": False,
        "protected_rows_read": False,
        "artifacts": artifacts,
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _atomic_write_bytes(output_dir / "parser-summary.json", _canonical_json_bytes(summary))
    return summary


def parse_catalog_metadata(payload: Any, expected_catalog_id: int) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        raise DeathCensusError("Binance catalog metadata payload is not an object")
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    catalogs = data.get("catalogs")
    if catalogs == []:
        return None
    if not isinstance(catalogs, list) or len(catalogs) != 1 or not isinstance(catalogs[0], dict):
        raise DeathCensusError("Binance catalog metadata schema mismatch")
    catalog = catalogs[0]
    catalog_id = int(catalog.get("catalogId"))
    if catalog_id != expected_catalog_id:
        raise DeathCensusError("Binance catalog metadata identity mismatch")
    name = str(catalog.get("catalogName") or "").strip()
    if not name:
        raise DeathCensusError("Binance catalog metadata name missing")
    parent = catalog.get("parentCatalogId")
    return {
        "catalog_id": catalog_id,
        "parent_catalog_id": "" if parent is None else int(parent),
        "catalog_name": name,
        "description": str(catalog.get("description") or "").strip(),
        "catalog_type": int(catalog.get("catalogType")),
        "article_total_current": int(catalog.get("total")),
    }


def fetch_catalog_metadata(catalog_id: int) -> dict[str, Any] | None:
    url = _url(
        ANNOUNCEMENT_API,
        {"type": 1, "catalogId": catalog_id, "pageNo": 1, "pageSize": 1},
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "exp-101-catalog-inventory/1"}
    )
    payload: Any | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code in {400, 404}:
                return None
            if exc.code not in {418, 429, 500, 502, 503, 504} or attempt == 2:
                raise DeathCensusError(
                    f"catalog metadata HTTP failure for id={catalog_id}: {exc.code}"
                ) from exc
            time.sleep(min(2**attempt, 4))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == 2:
                raise DeathCensusError(
                    f"catalog metadata network/schema failure for id={catalog_id}: {exc}"
                ) from exc
            time.sleep(min(2**attempt, 4))
    if payload is None:
        raise DeathCensusError(f"catalog metadata missing after retries: id={catalog_id}")
    return parse_catalog_metadata(payload, catalog_id)


def run_catalog_inventory(
    output_dir: Path,
    *,
    scan_start: int = CATALOG_SCAN_START,
    scan_end: int = CATALOG_SCAN_END,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if scan_start < 1 or scan_end < scan_start or scan_end > 1000:
        raise DeathCensusError("catalog scan range invalid")
    if workers < 1 or workers > 16:
        raise DeathCensusError("workers must be in [1, 16]")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_catalog_metadata, catalog_id): catalog_id
            for catalog_id in range(scan_start, scan_end + 1)
        }
        completed = 0
        for future in as_completed(futures):
            row = future.result()
            if row is not None:
                rows.append(row)
            completed += 1
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"catalog_inventory_progress={completed}/{len(futures)} discovered={len(rows)}",
                    flush=True,
                )
    rows.sort(key=lambda row: int(row["catalog_id"]))
    if not rows:
        raise DeathCensusError("Binance catalog inventory is empty")
    if DELIST_CATALOG_ID not in {int(row["catalog_id"]) for row in rows}:
        raise DeathCensusError("known delisting catalog absent from inventory")

    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "catalog-inventory.csv"
    _write_csv(inventory_path, CATALOG_INVENTORY_FIELDS, rows)
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "announcement_surface_inventory",
        "task_sha256": TASK_SHA256,
        "scan_start_inclusive": scan_start,
        "scan_end_inclusive": scan_end,
        "catalog_count": len(rows),
        "catalog_ids": [int(row["catalog_id"]) for row in rows],
        "inventory_sha256": file_sha256(inventory_path),
        "selection_basis": "catalog_name_and_description_only",
        "article_or_symbol_fields_persisted": False,
        "protected_rows_read": False,
        "performance_or_death_metrics_computed": False,
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _atomic_write_bytes(output_dir / "inventory-summary.json", _canonical_json_bytes(summary))
    return summary


def parse_full_asset_delist_title(title: str) -> tuple[tuple[str, ...], date] | None:
    match = re.fullmatch(
        r"Binance Will Delist (.+?) on (\d{4}-\d{2}-\d{2})",
        title.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    raw_assets, raw_date = match.groups()
    if any(word in raw_assets.upper() for word in ("MARGIN", "FUTURES", "OPTION")):
        return None
    normalized = re.sub(r"\s+(?:AND|&)\s+", ",", raw_assets, flags=re.IGNORECASE)
    assets = tuple(
        token.strip().upper()
        for token in normalized.split(",")
        if re.fullmatch(r"[A-Z0-9]+", token.strip().upper())
    )
    if not assets:
        return None
    return assets, date.fromisoformat(raw_date)


def _download_symbol(symbol: str, output: Path, fetcher: Callable[[str], Any]) -> dict[str, Any]:
    rows: list[list[Any]] = []
    start_ms = int(DOWNLOAD_START.timestamp() * 1000)
    end_ms = int(CUTOFF.timestamp() * 1000) - 1
    previous_last = -1
    while start_ms <= end_ms:
        payload = fetcher(
            _url(
                f"{MARKET_API}/api/v3/klines",
                {
                    "symbol": symbol,
                    "interval": "1d",
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": KLINE_LIMIT,
                },
            )
        )
        if isinstance(payload, dict) and "code" in payload:
            raise DeathCensusError(f"kline API rejected {symbol}: {payload}")
        if not isinstance(payload, list):
            raise DeathCensusError(f"kline payload is not a list: {symbol}")
        if not payload:
            break
        for raw in payload:
            if not isinstance(raw, list) or len(raw) < 11:
                raise DeathCensusError(f"kline schema mismatch: {symbol}")
            open_ms = int(raw[0])
            if open_ms >= int(CUTOFF.timestamp() * 1000):
                raise DeathCensusError(f"protected/post-cutoff kline returned: {symbol}")
            rows.append(
                [
                    datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).date().isoformat(),
                    raw[1],
                    raw[2],
                    raw[3],
                    raw[4],
                    raw[5],
                    raw[7],
                    raw[8],
                ]
            )
        last_ms = int(payload[-1][0])
        if last_ms <= previous_last:
            raise DeathCensusError(f"non-advancing kline pagination: {symbol}")
        previous_last = last_ms
        start_ms = last_ms + KLINE_INTERVAL_MS
        if len(payload) < KLINE_LIMIT:
            break
    unique: dict[str, list[Any]] = {}
    for row in rows:
        unique[str(row[0])] = row
    ordered = [unique[key] for key in sorted(unique)]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(KLINE_FIELDS)
        writer.writerows(ordered)
    temporary.replace(output)
    return {
        "symbol": symbol,
        "rows": len(ordered),
        "first": ordered[0][0] if ordered else None,
        "last": ordered[-1][0] if ordered else None,
        "sha256": file_sha256(output),
    }


def validate_symbol_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if tuple(frame.columns) != KLINE_FIELDS:
        raise DeathCensusError(f"daily file schema mismatch: {path.name}")
    if frame.empty:
        return frame.assign(date=pd.to_datetime(frame["date"], utc=True)).set_index("date")
    dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
    if dates.duplicated().any() or not dates.is_monotonic_increasing or (dates >= CUTOFF).any():
        raise DeathCensusError(f"daily date invariant failed: {path.name}")
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="raise").astype(int)
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise DeathCensusError(f"non-positive OHLC: {path.name}")
    if (frame[["volume", "quote_volume", "trade_count"]] < 0).any().any():
        raise DeathCensusError(f"negative activity field: {path.name}")
    frame["date"] = dates
    return frame.set_index("date")


def download_source(source_dir: Path, *, workers: int = DEFAULT_WORKERS) -> dict[str, Any]:
    if workers < 1 or workers > 16:
        raise DeathCensusError("workers must be in [1, 16]")
    source_dir.mkdir(parents=True, exist_ok=True)
    archive_symbols = list_archive_symbols()
    api_symbols = _api_usdt_symbols()
    usdt_symbols = tuple(sorted({symbol for symbol in archive_symbols if symbol.endswith("USDT")} | set(api_symbols)))
    if len(usdt_symbols) < 500:
        raise DeathCensusError("historical USDT symbol surface unexpectedly small")
    _atomic_write_bytes(source_dir / "archive-symbols.txt", ("\n".join(archive_symbols) + "\n").encode())
    _atomic_write_bytes(source_dir / "usdt-symbols.txt", ("\n".join(usdt_symbols) + "\n").encode())

    announcement_index = _announcement_index()
    _atomic_write_bytes(source_dir / "announcement-index.json", _canonical_json_bytes(announcement_index))
    details_dir = source_dir / "announcements"
    details_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(workers, 6)) as executor:
        futures = {
            executor.submit(_announcement_detail, row["code"]): row["code"]
            for row in announcement_index
            if not (details_dir / f"{row['code']}.json").is_file()
        }
        for future in as_completed(futures):
            detail = future.result()
            _atomic_write_bytes(details_dir / f"{detail['code']}.json", _canonical_json_bytes(detail))

    daily_dir = source_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    pending: list[str] = []
    for symbol in usdt_symbols:
        path = daily_dir / f"{symbol}.csv"
        if path.is_file():
            frame = validate_symbol_csv(path)
            completed.append(
                {
                    "symbol": symbol,
                    "rows": len(frame),
                    "first": frame.index.min().date().isoformat() if len(frame) else None,
                    "last": frame.index.max().date().isoformat() if len(frame) else None,
                    "sha256": file_sha256(path),
                }
            )
        else:
            pending.append(symbol)
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_download_symbol, symbol, daily_dir / f"{symbol}.csv", fetch_json): symbol
                for symbol in pending
            }
            done_count = 0
            for future in as_completed(futures):
                completed.append(future.result())
                done_count += 1
                if done_count % 50 == 0 or done_count == len(pending):
                    print(f"downloaded_symbols={done_count}/{len(pending)}", flush=True)

    completed.sort(key=lambda row: row["symbol"])
    files = {
        str(path.relative_to(source_dir)): file_sha256(path)
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.name != "source-manifest.json"
    }
    manifest: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "task_sha256": SOURCE_TASK_SHA256,
        "cutoff_end_exclusive": CUTOFF.isoformat(),
        "archive_symbol_count": len(archive_symbols),
        "usdt_symbol_count": len(usdt_symbols),
        "announcement_count_before_cutoff": len(announcement_index),
        "daily": completed,
        "files": files,
        "source_notes": {
            "market_data": "official Binance Spot REST market-data endpoint",
            "archive_index": "official data.binance.vision S3 index",
            "announcements": "official Binance frontend CMS endpoint; undocumented and snapshot-pinned",
            "post_cutoff_rows_persisted": False,
            "exchange_status_persisted_or_used": False,
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _atomic_write_bytes(source_dir / "source-manifest.json", _canonical_json_bytes(manifest))
    return manifest


def _load_source_manifest(source_dir: Path) -> dict[str, Any]:
    path = source_dir / "source-manifest.json"
    if not path.is_file():
        raise DeathCensusError("source manifest missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = payload.pop("manifest_sha256", None)
    if stored != canonical_json_sha256(payload):
        raise DeathCensusError("source manifest canonical checksum mismatch")
    payload["manifest_sha256"] = stored
    if payload.get("task_sha256") != SOURCE_TASK_SHA256 or payload.get("cutoff_end_exclusive") != CUTOFF.isoformat():
        raise DeathCensusError("source manifest binding mismatch")
    for relative, expected in payload.get("files", {}).items():
        candidate = source_dir / relative
        if not candidate.is_file() or file_sha256(candidate) != expected:
            raise DeathCensusError(f"source file checksum mismatch: {relative}")
    return payload


def _load_frames(source_dir: Path) -> dict[str, pd.DataFrame]:
    manifest = _load_source_manifest(source_dir)
    frames: dict[str, pd.DataFrame] = {}
    for row in manifest["daily"]:
        if int(row["rows"]) == 0:
            continue
        symbol = str(row["symbol"])
        frames[symbol] = validate_symbol_csv(source_dir / "daily" / f"{symbol}.csv")
    return frames


def _observed_return_count_before(frame: pd.DataFrame, selection_date: pd.Timestamp) -> int:
    dates = frame.index[frame.index < selection_date]
    if len(dates) < 2:
        return 0
    differences = dates.to_series().diff()
    return int(differences.eq(pd.Timedelta(days=1)).sum())


def select_at(
    frames: Mapping[str, pd.DataFrame],
    selection_date: pd.Timestamp,
    *,
    exclude_stable_fiat: bool = False,
) -> Selection:
    start = selection_date - pd.Timedelta(days=VOLUME_LOOKBACK_DAYS)
    candidates: list[dict[str, Any]] = []
    for symbol, frame in frames.items():
        if selection_date not in frame.index:
            continue
        observed_returns = _observed_return_count_before(frame, selection_date)
        if observed_returns < MIN_OBSERVED_RETURNS:
            continue
        recent = frame.loc[(frame.index >= start) & (frame.index < selection_date), "quote_volume"]
        if recent.empty:
            continue
        base = symbol[: -len("USDT")]
        if exclude_stable_fiat and base in STABLE_FIAT_BASES:
            continue
        candidates.append(
            {
                "symbol": symbol,
                "base_asset": base,
                "mean_quote_volume_180d": float(recent.mean()),
                "observed_returns_before_t": observed_returns,
                "leveraged_token": bool(LEVERAGED_BASE_RE.search(base)),
                "quote_like_asset": base in QUOTE_LIKE_BASES,
            }
        )
    candidates.sort(key=lambda row: (-row["mean_quote_volume_180d"], row["symbol"]))
    return Selection(selection_date, len(candidates), tuple(candidates[:TARGET_SIZE]))


def _selection_dates() -> tuple[pd.Timestamp, ...]:
    start = pd.Timestamp("2018-01-01T00:00:00Z")
    return tuple(pd.date_range(start, CUTOFF, inclusive="left", freq="MS"))


def _official_asset_delists(index: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    events: dict[str, dict[str, str]] = {}
    for row in index:
        parsed = parse_full_asset_delist_title(str(row["title"]))
        if parsed is None:
            continue
        assets, effective_date = parsed
        if pd.Timestamp(effective_date, tz="UTC") >= CUTOFF:
            continue
        for asset in assets:
            events[asset] = {
                "date": effective_date.isoformat(),
                "code": str(row["code"]),
                "title": str(row["title"]),
            }
    return events


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def selected_rows_from_available(
    selections: Sequence[Selection],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for selection in selections:
        if len(selection.rows) != TARGET_SIZE:
            continue
        for rank, row in enumerate(selection.rows, start=1):
            selected_counts[row["symbol"]] = selected_counts.get(row["symbol"], 0) + 1
            selected_rows.append(
                {
                    "selection_date": selection.selection_date.date().isoformat(),
                    "rank": rank,
                    **row,
                }
            )
    return selected_rows, selected_counts


def build_rename_successor_rows(
    frames: Mapping[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for predecessor, spec in RENAME_SUCCESSORS.items():
        successor = str(spec["successor"])
        predecessor_symbol = f"{predecessor}USDT"
        successor_symbol = f"{successor}USDT"
        if predecessor_symbol not in frames or successor_symbol not in frames:
            raise DeathCensusError(
                f"rename successor price surface missing: {predecessor}->{successor}"
            )
        end = pd.Timestamp(str(spec["predecessor_end"]), tz="UTC")
        predecessor_frame = frames[predecessor_symbol].loc[
            frames[predecessor_symbol].index <= end
        ]
        successor_frame = frames[successor_symbol]
        if predecessor_frame.empty or successor_frame.empty:
            raise DeathCensusError(
                f"rename successor price surface empty: {predecessor}->{successor}"
            )
        predecessor_last_date = predecessor_frame.index[-1]
        predecessor_last = float(predecessor_frame["close"].iloc[-1])
        predecessor_peak = float(predecessor_frame["close"].max())
        successor_first_date = successor_frame.index[0]
        successor_first = float(successor_frame["close"].iloc[0])
        multiplier = float(spec["multiplier"])
        adjusted_successor = successor_first * multiplier
        rows.append(
            {
                "predecessor": predecessor,
                "successor": successor,
                "multiplier": multiplier,
                "manager_status": str(spec["status"]),
                "predecessor_last_date": predecessor_last_date.date().isoformat(),
                "predecessor_last_close": predecessor_last,
                "predecessor_peak_close": predecessor_peak,
                "successor_first_date": successor_first_date.date().isoformat(),
                "successor_first_close": successor_first,
                "adjusted_successor_value_per_predecessor": adjusted_successor,
                "transition_continuity_ratio": adjusted_successor / predecessor_last,
                "successor_to_predecessor_peak_ratio": adjusted_successor / predecessor_peak,
                "unobserved_gap_days": max(
                    0, (successor_first_date - predecessor_last_date).days - 1
                ),
            }
        )
    return rows


def run_classification_audit(
    source_dir: Path,
    preflight_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest = _load_source_manifest(source_dir)
    frames = _load_frames(source_dir)
    preflight_summary_path = preflight_dir / "preflight-summary.json"
    if not preflight_summary_path.is_file():
        raise DeathCensusError("literal Phase-A preflight summary missing")
    preflight_summary = json.loads(preflight_summary_path.read_text(encoding="utf-8"))
    stored = preflight_summary.pop("summary_sha256", None)
    if stored != canonical_json_sha256(preflight_summary):
        raise DeathCensusError("literal Phase-A preflight checksum mismatch")
    preflight_summary["summary_sha256"] = stored
    if (
        preflight_summary.get("task_sha256") != SOURCE_TASK_SHA256
        or preflight_summary.get("source_manifest_sha256") != manifest["manifest_sha256"]
        or preflight_summary.get("m_or_r_exit_computed") is not False
    ):
        raise DeathCensusError("literal Phase-A preflight binding mismatch")

    selected_path = preflight_dir / "selected-symbols.csv"
    exits_path = preflight_dir / "exit-classification-candidates.csv"
    if (
        file_sha256(selected_path)
        != preflight_summary["artifacts"].get("selected-symbols.csv")
        or file_sha256(exits_path)
        != preflight_summary["artifacts"].get("exit-classification-candidates.csv")
    ):
        raise DeathCensusError("literal Phase-A artifact checksum mismatch")
    selected = pd.read_csv(selected_path)
    exits = pd.read_csv(exits_path).fillna("")
    if tuple(selected.columns) != SELECTED_FIELDS or tuple(exits.columns) != EXIT_CANDIDATE_FIELDS:
        raise DeathCensusError("literal Phase-A artifact schema mismatch")

    all_symbols = tuple(
        line.strip()
        for line in (source_dir / "usdt-symbols.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    stable_rows = []
    for base in sorted(STABLE_FIAT_BASES):
        matching = sorted(symbol for symbol in all_symbols if symbol == f"{base}USDT")
        stable_rows.append(
            {
                "base_asset": base,
                "symbols": " ".join(matching),
                "rule": "economic_design_pegged_to_USD_or_fiat",
            }
        )

    exit_classification = {
        str(row["base_asset"]): str(row["classification"])
        for row in exits.to_dict("records")
    }
    collapse_rows: list[dict[str, Any]] = []
    for base, event_date_raw in KNOWN_COLLAPSE_EVENTS.items():
        event_date = pd.Timestamp(event_date_raw)
        selected_dates = pd.to_datetime(
            selected.loc[selected["base_asset"].eq(base), "selection_date"]
        )
        if len(selected_dates) == 0:
            reason = "not_selected_in_literal_phase_a"
        elif base == "LUNA":
            reason = "old_and_new_assets_share_LUNAUSDT_source_symbol"
        elif base in exit_classification:
            reason = "present_in_exit_candidate_surface"
        else:
            reason = "source_symbol_has_rows_through_cutoff"
        collapse_rows.append(
            {
                "base_asset": base,
                "event_date": event_date.date().isoformat(),
                "selected_start_count": len(selected_dates),
                "selected_before_event": int((selected_dates < event_date).sum()),
                "selected_on_or_after_event": int((selected_dates >= event_date).sum()),
                "first_selected": selected_dates.min().date().isoformat() if len(selected_dates) else "",
                "last_selected": selected_dates.max().date().isoformat() if len(selected_dates) else "",
                "exit_candidate_classification": exit_classification.get(base, ""),
                "absence_reason": reason,
                "phase_b_blocker": base == "LUNA" and bool((selected_dates < event_date).any()),
            }
        )

    rename_rows = build_rename_successor_rows(frames)
    hidden_rename_rows = [
        row
        for row in rename_rows
        if row["manager_status"] == "hidden_economic_death"
        and row["successor_to_predecessor_peak_ratio"] < min((0.01, 0.05, 0.10, 0.25, 1.00))
    ]
    blocker = bool(hidden_rename_rows) or any(bool(row["phase_b_blocker"]) for row in collapse_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "stable-fiat-taxonomy.csv", STABLE_FIAT_FIELDS, stable_rows)
    _write_csv(output_dir / "known-collapse-audit.csv", COLLAPSE_AUDIT_FIELDS, collapse_rows)
    _write_csv(output_dir / "rename-successor-audit.csv", RENAME_AUDIT_FIELDS, rename_rows)
    artifacts = {
        path.name: file_sha256(path) for path in sorted(output_dir.glob("*.csv"))
    }
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "classification_audit",
        "task_sha256": TASK_SHA256,
        "source_task_sha256": SOURCE_TASK_SHA256,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "literal_preflight_summary_sha256": stored,
        "stable_fiat_rule": "economic_design_pegged_to_USD_or_fiat",
        "stable_fiat_frozen_base_count": len(STABLE_FIAT_BASES),
        "stable_fiat_present_symbol_count": sum(bool(row["symbols"]) for row in stable_rows),
        "known_collapse_count": len(collapse_rows),
        "rename_successor_count": len(rename_rows),
        "hidden_economic_death_count": len(hidden_rename_rows),
        "hidden_economic_death_symbols": [row["predecessor"] for row in hidden_rename_rows],
        "phase_b_blocked": blocker,
        "block_reason": "hidden_economic_death_requires_manager_review" if blocker else None,
        "m_hazard_or_exp094_conclusion_computed": False,
        "protected_rows_read": False,
        "artifacts": artifacts,
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _atomic_write_bytes(output_dir / "audit-summary.json", _canonical_json_bytes(summary))
    return summary


def run_preflight(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    manifest = _load_source_manifest(source_dir)
    frames = _load_frames(source_dir)
    index = json.loads((source_dir / "announcement-index.json").read_text(encoding="utf-8"))
    delists = _official_asset_delists(index)
    selections = [select_at(frames, stamp) for stamp in _selection_dates()]
    available = [selection for selection in selections if len(selection.rows) == TARGET_SIZE]
    first_available = available[0].selection_date if available else None

    preflight_rows: list[dict[str, Any]] = []
    selected_rows, selected_counts = selected_rows_from_available(selections)
    for selection in selections:
        symbols = [row["symbol"] for row in selection.rows]
        preflight_rows.append(
            {
                "selection_date": selection.selection_date.date().isoformat(),
                "pool_size": selection.pool_size,
                "selected_count": len(selection.rows),
                "selection_available": len(selection.rows) == TARGET_SIZE,
                "leveraged_count": sum(bool(row["leveraged_token"]) for row in selection.rows),
                "quote_like_count": sum(bool(row["quote_like_asset"]) for row in selection.rows),
                "selected_symbols": " ".join(symbols),
            }
        )

    exit_rows: list[dict[str, Any]] = []
    for symbol, count in sorted(selected_counts.items()):
        frame = frames[symbol]
        last = frame.index.max()
        if last >= CUTOFF - pd.Timedelta(days=1):
            continue
        base = symbol[: -len("USDT")]
        official = delists.get(base)
        exit_rows.append(
            {
                "symbol": symbol,
                "base_asset": base,
                "first_observation": frame.index.min().date().isoformat(),
                "last_observation": last.date().isoformat(),
                "selected_start_count": count,
                "official_full_delist_date": official["date"] if official else "",
                "official_announcement_code": official["code"] if official else "",
                "classification": "asset_exit" if official else "needs_classification",
            }
        )

    diagnostic_rows: list[dict[str, Any]] = []
    for stamp in (pd.Timestamp("2019-09-01T00:00:00Z"), pd.Timestamp("2020-03-02T00:00:00Z")):
        selection = select_at(frames, stamp)
        symbols = [row["symbol"] for row in selection.rows]
        bases = {row["base_asset"] for row in selection.rows}
        diagnostic_rows.append(
            {
                "selection_date": stamp.date().isoformat(),
                "pool_size": selection.pool_size,
                "selected_count": len(selection.rows),
                "overlap_with_current_20": len(bases & set(CURRENT_PRODUCT_BASES)),
                "selected_symbols": " ".join(symbols),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "selection-preflight.csv", PREFLIGHT_FIELDS, preflight_rows)
    _write_csv(output_dir / "selected-symbols.csv", SELECTED_FIELDS, selected_rows)
    _write_csv(output_dir / "exit-classification-candidates.csv", EXIT_CANDIDATE_FIELDS, exit_rows)
    _write_csv(output_dir / "required-date-diagnostics.csv", DIAGNOSTIC_FIELDS, diagnostic_rows)
    artifacts = {
        path.name: file_sha256(path)
        for path in sorted(output_dir.glob("*.csv"))
    }
    summary: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "classification_preflight",
        "task_sha256": TASK_SHA256,
        "cutoff_end_exclusive": CUTOFF.isoformat(),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "first_available_selection": first_available.isoformat() if first_available is not None else None,
        "monthly_selection_count": len(selections),
        "available_monthly_selection_count": len(available),
        "selected_unique_symbol_count": len(selected_counts),
        "exit_candidate_count": len(exit_rows),
        "needs_classification_count": sum(row["classification"] == "needs_classification" for row in exit_rows),
        "leveraged_selected_any": any(bool(row["leveraged_token"]) for row in selected_rows),
        "quote_like_selected_any": any(bool(row["quote_like_asset"]) for row in selected_rows),
        "m_or_r_exit_computed": False,
        "protected_rows_read": False,
        "ready_for_final_census": False,
        "artifacts": artifacts,
    }
    summary["summary_sha256"] = canonical_json_sha256(summary)
    _atomic_write_bytes(output_dir / "preflight-summary.json", _canonical_json_bytes(summary))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXP-101 death-census source and preflight")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--source-dir", type=Path, required=True)
    download.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--source-dir", type=Path, required=True)
    preflight.add_argument("--output-dir", type=Path, required=True)
    scan_prepass = subparsers.add_parser("selection-scan-prepass")
    scan_prepass.add_argument("--source-dir", type=Path, required=True)
    scan_prepass.add_argument("--output-dir", type=Path, required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--source-dir", type=Path, required=True)
    audit.add_argument("--preflight-dir", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)
    inventory = subparsers.add_parser("catalog-inventory")
    inventory.add_argument("--output-dir", type=Path, required=True)
    inventory.add_argument("--scan-start", type=int, default=CATALOG_SCAN_START)
    inventory.add_argument("--scan-end", type=int, default=CATALOG_SCAN_END)
    inventory.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    corpus = subparsers.add_parser("announcement-corpus")
    corpus.add_argument("--output-dir", type=Path, required=True)
    corpus.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    candidates = subparsers.add_parser("announcement-candidates")
    candidates.add_argument("--corpus-dir", type=Path, required=True)
    candidates.add_argument("--literal-preflight-dir", type=Path, required=True)
    candidates.add_argument("--selection-prepass-dir", type=Path, required=True)
    candidates.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "download":
        summary = download_source(args.source_dir, workers=args.workers)
        printable = {
            key: value
            for key, value in summary.items()
            if key not in {"daily", "files"}
        }
        printable["daily_file_count"] = len(summary["daily"])
        printable["sha_bound_file_count"] = len(summary["files"])
    elif args.command == "preflight":
        summary = run_preflight(args.source_dir, args.output_dir)
        printable = summary
    elif args.command == "selection-scan-prepass":
        summary = run_selection_scan_prepass(args.source_dir, args.output_dir)
        printable = summary
    elif args.command == "audit":
        summary = run_classification_audit(
            args.source_dir,
            args.preflight_dir,
            args.output_dir,
        )
        printable = summary
    elif args.command == "catalog-inventory":
        summary = run_catalog_inventory(
            args.output_dir,
            scan_start=args.scan_start,
            scan_end=args.scan_end,
            workers=args.workers,
        )
        printable = summary
    elif args.command == "announcement-corpus":
        summary = run_announcement_corpus(args.output_dir, workers=args.workers)
        printable = {
            key: value for key, value in summary.items() if key != "files"
        }
        printable["sha_bound_file_count"] = len(summary["files"])
    else:
        summary = run_announcement_candidate_parser(
            args.corpus_dir,
            args.literal_preflight_dir,
            args.selection_prepass_dir,
            args.output_dir,
        )
        printable = summary
    print(json.dumps(printable, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
