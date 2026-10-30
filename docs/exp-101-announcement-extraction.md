# EXP-101: an announcement corpus and LLM extraction that must quote its evidence

Condensed from the EXP-101 files in `research/experiments/` of the private research repo: the manifest `EXP-101-DEATH-CENSUS.md` and the result files `EXP-101-ANNOUNCEMENT-CORPUS-RESULT.md`, `EXP-101-KNOWN-EVENT-RECALL-AUDIT-RESULT.md`, `EXP-101-D3E-AS-EVENT-RECALL-RESULT.md`, `EXP-101-PAID-IDENTITY-CLASSIFICATION-RESULT.md`, `EXP-101-FLATTENED-CALIBRATION-RESULT.md`, `EXP-101-FIVE-ARTICLE-BATCH-STOP.md`, `EXP-101-SCHEMA-V3-BATCH-STOP.md`, `EXP-101-SCHEMA-V4-CONTROL-STOP.md`, `EXP-101-SCHEMA-V4-CLASSIFICATION-RESULT.md` and `EXP-101-DEATH-CENSUS-RESULT.md`. Those are already in English; numbers are copied, not recomputed. The source file is named in each section.

Code in this repo: `research_factory/death_census.py` (corpus and kline download with SHA-256 manifests, parser, recall controls), `research_factory/death_census_flatten.py` (rich-text bodies to readable text, its own manifest), `research_factory/death_census_classifier.py` (the paid call and every deterministic check), `research_factory/death_census_batch.py` (chunked runs, retries, aggregation), `research_factory/death_census_recall_audit.py` (known-event recall). The corpus itself, the model responses and the usage ledger are not included.

## Why

The portfolio research needed a base rate: how often does a coin in a top-20 basket officially die (delist) or change identity (rename, token swap, redenomination)? Binance's announcements are the primary source. An LLM can read them, but an LLM can also invent a sentence that sounds like the announcement. So every extracted event has to carry a sentence that is found, verbatim, in that article, and code checks it.

The experiment has two fail-closed phases. Phase A may only read public metadata, announcements and daily spot klines before the cutoff. Phase B (death rates) stays locked until every exit is classified and every rename is listed by name, so a classification can't be chosen after its loss is seen (manifest, "Status and authority").

## The corpus

Source: `EXP-101-ANNOUNCEMENT-CORPUS-RESULT.md`.

```text
runtime             279m 19s
included catalogs   48, 49, 51, 128, 157, 161
excluded catalogs   50, 93
article index       5,971 pre-cutoff rows
article details     5,971 / 5,971
acceptance          6 / 6 known identity events
bound files         5,980
post-cutoff rows    0
model calls         0
```

The canonical manifest SHA-256 is `656d381ec7a816a82f9a993e0ac9922cd806d2d239a75aefcf0947460b3f9c52`. Every file SHA, every article-code/filename pair, the `2025-01-29` end-exclusive cutoff and the manifest hash were replayed independently after the run. First stored release `2017-07-02T06:03:12Z`, last `2025-01-28T02:00:02.571Z`.

## A recall audit before any paid call

Sources: `EXP-101-KNOWN-EVENT-RECALL-AUDIT-RESULT.md`, `EXP-101-D3E-AS-EVENT-RECALL-RESULT.md`.

A deterministic parser narrows the corpus to candidate articles. Before paying for any model call, six known real events were traced through it:

| event | result |
|---|---|
| `MATIC_TO_POL`, `FTM_TO_S` | covered |
| `BTT_REDENOMINATION`, `PAX_TO_USDP`, `BCHABC_TO_BCH` | outside the systematic window (official article before `2022-02-01`) |
| `LUNA_TO_LUNC` | **blocked** |

The LUNA article (*"…LUNA (old) Renamed as LUNC"*) was in the broad `869`-article replay but not in the narrowed `154`. The frozen direction markers were `to`, `into`, `for`, `->` and `→`; `as` was missing. The paid call was not made. A free rerun added `as`: the narrowed set went from `154` to `159`, in-window known-event recall to `3/3`, and nothing else in the filter changed.

## First paid call: rejected whole

Source: `EXP-101-PAID-IDENTITY-CLASSIFICATION-RESULT.md`.

The 159 articles were split into two chunks (`80 + 79`), each with the six known events repeated as acceptance controls. The model was `z-ai/glm-5.2` at temperature `0`, with a strict JSON schema.

```text
chunk-1 prompt tokens       539,155
chunk-1 completion tokens     9,513
chunk-1 cost USD           0.1465206
classification rows               80
non-verbatim evidence rows         51
acceptance controls valid      5 / 6
```

51 of 80 evidence quotes were not in the source text, and not because of whitespace or HTML: many were not contiguous in the article at all. One control came back as `LUNA_TO_LUNA` instead of `LUNA_TO_LUNC`. The whole chunk was rejected; nothing was repaired or kept as a partial result, and chunk 2 was never sent. The `$0.1465206` stays in the append-only usage ledger.

## Calibration rounds

Sources: manifest section "Invalid first response and frozen calibration route", `EXP-101-FLATTENED-CALIBRATION-RESULT.md`.

The rich-text JSON bodies were flattened to readable text by one deterministic transformation, as a separate SHA-bound artifact (the original corpus is unchanged). Evidence comparison may normalize only repeated whitespace, NBSP, and Unicode quotes and dashes; raw and normalized failures are both reported. At most two calibration calls were allowed.

- **Round 1, 10 articles:** `finish_reason=length`, `16000` completion tokens of which `15878` were reasoning, no JSON. Cost `$0.04468144`. Replaying the same call hash used the saved response and made no second provider call.
- **Round 2, 5 articles:** raw and normalized evidence `5/5`, controls `6/6`, cost `$0.00584620`. All five rows still failed a semantic check: each said `is_identity_event=false` but named a positive event type.

The result file reports a conflict between the manager's written gate (normalized evidence `>= 95%` and `6/6` controls) and the implementation's extra zero-semantic-error condition; the manifest's calibration block lists all three. The manager ruled the semantic condition diagnostic. That diagnostic found a real schema gap (an article can describe a genuine identity event for a coin outside the scan set). The round-2 response was not repaired or used as data: every article was classified again under the later schemas.

## Three schema stops, each caused by asking the model for something code can compute

Sources: `EXP-101-FIVE-ARTICLE-BATCH-STOP.md`, `EXP-101-SCHEMA-V3-BATCH-STOP.md`, `EXP-101-SCHEMA-V4-CONTROL-STOP.md`.

- **Schema v2** asked for `in_scan_set` and `is_identity_event = real_identity_event AND in_scan_set`. At offset 10, three rows reported genuine out-of-scan swaps (`RAMP -> LEVER`, `MC -> BEAMX`, `TVK -> VANRY`) as `is_identity_event=true` with an unrelated symbol. The run stopped. Fix: the model returns primitives only; code computes the conjunction.
- **Schema v3** still asked for `real_identity_event` *and* an `event_type` that implies it. At offset 20, three rows (a native USDC integration, two Binance Convert additions) said `real_identity_event=false` with `event_type=relisting_different_asset`. The run stopped. Fix: one `events[]` array per article, `event_type` as the single decision, every boolean derived in code.
- **Schema v4** passed 18 chunks (90 articles), then failed the `6/6` control gate twice at offset 90: the model classified the `LUNA_TO_LUNC` control correctly but copied its evidence from a real LUNA article in the same batch, not from the control text. Counting that as `6/6` after seeing it would have relaxed a frozen gate. The manager ruled it a fixture-source collision: control evidence may come from any source bound into that call, production evidence only from its own article. Offset 90 was called again under a new hash.

## Schema v4: complete

Source: `EXP-101-SCHEMA-V4-CLASSIFICATION-RESULT.md`.

```text
articles                         159 / 159
accepted chunks                         32
classified events                      125
identity events in scan set             15
asset-exit signals                      77
out-of-scan identity articles           31
raw evidence                     115 / 125
normalized evidence              125 / 125
normalization rescues                   10
accepted-run cost USD          0.208997392
EXP-101 cumulative AI cost USD 0.541434292
```

One attempt hit the fixed `16,000` completion budget; it stays in the ledger, and the automatic retry at `32,000` passed. The 15 identity rows are announcement observations, not 15 deaths; identity segmentation deduplicates and orders them before any lifecycle replay.

## How the classifier is held to its evidence

All of this is in `research_factory/death_census_classifier.py` and covered by `tests/test_death_census_classifier.py`:

- Every article ID must appear exactly once; every event's `evidence_quote` must be a substring of that article's readable text (raw, then normalized); `subject_symbol` and `successor_ticker` must occur in the article as whole tokens; duplicate events fail; an article with events can't also give a no-event reason.
- All six known-event controls must come back identity-positive with verbatim evidence.
- The request is hashed (`model + prompt + canonical schema`) before it is sent. The raw response is written to disk before validation. A saved response with the same hash is replayed locally and makes no new provider call or ledger row.
- Only timeouts, connection errors and HTTP `408/425/429/500/502/503/504` may use the one retry. HTTP error bodies are saved with the API key redacted.
- Usage is appended once per call to a separate append-only ledger. A failed response is never edited.

## What it measured

Source: `EXP-101-DEATH-CENSUS-RESULT.md`.

For monthly 20-coin baskets, the mean one-year official death fraction was `h_1y = 0.6481%`. The unique-asset estimator gave `0.3474%` to `1.4004%` a year depending on the value-death threshold. The classified exits were LUNA (old), OMG, WAVES and XMR, clustered in 2022 and 2024.

The more useful finding: six selected coins (`AXS EOS FIL FTT SLP WIN`) stayed listed through the cutoff after falling at least 95% from a prior peak and never recovering above 10% of it. In the result's words: official death appears rare, but "still listed and nearly worthless" is not rare enough to ignore.
