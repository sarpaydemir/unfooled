# unfooled

A crypto research pipeline built not to fool itself: frozen splits, leakage gates, calibrated LLM judges and evidence-checked LLM extraction.

This is the method layer of a private research project (July–August 2026). It searched Binance data for a trading edge, then for a crypto portfolio product that could reach a return target of at least about 40% a year. In 120 experiments it found nothing that beat holding the market. The code here is what made that answer worth trusting: the gates every idea had to pass, and the checks on the LLMs that wrote and read things for it.

## Results

- **The LLM judge failed its own exam, so it got no vote.** On a frozen 20-case suite pinned by SHA-256 (10 correct, 10 with one material mutation), the reviewer scored 15/20: 3 false rejects, 2 false passes, for $0.064. The bar was 20/20. It got no authority to pass a candidate, the suite was never reused to tune its prompt, and the error class it missed (TA-Lib `AROON` output order) became a deterministic check. [EXP-074](docs/exp-074-llm-judge-calibration.md)
- **The event gate was too weak to see a small edge, and that was the result.** On 1,000 random-entry sets it passed 0.1% of null sets but only 1.5% of sets with a +0.5%/48h edge injected. 80% power needed 7,157–11,816 independent clusters; the open data had 877. A matched-null version caught a +1.5% edge from 550 events up, but at 300 events its Wilson lower bound was 74.18% against a 75% bar, so no new gate was adopted and the bar didn't move. [EXP-076/077](docs/exp-076-077-gate-power-calibration.md)
- **LLM extraction had to quote its source.** From a 5,971-article announcement corpus (SHA-256 manifest, 6/6 known events found), the first paid chunk returned 51 of 80 evidence quotes that weren't in the source text, and was rejected whole. A recall audit caught the parser dropping LUNA→LUNC because "as" wasn't a direction marker. The final run classified 159/159 articles with 125/125 quotes verified after normalization, for $0.54 in total. [EXP-101](docs/exp-101-announcement-extraction.md)
- **The best headline was survivorship bias.** A 64.28% XIRR on a basket of 20 coins came from coins picked knowing they had survived, over a window that skipped the 2018 crash. Same-window replays with a causal basket were 31–44% lower. From 2018, the corrected figure was +17.11% a year with a worst 365 days of −84%: below the target, and reported as "not reached". [Survivorship correction](docs/survivorship-correction.md)

## Method rules

1. **Splits are frozen before any return is read.** Warm-up 2022-02-01 → 2023-07-29, training → 2025-01-29, validation → 2025-07-29, locked holdout → 2026-07-30. 20 development pairs and 15 blind pairs were chosen from metadata, liquidity and spread, never from returns (the [survivorship correction](docs/survivorship-correction.md) shows what that rule still missed). The code refuses research windows that end after 2025-01-29 (`event_study.FINAL_CLOSED_BOUND`), and the announcement corpus stops at the same cutoff (`death_census.CUTOFF`).
2. **The alpha budget is fixed before data.** A campaign's familywise alpha and hypothesis count are written once and can't change (`storage.ResearchStore.ensure_alpha_campaign`): 150 earlier hypotheses counted as debt plus at most 30 new ones gives a Bonferroni threshold of 0.05/180 = 0.000278. A candidate takes a slot when its spec review passes, before any performance test, and a failed candidate's slot is never returned. `experiment_alpha.py` keeps the same kind of ledger for experiments outside the candidate pipeline.
3. **Lookahead is tested, not trusted.** Each candidate runs on 3 synthetic OHLCV series of 1,200 hourly bars, on the full frame and on prefixes cut at 384, 576, 768 and 960 bars; any difference means it used the future (`causality.py`). A static AST gate rejects negative shifts, manual dataframe merges, undeclared columns, imports outside an allowlist, and file, network, shell, `eval`/`exec` or dunder access (`safety.py`). A TA-Lib output contract catches swapped multi-output bindings (`indicator_contract.py`). Gates run in a fixed order with thresholds frozen in code (`gates.py`, `workflow.REQUIRED_PROMOTION_GATES`).
4. **LLM judges are calibrated on a frozen suite and never tuned on it.** The reviewer must come from a different model family than the generator, runs at temperature 0, and is scored on the SHA-pinned suite under a cost cap (`reviewer_benchmark_cases.py`, `reviewer_calibration.py`). A negative review only puts a candidate on hold; rejecting or releasing it takes a machine-checked artifact bound to the code and review hashes (`workflow.py`).
5. **LLM output must quote verbatim evidence.** Every extracted event carries a sentence that must be found in its own article, its tickers must appear in that article, and six known events ride along in every call as controls. Requests are hashed before sending, responses are saved before validation, failed responses are never repaired, and a replay makes no new provider call (`death_census_classifier.py`, `death_census_batch.py`).

## Run the tests

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13 or later (uv installs it).

```sh
uv sync
uv run pytest        # 170 tests: no network, no market data, no API key
uv run ruff check .
```

TA-Lib installs from the `ta-lib` 0.6.8 wheel, the exact version the output contract pins. The module entry points (`python -m research_factory.causality`, `event_study`, `event_gate_calibration`, `matched_null_calibration`, `reviewer_calibration`, `death_census*`) need inputs that aren't published: candidate files, Binance OHLCV data, the announcement corpus, or an OpenRouter key.

## Layout

| Area | Modules in `research_factory/` |
|---|---|
| Candidates and state | `models`, `validation`, `schemas`, `storage`, `policy` |
| Leakage gates | `safety`, `indicator_contract`, `talib_metadata` (+ `talib_multi_output_v1.json`), `static_validation`, `causality`, `gates`, `workflow`, `miner` |
| Statistics | `experiment_alpha`, `event_study`, `event_gate_calibration`, `matched_null_calibration` |
| LLM judge | `providers`, `reviewer_benchmark_cases`, `reviewer_calibration` |
| LLM extraction | `death_census`, `death_census_flatten`, `death_census_classifier`, `death_census_batch`, `death_census_recall_audit`, `hashing` |

`tests/` holds the unit tests and the two EXP-075 validator fixtures. `docs/` holds the four write-ups, translated or condensed from the original Turkish and English notes, each naming its source file.

## Not included, on purpose

- **Strategies and generated candidates.** The Freqtrade strategy files import Freqtrade (GPL-3.0), and the generated candidates derive from mechanisms mined from a GPL-3.0 strategy. Neither is here. `miner.py` is included only because `workflow.py` imports it; it parses a pinned upstream file at runtime and contains none of it.
- **Data.** No OHLCV, no announcement corpus or article text, no model responses, no research database. The write-ups publish hashes and summaries instead.
- **The rest of the project.** The experiment scripts and write-ups (EXP-000 to EXP-122), the portfolio and risk-overlay research, and the pipeline's CLI and scheduler stay private. Modules that depended on any of that were left out rather than trimmed.

The code is copied from the original repository without its history. Changes: `hashing.py` holds a four-line helper that two modules imported from an excluded one; test fixture paths are now relative to the test files; three unused names were removed to pass lint.

## How it was built

I built this by directing AI coding agents under written rules, as in [hunkrisk](https://github.com/sarpaydemir/hunkrisk). I set the target and made the major route calls: which lines of research to open, and which to close. A research-manager agent owned the task files, and for every hypothesis test the decision rules and pass gates were written before the run. The implementing agent worked under a separate contract and could not change scope or a gate without my approval. From 2026-08-08, every task was implemented twice in parallel, by Claude Opus and Claude Sonnet, blind to each other: matching results meant the task was clear, and diverging results meant it was ambiguous, which was reported as a finding.

Some rules were loosened over time to move faster. Four never were: the sealed research state, the alpha ledger, verifiable evidence, and no thresholds set after seeing results.

## License

MIT.
