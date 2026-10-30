# Survivorship correction: from a 64% headline to 17% a year

Sources, all in the private research repo:

- `docs/HANDOFF.md` (Turkish), sections on EXP-088 and EXP-089–093: the original headline.
- `research/experiments/EXP-104-PRODUCT-CORRECTION-RESULT.md` (English): the survivorship measurement.
- The research manager's notes, `CLAUDE.md` (Turkish, not published): §7e, the handover note dated 2026-08-12, for the corrected numbers and the verdict against the target; §3 items 37 and 39 for the rules that followed.
- `docs/PAIR_UNIVERSE.md` (Turkish): how the 20 coins were chosen.

Numbers are copied, not recomputed. None of this code is in this repo: the portfolio replays depend on modules and data that are left out.

## The headline

After the signal research found nothing it could certify (see [EXP-076/077](exp-076-077-gate-power-calibration.md)), the work moved to portfolio products funded by fixed monthly contributions. EXP-088 compared a weekly trend system with a passive basket of the 20 frozen coins ("Benchmark A"). The passive basket ended at `336.402/65 = 5.1754x` its 65 contributions (`2019-09-02` to `2025-01-06`): an **XIRR of `64.28%`**. EXP-092 replayed the same path for the frozen 20 ("S3") and got the same `5.1754x` and `64.28%`.

That is well above the target of at least about 40% a year.

## What was wrong with it

Two things, both named in the handover note:

1. **The 20 coins were picked knowing which ones survived.** The universe was frozen on `2026-07-29` from Binance's *currently active* markets, data coverage from `2023-06-17`, current quote volume and bid/ask spread. It deliberately used no returns. But "listed and liquid in 2026" is itself a survivor filter: a coin that died in 2022 can't be in it. EXP-092's own write-up flagged that S3 was a 2026 survivor pick and that comparisons in its favour were biased upward.
2. **The window skipped the 2018 crash.** The headline started in September 2019.

## Measuring the survivorship part (EXP-104)

EXP-104 replaced today's 20 with a **causal** top-20, chosen at the start date from data available then, and compared the two on the same calendar with the same daily engine:

| start | basket | buy and forget | monthly equal rebalance |
|---|---|---:|---:|
| `2019-09-01` | causal top-20 | `3.457x` | `4.578x` |
| `2019-09-01` | today's 20 | `5.028x` | `7.406x` |
| `2020-03-02` | causal top-20 | `2.630x` | `3.129x` |
| `2020-03-02` | today's 20 | `4.061x` | `5.617x` |

Relative to today's winners on the same dates, the causal basket was lower by `31.24%` / `38.18%` (2019 start, buy-and-forget / rebalanced) and by `35.25%` / `44.30%` (2020 start). Rebalancing still helped inside both honest baskets, but neither the current-20 return nor the rebalancing bonus is a defensible product number on its own. The worst trailing 365-day returns of the causal arms were `-70.81%` to `-81.45%`.

## The corrected numbers

From the handover note: a causal (survivorship-free) basket, `2018-03-05` to `2025-01-29`, `6.9` years, fees included.

| | money multiple | per year | worst 365 days |
|---|---:|---:|---:|
| pure crypto basket | `1.823x` | `+17.11%` | `-84.00%` |
| S&P 500, for comparison (approximate in the source) | `~2.2x` | `~+13%` | `~-25%` |

The note's verdict: the target of at least about 40% a year was **not reached, and no path to it was found**. Across `120` experiments, nothing that beat holding the market was found.

The two figures are not the same measurement: `64.28%` is an XIRR over 2019–2025 on today's 20, and the note labels `17.11%` only as "annual" for the causal basket over 2018–2025, without stating the annualization method. Both are reported exactly as written.

## What changed in the method

- **No product number alone** (§3 item 37, binding after EXP-104): every product number is presented with its downside tail and the survivorship-corrected base multiple. The earlier "+34% from monthly rebalancing" line was retired.
- **Survivorship inflates the bonus too, not just the level** (§3 item 39): on the same window, rebalancing added `+47.28%` on today's 20 but `+32.42%` on the causal basket, because diverging winners give rebalancing more to harvest. Correcting the multiple is not enough; the basket itself has to be causal.
- A universe rule that reads no returns is still not survivorship-free if it selects from what is listed today.
- The corrected number was reported against the target, including "not reached".
