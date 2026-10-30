# EXP-076 and EXP-077: can the event gate see a real edge?

Translated from the Turkish write-ups `research/experiments/EXP-076-EVENT-GATE-CALIBRATION.md` and `research/experiments/EXP-077-MATCHED-NULL-CALIBRATION.md` in the private research repo, including the independent-review note at the end of EXP-077; the matcher's label is from `docs/RESEARCH_FACTORY.md` in the same repo. Numbers are copied, not recomputed.

Code in this repo: `research_factory/event_study.py` (the gate's metrics), `research_factory/gates.py` (the frozen thresholds), `research_factory/event_gate_calibration.py` (EXP-076), `research_factory/matched_null_calibration.py` (EXP-077). Running them needs the Binance OHLCV files, which are not included; the unit tests use synthetic frames.

Neither experiment produces a strategy or a candidate. They measure the sieve.

## EXP-076: event-study gate and power calibration

### Question

How often does the existing event-study gate accept random long entries, and how often does it wrongly reject entries that carry a small but real edge?

### Frozen design

- Data: the 20 development pairs, range `[2019-01-01, 2025-01-29)`, open development data only.
- Fee: `0.001` on entry and on exit.
- Seed `76076`; `1000` Monte Carlo sets in total: densities of `300`, `550`, `900` and `1300` events, `250` sets each.
- Selected events on the same pair are at least `48h` apart.
- If the 49 timestamps on the signal–entry–forward path are not complete and contiguous hours, the event is dropped. A row offset is never taken as an hour.
- Cross-pair effective events are counted in `12h` UTC market clusters.
- Entry is the open of the `1h` candle after the signal candle. Horizon `48h`, scored with the existing `event_study.summarize()` metrics.

Each random set is scored twice:

1. **Null:** the real price path with random entry timing.
2. **Injected edge:** the same path and the same entries, with forward prices scaled by a factor that rises linearly from `0%` at entry to `+0.5%` at hour 48. Close, high and low get the same factor; fees don't change. Only a timing edge is added; the sample and the market path stay fixed.

### Gate conditions measured

`effective_events >= 60`, `mean_return_pct > 0`, `median_return_pct > 0`, `positive_pair_fraction >= 2/3`, `positive_half_fraction == 1.0`, `mfe_mae_ratio >= 1.20`, and all six together.

For each density and overall, the report gives the null pass rate, the injected-edge pass rate, the false-reject rate and metric percentiles. Any suitability call for a future campaign uses the **worst** density's false-reject rate, not the average, so a low density can't hide behind a high one. The MFE/MAE null 99% check also uses the highest value across densities.

### Power calculation

To avoid counting overlapping 48h outcomes as independent, the 12h UTC clusters are split into five phases by absolute bin number `mod 5`. Bin starts in the same phase are `60h` apart, and the nearest events inside them at least `49h`. The largest standard deviation across the five phases, and its 90th percentile across Monte Carlo sets, is the conservative input. For a two-sided test at 80% power, the report gives the clusters needed for a `+0.5% / 48h` edge at the current campaign's `alpha = 0.05 / 180` (and, for reference only, `0.05 / 30`), and the most independent clusters one purged phase can hold in the open range.

### Decision rules, written before the run

- The current campaign's thresholds and its six alpha slots do not change. No rejected candidate reopens; no new candidate or alpha slot.
- `effective_events` stays as the sample gate and positive mean as the economic-direction condition; neither is loosened here.
- Any of median, pair spread, temporal halves or MFE/MAE that rejects the injected `+0.5%` edge in more than 20% of sets at any density is unfit to be a hard gate in the next campaign.
- If the null 99th percentile of `mfe_mae_ratio` is `>= 1.20` at any density, that metric doesn't discriminate and must be replaced.
- If 80% power needs more clusters than the open training range can hold, the next campaign's alpha rule can't be chosen until that capacity problem is solved.
- Results may only shape the next campaign's pre-registered design.

### Result

Official run completed on 2026-07-30.

- `106/106` tests, compile and shell checks passed.
- Exactly `1000` sets, `250` per density. The event pool held `838739` contiguous windows.
- Schedule replay SHA-256 `a03c8f843d25e5c876d17d1490af9f894aa0f8d06b32a82c5c8d4696cb7b4b96`.
- **Null sets passed the combined gate `0.1%` of the time; `+0.5%` edge sets passed `1.5%` of the time.**
- Positive median's worst-density false-reject rate was `18.4%`, inside the pre-written `20%` limit.
- MFE/MAE (`100%`), positive-pair fraction (`81.2%`) and two positive temporal halves (`46.0%`) had worst-density false-reject rates above it: in their current form, none of the three is fit to be a hard gate.
- The null mean was positive in `76.4%` of sets, so `mean > 0` is an economic direction (beta) condition, not evidence of skill.
- **The open training range holds `877` purged clusters; 80% power for a `+0.5% / 48h` edge needs `7157–11816`.**
- Research DB SHA `17a6dbebe8bb1f0030ee15af990d3b4725f43af1cbcb9702a6cb655ec44ea809`, state `18 candidate / 6 alpha` and monthly AI total `$0.6388` unchanged.

**Decision:** every technical gate passed; EXP-076 **passed**. The current campaign and old rejections stand. Next: a calibration that separates market beta drift from skill with a matched-null lift, and measures `+0.5%` and `+1.5%` arms.

## EXP-077: matched-null calibration

### Question

Random long entries look positive partly because the market drifted up. Can a matched control (same pair, similar conditions before entry) remove that drift, and what are the Type-I and Type-II rates of a matched-lift sieve on separate calibration and evaluation sets?

### Frozen data and arms

- The same 20 development pairs, `[2019-01-01, 2025-01-29)`, fee `0.001` each way.
- Densities `300`, `550`, `900`, `1300`; `1000` sets per density: repetitions `0–499` only calibrate the null threshold, `500–999` are an independent evaluation.
- Seed `77077`. Expected schedule SHA-256 `fd3a4e00ac0fbd6593e6a63cb377769c78b7d105324765025b4d09a2a4986cad`, control-map SHA-256 `6d3bf592db2407dd0f27dc3103a0aebe07e2048c60b09368a57db902eada156b`.
- Arms: real null, linear `+0.5% / 48h`, linear `+1.5% / 48h`. All arms share event IDs. The synthetic edge is applied only to the event's 48h exit price; control paths stay real null.

### Frozen matcher

For each event, exactly 20 controls from the same pair:

1. Control entries come from `3, 4, …, 62` calendar days before the event, so event and control 48h outcome paths don't overlap, the UTC hour is the same, and control selection can't depend on the event's future. Controls may overlap each other; they aren't counted as independent, and the false-positive rate is measured directly on a held-out empirical null.
2. The control's own signal–entry–48h path must be complete and contiguous.
3. Using only the 168 hours closed before entry, for both event and control: pair 7-day log return, pair 7-day realized volatility, BTC 7-day log return, BTC 7-day realized volatility. Gaps of up to 3 hours in the pre-entry series are forward-filled causally from the last past close; a longer gap invalidates the feature for 168 hours. No backfill.
4. The four features are scaled by the standard deviation of the pair's finite research sample. The scale contains covariates only, no future returns.
5. The 20 controls with the smallest standardized Euclidean distance out of 60 candidate offsets are chosen; ties go to the older timestamp.

The matcher cannot read any outcome or return field. At least `95%` of the base event pool overall, and `90%` on each pair, must find all 20 controls, or the performance calibration doesn't run.

### Metrics and selection rule

`matched_lift = event's net 48h return − mean net 48h return of its 20 controls`.

Three future-campaign variants are scored: `matched_only` (mean lift above the null threshold), `matched_plus_median` (also positive absolute median return), and `matched_plus_median_blocks` (also at least three of four chronological blocks positive). The threshold is the 99th percentile (`method="higher"`) of mean matched lift in each density's null calibration half; only repetitions `500–999` are then evaluated, with one-sided 95% Wilson bounds.

A variant can become a future hard sieve only if, at **every** density, the null pass rate's Wilson upper bound is `<= 0.02` and the `+1.5%` pass rate's Wilson lower bound is `>= 0.75`. If the `+0.5%` Wilson lower bound is `>= 0.50` at every density, small edges count as resolvable; otherwise the pre-written conclusion is that a `+0.5% / 48h` edge is below this data's and this method's resolution. A selection is only a recommendation for a new campaign; `gates.py` does not change.

### Result

Official run completed on 2026-07-31.

- `114/114` tests, compile and shell checks passed. Exactly `4000` sets; `500` calibration and `500` evaluation sets per density, kept apart.
- `814034` of `838739` base events found all 20 controls: overall coverage `97.05%`, lowest pair `95.84%`.
- Schedule and control-map SHA-256 matched the preflight values above.

`matched_plus_median` behaved best:

| Density | Null pass | +0.5% pass | +1.5% pass | +1.5% Wilson lower |
|---:|---:|---:|---:|---:|
| 300 | `0.2%` | `10.6%` | `77.4%` | `74.18%` |
| 550 | `0.8%` | `11.6%` | `95.0%` | `93.14%` |
| 900 | `0.6%` | `30.6%` | `99.8%` | `99.11%` |
| 1300 | `0.6%` | `37.8%` | `100%` | `99.46%` |

- The null Wilson upper bound stayed under `2%` at every density.
- The `+1.5%` arm passed at 550, 900 and 1300 events. **At 300 events it was caught `77.4%` of the time, but the Wilson lower bound was `74.18% < 75%`, so it failed the pre-written gate.**
- The `+0.5%` arm's worst-density Wilson lower bound was `8.54%`; no variant passed the small-edge resolution gate.
- The four-block condition added no material discrimination to `matched_plus_median`.
- `selected_future_variant=None`, `edge_0_5_resolved=false`.
- Research DB SHA `17a6dbebe8bb1f0030ee15af990d3b4725f43af1cbcb9702a6cb655ec44ea809`, state `18 candidate / 6 alpha` and AI spend `$0.6388` unchanged.

**Decision:** EXP-077 **passed** technically, but no future hard gate was selected. The `75%` threshold and the 300-event density are not loosened after seeing the result. `matched_plus_median` is kept as a promising challenger; no new candidate, no `gates.py` change, no protected data.

### Independent review note

An independent review after EXP-077 confirmed the decisions and found no blocker. The null Wilson sentence above describes only `matched_plus_median`. `matched_only` also failed the Type-I rule: at 900 and 1300 events it had `5/500` null passes and a one-sided 95% Wilson upper bound of `2.03984% > 2%`.

Measured directly from `matched-runs.csv` afterwards, the null mean matched lift averages `0.1025`, `0.1213`, `0.1160` and `0.1181` percentage points by density, in the same direction in both halves. So the matcher reduces beta and regime drift but can't be presented as zero-centred or fully beta-neutral. Because the Q99 threshold comes from calibration null sets with the same structure, EXP-077's Type-I result and `selected_future_variant=None` don't change. The matcher is labelled `empirical_matched_residual_not_beta_neutral`.

## What it means

With the data that was open (`877` purged clusters), the gate could not see a `+0.5%` edge over 48 hours, and saw a `+1.5%` edge reliably only from 550 events up. That was reported as the result. Candidate generation stayed closed and no threshold was moved to make something pass.
