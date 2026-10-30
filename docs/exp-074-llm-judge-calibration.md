# EXP-074: calibrating the LLM judge

Translated from the Turkish write-up `research/experiments/EXP-074-REVIEWER-CALIBRATION.md` in the private research repo. The EXP-075 section is from `docs/RESEARCH_FACTORY.md` in the same repo. Numbers are copied, not recomputed.

Code in this repo: `research_factory/reviewer_benchmark_cases.py` (the frozen suite), `research_factory/reviewer_calibration.py` (the runner), `research_factory/workflow.py` (review prompt, payload validation, `review_hold`), `research_factory/gates.py` (`specification_review` gate).

## Question

In the pipeline, one LLM writes a research candidate (a hypothesis and a `signal(dataframe)` function), and a second LLM from a different model family reviews whether the code matches its own manifest. Before that reviewer is allowed to decide anything, how often does it reject a correct manifest–code pair, and how often does it miss a material mismatch? Measured on a frozen ground-truth set, so that a single LLM opinion can't permanently reject a candidate.

## Frozen design

- Generator model, unchanged: `openai/gpt-5.6-luna`.
- Reviewer: `z-ai/glm-5.2`, temperature `0`. The runner refuses a reviewer from the generator's model family.
- Suite: `EXP-074-REVIEWER-CALIBRATION-v1`, SHA-256 `4f537672c94e687d857cf4aa54aa11e7f485c1abdd02e6da38858bb34d4fca9e`. The suite in this repo still hashes to that value.
- 10 correct pairs and 10 pairs with exactly one material mutation; exactly 20 successful structured-output calls.
- Known-correct right-label and incomplete-bar gap behaviour is part of the passing set.
- Every call first takes an atomic idempotency claim keyed by suite, case, code, prompt, model and temperature.
- If total spend goes above `$0.80`, no further call is made.
- No candidate, alpha slot, market performance or protected data is used.

## Decision contract, written before the run

- A reviewer pass lets a candidate continue to the event study, after the deterministic gates.
- A reviewer fail produces `review_hold`, not a permanent rejection, and spends no alpha.
- `review_hold` can only be resolved by a machine artifact bound by SHA-256 to the code, the review and the evidence:
  - a confirmed material mismatch moves the candidate to `rejected`,
  - a confirmed reviewer false positive moves it back to `testing`.
- A single false pass in the benchmark means generation stays closed until that error class is caught by a deterministic gate.
- A false reject loses no data, because of the `review_hold` contract.

## Pass gate

- Suite structure and hash tests.
- 20/20 correct classifications are required for the reviewer to act as a hard gate.
- Every error is reported by case ID; Wilson 95% intervals are descriptive.
- All unit tests, compile, shell and the `18 candidate / 6 alpha` state audit must pass.

## Result

All 20 calls completed on 2026-07-30 with `z-ai/glm-5.2` at temperature `0`.

| | |
|---|---|
| Correct | `15/20` |
| False rejects | `3/10` passing cases: `P02-EMA-DIRECTION`, `P03-BBANDS-WIDTH`, `P09-HTF-AVAILABILITY` |
| False passes | `2/10` mismatch cases: `N07-AROON-OUTPUT`, `N10-ENTRY-SHIFT` |
| Passing-case acceptance | `0.70`, Wilson 95% `[0.3968, 0.8922]` |
| Mismatch detection | `0.80`, Wilson 95% `[0.4902, 0.9433]` |
| Total cost | `$0.06383972` |
| Unit tests | `74/74` |
| State audit | `18 candidate / 6 alpha`; protected data not opened |

In all three false rejects the model set all four semantic booleans to `true` but still returned `materiality_basis=confirmed_signal_mismatch`. The deterministic gate does not accept that contradiction; in the real candidate flow it becomes a `review_hold` that spends no alpha.

The reviewer passed `N10`, but the existing static safety gate already rejects `.shift(1)` on the returned signal. `N07` also passed static safety: TA-Lib's `AROON` returns `aroondown, aroonup`, and the mutation consumed the second output under the first one's name. On the three frozen synthetic OHLCV series, the passing and mutated signals differ on `1114`, `1131` and `1097` rows, `3342` in total.

## Decision

The hard-gate calibration **failed**, so the reviewer gets no authority to pass a candidate on its own. The same 20 cases are not reused to tune the prompt. Production generation stays off. Before any new candidate, a deterministic gate must catch at least the `N07` error class from the manifest, the code and canonical multi-output semantics, and the event-study gate/power calibration from the EXP-073 external audit must be finished (see [EXP-076/077](exp-076-077-gate-power-calibration.md)).

## Follow-up: EXP-075 turned the missed error into a deterministic check

EXP-075 added a schema-v2 indicator contract, a frozen TA-Lib output registry (`research_factory/talib_multi_output_v1.json`) and the code-bound `CompositeStaticValidator` (`research_factory/static_validation.py`, `research_factory/indicator_contract.py`). The same composite gate is required at generation, draft re-audit, the worker, the causality probe and the event study. Schema-v1 candidates are historical and read-only: they cannot be re-run or advanced.

Local and Docker smoke checks passed the correct output (`P07`) and rejected the wrong `AROON` tuple order (`N07`). The official run passed with `97/97` tests, a runtime metadata match, `P07` exit `0`, `N07` exit `2`, and the state unchanged at `18 candidate / 6 alpha`. No AI call and no market data.

In this repo: `tests/fixtures/EXP-075/`, `tests/test_indicator_contract.py`, and `tests/test_workflow.py::WorkflowTests::test_composite_rejects_n07_even_when_code_safety_passes`.
