from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any

from .models import canonical_json
from .policy import BudgetPolicy
from .providers import AIProvider, OpenRouterProvider
from .reviewer_benchmark_cases import benchmark_suite
from .schemas import SPECIFICATION_REVIEW_SCHEMA
from .storage import ResearchStore
from .workflow import REVIEW_SYSTEM_PROMPT, ResearchWorkflow


def _model_family(model: str) -> str:
    return model.casefold().strip().split("/", 1)[0].split("-", 1)[0]


def _predicted_pass(metrics: dict[str, Any]) -> bool:
    keys = (
        "hypothesis_code_consistent",
        "indicator_semantics_verified",
        "timeframe_semantics_verified",
        "entry_timing_consistent",
    )
    return all(metrics[key] is True for key in keys) and metrics["materiality_basis"] == "pass"


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    pass_cases = [row for row in results if row["expected"] == "pass"]
    reject_cases = [row for row in results if row["expected"] == "reject"]
    false_rejects = [row["case_id"] for row in pass_cases if not row["predicted_pass"]]
    false_passes = [row["case_id"] for row in reject_cases if row["predicted_pass"]]
    accepted_passes = len(pass_cases) - len(false_rejects)
    detected_rejects = len(reject_cases) - len(false_passes)
    return {
        "case_count": len(results),
        "expected_passes": len(pass_cases),
        "expected_rejects": len(reject_cases),
        "correct": len(results) - len(false_rejects) - len(false_passes),
        "false_reject_count": len(false_rejects),
        "false_reject_cases": false_rejects,
        "false_pass_count": len(false_passes),
        "false_pass_cases": false_passes,
        "pass_acceptance_rate": accepted_passes / len(pass_cases) if pass_cases else 0.0,
        "pass_acceptance_wilson95": _wilson(accepted_passes, len(pass_cases)),
        "mismatch_detection_rate": detected_rejects / len(reject_cases) if reject_cases else 0.0,
        "mismatch_detection_wilson95": _wilson(detected_rejects, len(reject_cases)),
        "accepted_for_hard_gate": not false_rejects and not false_passes and len(results) == 20,
    }


def run_suite(
    suite: dict[str, Any],
    store: ResearchStore,
    output_dir: Path,
    provider: AIProvider,
    monthly_budget_usd: float = 10.0,
    max_suite_cost_usd: float = 0.80,
) -> dict[str, Any]:
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("reviewer benchmark suite has no cases")
    expected_suite_hash = suite.get("suite_sha256")
    material = dict(suite)
    material.pop("suite_sha256", None)
    actual_suite_hash = sha256(canonical_json(material).encode()).hexdigest()
    if expected_suite_hash != actual_suite_hash:
        raise ValueError("reviewer benchmark suite SHA256 mismatch")
    configured_model = str(getattr(provider, "model", ""))
    if configured_model != suite["reviewer_model"]:
        raise RuntimeError(
            f"reviewer model mismatch: configured={configured_model} "
            f"frozen={suite['reviewer_model']}"
        )
    if _model_family(configured_model) == _model_family(suite["generator_model"]):
        raise RuntimeError("reviewer model family must differ from generator model family")
    configured_temperature = float(getattr(provider, "temperature", suite["temperature"]))
    if configured_temperature != float(suite["temperature"]):
        raise RuntimeError("reviewer temperature does not match frozen suite")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "suite.json").write_text(
        json.dumps(suite, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results: list[dict[str, Any]] = []
    total_cost = 0.0
    policy = BudgetPolicy(monthly_ai_budget_usd=monthly_budget_usd)
    for case in cases:
        if total_cost > max_suite_cost_usd:
            raise RuntimeError(
                f"reviewer calibration cost cap exceeded: "
                f"${total_cost:.4f}/${max_suite_cost_usd:.4f}"
            )
        case_id = str(case["case_id"])
        code = str(case["code"])
        code_hash = sha256(code.encode()).hexdigest()
        prompt = ResearchWorkflow._review_prompt(case["hypothesis"], code, code_hash)
        prompt_hash = sha256(prompt.encode()).hexdigest()
        result_path = output_dir / f"{case_id}.json"
        identity = {
            "suite_sha256": actual_suite_hash,
            "case_id": case_id,
            "code_sha256": code_hash,
            "prompt_sha256": prompt_hash,
            "model": configured_model,
            "temperature": configured_temperature,
        }
        digest = sha256(canonical_json(identity).encode()).hexdigest()[:20]
        job = store.begin_idempotent_job(
            f"review-calibration-{digest}",
            f"{suite['suite_id']}:{case_id}:{actual_suite_hash}",
            "reviewer_calibration",
            identity,
        )
        if not job["acquired"]:
            if job["status"] != "completed" or not result_path.is_file():
                raise RuntimeError(
                    f"calibration case is fail-closed in status={job['status']}: {case_id}"
                )
            stored = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(stored)
            total_cost += float(stored["cost_usd"])
            continue
        job_terminal = False
        try:
            policy.assert_ai_budget(store)
            response = provider.generate(prompt, SPECIFICATION_REVIEW_SCHEMA)
            metrics = ResearchWorkflow._validate_review_payload(response.payload)
            if metrics["reviewed_code_sha256"] != code_hash:
                raise ValueError(f"reviewer returned wrong code hash for {case_id}")
            store.record_usage(
                response.provider,
                response.model,
                response.prompt_tokens,
                response.completion_tokens,
                response.cost_usd,
            )
            row = {
                "case_id": case_id,
                "expected": case["expected"],
                "predicted_pass": _predicted_pass(metrics),
                "ground_truth_evidence": case["ground_truth_evidence"],
                "code_sha256": code_hash,
                "prompt_sha256": prompt_hash,
                "reviewer_provider": response.provider,
                "reviewer_model": response.model,
                "temperature": configured_temperature,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cost_usd": response.cost_usd,
                "request_id": response.request_id,
                "metrics": metrics,
            }
            result_path.write_text(
                json.dumps(row, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            store.finish_job(job["job_id"], "completed")
            job_terminal = True
            results.append(row)
            total_cost += response.cost_usd
            if total_cost > max_suite_cost_usd:
                raise RuntimeError(
                    f"reviewer calibration cost cap exceeded after {case_id}: "
                    f"${total_cost:.4f}/${max_suite_cost_usd:.4f}"
                )
        except Exception as error:
            if not job_terminal:
                store.finish_job(job["job_id"], "failed", f"{type(error).__name__}: {error}")
            raise

    summary = {
        "suite_id": suite["suite_id"],
        "suite_sha256": actual_suite_hash,
        "reviewer_model": configured_model,
        "temperature": configured_temperature,
        "total_cost_usd": total_cost,
        "max_suite_cost_usd": max_suite_cost_usd,
        **summarize_results(results),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, default=Path(".research_factory"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".research_factory/reviewer-benchmarks/EXP-074"),
    )
    parser.add_argument("--max-suite-cost-usd", type=float, default=0.80)
    args = parser.parse_args()
    model = os.getenv("RESEARCH_REVIEW_MODEL", "").strip()
    if not model:
        raise RuntimeError("RESEARCH_REVIEW_MODEL must be explicitly configured")
    temperature = float(os.getenv("RESEARCH_REVIEW_TEMPERATURE", "0"))
    provider = OpenRouterProvider(
        model=model,
        temperature=temperature,
        system_prompt=REVIEW_SYSTEM_PROMPT,
    )
    summary = run_suite(
        benchmark_suite(),
        # Benchmark accounting records paid AI usage in the factory ledger.
        ResearchStore(args.state_dir / "research.db", writable=True),
        args.output_dir,
        provider,
        monthly_budget_usd=float(os.getenv("AI_MONTHLY_BUDGET_USD", "10")),
        max_suite_cost_usd=args.max_suite_cost_usd,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
