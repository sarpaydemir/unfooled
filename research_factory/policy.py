from __future__ import annotations

from dataclasses import dataclass

from .storage import ResearchStore


@dataclass(frozen=True)
class BudgetPolicy:
    monthly_ai_budget_usd: float = 10.0
    max_candidates_per_cycle: int = 3
    cycle_cooldown_days: int = 14
    generation_enabled: bool = True

    def assert_ai_budget(self, store: ResearchStore) -> None:
        spent = store.monthly_ai_cost()
        if spent >= self.monthly_ai_budget_usd:
            raise RuntimeError(
                f"AI monthly budget exhausted: ${spent:.2f}/${self.monthly_ai_budget_usd:.2f}"
            )

    def assert_generation_enabled(self) -> None:
        if not self.generation_enabled:
            raise RuntimeError(
                "candidate generation is paused pending reviewer and gate calibration"
            )
