from __future__ import annotations

import ast
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from .models import Mechanism, SourceRef, canonical_json


INDICATORS = {
    "adx", "aroon", "atr", "bb", "bollinger", "cmf", "cti", "donchian",
    "ema", "ewo", "mfi", "obv", "roc", "rsi", "sma", "stoch", "tema",
    "ultimate", "volume", "vwap", "willr", "williams",
}
FORBIDDEN_FAMILIES = {
    "buyback": "buyback/position adjustment is outside the current contract",
    "dca": "position averaging is outside the current spot research contract",
    "short": "short/futures is outside the current product",
    "leverage": "leverage is outside the current product",
    "scalp": "5m scalping is not portable to the 1h engine",
    "grind": "grinding/position adjustment is outside the current contract",
    "rebuy": "rebuy/position adjustment is outside the current contract",
}
FAILED_FAMILIES = {"rsi", "obv", "squeeze", "momentum", "volume"}


def fetch_pinned_source(url: str, destination: Path, expected_sha256: str) -> str:
    request = Request(url, headers={"User-Agent": "SarpResearchFactory/0.1"})
    with urlopen(request, timeout=60) as response:
        content = response.read()
    actual = sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Source checksum mismatch: expected {expected_sha256}, got {actual}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return actual


class MechanismMiner:
    def __init__(self, repository: str, commit_sha: str, checksum: str) -> None:
        self.repository = repository
        self.commit_sha = commit_sha
        self.checksum = checksum

    def mine(self, path: Path) -> list[Mechanism]:
        source_text = path.read_text(encoding="utf-8")
        actual = sha256(source_text.encode()).hexdigest()
        if self.checksum and actual != self.checksum:
            raise ValueError("Pinned source checksum does not match local source")
        tree = ast.parse(source_text, filename=str(path))
        mechanisms: list[Mechanism] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name == "populate_entry_trend":
                mechanisms.extend(self._numbered_long_entries(path, node, source_text))
                continue
            name = node.name.casefold()
            if not any(token in name for token in ("buy", "entry", "sell", "exit", "condition")):
                continue
            segment = ast.get_source_segment(source_text, node) or ""
            mechanism = self._mechanism_from_function(path, node, segment)
            if mechanism is not None:
                mechanisms.append(mechanism)
        return sorted(mechanisms, key=lambda item: item.score, reverse=True)

    def _numbered_long_entries(
        self,
        path: Path,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        source_text: str,
    ) -> list[Mechanism]:
        results: list[Mechanism] = []
        for node in ast.walk(function):
            index = self._long_condition_index(node)
            if index is None:
                continue
            segment = ast.get_source_segment(source_text, node) or ""
            lowered = segment.casefold()
            indicators = tuple(sorted({name for name in INDICATORS if name in lowered}))
            if not indicators:
                continue
            rejected_reason = next(
                (reason for token, reason in FORBIDDEN_FAMILIES.items() if token in lowered),
                None,
            )
            novel = len(set(indicators) - FAILED_FAMILIES)
            novelty_score = min(1.0, 0.30 + novel * 0.12)
            if set(indicators).issubset(FAILED_FAMILIES):
                rejected_reason = rejected_reason or "mechanism only repeats exhausted OHLCV families"
                novelty_score = 0.0
            higher_tf = sum(token in lowered for token in ("_1h", "_4h", "_1d"))
            lower_tf = sum(token in lowered for token in ("_5m", "_15m"))
            portability = min(1.0, max(0.25, 0.65 + higher_tf * 0.10 - lower_tf * 0.20))
            conditions = tuple(self._condition_summaries(node)[:16])
            identity = {
                "name": f"long_entry_condition_{index}",
                "indicators": indicators,
                "conditions": conditions,
                "source_sha": self.commit_sha,
            }
            digest = sha256(canonical_json(identity).encode()).hexdigest()
            results.append(
                Mechanism(
                    mechanism_id=f"nfi-{digest[:12]}",
                    name=f"long_entry_condition_{index}",
                    intent="entry",
                    timeframe="mixed",
                    indicators=indicators,
                    conditions=conditions,
                    source=SourceRef(
                        repository=self.repository,
                        commit_sha=self.commit_sha,
                        checksum=self.checksum,
                        path=str(path),
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                    ),
                    portability_score=portability,
                    novelty_score=novelty_score,
                    sample_capacity_score=0.5,
                    safety_score=0.0 if rejected_reason else 0.8,
                    rejected_reason=rejected_reason,
                )
            )
        return results

    @staticmethod
    def _long_condition_index(node: ast.AST) -> int | None:
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            return None
        test = node.test
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "long_entry_condition_index"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, int)
        ):
            return test.comparators[0].value
        return None

    def _mechanism_from_function(
        self,
        path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        segment: str,
    ) -> Mechanism | None:
        lowered = segment.casefold()
        indicators = tuple(sorted({name for name in INDICATORS if name in lowered}))
        if not indicators:
            return None
        rejected_reason = next(
            (reason for token, reason in FORBIDDEN_FAMILIES.items() if token in lowered),
            None,
        )
        timeframe = self._timeframe(segment)
        if timeframe == "5m" or "5m" in node.name.casefold():
            rejected_reason = rejected_reason or "5m mechanism is outside the 1h execution contract"
        novel = len(set(indicators) - FAILED_FAMILIES)
        novelty_score = min(1.0, 0.25 + novel * 0.15)
        if set(indicators).issubset(FAILED_FAMILIES):
            rejected_reason = rejected_reason or "mechanism only repeats exhausted OHLCV families"
            novelty_score = 0.0
        portability = {"5m": 0.15, "15m": 0.35, "1h": 1.0, "4h": 0.9, "1d": 0.75}.get(
            timeframe, 0.65
        )
        conditions = tuple(self._condition_summaries(node)[:12])
        identity = {
            "name": node.name,
            "indicators": indicators,
            "conditions": conditions,
            "source_sha": self.commit_sha,
        }
        digest = sha256(canonical_json(identity).encode()).hexdigest()
        return Mechanism(
            mechanism_id=f"nfi-{digest[:12]}",
            name=node.name,
            intent="exit" if "exit" in node.name.casefold() or "sell" in node.name.casefold() else "entry",
            timeframe=timeframe,
            indicators=indicators,
            conditions=conditions,
            source=SourceRef(
                repository=self.repository,
                commit_sha=self.commit_sha,
                checksum=self.checksum,
                path=str(path),
                start_line=node.lineno,
                end_line=node.end_lineno or node.lineno,
            ),
            portability_score=portability,
            novelty_score=novelty_score,
            sample_capacity_score=0.5,
            safety_score=0.0 if rejected_reason else 0.8,
            rejected_reason=rejected_reason,
        )

    @staticmethod
    def _timeframe(segment: str) -> str:
        matches = re.findall(r"['\"](5m|15m|1h|4h|1d)['\"]", segment)
        return matches[0] if matches else "unknown"

    @staticmethod
    def _condition_summaries(node: ast.AST) -> list[str]:
        results: list[str] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                results.append(ast.dump(child, annotate_fields=False, include_attributes=False))
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in {"crossed_above", "crossed_below"}:
                    results.append(ast.dump(child, annotate_fields=False, include_attributes=False))
        return results


def write_catalog(path: Path, mechanisms: list[Mechanism]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) | {"score": item.score} for item in mechanisms], indent=2),
        encoding="utf-8",
    )
