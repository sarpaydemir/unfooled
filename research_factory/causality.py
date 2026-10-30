from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .static_validation import load_v2_candidate


FIXTURE_VERSION = "synthetic-ohlcv-v1"
PREFIX_CUTOFFS = (384, 576, 768, 960)


def synthetic_frames(length: int = 1200) -> list[pd.DataFrame]:
    x = np.arange(length, dtype=float)
    midpoint = (length - 1) / 2
    closes = (
        100 + 0.04 * x + 4 * np.sin(x / 13),
        160 - 0.04 * x + 4 * np.sin(x / 17),
        100 + 0.06 * np.abs(x - midpoint) + 3 * np.sin(x / 11),
    )
    dates = pd.date_range("2020-01-01", periods=length, freq="1h", tz="UTC")
    frames: list[pd.DataFrame] = []
    for close in closes:
        open_ = close * (1 + 0.001 * np.sin(x / 7))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.01,
                    "low": np.minimum(open_, close) * 0.99,
                    "close": close,
                    "volume": 1000 + 100 * (1 + np.sin(x / 19)),
                },
                index=dates,
            )
        )
    return frames


def _boolean_signal(
    signal_function: Callable[[pd.DataFrame], pd.Series],
    frame: pd.DataFrame,
) -> pd.Series:
    output = signal_function(frame.copy())
    if not isinstance(output, pd.Series):
        raise RuntimeError("signal(dataframe) must return a pandas Series")
    if len(output) != len(frame) or not output.index.equals(frame.index):
        raise RuntimeError("signal output must preserve candidate frame length and index")
    return output.fillna(False).astype(bool)


def prefix_causality_metrics(candidate: Path, manifest: Path) -> dict:
    code_hash = sha256(candidate.read_bytes()).hexdigest()
    metrics = {
        "candidate_code_sha256": code_hash,
        "fixture_version": FIXTURE_VERSION,
        "prefix_checks": 0,
        "mismatched_prefixes": 0,
        "mismatched_rows": 0,
        "execution_errors": 0,
    }
    try:
        signal_function = load_v2_candidate(candidate, manifest)
        for frame in synthetic_frames():
            full = _boolean_signal(signal_function, frame)
            for cutoff in PREFIX_CUTOFFS:
                prefix = _boolean_signal(signal_function, frame.iloc[:cutoff])
                mismatches = int((prefix != full.iloc[:cutoff]).sum())
                metrics["prefix_checks"] += 1
                metrics["mismatched_rows"] += mismatches
                if mismatches:
                    metrics["mismatched_prefixes"] += 1
    except Exception:
        metrics["execution_errors"] += 1
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = prefix_causality_metrics(args.candidate, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
