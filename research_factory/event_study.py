from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .static_validation import load_v2_candidate


FINAL_CLOSED_BOUND = pd.Timestamp("2025-01-29 00:00:00", tz="UTC")


def load_candidate(
    path: Path,
    manifest: Path,
) -> Callable[[pd.DataFrame], pd.Series]:
    return load_v2_candidate(path, manifest)


def load_frame(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_feather(path)
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"OHLCV columns missing: {path}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] < end)]
    if frame.empty:
        raise RuntimeError(f"no data in research range: {path}")
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def merge_closed_informative(
    base: pd.DataFrame,
    informative: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    duration = {"4h": pd.Timedelta(hours=4), "1d": pd.Timedelta(days=1)}[timeframe]
    right = informative.copy()
    right["available_at"] = right["date"] + duration
    right = right.drop(columns=["date"]).rename(
        columns={column: f"{column}_{timeframe}" for column in right.columns if column != "available_at"}
    )
    left = base.copy()
    left["available_at"] = left["date"] + pd.Timedelta(hours=1)
    merged = pd.merge_asof(
        left.sort_values("available_at"),
        right.sort_values("available_at"),
        on="available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.drop(columns=["available_at"])


def causal_dataset(
    data_dir: Path,
    base: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frame = load_frame(data_dir / f"{base}_USDT-1h.feather", start, end)
    warm_start = start - pd.Timedelta(days=400)
    for timeframe in ("4h", "1d"):
        informative = load_frame(data_dir / f"{base}_USDT-{timeframe}.feather", warm_start, end)
        frame = merge_closed_informative(frame, informative, timeframe)
    return frame


def _net_return(entry: float, exit_price: float, fee: float) -> float:
    return ((exit_price * (1 - fee)) / (entry * (1 + fee)) - 1) * 100


def pair_events(
    frame: pd.DataFrame,
    pair: str,
    signal_function: Callable[[pd.DataFrame], pd.Series],
    fee: float,
    decluster_hours: int = 48,
) -> list[dict]:
    candidate_frame = frame.copy().set_index("date", drop=False)
    raw_signal = signal_function(candidate_frame)
    if not isinstance(raw_signal, pd.Series) or len(raw_signal) != len(frame):
        raise RuntimeError("signal(dataframe) must return a same-length pandas Series")
    signal = raw_signal.fillna(False).astype(bool).to_numpy()
    events: list[dict] = []
    last_event: pd.Timestamp | None = None
    for index in np.flatnonzero(signal):
        if index + 48 >= len(frame):
            continue
        signal_date = frame.iloc[index]["date"]
        entry_index = index + 1
        entry_date = frame.iloc[entry_index]["date"]
        if last_event is not None and entry_date - last_event < pd.Timedelta(hours=decluster_hours):
            continue
        entry = float(frame.iloc[entry_index]["open"])
        future = frame.iloc[entry_index:index + 49]
        event = {
            "pair": pair,
            "signal_date": signal_date,
            "date": entry_date,
            "entry": entry,
            "return_12h_pct": _net_return(entry, float(frame.iloc[index + 12]["close"]), fee),
            "return_24h_pct": _net_return(entry, float(frame.iloc[index + 24]["close"]), fee),
            "return_48h_pct": _net_return(entry, float(frame.iloc[index + 48]["close"]), fee),
            "mfe_48h_pct": _net_return(entry, float(future["high"].max()), fee),
            "mae_48h_pct": _net_return(entry, float(future["low"].min()), fee),
        }
        events.append(event)
        last_event = entry_date
    return events


def summarize(events: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if events.empty:
        return {
            "raw_events": 0,
            "effective_events": 0,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "positive_pair_fraction": 0.0,
            "positive_half_fraction": 0.0,
            "mfe_mae_ratio": 0.0,
        }
    midpoint = start + (end - start) / 2
    pair_medians = events.groupby("pair")["return_48h_pct"].median()
    halves = [
        events.loc[events["date"] < midpoint, "return_48h_pct"],
        events.loc[events["date"] >= midpoint, "return_48h_pct"],
    ]
    if any(part.empty for part in halves):
        positive_half_fraction = 0.0
    else:
        positive_half_fraction = sum(float(part.median()) > 0 for part in halves) / 2
    market_clusters = events["date"].dt.floor("12h").nunique()
    median_mfe = float(events["mfe_48h_pct"].median())
    median_mae = abs(float(events["mae_48h_pct"].median()))
    return {
        "raw_events": int(len(events)),
        "effective_events": int(market_clusters),
        "mean_return_pct": float(events["return_48h_pct"].mean()),
        "median_return_pct": float(events["return_48h_pct"].median()),
        "positive_pair_fraction": float((pair_medians > 0).mean()),
        "positive_half_fraction": float(positive_half_fraction),
        "mfe_mae_ratio": median_mfe / median_mae if median_mae else 0.0,
    }


def run_study(
    candidate: Path,
    manifest: Path,
    data_dir: Path,
    pairs: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    fee: float,
    output_dir: Path,
) -> dict:
    if end > FINAL_CLOSED_BOUND:
        raise RuntimeError(f"research end {end} exceeds closed bound {FINAL_CLOSED_BOUND}")
    signal = load_candidate(candidate, manifest)
    records: list[dict] = []
    for pair in pairs:
        records.extend(pair_events(causal_dataset(data_dir, pair, start, end), pair, signal, fee))
    events = pd.DataFrame(records)
    if not events.empty:
        events["date"] = pd.to_datetime(events["date"], utc=True)
    metrics = summarize(events, start, end)
    output_dir.mkdir(parents=True, exist_ok=True)
    events.to_csv(output_dir / "events.csv", index=False)
    (output_dir / "event-study-metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-01-29")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    metrics = run_study(
        args.candidate,
        args.manifest,
        args.data_dir,
        args.pairs,
        pd.Timestamp(args.start, tz="UTC"),
        pd.Timestamp(args.end, tz="UTC"),
        args.fee,
        args.output_dir,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
