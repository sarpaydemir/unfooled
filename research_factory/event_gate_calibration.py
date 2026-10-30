from __future__ import annotations

import argparse
from bisect import bisect_left, insort
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from .event_study import FINAL_CLOSED_BOUND, load_frame, summarize


DEVELOPMENT_PAIRS = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "ADA",
    "XRP",
    "ZEC",
    "UNI",
    "AAVE",
    "FET",
    "XLM",
    "INJ",
    "BCH",
    "ARB",
    "HBAR",
    "LDO",
    "ICP",
    "OP",
    "ATOM",
    "CRV",
)
START = pd.Timestamp("2019-01-01", tz="UTC")
END = pd.Timestamp("2025-01-29", tz="UTC")
FEE = 0.001
EDGE_PCT = 0.5
SEED = 76076
DENSITIES = (300, 550, 900, 1300)
REPETITIONS_PER_DENSITY = 250
CURRENT_ALPHA = 0.05 / 180
FUTURE_REFERENCE_ALPHA = 0.05 / 30

_METRIC_COLUMNS = (
    "effective_events",
    "mean_return_pct",
    "median_return_pct",
    "positive_pair_fraction",
    "positive_half_fraction",
    "mfe_mae_ratio",
)
_CONDITIONS = (
    "effective_events",
    "mean_return",
    "median_return",
    "positive_pair_fraction",
    "positive_half_fraction",
    "mfe_mae_ratio",
    "joint",
)


@dataclass(frozen=True)
class SampleSpec:
    density: int
    repetition: int
    event_indices: tuple[int, ...]


def _net_returns(entry: np.ndarray, exit_price: np.ndarray, fee: float) -> np.ndarray:
    return ((exit_price * (1 - fee)) / (entry * (1 + fee)) - 1) * 100


def _validated_frame(frame: pd.DataFrame, pair: str) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{pair} OHLCV columns missing")
    checked = frame.loc[:, ["date", "open", "high", "low", "close", "volume"]].copy()
    checked["date"] = pd.to_datetime(checked["date"], utc=True)
    checked = checked.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(checked) < 49:
        raise ValueError(f"{pair} needs at least 49 hourly rows")
    prices = checked[["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError(f"{pair} contains invalid prices")
    return checked


def _pair_pool(
    pair: str,
    frame: pd.DataFrame,
    fee: float,
    edge_pct: float,
) -> pd.DataFrame:
    checked = _validated_frame(frame, pair)
    eligible = len(checked) - 48
    date_index = pd.DatetimeIndex(checked["date"]).as_unit("ns")
    one_hour_ns = 60 * 60 * 1_000_000_000
    hourly_steps = np.diff(date_index.asi8) == one_hour_ns
    contiguous = np.lib.stride_tricks.sliding_window_view(
        hourly_steps,
        48,
    ).all(axis=1)
    opens = checked["open"].to_numpy(dtype=float)
    highs = checked["high"].to_numpy(dtype=float)
    lows = checked["low"].to_numpy(dtype=float)
    closes = checked["close"].to_numpy(dtype=float)

    entry = opens[1 : 1 + eligible][contiguous]
    high_windows = np.lib.stride_tricks.sliding_window_view(highs[1:], 48)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows[1:], 48)
    high_windows = high_windows[contiguous]
    low_windows = low_windows[contiguous]
    path_factor = 1 + (edge_pct / 100) * (np.arange(1, 49) / 48)

    close_12 = closes[12 : 12 + eligible][contiguous]
    close_24 = closes[24 : 24 + eligible][contiguous]
    close_48 = closes[48 : 48 + eligible][contiguous]
    mfe = high_windows.max(axis=1)
    mae = low_windows.min(axis=1)
    edge_mfe = (high_windows * path_factor).max(axis=1)
    edge_mae = (low_windows * path_factor).min(axis=1)

    signal_dates = pd.Series(date_index[:eligible][contiguous], dtype="datetime64[ns, UTC]")
    event_dates = pd.Series(
        date_index[1 : 1 + eligible][contiguous],
        dtype="datetime64[ns, UTC]",
    )
    target_dates = pd.Series(
        date_index[48 : 48 + eligible][contiguous],
        dtype="datetime64[ns, UTC]",
    )
    event_ids = [
        f"{pair}:{timestamp.isoformat()}"
        for timestamp in event_dates
    ]
    return pd.DataFrame(
        {
            "event_id": event_ids,
            "pair": pair,
            "signal_date": signal_dates,
            "date": event_dates,
            "target_date": target_dates,
            "entry": entry,
            "return_12h_pct": _net_returns(entry, close_12, fee),
            "return_24h_pct": _net_returns(entry, close_24, fee),
            "return_48h_pct": _net_returns(entry, close_48, fee),
            "mfe_48h_pct": _net_returns(entry, mfe, fee),
            "mae_48h_pct": _net_returns(entry, mae, fee),
            "edge_return_12h_pct": _net_returns(
                entry,
                close_12 * (1 + (edge_pct / 100) * 12 / 48),
                fee,
            ),
            "edge_return_24h_pct": _net_returns(
                entry,
                close_24 * (1 + (edge_pct / 100) * 24 / 48),
                fee,
            ),
            "edge_return_48h_pct": _net_returns(
                entry,
                close_48 * (1 + edge_pct / 100),
                fee,
            ),
            "edge_mfe_48h_pct": _net_returns(entry, edge_mfe, fee),
            "edge_mae_48h_pct": _net_returns(entry, edge_mae, fee),
        }
    )


def build_event_pool(
    frames: dict[str, pd.DataFrame],
    *,
    fee: float,
    edge_pct: float,
) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one pair frame is required")
    if not 0 <= fee < 1:
        raise ValueError("fee must be in [0, 1)")
    if not math.isfinite(edge_pct) or edge_pct <= 0:
        raise ValueError("edge_pct must be positive and finite")
    pools = [
        _pair_pool(pair, frame, fee, edge_pct)
        for pair, frame in sorted(frames.items())
    ]
    pool = pd.concat(pools, ignore_index=True)
    if pool["event_id"].duplicated().any():
        raise RuntimeError("event pool contains duplicate event IDs")
    return pool.sort_values(["date", "pair"]).reset_index(drop=True)


def build_sample_schedule(
    pool: pd.DataFrame,
    *,
    densities: Iterable[int],
    repetitions_per_density: int,
    seed: int,
) -> tuple[SampleSpec, ...]:
    densities = tuple(int(value) for value in densities)
    if not densities or any(value <= 0 for value in densities):
        raise ValueError("densities must be positive")
    if repetitions_per_density <= 0:
        raise ValueError("repetitions_per_density must be positive")
    if pool.empty:
        raise ValueError("event pool is empty")

    pair_values = pool["pair"].astype(str).to_numpy()
    event_index = pd.DatetimeIndex(pd.to_datetime(pool["date"], utc=True)).as_unit("ns")
    event_hours = event_index.asi8 // (60 * 60 * 1_000_000_000)
    rng = np.random.default_rng(seed)
    schedule: list[SampleSpec] = []
    for density in densities:
        for repetition in range(repetitions_per_density):
            selected: set[int] = set()
            pair_hours: dict[str, list[int]] = {}
            attempts = 0
            max_attempts = max(10_000, density * 200)
            while len(selected) < density and attempts < max_attempts:
                attempts += 1
                index = int(rng.integers(0, len(pool)))
                if index in selected:
                    continue
                pair = pair_values[index]
                hour = int(event_hours[index])
                accepted_hours = pair_hours.setdefault(pair, [])
                position = bisect_left(accepted_hours, hour)
                if position and hour - accepted_hours[position - 1] < 48:
                    continue
                if (
                    position < len(accepted_hours)
                    and accepted_hours[position] - hour < 48
                ):
                    continue
                selected.add(index)
                insort(accepted_hours, hour)
            if len(selected) != density:
                raise RuntimeError(
                    f"could not sample {density} declustered events after "
                    f"{attempts} attempts"
                )
            schedule.append(
                SampleSpec(
                    density=density,
                    repetition=repetition,
                    event_indices=tuple(sorted(selected)),
                )
            )
    return tuple(schedule)


def _sample_hash(sample: SampleSpec, pool: pd.DataFrame) -> str:
    event_ids = sorted(pool.iloc[list(sample.event_indices)]["event_id"].astype(str))
    payload = "\n".join(
        [f"density={sample.density}", f"repetition={sample.repetition}", *event_ids]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def schedule_sha256(schedule: Iterable[SampleSpec], pool: pd.DataFrame) -> str:
    hashes = [_sample_hash(sample, pool) for sample in schedule]
    return hashlib.sha256("\n".join(hashes).encode("utf-8")).hexdigest()


def _condition_results(metrics: dict) -> dict[str, bool]:
    conditions = {
        "effective_events": metrics["effective_events"] >= 60,
        "mean_return": metrics["mean_return_pct"] > 0,
        "median_return": metrics["median_return_pct"] > 0,
        "positive_pair_fraction": metrics["positive_pair_fraction"] >= 2 / 3,
        "positive_half_fraction": metrics["positive_half_fraction"] >= 1.0,
        "mfe_mae_ratio": metrics["mfe_mae_ratio"] >= 1.2,
    }
    conditions["joint"] = all(conditions.values())
    return conditions


def _mode_events(selected: pd.DataFrame, mode: str) -> pd.DataFrame:
    prefix = "" if mode == "null" else "edge_"
    return pd.DataFrame(
        {
            "pair": selected["pair"].to_numpy(),
            "date": pd.to_datetime(selected["date"], utc=True),
            "return_12h_pct": selected[f"{prefix}return_12h_pct"].to_numpy(),
            "return_24h_pct": selected[f"{prefix}return_24h_pct"].to_numpy(),
            "return_48h_pct": selected[f"{prefix}return_48h_pct"].to_numpy(),
            "mfe_48h_pct": selected[f"{prefix}mfe_48h_pct"].to_numpy(),
            "mae_48h_pct": selected[f"{prefix}mae_48h_pct"].to_numpy(),
        }
    )


def _cluster_sd(events: pd.DataFrame) -> float:
    cluster_returns = events.groupby(events["date"].dt.floor("12h"))[
        "return_48h_pct"
    ].mean()
    if len(cluster_returns) < 2:
        return 0.0
    return float(cluster_returns.std(ddof=1))


def _purged_phase_ids(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    bins = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).floor("12h").as_unit("ns")
    bin_ordinals = bins.asi8 // (12 * 60 * 60 * 1_000_000_000)
    return np.mod(bin_ordinals, 5)


def purged_power_statistics(events: pd.DataFrame) -> dict:
    if events.empty:
        raise ValueError("power statistics require events")
    cluster_returns = events.groupby(events["date"].dt.floor("12h"))[
        "return_48h_pct"
    ].mean()
    phases = _purged_phase_ids(cluster_returns.index)
    phase_counts: dict[str, int] = {}
    phase_sds: dict[str, float | None] = {}
    finite_sds: list[float] = []
    for phase in range(5):
        values = cluster_returns.to_numpy(dtype=float)[phases == phase]
        phase_counts[str(phase)] = int(len(values))
        if len(values) < 2:
            phase_sds[str(phase)] = None
            continue
        standard_deviation = float(np.std(values, ddof=1))
        if not math.isfinite(standard_deviation):
            phase_sds[str(phase)] = None
            continue
        phase_sds[str(phase)] = standard_deviation
        finite_sds.append(standard_deviation)
    if not finite_sds:
        raise RuntimeError("no purged phase has enough clusters for power")
    return {
        "method": "12h_bins_mod_5_60h_spacing",
        "phase_counts": phase_counts,
        "phase_sd_pct": phase_sds,
        "conservative_sd_pct": max(finite_sds),
        "max_independent_clusters": max(phase_counts.values()),
    }


def run_schedule(
    pool: pd.DataFrame,
    schedule: Iterable[SampleSpec],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in schedule:
        selected = pool.iloc[list(sample.event_indices)]
        row: dict = {
            "density": sample.density,
            "repetition": sample.repetition,
            "sample_sha256": _sample_hash(sample, pool),
        }
        for mode in ("null", "edge"):
            events = _mode_events(selected, mode)
            metrics = summarize(events, start, end)
            conditions = _condition_results(metrics)
            row.update(
                {f"{mode}_{key}": value for key, value in metrics.items()}
            )
            row[f"{mode}_cluster_sd_pct"] = _cluster_sd(events)
            power_stats = purged_power_statistics(events)
            row[f"{mode}_purged_power_sd_pct"] = power_stats[
                "conservative_sd_pct"
            ]
            row[f"{mode}_purged_power_clusters"] = power_stats[
                "max_independent_clusters"
            ]
            row.update(
                {
                    f"{mode}_{name}_pass": passed
                    for name, passed in conditions.items()
                }
            )
        rows.append(row)
    runs = pd.DataFrame(rows)
    numeric = runs.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("calibration produced non-finite metrics")
    return runs.sort_values(["density", "repetition"]).reset_index(drop=True)


def _metric_percentiles(runs: pd.DataFrame, mode: str) -> dict:
    return {
        metric: {
            "p01": float(runs[f"{mode}_{metric}"].quantile(0.01)),
            "p50": float(runs[f"{mode}_{metric}"].quantile(0.50)),
            "p99": float(runs[f"{mode}_{metric}"].quantile(0.99)),
        }
        for metric in _METRIC_COLUMNS
    }


def _pass_rates(runs: pd.DataFrame, mode: str) -> dict[str, float]:
    return {
        condition: float(runs[f"{mode}_{condition}_pass"].mean())
        for condition in _CONDITIONS
    }


def _summary_slice(runs: pd.DataFrame) -> dict:
    null_pass = _pass_rates(runs, "null")
    edge_pass = _pass_rates(runs, "edge")
    return {
        "set_count": int(len(runs)),
        "null_pass_rates": null_pass,
        "edge_pass_rates": edge_pass,
        "injected_false_reject_rates": {
            condition: 1 - rate
            for condition, rate in edge_pass.items()
        },
        "null_metric_percentiles": _metric_percentiles(runs, "null"),
        "edge_metric_percentiles": _metric_percentiles(runs, "edge"),
    }


def calibration_summary(
    runs: pd.DataFrame,
    *,
    available_12h_clusters: int,
    densities: Iterable[int],
    repetitions_per_density: int,
) -> dict:
    densities = tuple(int(value) for value in densities)
    expected = len(densities) * repetitions_per_density
    counts = {
        str(density): int((runs["density"] == density).sum())
        for density in densities
    }
    if len(runs) != expected or any(
        count != repetitions_per_density for count in counts.values()
    ):
        raise RuntimeError("calibration run counts do not match frozen design")

    overall = _summary_slice(runs)
    by_density = {
        str(density): _summary_slice(runs.loc[runs["density"] == density])
        for density in densities
    }
    contested = (
        "median_return",
        "positive_pair_fraction",
        "positive_half_fraction",
        "mfe_mae_ratio",
    )
    decisions: dict[str, dict] = {}
    for condition in contested:
        false_reject = overall["injected_false_reject_rates"][condition]
        false_reject_by_density = {
            str(density): by_density[str(density)][
                "injected_false_reject_rates"
            ][condition]
            for density in densities
        }
        worst_false_reject = max(false_reject_by_density.values())
        decision = {
            "injected_false_reject_rate": false_reject,
            "false_reject_by_density": false_reject_by_density,
            "worst_density_false_reject_rate": worst_false_reject,
            "false_reject_within_20pct": worst_false_reject <= 0.20,
        }
        if condition == "mfe_mae_ratio":
            null_q99 = overall["null_metric_percentiles"]["mfe_mae_ratio"]["p99"]
            null_q99_by_density = {
                str(density): by_density[str(density)][
                    "null_metric_percentiles"
                ]["mfe_mae_ratio"]["p99"]
                for density in densities
            }
            worst_null_q99 = max(null_q99_by_density.values())
            decision.update(
                {
                    "null_p99": null_q99,
                    "null_p99_by_density": null_q99_by_density,
                    "worst_density_null_p99": worst_null_q99,
                    "null_p99_below_1_20": worst_null_q99 < 1.20,
                    "hard_gate_eligible_next_campaign": (
                        worst_false_reject <= 0.20 and worst_null_q99 < 1.20
                    ),
                }
            )
        else:
            decision["hard_gate_eligible_next_campaign"] = (
                worst_false_reject <= 0.20
            )
        decisions[condition] = decision

    return {
        "run_count": int(len(runs)),
        "density_counts": counts,
        "available_12h_clusters": int(available_12h_clusters),
        "decision_rules": {
            "max_injected_false_reject_rate": 0.20,
            "mfe_mae_null_p99_threshold": 1.20,
            "current_campaign_changes_allowed": False,
        },
        "overall": overall,
        "by_density": by_density,
        "next_campaign_decisions": decisions,
    }


def power_requirement(
    *,
    cluster_sd_pct: float,
    alpha: float,
    edge_pct: float,
    power: float,
) -> dict:
    if cluster_sd_pct <= 0 or not math.isfinite(cluster_sd_pct):
        raise ValueError("cluster_sd_pct must be positive and finite")
    if not 0 < alpha < 1 or not 0.5 < power < 1 or edge_pct <= 0:
        raise ValueError("invalid power inputs")
    normal = NormalDist()
    critical_z = normal.inv_cdf(1 - alpha / 2)
    power_z = normal.inv_cdf(power)
    required = math.ceil(
        ((critical_z + power_z) * cluster_sd_pct / edge_pct) ** 2
    )
    mde_at_300 = (
        (critical_z + power_z) * cluster_sd_pct / math.sqrt(300)
    )
    return {
        "alpha": alpha,
        "power": power,
        "edge_pct": edge_pct,
        "cluster_sd_pct": cluster_sd_pct,
        "critical_z": critical_z,
        "required_clusters": required,
        "minimum_detectable_mean_pct_at_n_300": mde_at_300,
    }


def _power_report(runs: pd.DataFrame, available_purged_clusters: int) -> dict:
    report: dict[str, dict] = {}
    for density, group in runs.groupby("density", sort=True):
        cluster_sd = float(group["null_purged_power_sd_pct"].quantile(0.90))
        current = power_requirement(
            cluster_sd_pct=cluster_sd,
            alpha=CURRENT_ALPHA,
            edge_pct=EDGE_PCT,
            power=0.80,
        )
        future = power_requirement(
            cluster_sd_pct=cluster_sd,
            alpha=FUTURE_REFERENCE_ALPHA,
            edge_pct=EDGE_PCT,
            power=0.80,
        )
        current["attainable_in_open_training"] = (
            current["required_clusters"] <= available_purged_clusters
        )
        future["attainable_in_open_training"] = (
            future["required_clusters"] <= available_purged_clusters
        )
        report[str(int(density))] = {
            "power_unit_method": "12h_bins_mod_5_60h_spacing",
            "p90_conservative_cluster_sd_pct": cluster_sd,
            "available_purged_clusters": available_purged_clusters,
            "current_alpha_0_05_over_180": current,
            "future_reference_alpha_0_05_over_30": future,
        }
    return report


def run_calibration(data_dir: Path, output_dir: Path) -> dict:
    if END > FINAL_CLOSED_BOUND:
        raise RuntimeError("calibration range exceeds closed development bound")
    frames = {
        pair: load_frame(data_dir / f"{pair}_USDT-1h.feather", START, END)
        for pair in DEVELOPMENT_PAIRS
    }
    pool = build_event_pool(frames, fee=FEE, edge_pct=EDGE_PCT)
    schedule = build_sample_schedule(
        pool,
        densities=DENSITIES,
        repetitions_per_density=REPETITIONS_PER_DENSITY,
        seed=SEED,
    )
    replay = build_sample_schedule(
        pool,
        densities=DENSITIES,
        repetitions_per_density=REPETITIONS_PER_DENSITY,
        seed=SEED,
    )
    schedule_hash = schedule_sha256(schedule, pool)
    if schedule_hash != schedule_sha256(replay, pool):
        raise RuntimeError("seed replay schedule hash mismatch")

    runs = run_schedule(pool, schedule, start=START, end=END)
    available_clusters = int(pool["date"].dt.floor("12h").nunique())
    available_phase_ids = _purged_phase_ids(
        pd.DatetimeIndex(pool["date"].dt.floor("12h").unique())
    )
    available_purged_clusters = max(
        int(np.count_nonzero(available_phase_ids == phase))
        for phase in range(5)
    )
    summary = calibration_summary(
        runs,
        available_12h_clusters=available_clusters,
        densities=DENSITIES,
        repetitions_per_density=REPETITIONS_PER_DENSITY,
    )
    summary.update(
        {
            "experiment_id": "EXP-076-EVENT-GATE-CALIBRATION",
            "config": {
                "pairs": list(DEVELOPMENT_PAIRS),
                "start": START.isoformat(),
                "end_exclusive": END.isoformat(),
                "fee": FEE,
                "edge_pct": EDGE_PCT,
                "seed": SEED,
                "densities": list(DENSITIES),
                "repetitions_per_density": REPETITIONS_PER_DENSITY,
            },
            "event_pool_rows": int(len(pool)),
            "schedule_sha256": schedule_hash,
            "available_purged_power_clusters": available_purged_clusters,
            "power": _power_report(runs, available_purged_clusters),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_dir / "calibration-runs.csv", index=False, float_format="%.12g")
    (output_dir / "calibration-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "schedule-sha256.txt").write_text(
        schedule_hash + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_calibration(args.data_dir, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
