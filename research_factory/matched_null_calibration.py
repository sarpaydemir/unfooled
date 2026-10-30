from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from .event_gate_calibration import (
    DEVELOPMENT_PAIRS,
    END,
    FEE,
    START,
    SampleSpec,
    build_event_pool,
    build_sample_schedule,
    schedule_sha256,
)
from .event_study import FINAL_CLOSED_BOUND, load_frame


EXPERIMENT_ID = "EXP-077-MATCHED-NULL-CALIBRATION"
SEED = 77077
DENSITIES = (300, 550, 900, 1300)
REPETITIONS_PER_DENSITY = 1000
CALIBRATION_REPETITIONS = 500
EDGE_PCTS = (0.5, 1.5)
CONTROL_COUNT = 20
CONTROL_OFFSETS_HOURS = tuple(range(-1488, -71, 24))
FEATURE_WINDOW_HOURS = 168
MINIMUM_OVERALL_COVERAGE = 0.95
MINIMUM_PAIR_COVERAGE = 0.90
VARIANTS = (
    "matched_only",
    "matched_plus_median",
    "matched_plus_median_blocks",
)
ARMS = ("null", "edge_0_5", "edge_1_5")


def _arm_name(edge_pct: float) -> str:
    return f"edge_{str(edge_pct).replace('.', '_')}"


def _edge_adjusted_return(
    null_return_pct: np.ndarray | pd.Series,
    edge_pct: float,
) -> np.ndarray:
    values = np.asarray(null_return_pct, dtype=float)
    return ((1 + values / 100) * (1 + edge_pct / 100) - 1) * 100


def trailing_features(
    frame: pd.DataFrame,
    *,
    window_hours: int = FEATURE_WINDOW_HOURS,
    maximum_forward_fill_hours: int = 3,
) -> pd.DataFrame:
    if window_hours < 2:
        raise ValueError("feature window must be at least two hours")
    required = {"date", "close"}
    if not required.issubset(frame.columns):
        raise ValueError("feature frame requires date and close")
    checked = frame.loc[:, ["date", "close"]].copy()
    checked["date"] = pd.to_datetime(checked["date"], utc=True)
    checked = checked.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    close = checked["close"].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("feature frame contains invalid close prices")
    if maximum_forward_fill_hours < 0:
        raise ValueError("maximum forward fill must be non-negative")
    full_index = pd.date_range(
        checked["date"].iloc[0],
        checked["date"].iloc[-1],
        freq="1h",
        tz="UTC",
    )
    actual_dates = pd.DatetimeIndex(checked["date"])
    hourly_close = pd.Series(
        close,
        index=actual_dates,
    ).reindex(full_index)
    if maximum_forward_fill_hours:
        hourly_close = hourly_close.ffill(limit=maximum_forward_fill_hours)
    log_close = np.log(hourly_close)
    log_return = log_close.diff()
    feature_return = log_close - log_close.shift(window_hours)
    feature_volatility = log_return.rolling(
        window_hours,
        min_periods=window_hours,
    ).std(ddof=0)
    valid_history = (
        hourly_close.notna()
        .rolling(window_hours + 1, min_periods=window_hours + 1)
        .sum()
        == window_hours + 1
    )
    feature_return = feature_return.where(valid_history)
    feature_volatility = feature_volatility.where(valid_history)
    result = pd.DataFrame(
        {
            "signal_date": actual_dates,
            "return_7d": feature_return.reindex(actual_dates).to_numpy(),
            "volatility_7d": feature_volatility.reindex(actual_dates).to_numpy(),
        }
    )
    return result


def _attach_features(
    pool: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    *,
    feature_window_hours: int,
) -> pd.DataFrame:
    if "BTC" not in frames:
        raise ValueError("BTC frame is required for matched features")
    btc = trailing_features(
        frames["BTC"],
        window_hours=feature_window_hours,
    ).rename(
        columns={
            "return_7d": "btc_return_7d",
            "volatility_7d": "btc_volatility_7d",
        }
    )
    pieces: list[pd.DataFrame] = []
    for pair, group in pool.groupby("pair", sort=True):
        if pair not in frames:
            raise ValueError(f"missing frame for {pair}")
        pair_features = trailing_features(
            frames[pair],
            window_hours=feature_window_hours,
        ).rename(
            columns={
                "return_7d": "pair_return_7d",
                "volatility_7d": "pair_volatility_7d",
            }
        )
        enriched = group.merge(pair_features, on="signal_date", how="left")
        enriched = enriched.merge(btc, on="signal_date", how="left")
        pieces.append(enriched)
    result = pd.concat(pieces, ignore_index=True)
    feature_columns = [
        "pair_return_7d",
        "pair_volatility_7d",
        "btc_return_7d",
        "btc_volatility_7d",
    ]
    finite = np.isfinite(result[feature_columns].to_numpy(dtype=float)).all(axis=1)
    return result.loc[finite].sort_values(["date", "pair"]).reset_index(drop=True)


def _candidate_matrix(
    dates_ns: np.ndarray,
    features: np.ndarray,
    offsets_hours: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    count = len(dates_ns)
    candidate_indices = np.full((count, len(offsets_hours)), -1, dtype=np.int32)
    distances = np.full((count, len(offsets_hours)), np.inf, dtype=np.float64)
    for column, offset in enumerate(offsets_hours):
        targets = dates_ns + np.int64(offset) * 60 * 60 * 1_000_000_000
        positions = np.searchsorted(dates_ns, targets)
        valid = positions < count
        safe_positions = np.minimum(positions, count - 1)
        valid &= dates_ns[safe_positions] == targets
        rows = np.flatnonzero(valid)
        if not len(rows):
            continue
        controls = safe_positions[rows]
        differences = features[rows] - features[controls]
        candidate_indices[rows, column] = controls.astype(np.int32)
        distances[rows, column] = np.einsum(
            "ij,ij->i",
            differences,
            differences,
        )
    return candidate_indices, distances


def _pair_matches(
    group: pd.DataFrame,
    *,
    offsets_hours: tuple[int, ...],
    control_count: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if control_count <= 0 or control_count > len(offsets_hours):
        raise ValueError("invalid control count")
    ordered = group.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(pd.to_datetime(ordered["date"], utc=True)).as_unit("ns")
    dates_ns = dates.asi8
    feature_columns = [
        "pair_return_7d",
        "pair_volatility_7d",
        "btc_return_7d",
        "btc_volatility_7d",
    ]
    feature_values = ordered[feature_columns].to_numpy(dtype=float)
    scales = np.std(feature_values, axis=0, ddof=0)
    scales[~np.isfinite(scales) | (scales <= 1e-12)] = 1.0
    standardized = feature_values / scales
    candidates, distances = _candidate_matrix(
        dates_ns,
        standardized,
        offsets_hours,
    )
    order = np.argsort(distances, axis=1, kind="stable")[:, :control_count]
    selected_distances = np.take_along_axis(distances, order, axis=1)
    selected_controls = np.take_along_axis(candidates, order, axis=1)
    matched_mask = np.isfinite(selected_distances).all(axis=1)
    rows = np.flatnonzero(matched_mask)
    controls = selected_controls[rows]
    matched = ordered.iloc[rows].copy().reset_index(drop=True)
    null_returns = ordered["return_48h_pct"].to_numpy(dtype=float)
    matched["matched_control_return_48h_pct"] = null_returns[controls].mean(axis=1)
    matched["control_count"] = control_count
    matched["minimum_control_gap_hours"] = min(abs(value) for value in offsets_hours)
    hour_ns = 60 * 60 * 1_000_000_000
    selected_offsets = (
        dates_ns[controls] - dates_ns[rows, np.newaxis]
    ) // hour_ns
    if (selected_offsets >= 0).any():
        raise RuntimeError("matched controls must be strictly in the past")
    if not np.isin(selected_offsets, np.asarray(offsets_hours)).all():
        raise RuntimeError("matched control offset is outside the frozen set")
    return matched, controls, selected_offsets.astype(np.int32)


def _offset_diagnostics(offsets: np.ndarray) -> dict:
    values = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if not len(values):
        raise RuntimeError("matched controls produced no offset diagnostics")
    unique, counts = np.unique(values, return_counts=True)
    histogram = {
        str(int(offset)): int(count)
        for offset, count in zip(unique, counts, strict=True)
    }
    return {
        "selection_count": int(len(values)),
        "minimum_hours": int(values.min()),
        "maximum_hours": int(values.max()),
        "mean_hours": float(values.mean()),
        "median_hours": float(np.median(values)),
        "p10_hours": float(np.quantile(values, 0.10)),
        "p90_hours": float(np.quantile(values, 0.90)),
        "histogram": histogram,
    }


def _lift_diagnostics(pool: pd.DataFrame) -> dict:
    def summarize(group: pd.DataFrame) -> dict:
        event = group["null_return_48h_pct"].to_numpy(dtype=float)
        control = group["matched_control_return_48h_pct"].to_numpy(dtype=float)
        lift = group["null_matched_lift_pct"].to_numpy(dtype=float)
        return {
            "events": int(len(group)),
            "mean_event_return_pct": float(event.mean()),
            "mean_control_return_pct": float(control.mean()),
            "mean_matched_lift_pct": float(lift.mean()),
            "median_matched_lift_pct": float(np.median(lift)),
        }

    dated = pool.copy()
    dated["_year"] = pd.to_datetime(dated["date"], utc=True).dt.year
    return {
        "interpretation": "empirical_matched_residual_not_beta_neutral",
        "overall": summarize(dated),
        "by_pair": {
            str(pair): summarize(group)
            for pair, group in dated.groupby("pair", sort=True)
        },
        "by_year": {
            str(int(year)): summarize(group)
            for year, group in dated.groupby("_year", sort=True)
        },
    }


def build_matched_event_pool(
    frames: dict[str, pd.DataFrame],
    *,
    fee: float,
    edge_pcts: Iterable[float] = EDGE_PCTS,
    offsets_hours: Iterable[int] = CONTROL_OFFSETS_HOURS,
    control_count: int = CONTROL_COUNT,
    feature_window_hours: int = FEATURE_WINDOW_HOURS,
    minimum_overall_coverage: float = MINIMUM_OVERALL_COVERAGE,
    minimum_pair_coverage: float = MINIMUM_PAIR_COVERAGE,
) -> tuple[pd.DataFrame, dict]:
    edge_pcts = tuple(float(value) for value in edge_pcts)
    offsets_hours = tuple(sorted({int(value) for value in offsets_hours}))
    if not edge_pcts or any(value <= 0 or not math.isfinite(value) for value in edge_pcts):
        raise ValueError("edge percentages must be positive and finite")
    if not offsets_hours or 0 in offsets_hours:
        raise ValueError("control offsets must be non-zero")
    if any(abs(value) < 49 for value in offsets_hours):
        raise ValueError("control offsets must be at least 49 hours")
    if any(value > 0 for value in offsets_hours):
        raise ValueError("control offsets must be strictly before the event")
    if not 0 <= minimum_overall_coverage <= 1 or not 0 <= minimum_pair_coverage <= 1:
        raise ValueError("coverage thresholds must be in [0, 1]")

    base = build_event_pool(frames, fee=fee, edge_pct=edge_pcts[0])
    base_counts = base.groupby("pair").size().to_dict()
    featured = _attach_features(
        base,
        frames,
        feature_window_hours=feature_window_hours,
    )
    matched_pieces: list[pd.DataFrame] = []
    selected_offset_pieces: list[np.ndarray] = []
    pair_diagnostics: dict[str, dict] = {}
    mapping_hash = hashlib.sha256()
    for pair, group in featured.groupby("pair", sort=True):
        matched, controls, selected_offsets = _pair_matches(
            group,
            offsets_hours=offsets_hours,
            control_count=control_count,
        )
        matched_pieces.append(matched)
        selected_offset_pieces.append(selected_offsets)
        base_count = int(base_counts.get(pair, 0))
        matched_count = int(len(matched))
        pair_coverage = matched_count / base_count if base_count else 0.0
        pair_diagnostics[str(pair)] = {
            "base_events": base_count,
            "feature_eligible_events": int(len(group)),
            "matched_events": matched_count,
            "coverage": pair_coverage,
            "selected_control_offsets": _offset_diagnostics(selected_offsets),
        }
        mapping_hash.update(str(pair).encode("utf-8"))
        mapping_hash.update(
            pd.DatetimeIndex(matched["date"]).as_unit("ns").asi8.astype("<i8").tobytes()
        )
        mapping_hash.update(controls.astype("<i4").tobytes())

    matched_pool = pd.concat(matched_pieces, ignore_index=True)
    matched_pool = matched_pool.sort_values(["date", "pair"]).reset_index(drop=True)
    base_count = int(len(base))
    overall_coverage = len(matched_pool) / base_count if base_count else 0.0
    if overall_coverage < minimum_overall_coverage:
        raise RuntimeError(
            f"matched coverage {overall_coverage:.6f} below "
            f"{minimum_overall_coverage:.6f}"
        )
    weak_pairs = {
        pair: values["coverage"]
        for pair, values in pair_diagnostics.items()
        if values["coverage"] < minimum_pair_coverage
    }
    if weak_pairs:
        raise RuntimeError(f"pair matched coverage below threshold: {weak_pairs}")

    null_return = matched_pool["return_48h_pct"].to_numpy(dtype=float)
    control_return = matched_pool["matched_control_return_48h_pct"].to_numpy(
        dtype=float
    )
    matched_pool["null_return_48h_pct"] = null_return
    matched_pool["null_matched_lift_pct"] = null_return - control_return
    for edge_pct in edge_pcts:
        arm = _arm_name(edge_pct)
        adjusted = _edge_adjusted_return(null_return, edge_pct)
        matched_pool[f"{arm}_return_48h_pct"] = adjusted
        matched_pool[f"{arm}_matched_lift_pct"] = adjusted - control_return

    keep = [
        "event_id",
        "pair",
        "signal_date",
        "date",
        "target_date",
        "matched_control_return_48h_pct",
        "control_count",
        "minimum_control_gap_hours",
        "null_return_48h_pct",
        "null_matched_lift_pct",
    ]
    for edge_pct in edge_pcts:
        arm = _arm_name(edge_pct)
        keep.extend(
            [
                f"{arm}_return_48h_pct",
                f"{arm}_matched_lift_pct",
            ]
        )
    matched_pool = matched_pool.loc[:, keep]
    diagnostics = {
        "base_event_count": base_count,
        "feature_eligible_event_count": int(len(featured)),
        "matched_event_count": int(len(matched_pool)),
        "overall_coverage": overall_coverage,
        "minimum_overall_coverage": minimum_overall_coverage,
        "minimum_pair_coverage": minimum_pair_coverage,
        "control_count": control_count,
        "offsets_hours": list(offsets_hours),
        "selected_control_offsets": _offset_diagnostics(
            np.concatenate(selected_offset_pieces, axis=0)
        ),
        "pair_diagnostics": pair_diagnostics,
        "control_map_sha256": mapping_hash.hexdigest(),
        "null_lift": _lift_diagnostics(matched_pool),
    }
    return matched_pool, diagnostics


def _sample_hash(sample: SampleSpec, pool: pd.DataFrame) -> str:
    event_ids = sorted(pool.iloc[list(sample.event_indices)]["event_id"].astype(str))
    payload = "\n".join(
        [f"density={sample.density}", f"repetition={sample.repetition}", *event_ids]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quarter_fraction(selected: pd.DataFrame, lift_column: str) -> float:
    ordered = selected.sort_values(["date", "pair"])
    quarters = np.array_split(ordered[lift_column].to_numpy(dtype=float), 4)
    if any(not len(values) for values in quarters):
        raise RuntimeError("four temporal blocks require at least four events")
    return float(np.mean([float(np.mean(values)) > 0 for values in quarters]))


def run_matched_schedule(
    pool: pd.DataFrame,
    schedule: Iterable[SampleSpec],
    *,
    calibration_repetitions: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for sample in schedule:
        selected = pool.iloc[list(sample.event_indices)]
        row: dict = {
            "density": sample.density,
            "repetition": sample.repetition,
            "split": (
                "calibration"
                if sample.repetition < calibration_repetitions
                else "evaluation"
            ),
            "sample_sha256": _sample_hash(sample, pool),
            "effective_events": int(
                pd.DatetimeIndex(selected["date"]).floor("12h").nunique()
            ),
        }
        for arm in ARMS:
            returns = selected[f"{arm}_return_48h_pct"].to_numpy(dtype=float)
            lifts = selected[f"{arm}_matched_lift_pct"].to_numpy(dtype=float)
            pair_means = selected.assign(_lift=lifts).groupby("pair")["_lift"].mean()
            row.update(
                {
                    f"{arm}_mean_return_pct": float(np.mean(returns)),
                    f"{arm}_median_return_pct": float(np.median(returns)),
                    f"{arm}_mean_matched_lift_pct": float(np.mean(lifts)),
                    f"{arm}_median_matched_lift_pct": float(np.median(lifts)),
                    f"{arm}_positive_pair_fraction": float((pair_means > 0).mean()),
                    f"{arm}_positive_quarter_fraction": _quarter_fraction(
                        selected,
                        f"{arm}_matched_lift_pct",
                    ),
                }
            )
        rows.append(row)
    runs = pd.DataFrame(rows).sort_values(["density", "repetition"]).reset_index(
        drop=True
    )
    numeric = runs.select_dtypes(include="number")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("matched calibration produced non-finite metrics")
    return runs


def wilson_bound(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid Wilson inputs")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence must be in (0.5, 1)")
    z = NormalDist().inv_cdf(confidence)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _variant_passes(
    runs: pd.DataFrame,
    *,
    arm: str,
    threshold: float,
    variant: str,
) -> pd.Series:
    passed = (
        (runs["effective_events"] >= 60)
        & (runs[f"{arm}_mean_matched_lift_pct"] > threshold)
    )
    if variant in {"matched_plus_median", "matched_plus_median_blocks"}:
        passed &= runs[f"{arm}_median_return_pct"] > 0
    if variant == "matched_plus_median_blocks":
        passed &= runs[f"{arm}_positive_quarter_fraction"] >= 0.75
    return passed


def calibration_summary(
    runs: pd.DataFrame,
    *,
    densities: Iterable[int],
    repetitions_per_density: int,
    calibration_repetitions: int,
) -> tuple[pd.DataFrame, dict]:
    densities = tuple(int(value) for value in densities)
    if not 0 < calibration_repetitions < repetitions_per_density:
        raise ValueError("calibration split must be inside repetitions")
    counts = {
        str(density): int((runs["density"] == density).sum())
        for density in densities
    }
    if any(value != repetitions_per_density for value in counts.values()):
        raise RuntimeError("matched run counts do not match frozen design")

    thresholds: dict[str, float] = {}
    augmented = runs.copy()
    for density in densities:
        calibration = augmented.loc[
            (augmented["density"] == density)
            & (augmented["split"] == "calibration"),
            "null_mean_matched_lift_pct",
        ]
        if len(calibration) != calibration_repetitions:
            raise RuntimeError("calibration split count mismatch")
        thresholds[str(density)] = float(
            calibration.quantile(0.99, interpolation="higher")
        )

    for variant in VARIANTS:
        for arm in ARMS:
            pass_column = f"{arm}_{variant}_pass"
            augmented[pass_column] = False
            for density in densities:
                mask = (
                    (augmented["density"] == density)
                    & (augmented["split"] == "evaluation")
                )
                augmented.loc[mask, pass_column] = _variant_passes(
                    augmented.loc[mask],
                    arm=arm,
                    threshold=thresholds[str(density)],
                    variant=variant,
                )

    evaluation_count = repetitions_per_density - calibration_repetitions
    decisions: dict[str, dict] = {}
    eligible: list[tuple[float, int, str]] = []
    for priority, variant in enumerate(VARIANTS):
        by_density: dict[str, dict] = {}
        variant_eligible = True
        resolved_0_5 = True
        worst_edge_0_5_lower = 1.0
        for density in densities:
            subset = augmented.loc[
                (augmented["density"] == density)
                & (augmented["split"] == "evaluation")
            ]
            if len(subset) != evaluation_count:
                raise RuntimeError("evaluation split count mismatch")
            arm_results: dict[str, dict] = {}
            for arm in ARMS:
                successes = int(subset[f"{arm}_{variant}_pass"].sum())
                lower, upper = wilson_bound(successes, evaluation_count)
                arm_results[arm] = {
                    "passes": successes,
                    "trials": evaluation_count,
                    "rate": successes / evaluation_count,
                    "wilson_lower_95_one_sided": lower,
                    "wilson_upper_95_one_sided": upper,
                }
            density_eligible = (
                arm_results["null"]["wilson_upper_95_one_sided"] <= 0.02
                and arm_results["edge_1_5"]["wilson_lower_95_one_sided"] >= 0.75
            )
            density_resolved = (
                arm_results["edge_0_5"]["wilson_lower_95_one_sided"] >= 0.50
            )
            variant_eligible &= density_eligible
            resolved_0_5 &= density_resolved
            worst_edge_0_5_lower = min(
                worst_edge_0_5_lower,
                arm_results["edge_0_5"]["wilson_lower_95_one_sided"],
            )
            by_density[str(density)] = {
                "threshold_pct": thresholds[str(density)],
                "arms": arm_results,
                "future_hard_gate_eligible": density_eligible,
                "edge_0_5_resolved": density_resolved,
            }
        decisions[variant] = {
            "by_density": by_density,
            "future_hard_gate_eligible": variant_eligible,
            "edge_0_5_resolved": resolved_0_5,
            "worst_density_edge_0_5_wilson_lower": worst_edge_0_5_lower,
        }
        if variant_eligible:
            eligible.append((worst_edge_0_5_lower, -priority, variant))

    selected_variant = max(eligible)[2] if eligible else None
    summary = {
        "run_count": int(len(augmented)),
        "density_counts": counts,
        "calibration_repetitions_per_density": calibration_repetitions,
        "evaluation_repetitions_per_density": evaluation_count,
        "null_thresholds_pct": thresholds,
        "decision_rules": {
            "null_wilson_upper_max": 0.02,
            "edge_1_5_wilson_lower_min": 0.75,
            "edge_0_5_resolution_wilson_lower_min": 0.50,
            "current_campaign_changes_allowed": False,
        },
        "variant_decisions": decisions,
        "selected_future_variant": selected_variant,
        "edge_0_5_resolved": (
            decisions[selected_variant]["edge_0_5_resolved"]
            if selected_variant
            else False
        ),
    }
    return augmented, summary


def run_calibration(data_dir: Path, output_dir: Path) -> dict:
    if END > FINAL_CLOSED_BOUND:
        raise RuntimeError("matched calibration exceeds closed development bound")
    frames = {
        pair: load_frame(data_dir / f"{pair}_USDT-1h.feather", START, END)
        for pair in DEVELOPMENT_PAIRS
    }
    replay_pool, replay_diagnostics = build_matched_event_pool(frames, fee=FEE)
    replay_control_hash = replay_diagnostics["control_map_sha256"]
    del replay_pool
    gc.collect()
    pool, match_diagnostics = build_matched_event_pool(frames, fee=FEE)
    if match_diagnostics["control_map_sha256"] != replay_control_hash:
        raise RuntimeError("control-map replay mismatch")

    replay_schedule = build_sample_schedule(
        pool,
        densities=DENSITIES,
        repetitions_per_density=REPETITIONS_PER_DENSITY,
        seed=SEED,
    )
    replay_schedule_hash = schedule_sha256(replay_schedule, pool)
    del replay_schedule
    gc.collect()
    schedule = build_sample_schedule(
        pool,
        densities=DENSITIES,
        repetitions_per_density=REPETITIONS_PER_DENSITY,
        seed=SEED,
    )
    schedule_hash = schedule_sha256(schedule, pool)
    if schedule_hash != replay_schedule_hash:
        raise RuntimeError("matched schedule replay mismatch")

    runs = run_matched_schedule(
        pool,
        schedule,
        calibration_repetitions=CALIBRATION_REPETITIONS,
    )
    augmented, summary = calibration_summary(
        runs,
        densities=DENSITIES,
        repetitions_per_density=REPETITIONS_PER_DENSITY,
        calibration_repetitions=CALIBRATION_REPETITIONS,
    )
    summary.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "config": {
                "pairs": list(DEVELOPMENT_PAIRS),
                "start": START.isoformat(),
                "end_exclusive": END.isoformat(),
                "fee": FEE,
                "seed": SEED,
                "densities": list(DENSITIES),
                "repetitions_per_density": REPETITIONS_PER_DENSITY,
                "calibration_repetitions": CALIBRATION_REPETITIONS,
                "edge_pcts": list(EDGE_PCTS),
                "control_count": CONTROL_COUNT,
                "control_offsets_hours": list(CONTROL_OFFSETS_HOURS),
                "feature_window_hours": FEATURE_WINDOW_HOURS,
            },
            "match_diagnostics": match_diagnostics,
            "schedule_sha256": schedule_hash,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(
        output_dir / "matched-runs.csv",
        index=False,
        float_format="%.12g",
    )
    (output_dir / "matched-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "schedule-sha256.txt").write_text(
        schedule_hash + "\n",
        encoding="utf-8",
    )
    (output_dir / "control-map-sha256.txt").write_text(
        match_diagnostics["control_map_sha256"] + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_calibration(args.data_dir, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
