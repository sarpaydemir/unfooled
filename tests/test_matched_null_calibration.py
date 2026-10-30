from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_factory.event_gate_calibration import build_sample_schedule
from research_factory.matched_null_calibration import (
    _pair_matches,
    build_matched_event_pool,
    calibration_summary,
    run_matched_schedule,
    trailing_features,
    wilson_bound,
)


def synthetic_frame(
    pair_offset: float = 0.0,
    *,
    periods: int = 1400,
) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=periods, freq="1h", tz="UTC")
    cycle = np.sin(np.arange(periods) / 19.0)
    slow = np.linspace(0.0, 6.0, periods)
    close = 100.0 + pair_offset + slow + cycle
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.05,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.full(periods, 1000.0),
        }
    )


class MatchedNullCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = {
            "BTC": synthetic_frame(0.0),
            "ETH": synthetic_frame(5.0),
            "SOL": synthetic_frame(10.0),
        }
        self.pool, self.diagnostics = build_matched_event_pool(
            self.frames,
            fee=0.001,
            edge_pcts=(0.5, 1.5),
            offsets_hours=(-288, -240, -192, -144, -96, -72),
            control_count=2,
            feature_window_hours=24,
            minimum_overall_coverage=0.50,
            minimum_pair_coverage=0.50,
        )

    def test_trailing_features_use_only_closed_contiguous_history(self) -> None:
        frame = synthetic_frame(periods=260)
        first = trailing_features(frame, window_hours=24)
        mutated = frame.copy()
        mutated.loc[200:, "close"] *= 10
        second = trailing_features(mutated, window_hours=24)
        pd.testing.assert_series_equal(
            first.loc[:199, "return_7d"],
            second.loc[:199, "return_7d"],
        )
        one_hour_gap = frame.drop(index=100).reset_index(drop=True)
        features = trailing_features(one_hour_gap, window_hours=24)
        target = pd.Timestamp("2020-01-05 14:00:00+00:00")
        value = features.loc[features["signal_date"] == target, "return_7d"]
        self.assertEqual(len(value), 1)
        self.assertTrue(np.isfinite(value).all())
        long_gap = frame.drop(index=range(100, 104)).reset_index(drop=True)
        features = trailing_features(long_gap, window_hours=24)
        target = pd.Timestamp("2020-01-05 18:00:00+00:00")
        value = features.loc[features["signal_date"] == target, "return_7d"]
        self.assertEqual(len(value), 1)
        self.assertTrue(value.isna().all())

    def test_matched_pool_is_replayable_and_pairs_all_edge_arms(self) -> None:
        replay, replay_diagnostics = build_matched_event_pool(
            self.frames,
            fee=0.001,
            edge_pcts=(0.5, 1.5),
            offsets_hours=(-288, -240, -192, -144, -96, -72),
            control_count=2,
            feature_window_hours=24,
            minimum_overall_coverage=0.50,
            minimum_pair_coverage=0.50,
        )
        self.assertFalse(self.pool.empty)
        self.assertEqual(
            self.diagnostics["control_map_sha256"],
            replay_diagnostics["control_map_sha256"],
        )
        self.assertTrue(self.pool["event_id"].equals(replay["event_id"]))
        self.assertTrue((self.pool["control_count"] == 2).all())
        self.assertTrue((self.pool["minimum_control_gap_hours"] >= 72).all())
        lift_0_5 = (
            self.pool["edge_0_5_matched_lift_pct"]
            - self.pool["null_matched_lift_pct"]
        )
        lift_1_5 = (
            self.pool["edge_1_5_matched_lift_pct"]
            - self.pool["null_matched_lift_pct"]
        )
        self.assertTrue((lift_0_5 > 0.49).all())
        self.assertTrue((lift_1_5 > lift_0_5).all())
        self.assertGreater(float(lift_1_5.median()), 1.45)
        self.assertLess(float(lift_1_5.median()), 1.55)

    def test_match_selection_does_not_read_outcome_column(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="72h", tz="UTC")
        group = pd.DataFrame(
            {
                "date": dates,
                "pair_return_7d": np.linspace(-1.0, 1.0, len(dates)),
                "pair_volatility_7d": np.linspace(0.1, 0.2, len(dates)),
                "btc_return_7d": np.linspace(-0.5, 0.5, len(dates)),
                "btc_volatility_7d": np.linspace(0.2, 0.3, len(dates)),
                "return_48h_pct": np.linspace(-2.0, 2.0, len(dates)),
            }
        )
        _, first_controls, first_offsets = _pair_matches(
            group,
            offsets_hours=(-216, -192, -168, -144, -96, -72),
            control_count=2,
        )
        mutated = group.copy()
        mutated["return_48h_pct"] = np.arange(len(group)) ** 3
        _, second_controls, second_offsets = _pair_matches(
            mutated,
            offsets_hours=(-216, -192, -168, -144, -96, -72),
            control_count=2,
        )
        np.testing.assert_array_equal(first_controls, second_controls)
        np.testing.assert_array_equal(first_offsets, second_offsets)

    def test_offset_and_null_lift_diagnostics_are_complete(self) -> None:
        offsets = self.diagnostics["selected_control_offsets"]
        expected = self.diagnostics["matched_event_count"] * 2
        self.assertEqual(offsets["selection_count"], expected)
        self.assertEqual(sum(offsets["histogram"].values()), expected)
        self.assertLessEqual(offsets["maximum_hours"], -72)
        self.assertGreaterEqual(offsets["minimum_hours"], -288)
        self.assertEqual(
            sum(
                values["selected_control_offsets"]["selection_count"]
                for values in self.diagnostics["pair_diagnostics"].values()
            ),
            expected,
        )
        lift = self.diagnostics["null_lift"]
        self.assertEqual(
            lift["interpretation"],
            "empirical_matched_residual_not_beta_neutral",
        )
        self.assertEqual(
            lift["overall"]["events"],
            self.diagnostics["matched_event_count"],
        )
        self.assertEqual(set(lift["by_pair"]), set(self.frames))
        self.assertTrue(lift["by_year"])

    def test_invalid_or_insufficient_control_contract_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            build_matched_event_pool(
                self.frames,
                fee=0.001,
                offsets_hours=(-72, 72),
                control_count=1,
                minimum_overall_coverage=0,
                minimum_pair_coverage=0,
            )
        with self.assertRaises(RuntimeError):
            build_matched_event_pool(
                self.frames,
                fee=0.001,
                offsets_hours=(-1296, -1224, -1152, -1080),
                control_count=4,
                feature_window_hours=24,
                minimum_overall_coverage=0.99,
                minimum_pair_coverage=0.99,
            )

    def test_schedule_runs_three_arms_with_disjoint_splits(self) -> None:
        schedule = build_sample_schedule(
            self.pool,
            densities=(12, 20),
            repetitions_per_density=6,
            seed=77077,
        )
        runs = run_matched_schedule(
            self.pool,
            schedule,
            calibration_repetitions=3,
        )
        self.assertEqual(len(runs), 12)
        self.assertEqual(
            runs.groupby(["density", "split"]).size().to_dict(),
            {
                (12, "calibration"): 3,
                (12, "evaluation"): 3,
                (20, "calibration"): 3,
                (20, "evaluation"): 3,
            },
        )
        self.assertTrue(
            (
                runs["edge_0_5_mean_matched_lift_pct"]
                > runs["null_mean_matched_lift_pct"]
            ).all()
        )
        self.assertTrue(
            (
                runs["edge_1_5_mean_matched_lift_pct"]
                > runs["edge_0_5_mean_matched_lift_pct"]
            ).all()
        )

    def test_wilson_bounds_are_one_sided_and_finite(self) -> None:
        lower_zero, upper_zero = wilson_bound(0, 500)
        lower_all, upper_all = wilson_bound(500, 500)
        self.assertEqual(lower_zero, 0.0)
        self.assertLess(upper_zero, 0.02)
        self.assertGreater(lower_all, 0.75)
        self.assertAlmostEqual(upper_all, 1.0)

    def test_summary_uses_calibration_only_and_selects_simplest_tie(self) -> None:
        rows: list[dict] = []
        for repetition in range(1000):
            split = "calibration" if repetition < 500 else "evaluation"
            null_lift = repetition / 100000 if split == "calibration" else -1.0
            row = {
                "density": 300,
                "repetition": repetition,
                "split": split,
                "sample_sha256": f"{repetition:064x}",
                "effective_events": 100,
            }
            for arm, lift, median in (
                ("null", null_lift, -1.0),
                ("edge_0_5", 1.0, 1.0),
                ("edge_1_5", 2.0, 2.0),
            ):
                row.update(
                    {
                        f"{arm}_mean_return_pct": median,
                        f"{arm}_median_return_pct": median,
                        f"{arm}_mean_matched_lift_pct": lift,
                        f"{arm}_median_matched_lift_pct": lift,
                        f"{arm}_positive_pair_fraction": float(lift > 0),
                        f"{arm}_positive_quarter_fraction": float(lift > 0),
                    }
                )
            rows.append(row)
        runs = pd.DataFrame(rows)
        augmented, summary = calibration_summary(
            runs,
            densities=(300,),
            repetitions_per_density=1000,
            calibration_repetitions=500,
        )
        self.assertEqual(summary["selected_future_variant"], "matched_only")
        self.assertTrue(summary["edge_0_5_resolved"])
        self.assertEqual(
            int(
                augmented.loc[
                    augmented["split"] == "evaluation",
                    "null_matched_only_pass",
                ].sum()
            ),
            0,
        )
        threshold = summary["null_thresholds_pct"]["300"]
        self.assertGreater(threshold, 0)
        self.assertLess(threshold, 0.01)

    def test_summary_rejects_wrong_frozen_counts(self) -> None:
        with self.assertRaises(RuntimeError):
            calibration_summary(
                pd.DataFrame({"density": [300], "split": ["calibration"]}),
                densities=(300,),
                repetitions_per_density=1000,
                calibration_repetitions=500,
            )


if __name__ == "__main__":
    unittest.main()
