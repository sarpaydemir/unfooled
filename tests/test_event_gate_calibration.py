from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research_factory.event_gate_calibration import (
    build_event_pool,
    build_sample_schedule,
    calibration_summary,
    power_requirement,
    purged_power_statistics,
    run_schedule,
    schedule_sha256,
)


def synthetic_frame(pair_offset: float = 0.0, periods: int = 900) -> pd.DataFrame:
    date = pd.date_range("2020-01-01", periods=periods, freq="1h", tz="UTC")
    trend = np.linspace(0.0, 4.0, periods)
    cycle = np.sin(np.arange(periods) / 17.0)
    close = 100.0 + pair_offset + trend + cycle
    return pd.DataFrame(
        {
            "date": date,
            "open": close - 0.05,
            "high": close + 0.7,
            "low": close - 0.7,
            "close": close,
            "volume": np.full(periods, 1000.0),
        }
    )


class EventGateCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = {
            "BTC": synthetic_frame(0.0),
            "ETH": synthetic_frame(5.0),
            "SOL": synthetic_frame(10.0),
        }
        self.pool = build_event_pool(self.frames, fee=0.001, edge_pct=0.5)

    def test_event_pool_pairs_null_and_linear_edge_paths(self) -> None:
        self.assertFalse(self.pool.empty)
        required = {
            "event_id",
            "pair",
            "date",
            "return_48h_pct",
            "mfe_48h_pct",
            "mae_48h_pct",
            "edge_return_48h_pct",
            "edge_mfe_48h_pct",
            "edge_mae_48h_pct",
        }
        self.assertTrue(required.issubset(self.pool.columns))
        lift = self.pool["edge_return_48h_pct"] - self.pool["return_48h_pct"]
        self.assertTrue((lift > 0).all())
        self.assertGreater(float(lift.median()), 0.49)
        self.assertLess(float(lift.median()), 0.51)
        self.assertEqual(self.pool["event_id"].nunique(), len(self.pool))

    def test_event_pool_excludes_windows_crossing_timestamp_gaps(self) -> None:
        frame = synthetic_frame(periods=140).drop(index=50).reset_index(drop=True)
        pool = build_event_pool({"BTC": frame}, fee=0.001, edge_pct=0.5)
        duration = pool["target_date"] - pool["signal_date"]
        self.assertTrue((duration == pd.Timedelta(hours=48)).all())
        self.assertLess(len(pool), len(frame) - 48)

    def test_schedule_is_exact_declustered_and_replayable(self) -> None:
        first = build_sample_schedule(
            self.pool,
            densities=(12, 20),
            repetitions_per_density=3,
            seed=76076,
        )
        second = build_sample_schedule(
            self.pool,
            densities=(12, 20),
            repetitions_per_density=3,
            seed=76076,
        )
        self.assertEqual(schedule_sha256(first, self.pool), schedule_sha256(second, self.pool))
        self.assertEqual(len(first), 6)
        for sample in first:
            self.assertEqual(len(sample.event_indices), sample.density)
            selected = self.pool.iloc[list(sample.event_indices)]
            for _, group in selected.groupby("pair"):
                ordered = group["date"].sort_values()
                if len(ordered) > 1:
                    gaps = ordered.diff().dropna()
                    self.assertTrue((gaps >= pd.Timedelta(hours=48)).all())

    def test_schedule_runs_paired_null_and_edge_metrics(self) -> None:
        schedule = build_sample_schedule(
            self.pool,
            densities=(12, 20),
            repetitions_per_density=3,
            seed=76076,
        )
        runs = run_schedule(
            self.pool,
            schedule,
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-02-07", tz="UTC"),
        )
        self.assertEqual(len(runs), 6)
        self.assertEqual(runs["sample_sha256"].nunique(), 6)
        self.assertTrue(np.isfinite(runs.select_dtypes(include="number")).all().all())
        self.assertTrue((runs["edge_mean_return_pct"] > runs["null_mean_return_pct"]).all())
        self.assertTrue(
            (runs["edge_median_return_pct"] > runs["null_median_return_pct"]).all()
        )

    def test_summary_applies_preregistered_false_reject_rules(self) -> None:
        schedule = build_sample_schedule(
            self.pool,
            densities=(12,),
            repetitions_per_density=4,
            seed=76076,
        )
        runs = run_schedule(
            self.pool,
            schedule,
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-02-07", tz="UTC"),
        )
        summary = calibration_summary(
            runs,
            available_12h_clusters=100,
            densities=(12,),
            repetitions_per_density=4,
        )
        self.assertEqual(summary["run_count"], 4)
        self.assertEqual(summary["density_counts"], {"12": 4})
        self.assertIn("median_return", summary["next_campaign_decisions"])
        self.assertIn("mfe_mae_ratio", summary["next_campaign_decisions"])
        self.assertEqual(
            summary["decision_rules"]["max_injected_false_reject_rate"],
            0.20,
        )

    def test_summary_demotes_overstrict_or_nondiscriminating_conditions(self) -> None:
        schedule = build_sample_schedule(
            self.pool,
            densities=(12,),
            repetitions_per_density=5,
            seed=76076,
        )
        runs = run_schedule(
            self.pool,
            schedule,
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-02-07", tz="UTC"),
        )
        runs.loc[:1, "edge_median_return_pass"] = False
        runs["edge_mfe_mae_ratio_pass"] = True
        runs["null_mfe_mae_ratio"] = 1.25
        summary = calibration_summary(
            runs,
            available_12h_clusters=100,
            densities=(12,),
            repetitions_per_density=5,
        )
        decisions = summary["next_campaign_decisions"]
        self.assertFalse(decisions["median_return"]["hard_gate_eligible_next_campaign"])
        self.assertFalse(
            decisions["mfe_mae_ratio"]["hard_gate_eligible_next_campaign"]
        )
        self.assertFalse(decisions["mfe_mae_ratio"]["null_p99_below_1_20"])

    def test_worst_density_cannot_be_hidden_by_pooled_false_reject_rate(self) -> None:
        schedule = build_sample_schedule(
            self.pool,
            densities=(12, 20),
            repetitions_per_density=5,
            seed=76076,
        )
        runs = run_schedule(
            self.pool,
            schedule,
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-02-07", tz="UTC"),
        )
        runs["edge_median_return_pass"] = True
        low_density = runs.index[runs["density"] == 12][:2]
        runs.loc[low_density, "edge_median_return_pass"] = False
        summary = calibration_summary(
            runs,
            available_12h_clusters=100,
            densities=(12, 20),
            repetitions_per_density=5,
        )
        decision = summary["next_campaign_decisions"]["median_return"]
        self.assertAlmostEqual(decision["injected_false_reject_rate"], 0.20)
        self.assertAlmostEqual(decision["worst_density_false_reject_rate"], 0.40)
        self.assertFalse(decision["hard_gate_eligible_next_campaign"])

    def test_stricter_alpha_requires_more_clusters(self) -> None:
        current = power_requirement(
            cluster_sd_pct=9.0,
            alpha=0.05 / 180,
            edge_pct=0.5,
            power=0.80,
        )
        future = power_requirement(
            cluster_sd_pct=9.0,
            alpha=0.05 / 30,
            edge_pct=0.5,
            power=0.80,
        )
        self.assertGreater(current["required_clusters"], future["required_clusters"])
        self.assertGreater(current["minimum_detectable_mean_pct_at_n_300"], 0.5)

    def test_power_units_are_purged_beyond_the_forward_horizon(self) -> None:
        dates = pd.date_range("2020-01-01", periods=25, freq="12h", tz="UTC")
        events = pd.DataFrame(
            {
                "date": dates,
                "return_48h_pct": np.linspace(-2.0, 2.0, len(dates)),
            }
        )
        stats = purged_power_statistics(events)
        self.assertEqual(stats["phase_counts"], {"0": 5, "1": 5, "2": 5, "3": 5, "4": 5})
        self.assertEqual(stats["max_independent_clusters"], 5)
        self.assertGreater(stats["conservative_sd_pct"], 0)


if __name__ == "__main__":
    unittest.main()
