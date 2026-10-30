from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research_factory.event_study import (
    load_candidate,
    merge_closed_informative,
    pair_events,
    summarize,
)


class EventStudyTests(unittest.TestCase):
    def test_candidate_requires_matching_v2_manifest(self) -> None:
        source = (
            "def signal(dataframe):\n"
            "    return dataframe['close'].gt(0).fillna(False)\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            missing = Path(directory) / "missing-manifest.json"
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                load_candidate(candidate, missing)

            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "candidate_code_sha256": "0" * 64,
                        "hypothesis": {
                            "hypothesis_id": "event-fixture",
                            "mechanism_ids": ["fixture"],
                            "title": "Event fixture",
                            "rationale": "Manifest binding test.",
                            "expected_regimes": ["fixture"],
                            "entry_logic": ["Synthetic signal."],
                            "exit_neutral_horizons": [48],
                            "controls": ["No market data."],
                            "failure_conditions": ["Hash mismatch."],
                            "required_timeframes": ["1h"],
                            "schema_version": 2,
                            "indicator_contracts": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertNotEqual(
                json.loads(manifest.read_text())["candidate_code_sha256"],
                sha256(source.encode()).hexdigest(),
            )
            with self.assertRaisesRegex(RuntimeError, "code SHA"):
                load_candidate(candidate, manifest)

    def test_zero_events_are_recorded_as_a_gate_failure_artifact(self) -> None:
        metrics = summarize(
            pd.DataFrame(),
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-02-01", tz="UTC"),
        )
        self.assertEqual(metrics["raw_events"], 0)
        self.assertEqual(metrics["effective_events"], 0)
        self.assertEqual(metrics["median_return_pct"], 0.0)

    def test_informative_candle_is_not_visible_before_close(self) -> None:
        base = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
                "close": [1, 1, 1, 1, 1],
            }
        )
        informative = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T00:00:00Z"]),
                "close": [99],
            }
        )
        merged = merge_closed_informative(base, informative, "4h")
        self.assertTrue(merged.loc[:2, "close_4h"].isna().all())
        self.assertEqual(merged.loc[3, "close_4h"], 99)

    def test_pair_events_are_declustered(self) -> None:
        dates = pd.date_range("2024-01-01", periods=120, freq="1h", tz="UTC")
        close = np.linspace(100, 120, len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1,
            }
        )
        events = pair_events(frame, "BTC", lambda data: pd.Series(True, index=data.index), 0.001)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1]["date"] - events[0]["date"], pd.Timedelta(hours=48))

    def test_entry_uses_next_candle_open(self) -> None:
        dates = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": np.arange(60, dtype=float) + 100,
                "high": np.arange(60, dtype=float) + 102,
                "low": np.arange(60, dtype=float) + 99,
                "close": np.arange(60, dtype=float) + 101,
                "volume": 1,
            }
        )
        setup = lambda data: pd.Series(data["date"] == dates[0], index=data.index)
        event = pair_events(frame, "BTC", setup, 0.001)[0]
        self.assertEqual(event["signal_date"], dates[0])
        self.assertEqual(event["date"], dates[1])
        self.assertEqual(event["entry"], frame.iloc[1]["open"])

    def test_candidate_receives_datetime_index(self) -> None:
        dates = pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1.0,
            }
        )

        def setup(data: pd.DataFrame) -> pd.Series:
            self.assertIsInstance(data.index, pd.DatetimeIndex)
            self.assertTrue(data.index.equals(pd.DatetimeIndex(data["date"])))
            return pd.Series(False, index=data.index)

        self.assertEqual(pair_events(frame, "BTC", setup, 0.001), [])

    def test_summary_counts_cross_pair_market_clusters(self) -> None:
        events = pd.DataFrame(
            {
                "pair": ["BTC", "ETH", "BTC", "ETH"],
                "date": pd.to_datetime(
                    ["2024-01-01T01:00Z", "2024-01-01T02:00Z",
                     "2024-01-03T01:00Z", "2024-01-03T02:00Z"]
                ),
                "return_48h_pct": [1, 2, 1, 2],
                "mfe_48h_pct": [2, 2, 2, 2],
                "mae_48h_pct": [-1, -1, -1, -1],
            }
        )
        metrics = summarize(
            events,
            pd.Timestamp("2024-01-01", tz="UTC"),
            pd.Timestamp("2024-01-05", tz="UTC"),
        )
        self.assertEqual(metrics["raw_events"], 4)
        self.assertEqual(metrics["effective_events"], 2)


if __name__ == "__main__":
    unittest.main()
