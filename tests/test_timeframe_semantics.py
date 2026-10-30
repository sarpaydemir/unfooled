import unittest

import numpy as np
import pandas as pd


def aligned_markers(index: pd.DatetimeIndex, rule: str, delay_bars: int) -> pd.Series:
    raw = pd.Series(np.arange(len(index), dtype=float), index=index)
    aggregate = raw.resample(rule, label="right", closed="left").last()
    return aggregate.shift(delay_bars).reindex(index, method="ffill")


class TimeframeSemanticsTests(unittest.TestCase):
    def test_four_hour_golden_availability_for_zero_one_two_delays(self) -> None:
        index = pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC")
        expected = {
            0: (pd.Timestamp("2024-01-01 04:00", tz="UTC"), 3.0),
            1: (pd.Timestamp("2024-01-01 08:00", tz="UTC"), 3.0),
            2: (pd.Timestamp("2024-01-01 12:00", tz="UTC"), 3.0),
        }
        for delay, (first_seen, marker) in expected.items():
            with self.subTest(delay=delay):
                aligned = aligned_markers(index, "4h", delay)
                self.assertEqual(aligned.loc[first_seen], marker)
                self.assertTrue(aligned.loc[: first_seen - pd.Timedelta(hours=1)].ne(marker).all())

    def test_daily_one_bar_delay_first_exposes_jan_one_on_jan_three(self) -> None:
        index = pd.date_range("2024-01-01", periods=96, freq="1h", tz="UTC")
        aligned = aligned_markers(index, "1D", 1)
        self.assertTrue(pd.isna(aligned.loc[pd.Timestamp("2024-01-02 23:00", tz="UTC")]))
        self.assertEqual(aligned.loc[pd.Timestamp("2024-01-03 00:00", tz="UTC")], 23.0)

    def test_non_utc_daily_bins_have_different_boundaries(self) -> None:
        utc_index = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
        values = pd.Series(np.arange(len(utc_index), dtype=float), index=utc_index)
        utc_close = values.resample("1D", label="right", closed="left").last().iloc[0]
        istanbul = values.copy()
        istanbul.index = istanbul.index.tz_convert("Europe/Istanbul")
        istanbul_close = istanbul.resample("1D", label="right", closed="left").last().iloc[0]
        self.assertEqual(utc_close, 23.0)
        self.assertEqual(istanbul_close, 20.0)
        self.assertNotEqual(utc_close, istanbul_close)


if __name__ == "__main__":
    unittest.main()
