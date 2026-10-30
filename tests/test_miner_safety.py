from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from research_factory.miner import MechanismMiner
from research_factory.safety import CandidateSafetyValidator


NFI_LIKE = """
def populate_entry_trend(dataframe):
    for enabled in []:
        long_entry_condition_index = 7
        if long_entry_condition_index == 7:
            logic = (dataframe['willr_1h'] < -80) & (dataframe['cmf_4h'] > 0)
    return dataframe

def buy_quick_5m(dataframe):
    condition = (dataframe['willr'] < -80) & (dataframe['ewo'] > 2)
    return condition

def buy_grind(dataframe):
    return dataframe['rsi'] < 20

def entry_plain(dataframe):
    return (dataframe['rsi'] < 30) & (dataframe['obv'] > 0)
"""


class MinerSafetyTests(unittest.TestCase):
    def test_external_framework_column_is_rejected_through_alias(self) -> None:
        source = """
def signal(dataframe):
    df = dataframe.copy()
    return df["protections_long_global"] & (df["close"] > 0)
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertFalse(report.passed)
        self.assertTrue(
            any("undeclared raw dataframe column 'protections_long_global'" in item
                for item in report.violations)
        )

    def test_candidate_derived_column_can_be_read_after_assignment(self) -> None:
        source = """
def signal(dataframe):
    df = dataframe.copy()
    df["derived_reclaim"] = df["close"] > df["open"]
    return df["derived_reclaim"]
"""
        self.assertTrue(CandidateSafetyValidator().validate(source).passed)

    def test_named_aggregation_output_is_a_derived_column(self) -> None:
        source = """
def signal(dataframe):
    grouped = dataframe.groupby(dataframe.index.floor("4h"))
    aggregate = grouped.agg(
        close=("close", "last"),
        count=("close", "count"),
    )
    complete = aggregate.loc[aggregate["count"] == 4]
    return dataframe["close"] > complete["close"].reindex(dataframe.index).ffill()
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertTrue(report.passed, report.violations)

    def test_named_aggregation_external_source_is_rejected(self) -> None:
        source = """
def signal(dataframe):
    grouped = dataframe.groupby(dataframe.index.floor("4h"))
    aggregate = grouped.agg(regime=("framework_regime", "last"))
    return aggregate["regime"].reindex(dataframe.index).ffill() == "risk_on"
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertFalse(report.passed)
        self.assertTrue(
            any("named-aggregation source column 'framework_regime'" in item
                for item in report.violations)
        )

    def test_dynamic_raw_column_access_is_rejected(self) -> None:
        source = """
def signal(dataframe):
    column = "close"
    return dataframe[column] > 0
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertFalse(report.passed)
        self.assertTrue(any("dynamic raw dataframe column" in item for item in report.violations))

    def test_signal_scalar_bool_returns_are_rejected(self) -> None:
        for returned in ("False", "bool(dataframe['close'].iloc[-1] > 0)"):
            source = f"""
def signal(dataframe):
    return {returned}
"""
            with self.subTest(returned=returned):
                report = CandidateSafetyValidator().validate(source)
                self.assertFalse(report.passed)
                self.assertTrue(
                    any("index-aligned boolean Series" in item for item in report.violations)
                )

    def test_helper_scalar_and_signal_series_return_are_allowed(self) -> None:
        source = """
def helper(value):
    return bool(value)

def signal(dataframe):
    setup = dataframe["close"] > dataframe["open"]
    return setup.fillna(False).astype(bool)
"""
        self.assertTrue(CandidateSafetyValidator().validate(source).passed)

    def test_signal_iloc_return_is_rejected(self) -> None:
        source = """
def signal(dataframe):
    setup = dataframe["close"] > dataframe["open"]
    return setup.iloc[-1]
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertFalse(report.passed)
        self.assertTrue(any("must not scalarize" in item for item in report.violations))

    def test_constant_ohlcv_selector_is_allowed(self) -> None:
        source = """
def signal(dataframe):
    required = ["open", "high", "low", "close", "volume"]
    frame = dataframe[required].copy()
    return (frame["close"] > frame["open"]).fillna(False).astype(bool)
"""
        self.assertTrue(CandidateSafetyValidator().validate(source).passed)

    def test_constant_selector_with_external_column_is_rejected(self) -> None:
        source = """
def signal(dataframe):
    required = ["open", "close", "regime"]
    frame = dataframe[required]
    return frame["close"] > frame["open"]
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertFalse(report.passed)
        self.assertTrue(any("undeclared raw dataframe column 'regime'" in item
                            for item in report.violations))

    def test_reassigned_or_runtime_built_selector_is_rejected(self) -> None:
        sources = (
            """
def signal(dataframe):
    required = ["open", "close"]
    required = make_columns()
    return dataframe[required]["close"] > 0
""",
            """
def signal(dataframe):
    required = ["open"] + ["close"]
    return dataframe[required]["close"] > 0
""",
        )
        for source in sources:
            with self.subTest(source=source):
                report = CandidateSafetyValidator().validate(source)
                self.assertFalse(report.passed)
                self.assertTrue(any("dynamic raw dataframe column" in item
                                    for item in report.violations))

    def test_miner_rejects_scalp_grind_and_exhausted_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "NFI.py"
            path.write_text(NFI_LIKE)
            checksum = sha256(NFI_LIKE.encode()).hexdigest()
            result = MechanismMiner("repo", "abc123", checksum).mine(path)
            self.assertEqual(len(result), 4)
            reasons = {item.name: item.rejected_reason for item in result}
            self.assertIsNone(reasons["long_entry_condition_7"])
            self.assertIsNotNone(reasons["buy_quick_5m"])
            self.assertIsNotNone(reasons["buy_grind"])
            self.assertIsNotNone(reasons["entry_plain"])

    def test_checksum_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "NFI.py"
            path.write_text(NFI_LIKE)
            with self.assertRaises(ValueError):
                MechanismMiner("repo", "abc123", "wrong").mine(path)

    def test_safety_rejects_future_network_and_shell(self) -> None:
        source = """
import requests
import subprocess
def signal(dataframe):
    leaked = dataframe['close'].shift(-1)
    return leaked
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertFalse(report.passed)
        self.assertTrue(any("negative shift" in item for item in report.violations))
        self.assertTrue(any("requests" in item for item in report.violations))

    def test_safety_accepts_causal_dataframe_code(self) -> None:
        source = """
import pandas as pd
def signal(dataframe):
    prior = dataframe['high'].rolling(20).max().shift(1)
    return dataframe['close'] > prior
"""
        self.assertTrue(CandidateSafetyValidator().validate(source).passed)

    def test_returned_signal_cannot_shift_execution(self) -> None:
        source = """
def signal(dataframe):
    setup = dataframe['close'] > 0
    return setup.shift(1, fill_value=False).astype(bool)
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertFalse(report.passed)
        self.assertTrue(any("must not shift" in item for item in report.violations))

    def test_helper_may_return_a_causal_positive_shift(self) -> None:
        source = """
def completed_higher_timeframe(dataframe):
    return dataframe.shift(1)

def signal(dataframe):
    prior = completed_higher_timeframe(dataframe)
    return prior['close'] > 0
"""
        report = CandidateSafetyValidator().validate(source)
        self.assertTrue(report.passed, report.violations)

    def test_string_join_is_not_dataframe_join(self) -> None:
        source = """
def signal(dataframe):
    missing = {'close'}.difference(dataframe.columns)
    if missing:
        raise ValueError(', '.join(sorted(missing)))
    return dataframe['close'] > 0
"""
        self.assertTrue(CandidateSafetyValidator().validate(source).passed)

    def test_dataframe_join_remains_forbidden(self) -> None:
        source = """
def signal(dataframe):
    joined = dataframe.join(dataframe)
    return joined['close'] > 0
"""
        self.assertFalse(CandidateSafetyValidator().validate(source).passed)


if __name__ == "__main__":
    unittest.main()
