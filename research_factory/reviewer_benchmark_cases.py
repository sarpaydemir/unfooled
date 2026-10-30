from __future__ import annotations

from hashlib import sha256
from typing import Any

from .models import canonical_json


def _hypothesis(
    case_id: str,
    title: str,
    entry_logic: list[str],
    timeframes: list[str],
) -> dict[str, Any]:
    return {
        "hypothesis_id": f"benchmark-{case_id}",
        "mechanism_ids": [f"benchmark-{case_id}"],
        "title": title,
        "rationale": "Frozen semantic reviewer calibration fixture.",
        "expected_regimes": ["benchmark-only"],
        "entry_logic": entry_logic,
        "exit_neutral_horizons": [48],
        "controls": ["No performance data; semantic comparison only."],
        "failure_conditions": ["Any material manifest-code mismatch."],
        "required_timeframes": timeframes,
    }


def _case(
    case_id: str,
    expected: str,
    title: str,
    entry_logic: list[str],
    timeframes: list[str],
    code: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "expected": expected,
        "hypothesis": _hypothesis(case_id, title, entry_logic, timeframes),
        "code": code.strip() + "\n",
        "ground_truth_evidence": evidence,
    }


RSI_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    rsi = pd.Series(
        talib.RSI(dataframe["close"].to_numpy(dtype=float), timeperiod=14),
        index=dataframe.index,
    )
    return rsi.lt(30.0).fillna(False).astype(bool)
"""

EMA_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    close = dataframe["close"].to_numpy(dtype=float)
    fast = pd.Series(talib.EMA(close, timeperiod=12), index=dataframe.index)
    slow = pd.Series(talib.EMA(close, timeperiod=26), index=dataframe.index)
    return fast.gt(slow).fillna(False).astype(bool)
"""

BBANDS_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    close = dataframe["close"].to_numpy(dtype=float)
    upper, middle, lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0
    )
    width = pd.Series(100.0 * (upper - lower) / middle, index=dataframe.index)
    return width.gt(8.0).fillna(False).astype(bool)
"""

WILLR_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    willr = pd.Series(
        talib.WILLR(
            dataframe["high"].to_numpy(dtype=float),
            dataframe["low"].to_numpy(dtype=float),
            dataframe["close"].to_numpy(dtype=float),
            timeperiod=14,
        ),
        index=dataframe.index,
    )
    return willr.lt(-80.0).fillna(False).astype(bool)
"""

ROC_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    roc = pd.Series(
        talib.ROC(dataframe["close"].to_numpy(dtype=float), timeperiod=9),
        index=dataframe.index,
    )
    return roc.gt(2.0).fillna(False).astype(bool)
"""

STOCHRSI_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    _, fastd = talib.STOCHRSI(
        dataframe["close"].to_numpy(dtype=float),
        timeperiod=14,
        fastk_period=14,
        fastd_period=3,
        fastd_matype=0,
    )
    fastd_series = pd.Series(fastd, index=dataframe.index)
    return fastd_series.lt(20.0).fillna(False).astype(bool)
"""

AROON_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    aroondown, _ = talib.AROON(
        dataframe["high"].to_numpy(dtype=float),
        dataframe["low"].to_numpy(dtype=float),
        timeperiod=14,
    )
    down = pd.Series(aroondown, index=dataframe.index)
    return down.gt(70.0).fillna(False).astype(bool)
"""

HTF_HELPERS = """
def _completed_4h(dataframe):
    grouped = dataframe.resample("4h", label="left", closed="left")
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    counts = grouped["close"].count()
    bars = bars.loc[counts.eq(4)].copy()
    bars.index = bars.index + pd.Timedelta(hours=4)
    return bars

def _available_on_1h(series, index):
    delayed = series.copy()
    delayed.index = delayed.index + pd.Timedelta(hours=4)
    return delayed.reindex(index, method="ffill")
"""

HTF_RSI_PASS = f"""
import pandas as pd
import talib

{HTF_HELPERS}

def signal(dataframe):
    bars = _completed_4h(dataframe)
    rsi = pd.Series(
        talib.RSI(bars["close"].to_numpy(dtype=float), timeperiod=14),
        index=bars.index,
    )
    aligned = _available_on_1h(rsi, dataframe.index)
    return aligned.lt(30.0).fillna(False).astype(bool)
"""

HTF_RSI_AFTER_FILL_FAIL = f"""
import pandas as pd
import talib

{HTF_HELPERS}

def signal(dataframe):
    bars = _completed_4h(dataframe)
    repeated_close = bars["close"].reindex(dataframe.index, method="ffill")
    rsi = pd.Series(
        talib.RSI(repeated_close.to_numpy(dtype=float), timeperiod=14),
        index=dataframe.index,
    )
    return rsi.lt(30.0).fillna(False).astype(bool)
"""

HTF_NO_DELAY_FAIL = HTF_RSI_PASS.replace(
    "delayed.index = delayed.index + pd.Timedelta(hours=4)",
    "delayed.index = delayed.index + pd.Timedelta(hours=0)",
)

RIGHT_LABEL_GAP_PASS = """
import pandas as pd
import talib

def signal(dataframe):
    grouped = dataframe.resample("4h", label="right", closed="left")
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    counts = grouped["close"].count()
    bars = bars.loc[counts.eq(4)].copy()
    rsi = pd.Series(
        talib.RSI(bars["close"].to_numpy(dtype=float), timeperiod=14),
        index=bars.index,
    )
    aligned = rsi.shift(1).reindex(dataframe.index, method="ffill")
    return aligned.lt(30.0).fillna(False).astype(bool)
"""


def benchmark_cases() -> list[dict[str, Any]]:
    cases = [
        _case(
            "P01-RSI-EXACT",
            "pass",
            "TA-Lib RSI oversold",
            ["TA-Lib RSI(close, timeperiod=14) output is below 30."],
            ["1h"],
            RSI_PASS,
            "Exact TA-Lib function, keyword, threshold direction and output.",
        ),
        _case(
            "N01-RSI-PERIOD",
            "reject",
            "TA-Lib RSI oversold",
            ["TA-Lib RSI(close, timeperiod=14) output is below 30."],
            ["1h"],
            RSI_PASS.replace("timeperiod=14", "timeperiod=13"),
            "Single material mutation: code period 13 contradicts manifest period 14.",
        ),
        _case(
            "P02-EMA-DIRECTION",
            "pass",
            "TA-Lib EMA trend",
            [
                "TA-Lib EMA(close, timeperiod=12) is above "
                "TA-Lib EMA(close, timeperiod=26)."
            ],
            ["1h"],
            EMA_PASS,
            "Exact EMA periods and comparison direction.",
        ),
        _case(
            "N02-EMA-DIRECTION",
            "reject",
            "TA-Lib EMA trend",
            [
                "TA-Lib EMA(close, timeperiod=12) is above "
                "TA-Lib EMA(close, timeperiod=26)."
            ],
            ["1h"],
            EMA_PASS.replace("fast.gt(slow)", "fast.lt(slow)"),
            "Single material mutation: below replaces manifest above.",
        ),
        _case(
            "P03-BBANDS-WIDTH",
            "pass",
            "Canonical TA-Lib Bollinger bandwidth",
            [
                "TA-Lib BBANDS(close, timeperiod=20, nbdevup=2.0, "
                "nbdevdn=2.0, matype=0); use 100*(upper-lower)/middle above 8."
            ],
            ["1h"],
            BBANDS_PASS,
            "Canonical full bandwidth uses both upper and lower outputs.",
        ),
        _case(
            "N03-BBANDS-HALF-WIDTH",
            "reject",
            "Canonical TA-Lib Bollinger bandwidth",
            [
                "TA-Lib BBANDS(close, timeperiod=20, nbdevup=2.0, "
                "nbdevdn=2.0, matype=0); use 100*(upper-lower)/middle above 8."
            ],
            ["1h"],
            BBANDS_PASS.replace(
                "100.0 * (upper - lower) / middle",
                "100.0 * (upper - middle) / middle",
            ),
            "Single material mutation: half-bandwidth replaces full bandwidth.",
        ),
        _case(
            "P04-WILLR-DIRECTION",
            "pass",
            "TA-Lib Williams percent R",
            ["TA-Lib WILLR(high, low, close, timeperiod=14) output is below -80."],
            ["1h"],
            WILLR_PASS,
            "Exact -100..0 scale and oversold direction.",
        ),
        _case(
            "N04-WILLR-DIRECTION",
            "reject",
            "TA-Lib Williams percent R",
            ["TA-Lib WILLR(high, low, close, timeperiod=14) output is below -80."],
            ["1h"],
            WILLR_PASS.replace("willr.lt(-80.0)", "willr.gt(-20.0)"),
            "Single material mutation: overbought direction replaces oversold.",
        ),
        _case(
            "P05-ROC-PERIOD",
            "pass",
            "TA-Lib rate of change",
            ["TA-Lib ROC(close, timeperiod=9) percentage output is above 2."],
            ["1h"],
            ROC_PASS,
            "Exact period, percentage scale and threshold direction.",
        ),
        _case(
            "N05-ROC-PERIOD",
            "reject",
            "TA-Lib rate of change",
            ["TA-Lib ROC(close, timeperiod=9) percentage output is above 2."],
            ["1h"],
            ROC_PASS.replace("timeperiod=9", "timeperiod=8"),
            "Single material mutation: code period 8 contradicts manifest period 9.",
        ),
        _case(
            "P06-STOCHRSI-FASTD",
            "pass",
            "TA-Lib STOCHRSI fast-D",
            [
                "TA-Lib STOCHRSI(close, timeperiod=14, fastk_period=14, "
                "fastd_period=3, fastd_matype=0); use returned fast-D below 20."
            ],
            ["1h"],
            STOCHRSI_PASS,
            "Exact keywords and returned fast-D output.",
        ),
        _case(
            "N06-STOCHRSI-MATYPE",
            "reject",
            "TA-Lib STOCHRSI fast-D",
            [
                "TA-Lib STOCHRSI(close, timeperiod=14, fastk_period=14, "
                "fastd_period=3, fastd_matype=0); use returned fast-D below 20."
            ],
            ["1h"],
            STOCHRSI_PASS.replace("fastd_matype=0", "fastd_matype=3"),
            "Single material mutation changes the consumed fast-D smoothing.",
        ),
        _case(
            "P07-AROON-OUTPUT",
            "pass",
            "TA-Lib AROON down",
            [
                "TA-Lib AROON(high, low, timeperiod=14); use returned "
                "aroondown output above 70."
            ],
            ["1h"],
            AROON_PASS,
            "TA-Lib returns aroondown first; code consumes that output.",
        ),
        _case(
            "N07-AROON-OUTPUT",
            "reject",
            "TA-Lib AROON down",
            [
                "TA-Lib AROON(high, low, timeperiod=14); use returned "
                "aroondown output above 70."
            ],
            ["1h"],
            AROON_PASS.replace("aroondown, _ =", "_, aroondown ="),
            "Single material mutation consumes aroonup as if it were aroondown.",
        ),
        _case(
            "P08-HTF-ORDER",
            "pass",
            "Completed 4h TA-Lib RSI",
            [
                "Aggregate distinct completed 4h OHLCV bars, compute TA-Lib "
                "RSI(close, timeperiod=14), delay availability by one 4h bar, "
                "then align the RSI result to 1h; use below 30."
            ],
            ["1h", "4h"],
            HTF_RSI_PASS,
            "Indicator is computed on distinct aggregate bars before result alignment.",
        ),
        _case(
            "N08-HTF-ORDER",
            "reject",
            "Completed 4h TA-Lib RSI",
            [
                "Aggregate distinct completed 4h OHLCV bars, compute TA-Lib "
                "RSI(close, timeperiod=14), delay availability by one 4h bar, "
                "then align the RSI result to 1h; use below 30."
            ],
            ["1h", "4h"],
            HTF_RSI_AFTER_FILL_FAIL,
            "Material mutation forward-fills raw 4h close before RSI calculation.",
        ),
        _case(
            "P09-HTF-AVAILABILITY",
            "pass",
            "One completed-bar 4h availability delay",
            [
                "For completed [00:00,04:00) 4h RSI, first availability is "
                "08:00 (T1 + one 4h period), then align to 1h."
            ],
            ["1h", "4h"],
            HTF_RSI_PASS,
            "Golden project anchor [00:00,04:00) -> 08:00.",
        ),
        _case(
            "N09-HTF-NO-DELAY",
            "reject",
            "One completed-bar 4h availability delay",
            [
                "For completed [00:00,04:00) 4h RSI, first availability is "
                "08:00 (T1 + one 4h period), then align to 1h."
            ],
            ["1h", "4h"],
            HTF_NO_DELAY_FAIL,
            "Single material mutation exposes T1 result without required D=1 delay.",
        ),
        _case(
            "P10-RIGHT-LABEL-GAPS",
            "pass",
            "Right-label completed 4h RSI with dropped incomplete bars",
            [
                "Discard incomplete 4h aggregates; TA-Lib RSI(close, timeperiod=14) "
                "uses the remaining ordered numeric observations. A right label names "
                "T1 and shift(1) supplies D=1, so [00:00,04:00) appears at 08:00. "
                "Return same-row setup for evaluator next-open entry."
            ],
            ["1h", "4h"],
            RIGHT_LABEL_GAP_PASS,
            "Known-correct cases from EXP-056 and EXP-064 are combined.",
        ),
        _case(
            "N10-ENTRY-SHIFT",
            "reject",
            "Right-label completed 4h RSI with dropped incomplete bars",
            [
                "Discard incomplete 4h aggregates; TA-Lib RSI(close, timeperiod=14) "
                "uses the remaining ordered numeric observations. A right label names "
                "T1 and shift(1) supplies D=1, so [00:00,04:00) appears at 08:00. "
                "Return same-row setup for evaluator next-open entry."
            ],
            ["1h", "4h"],
            RIGHT_LABEL_GAP_PASS.replace(
                "return aligned.lt(30.0).fillna(False).astype(bool)",
                "return aligned.lt(30.0).fillna(False).astype(bool).shift(1, fill_value=False)",
            ),
            "Single material mutation adds a forbidden extra 1h signal delay.",
        ),
    ]
    identifiers = [case["case_id"] for case in cases]
    if len(cases) != 20 or len(set(identifiers)) != 20:
        raise RuntimeError("reviewer benchmark must contain 20 unique cases")
    if sum(case["expected"] == "pass" for case in cases) != 10:
        raise RuntimeError("reviewer benchmark must contain 10 passing cases")
    return cases


def benchmark_suite() -> dict[str, Any]:
    suite = {
        "suite_id": "EXP-074-REVIEWER-CALIBRATION-v1",
        "generator_model": "openai/gpt-5.6-luna",
        "reviewer_model": "z-ai/glm-5.2",
        "temperature": 0.0,
        "cases": benchmark_cases(),
    }
    suite["suite_sha256"] = sha256(canonical_json(suite).encode()).hexdigest()
    return suite
