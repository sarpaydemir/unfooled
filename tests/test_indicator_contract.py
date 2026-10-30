from hashlib import sha256
import json
from pathlib import Path
import unittest

from research_factory.indicator_contract import IndicatorContractValidator
from research_factory.models import (
    Hypothesis,
    IndicatorContract,
    IndicatorOutputBinding,
    IndicatorParameter,
)
from research_factory.reviewer_benchmark_cases import (
    AROON_PASS,
    BBANDS_PASS,
    STOCHRSI_PASS,
)
from research_factory.talib_metadata import load_pinned_metadata
from research_factory.validation import hypothesis_from_stored_payload


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "EXP-075"


def hypothesis_contract(
    function: str,
    parameters: dict[str, int | float],
    bindings: dict[str, str],
) -> Hypothesis:
    return hypothesis_with_contracts(
        (
            IndicatorContract(
                function=function,
                keyword_parameters=tuple(
                    IndicatorParameter(name, value)
                    for name, value in sorted(parameters.items())
                ),
                output_bindings=tuple(
                    IndicatorOutputBinding(output, target)
                    for output, target in bindings.items()
                ),
            ),
        )
    )


def hypothesis_with_contracts(
    contracts: tuple[IndicatorContract, ...],
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="validator-fixture",
        mechanism_ids=("fixture",),
        title="TA-Lib output contract fixture",
        rationale="Deterministic structural validation only.",
        expected_regimes=("fixture",),
        entry_logic=("Use the declared TA-Lib output.",),
        exit_neutral_horizons=(48,),
        controls=("No performance data.",),
        failure_conditions=("Any contract mismatch.",),
        required_timeframes=("1h",),
        schema_version=2,
        indicator_contracts=contracts,
    )


def validator() -> IndicatorContractValidator:
    return IndicatorContractValidator(load_pinned_metadata())


class IndicatorContractTests(unittest.TestCase):
    def test_exp075_fixture_manifests_are_code_bound(self) -> None:
        expected = {
            "P07": (
                "46f721b64588dd55d8b98e9f72eb746fc1553a83cb812cc6b2022af422ec3d8f",
                True,
            ),
            "N07": (
                "58f5cdc0e86ad54a4d336f275e1c2e49cf0c42ecf557cf7b193a4c2741243814",
                False,
            ),
        }
        for case, (digest, should_pass) in expected.items():
            with self.subTest(case=case):
                directory = FIXTURE_ROOT / case
                source = (directory / "candidate.py").read_text(encoding="utf-8")
                manifest = json.loads(
                    (directory / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(sha256(source.encode()).hexdigest(), digest)
                self.assertEqual(manifest["candidate_code_sha256"], digest)
                hypothesis = hypothesis_from_stored_payload(manifest["hypothesis"])
                self.assertEqual(
                    validator().validate(hypothesis, source).passed,
                    should_pass,
                )

    def test_aroon_canonical_first_output_passes(self) -> None:
        source = AROON_PASS.strip() + "\n"
        self.assertEqual(
            sha256(source.encode()).hexdigest(),
            "46f721b64588dd55d8b98e9f72eb746fc1553a83cb812cc6b2022af422ec3d8f",
        )
        report = validator().validate(
            hypothesis_contract(
                "AROON",
                {"timeperiod": 14},
                {"aroondown": "aroondown"},
            ),
            source,
        )
        self.assertTrue(report.passed, report.violations)

    def test_aroon_second_output_mislabeled_as_down_fails(self) -> None:
        mutated = (AROON_PASS.strip() + "\n").replace(
            "aroondown, _ =", "_, aroondown ="
        )
        self.assertEqual(
            sha256(mutated.encode()).hexdigest(),
            "58f5cdc0e86ad54a4d336f275e1c2e49cf0c42ecf557cf7b193a4c2741243814",
        )
        report = validator().validate(
            hypothesis_contract(
                "AROON",
                {"timeperiod": 14},
                {"aroondown": "aroondown"},
            ),
            mutated,
        )
        self.assertFalse(report.passed)
        self.assertIn(
            "AROON output 'aroondown' is not bound at position 0",
            report.violations[0],
        )

    def test_stochrsi_fastd_second_output_passes(self) -> None:
        report = validator().validate(
            hypothesis_contract(
                "STOCHRSI",
                {
                    "timeperiod": 14,
                    "fastk_period": 14,
                    "fastd_period": 3,
                    "fastd_matype": 0,
                },
                {"fastd": "fastd"},
            ),
            STOCHRSI_PASS.strip() + "\n",
        )
        self.assertTrue(report.passed, report.violations)

    def test_bbands_three_outputs_pass(self) -> None:
        report = validator().validate(
            hypothesis_contract(
                "BBANDS",
                {
                    "timeperiod": 20,
                    "nbdevup": 2.0,
                    "nbdevdn": 2.0,
                    "matype": 0,
                },
                {
                    "upperband": "upper",
                    "middleband": "middle",
                    "lowerband": "lower",
                },
            ),
            BBANDS_PASS.strip() + "\n",
        )
        self.assertTrue(report.passed, report.violations)

    def test_repeated_identical_calls_with_distinct_bindings_pass(self) -> None:
        source = """
import pandas as pd
import talib

def signal(dataframe):
    first_down, _ = talib.AROON(
        dataframe["high"].to_numpy(dtype=float),
        dataframe["low"].to_numpy(dtype=float),
        timeperiod=14,
    )
    second_down, _ = talib.AROON(
        dataframe["high"].to_numpy(dtype=float),
        dataframe["low"].to_numpy(dtype=float),
        timeperiod=14,
    )
    first = pd.Series(first_down, index=dataframe.index)
    second = pd.Series(second_down, index=dataframe.index)
    combined = first.gt(70.0) & second.gt(65.0)
    return combined.fillna(False).astype(bool)
""".strip() + "\n"
        contract = lambda target: IndicatorContract(
            "AROON",
            (IndicatorParameter("timeperiod", 14),),
            (IndicatorOutputBinding("aroondown", target),),
        )
        report = validator().validate(
            hypothesis_with_contracts(
                (contract("first_down"), contract("second_down"))
            ),
            source,
        )
        self.assertTrue(report.passed, report.violations)

    def test_dependency_escapes_fail_closed(self) -> None:
        prefix = """
import numpy as np
import pandas as pd
import talib

def signal(dataframe):
    aroondown, _ = talib.AROON(
        dataframe["high"].to_numpy(dtype=float),
        dataframe["low"].to_numpy(dtype=float),
        timeperiod=14,
    )
    other_signal = dataframe["close"].gt(0)
"""
        cases = {
            "dummy_read": (
                "    seen = aroondown\n"
                "    return other_signal\n",
                "does not reach the final return",
            ),
            "unreachable_read": (
                "    return other_signal\n"
                "    seen = aroondown\n",
                "single final return",
            ),
            "reassignment": (
                "    aroondown = dataframe[\"close\"]\n"
                "    return aroondown\n",
                "reassigned",
            ),
            "augmented_reassignment": (
                "    aroondown += dataframe[\"close\"].to_numpy(dtype=float)\n"
                "    return pd.Series(aroondown, index=dataframe.index)\n",
                "reassigned",
            ),
            "helper_escape": (
                "    return helper(aroondown)\n",
                "unsupported dependency call",
            ),
            "fake_series_receiver": (
                "    return sink.fillna(aroondown)\n",
                "unsupported dependency call",
            ),
            "side_effect_escape": (
                "    mutate(aroondown)\n"
                "    return pd.Series(aroondown, index=dataframe.index)\n",
                "unsupported straight-line statement",
            ),
            "assigned_side_effect_escape": (
                "    ignored = mutate(aroondown)\n"
                "    return pd.Series(aroondown, index=dataframe.index)\n",
                "unsupported tainted dependency call",
            ),
            "numpy_out_mutation": (
                "    ignored = np.maximum(0.0, 0.0, out=aroondown)\n"
                "    return pd.Series(aroondown, index=dataframe.index)\n",
                "unsupported tainted dependency call",
            ),
            "numeric_inplace_mutation": (
                "    down = pd.Series(aroondown, index=dataframe.index)\n"
                "    ignored = down.clip(lower=0.0, inplace=1)\n"
                "    return down.gt(0.0)\n",
                "unsupported tainted dependency call",
            ),
            "annotated_side_effect_escape": (
                "    ignored: object = mutate(aroondown)\n"
                "    return pd.Series(aroondown, index=dataframe.index)\n",
                "unsupported tainted dependency call",
            ),
            "tuple_side_effect_escape": (
                "    ignored, other = mutate(aroondown)\n"
                "    return pd.Series(aroondown, index=dataframe.index)\n",
                "unsupported assignment target",
            ),
            "dataframe_parameter_rebound": (
                "    dataframe = sink\n"
                "    return dataframe[\"x\"].fillna(aroondown)\n",
                "dataframe parameter 'dataframe' is rebound",
            ),
            "dataframe_escape": (
                "    dataframe[\"x\"] = aroondown\n"
                "    return dataframe[\"x\"]\n",
                "does not reach the final return",
            ),
            "multiple_return": (
                "    if dataframe[\"close\"].gt(0).any():\n"
                "        return pd.Series(aroondown, index=dataframe.index)\n"
                "    return other_signal\n",
                "single final return",
            ),
            "unused_binding": (
                "    down = pd.Series(aroondown, index=dataframe.index)\n"
                "    return other_signal\n",
                "does not reach the final return",
            ),
        }
        contract = hypothesis_contract(
            "AROON",
            {"timeperiod": 14},
            {"aroondown": "aroondown"},
        )
        for name, (body, fragment) in cases.items():
            with self.subTest(name=name):
                report = validator().validate(contract, prefix + body)
                self.assertFalse(report.passed)
                self.assertTrue(
                    any(fragment in violation for violation in report.violations),
                    report.violations,
                )

    def test_call_shape_and_contract_errors_fail_closed(self) -> None:
        base = AROON_PASS.strip() + "\n"
        cases = (
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace("timeperiod=14", "timeperiod=period"),
                "literal numeric keyword",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"not_an_output": "aroondown"},
                ),
                base,
                "unknown canonical output",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace("aroondown, _ =", "outputs ="),
                "tuple/list",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace("import talib", "import talib as ta").replace(
                    "talib.AROON", "ta.AROON"
                ),
                "direct 'import talib'",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace("import talib", "import talib\nmodule = talib").replace(
                    "talib.AROON", "module.AROON"
                ),
                "without module aliases",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace("import talib", "import talib\ntalib = object()"),
                "name 'talib' is rebound",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import pandas as pd",
                    "import pandas as pd\npd = object()",
                ),
                "name 'pd' is rebound",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import talib",
                    "import talib\ntalib.AROON = replacement",
                ),
                "trusted namespace 'talib' is mutated",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace("import talib", "import talib\nclass talib:\n    pass"),
                "name 'talib' is rebound",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import talib",
                    "import talib\nimport pandas as talib",
                ),
                "name 'talib' is rebound",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import pandas as pd",
                    "import pandas as pd\nclass pd:\n    pass",
                ),
                "name 'pd' is rebound",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import talib",
                    "import talib\ndef talib():\n    return None",
                ),
                "name 'talib' is rebound",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import talib",
                    "import talib\nmatch {}:\n"
                    "    case {**talib}:\n"
                    "        pass",
                ),
                "name 'talib' is rebound",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import talib",
                    'import talib\nsetattr(pd, "Series", fake_series)',
                ),
                "unsupported trusted dependency use",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "import talib",
                    'import talib\nvars(pd)["Series"] = fake_series',
                ),
                "unsupported trusted dependency use",
            ),
            (
                hypothesis_contract(
                    "AROON",
                    {"timeperiod": 14},
                    {"aroondown": "aroondown"},
                ),
                base.replace(
                    "def signal(dataframe):",
                    "def signal(dataframe, /):",
                ),
                "exact signature 'signal(dataframe)'",
            ),
            (
                hypothesis_contract(
                    "STOCHRSI",
                    {},
                    {"fastd": "fastd"},
                ),
                """
import pandas as pd
import talib

def signal(dataframe):
    _, fastd = talib.STOCHRSI(
        dataframe["close"].to_numpy(dtype=float), 14, 14, 3, 0
    )
    fastd_series = pd.Series(fastd, index=dataframe.index)
    return fastd_series.lt(20.0).fillna(False).astype(bool)
""".strip() + "\n",
                "optional parameters must use keywords",
            ),
            (
                hypothesis_with_contracts(()),
                base,
                "undeclared multi-output call",
            ),
        )
        for contract, source, fragment in cases:
            with self.subTest(fragment=fragment):
                report = validator().validate(contract, source)
                self.assertFalse(report.passed)
                self.assertTrue(
                    any(fragment in violation for violation in report.violations),
                    report.violations,
                )
