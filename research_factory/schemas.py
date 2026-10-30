HYPOTHESIS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "mechanism_ids": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"},
                "rationale": {"type": "string"},
                "expected_regimes": {"type": "array", "items": {"type": "string"}},
                "entry_logic": {"type": "array", "items": {"type": "string"}},
                "exit_neutral_horizons": {
                    "type": "array",
                    "items": {"type": "integer", "enum": [12, 24, 48]},
                },
                "controls": {"type": "array", "items": {"type": "string"}},
                "failure_conditions": {"type": "array", "items": {"type": "string"}},
                "required_timeframes": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["1h", "4h", "1d"]},
                },
                "schema_version": {"type": "integer", "enum": [2]},
                "indicator_contracts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "function": {"type": "string"},
                            "keyword_parameters": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "value": {"type": "number"},
                                    },
                                    "required": ["name", "value"],
                                    "additionalProperties": False,
                                },
                            },
                            "output_bindings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "output": {"type": "string"},
                                        "target": {"type": "string"},
                                    },
                                    "required": ["output", "target"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "function",
                            "keyword_parameters",
                            "output_bindings",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "hypothesis_id",
                "mechanism_ids",
                "title",
                "rationale",
                "expected_regimes",
                "entry_logic",
                "exit_neutral_horizons",
                "controls",
                "failure_conditions",
                "required_timeframes",
                "schema_version",
                "indicator_contracts",
            ],
            "additionalProperties": False,
        },
        "generated_code": {"type": "string"},
    },
    "required": ["hypothesis", "generated_code"],
    "additionalProperties": False,
}


SPECIFICATION_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviewed_code_sha256": {"type": "string"},
        "hypothesis_code_consistent": {"type": "boolean"},
        "indicator_semantics_verified": {"type": "boolean"},
        "timeframe_semantics_verified": {"type": "boolean"},
        "entry_timing_consistent": {"type": "boolean"},
        "materiality_basis": {
            "type": "string",
            "enum": [
                "pass",
                "confirmed_signal_mismatch",
                "unresolved_material_ambiguity",
            ],
        },
        "review_notes": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "reviewed_code_sha256",
        "hypothesis_code_consistent",
        "indicator_semantics_verified",
        "timeframe_semantics_verified",
        "entry_timing_consistent",
        "materiality_basis",
        "review_notes",
        "findings",
    ],
    "additionalProperties": False,
}
