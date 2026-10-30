from __future__ import annotations

import ast
from dataclasses import dataclass
import math

from .models import Hypothesis, IndicatorContract
from .safety import SafetyReport
from .talib_metadata import TalibMetadata


_NUMPY_FUNCTIONS = {
    "abs",
    "where",
    "logical_and",
    "logical_or",
    "logical_not",
    "maximum",
    "minimum",
    "isfinite",
    "isnan",
}
_SERIES_METHODS = {
    "fillna",
    "astype",
    "gt",
    "ge",
    "lt",
    "le",
    "eq",
    "ne",
    "between",
    "abs",
    "clip",
    "shift",
}
_MULTI_OUTPUT_INPUT_ARITY = {
    "AROON": 2,
    "BBANDS": 1,
    "HT_PHASOR": 1,
    "HT_SINE": 1,
    "MACD": 1,
    "MACDEXT": 1,
    "MACDFIX": 1,
    "MAMA": 1,
    "MINMAX": 1,
    "MINMAXINDEX": 1,
    "STOCH": 3,
    "STOCHF": 3,
    "STOCHRSI": 1,
}
_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)


@dataclass(frozen=True)
class _CallRecord:
    function: str
    parameters: tuple[tuple[str, int | float], ...] | None
    targets: tuple[str, ...] | None
    line: int


class IndicatorContractValidator:
    def __init__(self, metadata: TalibMetadata) -> None:
        self.metadata = metadata

    def validate(self, hypothesis: Hypothesis, source: str) -> SafetyReport:
        violations: list[str] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return SafetyReport(
                False,
                (f"syntax error: {error.msg} at line {error.lineno}",),
            )
        if hypothesis.schema_version != 2:
            return SafetyReport(False, ("indicator contracts require schema v2",))

        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        self._validate_import_shape(tree, parents, violations)
        signal_functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "signal"
        ]
        signal = signal_functions[0] if len(signal_functions) == 1 else None
        if len(signal_functions) != 1:
            violations.append("schema v2 code must define exactly one signal() function")
        elif not self._has_exact_signal_signature(signal_functions[0]):
            violations.append(
                "schema v2 code requires exact signature 'signal(dataframe)'"
            )

        records = self._extract_calls(tree, parents, signal, violations)
        if hypothesis.indicator_contracts or records:
            self._validate_canonical_talib_binding(tree, violations)
            self._validate_dependency_namespaces(tree, parents, violations)
        successful = self._match_contracts(
            hypothesis.indicator_contracts,
            records,
            violations,
        )
        if signal is not None and (records or hypothesis.indicator_contracts):
            self._validate_dependencies(signal, successful, violations)
        return SafetyReport(
            not violations,
            tuple(sorted(set(violations))),
        )

    @staticmethod
    def _validate_canonical_talib_binding(
        tree: ast.Module,
        violations: list[str],
    ) -> None:
        direct_imports = 0
        for statement in tree.body:
            if not isinstance(statement, ast.Import):
                continue
            direct_imports += sum(
                alias.name == "talib" and alias.asname is None
                for alias in statement.names
            )
        if direct_imports != 1:
            violations.append(
                "multi-output contracts require exactly one module-level "
                "direct 'import talib'"
            )
        for line in IndicatorContractValidator._binding_lines(
            tree,
            "talib",
            allowed_import=("talib", None),
        ):
            violations.append(
                f"canonical TA-Lib name 'talib' is rebound at line {line}"
            )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and IndicatorContractValidator._attribute_root_name(node) == "talib"
            ):
                violations.append(
                    f"trusted namespace 'talib' is mutated at line {node.lineno}"
                )

    @staticmethod
    def _validate_dependency_namespaces(
        tree: ast.Module,
        parents: dict[ast.AST, ast.AST],
        violations: list[str],
    ) -> None:
        used: dict[str, tuple[str, str | None]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
                continue
            if node.value.id == "pd" and node.attr == "Series":
                used["pd"] = ("pandas", "pd")
            elif node.value.id == "np" and node.attr in _NUMPY_FUNCTIONS:
                used["np"] = ("numpy", "np")
            elif node.value.id == "numpy" and node.attr in _NUMPY_FUNCTIONS:
                used["numpy"] = ("numpy", None)
        for trusted_name, (module_name, import_alias) in used.items():
            imports = 0
            for statement in tree.body:
                if not isinstance(statement, ast.Import):
                    continue
                imports += sum(
                    alias.name == module_name and alias.asname == import_alias
                    for alias in statement.names
                )
            if imports != 1:
                expected = (
                    f"import {module_name} as {import_alias}"
                    if import_alias is not None
                    else f"import {module_name}"
                )
                violations.append(
                    f"trusted dependency namespace '{trusted_name}' requires "
                    f"exactly one module-level '{expected}'"
                )
            for line in IndicatorContractValidator._binding_lines(
                tree,
                trusted_name,
                allowed_import=(module_name, import_alias),
            ):
                violations.append(
                    f"trusted dependency name '{trusted_name}' is rebound "
                    f"at line {line}"
                )
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Name)
                    and node.id == trusted_name
                    and isinstance(node.ctx, ast.Load)
                    and not IndicatorContractValidator._allowed_namespace_load(
                        node,
                        trusted_name,
                        parents,
                    )
                ):
                    violations.append(
                        f"unsupported trusted dependency use '{trusted_name}' "
                        f"at line {node.lineno}"
                    )
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.ctx, (ast.Store, ast.Del))
                    and IndicatorContractValidator._attribute_root_name(node)
                    == trusted_name
                ):
                    violations.append(
                        f"trusted namespace '{trusted_name}' is mutated at line "
                        f"{node.lineno}"
                    )

    @staticmethod
    def _binding_lines(
        tree: ast.AST,
        trusted_name: str,
        *,
        allowed_import: tuple[str, str | None] | None,
    ) -> tuple[int, ...]:
        lines: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and node.id == trusted_name
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ) or (isinstance(node, ast.arg) and node.arg == trusted_name):
                lines.add(getattr(node, "lineno", 0))
            elif (
                isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                )
                and node.name == trusted_name
            ):
                lines.add(node.lineno)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    bound_name = alias.asname or alias.name.split(".", 1)[0]
                    if bound_name != trusted_name:
                        continue
                    if allowed_import == (alias.name, alias.asname):
                        continue
                    lines.add(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    bound_name = alias.asname or alias.name
                    if bound_name == trusted_name:
                        lines.add(node.lineno)
            elif (
                isinstance(node, ast.ExceptHandler)
                and node.name == trusted_name
            ):
                lines.add(node.lineno)
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
                if node.name == trusted_name:
                    lines.add(node.lineno)
            elif (
                isinstance(node, ast.MatchMapping)
                and node.rest == trusted_name
            ):
                lines.add(node.lineno)
        return tuple(sorted(lines))

    @staticmethod
    def _allowed_namespace_load(
        node: ast.Name,
        trusted_name: str,
        parents: dict[ast.AST, ast.AST],
    ) -> bool:
        attribute = parents.get(node)
        if not (
            isinstance(attribute, ast.Attribute)
            and attribute.value is node
        ):
            return False
        allowed_attributes = (
            {"Series"} if trusted_name == "pd" else _NUMPY_FUNCTIONS
        )
        if attribute.attr not in allowed_attributes:
            return False
        call = parents.get(attribute)
        return isinstance(call, ast.Call) and call.func is attribute

    @staticmethod
    def _has_exact_signal_signature(
        signal: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        arguments = signal.args
        return (
            not arguments.posonlyargs
            and len(arguments.args) == 1
            and arguments.args[0].arg == "dataframe"
            and arguments.vararg is None
            and not arguments.kwonlyargs
            and arguments.kwarg is None
            and not arguments.defaults
            and not arguments.kw_defaults
        )

    @staticmethod
    def _attribute_root_name(node: ast.Attribute) -> str | None:
        root: ast.AST = node
        while isinstance(root, ast.Attribute):
            root = root.value
        return root.id if isinstance(root, ast.Name) else None

    def _validate_import_shape(
        self,
        tree: ast.Module,
        parents: dict[ast.AST, ast.AST],
        violations: list[str],
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "talib" and alias.asname is not None:
                        violations.append(
                            f"TA-Lib contracts require direct 'import talib' at line {node.lineno}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module == "talib":
                violations.append(
                    f"TA-Lib contracts require direct 'import talib' at line {node.lineno}"
                )
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "talib"
                and node.attr in self.metadata.outputs
            ):
                parent = parents.get(node)
                if not (
                    isinstance(parent, ast.Call)
                    and parent.func is node
                ):
                    violations.append(
                        f"indirect TA-Lib multi-output reference '{node.attr}' "
                        f"at line {node.lineno}"
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "talib"
            ):
                violations.append(
                    f"dynamic TA-Lib function lookup at line {node.lineno}"
                )
            elif (
                isinstance(node, ast.Name)
                and node.id == "talib"
                and isinstance(node.ctx, ast.Load)
            ):
                parent = parents.get(node)
                attribute = parent if isinstance(parent, ast.Attribute) else None
                call = parents.get(attribute) if attribute is not None else None
                if not (
                    attribute is not None
                    and attribute.value is node
                    and isinstance(call, ast.Call)
                    and call.func is attribute
                ):
                    violations.append(
                        f"TA-Lib must be called directly without module aliases "
                        f"at line {node.lineno}"
                    )

    def _extract_calls(
        self,
        tree: ast.Module,
        parents: dict[ast.AST, ast.AST],
        signal: ast.FunctionDef | ast.AsyncFunctionDef | None,
        violations: list[str],
    ) -> list[_CallRecord]:
        records: list[_CallRecord] = []
        for node in ast.walk(tree):
            function = self._direct_multi_output_function(node)
            if function is None:
                continue
            if signal is None or not self._is_descendant_of(node, signal, parents):
                violations.append(
                    f"{function} multi-output call is outside signal() at line {node.lineno}"
                )
                continue
            parent = parents.get(node)
            targets: tuple[str, ...] | None = None
            if (
                isinstance(parent, ast.Assign)
                and parent.value is node
                and len(parent.targets) == 1
                and isinstance(parent.targets[0], (ast.Tuple, ast.List))
                and all(isinstance(item, ast.Name) for item in parent.targets[0].elts)
            ):
                targets = tuple(item.id for item in parent.targets[0].elts)
            else:
                violations.append(
                    f"{function} at line {node.lineno} must use one tuple/list "
                    "assignment of simple names"
                )
            parameters = self._literal_parameters(node, function, violations)
            records.append(
                _CallRecord(function, parameters, targets, node.lineno)
            )
        return sorted(records, key=lambda item: item.line)

    def _direct_multi_output_function(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return None
        if not isinstance(node.func.value, ast.Name) or node.func.value.id != "talib":
            return None
        return node.func.attr if node.func.attr in self.metadata.outputs else None

    @staticmethod
    def _is_descendant_of(
        node: ast.AST,
        ancestor: ast.AST,
        parents: dict[ast.AST, ast.AST],
    ) -> bool:
        current = node
        while current in parents:
            current = parents[current]
            if current is ancestor:
                return True
        return False

    def _literal_parameters(
        self,
        node: ast.Call,
        function: str,
        violations: list[str],
    ) -> tuple[tuple[str, int | float], ...] | None:
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            violations.append(
                f"{function} at line {node.lineno} forbids starred arguments"
            )
            return None
        input_arity = _MULTI_OUTPUT_INPUT_ARITY[function]
        if len(node.args) != input_arity:
            violations.append(
                f"{function} optional parameters must use keywords; expected "
                f"{input_arity} positional input arrays at line {node.lineno}"
            )
            return None
        parameters: list[tuple[str, int | float]] = []
        names: set[str] = set()
        for keyword in node.keywords:
            if keyword.arg is None:
                violations.append(
                    f"{function} at line {node.lineno} forbids **kwargs"
                )
                return None
            value = self._numeric_literal(keyword.value)
            if value is None:
                violations.append(
                    f"{function} keyword '{keyword.arg}' must be a literal numeric "
                    f"keyword value at line {node.lineno}"
                )
                return None
            if keyword.arg in names:
                violations.append(
                    f"{function} duplicate keyword '{keyword.arg}' at line {node.lineno}"
                )
                return None
            names.add(keyword.arg)
            parameters.append((keyword.arg, self._canonical_number(value)))
        return tuple(sorted(parameters))

    @staticmethod
    def _numeric_literal(node: ast.AST) -> int | float | None:
        if (
            isinstance(node, ast.Constant)
            and not isinstance(node.value, bool)
            and isinstance(node.value, (int, float))
            and math.isfinite(node.value)
        ):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and not isinstance(node.operand.value, bool)
            and isinstance(node.operand.value, (int, float))
            and math.isfinite(node.operand.value)
        ):
            return (
                -node.operand.value
                if isinstance(node.op, ast.USub)
                else node.operand.value
            )
        return None

    @staticmethod
    def _canonical_number(value: int | float) -> int | float:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _match_contracts(
        self,
        contracts: tuple[IndicatorContract, ...],
        records: list[_CallRecord],
        violations: list[str],
    ) -> list[tuple[_CallRecord, IndicatorContract]]:
        valid_contracts: list[IndicatorContract] = []
        global_targets: set[str] = set()
        contract_identities: set[tuple] = set()
        for contract in contracts:
            canonical_outputs = self.metadata.outputs.get(contract.function)
            if canonical_outputs is None:
                violations.append(
                    f"unknown canonical multi-output function '{contract.function}'"
                )
                continue
            parameters = self._contract_parameters(contract, violations)
            bindings = tuple(
                sorted(
                    (
                        binding.output,
                        binding.target,
                    )
                    for binding in contract.output_bindings
                )
            )
            identity = (contract.function, parameters, bindings)
            if identity in contract_identities:
                violations.append(
                    f"duplicate indicator contract for {contract.function}"
                )
                continue
            contract_identities.add(identity)
            local_outputs: set[str] = set()
            local_targets: set[str] = set()
            for binding in contract.output_bindings:
                if binding.output not in canonical_outputs:
                    violations.append(
                        f"{contract.function} unknown canonical output "
                        f"'{binding.output}'"
                    )
                if (
                    binding.output in local_outputs
                    or binding.target in local_targets
                    or binding.target == "_"
                ):
                    violations.append(
                        f"{contract.function} has duplicate or placeholder output binding"
                    )
                if binding.target in global_targets:
                    violations.append(
                        f"TA-Lib output target '{binding.target}' is reused across contracts"
                    )
                local_outputs.add(binding.output)
                local_targets.add(binding.target)
                global_targets.add(binding.target)
            valid_contracts.append(contract)

        unmatched = set(range(len(records)))
        successful: list[tuple[_CallRecord, IndicatorContract]] = []
        for contract in valid_contracts:
            parameters = self._contract_parameters(contract, violations)
            candidates = [
                index
                for index in sorted(unmatched)
                if records[index].function == contract.function
                and records[index].parameters == parameters
            ]
            exact = [
                index
                for index in candidates
                if self._targets_match(contract, records[index])
            ]
            chosen = exact[0] if exact else (candidates[0] if candidates else None)
            if chosen is None:
                violations.append(
                    f"orphan {contract.function} indicator contract has no exact call"
                )
                continue
            unmatched.remove(chosen)
            record = records[chosen]
            if self._validate_binding_positions(contract, record, violations):
                successful.append((record, contract))
        for index in sorted(unmatched):
            record = records[index]
            violations.append(
                f"undeclared multi-output call {record.function} at line {record.line}"
            )
        return successful

    def _contract_parameters(
        self,
        contract: IndicatorContract,
        violations: list[str],
    ) -> tuple[tuple[str, int | float], ...]:
        parameters: list[tuple[str, int | float]] = []
        names: set[str] = set()
        for item in contract.keyword_parameters:
            if (
                item.name in names
                or isinstance(item.value, bool)
                or not isinstance(item.value, (int, float))
                or not math.isfinite(item.value)
            ):
                violations.append(
                    f"{contract.function} contract has invalid keyword parameters"
                )
                continue
            names.add(item.name)
            parameters.append((item.name, self._canonical_number(item.value)))
        return tuple(sorted(parameters))

    def _targets_match(
        self,
        contract: IndicatorContract,
        record: _CallRecord,
    ) -> bool:
        targets = record.targets
        canonical_outputs = self.metadata.outputs[contract.function]
        if targets is None or len(targets) != len(canonical_outputs):
            return False
        return all(
            targets[canonical_outputs.index(binding.output)] == binding.target
            for binding in contract.output_bindings
            if binding.output in canonical_outputs
        )

    def _validate_binding_positions(
        self,
        contract: IndicatorContract,
        record: _CallRecord,
        violations: list[str],
    ) -> bool:
        canonical_outputs = self.metadata.outputs[contract.function]
        if record.targets is None:
            return False
        if len(record.targets) != len(canonical_outputs):
            violations.append(
                f"{contract.function} at line {record.line} returns "
                f"{len(canonical_outputs)} outputs but binds {len(record.targets)}"
            )
            return False
        passed = True
        for binding in contract.output_bindings:
            if binding.output not in canonical_outputs:
                passed = False
                continue
            position = canonical_outputs.index(binding.output)
            if record.targets[position] != binding.target:
                violations.append(
                    f"{contract.function} output '{binding.output}' is not bound "
                    f"at position {position} to '{binding.target}' at line {record.line}"
                )
                passed = False
        return passed

    def _validate_dependencies(
        self,
        signal: ast.FunctionDef | ast.AsyncFunctionDef,
        successful: list[tuple[_CallRecord, IndicatorContract]],
        violations: list[str],
    ) -> None:
        returns = [
            node
            for node in ast.walk(signal)
            if isinstance(node, ast.Return)
            and self._nearest_function(node, signal) is signal
        ]
        if (
            len(returns) != 1
            or not signal.body
            or signal.body[-1] is not returns[0]
        ):
            violations.append(
                f"signal() must have a single final return at line {signal.lineno}"
            )
        if any(
            isinstance(node, _BRANCH_NODES)
            for statement in signal.body
            for node in ast.walk(statement)
        ):
            violations.append(
                f"signal() with multi-output contracts must be straight-line at line {signal.lineno}"
            )
        for index, statement in enumerate(signal.body):
            allowed = isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return))
            docstring = (
                index == 0
                and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
            if not allowed and not docstring:
                violations.append(
                    f"unsupported straight-line statement "
                    f"'{type(statement).__name__}' at line {statement.lineno}"
                )
            if isinstance(statement, ast.Assign):
                simple_assignment = (
                    len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                )
                multi_output_assignment = (
                    len(statement.targets) == 1
                    and isinstance(statement.targets[0], (ast.Tuple, ast.List))
                    and all(
                        isinstance(item, ast.Name)
                        for item in statement.targets[0].elts
                    )
                    and self._direct_multi_output_function(statement.value)
                    is not None
                )
                if not simple_assignment and not multi_output_assignment:
                    violations.append(
                        f"unsupported assignment target at line {statement.lineno}"
                    )
            elif isinstance(statement, ast.AnnAssign) and not isinstance(
                statement.target,
                ast.Name,
            ):
                violations.append(
                    f"unsupported assignment target at line {statement.lineno}"
                )
        if len(returns) != 1 or returns[0].value is None:
            return

        bound_targets = {
            binding.target
            for _, contract in successful
            for binding in contract.output_bindings
        }
        assignments: dict[str, ast.AST] = {}
        for statement in signal.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name):
                    assignments[target.id] = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                assignments[statement.target.id] = statement.value
        dataframe_names = {argument.arg for argument in signal.args.args}
        for node in ast.walk(signal):
            if (
                isinstance(node, ast.Name)
                and node.id in dataframe_names
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                violations.append(
                    f"dataframe parameter '{node.id}' is rebound at line "
                    f"{node.lineno}"
                )
        self._validate_tainted_assignments(
            signal,
            bound_targets,
            dataframe_names,
            violations,
        )
        for target in sorted(bound_targets):
            store_count = sum(
                1
                for node in ast.walk(signal)
                if (
                    isinstance(node, ast.Name)
                    and node.id == target
                    and isinstance(node.ctx, ast.Store)
                )
            )
            mutated_subscript = any(
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and any(
                    isinstance(child, ast.Name) and child.id == target
                    for child in ast.walk(node.value)
                )
                for node in ast.walk(signal)
            )
            deleted = any(
                isinstance(node, ast.Name)
                and node.id == target
                and isinstance(node.ctx, ast.Del)
                for node in ast.walk(signal)
            )
            if store_count != 1 or mutated_subscript or deleted:
                violations.append(
                    f"TA-Lib output target '{target}' is reassigned in signal()"
                )

        unsupported: list[str] = []
        reachable = self._reachable_names(
            returns[0].value,
            assignments,
            dataframe_names,
            unsupported,
        )
        violations.extend(unsupported)
        for record, contract in successful:
            declared = {binding.output for binding in contract.output_bindings}
            for binding in contract.output_bindings:
                if binding.target not in reachable:
                    violations.append(
                        f"{contract.function} output '{binding.output}' target "
                        f"'{binding.target}' does not reach the final return "
                        f"at line {record.line}"
                    )
            if record.targets is None:
                continue
            for position, target in enumerate(record.targets):
                output = self.metadata.outputs[record.function][position]
                if target != "_" and target in reachable and output not in declared:
                    violations.append(
                        f"{record.function} reachable output '{output}' is missing "
                        f"from the manifest at line {record.line}"
                    )

    @staticmethod
    def _nearest_function(
        node: ast.AST,
        expected: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        # ast.walk(expected) is used only for the expected function; nested
        # function definitions are rejected by the straight-line grammar.
        return expected

    def _reachable_names(
        self,
        expression: ast.AST,
        assignments: dict[str, ast.AST],
        dataframe_names: set[str],
        violations: list[str],
    ) -> set[str]:
        reachable: set[str] = set()
        visiting: set[str] = set()

        def visit_name(name: str) -> None:
            reachable.add(name)
            if name not in assignments or name in visiting:
                return
            visiting.add(name)
            for dependency in expression_dependencies(assignments[name]):
                visit_name(dependency)
            visiting.remove(name)

        def expression_dependencies(node: ast.AST) -> set[str]:
            if isinstance(node, ast.Name):
                return {node.id}
            if isinstance(node, ast.Constant):
                return set()
            if isinstance(node, ast.Attribute):
                return expression_dependencies(node.value)
            if isinstance(node, ast.Subscript):
                return (
                    expression_dependencies(node.value)
                    | expression_dependencies(node.slice)
                )
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                return self._union(expression_dependencies(item) for item in node.elts)
            if isinstance(node, ast.Dict):
                items = [item for item in node.keys + node.values if item is not None]
                return self._union(expression_dependencies(item) for item in items)
            if isinstance(node, ast.UnaryOp):
                return expression_dependencies(node.operand)
            if isinstance(node, ast.BinOp):
                return (
                    expression_dependencies(node.left)
                    | expression_dependencies(node.right)
                )
            if isinstance(node, ast.BoolOp):
                return self._union(expression_dependencies(item) for item in node.values)
            if isinstance(node, ast.Compare):
                return (
                    expression_dependencies(node.left)
                    | self._union(
                        expression_dependencies(item) for item in node.comparators
                    )
                )
            if isinstance(node, ast.Call):
                if not self._allowed_dependency_call(
                    node,
                    assignments,
                    dataframe_names,
                ):
                    violations.append(
                        f"unsupported dependency call at line {node.lineno}"
                    )
                parts = list(node.args) + [
                    keyword.value for keyword in node.keywords
                ]
                if isinstance(node.func, ast.Attribute):
                    parts.append(node.func.value)
                return self._union(expression_dependencies(item) for item in parts)
            violations.append(
                f"unsupported dependency expression at line {getattr(node, 'lineno', 0)}"
            )
            return {
                child.id
                for child in ast.walk(node)
                if isinstance(child, ast.Name)
            }

        for name in expression_dependencies(expression):
            visit_name(name)
        return reachable

    @staticmethod
    def _union(groups) -> set[str]:
        combined: set[str] = set()
        for group in groups:
            combined.update(group)
        return combined

    def _allowed_dependency_call(
        self,
        node: ast.Call,
        assignments: dict[str, ast.AST],
        dataframe_names: set[str],
    ) -> bool:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pd"
            and node.func.attr == "Series"
        ):
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"np", "numpy"}
            and node.func.attr in _NUMPY_FUNCTIONS
        ):
            return not any(
                keyword.arg == "out"
                for keyword in node.keywords
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr in _SERIES_METHODS:
            if not self._is_series_expression(
                node.func.value,
                assignments,
                dataframe_names,
                set(),
            ):
                return False
            for keyword in node.keywords:
                if keyword.arg != "inplace":
                    continue
                if not (
                    isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is False
                ):
                    return False
            if node.func.attr != "shift":
                return True
            amount_node = (
                node.args[0]
                if node.args
                else next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "periods"
                    ),
                    ast.Constant(value=1),
                )
            )
            amount = self._numeric_literal(amount_node)
            return amount is not None and amount > 0
        return False

    def _is_series_expression(
        self,
        node: ast.AST,
        assignments: dict[str, ast.AST],
        dataframe_names: set[str],
        visiting: set[str],
    ) -> bool:
        if isinstance(node, ast.Name):
            if node.id in visiting or node.id not in assignments:
                return False
            visiting.add(node.id)
            result = self._is_series_expression(
                assignments[node.id],
                assignments,
                dataframe_names,
                visiting,
            )
            visiting.remove(node.id)
            return result
        if isinstance(node, ast.Subscript):
            return isinstance(node.value, ast.Name) and node.value.id in dataframe_names
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pd"
                and node.func.attr == "Series"
            ):
                return True
            return (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in _SERIES_METHODS
                and self._is_series_expression(
                    node.func.value,
                    assignments,
                    dataframe_names,
                    visiting,
                )
            )
        if isinstance(node, ast.UnaryOp):
            return self._is_series_expression(
                node.operand,
                assignments,
                dataframe_names,
                visiting,
            )
        if isinstance(node, ast.BinOp):
            return any(
                self._is_series_expression(
                    item,
                    assignments,
                    dataframe_names,
                    visiting,
                )
                for item in (node.left, node.right)
            )
        if isinstance(node, ast.BoolOp):
            return any(
                self._is_series_expression(
                    item,
                    assignments,
                    dataframe_names,
                    visiting,
                )
                for item in node.values
            )
        if isinstance(node, ast.Compare):
            return any(
                self._is_series_expression(
                    item,
                    assignments,
                    dataframe_names,
                    visiting,
                )
                for item in (node.left, *node.comparators)
            )
        return False

    def _validate_tainted_assignments(
        self,
        signal: ast.FunctionDef | ast.AsyncFunctionDef,
        bound_targets: set[str],
        dataframe_names: set[str],
        violations: list[str],
    ) -> None:
        tainted = set(bound_targets)
        known_assignments: dict[str, ast.AST] = {}
        for statement in signal.body:
            target: ast.Name | None = None
            value: ast.AST | None = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                target = statement.targets[0]
                value = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                target = statement.target
                value = statement.value
            if target is None or value is None:
                continue
            loaded_names = {
                node.id
                for node in ast.walk(value)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            }
            if loaded_names & tainted:
                for call in (
                    node
                    for node in ast.walk(value)
                    if isinstance(node, ast.Call)
                ):
                    if not self._allowed_dependency_call(
                        call,
                        known_assignments,
                        dataframe_names,
                    ):
                        violations.append(
                            f"unsupported tainted dependency call at line "
                            f"{call.lineno}"
                        )
                tainted.add(target.id)
            known_assignments[target.id] = value
