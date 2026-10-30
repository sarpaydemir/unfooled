from __future__ import annotations

import ast
from dataclasses import dataclass


ALLOWED_IMPORT_ROOTS = {"freqtrade", "numpy", "pandas", "pandas_ta", "talib", "technical"}
FORBIDDEN_NAMES = {
    "__import__", "compile", "eval", "exec", "input", "open",
    "breakpoint", "globals", "locals",
}
FORBIDDEN_CALL_ROOTS = {
    "os", "pathlib", "requests", "shutil", "socket", "subprocess", "sys", "urllib",
}
RAW_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class SafetyReport:
    passed: bool
    violations: tuple[str, ...]


class CandidateSafetyValidator(ast.NodeVisitor):
    def __init__(self) -> None:
        self.violations: list[str] = []
        self.function_stack: list[str] = []
        self.dataframe_aliases: set[str] = set()
        self.derived_columns: set[str] = set()
        self.constant_column_sets: dict[str, tuple[str, ...]] = {}

    def validate(self, source: str) -> SafetyReport:
        self.violations = []
        self.function_stack = []
        self.dataframe_aliases = set()
        self.derived_columns = set()
        self.constant_column_sets = {}
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            return SafetyReport(False, (f"syntax error: {error.msg} at line {error.lineno}",))
        self.visit(tree)
        return SafetyReport(not self.violations, tuple(sorted(set(self.violations))))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        if node.name == "signal" and node.args.args:
            self.dataframe_aliases.add(node.args.args[0].arg)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        # Validate the right-hand side before making newly derived columns
        # available to subsequent reads.
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                literal_columns = (
                    self._literal_columns(node.value)
                    if isinstance(node.value, (ast.List, ast.Tuple))
                    else None
                )
                if literal_columns is None:
                    self.constant_column_sets.pop(target.id, None)
                else:
                    self.constant_column_sets[target.id] = literal_columns
            if isinstance(target, ast.Name) and self._is_dataframe_alias_expression(node.value):
                self.dataframe_aliases.add(target.id)
                for output_column, source_column in self._named_aggregation_columns(node.value):
                    if (
                        source_column not in RAW_OHLCV_COLUMNS
                        and source_column not in self.derived_columns
                    ):
                        self.violations.append(
                            "undeclared named-aggregation source column "
                            f"'{source_column}' at line {node.lineno}"
                        )
                    else:
                        self.derived_columns.add(output_column)
            column = self._literal_dataframe_column(target)
            if column is not None:
                self.derived_columns.add(column)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            if isinstance(node.target, ast.Name) and self._is_dataframe_alias_expression(node.value):
                self.dataframe_aliases.add(node.target.id)
        column = self._literal_dataframe_column(node.target)
        if column is not None:
            self.derived_columns.add(column)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load) and self._is_dataframe_alias(node.value):
            columns = self._selector_columns(node.slice)
            if columns is None:
                self.violations.append(
                    f"dynamic raw dataframe column access at line {node.lineno}"
                )
            else:
                for column in columns:
                    if column not in RAW_OHLCV_COLUMNS and column not in self.derived_columns:
                        self.violations.append(
                            f"undeclared raw dataframe column '{column}' at line {node.lineno}"
                        )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                self.violations.append(f"forbidden import '{root}' at line {node.lineno}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".", 1)[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            self.violations.append(f"forbidden import '{root}' at line {node.lineno}")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.violations.append(f"forbidden name '{node.id}' at line {node.lineno}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in FORBIDDEN_CALL_ROOTS:
            self.violations.append(f"forbidden API '{root.id}' at line {node.lineno}")
        if node.attr.startswith("__"):
            self.violations.append(f"dunder access at line {node.lineno}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else ""
        )
        if name == "shift" and node.args and self._negative_number(node.args[0]):
            self.violations.append(f"future-looking negative shift at line {node.lineno}")
        is_string_join = (
            name == "join"
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Constant)
            and isinstance(node.func.value.value, str)
        )
        if name in {"merge", "merge_asof", "join"} and not is_string_join:
            self.violations.append(
                f"manual dataframe merge '{name}' requires reviewed informative merge at line {node.lineno}"
            )
        if (
            name == "get"
            and isinstance(node.func, ast.Attribute)
            and self._is_dataframe_alias(node.func.value)
        ):
            columns = self._selector_columns(node.args[0]) if node.args else None
            if columns is None:
                self.violations.append(
                    f"dynamic raw dataframe column access at line {node.lineno}"
                )
            else:
                for column in columns:
                    if column not in RAW_OHLCV_COLUMNS and column not in self.derived_columns:
                        self.violations.append(
                            f"undeclared raw dataframe column '{column}' at line {node.lineno}"
                        )
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if self.function_stack and self.function_stack[-1] == "signal" and node.value is not None:
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, bool):
                self.violations.append(
                    f"signal must return an index-aligned boolean Series, not scalar bool at line {node.lineno}"
                )
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "bool"
            ):
                self.violations.append(
                    f"signal must return an index-aligned boolean Series, not bool() at line {node.lineno}"
                )
            if any(
                isinstance(child, ast.Attribute) and child.attr in {"iloc", "iat", "at"}
                for child in ast.walk(node.value)
            ):
                self.violations.append(
                    f"signal return must not scalarize with row access at line {node.lineno}"
                )
            for child in ast.walk(node.value):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "shift"
                ):
                    self.violations.append(
                        f"returned signal must not shift execution at line {node.lineno}"
                    )
        self.generic_visit(node)

    @staticmethod
    def _negative_number(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        )

    def _is_dataframe_alias(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id in self.dataframe_aliases

    def _is_dataframe_alias_expression(self, node: ast.AST) -> bool:
        if self._is_dataframe_alias(node):
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return self._is_dataframe_alias_expression(node.func.value)
        if isinstance(node, ast.Subscript) and self._is_dataframe_alias(node.value):
            columns = self._selector_columns(node.slice)
            return columns is not None and len(columns) > 1
        return False

    def _literal_dataframe_column(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Subscript) or not self._is_dataframe_alias(node.value):
            return None
        columns = self._selector_columns(node.slice)
        if columns is None or len(columns) != 1:
            return None
        return columns[0]

    @staticmethod
    def _literal_columns(node: ast.AST) -> tuple[str, ...] | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return (node.value,)
        if isinstance(node, (ast.List, ast.Tuple)):
            values: list[str] = []
            for element in node.elts:
                if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                    return None
                values.append(element.value)
            return tuple(values)
        return None

    def _selector_columns(self, node: ast.AST) -> tuple[str, ...] | None:
        literal = self._literal_columns(node)
        if literal is not None:
            return literal
        if isinstance(node, ast.Name):
            return self.constant_column_sets.get(node.id)
        return None

    @staticmethod
    def _named_aggregation_columns(node: ast.AST) -> tuple[tuple[str, str], ...]:
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "agg"
        ):
            return ()
        columns: list[tuple[str, str]] = []
        for keyword in node.keywords:
            if keyword.arg is None or not isinstance(keyword.value, (ast.List, ast.Tuple)):
                continue
            if len(keyword.value.elts) != 2:
                continue
            source = keyword.value.elts[0]
            reducer = keyword.value.elts[1]
            if not (
                isinstance(source, ast.Constant)
                and isinstance(source.value, str)
                and isinstance(reducer, ast.Constant)
                and isinstance(reducer.value, str)
            ):
                continue
            columns.append((keyword.arg, source.value))
        return tuple(columns)
