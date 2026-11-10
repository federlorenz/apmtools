"""Restricted, reproducible column expressions for Summary.calculate_column.

Supported: numeric/bool constants, column names, col('any column name'),
arithmetic, comparisons, bitwise boolean (& | ~), and approved functions.
No Python eval, attributes, indexing, comprehensions or dynamic calls.
"""
from __future__ import annotations

import ast
import operator

import numpy as np
import pandas as pd

_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
}
_COMPARE = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Invert: operator.invert}


def evaluate_column_expression(frame: pd.DataFrame, expression: str):
    """Evaluate an allowlisted vector expression, aligned to ``frame`` rows.

    ``col('name with spaces')`` references non-identifier column names.
    Operators are pandas elementwise operators, so comparison expressions
    should use ``&``/``|`` with parentheses rather than Python ``and``/``or``.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError('expression must be a non-empty string')
    if len(expression) > 4096:
        raise ValueError('expression is too long')
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as exc:
        raise ValueError(f'Invalid expression: {exc.msg}') from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError('expression is too complex')

    def series(value):
        if isinstance(value, pd.Series):
            return value
        return pd.Series(value, index=frame.index)

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float, bool):
                return node.value
            raise ValueError('Only numeric and boolean literals are permitted')
        if isinstance(node, ast.Name):
            if node.id not in frame.columns:
                raise ValueError(f'Unknown column: {node.id!r}')
            if not frame.columns.is_unique:
                raise ValueError('Expressions require unique column names')
            return frame[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or type(node.ops[0]) not in _COMPARE:
                raise ValueError('Only single comparisons are supported; combine using & or |')
            return _COMPARE[type(node.ops[0])](evaluate(node.left), evaluate(node.comparators[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name == 'col':
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                    raise ValueError("col() requires one literal string column name")
                key = node.args[0].value
                if key not in frame.columns:
                    raise ValueError(f'Unknown column: {key!r}')
                if not frame.columns.is_unique:
                    raise ValueError('Expressions require unique column names')
                return frame[key]
            if name not in {'log', 'log10', 'exp', 'sqrt', 'abs', 'round', 'where', 'fillna', 'clip', 'mean'}:
                raise ValueError(f'Unsupported function: {name!r}')
            args = [evaluate(arg) for arg in node.args]
            lengths = {'log': 1, 'log10': 1, 'exp': 1, 'sqrt': 1, 'abs': 1,
                       'round': 2, 'where': 3, 'fillna': 2, 'clip': 3}
            if name in lengths and len(args) != lengths[name]:
                raise ValueError(f'{name}() requires {lengths[name]} arguments')
            if name == 'log':
                return np.log(args[0])
            if name == 'log10':
                return np.log10(args[0])
            if name == 'exp':
                return np.exp(args[0])
            if name == 'sqrt':
                return np.sqrt(args[0])
            if name == 'abs':
                return abs(args[0])
            if name == 'round':
                if type(args[1]) is not int or not -15 <= args[1] <= 15:
                    raise ValueError('round() decimals must be an integer from -15 to 15')
                return np.round(args[0], args[1])
            if name == 'where':
                return pd.Series(np.where(series(args[0]).fillna(False).astype(bool),
                                          series(args[1]), series(args[2])), index=frame.index)
            if name == 'fillna':
                return series(args[0]).fillna(args[1])
            if name == 'clip':
                return series(args[0]).clip(lower=args[1], upper=args[2])
            if name == 'mean':
                if not args:
                    raise ValueError('mean() requires at least one argument')
                # Positional arrays preserve duplicate row indices.
                return pd.Series(np.nanmean(np.column_stack([
                    pd.to_numeric(series(arg), errors='raise').to_numpy(dtype=float)
                    for arg in args]), axis=1), index=frame.index)
        raise ValueError(f'Unsupported expression syntax: {type(node).__name__}')

    result = evaluate(tree.body)
    if isinstance(result, pd.Series):
        if len(result) != len(frame):
            raise ValueError('Expression result length does not match input rows')
        return result
    if np.isscalar(result):
        return pd.Series(result, index=frame.index)
    raise ValueError('Expression must produce a scalar or a column')
