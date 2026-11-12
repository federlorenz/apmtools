"""Validated, replayable transformations used by the Modify dataset sidebar.

Every mutation delegates to a recorded Summary method. The input Summary is
never changed, and multi-step operations are committed only on success.
"""
from __future__ import annotations

import ast
import re


def parse_rename_lines(text):
    """Parse one old=new column rename per line, preserving spaces in names."""
    result = []
    for line in (text or '').splitlines():
        if not line.strip():
            continue
        if '=' not in line:
            raise ValueError(f"Expected old=new: {line!r}")
        old, new = (part.strip() for part in line.split('=', 1))
        if not old or not new:
            raise ValueError(f"Both names are required: {line!r}")
        result.append((old, new))
    if not result:
        raise ValueError('Enter at least one old=new mapping')
    return result


def _value(text, dtype=None):
    """Interpret literals, but leave unquoted strings as text."""
    text = str(text).strip()
    if not text:
        return ''
    try:
        val = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        val = text
    if dtype is not None:
        from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype
        if is_bool_dtype(dtype) and isinstance(val, str):
            if val.lower() in ('true', 'false'):
                return val.lower() == 'true'
        if is_string_dtype(dtype) and not is_numeric_dtype(dtype) and val is not None:
            return str(val)
    return val


def parse_value_lines(text, dtype):
    """Parse value substitutions into recorded mapping pairs, keeping key types."""
    lines = parse_rename_lines(text)
    return [(_value(old, dtype), _value(new, dtype)) for old, new in lines]


def _columns(summary, names):
    if not summary.columns.is_unique:
        raise ValueError('Column editing requires unique column names')
    missing = [name for name in names if name not in summary.columns]
    if missing:
        raise ValueError(f"Unknown columns: {', '.join(missing)}")


def rename_defaults(summary):
    """Editable, one-to-one old=new defaults for each column."""
    return "\n".join(f"{name}={name}" for name in map(str, summary.columns))


def values_defaults(summary, column, *, max_values=200):
    """Prepopulate unique, non-missing values without scanning them repeatedly.

    Cap the editor size for high-cardinality columns; users can still add
    replacements manually. Values are represented as literal text so numeric
    and boolean values round-trip through parse_value_lines().
    """
    import pandas as pd
    _columns(summary, [column])
    values = pd.unique(summary[column].dropna())
    entries = []
    for value in values[:max_values]:
        label = str(value)
        if "\n" in label or "=" in label:
            continue  # These require a more expressive editor than old=new lines.
        entries.append(f"{label}={label}")
    return "\n".join(entries)


def keep_defaults(summary):
    """Comma-separated column names to retain, in their current order."""
    return ", ".join(map(str, summary.columns))


def edit_keep_text(summary, text):
    """Remove all columns except those named in a comma-separated list."""
    names = [name.strip() for name in (text or "").split(",")]
    if any(not name for name in names):
        raise ValueError("Keep columns requires comma-separated, nonempty names")
    return edit_keep(summary, names)


def edit_rename(summary, text):
    pairs = parse_rename_lines(text)
    _columns(summary, [old for old, _ in pairs])
    pairs = [(old, new) for old, new in pairs if old != new]
    if not pairs:
        raise ValueError("No column names were changed")
    if len({old for old, _ in pairs}) != len(pairs):
        raise ValueError('A column cannot be renamed twice in one operation')
    targets = [new for _, new in pairs]
    if len(set(targets)) != len(targets):
        raise ValueError('New column names must be unique')
    # Keep recorded rename actions valid even when names are swapped or chained.
    untouched = set(summary.columns) - {old for old, _ in pairs}
    if untouched.intersection(targets):
        raise ValueError('A new name is already in use by a column not being renamed')
    current = summary
    reserved = set(summary.columns) | set(targets)
    temporaries = []
    for i, (old, new) in enumerate(pairs):
        tmp = f'__apmtools_rename_{i}__'
        while tmp in reserved:
            tmp += '_'
        reserved.add(tmp)
        current = current.edit_columns('rename', old, new_name=tmp)
        temporaries.append((tmp, new))
    for tmp, new in temporaries:
        current = current.edit_columns('rename', tmp, new_name=new)
    return current


def edit_values(summary, column, text):
    _columns(summary, [column])
    mappings = parse_value_lines(text, summary[column].dtype)
    mappings = [(old, new) for old, new in mappings if old != new]
    if not mappings:
        raise ValueError("No column values were changed")
    return summary.change_values(column, mappings)


def edit_remove(summary, columns):
    names = list(dict.fromkeys(columns or []))
    if not names:
        raise ValueError('Choose at least one column to remove')
    _columns(summary, names)
    if len(names) == len(summary.columns):
        raise ValueError('At least one column must remain')
    current = summary
    for name in names:
        current = current.edit_columns('remove', name)
    return current


def edit_keep(summary, columns):
    names = list(dict.fromkeys(columns or []))
    if not names:
        raise ValueError('Choose at least one column to keep')
    _columns(summary, names)
    if len(names) == len(summary.columns):
        raise ValueError("All columns are already kept")
    current = summary
    for name in summary.columns:
        if name not in names:
            current = current.edit_columns('remove', name)
    return current


def create_column(summary, name, fill=''):
    name = (name or '').strip()
    if not name:
        raise ValueError('Enter a column name')
    if name in summary.columns:
        raise ValueError(f'Column already exists: {name}')
    return summary.edit_columns('add', name, fill=_value(fill))


def calculate_column(summary, name, expression):
    return summary.calculate_column((name or '').strip(), (expression or '').strip())


def _map_condition(summary, column, operator, value):
    """Validate and parse one user-entered mapping predicate."""
    _columns(summary, [column])
    allowed = {'==', '!=', '>', '>=', '<', '<=', 'isin', 'isna', 'notna'}
    if operator not in allowed:
        raise ValueError(f'Unsupported condition: {operator}')
    condition = {'column': column, 'op': operator}
    if operator == 'isin':
        condition['value'] = [_value(item.strip(), summary[column].dtype)
                              for item in str(value or '').split(',') if item.strip()]
        if not condition['value']:
            raise ValueError('Enter one or more values for isin')
    elif operator not in ('isna', 'notna'):
        if value is None or not str(value).strip():
            raise ValueError(f'Enter a comparison value for {column}')
        condition['value'] = _value(value, summary[column].dtype)
    return condition


def map_column_conditions(summary, target_column, conditions, when_true,
                          when_false='', has_false=False):
    """Map values using multiple predicates joined by AND/OR, left to right.

    ``conditions`` is a list of dictionaries with ``column``, ``op``,
    ``value`` and, after the first entry, ``join`` (``and`` or ``or``).
    The resulting expression is fully nested and JSON-history replayable.
    Example: (A AND B) OR C, rather than A AND (B OR C).
    """
    _columns(summary, [target_column])
    if not isinstance(conditions, (list, tuple)) or not conditions:
        raise ValueError('Enter at least one mapping condition')
    if not all(isinstance(entry, dict) for entry in conditions):
        raise ValueError('Each mapping condition must be a dictionary')
    combined = None
    for index, entry in enumerate(conditions):
        condition = _map_condition(summary, entry.get('column'), entry.get('op'),
                                   entry.get('value'))
        if index == 0:
            combined = condition
        else:
            connector = str(entry.get('join', 'and')).lower()
            if connector not in ('and', 'or'):
                raise ValueError(f'Unsupported logical connector: {connector}')
            combined = {connector: [combined, condition]}
    true_value = _value(when_true, summary[target_column].dtype)
    false_value = (_value(when_false, summary[target_column].dtype)
                   if has_false else None)
    return summary.map_values(target_column, combined, true_value, false_value)


def map_column(summary, target_column, condition_column, operator, value,
               when_true, when_false='', has_false=False):
    """Backward-compatible single-condition mapping interface."""
    return map_column_conditions(
        summary, target_column,
        [{'column': condition_column, 'op': operator, 'value': value}],
        when_true, when_false, has_false)
