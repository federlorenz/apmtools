"""Strict, versioned JSON encoding and replay validation for Summary history.

No pickle, arbitrary Python class construction, or dynamic method resolution is
used when deserializing a history document.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY_VERSION = 1
HISTORY_METHODS = frozenset({'change_values', 'map_values', 'edit_columns', 'subset', 'dtfilter', 'get_monitor'})
_TAG = '__apmtools_history_type__'


def encode_value(value):
    """Convert supported Python/pandas scalars and nested containers to JSON."""
    if value is pd.NA:
        return {_TAG: 'pd.NA'}
    if value is pd.NaT:
        return {_TAG: 'pd.NaT'}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.datetime64):
        return encode_value(pd.Timestamp(value))
    if isinstance(value, np.timedelta64):
        return encode_value(pd.Timedelta(value))
    if isinstance(value, np.generic):
        return encode_value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return {_TAG: 'float', 'value': 'nan'}
        if math.isinf(value):
            return {_TAG: 'float', 'value': 'inf' if value > 0 else '-inf'}
        return value
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return {_TAG: 'pd.NaT'}
        return {_TAG: 'Timestamp', 'value': value.isoformat(),
                'tz': str(value.tz) if value.tz is not None else None}
    if isinstance(value, pd.Timedelta):
        if pd.isna(value):
            return {_TAG: 'pd.NaT'}
        return {_TAG: 'Timedelta', 'value': int(value.value)}
    if isinstance(value, datetime):
        return {_TAG: 'datetime', 'value': value.isoformat()}
    if isinstance(value, date):
        return {_TAG: 'date', 'value': value.isoformat()}
    if isinstance(value, time):
        return {_TAG: 'time', 'value': value.isoformat()}
    if isinstance(value, timedelta):
        return {_TAG: 'timedelta', 'days': value.days,
                'seconds': value.seconds, 'microseconds': value.microseconds}
    if isinstance(value, tuple):
        return {_TAG: 'tuple', 'items': [encode_value(item) for item in value]}
    if isinstance(value, list):
        return [encode_value(item) for item in value]
    if isinstance(value, dict):
        if _TAG in value:
            raise TypeError(f'{_TAG} is reserved for history serialization')
        if not all(isinstance(k, str) for k in value):
            raise TypeError('History dictionaries must have string keys; use pairs for mappings')
        return {k: encode_value(v) for k, v in value.items()}
    raise TypeError(f'Unsupported history argument type: {type(value).__name__}')


def decode_value(value):
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if _TAG not in value:
        return {k: decode_value(v) for k, v in value.items()}
    kind = value[_TAG]
    keys = set(value)
    expected = {
        'pd.NA': {_TAG}, 'pd.NaT': {_TAG},
        'float': {_TAG, 'value'}, 'tuple': {_TAG, 'items'},
        'Timestamp': {_TAG, 'value', 'tz'},
        'Timedelta': {_TAG, 'value'},
        'datetime': {_TAG, 'value'}, 'date': {_TAG, 'value'},
        'time': {_TAG, 'value'},
        'timedelta': {_TAG, 'days', 'seconds', 'microseconds'},
    }
    if kind not in expected or keys != expected[kind]:
        raise ValueError(f'Invalid history value tag: {kind!r}')
    if kind == 'tuple':
        if not isinstance(value['items'], list):
            raise ValueError('Invalid tuple history payload')
        return tuple(decode_value(item) for item in value['items'])
    if kind == 'pd.NA':
        return pd.NA
    if kind == 'pd.NaT':
        return pd.NaT
    if kind == 'float':
        if value['value'] not in ('nan', 'inf', '-inf'):
            raise ValueError('Invalid special float')
        return float(value['value'])
    if kind == 'Timestamp':
        ts = pd.Timestamp(value['value'])
        tz = value['tz']
        if tz is not None:
            ts = ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)
        return ts
    if kind == 'Timedelta':
        return pd.Timedelta(value['value'], unit='ns')
    if kind == 'datetime':
        return datetime.fromisoformat(value['value'])
    if kind == 'date':
        return date.fromisoformat(value['value'])
    if kind == 'time':
        return time.fromisoformat(value['value'])
    if kind == 'timedelta':
        return timedelta(days=value['days'], seconds=value['seconds'],
                         microseconds=value['microseconds'])
    raise ValueError(f'Invalid history value tag: {kind!r}')


def validate_steps(steps):
    """Validate an encoded history without evaluating any transformations."""
    if not isinstance(steps, list):
        raise ValueError('History must be a list of steps')
    normalized = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) != {'method', 'args'}:
            raise ValueError(f'Invalid history step at index {index}')
        if step['method'] not in HISTORY_METHODS:
            raise ValueError(f'History method not permitted: {step["method"]!r}')
        if not isinstance(step['args'], dict):
            raise ValueError(f'History step {index} args must be an object')
        # Decode tags and enforce supported types, including strict JSON-safe floats.
        normalized.append({'method': step['method'],
                           'args': encode_value(decode_value(step['args']))})
    return normalized


def load_history(source):
    """Read a versioned JSON path/document or an existing list of steps."""
    if isinstance(source, (str, bytes, Path)) or hasattr(source, '__fspath__'):
        with open(source, encoding='utf-8') as file:
            source = json.load(file, parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError(f'Nonstandard JSON constant: {x}')))
    if isinstance(source, dict):
        if set(source) != {'version', 'history'} or type(source['version']) is not int or source['version'] != HISTORY_VERSION:
            raise ValueError(f'Unsupported history document version (expected {HISTORY_VERSION})')
        source = source['history']
    return validate_steps(source)


def save_history(path, steps):
    document = {'version': HISTORY_VERSION, 'history': validate_steps(steps)}
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(document, file, indent=2, allow_nan=False)
