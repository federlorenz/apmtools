"""Validation and serialization of user-defined straight-line overlays."""
import json
import math


def parse_reference_lines(value):
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else []
        except (ValueError, TypeError):
            return []
    if not isinstance(value, (list, tuple)):
        return []
    lines = []
    for idx, entry in enumerate(value):
        if not isinstance(entry, dict):
            continue
        try:
            intercept = float(entry.get('intercept', 0))
            slope = float(entry.get('slope', 0))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(intercept) and math.isfinite(slope)):
            continue
        label = str(entry.get('label', '')).strip() or f'Straight line {idx + 1}'
        # Stable identity across label edits, for colour and visibility selectors.
        key = f'Reference line {int(entry.get("id", idx + 1))}' if str(entry.get('id', idx + 1)).isdigit() else f'Reference line {idx + 1}'
        # Datetime lines use a dated intercept and a slope per selected unit.
        # Empty origin is resolved to the earliest X in the full plot dataset.
        origin = str(entry.get('origin', '') or '').strip()
        unit = str(entry.get('time_unit', 'day') or 'day').lower()
        if unit not in ('second', 'minute', 'hour', 'day', 'week'):
            unit = 'day'
        lines.append({'id': int(key.split()[-1]), 'label': label,
                      'intercept': intercept, 'slope': slope, 'key': key,
                      'origin': origin, 'time_unit': unit})
    return lines


def encode_reference_lines(lines):
    return json.dumps([{'id': line['id'], 'label': line['label'],
                        'intercept': line['intercept'], 'slope': line['slope'],
                        'origin': line['origin'], 'time_unit': line['time_unit']}
                       for line in parse_reference_lines(lines)], separators=(',', ':'))
