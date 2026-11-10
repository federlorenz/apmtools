"""Multiple include/exclude datetime windows for Shiny filters and saved views.

The original datetime_start/datetime_end fields remain valid. Positive windows
are unioned, then negative windows are subtracted. Each window delegates to
Summary.dtfilter, retaining its date/time and exclusive-end semantics.
"""
from __future__ import annotations

MAX_DATETIME_INTERVALS = 16


def normalize_datetime_intervals(intervals=None, start=None, end=None):
    """Return a detached, serializable list of nonempty intervals (legacy-aware)."""
    if intervals is None:
        intervals = [{'start': start or '', 'end': end or '', 'type': 'positive'}]
    result = []
    for item in intervals:
        if not isinstance(item, dict):
            continue
        lower = str(item.get('start') or '').strip()
        upper = str(item.get('end') or '').strip()
        if lower or upper:
            result.append({'start': lower, 'end': upper,
                           'type': 'negative' if item.get('type') == 'negative' else 'positive'})
    return result


def saved_datetime_intervals(settings):
    return normalize_datetime_intervals(settings.get('datetime_intervals'),
                                        settings.get('datetime_start'), settings.get('datetime_end'))


def input_datetime_intervals(read, settings=None):
    """Read mounted interval controls, using saved values for unmounted inputs.

    `read(name, fallback)` must distinguish a missing widget from a present
    empty widget; this permits the user to clear a previously saved interval.
    """
    settings = settings or {}
    saved = saved_datetime_intervals(settings)
    result = []
    for i in range(MAX_DATETIME_INTERVALS):
        original = saved[i] if i < len(saved) else {}
        start_id = 'datetime_start' if i == 0 else f'datetime_start_{i}'
        end_id = 'datetime_end' if i == 0 else f'datetime_end_{i}'
        start = read(start_id, original.get('start', ''))
        end = read(end_id, original.get('end', ''))
        mode = read(f'datetime_mode_{i}', original.get('type', 'positive'))
        if not str(start or '').strip() and not str(end or '').strip():
            break  # Later controls are hidden until the preceding interval is filled.
        result.append({'start': start, 'end': end, 'type': mode})
    return normalize_datetime_intervals(result)


def datetime_interval_controls(ui, settings=None):
    """Static Shiny inputs, showing the next window as the preceding is filled.

    Input identifiers remain stable across restoration and subsequent edits.
    """
    saved = saved_datetime_intervals(settings or {})
    elements = []
    for i in range(MAX_DATETIME_INTERVALS):
        item = saved[i] if i < len(saved) else {}
        start_id = 'datetime_start' if i == 0 else f'datetime_start_{i}'
        end_id = 'datetime_end' if i == 0 else f'datetime_end_{i}'
        fields = ui.div(
            ui.tags.strong(f'Datetime interval {i + 1}'),
            ui.input_select(f'datetime_mode_{i}', 'Filter type',
                choices={'positive': 'Positive (keep)', 'negative': 'Negative (exclude)'},
                selected=item.get('type', 'positive')),
            ui.input_text(start_id, 'Start datetime', value=item.get('start', ''),
                placeholder='YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS'),
            ui.input_text(end_id, 'End datetime', value=item.get('end', ''),
                placeholder='YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS'),
            class_='mb-3')
        if i:
            prev_start = 'datetime_start' if i == 1 else f'datetime_start_{i - 1}'
            prev_end = 'datetime_end' if i == 1 else f'datetime_end_{i - 1}'
            fields = ui.panel_conditional(
                f"(input.{prev_start} && input.{prev_start}.trim() !== '') || "
                f"(input.{prev_end} && input.{prev_end}.trim() !== '')", fields)
        elements.append(fields)
    elements.append(ui.help_text(
        'Positive intervals are combined (keep any matching interval); negative '
        'intervals exclude matching observations. Clear start and end to remove an interval.'))
    return elements
