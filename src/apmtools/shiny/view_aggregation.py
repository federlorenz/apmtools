"""Shared aggregation for raw observations and previously summarized views.

A materialized row is the unit of observation for a second aggregation: means
are equal-weighted means of the existing rows, not means of original readings.
"""
from __future__ import annotations

import pandas as pd

from .config import FREQUENCIES, AGGREGATES


class ViewAggregationError(ValueError):
    """Invalid request to aggregate a view."""


def _option(value, choices):
    # Both UI labels ("Hour", "Mean") and normalized values are accepted.
    return choices.get(value, value)


def normalize_statistics(value):
    """Normalize legacy scalar or multi-select statistical functions."""
    if value is None:
        return []
    requested = [value] if isinstance(value, str) else list(value)
    result = []
    for item in requested:
        statistic = _option(item, AGGREGATES)
        if statistic is None:
            continue
        if statistic not in AGGREGATES.values():
            raise ViewAggregationError(f'Unknown statistical function: {item!r}')
        if statistic not in result:
            result.append(statistic)
    return result


def aggregate_view(data, value_columns, frequency=None, grouping_by=None,
                   aggregate='mean', *, only_complete=False,
                   flooring_timestamp=True, time_column=None):
    """Aggregate a raw Summary or an already materialized table/plot view.

    For raw timestamped data, defer to Summary.group_stat to retain its
    established completeness and frequency semantics. Without a raw datetime,
    use the present rows, never reconstruct completeness from summary values.
    A missing timestamp is fine when frequency=None. For temporal regrouping,
    a datetime-like column (e.g. `hour`) must be available or supplied.
    """
    frequency = _option(frequency, FREQUENCIES)
    statistics = normalize_statistics(aggregate)
    groups = list(grouping_by or [])
    values = list(value_columns or [])
    if data is None or data.empty:
        return pd.DataFrame()
    if not statistics:
        return data.copy()
    if not groups and frequency is None:
        return pd.DataFrame()
    missing = [c for c in groups + values if c not in data.columns]
    if missing:
        raise ViewAggregationError(f'Aggregation columns unavailable: {missing}')
    if not values:
        return pd.DataFrame()

    raw_time = ('datetime' in data.columns and
                pd.api.types.is_datetime64_any_dtype(data['datetime']))
    if raw_time and hasattr(data, 'group_stat'):
        kwargs = ({'only_complete': only_complete,
                   'flooring_timestamp': flooring_timestamp}
                  if frequency is not None else {})
        # Completeness belongs to each measured variable, not to the union of
        # selected Y variables. Otherwise a missing CO observation discards a
        # perfectly complete PM2.5 interval (and vice versa).
        if frequency is not None and only_complete and len(values) > 1:
            parts = []
            for value in values:
                part = data.group_stat(value_cols=[value], frequency=frequency,
                                       grouping_by=groups, aggregate=statistics,
                                       **kwargs)
                if part is not None and not part.empty:
                    parts.append(pd.DataFrame(part).reset_index(drop=True))
            if not parts:
                return pd.DataFrame()
            keys = list(dict.fromkeys(groups +
                ([frequency] if isinstance(frequency, str) else list(frequency))))
            result = parts[0]
            for part in parts[1:]:
                result = result.merge(part, on=keys, how='outer', sort=False,
                                      validate='one_to_one')
            return data._constructor(result)
        result = data.group_stat(value_cols=values, frequency=frequency,
                                 grouping_by=groups, aggregate=statistics,
                                 **kwargs)
        return pd.DataFrame() if result is None else result

    if frequency is not None and only_complete:
        raise ViewAggregationError(
            'Completeness cannot be recalculated from previously aggregated values. '
            'Disable "Only complete intervals" for this additional aggregation.')

    work = pd.DataFrame(data).copy()
    keys = list(groups)
    if frequency is not None:
        if frequency not in ('day', 'hour', 'minute', 'second'):
            raise ViewAggregationError(
                f'Frequency {frequency!r} is unsupported for already aggregated timestamps')
        candidates = ([time_column] if time_column else []) + [
            name for name in ('datetime', 'second', 'minute', 'hour', 'day')
            if name in work.columns]
        source = next((name for name in candidates if name in work.columns and
                       pd.api.types.is_datetime64_any_dtype(work[name])), None)
        if source is None:
            raise ViewAggregationError(
                'Temporal aggregation requires a datetime-like column '
                '(for example, hour or day) in the current view.')
        work[frequency] = pd.to_datetime(work[source]).dt.floor(
            {'day': 'D', 'hour': 'h', 'minute': 'min', 'second': 's'}[frequency])
        if frequency not in keys:
            keys.append(frequency)
    result = work.groupby(keys, dropna=False, sort=True)[values].agg(statistics)
    result.columns = [f'{column}_{stat}' for column, stat in result.columns]
    result = result.reset_index()
    return data._constructor(result) if hasattr(data, '_constructor') else result
