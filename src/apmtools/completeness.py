"""Sampling completeness calculations shared by Summary and future quality views.

Coverage is estimated from each identifier's median positive sampling gap.
The first and last partially observed bins cannot be classified as complete
without a known deployment schedule: results describe coverage inside bins
within the observed recording span, not compliance with a planned deployment.
"""
from __future__ import annotations

import re
import pandas as pd

_UNITS = {'second': 's', 'minute': 'min', 'hour': 'h', 'day': 'D', 'week': 'W'}
_DURATION = {'second': pd.Timedelta(seconds=1),
             'minute': pd.Timedelta(minutes=1),
             'hour': pd.Timedelta(hours=1),
             'day': pd.Timedelta(days=1),
             'week': pd.Timedelta(weeks=1),
             'weekday': pd.Timedelta(days=1)}


def _primary_frequency(frequency):
    if isinstance(frequency, (tuple, list)):
        if not frequency:
            raise ValueError('frequency cannot be empty')
        # Match the historical group_stat ranking of coarsest first.
        ranking = {'week': 0, 'weekday': 1, 'day': 2, 'hour': 3,
                   'minute': 4, 'second': 5}
        return min(frequency, key=lambda f: ranking[_parse_frequency(f)[1]])
    return frequency


def _parse_frequency(frequency):
    if not isinstance(frequency, str):
        raise ValueError('frequency must be a string or a sequence of strings')
    match = re.fullmatch(r'(?:(\d+(?:\.\d+)?)\s+)?(second|minute|hour|day|week|weekday)', frequency.strip())
    if not match:
        raise ValueError(f'Unsupported completeness frequency: {frequency!r}')
    count = float(match.group(1) or 1)
    if not count.is_integer() or count <= 0:
        raise ValueError('Completeness frequency multiplier must be a positive integer')
    return int(count), match.group(2)


def _interval_bins(timestamps, frequency):
    multiplier, unit = _parse_frequency(frequency)
    series = pd.to_datetime(timestamps)
    if unit == 'weekday':
        # Historical weekday grouping pools weekdays across calendar weeks.
        return (series.dt.weekday // multiplier) * multiplier
    if unit == 'week':
        # Monday-anchored calendar weeks, including multiweek bins.
        starts = series.dt.to_period('W-SUN').dt.start_time
        if multiplier == 1:
            return starts
        epoch = pd.Timestamp('1970-01-05')
        return epoch + ((starts - epoch) // pd.Timedelta(weeks=multiplier)) * pd.Timedelta(weeks=multiplier)
    return series.dt.floor(f'{multiplier}{_UNITS[unit]}')


def analyse_completeness(data, value_cols, frequency, *, time_col='datetime',
                         id_col='identifier', min_completeness=1.0,
                         include_empty_intervals=False):
    """Return one row per (identifier, frequency bin, measurement).

    Columns: identifier (or *id_col*), ``interval``, ``variable``,
    ``observed``, ``expected``, ``missing``, ``completeness``, ``is_complete``,
    ``sampling_interval``. ``observed`` counts non-null measurements, not
    distinct timestamps; ``missing`` is clipped at zero. NaN expected counts
    flag identifiers with insufficient timing information.

    ``include_empty_intervals`` adds bins between each identifier's first and
    last observed bin for ordinary temporal frequencies, except when a bin
    is shorter than the identifier's sampling cadence (unscheduled bins are
    not counted as missing); it does not infer
    missing deployment time before the first or after the last observation.
    """
    if not 0 <= min_completeness <= 1:
        raise ValueError('min_completeness must be between 0 and 1')
    frequency = _primary_frequency(frequency)
    multiplier, unit = _parse_frequency(frequency)
    values = list(value_cols)
    if not values:
        raise ValueError('value_cols must not be empty')
    required = [id_col, time_col, *values]
    absent = [name for name in required if name not in data.columns]
    if absent:
        raise KeyError(f'Missing completeness columns: {absent}')
    frame = pd.DataFrame(data)[required].copy()
    frame[time_col] = pd.to_datetime(frame[time_col], errors='coerce')
    frame = frame.dropna(subset=[id_col, time_col])
    columns = [id_col, 'interval', 'variable', 'observed', 'expected',
               'missing', 'completeness', 'is_complete', 'sampling_interval']
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame['_interval'] = _interval_bins(frame[time_col], frequency)
    # Nonpositive differences (duplicates) cannot describe a sampling cadence.
    ordered = frame.sort_values([id_col, time_col])
    diffs = ordered.groupby(id_col, sort=False)[time_col].diff()
    cadence = (pd.DataFrame({id_col: ordered[id_col].to_numpy(),
                             'gap': diffs.to_numpy()})
               .loc[lambda x: x.gap > pd.Timedelta(0)]
               .groupby(id_col)['gap'].median())
    groups = frame.groupby([id_col, '_interval'], dropna=False, sort=True)[values].count()
    if include_empty_intervals and unit != 'weekday':
        duration = _DURATION[unit] * multiplier
        new_parts = []
        for identifier, portion in frame.groupby(id_col, sort=False):
            first, last = portion['_interval'].min(), portion['_interval'].max()
            # An interval shorter than the monitor cadence is not a scheduled
            # observation interval. Do not manufacture missing sub-cadence bins.
            monitor_cadence = cadence.get(identifier, pd.NaT)
            if pd.notna(monitor_cadence) and duration < monitor_cadence:
                bins = pd.DatetimeIndex(sorted(portion['_interval'].unique()))
            else:
                bins = pd.date_range(first, last, freq=duration)
            new_parts.append(pd.MultiIndex.from_product([[identifier], bins], names=[id_col, '_interval']))
        if new_parts:
            all_bins = new_parts[0]
            for additional in new_parts[1:]:
                all_bins = all_bins.append(additional)
            groups = groups.reindex(all_bins, fill_value=0)
    long = (groups.rename_axis(index=[id_col, 'interval'])
            .reset_index().melt(id_vars=[id_col, 'interval'],
                                value_vars=values, var_name='variable',
                                value_name='observed'))
    long['sampling_interval'] = long[id_col].map(cadence)
    expected = (_DURATION[unit] * multiplier) / long['sampling_interval']
    expected = pd.to_numeric(expected, errors='coerce').round()
    # A bin containing a recorded sample has one expected opportunity even
    # when its width is smaller than the inferred sampling cadence.
    # Such sub-cadence bins without records were excluded above.
    long['expected'] = expected.clip(lower=1)
    long['missing'] = (long['expected'] - long['observed']).clip(lower=0)
    long['completeness'] = long['observed'] / long['expected']
    long['is_complete'] = long['completeness'].ge(min_completeness) & long['expected'].notna()
    return long[columns]


def complete_interval_mask(data, value_cols, frequency, *, time_col='datetime',
                           id_col='identifier', min_completeness=1.0):
    """Row mask preserving legacy group_stat semantics (all values complete)."""
    report = analyse_completeness(data, value_cols, frequency, time_col=time_col,
                                  id_col=id_col, min_completeness=min_completeness)
    complete = report.groupby([id_col, 'interval'])['is_complete'].all()
    intervals = _interval_bins(pd.to_datetime(data[time_col]), _primary_frequency(frequency))
    index = pd.MultiIndex.from_arrays([data[id_col], intervals], names=[id_col, 'interval'])
    return pd.Series(index.isin(complete.index[complete]), index=data.index)
