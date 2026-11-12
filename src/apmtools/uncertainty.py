"""Descriptive uncertainty summaries at a declared unit of observation.

The input rows are the statistical units. Aggregating pre-aggregated rows
therefore describes their variability, not the original underlying readings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t


METHODS = frozenset({'ci', 'sd', 'se', 'percentile'})


def validate_uncertainty(uncertainty):
    """Normalize an optional uncertainty specification without mutating it."""
    if uncertainty is None:
        return None
    if isinstance(uncertainty, str):
        uncertainty = {'method': uncertainty}
    if not isinstance(uncertainty, dict):
        raise ValueError('uncertainty must be None, a method string or a dictionary')
    method = str(uncertainty.get('method', 'ci')).lower()
    if method not in METHODS:
        raise ValueError(f'Unsupported uncertainty method: {method!r}')
    confidence = float(uncertainty.get('confidence', 95))
    if not np.isfinite(confidence) or not 0 < confidence < 100:
        raise ValueError('uncertainty confidence must be between 0 and 100')
    unit = uncertainty.get('unit', 'rows')
    if unit != 'rows':
        raise ValueError('Only unit="rows" is supported; use separate aggregation to define units')
    return {'method': method, 'confidence': confidence, 'unit': unit}


def grouped_uncertainty(frame, keys, value_columns, uncertainty):
    """Return one row per group with mean, SD, N and requested bounds.

    CI/SD/SE and percentile intervals need at least two finite values.
    All-null groups are retained with NaN mean/bounds and N=0.
    """
    options = validate_uncertainty(uncertainty)
    if options is None:
        return None
    keys = list(dict.fromkeys(keys))
    if not keys:
        raise ValueError('Uncertainty aggregation requires at least one grouping key')
    grouped = frame.groupby(keys, dropna=False, sort=True)
    base = grouped.size().rename('__rows').reset_index()[keys]
    method = options['method']
    confidence = options['confidence']
    tail = (100 - confidence) / 200
    for value in value_columns:
        raw = frame[keys].copy()
        raw['__value'] = pd.to_numeric(frame[value], errors='coerce')
        raw['__value'] = raw['__value'].where(np.isfinite(raw['__value']))
        stats = raw.groupby(keys, dropna=False)['__value'].agg(['mean', 'std', 'count']).reset_index()
        n = stats['count'].to_numpy(dtype=float, copy=True)
        mu = stats['mean'].to_numpy(dtype=float, copy=True)
        sd = stats['std'].to_numpy(dtype=float, copy=True)
        if method == 'percentile':
            quantiles = raw.groupby(keys, dropna=False)['__value'].quantile([tail, 1-tail]).unstack(-1).reset_index()
            quantiles = stats[keys].merge(quantiles, on=keys, how='left', sort=False)
            lo = quantiles[tail].to_numpy(dtype=float, copy=True)
            hi = quantiles[1-tail].to_numpy(dtype=float, copy=True)
        else:
            spread = sd.copy()
            if method in ('ci', 'se'):
                spread /= np.sqrt(n)
            if method == 'ci':
                spread *= t.ppf(1-tail, n-1)
            lo, hi = mu-spread, mu+spread
        lo[n < 2] = np.nan
        hi[n < 2] = np.nan
        stats[f'{value}_n'] = stats['count'].astype('int64')
        stats[f'{value}_sd'] = stats['std']
        stats[f'{value}_se'] = stats['std'] / np.sqrt(n)
        stats[f'{value}_{method}_lower'] = lo
        stats[f'{value}_{method}_upper'] = hi
        stats = stats.drop(columns=['mean', 'std', 'count'])
        base = base.merge(stats, on=keys, how='left', sort=False, validate='one_to_one')
    return base
