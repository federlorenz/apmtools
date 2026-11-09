"""Histogram bins and normalization shared by drawing and automatic bounds.

No Shiny dependency: exported scripts can embed this function unchanged.
"""
from __future__ import annotations

import numpy as np



def effective_histogram_bin_mode(values, bin_mode='count', *, range_min=None, range_max=None):
    """Use continuous bins if a discrete request has noninteger data.

    Only observations inside the selected histogram domain matter. The public
    histogram_bin_edges validator remains strict for direct API callers.
    """
    if bin_mode != 'discrete':
        return bin_mode
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if range_min not in (None, ''):
        values = values[values >= float(range_min)]
    if range_max not in (None, ''):
        values = values[values <= float(range_max)]
    return ('discrete' if not len(values) or np.allclose(values, np.rint(values))
            else 'count')


def histogram_bin_edges(values, *, bins=30, bin_mode='count', bin_width=1.0,
                        range_min=None, range_max=None):
    """Select bin boundaries consistently for charts, bounds and facet views.

    The explicit range is a histogram domain (out-of-domain observations are
    excluded), not a post-render axis crop. Discrete bins are integer-centred.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([0., 1.])
    low = float(np.min(values) if range_min in (None, '') else range_min)
    high = float(np.max(values) if range_max in (None, '') else range_max)
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        raise ValueError('Histogram bin range must contain finite, ordered bounds')
    if bin_mode == 'discrete':
        if not np.allclose(values, np.rint(values)):
            raise ValueError('Discrete histogram mode requires integer observations')
        lo, hi = int(np.ceil(low)), int(np.floor(high))
        if hi < lo or hi - lo > 10000:
            raise ValueError('Discrete histogram requires 1–10001 integer bins')
        return np.arange(lo - .5, hi + 1.5, dtype=float)
    if high == low:
        high, low = high + .5, low - .5
    if bin_mode == 'width':
        width = float(bin_width)
        if not np.isfinite(width) or width <= 0:
            raise ValueError('Histogram bin width must be positive and finite')
        count = int(np.ceil((high - low) / width))
        if count > 10000:
            raise ValueError('Histogram bin width would create more than 10000 bins')
        return low + np.arange(max(count, 1) + 1) * width
    if bin_mode not in ('count', 'auto', 'fd', 'scott', 'sturges'):
        raise ValueError(f'Unknown histogram bin mode: {bin_mode}')
    rule = max(1, int(bins)) if bin_mode == 'count' else bin_mode
    edges = np.histogram_bin_edges(values, bins=rule, range=(low, high))
    if len(edges) > 10001:
        raise ValueError('Histogram bin selection would create more than 10000 bins')
    return edges


def histogram_kde(values, grid, *, bandwidth=1.0):
    """Gaussian KDE using NumPy; bounded work and deterministic subsampling.

    Returns density per unit X. Only drawn for continuous layered histograms.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    grid = np.asarray(grid, dtype=float)
    if len(values) < 2 or not len(grid):
        return np.zeros_like(grid)
    factor = float(bandwidth)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('KDE bandwidth factor must be positive and finite')
    if len(values) > 4096:
        values = values[np.linspace(0, len(values) - 1, 4096, dtype=int)]
    sigma = float(np.std(values, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.zeros_like(grid)
    width = max(sigma * len(values) ** (-1 / 5) * factor, np.finfo(float).eps)
    result = np.zeros_like(grid)
    # Avoid allocating grid.size * sample.size when many bins are present.
    for i in range(0, len(grid), 256):
        distance = (grid[i:i + 256, None] - values[None, :]) / width
        result[i:i + 256] = np.exp(-0.5 * distance ** 2).sum(axis=1) / (
            len(values) * width * np.sqrt(2 * np.pi))
    return result


def histogram_heights(arrays, *, bins=30, stat='count', arrangement='layer',
                      common_bins=True, common_norm=False, cumulative=0,
                      common_edges=None, norm_total=None, bin_mode='count',
                      bin_width=1.0, range_min=None, range_max=None):
    """Return edges and bin heights for each dataset.

    Independent bins are supported for layered distributions; arrangements
    requiring bin alignment always use common edges.  Empty inputs are safe.
    """
    if stat not in ('count', 'frequency', 'probability', 'percent', 'density'):
        raise ValueError(f'Unknown histogram statistic: {stat}')
    if arrangement not in ('layer', 'dodge', 'stack', 'fill'):
        raise ValueError(f'Unknown histogram arrangement: {arrangement}')
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    arrays = [a[np.isfinite(a)] for a in arrays]
    # The optional bin range limits the analyzed observations as well as the
    # display domain; probabilities/densities integrate to one over that domain.
    if range_min not in (None, ''):
        arrays = [a[a >= float(range_min)] for a in arrays]
    if range_max not in (None, ''):
        arrays = [a[a <= float(range_max)] for a in arrays]
    pooled = np.concatenate(arrays) if arrays else np.empty(0)
    bin_mode = effective_histogram_bin_mode(pooled, bin_mode)
    if not len(pooled):
        return [], []
    aligned = bool(common_bins or arrangement != 'layer')
    pooled_edges = (np.asarray(common_edges, dtype=float) if common_edges is not None
                    else histogram_bin_edges(pooled, bins=bins, bin_mode=bin_mode,
                        bin_width=bin_width, range_min=range_min, range_max=range_max))
    edges = [pooled_edges if aligned else
             histogram_bin_edges(a, bins=bins, bin_mode=bin_mode,
                                 bin_width=bin_width, range_min=range_min, range_max=range_max)
             if len(a) else pooled_edges
             for a in arrays]
    counts = [np.histogram(a, bins=e)[0].astype(float) for a, e in zip(arrays, edges)]
    total = int(norm_total) if norm_total is not None else sum(len(a) for a in arrays)
    heights = []
    for a, e, n in zip(arrays, edges, counts):
        denominator = total if common_norm else len(a)
        widths = np.diff(e)
        if stat == 'frequency':
            value = np.divide(n, widths, out=np.zeros_like(n), where=widths > 0)
        elif stat == 'density':
            value = np.divide(n, widths * denominator, out=np.zeros_like(n),
                              where=(widths > 0) & (denominator > 0))
        elif stat == 'probability':
            value = n / denominator if denominator else n
        elif stat == 'percent':
            value = n * 100 / denominator if denominator else n
        else:
            value = n
        if cumulative:
            # Density and frequency are per-unit-X quantities: integrate
            # over each bin before accumulating. Probability and percent
            # already represent per-bin mass; count needs no conversion.
            increments = value * widths if stat in ('density', 'frequency') else value
            value = np.cumsum(increments)
            if int(cumulative) < 0:
                value = np.cumsum(increments[::-1])[::-1]
        heights.append(value)
    if arrangement == 'fill':
        matrix = np.stack(heights)
        sums = matrix.sum(axis=0)
        heights = [np.divide(row, sums, out=np.zeros_like(row), where=sums != 0)
                   for row in matrix]
    return edges, heights
