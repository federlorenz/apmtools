"""Shared eligibility and aggregation settings for line-plot uncertainty.

Only explicitly aggregated mean series can request computed intervals.  Keep
this rule in the UI, renderer and derived-table replay, so stale saved inputs
do not produce bands on unaggregated data.
"""
from __future__ import annotations

from .view_aggregation import normalize_statistics


def line_bands_enabled(plot_type, aggregate, checked):
    """Whether bands are enabled for the current plot state."""
    return plot_type == "line" and bool(normalize_statistics(aggregate)) and bool(checked)


def line_band_uncertainty(settings):
    """Aggregation kwargs for computed mean-based bands, or None.

    A custom lower/upper pair is never re-estimated.  It remains a renderer
    option, while the aggregation engine only computes requested statistics.
    """
    if not line_bands_enabled(settings.get("plot_type", "line"),
                              settings.get("aggregate"), settings.get("error_bands")):
        return None
    if settings.get("error_band_method", "ci") == "custom":
        return None
    if "mean" not in normalize_statistics(settings.get("aggregate")):
        return None
    return {"method": settings.get("error_band_method", "ci"),
            "confidence": float(settings.get("error_band_confidence", 95) or 95)}
