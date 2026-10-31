"""Figure labels and appearance controls."""
from __future__ import annotations
import re
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES


def _axis_label(variable: str) -> str:
    """Return a variable label with units, including aggregated columns.

    Aggregation produces names such as ``pm25_mean`` or ``CO(ppm)_max``.
    Strip only a *recognized* aggregation suffix, and only when the full
    column name does not already have its own units entry.
    """
    name = str(variable)
    unit = VARIABLE_UNITS.get(name)
    if unit is None:
        for suffix in ("count", "max", "mean", "median", "min", "std", "sum"):
            marker = f"_{suffix}"
            if name.endswith(marker):
                unit = VARIABLE_UNITS.get(name[:-len(marker)])
                if unit is not None:
                    break
    return f"{name} [{unit}]" if unit else name



def _label_plot_axes(fig, plot_type, x_column, y_columns, *,
                     category_columns=None, x_label_override=None):
    """Label plot axes while retaining special histogram and facet labels."""
    if plot_type == "histogram":
        fig.update_xaxes(title_text=_axis_label(x_column))
    elif plot_type in {"box", "bar"}:
        categories = category_columns or [x_column]
        fig.update_xaxes(title_text=" | ".join(_axis_label(c) for c in categories))
        fig.update_yaxes(title_text="<br>".join(_axis_label(y) for y in y_columns))
    elif plot_type == "line" and len(y_columns) > 1:
        fig.update_xaxes(title_text=_axis_label(x_column))
        fig.update_yaxes(title_text="<br>".join(_axis_label(y) for y in y_columns))
        for annotation in fig.layout.annotations:
            label = str(annotation.text)
            if label.startswith("variable="):
                annotation.text = _axis_label(label.split("=", 1)[1])
    else:
        fig.update_xaxes(title_text=_axis_label(x_column))
        if y_columns:
            fig.update_yaxes(title_text=_axis_label(y_columns[0]))
    if x_label_override is not None:
        fig.update_xaxes(title_text=x_label_override)
    return fig



def _parse_label_map(value):
    """Parse semicolon-separated original=replacement entries."""
    result = {}
    for entry in re.split(r"[;\r\n]+", value or ""):
        if "=" in entry:
            key, replacement = entry.split("=", 1)
            if key.strip() and replacement.strip():
                result[key.strip()] = replacement.strip()
    return result



def _numeric_axis_controls(kind, box_orientation="vertical"):
    """Return axes on which numeric/date tick-density controls apply."""
    if kind in ("line", "scatter", "histogram"):
        return True, True
    if kind == "box" and box_orientation == "horizontal":
        return True, False
    return False, kind in ("box", "bar")



def _matplotlib_tick_controls(ax, *, kind, box_orientation="vertical", show_grid=True,
                              x_major_ticks=6, y_major_ticks=6,
                              x_minor_ticks=False, y_minor_ticks=False):
    from matplotlib.ticker import MaxNLocator, AutoMinorLocator
    from matplotlib.dates import AutoDateLocator
    use_x, use_y = _numeric_axis_controls(kind, box_orientation)
    for active, axis, count, minor in ((use_x, ax.xaxis, x_major_ticks, x_minor_ticks),
                                       (use_y, ax.yaxis, y_major_ticks, y_minor_ticks)):
        if not active:
            continue
        count = max(2, min(30, int(count or 6)))
        if axis.get_scale() == "linear":
            if isinstance(axis.get_major_locator(), AutoDateLocator):
                axis.set_major_locator(AutoDateLocator(minticks=max(2, count - 2), maxticks=count))
            else:
                axis.set_major_locator(MaxNLocator(nbins=count))
        if minor:
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(__import__("matplotlib.ticker", fromlist=["NullLocator"]).NullLocator())
        axis.grid(bool(show_grid and minor), which="minor")



def _series_style_maps(line_alpha="", line_style="", scatter_size="", scatter_marker="", scatter_alpha=""):
    """Parse per-trace styles, keyed by the original legend/series name."""
    return tuple(_parse_label_map(v) for v in (line_alpha, line_style, scatter_size, scatter_marker, scatter_alpha))



def _bounded_float(mapping, name, default, low, high):
    try:
        value = float(mapping.get(name, default))
        return value if np.isfinite(value) and low <= value <= high else float(default)
    except (ValueError, TypeError):
        return float(default)



def _line_style(mapping, name):
    return {"solid": "-", "dotted": ":", "dashed": "--", "-": "-", ":": ":", "--": "--"}.get(
        mapping.get(name, "solid").strip().lower(), "-")



def _scatter_marker(mapping, name):
    return {"dot": "o", "cross": "x", "o": "o", "x": "x"}.get(
        mapping.get(name, "dot").strip().lower(), "o")



def _parse_axis_limit(value, *, datetime_axis=False):
    """Parse numeric bounds or an ISO timestamp for a datetime X axis."""
    if value is None or not str(value).strip():
        return None
    if datetime_axis:
        try:
            parsed = pd.to_datetime(value, errors="raise")
            if not pd.isna(parsed):
                return parsed
        except (TypeError, ValueError, OverflowError):
            return None
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None

