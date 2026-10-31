"""Pure plot-state helpers shared by Shiny and persistence.

These functions deliberately have no reactive dependencies. They make
transitions testable without a running Shiny session.
"""
from __future__ import annotations

SNAPSHOT_INPUT_KEYS = ("plot_type", "x", "y", "numeric_subset_columns", "datetime_start", "datetime_end",
        "group_by", "frequency", "aggregate", "only_complete", "flooring_timestamp",
        "palette", "template", "title", "title_font_size", "plot_dpi", "legend_position",
        "legend_inside", "legend_labels", "legend_columns", "legend_font_size", "multi_y_color_by",
        "x_axis_label", "y_axis_label", "x_tick_labels", "y_tick_labels", "x_min", "x_max", "y_min", "y_max",
        "show_grid", "x_major_ticks", "x_minor_ticks", "y_major_ticks",
        "y_minor_ticks", "axis_font_size", "tick_font_size",
        "line_width", "line_alpha", "line_style", "scatter_size", "scatter_marker",
        "scatter_alpha", "bins", "hist_density", "hist_stacked", "hist_type",
        "hist_cumulative", "hist_alpha", "box_showfliers", "box_orientation", "bar_box_width",
        "multi_panel", "independent_panel_limits", "panel_by", "panel_titles",
        "panel_font_size", "panel_hspace", "panel_wspace", "visible_elements")


def reconcile_visibility(keys, previous, saved):
    """Return visible keys after a component-list change, or None if transient.

    An empty renderer key set is not an instruction to erase saved visibility.
    New elements default to visible, while previously hidden elements remain
    hidden when the renderer changes grouping or panel layout.
    """
    keys = tuple(keys)
    if not keys:
        return None
    if saved is None:
        return keys
    selected = tuple(key for key in keys if key in saved)
    if previous is None and not selected:
        return keys
    if previous is not None:
        old = set(previous)
        selected = tuple(key for key in keys if key in selected or key not in old)
    return selected


def snapshot_inputs(current, input_values):
    """Merge mounted controls into a detached canonical configuration.

    Unmounted inputs must not erase settings restored from a saved plot.
    """
    from copy import deepcopy
    result = deepcopy(current)
    for key, value in input_values.items():
        if key in SNAPSHOT_INPUT_KEYS and value is not None:
            result[key] = list(value) if isinstance(value, (tuple, list)) else value
    return result
