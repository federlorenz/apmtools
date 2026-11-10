"""Pure plot-state helpers shared by Shiny and persistence.

These functions deliberately have no reactive dependencies. They make
transitions testable without a running Shiny session.
"""
from __future__ import annotations

SNAPSHOT_INPUT_KEYS = ("plot_type", "x", "y", "numeric_subset_columns", "datetime_start", "datetime_end", "datetime_intervals",
        "group_by", "frequency", "aggregate", "only_complete", "flooring_timestamp",
        "palette", "template", "title", "title_font_size", "plot_dpi", "plot_height_pct", "legend_position",
        "legend_inside", "legend_labels", "legend_columns", "legend_font_size", "multi_y_color_by",
        "x_axis_label", "y_axis_label", "x_tick_labels", "y_tick_labels", "x_min", "x_max", "y_min", "y_max",
        "show_grid", "x_major_ticks", "x_minor_ticks", "y_major_ticks",
        "y_minor_ticks", "axis_font_size", "tick_font_size",
        "line_width", "line_alpha", "line_style", "scatter_size", "scatter_marker",
        "scatter_alpha", "bins", "hist_density", "hist_stacked", "hist_type",
        "hist_cumulative", "hist_alpha", "hist_appearance", "hist_arrangement", "hist_stat", "hist_common_bins", "hist_common_norm", "hist_panel_norm_scope", "hist_bin_mode", "hist_bin_width",
        "hist_range_min", "hist_range_max", "hist_discrete", "hist_kde", "hist_kde_bw", "box_showfliers", "box_orientation", "bar_box_width", "bar_stacking",
        "multi_panel", "independent_panel_limits", "panel_by", "panel_titles",
        "panel_font_size", "panel_hspace", "panel_wspace", "visible_elements")


def normalize_histogram_view(settings):
    """Make legacy Summary.views histogram settings usable by current controls.

    Old keys are retained for external consumers; new keys take precedence.
    This is deliberately a pure function so loading/snapshotting never mutates
    the stored or caller-owned view dictionary.
    """
    settings = dict(settings or {})
    old_keys = {"hist_density", "hist_stacked", "hist_type"}
    if not old_keys.intersection(settings):
        return settings
    old_type = settings.get("hist_type", "bar")
    old_density = str(settings.get("hist_density", "False")).lower() == "true"
    old_stack = str(settings.get("hist_stacked", "False")).lower() == "true"
    settings.setdefault("hist_appearance", {
        "bar": "bars", "barstacked": "bars", "step": "step",
        "stepfilled": "stepfilled"}.get(old_type, "bars"))
    settings.setdefault("hist_arrangement", "stack" if old_stack or old_type == "barstacked" else "layer")
    settings.setdefault("hist_stat", "density" if old_density else "count")
    settings.setdefault("hist_common_bins", True)
    # Matplotlib's legacy density=True with multiple groups used one pooled
    # denominator; preserve that exact behaviour for restored views.
    settings.setdefault("hist_common_norm", bool(old_density))
    return settings



def restore_dynamic_histogram_x(current, saved, *, mounted=False, kind=None):
    """Resolve the dynamically mounted X selector while restoring a histogram.

    An absent/initially empty selector is not a user edit. After the first
    nonempty value is seen, empty selections are intentional and must not be
    replaced by the saved configuration. Other plot types never use this
    restoration fallback.
    """
    empty = current is None or current == '' or current == [] or current == ()
    if empty and not mounted and kind == 'histogram':
        return list(saved) if isinstance(saved, (tuple, list)) else saved
    return current


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
    result = deepcopy(normalize_histogram_view(current))
    # Retired option: retain compatibility when loading older saved plots,
    # but do not write a no-longer-effective width control on new saves.
    result.pop("plot_width_pct", None)
    for key, value in input_values.items():
        if key in SNAPSHOT_INPUT_KEYS and value is not None:
            result[key] = list(value) if isinstance(value, (tuple, list)) else value
    return result


def restore_panel_selection(groups, current, saved, pending=False):
    """Reconcile facet choices without erasing a not-yet-mounted saved facet.

    Returns (selection, pending).  Once restored, user edits take precedence.
    """
    groups = list(groups or [])
    saved = list(saved or [])
    if pending:
        if not all(name in groups for name in saved):
            return [], True
        return [name for name in saved if name in groups], False
    return [name for name in (current or []) if name in groups], False
