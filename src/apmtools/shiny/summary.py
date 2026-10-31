"""Public compatibility facade for the modular Shiny Summary application.

The implementation is organised in columns, data, appearance, renderers,
export and ui modules. Existing imports from summary remain supported.
"""
from __future__ import annotations
from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES
from .columns import _default_x, _default_group, _numeric_columns, _categorical_columns, _datetime_columns, _timedelta_columns, _alphabetical, _normalise_y
from .data import _composite_labels, _subset_choices, _parse_datetime_filter, _apply_filters, _aggregate, _aggregated_value_columns, _frequency_timedelta, _add_gap_sections
from .appearance import _axis_label, _label_plot_axes, _parse_label_map, _numeric_axis_controls, _matplotlib_tick_controls, _series_style_maps, _bounded_float, _line_style, _scatter_marker, _parse_axis_limit
from .matplotlib_renderer import _matplotlib_defaults, _axis_bounds_from_spec, _pad_axis_bounds, _panel_columns, _panel_partitions, _panel_trace_defaults, _panel_axis_bounds, _panel_limit_value, _format_panel_bounds, _matplotlib_direct_figure, _matplotlib_figure
from .export import _safe_csv_filename, _standalone_matplotlib_script
from .ui import create_summary_app, run_summary_app


