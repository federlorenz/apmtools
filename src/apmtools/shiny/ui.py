"""Shiny application controls, reactive server and launch."""
from __future__ import annotations
from .filter_config import filter_config_fields as _filter_config_fields
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES
from .columns import _alphabetical, _categorical_columns, _datetime_columns, _default_group, _default_x, _normalise_y, _numeric_columns, _timedelta_columns
from .data import _aggregate, _apply_filters, _composite_labels, _subset_choices
from .appearance import _apply_plot_appearance, _label_plot_axes
from .matplotlib_renderer import _axis_bounds_from_spec, _format_panel_bounds, _matplotlib_defaults, _matplotlib_direct_figure, _pad_axis_bounds, _panel_axis_bounds, _panel_partitions, _panel_trace_defaults
from .plotly_renderer import _barplot_figure, _boxplot_figure, _histogram_figure, _lineplot_figure, _scatter_figure
from .elements import element_keys, filter_elements, default_visible_keys
from .export import _safe_csv_filename, _standalone_matplotlib_script


def _build_summary_ui_server(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
    backend: str = "plotly",
    include_theme_picker: bool = True,
    config_manager=None,
):
    """Create, but do not start, an interactive Shiny app for a Summary."""
    from shiny import App, reactive, render, ui, module
    try:
        import shinyswatch
    except ImportError as exc:
        raise ImportError(
            "Summary.plot_app() requires shinyswatch for the app theme selector. "
            'Install the optional dependencies with: pip install "apmtools[shiny]"'
        ) from exc
    if backend not in ("plotly", "matplotlib"):
        raise ValueError("backend must be 'plotly' or 'matplotlib'")
    if backend == "plotly":
        from shinywidgets import output_widget, render_plotly
    # Keep the Summary object intact: subset(), dtfilter() and group_stat()
    # are methods on Summary and are intentionally used below.
    summary_description = str(getattr(data, "description", "") or "")
    summary = data.copy(deep=False)
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame/Summary")
    if summary.empty:
        raise ValueError("Cannot create a Shiny application from an empty Summary.")

    columns = _alphabetical(summary.columns)
    column_map = {str(c): c for c in summary.columns}
    numeric_columns = _alphabetical(_numeric_columns(summary))
    categorical_columns = _categorical_columns(summary)
    datetime_columns = _datetime_columns(summary)
    timedelta_columns = _timedelta_columns(summary)
    time_group_columns = [f for f in FREQUENCIES.values() if f is not None and f not in column_map]
    group_columns = _alphabetical(
        [c for c in columns if not pd.api.types.is_numeric_dtype(summary[column_map[c]])]
        + time_group_columns
    )
    for name in time_group_columns:
        column_map[name] = name

    x = _default_x(summary) if x is None else str(x)
    if x not in column_map:
        raise ValueError(f"Unknown x column: {x!r}")

    initial_y = _normalise_y(summary, x, y)
    if group_by is None:
        initial_groups = [_default_group(summary)] if _default_group(summary) is not None else []
    elif isinstance(group_by, str):
        initial_groups = [group_by]
    else:
        initial_groups = list(group_by)
    unknown_groups = [c for c in initial_groups if c not in column_map]
    if unknown_groups:
        raise ValueError(f"Unknown group_by column(s): {unknown_groups!r}")
    # Numeric columns are not offered by the Group by selector.
    initial_groups = [c for c in initial_groups if c in group_columns]

    if plot_type not in PLOT_TYPES:
        raise ValueError(f"Unknown plot_type {plot_type!r}. Choose one of: {', '.join(PLOT_TYPES)}")

    subset_columns = columns
    max_subset_stages = max(2, len(subset_columns) + 1)
    # Stable choices are shared across tabs. Computing unique values from the
    # full dataset on every filter change was expensive for large summaries.
    subset_choice_cache = {}

    def cached_subset_choices(name):
        if name not in subset_choice_cache:
            subset_choice_cache[name] = _subset_choices(summary, column_map[name])
        return subset_choice_cache[name]

    # The Info tab is the only tab at startup. Plot modules are created lazily
    # when the user presses New plot, so no plotting inputs need to initialize
    # until a plot actually exists.
    info_tab = ui.nav_panel(
        "Info",
        ui.div(
            ui.h2("Summary information"),
            ui.p("Description of the Summary object used to launch this application."),
            ui.tags.pre(summary_description, class_="summary-info-description"),
            ui.h3("Using the plotting application"),
            ui.p("Select + New plot to create Plot 1. Each additional plot opens "
                 "in its own tab. When a plot is selected, New plot copies its "
                 "current settings, which you can then edit independently."),
            ui.h4("Left sidebar — data and plot settings"),
            ui.tags.ul(
                ui.tags.li(ui.strong("Plot settings: "), "Choose the plot type and the X and Y variables."),
                ui.tags.li(ui.strong("Plot filters: "), "Select categorical subsets, numerical "
                           "greater-than and less-than bounds, and start/end datetime limits."),
                ui.tags.li(ui.strong("Group by: "), "Choose one or more grouping variables "
                           "to distinguish data series."),
                ui.tags.li(ui.strong("Frequency / statistical aggregation: "),
                           "Select a time frequency and statistical function for aggregated plots."),
            ),
            ui.h4("Right sidebar — appearance and layout"),
            ui.tags.ul(
                ui.tags.li(ui.strong("Plot appearance: "), "Set the palette, plot theme, "
                           "title and title font size."),
                ui.tags.li(ui.strong("Legend: "), "Control legend position, inside/outside "
                           "placement, labels, columns and font size."),
                ui.tags.li(ui.strong("Axes: "), "Set axis labels, minimum and maximum "
                           "values, grid, tick controls and fonts."),
                ui.tags.li(ui.strong("Plot components: "), "Customize trace colours, "
                           "lines, markers, histograms and boxplot options."),
                ui.tags.li(ui.strong("Panel appearance: "), "Configure multi-panel "
                           "plots and panel titles (Matplotlib backend)."),
            ),
            ui.h4("Plot tabs, export and application theme"),
            ui.p("Use + New plot to add a plot tab, or Close selected plot to "
                 "remove the current plot. The Info tab remains available."),
            ui.p("Export plotted data as CSV saves the data displayed in the plot. "
                 "Export plot saves the chart. With the Matplotlib backend, "
                 "Export as a Python file and associated data creates a "
                 "reproducible script and its data."),
            ui.p("Use the theme selector at the top right to change the Shiny "
                 "application's appearance. This is separate from the plot theme "
                 "in the Plot appearance tab."),
            class_="summary-info-content",
        ),
        value="info",
    )

    @module.ui
    def plot_tab_ui(initial_filters=None):
        initial_filters = initial_filters or {}
        def initial(name, default):
            return initial_filters.get(name, default)
        return ui.div(
            ui.tags.style("""
.summary-sidebar-tabs .nav-tabs {display:flex; flex-wrap:wrap; gap:3px; border-bottom:0; margin-bottom:0.5rem;}
.summary-sidebar-tabs .nav-link {border:0; border-radius:0.4rem; padding:0.45rem 0.65rem; font-size:0.82rem; font-weight:600; color:#17212b; background:var(--tab-color,#dce8f7);}
.summary-sidebar-tabs .nav-link.active {background:var(--tab-color,#dce8f7); color:#17212b; box-shadow:inset 0 -3px 0 rgba(0,0,0,.35);}
.summary-sidebar-tabs .nav-item:nth-child(5n+1) {--tab-color:#a7d7f3;}
.summary-sidebar-tabs .nav-item:nth-child(5n+2) {--tab-color:#b8e5c5;}
.summary-sidebar-tabs .nav-item:nth-child(5n+3) {--tab-color:#ffe0a3;}
.summary-sidebar-tabs .nav-item:nth-child(5n+4) {--tab-color:#d8c5f2;}
.summary-sidebar-tabs .nav-item:nth-child(5n+5) {--tab-color:#f7c7d5;}
.summary-sidebar-tabs .tab-content {padding:0.65rem 0.25rem;}
"""),
            ui.tags.style("""
.summary-color-row {display:flex;justify-content:space-between;align-items:center;gap:8px;margin:7px 0;}
.summary-color-name {overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;font-size:.85rem;}
.summary-color-trigger {display:inline-flex;align-items:center;gap:7px;white-space:nowrap;}
.summary-color-swatch {display:inline-block;width:27px;height:23px;border:1px solid #888;border-radius:4px;}
.summary-color-hex {font-family:monospace;font-size:.8rem;}
.summary-native-color-picker {display:block;width:100%;height:110px;cursor:pointer;margin:8px 0 14px;}
.summary-color-presets {display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:14px 0;}
.summary-color-preset {height:32px;border:1px solid #888;border-radius:5px;cursor:pointer;}
"""),
            ui.tags.script("""
// Event delegation survives Shiny's dynamic modal and panel replacement.
if (!window.__apmColorPickerInstalled) {
  window.__apmColorPickerInstalled = true;
  document.addEventListener('input', function(ev) {
    if (!ev.target.matches('.summary-native-color-picker')) return;
    const field = document.getElementById(ev.target.dataset.hexTarget);
    if (field) {
      field.value = ev.target.value.toUpperCase();
      field.dispatchEvent(new Event('input', {bubbles:true}));
      field.dispatchEvent(new Event('change', {bubbles:true}));
    }
  });
  document.addEventListener('click', function(ev) {
    const swatch = ev.target.closest('.summary-color-preset');
    if (!swatch) return;
    ev.preventDefault();
    const field = document.getElementById(swatch.dataset.hexTarget);
    if (field) {
      field.value = swatch.dataset.colorValue;
      field.dispatchEvent(new Event('input', {bubbles:true}));
      field.dispatchEvent(new Event('change', {bubbles:true}));
    }
    const picker = document.querySelector('.summary-native-color-picker');
    if (picker) picker.value = swatch.dataset.colorValue;
  });
  document.addEventListener('input', function(ev) {
    if (!ev.target.matches('input[id$="picker_hex"]')) return;
    if (!/^#[0-9a-f]{6}$/i.test(ev.target.value)) return;
    const picker = document.querySelector('.summary-native-color-picker');
    if (picker) picker.value = ev.target.value;
  });
}
"""),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.div(ui.navset_tab(
                    ui.nav_panel("Plot settings",
                        ui.input_select("plot_type", "Plot type", choices=PLOT_TYPES, selected=initial("plot_type", plot_type)),
                        ui.output_ui("x_control"),
                        ui.output_ui("y_control")),
                    ui.nav_panel("Plot filters",
                        *[
                            ui.div(
                                ui.input_select(f"subset_column_{i}", f"Subset column {i + 1}",
                                    choices={"": "Select a column", **{c: c for c in subset_columns}},
                                    selected=(initial("subset_chain", [{}] * max_subset_stages)[i].get("column", "")
                                        if i < len(initial("subset_chain", [])) else "")),
                                ui.output_ui(f"subset_values_ui_{i}"),
                                ui.input_select(f"subset_style_{i}", "Filter style",
                                    choices={k: k for k in ("all", "any", "negative all", "negative any")},
                                    selected=(initial("subset_chain", [])[i].get("style", "all")
                                        if i < len(initial("subset_chain", [])) else "all")),
                                class_="mb-3",
                            ) if i == 0 else ui.panel_conditional(
                                " && ".join(f"input.subset_column_{j} !== ''" for j in range(i)),
                                ui.div(
                                    ui.input_select(f"subset_column_{i}", f"Subset column {i + 1}",
                                        choices={"": "Select a column", **{c: c for c in subset_columns}},
                                        selected=(initial("subset_chain", [])[i].get("column", "")
                                            if i < len(initial("subset_chain", [])) else "")),
                                    ui.output_ui(f"subset_values_ui_{i}"),
                                    ui.input_select(f"subset_style_{i}", "Filter style",
                                        choices={k: k for k in ("all", "any", "negative all", "negative any")},
                                        selected=(initial("subset_chain", [])[i].get("style", "all")
                                            if i < len(initial("subset_chain", [])) else "all")),
                                    class_="mb-3",
                                ),
                            )
                            for i in range(max_subset_stages)
                        ],
                        ui.input_selectize("numeric_subset_columns", "Numerical subset",
                            choices={c: c for c in numeric_columns},
                            selected=initial("numeric_subset_columns", []), multiple=True),
                        ui.output_ui("numeric_subset_bounds"),
                        ui.input_text("datetime_start", "Start datetime", value=initial("datetime_start", ""),
                            placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                        ui.input_text("datetime_end", "End datetime", value=initial("datetime_end", ""),
                            placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                        ui.help_text("Filters affect this plot only. New plots inherit the selected plot settings.")),
                    ui.nav_panel("Group by",
                        ui.input_selectize("group_by", "Group by", choices={c:c for c in group_columns},
                            selected=initial("group_by", []), multiple=True)),
                    ui.nav_panel("Frequency / statistical aggregation",
                        ui.input_select("frequency", "Frequency / aggregation mode",
                            choices={key: key for key in FREQUENCIES}, selected=initial("frequency", "None")),
                        ui.input_select("aggregate", "Apply statistical function",
                            choices={key: key for key in AGGREGATES}, selected=initial("aggregate", "None")),
                        ui.help_text("Choose a statistical function to aggregate by the selected groups, even when frequency is None."),
                        ui.output_ui("frequency_options")),
                    id="left_sidebar_tabs"), class_="summary-sidebar-tabs"),
                    title="Plot controls", position="left", open="open", width="320px",
                ),
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.div(ui.navset_tab(
                        ui.nav_panel("Plot appearance",
                            ui.input_select("palette", "Color palette", choices=["Default", "Plotly", "D3", "G10", "T10", "Set1", "Set2", "Pastel", "Dark24"], selected=initial("palette", "Default")),
                            ui.input_select("template", "Theme", choices={"plotly":"Light", "plotly_white":"White", "plotly_dark":"Dark", "simple_white":"Minimal", "ggplot2":"GGPlot"}, selected=initial("template", "plotly")),
                            ui.input_text("title", "Plot title", value=initial("title", "")),
                            ui.input_numeric("title_font_size", "Plot title font size", value=initial("title_font_size", 14), min=6, max=72),
                            (ui.div(
                                ui.help_text("Adjust all plot font sizes together (1 point per click)."),
                                ui.div(ui.input_action_button("decrease_plot_fonts", "− Decrease fonts"),
                                       ui.input_action_button("increase_plot_fonts", "+ Increase fonts"),
                                       class_="d-flex flex-wrap gap-2 mb-3"),
                                ui.input_numeric("plot_dpi", "Plot DPI", value=initial("plot_dpi", 100), min=30, max=600, step=25),
                                ui.div(ui.input_action_button("decrease_plot_dpi", "− DPI"),
                                       ui.input_action_button("increase_plot_dpi", "+ DPI"),
                                       class_="d-flex flex-wrap gap-2"),
                                ui.help_text("DPI affects Matplotlib plot rendering. Export DPI is set separately when exporting."),
                            ) if backend == "matplotlib" else ui.div())),
                        ui.nav_panel("Legend",
                            ui.panel_conditional("(input.plot_type === 'line' || input.plot_type === 'box' || input.plot_type === 'bar' || input.plot_type === 'scatter') && input.y && input.y.length > 1", ui.input_select("multi_y_color_by", "Multiple Y variables: colour by", choices={"group":"Grouping (default)", "variable":"Y variable", "both":"Both", "none":"None"}, selected=initial("multi_y_color_by", "group"))),
                            ui.input_select("legend_position", "Legend position", choices={k:k.title() for k in ("right", "left", "top", "bottom")}, selected=initial("legend_position", "right")),
                            ui.input_checkbox("legend_inside", "Place legend inside plot", value=initial("legend_inside", False)),
                            ui.output_ui("legend_labels_editor"),
                            (ui.input_numeric("legend_columns", "Legend columns", value=initial("legend_columns", 1), min=1, max=100, step=1) if backend == "matplotlib" else ui.div()),
                            ui.input_numeric("legend_font_size", "Legend font size", value=initial("legend_font_size", 12), min=6, max=72)),
                        ui.nav_panel("Axes",
                            ui.input_text("x_axis_label", "X-axis label", value=initial("x_axis_label", "")),
                            ui.input_text_area("y_axis_label", "Y-axis label", value=initial("y_axis_label", ""), rows=3),
                            ui.layout_columns(
                                (ui.input_text_area("x_min", "X minimum", value=initial("x_min", ""), rows=4) if backend == "matplotlib" else ui.input_text("x_min", "X minimum", value="")),
                                (ui.input_text_area("x_max", "X maximum", value=initial("x_max", ""), rows=4) if backend == "matplotlib" else ui.input_text("x_max", "X maximum", value="")),
                                col_widths=(6, 6)),
                            ui.layout_columns(
                                (ui.input_text_area("y_min", "Y minimum", value=initial("y_min", ""), rows=4) if backend == "matplotlib" else ui.input_text("y_min", "Y minimum", value="")),
                                (ui.input_text_area("y_max", "Y maximum", value=initial("y_max", ""), rows=4) if backend == "matplotlib" else ui.input_text("y_max", "Y maximum", value="")),
                                col_widths=(6, 6)),
                            ui.output_ui("x_tick_labels_editor"),
                            ui.output_ui("y_tick_labels_editor"),
                            ui.input_checkbox("show_grid", "Show grid", value=initial("show_grid", True)),
                            ui.panel_conditional("input.plot_type === 'line' || input.plot_type === 'scatter' || input.plot_type === 'histogram' || (input.plot_type === 'box' && input.box_orientation === 'horizontal')",
                                ui.input_numeric("x_major_ticks", "X-axis major ticks (approximate)", value=initial("x_major_ticks", 6), min=2, max=30, step=1),
                                ui.input_checkbox("x_minor_ticks", "X-axis minor ticks / gridlines", value=initial("x_minor_ticks", False))),
                            ui.panel_conditional("input.plot_type === 'line' || input.plot_type === 'scatter' || input.plot_type === 'histogram' || input.plot_type === 'bar' || (input.plot_type === 'box' && input.box_orientation !== 'horizontal')",
                                ui.input_numeric("y_major_ticks", "Y-axis major ticks (approximate)", value=initial("y_major_ticks", 6), min=2, max=30, step=1),
                                ui.input_checkbox("y_minor_ticks", "Y-axis minor ticks / gridlines", value=initial("y_minor_ticks", False))),
                            ui.input_numeric("axis_font_size", "Axis title font size", value=initial("axis_font_size", 12), min=6, max=72),
                            ui.input_numeric("tick_font_size", "Axis tick font size", value=initial("tick_font_size", 10), min=6, max=72)),
                        ui.nav_panel("Plot components",
                            ui.output_ui("element_visibility_ui"),
                            ui.output_ui("trace_colors_editor"),
                            ui.panel_conditional("input.plot_type === 'line'",
                                ui.input_numeric("line_width", "Line width", value=initial("line_width", 2), min=0.5, max=15, step=0.5),
                                ui.input_text_area("line_alpha", "Line alpha (name=0..1; ...)", value=initial("line_alpha", ""), rows=4),
                                ui.input_text_area("line_style", "Line style (name=solid/dotted/dashed; ...)", value=initial("line_style", ""), rows=4)),
                            ui.panel_conditional("input.plot_type === 'scatter'",
                                ui.input_text_area("scatter_size", "Marker size (name=size; ...)", value=initial("scatter_size", ""), rows=4),
                                ui.input_text_area("scatter_marker", "Marker type (name=dot/cross; ...)", value=initial("scatter_marker", ""), rows=4),
                                ui.input_text_area("scatter_alpha", "Marker alpha (name=0..1; ...)", value=initial("scatter_alpha", ""), rows=4)),
                            ui.panel_conditional("input.plot_type === 'histogram'",
                                ui.input_numeric("bins", "Histogram bins", value=initial("bins", 30), min=1, max=200),
                                ui.input_select("hist_density", "Density", choices={"False":"False","True":"True"}, selected=initial("hist_density", "False")),
                                ui.input_select("hist_stacked", "Stacked", choices={"False":"False","True":"True"}, selected=initial("hist_stacked", "False")),
                                ui.input_select("hist_type", "Histogram type", choices={"bar":"bar","barstacked":"barstacked","step":"step","stepfilled":"stepfilled"}, selected=initial("hist_type", "bar")),
                                ui.input_select("hist_cumulative", "Cumulative", choices={"0":"False","1":"True (ascending)","-1":"True (descending)"}, selected=initial("hist_cumulative", "0")),
                                ui.input_text_area("hist_alpha", "Histogram alpha (name=0..1; ...)", value=initial("hist_alpha", ""), rows=4)),
                            ui.panel_conditional("input.plot_type === 'bar' || input.plot_type === 'box'",
                                ui.input_numeric("bar_box_width", "Bar / box width", min=0.20, max=3.00,
                                                 value=initial("bar_box_width", 1.0), step=0.05)),
                            ui.panel_conditional("input.plot_type === 'box'",
                                ui.input_checkbox("box_showfliers", "Show outliers (fliers)", value=initial("box_showfliers", True)),
                                ui.input_select("box_orientation", "Box orientation", choices={"vertical":"Vertical", "horizontal":"Horizontal"}, selected=initial("box_orientation", "vertical")))),
                        ui.nav_panel("Panel appearance",
                            (ui.div(
                                ui.panel_conditional("input.group_by && input.group_by.length > 0",
                                    ui.input_checkbox("multi_panel", "Separate panels by group", value=initial("multi_panel", False)),
                                    ui.panel_conditional("input.multi_panel === true",
                                        ui.input_checkbox("independent_panel_limits",
                                            "Independent axis limits for each panel", value=initial("independent_panel_limits", True))),
                                    ui.panel_conditional("input.multi_panel === true && input.group_by.length > 1",
                                        ui.input_selectize("panel_by", "Panel grouping variables",
                                            choices={name: name for name in group_columns}, selected=initial("panel_by", []), multiple=True))),
                                ui.panel_conditional("input.multi_panel === true",
                                    ui.output_ui("panel_titles_editor"),
                                    ui.input_numeric("panel_hspace", "Vertical panel spacing", value=initial("panel_hspace", 0.90), min=0, max=5, step=0.05),
                                    ui.input_numeric("panel_wspace", "Horizontal panel spacing", value=initial("panel_wspace", 0.38), min=0, max=5, step=0.05)),
                                ui.input_numeric("panel_font_size", "Panel title font size", value=initial("panel_font_size", 14), min=6, max=72))
                                if backend == "matplotlib" else ui.help_text("Panel formatting is available with the Matplotlib backend."))),
                        id="right_sidebar_tabs"), class_="summary-sidebar-tabs"),
                        title="Plot appearance", position="right", open="closed", width="320px",
                    ),
                    ui.card(ui.card_header("Plot"),
                        ui.div((output_widget("plot", height="100%", width="100%") if backend == "plotly" else ui.output_plot("plot", height="100%", width="100%")),
                            class_="summary-plot-container"),
                        ui.input_action_button("export_csv", "Export plotted data as CSV", class_="btn-primary mt-3"),
                        ui.input_action_button("export_plot", "Export plot", class_="btn-secondary mt-3"),
                        (ui.download_button("export_python", "Export as a Python file and associated data",
                                            class_="btn-secondary mt-3") if backend == "matplotlib" else ui.div())),
                ),
            ),
        )

    @module.server
    def plot_tab_server(input, output, session, initial_filters=None, config_tab_id=None):
            initial_filters = initial_filters or {}
            # One stable output per stage: changing a later stage does not
            # recreate the inputs belonging to earlier stages.
            def _make_subset_values_ui(stage):
                def _values_ui():
                    try:
                        name = input[f"subset_column_{stage}"]()
                    except Exception:
                        name = ""
                    if name not in column_map:
                        return ui.div()
                    choices, _ = cached_subset_choices(name)
                    saved = _live_setting("subset_chain", []) or []
                    selected = (saved[stage].get("values", [])
                                if stage < len(saved) and saved[stage].get("column") == name else [])
                    return ui.input_selectize(f"subset_values_{stage}", f"{name} values",
                                              choices=choices, selected=selected, multiple=True)
                # Shiny registers outputs by the decorated function name.
                # Give each generated output its own stable name before rendering.
                _values_ui.__name__ = f"subset_values_ui_{stage}"
                return render.ui(_values_ui)

            for _stage in range(max_subset_stages):
                output(_make_subset_values_ui(_stage))

            @render.ui
            def numeric_subset_bounds():
                controls = []
                for name in list(input.numeric_subset_columns() or []):
                    if name not in numeric_columns:
                        continue
                    # Stable IDs per column preserve entered bounds when selections change.
                    index = numeric_columns.index(name)
                    bounds = (_live_setting("numeric_subset_bounds", {}) or {}).get(name, {})
                    controls.append(ui.div(
                        ui.tags.strong(name),
                        ui.input_text(f"numeric_gt_{index}", "Greater than",
                            value=str(bounds.get("gt", "")), placeholder="No lower bound"),
                        ui.input_text(f"numeric_lt_{index}", "Less than",
                            value=str(bounds.get("lt", "")), placeholder="No upper bound"),
                        class_="mb-3",
                    ))
                return ui.div(*controls)


            def _restored_subset_chain():
                """Read mounted controls, falling back to saved values if absent.

                A hidden/dynamic input may not exist until its UI output mounts.
                An existing empty selection, however, is an intentional user edit.
                """
                saved = _live_setting("subset_chain", []) or []
                stages = []
                for i in range(max_subset_stages):
                    prior = saved[i] if i < len(saved) else {}
                    try:
                        name = input[f"subset_column_{i}"]()
                    except Exception:
                        name = prior.get("column", "")
                    if name not in column_map:
                        break
                    same_column = name == prior.get("column")
                    try:
                        keys = list(input[f"subset_values_{i}"]() or [])
                    except Exception:
                        keys = list(prior.get("values", []) or []) if same_column else []
                    try:
                        style = input[f"subset_style_{i}"]() or "all"
                    except Exception:
                        style = prior.get("style", "all") if same_column else "all"
                    stages.append({"column": name, "values": keys, "style": style})
                return stages

            def _restored_numeric_bounds():
                saved = _live_setting("numeric_subset_bounds", {}) or {}
                try:
                    names = list(input.numeric_subset_columns() or [])
                except Exception:
                    names = list(_live_setting("numeric_subset_columns", []) or [])
                result = {}
                for name in names:
                    if name not in numeric_columns:
                        continue
                    index = numeric_columns.index(name)
                    result[name] = {}
                    for suffix in ("gt", "lt"):
                        try:
                            raw = input[f"numeric_{suffix}_{index}"]()
                        except Exception:
                            raw = saved.get(name, {}).get(suffix, "")
                        result[name][suffix] = raw
                return result

            def _patch_live_config(tab_id, changes):
                """Update a tab draft with older and newer config managers."""
                if hasattr(config_manager, "patch"):
                    return config_manager.patch(tab_id, changes)
                current = config_manager.get(tab_id) if tab_id in config_manager.drafts else {}
                current.update(changes)
                return config_manager.update(tab_id, current)

            def _live_setting(name, default=None):
                """Read the current tab draft, without subscribing to draft writes.

                Shiny controls remain the source of user events during migration;
                the canonical draft supplies values for unmounted controls.
                """
                if config_manager is not None and config_tab_id is not None:
                    with reactive.isolate():
                        if config_tab_id in config_manager.drafts:
                            return config_manager.value(config_tab_id, name, default) if hasattr(config_manager, "value") else config_manager.get(config_tab_id).get(name, default)
                return initial_filters.get(name, default)

            @reactive.calc
            def filter_state():
                def text_value(name):
                    try:
                        return str(getattr(input, name)() or "").strip()
                    except Exception:
                        return str(_live_setting(name, "") or "").strip()

                return {"stages": _restored_subset_chain(),
                        "numeric_bounds": _restored_numeric_bounds(),
                        "start": text_value("datetime_start"),
                        "end": text_value("datetime_end")}


            def _commit_filter_event():
                """Commit a user filter event without subscribing to configuration writes.

                Reading all effective controls in an isolated scope prevents an
                unrelated input from becoming a dependency of this event handler.
                Missing dynamic controls retain their canonical draft values.
                """
                if config_manager is None or config_tab_id is None:
                    return
                with reactive.isolate():
                    changes = _filter_config_fields(filter_state())
                    _patch_live_config(config_tab_id, changes)

            def _register_filter_event(input_name):
                """Register a per-control event, including dynamically mounted inputs."""
                def _event_value():
                    try:
                        return input[input_name]()
                    except Exception:
                        return None

                @reactive.effect
                @reactive.event(_event_value, ignore_init=True)
                def _on_filter_change():
                    _commit_filter_event()

                return _on_filter_change

            # Retain references for the lifetime of this plot module. Unlike a
            # broad synchronization effect, each handler observes one UI input;
            # no handler subscribes to the draft it writes.
            _filter_event_handlers = []
            for _name in ("numeric_subset_columns", "datetime_start", "datetime_end"):
                _filter_event_handlers.append(_register_filter_event(_name))
            for _stage in range(max_subset_stages):
                for _prefix in ("subset_column", "subset_values", "subset_style"):
                    _filter_event_handlers.append(
                        _register_filter_event(f"{_prefix}_{_stage}")
                    )
            for _index in range(len(numeric_columns)):
                for _suffix in ("gt", "lt"):
                    _filter_event_handlers.append(
                        _register_filter_event(f"numeric_{_suffix}_{_index}")
                    )

            @reactive.calc
            def filtered_summary():
                state = filter_state()
                # Apply each stage sequentially through Summary.subset().
                # Empty value selections are intentionally no-ops.
                stages = state["stages"]
                # Preserve original row adjacency before numerical filtering so a
                # line cannot bridge observations removed by a numeric bound.
                source = summary
                if input.plot_type() == "line" and AGGREGATES.get(input.aggregate()) is None and state["numeric_bounds"]:
                    source = summary.copy()
                    order = source.sort_values((["identifier"] if "identifier" in source.columns else []) + ["datetime"], kind="stable")
                    if "identifier" in order.columns:
                        source.loc[order.index, "__source_row_order"] = order.groupby("identifier", dropna=False).cumcount().to_numpy()
                    else:
                        source.loc[order.index, "__source_row_order"] = np.arange(len(order))
                current = source
                for stage in stages:
                    _, mapping = cached_subset_choices(stage["column"])
                    selected = [mapping[key] for key in stage["values"] if key in mapping]
                    if selected:
                        current = current.subset(
                            {column_map[stage["column"]]: selected},
                            filter_style=stage["style"],
                        )
                current = _apply_filters(current, {}, state["start"], state["end"])
                for name, bounds in state["numeric_bounds"].items():
                    values = pd.to_numeric(current[column_map[name]], errors="coerce")
                    for suffix, comparison in (("gt", "gt"), ("lt", "lt")):
                        raw = bounds.get(suffix, "")
                        try:
                            bound = float(str(raw).strip())
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(bound):
                            continue
                        mask = values.gt(bound) if comparison == "gt" else values.lt(bound)
                        current = current.loc[mask.fillna(False) | values.isna()].copy()
                        values = values.loc[current.index]
                return current

            # Individual editors: original labels are stable keys, never parsed
            # from user-entered text. Changes invalidate only figure styling.
            import hashlib
            _label_fields = ("legend_labels", "panel_titles", "x_tick_labels", "y_tick_labels")
            _label_keys = {name: reactive.value(()) for name in _label_fields}
            from .appearance import _parse_label_map
            def _editor_map(field, *, watch=False):
                # The tab draft owns edited mappings; no parallel reactive dictionary.
                if config_manager is not None and config_tab_id is not None:
                    if watch and config_tab_id in config_manager.drafts:
                        config_manager.watch(config_tab_id, "appearance")
                    return _parse_label_map(_live_setting(field, initial_filters.get(field, "")))
                return _parse_label_map(initial_filters.get(field, ""))

            def _set_label_keys(field, keys):
                keys = tuple(dict.fromkeys(str(k) for k in keys if str(k).strip()))
                with reactive.isolate():
                    previous = _label_keys[field].get()
                if keys != previous:
                    _label_keys[field].set(keys)

            def _label_string(field):
                # Internal adapter for the existing renderer/export APIs.
                # Label discovery is downstream of figure rendering. Do not
                # subscribe the figure to its own discovered key set: this
                # otherwise forms plot -> legend keys -> plot feedback cycles.
                with reactive.isolate():
                    keys = _label_keys[field].get()
                values = _editor_map(field)
                active = keys if keys else tuple(values)
                return "; ".join(f"{key}={values.get(key, key)}" for key in active
                                 if values.get(key, key) != key)

            for _field in _label_fields:
                def _make_editor(field):
                    @output(id=f"{field}_editor")
                    @render.ui
                    def _editor():
                        keys = _label_keys[field].get()
                        with reactive.isolate():
                            values = _editor_map(field, watch=True)
                        headings = {
                            "legend_labels": "Legend labels",
                            "panel_titles": "Panel titles",
                            "x_tick_labels": "X tick labels",
                            "y_tick_labels": "Y tick labels",
                        }
                        return ui.div(
                            ui.h6(headings[field]),
                            *[
                                ui.input_text(
                                    f"edit_{field}_{hashlib.sha1(key.encode()).hexdigest()[:16]}",
                                    None, value=values.get(key, key)) for key in keys
                            ] if keys else [ui.help_text("No displayed labels")],
                        )
                _make_editor(_field)

                def _make_watcher(field):
                    @reactive.effect
                    def _watch():
                        keys = _label_keys[field].get()
                        # A plot-type transition unmounts the old inputs asynchronously.
                        # Do not consume values from the previous plot's editors while
                        # Shiny is replacing their DOM nodes.
                        with reactive.isolate():
                            if not keys:
                                return
                        # Read current mappings without subscribing this watcher to
                        # its own writes: otherwise each keystroke can retrigger it.
                        with reactive.isolate():
                            updated = dict(_editor_map(field))
                        changed = False
                        for key in keys:
                            ident = f"edit_{field}_{hashlib.sha1(key.encode()).hexdigest()[:16]}"
                            value = _optional_input(ident, None)
                            if value is not None and updated.get(key, key) != value:
                                updated[key] = value
                                changed = True
                        if changed:
                            if config_manager is not None and config_tab_id is not None:
                                _patch_live_config(config_tab_id, {
                                    field: "; ".join(f"{k}={v}" for k, v in updated.items() if v != k)
                                })
                _make_watcher(_field)

            # Colour keys describe the actual colour encoding, not the number
            # of categorical boxes/points.  Facet variables are removed before
            # determining the active within-panel grouping.
            from .color_editor import active_color_keys, color_defaults, valid_hex_color
            _color_keys = reactive.value(())
            def _color_map(*, watch=False):
                return _editor_map("trace_colors", watch=watch)

            def _color_string():
                keys = _color_keys.get()
                values = _color_map(watch=True)
                return "; ".join(f"{key}={values[key]}" for key in keys if key in values)

            # Each rendered colour key has a stable swatch button. The popup
            # itself is mounted only on demand, avoiding text-input remounts
            # and the feedback loop previously seen in the label editors.
            _picker_key = reactive.value(None)
            _picker_defaults = ("#636EFA", "#EF553B", "#00CC96", "#AB63FA",
                                "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
                                "#FF97FF", "#FECB52", "#000000", "#FFFFFF")

            @output(id="trace_colors_editor")
            @render.ui
            def trace_colors_editor():
                keys = _color_keys.get()
                values = _color_map(watch=True)
                with reactive.isolate():
                    palette = _optional_input("palette", "Default")
                    spec = prepared_plot()
                    defaults = color_defaults(spec, keys, palette)
                return ui.div(
                    ui.h6("Trace colors"),
                    *[ui.div(
                        ui.tags.span(k, class_="summary-color-name", title=k),
                        ui.input_action_button(
                            f"pick_trace_color_{hashlib.sha1(k.encode()).hexdigest()[:16]}",
                            ui.tags.span(
                                ui.tags.span("", class_="summary-color-swatch",
                                             style=f"background-color:{values.get(k, defaults.get(k, '#636efa'))}"),
                                ui.tags.span(values.get(k, defaults.get(k, "#636efa")),
                                             class_="summary-color-hex"),
                            ),
                            class_="summary-color-trigger btn btn-outline-secondary btn-sm"),
                        class_="summary-color-row") for k in keys]
                    if keys else [ui.help_text("No coloured elements")],
                )

            _picker_click_counts = {}
            @reactive.effect
            def _watch_picker_buttons():
                keys = _color_keys.get()
                clicked = None
                for key in keys:
                    field = f"pick_trace_color_{hashlib.sha1(key.encode()).hexdigest()[:16]}"
                    count = int(_optional_input(field, 0) or 0)
                    previous = _picker_click_counts.get(key, 0)
                    if count > previous:
                        clicked = key
                    _picker_click_counts[key] = count
                if clicked is None:
                    return
                with reactive.isolate():
                    values = _color_map()
                    defaults = color_defaults(prepared_plot(), (clicked,),
                                              _optional_input("palette", "Default"))
                    current = values.get(clicked, defaults.get(clicked, "#636efa"))
                _picker_key.set(clicked)
                ui.modal_show(ui.modal(
                    ui.p(f"Choose colour for {clicked}"),
                    ui.tags.input(type="color", value=current,
                                  class_="summary-native-color-picker",
                                  **{"data-hex-target": session.ns("picker_hex")}),
                    ui.input_text("picker_hex", "Hex colour", value=current),
                    ui.div(*[ui.tags.button("", type="button",
                        class_="summary-color-preset",
                        title=color, style=f"background-color:{color}",
                        **{"data-color-value": color,
                           "data-hex-target": session.ns("picker_hex")})
                        for color in _picker_defaults],
                        class_="summary-color-presets"),
                    title="Trace colour", easy_close=True,
                    footer=ui.div(ui.modal_button("Cancel"),
                        ui.input_action_button("picker_apply", "Apply colour",
                                               class_="btn-primary")),
                ))

            @reactive.effect
            @reactive.event(input.picker_apply)
            def _apply_picked_color():
                key = _picker_key.get()
                value = str(_optional_input("picker_hex", "")).strip()
                if key is not None and valid_hex_color(value):
                    with reactive.isolate():
                        values = dict(_color_map())
                    values[key] = value.upper()
                    if config_manager is not None and config_tab_id is not None:
                        _patch_live_config(config_tab_id, {
                            "trace_colors": "; ".join(f"{k}={v}" for k, v in values.items())
                        })
                    ui.modal_remove()
                else:
                    ui.notification_show("Enter a valid six-digit hex colour (#RRGGBB).",
                                         type="error", duration=4)

            @reactive.effect
            def _sync_color_keys():
                spec = prepared_plot()
                if "error" in spec:
                    keys = ()
                else:
                    groups = list(input.group_by() or [])
                    faceted = backend == "matplotlib" and bool(groups) and bool(_optional_input("multi_panel", False))
                    panel_by = ((list(_optional_input("panel_by", []) or []) or groups)
                                if faceted and len(groups) > 1 else groups if faceted else [])
                    keys = active_color_keys(spec, mode=_optional_input("multi_y_color_by", "group"),
                                             panel_by=panel_by)
                with reactive.isolate():
                    previous = _color_keys.get()
                if keys != previous:
                    _color_keys.set(keys)

            @reactive.effect
            @reactive.event(input.export_csv)
            def show_export_dialog():
                ui.modal_show(ui.modal(
                    ui.input_text("export_filename", "CSV filename", value="export.csv"),
                    ui.download_button("download_csv", "Download CSV", class_="btn-primary"),
                    title="Export plotted dataframe",
                    easy_close=True,
                    footer=ui.modal_button("Cancel"),
                ))

            @reactive.effect
            @reactive.event(input.export_plot)
            def show_plot_export_dialog():
                ui.modal_show(ui.modal(
                    ui.input_numeric("plot_export_width", "Width (inches)", value=10, min=1, max=100, step=0.5),
                    ui.input_numeric("plot_export_height", "Height (inches)", value=5, min=1, max=100, step=0.5),
                    ui.input_numeric("plot_export_dpi", "Resolution (DPI)", value=150, min=30, max=1200, step=25),
                    ui.download_button("download_plot_png", "Download plot.png", class_="btn-primary"),
                    title="Export plot", easy_close=True,
                    footer=ui.modal_button("Cancel"),
                ))

            @render.download(filename="plot.png", media_type="image/png")
            def download_plot_png():
                import io
                width = float(input.plot_export_width())
                height = float(input.plot_export_height())
                dpi = int(input.plot_export_dpi())
                if not (1 <= width <= 100 and 1 <= height <= 100 and 30 <= dpi <= 1200):
                    raise ValueError("Invalid plot export dimensions or resolution")
                if width * dpi * height * dpi > 100_000_000:
                    raise ValueError("Export exceeds 100 megapixels; reduce size or DPI")
                if backend == "matplotlib":
                    import matplotlib.pyplot as plt
                    fig = _matplotlib_direct_figure(prepared_plot(), **matplotlib_options())
                    try:
                        fig.set_size_inches(width, height, forward=True)
                        buffer = io.BytesIO()
                        fig.savefig(buffer, format="png", dpi=dpi)
                        yield buffer.getvalue()
                    finally:
                        plt.close(fig)
                else:
                    # Plotly static image export requires kaleido.
                    # Use the same styled figure as the interactive plot.
                    fig = styled_plotly_figure()
                    try:
                        yield fig.to_image(format="png", width=round(width * dpi),
                                           height=round(height * dpi), scale=1)
                    except Exception as exc:
                        raise RuntimeError(
                            'Plotly PNG export requires kaleido: pip install kaleido'
                        ) from exc

            @render.download(filename=lambda: _safe_csv_filename(input.export_filename()))
            def download_csv():
                # Use the same post-filter/post-aggregation dataframe as the plot.
                # The original dataframe is never mutated.
                frame = pd.DataFrame(selected_data()).reset_index(drop=True)
                yield frame.to_csv(index=False).encode("utf-8-sig")

            @render.ui
            def x_control():
                kind = input.plot_type()
                if kind in ("box", "bar"):
                    choices = categorical_columns
                    multiple = True
                elif kind == "line":
                    choices = _alphabetical(datetime_columns + timedelta_columns)
                    multiple = False
                else:
                    choices = numeric_columns
                    multiple = kind in ("scatter", "histogram")

                if not choices:
                    return ui.help_text(f"No compatible X variables for {kind}.")
                selected = initial_filters.get("x", [])
                # This control is remounted on plot-type changes. Never feed
                # a datetime line X into a categorical box/bar selectize:
                # Shiny otherwise has to reconcile a selection absent from
                # the new choice set while dependent reactive plots run.
                candidate = [selected] if isinstance(selected, str) else list(selected or [])
                candidate = [value for value in candidate if value in choices]
                selected = candidate if multiple else (candidate[0] if candidate else "")
                if multiple:
                    return ui.input_selectize(
                        "x", "X-axis / variables" if kind in ("scatter", "histogram") else "X-axis / categories",
                        choices={c: c for c in choices},
                        selected=selected, multiple=True,
                    )
                return ui.input_select(
                    "x", "X-axis / variable",
                    choices={"": "None", **{c: c for c in choices}},
                    selected=initial_filters.get("x", ""),
                )

            @render.ui
            def y_control():
                kind = input.plot_type()
                if kind == "histogram":
                    return ui.div()
                if not numeric_columns:
                    return ui.help_text("No numeric Y variables available.")
                return ui.input_selectize(
                    "y", "Y-axis / variables",
                    choices={c: c for c in numeric_columns},
                    selected=initial_filters.get("y", []),
                    multiple=True,
                )

            def _optional_input(name, default=None):
                """Read a conditionally mounted Shiny input safely.

                Dynamic controls do not exist before their UI has been mounted.
                Their absence must not prevent the first plot from rendering.
                """
                try:
                    value = getattr(input, name)()
                except (AttributeError, KeyError, TypeError):
                    return default
                return default if value is None else value

            # Panel controls are static UI inputs.  Previously they were created
            # by a render.ui output inside an inactive tab, so Shiny did not
            # initialize them until the user opened Panel appearance.  This
            # prevented Matplotlib grouped plots from rendering immediately.
            @reactive.effect
            @reactive.event(input.group_by, ignore_none=False)
            def sync_panel_group_choices():
                selected = list(input.group_by() or [])
                with reactive.isolate():
                    current_panels = list(_optional_input("panel_by", []) or [])
                valid_panels = [name for name in current_panels if name in selected]
                ui.update_selectize("panel_by", choices={name: name for name in selected},
                                    selected=valid_panels)

            @render.ui
            def frequency_options():
                if FREQUENCIES.get(input.frequency()) is None:
                    return ui.div()
                return ui.div(
                    ui.input_checkbox("only_complete", "Only complete intervals", value=initial_filters.get("only_complete", True)),
                    ui.input_checkbox("flooring_timestamp", "Floor timestamps", value=initial_filters.get("flooring_timestamp", True)),
                )

            # The virtual group choices depend on plot type and selected axes,
            # not on the Group by value. Updating a selectize control while
            # subscribing to that same control can create an invalidation loop,
            # especially during a line -> box/bar transition with multiple Y's.
            _last_axis_group_choices = reactive.value(None)

            @reactive.effect
            def sync_axis_group_choices():
                kind = input.plot_type()
                x_raw = _optional_input("x", [])
                y_raw = _optional_input("y", [])
                xs = [x_raw] if isinstance(x_raw, str) else list(x_raw or [])
                ys = [y_raw] if isinstance(y_raw, str) else list(y_raw or [])
                choices = list(group_columns)
                if kind in ("histogram", "scatter") and len(xs) > 1:
                    choices.append("variables_x")
                if kind != "histogram" and len(ys) > 1:
                    choices.append("variables_y")
                signature = tuple(choices)
                with reactive.isolate():
                    previous = _last_axis_group_choices.get()
                    current = list(_optional_input("group_by", []) or [])
                if signature == previous:
                    return
                _last_axis_group_choices.set(signature)
                selected = [g for g in current if g in choices]
                # The virtual axis groups are not part of the static selectize
                # choices at mount time. On restoring a saved plot, Shiny may
                # discard them before this effect populates the choices.
                # Restore only eligible saved virtual groups; do not override
                # physical groups or subsequent user changes.
                if previous is None:
                    restored = initial_filters.get("group_by", []) or []
                    if isinstance(restored, str):
                        restored = [restored]
                    for name in restored:
                        if name in ("variables_x", "variables_y") and name in choices and name not in selected:
                            selected.append(name)
                ui.update_selectize("group_by", choices={c: c for c in choices}, selected=selected)

            @reactive.calc
            def selected_data():
                current = filtered_summary()
                # Chained subsets may legitimately produce zero rows. Do not
                # call group_stat() on an empty Summary (it returns None).
                if current is None or current.empty:
                    return pd.DataFrame()
                kind = input.plot_type()
                x_input = input.x()
                selected_x = ([x_input] if isinstance(x_input, str) else list(x_input or []))
                if not selected_x or any(c not in column_map for c in selected_x):
                    return pd.DataFrame()

                if kind == "histogram":
                    value_names = selected_x
                else:
                    y_input = input.y()
                    value_names = ([y_input] if isinstance(y_input, str) else list(y_input or []))
                if not value_names or any(c not in column_map for c in value_names):
                    return pd.DataFrame()
                # Changing plot type updates the X selector asynchronously.
                # Until that update arrives, a histogram may still hold the
                # previous line plot's string/datetime X column. Never pass
                # such a column to group_stat(mean/median/etc.).
                if kind == "histogram" and any(
                    not pd.api.types.is_numeric_dtype(current[column_map[name]])
                    for name in value_names
                ):
                    return pd.DataFrame()

                selected_groups = [g for g in (input.group_by() or [])
                                   if g not in ("variables_x", "variables_y")]
                # Derive hour/day/etc. through group_stat without aggregation.
                # The returned index column identifies source rows.
                for component in selected_groups:
                    if component not in time_group_columns:
                        continue
                    generated = current.group_stat(
                        value_cols=value_names,
                        frequency=component,
                        grouping_by=None,
                        aggregate=None,
                        only_complete=False,
                        flooring_timestamp=False,
                    )
                    if generated is None or component not in generated.columns:
                        return pd.DataFrame()
                    current = current.copy()
                    if "index" in generated.columns and current.index.is_unique:
                        values = generated.set_index("index")[component]
                        current[component] = values.reindex(current.index)
                    else:
                        valid = current["datetime"].notna().to_numpy()
                        current[component] = pd.NA
                        if len(generated) == int(valid.sum()):
                            current.loc[valid, component] = generated[component].to_numpy()
                grouping_by = [column_map[c] for c in selected_groups]
                frequency = FREQUENCIES.get(input.frequency())
                should_aggregate = AGGREGATES.get(input.aggregate()) is not None
                aggregate = AGGREGATES.get(input.aggregate())
                value_columns = [column_map[c] for c in value_names]

                if should_aggregate and kind in ("box", "bar"):
                    for c in selected_x:
                        if column_map[c] not in grouping_by:
                            grouping_by.append(column_map[c])
                if kind == "scatter" and should_aggregate:
                    # Every X and Y variable must be aggregated on identical rows.
                    value_columns = list(dict.fromkeys(
                        [*(column_map[c] for c in selected_x), *value_columns]
                    ))
                if (kind == "line" and selected_x[0] in timedelta_columns
                        and should_aggregate):
                    # group_stat aggregation drops the original timedelta X
                    # unless it is explicitly aggregated. Never silently plot
                    # an unrelated frequency column instead.
                    return pd.DataFrame()
                if should_aggregate:
                    # An aggregation without a frequency or grouping has no
                    # plotting groups. Keep the reactive chain alive; the
                    # prepared_plot() validation displays a useful message.
                    if frequency is None and not grouping_by:
                        return pd.DataFrame()
                    if current is None or current.empty:
                        return pd.DataFrame()
                    current = _aggregate(
                        current, value_columns, frequency, grouping_by, aggregate,
                        only_complete=(bool(input.only_complete()) if frequency is not None else True),
                        flooring_timestamp=(bool(input.flooring_timestamp()) if frequency is not None else True),
                    )
                    if current is None or current.empty:
                        return pd.DataFrame()
                return current

            @reactive.calc
            def unfiltered_plot():
                # Validate before evaluating selected_data(), since invalid
                # aggregation options must not poison reactive calculations.
                if (AGGREGATES.get(input.aggregate()) is not None
                        and FREQUENCIES.get(input.frequency()) is None
                        and not list(input.group_by() or [])):
                    return {"error": "Select a frequency or at least one Group by variable to apply a statistical function"}
                plot_df = selected_data()
                selected_plot = input.plot_type()
                x_input = input.x()
                selected_x = ([x_input] if isinstance(x_input, str) else list(x_input or []))
                y_input = None if selected_plot == "histogram" else input.y()
                selected_y = ([y_input] if isinstance(y_input, str) else list(y_input or []))
                selected_groups = list(input.group_by() or [])
                physical_groups = [g for g in selected_groups if g not in ("variables_x", "variables_y")]
                frequency = FREQUENCIES.get(input.frequency())
                should_aggregate = AGGREGATES.get(input.aggregate()) is not None

                if (selected_plot == "line" and selected_x
                        and selected_x[0] in timedelta_columns
                        and should_aggregate):
                    return {"error": "Timedelta X-axis requires Aggregation: None. Disable aggregation to plot elapsed time."}
                if plot_df.empty or not selected_x or (
                    selected_plot != "histogram" and not selected_y
                ):
                    return {"error": "No data for the selected filters"}
                # The X widget is replaced asynchronously by render.ui. Until
                # the browser mounts it, the old line X can remain in input.x.
                # Treat that intermediate state as unready, not as box data.
                if selected_plot in ("box", "bar") and any(
                    x not in categorical_columns for x in selected_x
                ):
                    return {"error": "Waiting for categorical X-axis selection"}

                aggregate = AGGREGATES.get(input.aggregate())
                def mapped(column):
                    original = column_map[column]
                    return f"{original}_{aggregate}" if should_aggregate else original

                if selected_plot == "histogram":
                    x_column = mapped(selected_x[0])
                    y_columns = []
                elif selected_plot in ("box", "bar"):
                    x_columns = [column_map[c] for c in selected_x]
                    x_column = x_columns[0]
                    y_columns = [mapped(c) for c in selected_y]
                elif selected_plot == "scatter":
                    x_column = mapped(selected_x[0])
                    y_columns = [mapped(c) for c in selected_y]
                else:
                    x_column = (column_map[selected_x[0]] if selected_x[0] in timedelta_columns
                                else frequency if frequency is not None
                                else column_map[selected_x[0]])
                    y_columns = [mapped(c) for c in selected_y]

                if selected_plot == "line" and should_aggregate and frequency is None:
                    return {"error": "Categorical-only aggregation has no datetime axis; use a bar plot or select a frequency"}

                x_variables = [mapped(c) for c in selected_x] if selected_plot in ("scatter", "histogram") else [x_column]
                required = [*x_variables, *y_columns]
                if selected_plot in ("box", "bar"):
                    required.extend(x_columns)
                if any(c not in plot_df.columns for c in required):
                    return {"error": "Selected variables are unavailable after aggregation"}

                if selected_plot == "histogram" and len(x_variables) > 1:
                    from .histogram_multi import histogram_long_data
                    plot_df = histogram_long_data(plot_df, x_variables, group_columns=[
                        column_map[c] for c in physical_groups if column_map[c] in plot_df.columns
                    ])
                    plot_df["variables_x"] = plot_df["__hist_variable"]
                    x_column = "__hist_value"
                    x_label_override = " | ".join(x_variables)

                if (selected_plot == "scatter" and "variables_x" in selected_groups
                        and len(x_variables) > 1):
                    # Melt copies rather than the original X columns.  A source
                    # variable can simultaneously be an X and a Y selection;
                    # removing it here leaves the subsequent Y melt and the
                    # renderer with a stale reference to a missing column.
                    plot_df = plot_df.copy()
                    x_copies = []
                    x_names = {}
                    for index, column in enumerate(x_variables):
                        temporary = f"__plot_x_melt_{index}"
                        while temporary in plot_df.columns:
                            temporary += "_"
                        plot_df[temporary] = plot_df[column]
                        x_copies.append(temporary)
                        x_names[temporary] = column
                    id_columns = [c for c in plot_df.columns if c not in x_copies]
                    plot_df = plot_df.melt(id_vars=id_columns, value_vars=x_copies,
                                           var_name="variables_x", value_name="__axis_x_value")
                    plot_df["variables_x"] = plot_df["variables_x"].map(x_names)
                    x_column = "__axis_x_value"
                    x_variables = [x_column]

                if selected_plot in ("box", "bar"):
                    plot_df = plot_df.copy(deep=False)
                    plot_df["__plot_category"] = _composite_labels(plot_df, x_columns)
                    x_column = "__plot_category"

                if "variables_y" in selected_groups and len(y_columns) > 1:
                    # Make the selected Y variable an actual categorical dimension,
                    # so it works with ordinary grouping and panel partitioning.
                    # A variable may be selected as both scatter X and Y.
                    # Melting the original Y columns would remove it from the
                    # frame, leaving the X axis pointing at a missing column.
                    # Melt temporary Y copies instead, retaining X columns.
                    overlapping = set(y_columns).intersection(x_variables)
                    if overlapping:
                        plot_df = plot_df.copy()
                        copies = []
                        original_names = {}
                        for index, column in enumerate(y_columns):
                            temporary = f"__plot_y_melt_{index}"
                            while temporary in plot_df.columns:
                                temporary += "_"
                            plot_df[temporary] = plot_df[column]
                            copies.append(temporary)
                            original_names[temporary] = column
                        id_columns = [c for c in plot_df.columns if c not in copies]
                        plot_df = plot_df.melt(id_vars=id_columns, value_vars=copies,
                                               var_name="variables_y", value_name="__axis_y_value")
                        plot_df["variables_y"] = plot_df["variables_y"].map(original_names)
                    else:
                        id_columns = [c for c in plot_df.columns if c not in y_columns]
                        plot_df = plot_df.melt(id_vars=id_columns, value_vars=y_columns,
                                               var_name="variables_y", value_name="__axis_y_value")
                    y_columns = ["__axis_y_value"]
                    # The categorical X labels have already been computed above.
                    # Scatter and bar/box now use the standard single-Y renderer.

                # Convert durations once for Plotly, Matplotlib, editable
                # numeric bounds, faceting and exported plots. Retain the
                # original variable name and label its displayed unit.
                x_label_override = None
                if (selected_plot == "line" and
                        pd.api.types.is_timedelta64_dtype(plot_df[x_column])):
                    plot_df = plot_df.copy()
                    plot_df[x_column] = plot_df[x_column] / pd.Timedelta(hours=1)
                    x_label_override = f"{x_column} [hours]"
                    # Frequency-based gap detection is only meaningful for
                    # aggregated timestamp series, not raw durations.
                    frequency = None

                # Multiple grouping columns are supported by group_stat(). For
                # plotting, Plotly needs one color/category column, so combine the
                # selected grouping columns into a readable composite label.
                group_columns = [column_map[c] for c in physical_groups
                                 if column_map[c] in plot_df.columns]
                if "variables_x" in selected_groups and "variables_x" in plot_df:
                    group_columns.append("variables_x")
                if "variables_y" in selected_groups and "variables_y" in plot_df:
                    group_columns.append("variables_y")
                if group_columns:
                    plot_df = plot_df.copy(deep=False)
                    plot_df["__plot_group"] = (
                        _composite_labels(plot_df, group_columns)
                    )

                # The scatter Y-variable grouping uses a long-form value column.
                # Keep the spec synchronized with that column even during
                # reactive transitions between single- and multi-Y selections.
                if selected_plot == "scatter" and "__axis_y_value" in plot_df.columns:
                    y_columns = ["__axis_y_value"]

                return {"data": plot_df, "kind": selected_plot, "x": x_column,
                        "ys": y_columns, "groups": group_columns,
                        "categories": x_columns if selected_plot in ("box", "bar") else None,
                        "frequency": frequency, "x_label_override": x_label_override,
                        "x_variables": x_variables,
                        "hist_variables": x_variables if selected_plot == "histogram" and len(x_variables) > 1 else [],
                        "multi_axis": (selected_plot in ("box", "bar") and len(y_columns) > 1)
                            or (selected_plot == "scatter" and (len(x_variables) > 1 or len(y_columns) > 1))}

            @reactive.calc
            def available_elements():
                return element_keys(unfiltered_plot())

            # The live configuration is authoritative for component visibility.
            # Keep only the previous *available keys* locally: this is derived
            # renderer metadata, not another copy of the user's selection.
            inherited_visible = initial_filters.get("visible_elements")
            element_key_signature = reactive.Value(None)

            def configured_visibility():
                """Read canonical visibility; None means all components visible."""
                if config_manager is not None and config_tab_id is not None:
                    if config_tab_id in config_manager.drafts:
                        return config_manager.value(config_tab_id, "visible_elements", inherited_visible)
                return inherited_visible

            @reactive.effect
            def reconcile_visible_elements():
                keys = tuple(available_elements())
                with reactive.isolate():
                    previous = element_key_signature.get()
                    saved = configured_visibility()
                # Dynamic controls can briefly expose no components on restore.
                # Never overwrite saved visibility during that transient phase.
                if not keys:
                    return
                if previous == keys:
                    return
                if saved is None:
                    selected = keys
                else:
                    selected = tuple(k for k in keys if k in saved)
                    if previous is None and not selected:
                        # Old snapshots may contain stale/empty component IDs.
                        selected = keys
                    elif previous is not None:
                        old_keys = set(previous)
                        selected = tuple(k for k in keys if k in selected or k not in old_keys)
                element_key_signature.set(keys)
                if config_manager is not None and config_tab_id is not None:
                    _patch_live_config(config_tab_id, {"visible_elements": list(selected)})

            @render.ui
            def element_visibility_ui():
                keys = tuple(available_elements())
                with reactive.isolate():
                    saved = configured_visibility()
                    chosen = keys if saved is None else tuple(k for k in keys if k in saved)
                if not keys:
                    # An empty filtered dataset is temporary. Keep the last
                    # component controls mounted instead of replacing them
                    # with help text: unmounting sends an empty checkbox value
                    # which can later hide every line when data returns.
                    with reactive.isolate():
                        previous = element_key_signature.get() or ()
                    if not previous:
                        return ui.help_text("No plotted elements available")
                    keys = previous
                    chosen = keys if saved is None else tuple(k for k in keys if k in saved)
                return ui.input_checkbox_group(
                    "visible_elements", "Visible plotted elements",
                    choices={key: key for key in keys}, selected=list(chosen),
                )

            @reactive.effect
            @reactive.event(input.visible_elements, ignore_init=True)
            def sync_visible_elements():
                chosen = _optional_input("visible_elements", None)
                if chosen is None:
                    return
                keys = tuple(available_elements())
                with reactive.isolate():
                    signature = element_key_signature.get()
                    saved = configured_visibility()
                # Ignore stale events from an unmounted/rebuilt checkbox group.
                if not keys or signature != keys:
                    return
                chosen = tuple(k for k in chosen if k in keys)
                effective = keys if saved is None else tuple(k for k in keys if k in saved)
                if chosen != effective and config_manager is not None and config_tab_id is not None:
                    _patch_live_config(config_tab_id, {"visible_elements": list(chosen)})

            @reactive.calc
            def prepared_plot():
                raw = unfiltered_plot()
                # Visibility changes alter the plot specification. Other appearance
                # edits (labels, colours, panel spacing) must not invalidate it:
                # renderers can reuse the already prepared data/specification.
                if config_manager is not None and config_tab_id is not None:
                    config_manager.watch(config_tab_id, "visibility")
                chosen = configured_visibility()
                if chosen is None:
                    chosen = tuple(default_visible_keys(raw))
                return filter_elements(raw, chosen)

            @reactive.calc
            def base_figure():
                import plotly.express as px
                spec = prepared_plot()
                if "error" in spec:
                    return px.scatter(title=spec["error"])
                plot_df = spec["data"]
                if plot_df.empty:
                    return px.scatter(title="No visible plot elements")
                selected_plot = spec["kind"]
                x_column = spec["x"]
                y_columns = spec["ys"]
                group_columns = spec["groups"]
                frequency = spec["frequency"]
                x_columns = spec["categories"]
                if spec.get("multi_axis"):
                    from .multi_axis import plotly_multi_axis
                    fig = plotly_multi_axis(spec, mode=_optional_input("multi_y_color_by", "group"),
                                            palette=input.palette(), legend_labels=_label_string("legend_labels"),
                                            trace_colors=_color_string())
                elif selected_plot == "histogram":
                    fig = _histogram_figure(
                        plot_df, x_column,
                        group_column="__hist_series" if spec.get("hist_variables") else
                                     ("__plot_group" if group_columns else None),
                        bins=int(input.bins()),
                        density=input.hist_density() == "True",
                        stacked=input.hist_stacked() == "True",
                        histtype=input.hist_type(),
                        cumulative=int(input.hist_cumulative()),
                        overlay=bool(spec.get("hist_variables")),
                    )
                elif selected_plot == "box":
                    fig = _boxplot_figure(plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
                elif selected_plot == "bar":
                    fig = _barplot_figure(plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
                elif selected_plot == "scatter":
                    fig = _scatter_figure(plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
                else:
                    fig = _lineplot_figure(plot_df, x_column, y_columns, group_columns=group_columns, frequency=frequency, column_map=column_map)

                # A removed group/variable must not leave an empty Plotly trace
                # or a legend entry behind.
                if selected_plot == "line":
                    fig.data = tuple(trace for trace in fig.data
                                     if getattr(trace, "y", None) is not None
                                     and any(pd.notna(v) for v in trace.y))
                if not group_columns and not spec.get("multi_axis") and not spec.get("hist_variables"):
                    # Retain trace names/colours for appearance editing, but
                    # suppress the redundant ungrouped legend.
                    fig.update_traces(showlegend=False)
                    fig.update_layout(showlegend=False)

                _label_plot_axes(
                    fig, selected_plot, x_column, y_columns,
                    category_columns=x_columns if selected_plot in ("box", "bar") else None,
                    x_label_override=spec.get("x_label_override"),
                )
                if spec.get("multi_axis"):
                    from .appearance import _axis_label
                    if selected_plot == "scatter" and len(spec["x_variables"]) > 1:
                        fig.update_xaxes(title_text="<br>".join(_axis_label(c) for c in spec["x_variables"]))
                    if len(y_columns) > 1:
                        fig.update_yaxes(title_text="<br>".join(_axis_label(c) for c in y_columns))
                fig.update_layout(
                    hovermode="x unified" if selected_plot == "line" else "closest",
                    margin=dict(l=60, r=30, t=70, b=60),
                    autosize=True,
                )
                return fig

            # Populate appearance controls only after a valid plot has been
            # prepared.  The effect depends on the plot specification, not on
            # appearance inputs, so editing a label or colour cannot reset it.
            _last_line_style_keys = [None]
            _last_scatter_style_keys = [None]

            @reactive.effect
            @reactive.event(prepared_plot, lambda: _optional_input("multi_panel", False) if input.group_by() else False,
                            lambda: tuple(_optional_input("panel_by", []) or []) if input.group_by() and len(input.group_by()) > 1 else (),
                            lambda: _optional_input("independent_panel_limits", True) if input.group_by() and _optional_input("multi_panel", False) else True, ignore_none=False)
            def populate_appearance_from_plot():
                from .appearance import _parse_label_map
                spec = prepared_plot()
                if "error" in spec:
                    return
                defaults = _matplotlib_defaults(spec, palette="Default")
                names = defaults["names"]
                selected_groups_for_legend = list(input.group_by() or [])
                if (backend == "matplotlib" and selected_groups_for_legend
                        and bool(_optional_input("multi_panel", False))):
                    panel_variables_for_legend = (
                        (list(_optional_input("panel_by", []) or []) or selected_groups_for_legend)
                        if len(selected_groups_for_legend) > 1
                        else selected_groups_for_legend
                    )
                    if panel_variables_for_legend:
                        trace_defaults = _panel_trace_defaults(
                            spec, panel_variables_for_legend, palette="Default"
                        )
                        names = trace_defaults["names"]
                        defaults["colors"] = trace_defaults["colors"]
                # On a restored tab, the saved labels/title are authoritative.
                # The preparation effect can run after the controls mount and
                # otherwise overwrite the saved values with generated defaults.
                # Resume automatic labels when the underlying data selection changes.
                restored_data_keys = ("plot_type", "x", "y", "group_by", "frequency", "aggregate")
                same_restored_data = bool(initial_filters) and all(
                    (list(_optional_input(k, []) or []) if k in ("x", "y", "group_by") else _optional_input(k, None))
                    == (list(initial_filters.get(k, []) or []) if k in ("x", "y", "group_by") else initial_filters.get(k))
                    for k in restored_data_keys if k in initial_filters
                )
                if not same_restored_data or "title" not in initial_filters:
                    ui.update_text("title", value="Summary plot")
                if not same_restored_data or "x_axis_label" not in initial_filters:
                    ui.update_text("x_axis_label", value=defaults["x_label"])
                if not same_restored_data or "y_axis_label" not in initial_filters:
                    ui.update_text_area("y_axis_label", value=defaults["y_label"])
                limits = _axis_bounds_from_spec(
                    spec,
                    bins=int(input.bins()) if spec["kind"] == "histogram" else 30,
                    density=input.hist_density() == "True" if spec["kind"] == "histogram" else False,
                    stacked=(input.hist_stacked() == "True" or input.hist_type() == "barstacked") if spec["kind"] == "histogram" else False,
                    cumulative=int(input.hist_cumulative()) if spec["kind"] == "histogram" else 0,
                )
                if spec["kind"] == "box" and not spec.get("multi_axis") and not input.box_showfliers():
                    from .data import _box_whisker_limits
                    box_groups = [spec["x"]] + (["__plot_group"] if spec["groups"] else [])
                    whiskers = _box_whisker_limits(spec["data"], spec["ys"][0], box_groups)
                    if whiskers is not None:
                        axis = "x" if input.box_orientation() == "horizontal" else "y"
                        limits[f"{axis}_min"], limits[f"{axis}_max"] = map(str, whiskers)
                limits = _pad_axis_bounds(limits)
                selected_groups = list(input.group_by() or [])
                if (backend == "matplotlib" and selected_groups and bool(_optional_input("multi_panel", False))
                        and bool(_optional_input("independent_panel_limits", True))):
                    panel_variable = ((list(_optional_input("panel_by", []) or []) or selected_groups) if len(selected_groups) > 1
                                      else selected_groups)
                    panel_bounds = _panel_axis_bounds(
                        spec, panel_variable,
                        bins=int(input.bins()) if spec["kind"] == "histogram" else 30,
                        density=input.hist_density() == "True" if spec["kind"] == "histogram" else False,
                        stacked=(input.hist_stacked() == "True" or input.hist_type() == "barstacked") if spec["kind"] == "histogram" else False,
                        cumulative=int(input.hist_cumulative()) if spec["kind"] == "histogram" else 0,
                    )
                    if panel_bounds:
                        if spec["kind"] == "box" and not spec.get("multi_axis") and not input.box_showfliers():
                            from .data import _box_whisker_limits
                            panel_groups = [spec["x"]] + (["__plot_group"] if spec["groups"] else [])
                            axis = "x" if input.box_orientation() == "horizontal" else "y"
                            for panel_title, frame, *_ in _panel_partitions(spec, panel_variable):
                                whiskers = _box_whisker_limits(frame, spec["ys"][0], panel_groups)
                                if whiskers is not None and panel_title in panel_bounds:
                                    panel_bounds[panel_title][f"{axis}_min"], panel_bounds[panel_title][f"{axis}_max"] = map(str, whiskers)
                        panel_bounds = {name: _pad_axis_bounds(bounds) for name, bounds in panel_bounds.items()}
                        limits = {key: _format_panel_bounds(panel_bounds, key) for key in limits}
                for field, value in limits.items():
                    if backend == "matplotlib":
                        ui.update_text_area(field, value=value)
                    else:
                        ui.update_text(field, value=value)
                # Legend labels are maintained by the rendered-figure observer
                # below, not predicted from the plot specification.
                names = list(dict.fromkeys(str(n) for n in names))
                # Trace-colour editor keys are synced independently from the
                # active plot and never reset by appearance defaults.
                kind = spec["kind"]
                if kind == "line":
                    # Only reconcile the editors when their trace identities change.
                    # Plot preparation may rerun for unrelated reasons (filters,
                    # panels, etc.); resetting these inputs on each run races with
                    # the user's edits and makes repeated style changes unreliable.
                    current_keys = tuple(names)
                    previous_keys = _last_line_style_keys[0]
                    _last_line_style_keys[0] = current_keys
                    if previous_keys != current_keys:
                        for field, default in (("line_alpha", "1"), ("line_style", "solid")):
                            if previous_keys is None and same_restored_data and field in initial_filters:
                                # The saved configuration is already mounted.
                                continue
                            with reactive.isolate():
                                existing = _parse_label_map(_optional_input(field, ""))
                            # Preserve edits for identities that survive a data
                            # change; initialize only newly discovered traces.
                            updated = "\n".join(
                                f"{name}={existing.get(name, default)}" for name in names
                            )
                            ui.update_text_area(field, value=updated)
                elif kind == "histogram":
                    ui.update_text_area("hist_alpha", value="\n".join(f"{name}=0.65" for name in names))
                elif kind == "scatter":
                    # Preserve user edits across preparation updates; only
                    # synchronize when the actual trace identities change.
                    current_keys = tuple(names)
                    previous_keys = _last_scatter_style_keys[0]
                    _last_scatter_style_keys[0] = current_keys
                    if previous_keys != current_keys:
                        for field, default in (("scatter_size", "7"), ("scatter_marker", "dot"), ("scatter_alpha", "1")):
                            if previous_keys is None and same_restored_data and field in initial_filters:
                                continue
                            with reactive.isolate():
                                existing = _parse_label_map(_optional_input(field, ""))
                            ui.update_text_area(field, value="\n".join(
                                f"{name}={existing.get(name, default)}" for name in names))
                if backend == "matplotlib":
                    panel_variables = ((list(_optional_input("panel_by", []) or []) or selected_groups) if len(selected_groups) > 1
                                       else selected_groups)
                    panel_names = (list(dict.fromkeys(name for name, part, _ in _panel_partitions(spec, panel_variables)
                                                        if not part.empty))
                                   if selected_groups and bool(_optional_input("multi_panel", False)) else [])
                    _set_label_keys("panel_titles", panel_names)

            @reactive.effect
            @reactive.event(input.box_showfliers, input.box_orientation,
                            input.multi_panel, input.independent_panel_limits,
                            input.panel_by, ignore_init=True)
            def update_box_axis_for_fliers():
                """Recompute both axes after orientation, outlier or facet changes.

                The category axis must never inherit a numeric data range from
                the previous orientation. Facet limits are recomputed from the
                actual partition, not from the preceding single-panel view.
                """
                spec = prepared_plot()
                if spec.get("kind") != "box" or "error" in spec:
                    return
                from .data import _box_whisker_limits

                horizontal = input.box_orientation() == "horizontal"
                value_axis = "x" if horizontal else "y"
                category_axis = "y" if horizontal else "x"

                def box_bounds(frame, grouped):
                    vals = pd.to_numeric(frame[spec["ys"][0]], errors="coerce")
                    vals = vals[np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))]
                    if vals.empty:
                        return None
                    if input.box_showfliers():
                        low, high = float(vals.min()), float(vals.max())
                    else:
                        group_cols = [spec["x"]] + (["__plot_group"] if grouped else [])
                        whiskers = _box_whisker_limits(frame, spec["ys"][0], group_cols)
                        if whiskers is None:
                            return None
                        low, high = map(float, whiskers)
                    margin = .05 * (high - low) if high > low else max(abs(low) * .05, .5)
                    return (low - margin, high + margin)

                groups = list(input.group_by() or [])
                panels = (backend == "matplotlib" and groups
                          and bool(_optional_input("multi_panel", False))
                          and bool(_optional_input("independent_panel_limits", True)))
                if panels:
                    panel_by = ((list(_optional_input("panel_by", []) or []) or groups)
                                if len(groups) > 1 else groups)
                    parts = _panel_partitions(spec, panel_by)
                    values = {name: box_bounds(part, remaining)
                              for name, part, remaining in parts}
                    values = {name: bounds for name, bounds in values.items() if bounds is not None}
                    if values:
                        for index, suffix in enumerate(("min", "max")):
                            ui.update_text_area(f"{value_axis}_{suffix}", value="\n".join(
                                f"{name}={bounds[index]}" for name, bounds in values.items()))
                else:
                    bounds = box_bounds(spec["data"], spec["groups"])
                    if bounds is not None:
                        for suffix, value in zip(("min", "max"), bounds):
                            if backend == "matplotlib":
                                ui.update_text_area(f"{value_axis}_{suffix}", value=str(value))
                            else:
                                ui.update_text(f"{value_axis}_{suffix}", value=str(value))
                # Categorical positions are determined by the renderer.
                # Clearing these prevents old numeric limits from clipping them.
                for suffix in ("min", "max"):
                    if backend == "matplotlib":
                        ui.update_text_area(f"{category_axis}_{suffix}", value="")
                    else:
                        ui.update_text(f"{category_axis}_{suffix}", value="")

            @reactive.effect
            @reactive.event(input.plot_type, ignore_init=True)
            def reset_on_plot_type_change():
                # Reset every plot-specific selection when the plot type changes.
                for _i in range(max_subset_stages):
                    ui.update_select(f"subset_column_{_i}", selected="")
                    ui.update_select(f"subset_style_{_i}", selected="all")
                ui.update_selectize("numeric_subset_columns", selected=[])
                ui.update_text("datetime_start", value="")
                ui.update_text("datetime_end", value="")
                ui.update_selectize("group_by", selected=[])
                ui.update_checkbox("multi_panel", value=False)
                ui.update_checkbox("independent_panel_limits", value=True)
                ui.update_select("frequency", selected="None")
                ui.update_select("aggregate", selected="None")
                ui.update_text("title", value="")
                ui.update_text("x_axis_label", value="")
                ui.update_text_area("y_axis_label", value="")
                for field in ("x_min", "x_max", "y_min", "y_max"):
                    if backend == "matplotlib":
                        ui.update_text_area(field, value="")
                    else:
                        ui.update_text(field, value="")
                for field in ("legend_labels", "x_tick_labels", "y_tick_labels", "panel_titles"):
                    _set_label_keys(field, ())
                # Invalidate figure-derived editor identities immediately. The
                # previous lineplot legend/ticks must never be treated as boxplot
                # identities during the asynchronous input transition.
                rendered_legend_keys.set(())
                rendered_tick_keys.set(((), ()))
                _color_keys.set(())
                ui.update_select("palette", selected="Default")
                ui.update_select("template", selected="plotly")
                ui.update_select("legend_position", selected="right")
                ui.update_checkbox("legend_inside", value=False)
                if backend == "matplotlib":
                    ui.update_numeric("legend_columns", value=1)
                ui.update_checkbox("show_grid", value=True)
                for field, value in (("title_font_size", 14), ("axis_font_size", 12),
                                     ("tick_font_size", 10), ("legend_font_size", 12),
                                     ("panel_font_size", 14)):
                    ui.update_numeric(field, value=value)
                if backend == "matplotlib":
                    ui.update_numeric("plot_dpi", value=100)
                ui.update_numeric("line_width", value=2)
                ui.update_checkbox("box_showfliers", value=True)
                ui.update_select("box_orientation", selected="vertical")
                ui.update_text_area("hist_alpha", value="")
                for field in ("line_alpha", "line_style", "scatter_size", "scatter_marker", "scatter_alpha"):
                    ui.update_text_area(field, value="")
                ui.update_numeric("bins", value=30)
                ui.update_select("hist_density", selected="False")
                ui.update_select("hist_stacked", selected="False")
                ui.update_select("hist_type", selected="bar")
                ui.update_select("hist_cumulative", selected="0")
                ui.update_checkbox("only_complete", value=True)
                ui.update_checkbox("flooring_timestamp", value=True)

            if backend == "matplotlib":
                _font_fields = ("title_font_size", "axis_font_size", "tick_font_size",
                                "legend_font_size", "panel_font_size")

                def _change_fonts(delta):
                    # Use current values, not the defaults, to retain individual
                    # adjustments made in the other appearance tabs.
                    with reactive.isolate():
                        for field in _font_fields:
                            current = _optional_input(field, None)
                            if current is None:
                                current = initial_filters.get(field, {
                                    "title_font_size": 14, "axis_font_size": 12,
                                    "tick_font_size": 10, "legend_font_size": 12,
                                    "panel_font_size": 14,
                                }[field])
                            ui.update_numeric(field, value=max(6, min(72, float(current) + delta)))

                @reactive.effect
                @reactive.event(input.increase_plot_fonts)
                def _increase_plot_fonts():
                    _change_fonts(1)

                @reactive.effect
                @reactive.event(input.decrease_plot_fonts)
                def _decrease_plot_fonts():
                    _change_fonts(-1)

                def _change_dpi(delta):
                    with reactive.isolate():
                        current = _optional_input("plot_dpi", initial_filters.get("plot_dpi", 100))
                        ui.update_numeric("plot_dpi", value=max(30, min(600, int(current or 100) + delta)))

                @reactive.effect
                @reactive.event(input.increase_plot_dpi)
                def _increase_plot_dpi():
                    _change_dpi(25)

                @reactive.effect
                @reactive.event(input.decrease_plot_dpi)
                def _decrease_plot_dpi():
                    _change_dpi(-25)

            # The figure renderer is the source of truth for legend entries.
            # Store original keys, not renamed display strings, to avoid
            # feedback loops when the user edits the mapping.
            rendered_legend_keys = reactive.value(())

            def _capture_legend(fig, is_plotly=False):
                from .appearance import _parse_label_map
                # Use the full saved mapping during initial legend discovery.
                # _label_string() deliberately filters by discovered keys, which
                # are still empty when a restored tab first renders.
                with reactive.isolate():
                    mapping = dict(_editor_map("legend_labels"))
                # Resolve displayed labels against stable, original keys.  A
                # reverse dictionary is unsafe: two different keys may have
                # the same edited label, and mapping order then makes keys
                # alternate as the figure is rendered repeatedly.
                with reactive.isolate():
                    known_keys = _label_keys["legend_labels"].get()
                if is_plotly:
                    if getattr(fig.layout, "showlegend", None) is False:
                        labels = []
                    else:
                        labels = [str(t.name) for t in fig.data
                                  if getattr(t, "showlegend", None) is True and t.name]
                else:
                    legends = list(getattr(fig, "legends", []))
                    legends += [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
                    labels = [txt.get_text() for leg in legends for txt in leg.get_texts()]
                excluded = ("Y variable", "Grouping", "Legend", "All observations")
                labels = [label for label in labels if label.strip() and label not in excluded]
                # If this is merely a relabelled rendering of the existing
                # traces, do not rediscover/rename their identities.
                expected = [mapping.get(key, key) for key in known_keys]
                if known_keys and set(labels) == set(expected):
                    keys = tuple(known_keys)
                else:
                    reverse = {}
                    for key in known_keys:
                        display = mapping.get(key, key)
                        reverse.setdefault(display, []).append(key)
                    keys = tuple(dict.fromkeys(
                        reverse[label][0] if len(reverse.get(label, ())) == 1
                        else label for label in labels
                    ))
                with reactive.isolate():
                    old = rendered_legend_keys.get()
                if keys != old:
                    rendered_legend_keys.set(keys)

            @reactive.effect
            def _refresh_labels_from_rendered_legend():
                keys = rendered_legend_keys.get()
                _set_label_keys("legend_labels", keys)

            # Tick editors are populated from the actual axes after the plot
            # renderer has finished choosing tick positions and categories.
            # Keep the unmodified keys separate from user-visible replacements.
            rendered_tick_keys = reactive.value(((), ()))

            def _capture_ticks(fig, is_plotly=False):
                if is_plotly:
                    keys = []
                    for axis_name in ("xaxis", "yaxis"):
                        axis = getattr(fig.layout, axis_name)
                        if axis.tickmode == "array" and axis.ticktext:
                            labels = tuple(map(str, axis.ticktext))
                        elif axis.type == "category" and axis.categoryarray:
                            labels = tuple(map(str, axis.categoryarray))
                        else:
                            # Plotly chooses automatic numeric ticks in the
                            # browser, not in the Python figure.
                            labels = ()
                        keys.append(tuple(dict.fromkeys(labels)))
                    keys = tuple(keys)
                else:
                    fig.canvas.draw()
                    keys = tuple(tuple(dict.fromkeys(
                        text.get_text() for ax in fig.axes if ax.get_visible()
                        for text in (ax.get_xticklabels() if direction == "x" else ax.get_yticklabels())
                        if text.get_text().strip()
                    )) for direction in ("x", "y"))
                with reactive.isolate():
                    previous = rendered_tick_keys.get()
                if keys != previous:
                    rendered_tick_keys.set(keys)

            @reactive.effect
            def _refresh_tick_editors():
                xkeys, ykeys = rendered_tick_keys.get()
                _set_label_keys("x_tick_labels", xkeys)
                _set_label_keys("y_tick_labels", ykeys)

            def _apply_matplotlib_tick_labels(fig):
                from .appearance import _parse_label_map
                from matplotlib.ticker import FixedLocator, FixedFormatter
                maps = tuple(_parse_label_map(_label_string(field))
                             for field in ("x_tick_labels", "y_tick_labels"))
                if not any(maps):
                    return
                # A fixed locator preserves the exact tick positions chosen
                # by Matplotlib, while the formatter changes text only.
                for ax in fig.axes:
                    for axis, mapping in zip((ax.xaxis, ax.yaxis), maps):
                        if not mapping:
                            continue
                        positions = axis.get_majorticklocs()
                        labels = [label.get_text() for label in axis.get_ticklabels()]
                        if len(positions) == len(labels):
                            axis.set_major_locator(FixedLocator(positions))
                            axis.set_major_formatter(FixedFormatter(
                                [mapping.get(label, label) for label in labels]))

            def _apply_plotly_tick_labels(fig):
                from .appearance import _parse_label_map
                maps = tuple(_parse_label_map(_label_string(field))
                             for field in ("x_tick_labels", "y_tick_labels"))
                for axis_name, mapping in zip(("xaxis", "yaxis"), maps):
                    if not mapping:
                        continue
                    axis = getattr(fig.layout, axis_name)
                    if axis.tickmode == "array" and axis.tickvals:
                        original = list(axis.ticktext or [str(v) for v in axis.tickvals])
                        axis.ticktext = [mapping.get(str(t), str(t)) for t in original]
                    else:
                        # Plotly category tick positions are category strings.
                        # Explicit tick arrays avoid changing category order.
                        values = list(axis.categoryarray or ())
                        if values:
                            axis.tickmode = "array"
                            axis.tickvals = values
                            axis.ticktext = [mapping.get(str(v), str(v)) for v in values]

            if backend == "plotly":
                @reactive.calc
                def styled_plotly_figure():
                    import plotly.graph_objects as go
                    fig = go.Figure(base_figure())
                    styled = _apply_plot_appearance(
                        fig, title=input.title(), x_label=input.x_axis_label(),
                        y_label=input.y_axis_label(), legend_labels=_label_string("legend_labels"),
                        trace_colors=_color_string(), palette=input.palette(),
                        x_min=input.x_min(), x_max=input.x_max(),
                        y_min=input.y_min(), y_max=input.y_max(),
                        template=input.template(), legend_position=input.legend_position(),
                        legend_inside=input.legend_inside(), show_grid=input.show_grid(),
                        x_major_ticks=input.x_major_ticks(), y_major_ticks=input.y_major_ticks(),
                        x_minor_ticks=input.x_minor_ticks(), y_minor_ticks=input.y_minor_ticks(),
                        line_width=(input.line_width() if input.plot_type() == "line" else 2),
                        hist_alpha=input.hist_alpha(), box_showfliers=input.box_showfliers(),
                        box_orientation=input.box_orientation(),
                        title_font_size=input.title_font_size(), axis_font_size=input.axis_font_size(),
                        tick_font_size=input.tick_font_size(), legend_font_size=input.legend_font_size(),
                        line_alpha=input.line_alpha(), line_style=input.line_style(),
                        scatter_size=input.scatter_size(), scatter_marker=input.scatter_marker(),
                        scatter_alpha=input.scatter_alpha(), plot_kind=input.plot_type(),
                    )
                    if input.plot_type() in ("bar", "box"):
                        from .plotly_renderer import _set_bar_box_width
                        _set_bar_box_width(styled, input.plot_type(), float(input.bar_box_width()))
                    if prepared_plot().get("multi_axis"):
                        # The generic appearance helper applies palette colours
                        # per trace and resets scatter symbols. Preserve the
                        # multi-variable colour/shape mapping after styling.
                        original = base_figure()
                        for target, source in zip(styled.data, original.data):
                            if getattr(source, "marker", None) is not None:
                                if source.marker.color is not None:
                                    target.marker.color = source.marker.color
                                if getattr(source.marker, "symbol", None) is not None:
                                    target.marker.symbol = source.marker.symbol
                                if getattr(source.marker, "pattern", None) is not None:
                                    target.marker.pattern = source.marker.pattern
                            if getattr(source, "line", None) is not None and source.line.color is not None:
                                target.line.color = source.line.color
                    if input.plot_type() == "line" and len(prepared_plot().get("ys", [])) > 1:
                        from .plotly_renderer import _style_multi_y_legend
                        _style_multi_y_legend(styled, prepared_plot(), input.multi_y_color_by(),
                                              input.palette(), _color_string(), input.line_style(),
                                              _label_string("legend_labels"))
                    if input.plot_type() == "box" and not input.box_showfliers():
                        from .data import _box_whisker_limits
                        spec = prepared_plot()
                        if "error" not in spec:
                            axis = "x" if input.box_orientation() == "horizontal" else "y"
                            lower_input = input.x_min() if axis == "x" else input.y_min()
                            upper_input = input.x_max() if axis == "x" else input.y_max()
                            if not str(lower_input or "").strip() and not str(upper_input or "").strip():
                                limits = _box_whisker_limits(
                                    spec["data"], spec["ys"][0],
                                    [spec["x"]] + (["__plot_group"] if spec["groups"] else []),
                                )
                                if limits is not None:
                                    styled.update_layout(**{f"{axis}axis_range": limits})
                    return styled

                @render_plotly
                def plot():
                    fig = styled_plotly_figure()
                    _capture_legend(fig, is_plotly=True)
                    _capture_ticks(fig, is_plotly=True)
                    _apply_plotly_tick_labels(fig)
                    return fig
            else:
                @reactive.calc
                def matplotlib_options():
                    # Shared with the export so that all appearance options,
                    # including per-panel overrides, are reproduced exactly.
                    # Read grouping/panel controls once per reactive evaluation.
                    # These options affect rendering, not selected_data().
                    active_groups = list(input.group_by() or [])
                    multi_panel = bool(_optional_input("multi_panel", False)) if active_groups else False
                    panel_groups = (list(_optional_input("panel_by", []) or []) or active_groups) if len(active_groups) > 1 else active_groups
                    return dict(title=_optional_input("title", ""),
                        x_label=_optional_input("x_axis_label", ""), y_label=_optional_input("y_axis_label", ""),
                        legend_labels=_label_string("legend_labels"), trace_colors=_color_string(),
                        palette=_optional_input("palette", "Default"), x_min=_optional_input("x_min", ""), x_max=_optional_input("x_max", ""),
                        y_min=_optional_input("y_min", ""), y_max=_optional_input("y_max", ""), template=_optional_input("template", "plotly"),
                        legend_position=_optional_input("legend_position", "right"), legend_inside=_optional_input("legend_inside", False),
                        legend_columns=max(1, int(_optional_input("legend_columns", 1) or 1)),
                        multi_y_color_by=_optional_input("multi_y_color_by", "group"),
                        show_grid=_optional_input("show_grid", True),
                        x_major_ticks=_optional_input("x_major_ticks", 6), y_major_ticks=_optional_input("y_major_ticks", 6),
                        x_minor_ticks=_optional_input("x_minor_ticks", False), y_minor_ticks=_optional_input("y_minor_ticks", False),
                        title_font_size=_optional_input("title_font_size", 14), axis_font_size=_optional_input("axis_font_size", 12),
                        tick_font_size=_optional_input("tick_font_size", 10), legend_font_size=_optional_input("legend_font_size", 12),
                        panel_font_size=_optional_input("panel_font_size", 14), plot_dpi=int(_optional_input("plot_dpi", 100) or 100), hist_alpha=_optional_input("hist_alpha", ""),
                        box_showfliers=_optional_input("box_showfliers", True), box_orientation=_optional_input("box_orientation", "vertical"),
                        bar_box_width=float(_optional_input("bar_box_width", 1.0)),
                        line_width=(_optional_input("line_width", 2) if input.plot_type() == "line" else 2),
                        line_alpha=(_optional_input("line_alpha", "") if input.plot_type() == "line" else ""),
                        line_style=(_optional_input("line_style", "") if input.plot_type() == "line" else ""),
                        scatter_size=(_optional_input("scatter_size", "") if input.plot_type() == "scatter" else ""),
                        scatter_marker=(_optional_input("scatter_marker", "") if input.plot_type() == "scatter" else ""),
                        scatter_alpha=(_optional_input("scatter_alpha", "") if input.plot_type() == "scatter" else ""),
                        bins=int(_optional_input("bins", 30)) if input.plot_type() == "histogram" else 30,
                        density=_optional_input("hist_density", "False") == "True" if input.plot_type() == "histogram" else False,
                        stacked=_optional_input("hist_stacked", "False") == "True" if input.plot_type() == "histogram" else False,
                        histtype=_optional_input("hist_type", "bar") if input.plot_type() == "histogram" else "bar",
                        cumulative=int(_optional_input("hist_cumulative", "0")) if input.plot_type() == "histogram" else 0,
                        panel_by=(panel_groups if multi_panel else None),
                        independent_panel_limits=(bool(_optional_input("independent_panel_limits", True))
                                                  if multi_panel else True),
                        panel_titles=(_label_string("panel_titles") if multi_panel else ""),
                        panel_hspace=(float(_optional_input("panel_hspace", 0.90) or 0) if multi_panel else 0.90),
                        panel_wspace=(float(_optional_input("panel_wspace", 0.38) or 0) if multi_panel else 0.38),
                    )

                @render.download(filename="plot_export.zip", media_type="application/zip")
                def export_python():
                    import io
                    import zipfile
                    spec = prepared_plot()
                    if "error" in spec:
                        raise ValueError(spec["error"])
                    frame = pd.DataFrame(spec["data"])
                    script = _standalone_matplotlib_script(spec, matplotlib_options())
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                        archive.writestr("plot_data.csv", frame.to_csv(index=False))
                        archive.writestr("plot_script.py", script)
                    yield buffer.getvalue()

                @render.plot
                def plot():
                    fig = _matplotlib_direct_figure(prepared_plot(), **matplotlib_options())
                    _capture_legend(fig)
                    _capture_ticks(fig)
                    _apply_matplotlib_tick_labels(fig)
                    return fig

            def snapshot():
                keys = ("plot_type", "x", "y", "numeric_subset_columns", "datetime_start", "datetime_end",
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
                # The live draft is the canonical state. UI controls contribute
                # only the fields currently mounted, preserving settings from
                # inactive/dynamic panels instead of silently dropping them.
                with reactive.isolate():
                    result = (config_manager.get(config_tab_id)
                              if config_manager is not None and config_tab_id in config_manager.drafts
                              else dict(initial_filters))
                for key in keys:
                    try:
                        value = (_label_string(key) if key in _label_fields else getattr(input, key)())
                        if value is not None:
                            result[key] = list(value) if isinstance(value, (list, tuple)) else value
                    except Exception:
                        # Inputs in inactive tabs may not have been initialized yet.
                        pass
                # Snapshot the same effective filter state used by the renderer,
                # including saved values for controls not yet mounted.
                with reactive.isolate():
                    effective_filters = filter_state()
                result.update(_filter_config_fields(effective_filters))
                # These editors are reactive dictionaries, not Shiny inputs.
                # Persist their full mappings rather than relying on input.*.
                for field in _label_fields:
                    with reactive.isolate():
                        values = dict(_editor_map(field))
                    result[field] = "; ".join(f"{key}={value}" for key, value in values.items()
                                              if value != key)
                with reactive.isolate():
                    colors = dict(_color_map())
                    chosen = configured_visibility()
                    visible_keys = tuple(available_elements())
                result["trace_colors"] = "; ".join(f"{key}={value}" for key, value in colors.items())
                # No available components must not erase visibility in a
                # retained configuration (e.g. threshold above maximum).
                if not visible_keys:
                    result["visible_elements"] = None if chosen is None else list(chosen)
                else:
                    result["visible_elements"] = list(visible_keys if chosen is None else
                                                      (k for k in chosen if k in visible_keys))
                return result

            # Expose the same dataframe that feeds plot preparation/export.
            # The app-level New table action captures it as a static snapshot.
            snapshot.plot_dataframe = selected_data
            return snapshot


    @module.ui
    def table_tab_ui(frame):
        available = _alphabetical(frame.columns)
        numeric = _alphabetical(_numeric_columns(frame))
        groups = _alphabetical([c for c in available if c not in numeric])
        stages = max(2, len(available) + 1)
        return ui.div(
            ui.layout_sidebar(
                ui.sidebar(
                    ui.div(ui.navset_tab(
                        ui.nav_panel("Table filters",
                            *[
                                ui.div(
                                    ui.input_select(f"subset_column_{i}", f"Subset column {i + 1}",
                                        choices={"": "Select a column", **{c: c for c in available}}),
                                    ui.output_ui(f"subset_values_ui_{i}"),
                                    ui.input_select(f"subset_style_{i}", "Filter style",
                                        choices={k: k for k in ("all", "any", "negative all", "negative any")}),
                                    class_="mb-3",
                                ) if i == 0 else ui.panel_conditional(
                                    " && ".join(f"input.subset_column_{j} !== ''" for j in range(i)),
                                    ui.div(
                                        ui.input_select(f"subset_column_{i}", f"Subset column {i + 1}",
                                            choices={"": "Select a column", **{c: c for c in available}}),
                                        ui.output_ui(f"subset_values_ui_{i}"),
                                        ui.input_select(f"subset_style_{i}", "Filter style",
                                            choices={k: k for k in ("all", "any", "negative all", "negative any")}),
                                        class_="mb-3",
                                    ),
                                ) for i in range(stages)
                            ],
                            ui.input_selectize("numeric_subset_columns", "Numerical subset",
                                choices={c: c for c in numeric}, multiple=True),
                            ui.output_ui("numeric_subset_bounds"),
                            ui.input_text("datetime_start", "Start datetime",
                                placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                            ui.input_text("datetime_end", "End datetime",
                                placeholder="YYYY-MM-DD, HH:MM:SS, or YYYY-MM-DD HH:MM:SS"),
                        ),
                        ui.nav_panel("Frequency/Grouping/Statistical aggregation",
                            ui.input_selectize("group_by", "Group by", choices={c: c for c in groups},
                                multiple=True),
                            ui.input_select("frequency", "Frequency / aggregation mode",
                                choices={k: k for k in FREQUENCIES}),
                            ui.input_select("aggregate", "Apply statistical function",
                                choices={k: k for k in AGGREGATES}),
                            ui.output_ui("frequency_options"),
                        ),
                        id="table_sidebar_tabs"), class_="summary-sidebar-tabs"),
                    title="Table controls", open="closed", width="320px",
                ),
                ui.card(
                    ui.card_header("Data table"),
                    ui.div(ui.output_data_frame("table_grid"), class_="summary-table-grid-area"),
                    class_="summary-table-grid-card",
                ),
            ),
            class_="summary-table-tab",
        )

    @module.server
    def table_tab_server(input, output, session, frame):
        # Preserve Summary methods on snapshots made from plots and tables.
        if hasattr(summary, "subset") and not hasattr(frame, "subset"):
            frame = summary._constructor(frame)
        # Each table starts from its own snapshot and processes it independently.
        available = _alphabetical(frame.columns)
        numeric = _alphabetical(_numeric_columns(frame))
        stages = max(2, len(available) + 1)
        cache = {}

        def choices_for(name):
            if name not in cache:
                cache[name] = _subset_choices(frame, name)
            return cache[name]

        def make_values_ui(stage):
            def values_ui():
                try:
                    name = input[f"subset_column_{stage}"]()
                except Exception:
                    return ui.div()
                if name not in frame.columns:
                    return ui.div()
                choices, _ = choices_for(name)
                return ui.input_selectize(f"subset_values_{stage}", f"{name} values",
                    choices=choices, multiple=True)
            values_ui.__name__ = f"subset_values_ui_{stage}"
            return render.ui(values_ui)

        for stage in range(stages):
            output(make_values_ui(stage))

        @render.ui
        def numeric_subset_bounds():
            controls = []
            for name in list(input.numeric_subset_columns() or []):
                if name not in numeric:
                    continue
                index = numeric.index(name)
                controls.append(ui.div(
                    ui.tags.strong(name),
                    ui.input_text(f"numeric_gt_{index}", "Greater than", placeholder="No lower bound"),
                    ui.input_text(f"numeric_lt_{index}", "Less than", placeholder="No upper bound"),
                ))
            return ui.div(*controls)

        @render.ui
        def frequency_options():
            if FREQUENCIES.get(input.frequency()) is None:
                return ui.div()
            return ui.div(
                ui.input_checkbox("only_complete", "Only complete intervals", value=True),
                ui.input_checkbox("flooring_timestamp", "Floor timestamps", value=True),
            )

        @reactive.calc
        def table_data():
            current = frame.copy(deep=True)
            for stage in range(stages):
                try:
                    name = input[f"subset_column_{stage}"]()
                except Exception:
                    break
                if name not in frame.columns:
                    break
                try:
                    keys = list(input[f"subset_values_{stage}"]() or [])
                except Exception:
                    keys = []
                _, mapping = choices_for(name)
                values = [mapping[key] for key in keys if key in mapping]
                if values:
                    try:
                        style = input[f"subset_style_{stage}"]() or "all"
                    except Exception:
                        style = "all"
                    current = current.subset({name: values}, filter_style=style)
            current = _apply_filters(current, {}, input.datetime_start(), input.datetime_end())
            for name in list(input.numeric_subset_columns() or []):
                if name not in numeric:
                    continue
                index = numeric.index(name)
                values = pd.to_numeric(current[name], errors="coerce")
                for suffix in ("gt", "lt"):
                    try:
                        bound = float(str(input[f"numeric_{suffix}_{index}"]()).strip())
                    except (TypeError, ValueError, KeyError):
                        continue
                    if not np.isfinite(bound):
                        continue
                    mask = values.gt(bound) if suffix == "gt" else values.lt(bound)
                    current = current.loc[mask.fillna(False) | values.isna()].copy()
                    values = values.loc[current.index]
            if current.empty:
                return current
            aggregate = AGGREGATES.get(input.aggregate())
            if aggregate is not None:
                frequency = FREQUENCIES.get(input.frequency())
                grouping = list(input.group_by() or [])
                if frequency is None and not grouping:
                    return pd.DataFrame()
                # Do not exclude identifier by name: preserve the existing
                # Summary.group_stat() semantics and let the grouping selection
                # determine whether a numeric column is an aggregation value.
                # Entirely missing numeric columns remain excluded.
                value_cols = [name for name in numeric if name not in grouping
                              and current[name].notna().any()]
                if not value_cols:
                    return pd.DataFrame()
                if not hasattr(current, "group_stat"):
                    return pd.DataFrame()
                complete = bool(input.only_complete()) if frequency is not None else False
                floor = bool(input.flooring_timestamp()) if frequency is not None else True
                if complete and frequency is not None:
                    # group_stat requires *every* value column to be complete in
                    # the same interval. Evaluate each measurement independently,
                    # then align the resulting aggregate columns on group keys.
                    # This avoids one sparse sensor hiding all other measurements.
                    parts = []
                    for value_col in value_cols:
                        part = _aggregate(current, [value_col], frequency, grouping,
                            aggregate, only_complete=True, flooring_timestamp=floor)
                        if part is not None and not part.empty:
                            parts.append(pd.DataFrame(part).reset_index(drop=True))
                    if parts:
                        current = parts[0]
                        for part in parts[1:]:
                            # Merge on actual grouping/time columns produced by
                            # group_stat(), not on the frequency option string.
                            measurement_cols = {f"{name}_{aggregate}" for name in value_cols}
                            keys = [col for col in current.columns
                                    if col in part.columns and col not in measurement_cols]
                            if keys:
                                current = current.merge(part, on=keys, how="outer", sort=False)
                            else:
                                # No reliable shared keys: preserve the rows of
                                # each result rather than triggering a merge error.
                                current = pd.concat([current, part], ignore_index=True)
                    else:
                        current = pd.DataFrame()
                else:
                    current = _aggregate(current, value_cols, frequency, grouping, aggregate,
                        only_complete=False, flooring_timestamp=floor)
            return pd.DataFrame() if current is None else current

        @render.data_frame
        def table_grid():
            result = table_data()
            # Convert Summary/Group output to plain, grid-compatible pandas
            # columns. Completeness filtering may legitimately produce zero rows.
            result = pd.DataFrame(result).reset_index(drop=True)
            # Hide columns without any observed values in the current table.
            # This is presentation-only: the underlying table data is retained
            # for subsequent filtering, grouping and new-table snapshots.
            result = result.dropna(axis=1, how="all")
            result.columns = [str(c) for c in result.columns]
            result = result.loc[:, ~result.columns.duplicated()].copy()
            for col in result.columns:
                series = result[col]
                if pd.api.types.is_timedelta64_dtype(series.dtype):
                    result[col] = series.astype(str)
                elif pd.api.types.is_object_dtype(series.dtype):
                    # Shiny's browser grid cannot reliably infer mixed Python
                    # objects (including nested values and pd.NA).
                    nonnull = series.dropna()
                    if len(nonnull) and not nonnull.map(lambda v: isinstance(v, (str, int, float, bool, datetime, date, pd.Timestamp))).all():
                        result[col] = series.map(lambda v: str(v) if not pd.isna(v) else None)
                elif pd.api.types.is_numeric_dtype(series.dtype):
                    result[col] = series.replace([np.inf, -np.inf], np.nan)
            return render.DataGrid(result, filters=True, width="100%", height="100%")


    app_ui = ui.page_fluid(
        ui.tags.style("""
            .container, .container-fluid {max-width: none !important;}
            /* Give table grid headers and their filter inputs 50% more space. */
            .summary-table-tab .shiny-data-grid table th,
            .summary-table-tab .shiny-data-grid table td,
            .summary-table-tab .shiny-data-grid .cell,
            .summary-table-tab .shiny-data-grid .header {
                min-width: 150px !important;
            }
            .summary-table-tab .shiny-data-grid input {min-width: 120px;}
            /* Explicit viewport sizing avoids zero-height nested flex children. */
            .summary-table-tab {
                height: calc(100dvh - 185px);
                min-height: 360px;
                overflow: hidden;
            }
            .summary-table-tab > .bslib-sidebar-layout {
                height: 100%;
                min-height: 0;
            }
            .summary-table-tab .bslib-sidebar-layout > aside,
            .summary-table-tab .bslib-sidebar-layout > .sidebar {
                height: 100%;
                overflow-y: auto;
            }
            .summary-table-tab .summary-table-grid-card {
                height: 100%;
                min-height: 0;
            }
            .summary-table-tab .summary-table-grid-card .card-body {
                height: 100%;
                min-height: 0;
                display: flex;
                flex-direction: column;
            }
            .summary-table-tab .summary-table-grid-area {
                flex: 1 1 auto;
                min-height: 0;
                height: 100%;
                overflow: hidden;
            }
            .summary-table-tab .summary-table-grid-area > * {
                height: 100% !important;
                min-height: 0;
            }

            .summary-plot-container {width:100%; height:auto; aspect-ratio: 2 / 1; min-width:0; position:relative;}
            .summary-plot-container > * {width:100% !important;}
            .summary-plot-container .shiny-plot-output {min-height:400px;}
            /* Keep the shared app theme picker right-aligned in the top toolbar.
               This selector is independent of each plot's Plot appearance theme. */
            .summary-top-toolbar {display:flex; align-items:flex-end; gap:0.75rem;
                                  flex-wrap:wrap; margin-bottom:1rem;}
            .summary-toolbar-actions {display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;}
            .summary-app-theme {margin-left:auto; width:220px; max-width:100%;}
            .summary-app-theme .shiny-input-container {width:100%; margin-bottom:0;}
            @media (max-width: 480px) {
                .summary-app-theme {width:100%;}
            }
        """),
        ui.tags.script("""
        (() => {
          // A Plotly relayout mutates its DOM. Observing all descendants and
          // triggering another relayout causes a self-sustaining render loop.
          const observed = new WeakSet();
          const install = () => document.querySelectorAll('.summary-plot-container').forEach(el => {
            if (observed.has(el)) return;
            observed.add(el);
            let lastWidth = -1;
            let pending = false;
            const resize = () => {
              if (pending) return;
              pending = true;
              requestAnimationFrame(() => {
                pending = false;
                const width = Math.round(el.getBoundingClientRect().width);
                if (!width) return;
                const height = Math.round(width / 2);
                if (width === lastWidth && el.style.height === height + 'px') return;
                lastWidth = width;
                el.style.height = height + 'px';
                const graph = el.querySelector('.js-plotly-plot');
                if (graph && window.Plotly) window.Plotly.relayout(graph, {width, height});
              });
            };
            new ResizeObserver(resize).observe(el);
            resize();
          });
          // Only watch tab/container insertions, never individual Plotly nodes.
          let installPending = false;
          new MutationObserver(() => {
            if (installPending) return;
            installPending = true;
            requestAnimationFrame(() => {installPending = false; install();});
          }).observe(document.body, {childList:true, subtree:true});
          document.addEventListener('shown.bs.tab', () => requestAnimationFrame(install));
          install();
        })();
        """),
        ui.div(
            ui.div(
                ui.div(
                    ui.input_action_button("add_tab", "+ New plot", class_="btn-primary"),
                    ui.input_action_button("add_table", "+ New table", class_="btn-primary"),
                    ui.input_action_button("close_tab", "Close selected tab", class_="btn-outline-danger"),
                    ui.input_action_button("retain_plot", "Save selected plot", class_="btn-outline-primary"),
                    ui.input_action_button("restore_plot", "Load saved plot", class_="btn-outline-primary"),
                    class_="summary-toolbar-actions",
                ),
                # Exactly one app-wide theme picker, outside dynamically added
                # plot modules. shinyswatch provides its full theme catalogue.
                ui.div(shinyswatch.theme_picker_ui(), class_="summary-app-theme") if include_theme_picker else ui.div(class_="summary-app-theme"),
                class_="summary-top-toolbar",
            ),
            ui.tags.style("""
                .summary-info-content {max-width: 980px; margin: 1.2rem auto 3rem;
                                       padding: 0 1.25rem; line-height: 1.6;}
                .summary-info-content h3 {margin-top: 2rem;}
                .summary-info-content h4 {margin-top: 1.4rem;}
                .summary-info-content li {margin-bottom: .45rem;}
                .summary-info-description {white-space: pre-wrap; overflow-wrap: anywhere;
                    font: inherit; padding: 1rem 1.25rem; border: 1px solid var(--bs-border-color);
                    border-radius: .5rem; background: var(--bs-tertiary-bg); min-height: 3rem;}
            """),
            ui.navset_tab(info_tab, id="plot_tabs"),
        ),
        title="Summary plots",
    )

    def server(input, output, session):
        nonlocal config_manager
        # Register shinyswatch at the application root (not inside plot modules).
        # Its picker dynamically switches the app's Bootstrap theme without
        # touching plot-specific options or the plot cloning logic.
        if include_theme_picker:
            shinyswatch.theme_picker_server()
        next_tab = reactive.value(0)
        active_tabs = reactive.value(())
        tab_filters = {}
        table_frames = {}
        next_table = reactive.value(0)
        from ..plot_configs import PlotConfigManager
        manager = config_manager or PlotConfigManager(data)
        # Use the same manager in each plot module and in the toolbar actions.
        config_manager = manager
        saved_revision = reactive.value(0)
        # Drafts track live input state; only Save selected plot changes Summary.plots.
        def register_draft(tab_id, snapshot):
            @reactive.effect
            def sync_draft():
                try:
                    state = snapshot()
                    manager.update(tab_id, state)
                except (TypeError, ValueError):
                    # A transient invalid input must not overwrite the last good draft.
                    pass

        @reactive.effect
        @reactive.event(input.retain_plot)
        def retain_selected():
            selected = input.plot_tabs()
            if selected not in tab_filters:
                ui.notification_show("Select a plot tab to save", type="warning")
                return
            ui.modal_show(ui.modal(
                ui.input_text("retained_plot_name", "Configuration name", value=f"Saved {selected}"),
                title="Save plot configuration",
                footer=ui.div(ui.modal_button("Cancel"),
                              ui.input_action_button("confirm_retain_plot", "Save", class_="btn-primary")),
            ))

        @reactive.effect
        @reactive.event(input.confirm_retain_plot)
        def confirm_retain():
            selected = input.plot_tabs()
            try:
                with reactive.isolate():
                    manager.update(selected, tab_filters[selected]())
                    manager.retain(selected, input.retained_plot_name())
                saved_revision.set(saved_revision.get() + 1)
                ui.modal_remove()
                ui.notification_show("Plot configuration retained in Summary.plots", type="message")
            except Exception as exc:
                ui.notification_show(f"Could not save configuration: {exc}", type="error")

        @reactive.effect
        @reactive.event(input.restore_plot)
        def choose_saved():
            saved_revision.get()
            presets = [p for p in manager.saved() if p.get("type") == "plot"]
            if not presets:
                ui.notification_show("No retained plot configurations", type="warning")
                return
            ui.modal_show(ui.modal(
                ui.input_select("retained_plot_choice", "Saved plot",
                                choices={p["id"]: p.get("name", p["id"]) for p in presets}),
                title="Load saved plot",
                footer=ui.div(ui.modal_button("Cancel"),
                              ui.input_action_button("confirm_restore_plot", "Load", class_="btn-primary")),
            ))

        @reactive.effect
        @reactive.event(input.confirm_restore_plot)
        def restore_selected():
            try:
                with reactive.isolate():
                    preset_id = input.retained_plot_choice()
                    number = next_tab.get() + 1
                    tab_id = f"plot_{number}"
                    inherited = manager.restore(preset_id, tab_id)
                    panel = ui.nav_panel(f"Plot {number}", plot_tab_ui(tab_id, inherited), value=tab_id)
                    tab_filters[tab_id] = plot_tab_server(tab_id, inherited, config_tab_id=tab_id)
                    register_draft(tab_id, tab_filters[tab_id])
                    next_tab.set(number)
                    active_tabs.set((*active_tabs.get(), tab_id))
                    ui.insert_nav_panel("plot_tabs", panel, select=True)
                ui.modal_remove()
            except Exception as exc:
                ui.notification_show(f"Could not restore plot: {exc}", type="error")

        @reactive.effect
        @reactive.event(input.add_tab)
        def add_plot():
            number = next_tab.get() + 1
            next_tab.set(number)
            tab_id = f"plot_{number}"
            # Register module before inserting its UI.
            selected = input.plot_tabs()
            # Clone the currently selected plot's complete live configuration.
            with reactive.isolate():
                try:
                    inherited = tab_filters[selected]() if selected in tab_filters else {}
                except Exception:
                    # A missing dynamic input must never disable New plot.
                    inherited = {}
            # Build the new UI before registering the server module. This also
            # avoids leaving a half-registered module when UI creation fails.
            new_panel = ui.nav_panel(f"Plot {number}", plot_tab_ui(tab_id, inherited), value=tab_id)
            tab_filters[tab_id] = plot_tab_server(tab_id, inherited, config_tab_id=tab_id)
            register_draft(tab_id, tab_filters[tab_id])
            active_tabs.set((*active_tabs.get(), tab_id))
            ui.insert_nav_panel("plot_tabs", new_panel, select=True)

        @reactive.effect
        @reactive.event(input.add_table)
        def add_table():
            number = next_table.get() + 1
            next_table.set(number)
            tab_id = f"table_{number}"
            selected = input.plot_tabs()
            with reactive.isolate():
                if selected in table_frames:
                    frame = table_frames[selected].copy(deep=True)
                elif selected in tab_filters:
                    try:
                        frame = tab_filters[selected].plot_dataframe()
                        frame = pd.DataFrame(frame).copy(deep=True)
                    except Exception:
                        # A plot may have incomplete settings; show an empty
                        # grid rather than disabling the New table action.
                        frame = pd.DataFrame()
                else:
                    frame = pd.DataFrame(summary).copy(deep=True)
            # Datagrid requires plain column names; preserve the values and
            # index as displayed columns where useful.
            frame = frame.reset_index(drop=True)
            frame.columns = [str(c) for c in frame.columns]
            panel = ui.nav_panel(f"Table {number}", table_tab_ui(tab_id, frame), value=tab_id)
            table_frames[tab_id] = frame
            table_tab_server(tab_id, frame)
            active_tabs.set((*active_tabs.get(), tab_id))
            ui.insert_nav_panel("plot_tabs", panel, select=True)

        @reactive.effect
        @reactive.event(input.close_tab)
        def close_plot():
            selected = input.plot_tabs()
            tabs = active_tabs.get()
            if selected not in tabs:
                return
            ui.remove_nav_panel("plot_tabs", target=selected)
            table_frames.pop(selected, None)
            tab_filters.pop(selected, None)
            manager.discard(selected)
            active_tabs.set(tuple(t for t in tabs if t != selected))
            # Shiny does not provide a public API to unregister an output module.
            # The removed tab cannot be selected again and tab IDs are never reused.

    return app_ui, server


def create_summary_app(data, **kwargs):
    """Create a dataset-bound Shiny app (legacy compatibility API)."""
    from shiny import App
    app_ui, server = _build_summary_ui_server(data, **kwargs)
    return App(app_ui, server)


def run_summary_app(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
    host: str = "127.0.0.1",
    port: int = 0,
    launch_browser: bool = True,
    daemon: bool = True,
    backend: str = "plotly",
):
    """Create and start the Shiny app in a background thread."""
    from shiny import run_app

    app = create_summary_app(data, x=x, y=y, group_by=group_by, plot_type=plot_type, title=title, backend=backend)

    def _run() -> None:
        run_app(app, host=host, port=port, launch_browser=launch_browser, dev_mode=False)

    thread = threading.Thread(target=_run, name="apmtools-shiny-summary", daemon=daemon)
    thread.start()
    return thread

