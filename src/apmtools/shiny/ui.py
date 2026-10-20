"""Shiny application controls, reactive server and launch."""
from __future__ import annotations
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
from .export import _safe_csv_filename, _standalone_matplotlib_script


def create_summary_app(
    data: pd.DataFrame,
    *,
    x: str | None = None,
    y: str | Sequence[str] | None = None,
    group_by: str | Sequence[str] | None = None,
    plot_type: str = "line",
    title: str = "Summary plot",
    backend: str = "plotly",
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
    # Stable choices are shared across tabs. Computing unique values from the
    # full dataset on every filter change was expensive for large summaries.
    subset_choice_cache = {}

    def cached_subset_choices(name):
        if name not in subset_choice_cache:
            subset_choice_cache[name] = _subset_choices(summary, column_map[name])
        return subset_choice_cache[name]

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
            ui.layout_sidebar(
                ui.sidebar(
                    ui.div(ui.navset_tab(
                    ui.nav_panel("Plot settings",
                        ui.input_select("plot_type", "Plot type", choices=PLOT_TYPES, selected=initial("plot_type", plot_type)),
                        ui.output_ui("x_control"),
                        ui.output_ui("y_control")),
                    ui.nav_panel("Plot filters",
                        ui.input_selectize("subset_columns", "Subset columns",
                            choices={c: c for c in subset_columns}, selected=initial("subset_columns", []), multiple=True),
                        ui.output_ui("subset_filters"),
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
                            ui.input_numeric("title_font_size", "Plot title font size", value=initial("title_font_size", 14), min=6, max=72)),
                        ui.nav_panel("Legend",
                            ui.input_select("legend_position", "Legend position", choices={k:k.title() for k in ("right", "left", "top", "bottom")}, selected=initial("legend_position", "right")),
                            ui.input_checkbox("legend_inside", "Place legend inside plot", value=initial("legend_inside", False)),
                            ui.input_text_area("legend_labels", "Legend labels (old=new; ...)", value=initial("legend_labels", ""), rows=6),
                            (ui.input_numeric("legend_columns", "Legend columns", value=initial("legend_columns", 1), min=1, max=100, step=1) if backend == "matplotlib" else ui.div()),
                            ui.input_numeric("legend_font_size", "Legend font size", value=initial("legend_font_size", 12), min=6, max=72)),
                        ui.nav_panel("Axes",
                            ui.input_text("x_axis_label", "X-axis label", value=initial("x_axis_label", "")),
                            ui.input_text("y_axis_label", "Y-axis label", value=initial("y_axis_label", "")),
                            ui.layout_columns(
                                (ui.input_text_area("x_min", "X minimum", value=initial("x_min", ""), rows=4) if backend == "matplotlib" else ui.input_text("x_min", "X minimum", value="")),
                                (ui.input_text_area("x_max", "X maximum", value=initial("x_max", ""), rows=4) if backend == "matplotlib" else ui.input_text("x_max", "X maximum", value="")),
                                col_widths=(6, 6)),
                            ui.layout_columns(
                                (ui.input_text_area("y_min", "Y minimum", value=initial("y_min", ""), rows=4) if backend == "matplotlib" else ui.input_text("y_min", "Y minimum", value="")),
                                (ui.input_text_area("y_max", "Y maximum", value=initial("y_max", ""), rows=4) if backend == "matplotlib" else ui.input_text("y_max", "Y maximum", value="")),
                                col_widths=(6, 6)),
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
                            ui.input_text_area("trace_colors", "Trace colors (name=#hex; ...)", value=initial("trace_colors", ""), rows=6),
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
                                    ui.input_text_area("panel_titles", "Panel titles (old=new; ...)", value=initial("panel_titles", ""), rows=6)),
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
    def plot_tab_server(input, output, session, initial_filters=None):
            initial_filters = initial_filters or {}
            @render.ui
            def subset_filters():
                controls = []
                for i, name in enumerate(list(input.subset_columns())):
                    if name not in column_map:
                        continue
                    choices, _ = cached_subset_choices(name)
                    controls.append(ui.input_selectize(
                        f"subset_values_{i}", f"{name} values", choices=choices,
                        selected=initial_filters.get("subset_values", {}).get(name, []),
                        multiple=True,
                    ))
                return ui.div(*controls) if controls else ui.help_text("Select columns to filter this plot.")

            @render.ui
            def numeric_subset_bounds():
                controls = []
                for name in list(input.numeric_subset_columns() or []):
                    if name not in numeric_columns:
                        continue
                    # Stable IDs per column preserve entered bounds when selections change.
                    index = numeric_columns.index(name)
                    bounds = initial_filters.get("numeric_subset_bounds", {}).get(name, {})
                    controls.append(ui.div(
                        ui.tags.strong(name),
                        ui.input_text(f"numeric_gt_{index}", "Greater than",
                            value=str(bounds.get("gt", "")), placeholder="No lower bound"),
                        ui.input_text(f"numeric_lt_{index}", "Less than",
                            value=str(bounds.get("lt", "")), placeholder="No upper bound"),
                        class_="mb-3",
                    ))
                return ui.div(*controls)

            @reactive.calc
            def filter_state():
                names = list(input.subset_columns())
                values = {}
                for i, name in enumerate(names):
                    if name in column_map:
                        values[name] = list(input[f"subset_values_{i}"]())
                return {"columns": names, "values": values,
                        "start": input.datetime_start().strip(),
                        "end": input.datetime_end().strip()}

            @reactive.calc
            def filtered_summary():
                state = filter_state()
                filters = {}
                for name, keys in state["values"].items():
                    _, mapping = cached_subset_choices(name)
                    selected = [mapping[key] for key in keys if key in mapping]
                    if selected:
                        filters[column_map[name]] = selected
                # Preserve original row adjacency before numerical filtering so a
                # line cannot bridge observations removed by a numeric bound.
                source = summary
                if input.plot_type() == "line" and AGGREGATES.get(input.aggregate()) is None and list(input.numeric_subset_columns() or []):
                    source = summary.copy()
                    order = source.sort_values((["identifier"] if "identifier" in source.columns else []) + ["datetime"], kind="stable")
                    if "identifier" in order.columns:
                        source.loc[order.index, "__source_row_order"] = order.groupby("identifier", dropna=False).cumcount().to_numpy()
                    else:
                        source.loc[order.index, "__source_row_order"] = np.arange(len(order))
                current = _apply_filters(source, filters, state["start"], state["end"])
                for name in list(input.numeric_subset_columns() or []):
                    if name not in numeric_columns:
                        continue
                    index = numeric_columns.index(name)
                    values = pd.to_numeric(current[column_map[name]], errors="coerce")
                    for suffix, comparison in (("gt", "gt"), ("lt", "lt")):
                        try:
                            raw = input[f"numeric_{suffix}_{index}"]()
                        except Exception:
                            raw = ""
                        try:
                            bound = float(str(raw).strip())
                        except (TypeError, ValueError):
                            continue  # Empty/incomplete inputs do not filter.
                        if not np.isfinite(bound):
                            continue
                        mask = values.gt(bound) if comparison == "gt" else values.lt(bound)
                        current = current.loc[mask.fillna(False)].copy()
                        values = values.loc[current.index]
                return current

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
                    multiple = False

                if not choices:
                    return ui.help_text(f"No compatible X variables for {kind}.")
                selected = initial_filters.get("x", [])
                if multiple:
                    return ui.input_selectize(
                        "x", "X-axis / categories",
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
                selected = ""
                if kind in ("box", "bar", "scatter"):
                    return ui.input_select(
                        "y", "Y-axis / variable",
                        choices={"": "None", **{c: c for c in numeric_columns}},
                        selected=initial_filters.get("y", ""),
                    )
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
                ui.update_selectize("panel_by", choices={name: name for name in selected},
                                    selected=selected)

            @render.ui
            def frequency_options():
                if FREQUENCIES.get(input.frequency()) is None:
                    return ui.div()
                return ui.div(
                    ui.input_checkbox("only_complete", "Only complete intervals", value=initial_filters.get("only_complete", True)),
                    ui.input_checkbox("flooring_timestamp", "Floor timestamps", value=initial_filters.get("flooring_timestamp", True)),
                )

            @reactive.calc
            def selected_data():
                current = filtered_summary()
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

                selected_groups = list(input.group_by() or [])
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
                    # Both numerical axes must be aggregated on the same rows.
                    value_columns = list(dict.fromkeys(
                        [column_map[selected_x[0]], *value_columns]
                    ))
                if (kind == "line" and selected_x[0] in timedelta_columns
                        and should_aggregate):
                    # group_stat aggregation drops the original timedelta X
                    # unless it is explicitly aggregated. Never silently plot
                    # an unrelated frequency column instead.
                    return pd.DataFrame()
                if should_aggregate:
                    current = _aggregate(
                        current, value_columns, frequency, grouping_by, aggregate,
                        only_complete=(bool(input.only_complete()) if frequency is not None else True),
                        flooring_timestamp=(bool(input.flooring_timestamp()) if frequency is not None else True),
                    )
                return current

            @reactive.calc
            def prepared_plot():
                plot_df = selected_data()
                selected_plot = input.plot_type()
                x_input = input.x()
                selected_x = ([x_input] if isinstance(x_input, str) else list(x_input or []))
                y_input = None if selected_plot == "histogram" else input.y()
                selected_y = ([y_input] if isinstance(y_input, str) else list(y_input or []))
                selected_groups = list(input.group_by() or [])
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
                    y_columns = [mapped(selected_y[0])]
                elif selected_plot == "scatter":
                    x_column = mapped(selected_x[0])
                    y_columns = [mapped(selected_y[0])]
                else:
                    x_column = (column_map[selected_x[0]] if selected_x[0] in timedelta_columns
                                else frequency if frequency is not None
                                else column_map[selected_x[0]])
                    y_columns = [mapped(c) for c in selected_y]

                if selected_plot == "line" and should_aggregate and frequency is None:
                    return {"error": "Categorical-only aggregation has no datetime axis; use a bar plot or select a frequency"}

                required = [x_column, *y_columns]
                if selected_plot in ("box", "bar"):
                    required.extend(x_columns)
                if any(c not in plot_df.columns for c in required):
                    return {"error": "Selected variables are unavailable after aggregation"}

                if selected_plot in ("box", "bar"):
                    plot_df = plot_df.copy(deep=False)
                    plot_df["__plot_category"] = _composite_labels(plot_df, x_columns)
                    x_column = "__plot_category"

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
                group_columns = [column_map[c] for c in selected_groups
                                 if column_map[c] in plot_df.columns]
                if group_columns:
                    plot_df = plot_df.copy(deep=False)
                    plot_df["__plot_group"] = (
                        _composite_labels(plot_df, group_columns)
                    )

                return {"data": plot_df, "kind": selected_plot, "x": x_column,
                        "ys": y_columns, "groups": group_columns,
                        "categories": x_columns if selected_plot in ("box", "bar") else None,
                        "frequency": frequency, "x_label_override": x_label_override}

            @reactive.calc
            def base_figure():
                import plotly.express as px
                spec = prepared_plot()
                if "error" in spec:
                    return px.scatter(title=spec["error"])
                plot_df = spec["data"]
                selected_plot = spec["kind"]
                x_column = spec["x"]
                y_columns = spec["ys"]
                group_columns = spec["groups"]
                frequency = spec["frequency"]
                x_columns = spec["categories"]
                if selected_plot == "histogram":
                    fig = _histogram_figure(
                        plot_df, x_column,
                        group_column="__plot_group" if group_columns else None,
                        bins=int(input.bins()),
                        density=input.hist_density() == "True",
                        stacked=input.hist_stacked() == "True",
                        histtype=input.hist_type(),
                        cumulative=int(input.hist_cumulative()),
                    )
                elif selected_plot == "box":
                    fig = _boxplot_figure(plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
                elif selected_plot == "bar":
                    fig = _barplot_figure(plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
                elif selected_plot == "scatter":
                    fig = _scatter_figure(plot_df, x_column, y_columns[0], group_column="__plot_group" if group_columns else None)
                else:
                    fig = _lineplot_figure(plot_df, x_column, y_columns, group_columns=group_columns, frequency=frequency, column_map=column_map)

                if not group_columns:
                    # Retain trace names/colours for appearance editing, but
                    # suppress the redundant ungrouped legend.
                    fig.update_traces(showlegend=False)
                    fig.update_layout(showlegend=False)

                _label_plot_axes(
                    fig, selected_plot, x_column, y_columns,
                    category_columns=x_columns if selected_plot in ("box", "bar") else None,
                    x_label_override=spec.get("x_label_override"),
                )
                fig.update_layout(
                    hovermode="x unified" if selected_plot == "line" else "closest",
                    margin=dict(l=60, r=30, t=70, b=60),
                    autosize=True,
                )
                return fig

            # Populate appearance controls only after a valid plot has been
            # prepared.  The effect depends on the plot specification, not on
            # appearance inputs, so editing a label or colour cannot reset it.
            @reactive.effect
            @reactive.event(prepared_plot, lambda: _optional_input("multi_panel", False) if input.group_by() else False,
                            lambda: tuple(_optional_input("panel_by", []) or []) if input.group_by() and len(input.group_by()) > 1 else (),
                            lambda: _optional_input("independent_panel_limits", True) if input.group_by() and _optional_input("multi_panel", False) else True, ignore_none=False)
            def populate_appearance_from_plot():
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
                ui.update_text("title", value="Summary plot")
                ui.update_text("x_axis_label", value=defaults["x_label"])
                ui.update_text("y_axis_label", value=defaults["y_label"])
                limits = _axis_bounds_from_spec(
                    spec,
                    bins=int(input.bins()) if spec["kind"] == "histogram" else 30,
                    density=input.hist_density() == "True" if spec["kind"] == "histogram" else False,
                    stacked=(input.hist_stacked() == "True" or input.hist_type() == "barstacked") if spec["kind"] == "histogram" else False,
                    cumulative=int(input.hist_cumulative()) if spec["kind"] == "histogram" else 0,
                )
                if spec["kind"] == "box" and not input.box_showfliers():
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
                        if spec["kind"] == "box" and not input.box_showfliers():
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
                ui.update_text_area("legend_labels", value="; ".join(
                    f"{name}={name}" for name in names))
                ui.update_text_area("trace_colors", value="; ".join(
                    f"{name}={defaults['colors'][name]}" for name in names))
                kind = spec["kind"]
                if kind == "line":
                    ui.update_text_area("line_alpha", value="; ".join(f"{name}=1" for name in names))
                    ui.update_text_area("line_style", value="; ".join(f"{name}=solid" for name in names))
                elif kind == "histogram":
                    ui.update_text_area("hist_alpha", value="; ".join(f"{name}=0.65" for name in names))
                elif kind == "scatter":
                    ui.update_text_area("scatter_size", value="; ".join(f"{name}=7" for name in names))
                    ui.update_text_area("scatter_marker", value="; ".join(f"{name}=dot" for name in names))
                    ui.update_text_area("scatter_alpha", value="; ".join(f"{name}=1" for name in names))
                if backend == "matplotlib":
                    panel_variables = ((list(_optional_input("panel_by", []) or []) or selected_groups) if len(selected_groups) > 1
                                       else selected_groups)
                    panel_names = ([name for name, _, _ in _panel_partitions(spec, panel_variables)]
                                   if selected_groups and bool(_optional_input("multi_panel", False)) else [])
                    ui.update_text_area("panel_titles", value="; ".join(
                        f"{name}={name}" for name in panel_names))

            @reactive.effect
            @reactive.event(input.box_showfliers, input.box_orientation, ignore_init=True)
            def update_box_axis_for_fliers():
                """Refresh only the boxplot value-axis bounds when fliers change."""
                spec = prepared_plot()
                if spec.get("kind") != "box" or "error" in spec:
                    return
                axis = "x" if input.box_orientation() == "horizontal" else "y"
                if input.box_showfliers():
                    vals = pd.to_numeric(spec["data"][spec["ys"][0]], errors="coerce")
                    vals = vals[np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))]
                    if vals.empty:
                        return
                    lo, hi = float(vals.min()), float(vals.max())
                    pad = .05 * (hi - lo) if hi > lo else max(abs(lo) * .05, .5)
                    limits = [lo - pad, hi + pad]
                else:
                    from .data import _box_whisker_limits
                    grouping = [spec["x"]] + (["__plot_group"] if spec["groups"] else [])
                    limits = _box_whisker_limits(spec["data"], spec["ys"][0], grouping)
                if limits is None:
                    return
                if (backend == "matplotlib" and input.group_by()
                        and bool(_optional_input("multi_panel", False))
                        and bool(_optional_input("independent_panel_limits", True))):
                    groups = list(input.group_by() or [])
                    panel_by = ((list(_optional_input("panel_by", []) or []) or groups)
                                if len(groups) > 1 else groups)
                    from .data import _box_whisker_limits
                    panel_values = {}
                    for panel_title, part, remaining in _panel_partitions(spec, panel_by):
                        if input.box_showfliers():
                            vals = pd.to_numeric(part[spec["ys"][0]], errors="coerce")
                            vals = vals[np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan))]
                            if vals.empty:
                                continue
                            lo, hi = float(vals.min()), float(vals.max())
                            pad = .05 * (hi - lo) if hi > lo else max(abs(lo) * .05, .5)
                            panel_values[panel_title] = [lo - pad, hi + pad]
                        else:
                            group_cols = [spec["x"]] + (["__plot_group"] if remaining else [])
                            panel_values[panel_title] = _box_whisker_limits(part, spec["ys"][0], group_cols)
                    panel_values = {name: val for name, val in panel_values.items() if val is not None}
                    if panel_values:
                        for idx, key in enumerate((f"{axis}_min", f"{axis}_max")):
                            ui.update_text_area(key, value="\n".join(
                                f"{name}={values[idx]}" for name, values in panel_values.items()))
                        return
                for key, value in zip((f"{axis}_min", f"{axis}_max"), limits):
                    if backend == "matplotlib":
                        ui.update_text_area(key, value=str(value))
                    else:
                        ui.update_text(key, value=str(value))

            @reactive.effect
            @reactive.event(input.plot_type, ignore_init=True)
            def reset_on_plot_type_change():
                # Reset every plot-specific selection when the plot type changes.
                ui.update_selectize("subset_columns", selected=[])
                ui.update_selectize("numeric_subset_columns", selected=[])
                ui.update_text("datetime_start", value="")
                ui.update_text("datetime_end", value="")
                ui.update_selectize("group_by", selected=[])
                ui.update_checkbox("multi_panel", value=False)
                ui.update_checkbox("independent_panel_limits", value=True)
                ui.update_select("frequency", selected="None")
                ui.update_select("aggregate", selected="None")
                ui.update_text("title", value="")
                for field in ("x_axis_label", "y_axis_label"):
                    ui.update_text(field, value="")
                for field in ("x_min", "x_max", "y_min", "y_max"):
                    if backend == "matplotlib":
                        ui.update_text_area(field, value="")
                    else:
                        ui.update_text(field, value="")
                for field in ("legend_labels", "trace_colors"):
                    ui.update_text_area(field, value="")
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

            if backend == "plotly":
                @reactive.calc
                def styled_plotly_figure():
                    import plotly.graph_objects as go
                    fig = go.Figure(base_figure())
                    styled = _apply_plot_appearance(
                        fig, title=input.title(), x_label=input.x_axis_label(),
                        y_label=input.y_axis_label(), legend_labels=input.legend_labels(),
                        trace_colors=input.trace_colors(), palette=input.palette(),
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
                    return styled_plotly_figure()
            else:
                @reactive.calc
                def matplotlib_options():
                    # Shared with the export so that all appearance options,
                    # including per-panel overrides, are reproduced exactly.
                    return dict(title=_optional_input("title", ""),
                        x_label=_optional_input("x_axis_label", ""), y_label=_optional_input("y_axis_label", ""),
                        legend_labels=_optional_input("legend_labels", ""), trace_colors=_optional_input("trace_colors", ""),
                        palette=_optional_input("palette", "Default"), x_min=_optional_input("x_min", ""), x_max=_optional_input("x_max", ""),
                        y_min=_optional_input("y_min", ""), y_max=_optional_input("y_max", ""), template=_optional_input("template", "plotly"),
                        legend_position=_optional_input("legend_position", "right"), legend_inside=_optional_input("legend_inside", False),
                        legend_columns=max(1, int(_optional_input("legend_columns", 1) or 1)),
                        show_grid=_optional_input("show_grid", True),
                        x_major_ticks=_optional_input("x_major_ticks", 6), y_major_ticks=_optional_input("y_major_ticks", 6),
                        x_minor_ticks=_optional_input("x_minor_ticks", False), y_minor_ticks=_optional_input("y_minor_ticks", False),
                        title_font_size=_optional_input("title_font_size", 14), axis_font_size=_optional_input("axis_font_size", 12),
                        tick_font_size=_optional_input("tick_font_size", 10), legend_font_size=_optional_input("legend_font_size", 12),
                        panel_font_size=_optional_input("panel_font_size", 14), hist_alpha=_optional_input("hist_alpha", ""),
                        box_showfliers=_optional_input("box_showfliers", True), box_orientation=_optional_input("box_orientation", "vertical"),
                        line_width=(_optional_input("line_width", 2) if input.plot_type() == "line" else 2), line_alpha=_optional_input("line_alpha", ""),
                        line_style=_optional_input("line_style", ""), scatter_size=_optional_input("scatter_size", ""),
                        scatter_marker=_optional_input("scatter_marker", ""), scatter_alpha=_optional_input("scatter_alpha", ""),
                        bins=int(_optional_input("bins", 30)) if input.plot_type() == "histogram" else 30,
                        density=_optional_input("hist_density", "False") == "True" if input.plot_type() == "histogram" else False,
                        stacked=_optional_input("hist_stacked", "False") == "True" if input.plot_type() == "histogram" else False,
                        histtype=_optional_input("hist_type", "bar") if input.plot_type() == "histogram" else "bar",
                        cumulative=int(_optional_input("hist_cumulative", "0")) if input.plot_type() == "histogram" else 0,
                        panel_by=(
                            ((list(_optional_input("panel_by", []) or []) or list(input.group_by() or [])) if len(list(input.group_by() or [])) > 1
                             else list(input.group_by() or []))
                            if list(input.group_by() or []) and bool(_optional_input("multi_panel", False)) else None
                        ),
                        independent_panel_limits=(bool(_optional_input("independent_panel_limits", True))
                                                  if input.group_by() and _optional_input("multi_panel", False) else True),
                        panel_titles=(_optional_input("panel_titles", "") if input.group_by() and _optional_input("multi_panel", False) else ""),                    )

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
                    return _matplotlib_direct_figure(prepared_plot(), **matplotlib_options())

            def snapshot():
                keys = ("plot_type", "x", "y", "subset_columns", "numeric_subset_columns", "datetime_start", "datetime_end",
                        "group_by", "frequency", "aggregate", "only_complete", "flooring_timestamp",
                        "palette", "template", "title", "title_font_size", "legend_position",
                        "legend_inside", "legend_labels", "legend_columns", "legend_font_size",
                        "x_axis_label", "y_axis_label", "x_min", "x_max", "y_min", "y_max",
                        "show_grid", "x_major_ticks", "x_minor_ticks", "y_major_ticks",
                        "y_minor_ticks", "axis_font_size", "tick_font_size", "trace_colors",
                        "line_width", "line_alpha", "line_style", "scatter_size", "scatter_marker",
                        "scatter_alpha", "bins", "hist_density", "hist_stacked", "hist_type",
                        "hist_cumulative", "hist_alpha", "box_showfliers", "box_orientation",
                        "multi_panel", "independent_panel_limits", "panel_by", "panel_titles",
                        "panel_font_size")
                result = dict(initial_filters)
                for key in keys:
                    try:
                        value = getattr(input, key)()
                        if value is not None:
                            result[key] = list(value) if isinstance(value, (list, tuple)) else value
                    except Exception:
                        # Inputs in inactive tabs may not have been initialized yet.
                        pass
                subset = {}
                for i, name in enumerate(result.get("subset_columns", []) or []):
                    try:
                        subset[name] = list(input[f"subset_values_{i}"]() or [])
                    except Exception:
                        subset[name] = initial_filters.get("subset_values", {}).get(name, [])
                result["subset_values"] = subset
                numeric_bounds = {}
                for name in result.get("numeric_subset_columns", []) or []:
                    if name not in numeric_columns:
                        continue
                    index = numeric_columns.index(name)
                    numeric_bounds[name] = {}
                    for suffix in ("gt", "lt"):
                        try:
                            numeric_bounds[name][suffix] = input[f"numeric_{suffix}_{index}"]()
                        except Exception:
                            numeric_bounds[name][suffix] = initial_filters.get("numeric_subset_bounds", {}).get(name, {}).get(suffix, "")
                result["numeric_subset_bounds"] = numeric_bounds
                return result

            return snapshot


    app_ui = ui.page_fluid(
        ui.tags.style("""
            .container, .container-fluid {max-width: none !important;}
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
                    ui.input_action_button("close_tab", "Close selected plot", class_="btn-outline-danger"),
                    class_="summary-toolbar-actions",
                ),
                # Exactly one app-wide theme picker, outside dynamically added
                # plot modules. shinyswatch provides its full theme catalogue.
                ui.div(shinyswatch.theme_picker_ui(), class_="summary-app-theme"),
                class_="summary-top-toolbar",
            ),
            ui.navset_tab(ui.nav_panel("Plot 1", plot_tab_ui("plot_1", {}), value="plot_1"), id="plot_tabs"),
        ),
        title="Summary plots",
    )

    def server(input, output, session):
        # Register shinyswatch at the application root (not inside plot modules).
        # Its picker dynamically switches the app's Bootstrap theme without
        # touching plot-specific options or the plot cloning logic.
        shinyswatch.theme_picker_server()
        next_tab = reactive.value(1)
        active_tabs = reactive.value(("plot_1",))
        tab_filters = {"plot_1": plot_tab_server("plot_1", {})}

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
            tab_filters[tab_id] = plot_tab_server(tab_id, inherited)
            active_tabs.set((*active_tabs.get(), tab_id))
            ui.insert_nav_panel("plot_tabs", new_panel, select=True)

        @reactive.effect
        @reactive.event(input.close_tab)
        def close_plot():
            selected = input.plot_tabs()
            tabs = active_tabs.get()
            if selected not in tabs or len(tabs) <= 1:
                return
            ui.remove_nav_panel("plot_tabs", target=selected)
            active_tabs.set(tuple(t for t in tabs if t != selected))
            # Shiny does not provide a public API to unregister an output module.
            # The removed tab cannot be selected again and tab IDs are never reused.

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

