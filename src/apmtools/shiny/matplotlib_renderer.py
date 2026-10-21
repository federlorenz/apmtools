"""Native and converted Matplotlib figures and panel layout."""
from __future__ import annotations
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES
from .appearance import _axis_label, _bounded_float, _line_style, _matplotlib_tick_controls, _parse_axis_limit, _parse_label_map, _scatter_marker, _series_style_maps
from .data import _add_gap_sections, _add_source_sections, _composite_labels


def _matplotlib_defaults(spec, *, palette="Default"):
    """Derive labels and series colours directly from prepared pandas data."""
    kind, df = spec["kind"], spec["data"]
    x, ys, groups = spec["x"], spec["ys"], spec["groups"]
    names = ([str(v) for v in pd.unique(df["__plot_group"])]
             if groups and "__plot_group" in df else [])
    if not names and kind == "line" and len(ys) > 1:
        names = [str(y) for y in ys]
    if not names:
        names = ["All observations"]
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    colors = {name: palette_values[i % len(palette_values)] for i, name in enumerate(names)}
    x_label = (" | ".join(_axis_label(c) for c in spec["categories"])
               if kind in ("box", "bar") else _axis_label(x))
    y_label = ("Probability density" if kind == "histogram" else
               "Value" if kind == "line" and len(ys) > 1 else
               _axis_label(ys[0]) if ys else "Count")
    x_label = spec.get("x_label_override") or x_label
    return dict(x_label=x_label, y_label=y_label, names=names, colors=colors)



def _axis_bounds_from_spec(spec, *, bins=30, density=False, stacked=False, cumulative=0):
    """Compute editable axis bounds from the prepared data, without rendering.

    Datetime X bounds are ISO timestamps; categorical X bounds are intentionally
    omitted. Values are calculated only when a valid plot specification exists.
    """
    if "error" in spec:
        return {key: "" for key in ("x_min", "x_max", "y_min", "y_max")}
    df, kind, x, ys = spec["data"], spec["kind"], spec["x"], spec["ys"]
    bounds = {key: "" for key in ("x_min", "x_max", "y_min", "y_max")}

    def fill(axis, values):
        if pd.api.types.is_datetime64_any_dtype(values):
            valid = pd.to_datetime(values, errors="coerce").dropna()
            if len(valid):
                bounds[f"{axis}_min"] = valid.min().isoformat(sep=" ")
                bounds[f"{axis}_max"] = valid.max().isoformat(sep=" ")
        else:
            valid = pd.to_numeric(values, errors="coerce")
            valid = valid[np.isfinite(valid.to_numpy(dtype=float, na_value=np.nan))]
            if len(valid):
                bounds[f"{axis}_min"] = format(float(valid.min()), ".12g")
                bounds[f"{axis}_max"] = format(float(valid.max()), ".12g")

    if kind not in ("bar", "box"):
        fill("x", df[x])
    if kind == "histogram":
        values = pd.to_numeric(df[x], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        if len(values):
            groups = ([pd.to_numeric(part[x], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                       for _, part in df.groupby("__plot_group", sort=False)]
                      if spec["groups"] and "__plot_group" in df else [values])
            edges = np.histogram_bin_edges(values, bins=max(1, int(bins)))
            histograms = np.array([np.histogram(v[np.isfinite(v)], bins=edges)[0]
                                   for v in groups], dtype=float)
            if density:
                total = histograms.sum()
                if total:
                    histograms /= total * np.diff(edges)
            if cumulative:
                histograms = histograms.cumsum(axis=1)
            ymax = (histograms.sum(axis=0).max() if stacked and histograms.size
                    else histograms.max() if histograms.size else 0)
            bounds["y_min"], bounds["y_max"] = "0", format(float(ymax), ".12g")
    elif kind == "bar":
        values = pd.to_numeric(df[ys[0]], errors="coerce")
        cols = [x, "__plot_group"] if spec["groups"] else [x]
        totals = (df.assign(__axis_value=values).groupby(cols, dropna=False)["__axis_value"]
                  .sum(min_count=1).dropna())
        if len(totals):
            bounds["y_min"] = format(float(min(0, totals.min())), ".12g")
            bounds["y_max"] = format(float(max(0, totals.max())), ".12g")
    else:
        values = pd.concat([pd.to_numeric(df[y], errors="coerce") for y in ys], ignore_index=True)
        fill("y", values)
    return bounds



def _pad_axis_bounds(bounds, fraction=0.05):
    """Return bounds expanded by 5% of the data span on each side.

    Datetime X bounds retain their timestamp representation. Zero-width
    numeric ranges receive a small nonzero span to avoid singular axes.
    """
    result = dict(bounds)
    for axis in ("x", "y"):
        lo_key, hi_key = f"{axis}_min", f"{axis}_max"
        lo, hi = result.get(lo_key, ""), result.get(hi_key, "")
        if not lo or not hi:
            continue
        try:
            low, high = float(lo), float(hi)
            if not (np.isfinite(low) and np.isfinite(high)):
                continue
            span = high - low
            margin = fraction * (span if span > 0 else max(abs(low), 1.0))
            result[lo_key] = format(low - margin, ".12g")
            result[hi_key] = format(high + margin, ".12g")
        except (ValueError, TypeError):
            if axis != "x":
                continue
            try:
                low, high = pd.Timestamp(lo), pd.Timestamp(hi)
                span = high - low
                margin = span * fraction if span > pd.Timedelta(0) else pd.Timedelta(seconds=1)
                result[lo_key] = (low - margin).isoformat(sep=" ")
                result[hi_key] = (high + margin).isoformat(sep=" ")
            except (ValueError, TypeError, OverflowError):
                continue
    return result



def _panel_columns(spec, panel_by):
    """Resolve one or more selected facet columns in their selected order."""
    selected = [panel_by] if isinstance(panel_by, str) else list(panel_by or [])
    return [column for column in selected
            if column in spec.get("groups", ()) and column in spec.get("data", ())]



def _panel_partitions(spec, panel_by):
    """Yield panel title, panel data and remaining colour-group columns."""
    columns = _panel_columns(spec, panel_by)
    if not columns:
        return
    remaining = [g for g in spec["groups"] if g not in columns]
    grouper = columns[0] if len(columns) == 1 else columns
    for value, part in spec["data"].groupby(grouper, sort=False, dropna=False):
        values = value if isinstance(value, tuple) else (value,)
        title = " | ".join("None" if pd.isna(v) else str(v) for v in values)
        if remaining:
            part = part.copy(deep=False)
            part["__plot_group"] = (_composite_labels(part, remaining))
        yield title, part, remaining



def _panel_trace_defaults(spec, panel_by, *, palette="Default"):
    """Appearance entries for the series actually drawn inside each panel.

    Facet columns are removed from the colour grouping.  When all grouping
    columns are facets, each panel contains an ungrouped "All observations"
    trace; when some grouping columns remain, their values label the traces.
    """
    names = []
    for _, part, remaining in _panel_partitions(spec, panel_by):
        panel_spec = {**spec, "data": part, "groups": remaining}
        for name in _matplotlib_defaults(panel_spec, palette=palette)["names"]:
            if name not in names:
                names.append(name)
    if not names:
        return _matplotlib_defaults(spec, palette=palette)
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    return {"names": names,
            "colors": {name: palette_values[i % len(palette_values)]
                       for i, name in enumerate(names)}}



def _panel_axis_bounds(spec, panel_by, **kwargs):
    """Per-facet data limits, keeping categorical X limits blank."""
    if "error" in spec:
        return {}
    result = {}
    for title, part, remaining in _panel_partitions(spec, panel_by):
        sub_spec = {**spec, "data": part, "groups": remaining}
        result[title] = _axis_bounds_from_spec(sub_spec, **kwargs)
    return result



def _panel_limit_value(raw, panel, fallback=""):
    """Read a panel-specific limit from `panel: value` lines or a scalar limit."""
    raw = str(raw or "").strip()
    if not raw:
        return fallback
    if "\n" not in raw and ": " not in raw:
        return raw
    for line in raw.replace("; ", "\n").splitlines():
        label, separator, value = line.partition(": ")
        if separator and label.strip() == str(panel):
            return value.strip()
    return fallback



def _format_panel_bounds(bounds, key):
    return "\n".join(f"{panel}: {limits[key]}" for panel, limits in bounds.items()
                     if limits.get(key, "") != "")



def _fit_inside_legends(fig, axes, font_size=12, ncol=1):
    """Single-pass inside legend layout; grow the canvas when necessary.

    Keep the existing axes geometry (in inches), increasing figure dimensions
    if the requested legend is larger than the axes. No iterative redraws.
    """
    ncol = max(1, int(ncol or 1))
    legends = []
    for ax in axes:
        old = ax.get_legend()
        if old is None:
            continue
        handles = list(old.legend_handles)
        labels = [item.get_text() for item in old.get_texts()]
        if not handles:
            continue
        title = old.get_title().get_text()
        loc = old._loc
        old.remove()
        leg = ax.legend(handles, labels, title=title, loc=loc,
                        ncol=min(ncol, len(labels)), fontsize=font_size,
                        title_fontsize=font_size, frameon=True, fancybox=True, framealpha=.85,
                        borderaxespad=.25, labelspacing=.2,
                        handletextpad=.3, columnspacing=.6,
                        handlelength=1.1)
        leg.set_in_layout(False)
        legends.append((ax, leg))
    if not legends:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Expand the physical canvas only when the legend needs more room.
    # Preserve axes dimensions in inches rather than stretching plot data.
    extra_w = extra_h = 0.0
    for ax, leg in legends:
        lb = leg.get_window_extent(renderer)
        ab = ax.get_window_extent(renderer)
        extra_w = max(extra_w, (lb.width - ab.width + 8) / fig.dpi)
        extra_h = max(extra_h, (lb.height - ab.height + 8) / fig.dpi)
    extra_w, extra_h = max(0., extra_w), max(0., extra_h)
    if extra_w or extra_h:
        old_w, old_h = fig.get_size_inches()
        # Preserve the original plot size in inches and allow it to grow into
        # the new canvas as well, instead of shrinking to accommodate legends.
        positions = [(ax, ax.get_position().frozen()) for ax in axes]
        fig.set_size_inches(old_w + extra_w, old_h + extra_h, forward=True)
        for ax, pos in positions:
            ax.set_position((pos.x0 * old_w / (old_w + extra_w),
                             pos.y0 * old_h / (old_h + extra_h),
                             pos.width * old_w / (old_w + extra_w) + extra_w / (old_w + extra_w),
                             pos.height * old_h / (old_h + extra_h) + extra_h / (old_h + extra_h)))


def _place_outside_legend(fig, axes, position="right", font_size=12, ncol=1):
    """Use exactly the requested columns and reserve their measured footprint.

    A single legend is shared across panels. Only one full canvas draw is used
    for measuring; no automatic column search or placement fallback.
    """
    seen = {}
    for ax in axes:
        old = ax.get_legend()
        if old is None:
            continue
        for handle, label in zip(old.legend_handles,
                                 [t.get_text() for t in old.get_texts()]):
            if label and label not in seen:
                seen[label] = handle
        old.set_visible(False)
        old.set_in_layout(False)
    if not seen:
        fig.tight_layout()
        return
    labels, handles = list(seen), list(seen.values())
    pos = position if position in ("right", "left", "top", "bottom") else "right"
    ncol = max(1, min(int(ncol or 1), len(labels)))
    legend = fig.legend(handles, labels, ncol=ncol, loc="center",
                        bbox_to_anchor=(.5, .5), bbox_transform=fig.transFigure,
                        fontsize=font_size, title="Legend", title_fontsize=font_size,
                        frameon=False, borderaxespad=0, labelspacing=.2,
                        handletextpad=.3, columnspacing=.6, handlelength=1.1)
    legend.set_in_layout(False)
    fig.canvas.draw()
    bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
    fw, fh = fig.canvas.get_width_height()
    # Minimal canvas padding and a narrow separation from the axes.
    # Do not charge the legend for a second outer margin on both sides.
    margin = max(2., fig.dpi * .025)
    gap = max(3., fig.dpi * .04)
    if pos in ("right", "left"):
        reserve = min(.95, (bbox.width + gap + margin) / fw)
        rect = (0, 0, 1-reserve, 1) if pos == "right" else (reserve, 0, 1, 1)
        fig.tight_layout(rect=rect, pad=.25, w_pad=.2, h_pad=.2)
        x = (1 - (bbox.width/2 + margin)/fw if pos == "right"
             else (bbox.width/2 + margin)/fw)
        legend.set_bbox_to_anchor((x, .5), transform=fig.transFigure)
    else:
        reserve = min(.95, (bbox.height + gap + margin) / fh)
        rect = (0, 0, 1, 1-reserve) if pos == "top" else (0, reserve, 1, 1)
        fig.tight_layout(rect=rect, pad=.25, w_pad=.2, h_pad=.2)
        y = (1 - (bbox.height/2 + margin)/fh if pos == "top"
             else (bbox.height/2 + margin)/fh)
        legend.set_bbox_to_anchor((.5, y), transform=fig.transFigure)
    legend.set_in_layout(False)
    if len(axes) > 1:
        # Keep the legend reservation, but restore V77-like breathing room
        # between facet axes.  This changes only the internal subplot gaps.
        fig.subplots_adjust(wspace=0.34, hspace=0.44)

def _matplotlib_direct_figure(spec, *, title="", x_label="", y_label="",
                              legend_labels="", trace_colors="", palette="Default",
                              x_min="", x_max="", y_min="", y_max="",
                              template="plotly", legend_position="right",
                              legend_inside=False, legend_columns=1, show_grid=True, font_size=12,
                              line_width=2, marker_size=7, line_alpha="", line_style="",
                              scatter_size="", scatter_marker="", scatter_alpha="", hist_alpha="",
                              box_showfliers=True, box_orientation="vertical",
                              title_font_size=14, axis_font_size=12, tick_font_size=10,
                              legend_font_size=12, panel_font_size=14, plot_dpi=100, bins=30, density=False,
                              stacked=False, histtype="bar", cumulative=0,
                              panel_by=None, independent_panel_limits=True, panel_titles="", _axes=None,
                              x_major_ticks=6, y_major_ticks=6, x_minor_ticks=False, y_minor_ticks=False):
    """Draw directly from the filtered/aggregated DataFrame, without Plotly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    if _axes is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=plot_dpi)
    else:
        ax = _axes
        fig = ax.figure
    if "error" in spec:
        ax.text(.5, .5, spec["error"], ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    df, kind, x, ys = spec["data"], spec["kind"], spec["x"], spec["ys"]
    # Facet only the native Matplotlib renderer.  Each panel receives the
    # same filtered/aggregated data subset and the same appearance settings.
    if panel_by and _axes is None:
        import math
        partitions = list(_panel_partitions(spec, panel_by))
        panel_title_map = _parse_label_map(panel_titles)
        if partitions:
            if len(partitions) > 36:
                ax.text(.5, .5, f"{len(partitions)} panels requested; filter the data to 36 or fewer panels.",
                        ha="center", va="center", transform=ax.transAxes, wrap=True)
                ax.set_axis_off()
                return fig
            plt.close(fig)
            ncols = (2 if len(partitions) == 4 else min(3, len(partitions)))
            nrows = math.ceil(len(partitions) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.8*nrows), dpi=plot_dpi,
                                     squeeze=False, sharex=False, sharey=False)
            for idx, (panel_title, panel_df, other_groups) in enumerate(partitions):
                panel_spec = {**spec, "data": panel_df, "groups": other_groups,
                              "_hide_ungrouped_legend": not bool(other_groups)}
                panel_ax = axes.flat[idx]
                _matplotlib_direct_figure(panel_spec, title=panel_title_map.get(panel_title, panel_title), x_label=x_label,
                    y_label=y_label, legend_labels=legend_labels, trace_colors=trace_colors,
                    palette=palette,
                    x_min=_panel_limit_value(x_min, panel_title), x_max=_panel_limit_value(x_max, panel_title),
                    y_min=_panel_limit_value(y_min, panel_title), y_max=_panel_limit_value(y_max, panel_title),
                    template=template, legend_position=legend_position,
                    legend_inside=legend_inside, legend_columns=legend_columns, show_grid=show_grid, font_size=font_size,
                    line_width=line_width, marker_size=marker_size, line_alpha=line_alpha, line_style=line_style,
                    scatter_size=scatter_size, scatter_marker=scatter_marker, scatter_alpha=scatter_alpha,
                    hist_alpha=hist_alpha, box_showfliers=box_showfliers, box_orientation=box_orientation,
                    title_font_size=panel_font_size, axis_font_size=axis_font_size,
                    tick_font_size=tick_font_size, legend_font_size=legend_font_size,
                    panel_font_size=panel_font_size, plot_dpi=plot_dpi,
                    bins=bins, density=density,
                    stacked=stacked, histtype=histtype, cumulative=cumulative, _axes=panel_ax,
                    x_major_ticks=x_major_ticks, y_major_ticks=y_major_ticks,
                    x_minor_ticks=x_minor_ticks, y_minor_ticks=y_minor_ticks)
            for extra in list(axes.flat)[len(partitions):]:
                extra.set_visible(False)
            if legend_inside:
                for panel_ax in list(axes.flat)[:len(partitions)]:
                    if panel_ax.get_legend() is not None:
                        panel_ax.get_legend().set_in_layout(False)
                fig.tight_layout()
                # V77 used tight_layout for facets; restore visible gutters
                # without changing V83's legend behaviour or column control.
                fig.subplots_adjust(wspace=0.34, hspace=0.44)
                _fit_inside_legends(fig, list(axes.flat)[:len(partitions)], legend_font_size, legend_columns)
            else:
                _place_outside_legend(fig, list(axes.flat)[:len(partitions)],
                                      legend_position, legend_font_size, legend_columns)
            return fig
    group_col = "__plot_group" if spec["groups"] and "__plot_group" in df.columns else None
    defaults = _matplotlib_defaults(spec, palette=palette)
    rename = _parse_label_map(legend_labels)
    overrides = _parse_label_map(trace_colors)
    alphas, styles, sizes, markers, marker_alphas = _series_style_maps(
        line_alpha, line_style, scatter_size, scatter_marker, scatter_alpha)
    histogram_alphas = _parse_label_map(hist_alpha)
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    labels = defaults["names"] if group_col else ([str(y) for y in ys] if len(ys) > 1 else ["All observations"])
    colors = {name: overrides.get(name, defaults["colors"].get(name, palette_values[i % len(palette_values)]))
              for i, name in enumerate(labels)}
    for name, color in list(colors.items()):
        if not mcolors.is_color_like(color):
            colors[name] = defaults["colors"].get(name, palette_values[0])

    def series():
        if group_col:
            for name in labels:
                yield name, df.loc[df[group_col] == name]
        else:
            yield "All observations", df

    if kind == "histogram":
        arrays = []
        for name, rows in series():
            vals = pd.to_numeric(rows[x], errors="coerce").to_numpy(dtype=float)
            arrays.append(vals[np.isfinite(vals)])
        all_vals = np.concatenate(arrays) if arrays else np.array([])
        if len(all_vals):
            edges = np.histogram_bin_edges(all_vals, bins=max(1, int(bins)))
            _, _, hist_patches = ax.hist(
                arrays, bins=edges, density=density,
                stacked=stacked or histtype == "barstacked",
                histtype=("bar" if histtype == "barstacked" else histtype),
                cumulative=cumulative if cumulative else False,
                color=[colors[n] for n in labels],
                label=[rename.get(n, n) for n in labels])
            # Matplotlib returns one patch container per group for bar and
            # one polygon per group for step/stepfilled.
            for name, group_patches in zip(labels, hist_patches):
                alpha = _bounded_float(histogram_alphas, name,
                                       1 if histtype == "step" else .65, 0, 1)
                if hasattr(group_patches, "set_alpha"):
                    group_patches.set_alpha(alpha)
                else:
                    for patch in group_patches:
                        patch.set_alpha(alpha)
        defaults["y_label"] = ("Cumulative probability" if cumulative and density else
                               "Cumulative count" if cumulative else
                               "Probability density" if density else "Count")
    elif kind == "scatter":
        for name, rows in series():
            color = colors.get(name, palette_values[0])
            ax.scatter(rows[x], rows[ys[0]],
                       s=_bounded_float(sizes, name, marker_size, 0.1, 100)**2,
                       marker=_scatter_marker(markers, name),
                       alpha=_bounded_float(marker_alphas, name, 1, 0, 1),
                       color=color, label=rename.get(name, name))
    elif kind == "line":
        line_df = df
        id_col = "identifier" if "identifier" in df.columns else None
        section_col = None
        if "__source_row_order" in df.columns:
            line_df = _add_source_sections(df, x_column=x)
            section_col = "__line_section"
        elif spec["frequency"] is not None and id_col is None:
            line_df = _add_gap_sections(df, x_column=x, group_columns=spec["groups"], frequency=spec["frequency"])
            section_col = "__line_section"
        for y in ys:
            if group_col:
                sets = [(n, line_df.loc[line_df[group_col] == n]) for n in labels]
            else:
                sets = [(str(y) if len(ys) > 1 else "All observations", line_df)]
            for name, rows in sets:
                segment_col = section_col or id_col
                parts = (rows.groupby(segment_col, dropna=False, sort=False) if segment_col else [(None, rows)])
                for _, part in parts:
                    part = part.sort_values(x)
                    ax.plot(part[x], part[y], color=colors.get(name, palette_values[0]),
                            linewidth=float(line_width),
                            linestyle=_line_style(styles, name),
                            alpha=_bounded_float(alphas, name, 1, 0, 1),
                            label=rename.get(name, name))
    elif kind in ("box", "bar"):
        # Build categorical groups in a single pass. The former implementation
        # repeatedly scanned the full DataFrame for every category × group,
        # which became prohibitively expensive for many identifiers.
        from collections import defaultdict

        category_values = df[x].astype(str)
        cats = list(pd.unique(category_values))
        cat_idx = {category: i for i, category in enumerate(cats)}
        group_names = labels if group_col else ["All observations"]
        group_idx = {name: i for i, name in enumerate(group_names)}
        n = max(1, len(group_names))
        # A boxplot creates many artists per category/group (whiskers, caps,
        # median, fliers). Thousands of combinations can stall the render
        # thread for minutes. Fail fast with an actionable message rather than
        # blocking the plot output indefinitely. No data is silently omitted.
        observed_pairs = df[[x, group_col]].drop_duplicates().shape[0] if group_col else len(cats)
        limit = 250 if kind == "box" else 1500
        if observed_pairs > limit:
            ax.text(.5, .5,
                    f"Too many category/group combinations ({observed_pairs:,}) for a {kind} plot.\n"
                    f"Use the filters, reduce Group by cardinality, or aggregate the data\n"
                    f"(rendering limit: {limit:,} combinations).",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, wrap=True)
            ax.set_axis_off()
            return fig
        width = .82 / n
        values = pd.to_numeric(df[ys[0]], errors="coerce")
        valid = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
        plot_groups = df[group_col].astype(str) if group_col else pd.Series("All observations", index=df.index)
        # Use integer group indices, avoiding repeated string comparisons.
        ci = category_values.map(cat_idx).to_numpy()
        gi = plot_groups.map(group_idx).to_numpy()
        valid_array = valid.to_numpy() & pd.notna(gi)
        positions = ci[valid_array] + (gi[valid_array] - (n - 1) / 2) * width
        numeric = values.to_numpy(dtype=float, na_value=np.nan)[valid_array]
        names = gi[valid_array].astype(int)
        if kind == "bar":
            # One bar artist collection instead of one call per category/group.
            totals = defaultdict(float)
            for position, value in zip(positions, numeric):
                totals[float(position)] += float(value)
            xpos = list(totals)
            # Derive group colour from the position's group index.
            palette_for_positions = {}
            for category_index, group_index, position in zip(ci[valid_array], names, positions):
                palette_for_positions[float(position)] = colors.get(group_names[group_index], palette_values[0])
            if xpos:
                ax.bar(xpos, [totals[pos] for pos in xpos], width=width * .92,
                       color=[palette_for_positions[pos] for pos in xpos])
        else:
            # Calculate group memberships once and make one boxplot call.
            buckets = defaultdict(list)
            bucket_colors = {}
            for position, value, group_index in zip(positions, numeric, names):
                key = float(position)
                buckets[key].append(float(value))
                bucket_colors[key] = colors.get(group_names[group_index], palette_values[0])
            if buckets:
                box_positions = sorted(buckets)
                bp = ax.boxplot([buckets[pos] for pos in box_positions],
                                positions=box_positions, widths=width * .92,
                                manage_ticks=False, patch_artist=True,
                                showfliers=bool(box_showfliers),
                                vert=(box_orientation != "horizontal"))
                for patch, position in zip(bp["boxes"], box_positions):
                    patch.set_facecolor(bucket_colors[position])
                    patch.set_alpha(.7)
                if not box_showfliers:
                    # Autoscaling normally includes every observation in the
                    # data; instead use only the whisker endpoints.
                    whisker_values = np.concatenate([
                        line.get_xdata() if box_orientation == "horizontal" else line.get_ydata()
                        for line in bp["whiskers"]
                    ])
                    whisker_values = np.asarray(whisker_values, dtype=float)
                    whisker_values = whisker_values[np.isfinite(whisker_values)]
                    if whisker_values.size:
                        low, high = float(whisker_values.min()), float(whisker_values.max())
                        margin = .05 * (high - low) if high > low else max(abs(low) * .05, .5)
                        if box_orientation == "horizontal":
                            ax.set_xlim(low - margin, high + margin)
                        else:
                            ax.set_ylim(low - margin, high + margin)
        if kind == "box" and box_orientation == "horizontal":
            ax.set_yticks(range(len(cats)), cats)
            ax.set_ylim(-.5, max(.5, len(cats) - .5))
        else:
            ax.set_xticks(range(len(cats)), cats)
            ax.set_xlim(-.5, max(.5, len(cats) - .5))
        # A boxplot with a single ungrouped series does not need a legend.
        # Keep legends for genuinely grouped boxplots and for bar plots.
        show_category_legend = not (
            kind == "box" and len(group_names) == 1
            and group_names[0] == "All observations"
        )
        if len(group_names) <= 40 and show_category_legend and not spec.get("_hide_ungrouped_legend", not bool(spec["groups"])):
            ax.legend(handles=[Patch(facecolor=colors[name], label=rename.get(name, name))
                               for name in group_names], title="Legend")
    else:
        raise ValueError(f"Unknown plot type: {kind}")

    if kind not in ("box", "bar"):
        handles, texts = ax.get_legend_handles_labels()
        unique = dict(zip(texts, handles))
        if unique and not spec.get("_hide_ungrouped_legend", not bool(spec["groups"])):
            ax.legend(unique.values(), unique.keys(), title="Legend")
    legend = ax.get_legend()
    if legend is not None:
        locs = {"right":"upper right", "left":"upper left", "top":"upper center", "bottom":"lower center"}
        if legend_inside:
            legend.set_loc(locs.get(legend_position, "upper right"))
            legend.set_bbox_to_anchor(None)
        else:
            anchors = {"right":(1.01,1), "left":(-.01,1), "top":(.5,1.15), "bottom":(.5,-.2)}
            legend.set_loc(locs.get(legend_position, "upper right"))
            legend.set_bbox_to_anchor(anchors.get(legend_position, (1.01,1)))
    if template == "plotly_dark":
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#222222")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    elif template == "ggplot2":
        ax.set_facecolor("#EBEBEB")
    elif template in ("plotly_white", "simple_white"):
        ax.set_facecolor("white")
    xlabel = x_label.strip() or defaults["x_label"]
    ylabel = y_label.strip() or defaults["y_label"]
    if kind == "box" and box_orientation == "horizontal":
        xlabel, ylabel = ylabel, xlabel
    ax.set_xlabel(xlabel, fontsize=axis_font_size)
    ax.set_ylabel(ylabel, fontsize=axis_font_size)
    ax.set_title(title, fontsize=title_font_size)
    ax.tick_params(labelsize=tick_font_size)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_font_size)
        legend.get_title().set_fontsize(legend_font_size)
    ax.grid(bool(show_grid))
    _matplotlib_tick_controls(ax, kind=kind, box_orientation=box_orientation,
                              show_grid=show_grid, x_major_ticks=x_major_ticks,
                              y_major_ticks=y_major_ticks, x_minor_ticks=x_minor_ticks,
                              y_minor_ticks=y_minor_ticks)
    for lo, hi, getter, setter in ((x_min, x_max, ax.get_xlim, ax.set_xlim),
                                   (y_min, y_max, ax.get_ylim, ax.set_ylim)):
        try:
            from matplotlib.dates import date2num
            is_datetime_x = getter == ax.get_xlim and pd.api.types.is_datetime64_any_dtype(df[x])
            lower = _parse_axis_limit(lo, datetime_axis=is_datetime_x)
            upper = _parse_axis_limit(hi, datetime_axis=is_datetime_x)
            if is_datetime_x:
                lower = date2num(lower.to_pydatetime()) if lower is not None else None
                upper = date2num(upper.to_pydatetime()) if upper is not None else None
            if lower is not None or upper is not None:
                current = getter()
                setter(lower if lower is not None else current[0], upper if upper is not None else current[1])
        except (TypeError, ValueError):
            pass
    if _axes is None:
        if legend_inside:
            if ax.get_legend() is not None:
                ax.get_legend().set_in_layout(False)
            fig.tight_layout()
            _fit_inside_legends(fig, [ax], legend_font_size, legend_columns)
        else:
            _place_outside_legend(fig, [ax], legend_position, legend_font_size, legend_columns)
    return fig



def _matplotlib_figure(plotly_figure):
    """Render the shared Plotly trace specification with Matplotlib (Agg).

    The Shiny controls, data selection, aggregation, and appearance settings
    are shared between backends. No browser-side Plotly widget is created.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 5), dpi=plot_dpi)
    source = plotly_figure
    category_labels = list(getattr(source.layout.xaxis, "categoryarray", None) or [])
    categorical = (getattr(source.layout.xaxis, "type", None) == "category")
    category_map = {str(value): i for i, value in enumerate(category_labels)}
    legend_seen = set()
    legend_handles = []
    legend_texts = []

    def xvalues(values):
        vals = list(values) if values is not None else []
        if categorical:
            for val in vals:
                if str(val) not in category_map:
                    category_map[str(val)] = len(category_map)
            return [category_map[str(val)] for val in vals]
        return vals

    traces = list(source.data)
    box_traces = [t for t in traces if t.type == "box"]
    for index, trace in enumerate(traces):
        name = str(trace.name or "")
        color = (getattr(getattr(trace, "line", None), "color", None)
                 or getattr(getattr(trace, "marker", None), "color", None)
                 or f"C{index % 10}")
        if not isinstance(color, str) or not mcolors.is_color_like(color):
            color = f"C{index % 10}"
        label = name if name and name not in legend_seen and trace.showlegend is not False else "_nolegend_"
        if label != "_nolegend_":
            legend_seen.add(name)
        if trace.type in ("scatter", "scattergl"):
            xs = xvalues(trace.x)
            ys = list(trace.y)
            mode = str(trace.mode or "lines")
            if "lines" in mode:
                ax.plot(xs, ys, color=color, label=label,
                        linewidth=getattr(trace.line, "width", None) or 2,
                        marker="o" if "markers" in mode else None,
                        markersize=getattr(trace.marker, "size", None) or 5)
            elif "markers" in mode:
                ax.scatter(xs, ys, color=color, label=label,
                           s=(getattr(trace.marker, "size", None) or 7) ** 2)
        elif trace.type == "box":
            # Position boxes at category centres, distributing grouped boxes
            # within each category to avoid excessive whitespace.
            xs = xvalues(trace.x)
            ys = np.asarray(trace.y, dtype=float)
            categories = list(dict.fromkeys(xs))
            box_index = box_traces.index(trace)
            width = 0.8 / max(len(box_traces), 1)
            offset = (box_index - (len(box_traces) - 1) / 2) * width
            groups = [ys[np.asarray(xs) == cat] for cat in categories]
            groups = [g[np.isfinite(g)] for g in groups]
            valid = [(cat, group) for cat, group in zip(categories, groups) if len(group)]
            if valid:
                result = ax.boxplot([group for _, group in valid],
                                    positions=[cat + offset for cat, _ in valid],
                                    widths=width * .92, patch_artist=True,
                                    manage_ticks=False, showfliers=True)
                for box in result["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(.65)
                if label != "_nolegend_":
                    legend_handles.append(Patch(facecolor=color, alpha=.65))
                    legend_texts.append(label)
        elif trace.type == "bar":
            xs = xvalues(trace.x)
            if not categorical and xs and isinstance(xs[0], str):
                categorical = True
                xs = xvalues(trace.x)
            width = .8 / max(sum(t.type == "bar" for t in traces), 1)
            offset = (index - (len(traces) - 1) / 2) * width
            ax.bar(np.asarray(xs, dtype=float) + offset, trace.y,
                   width=width * .95, color=color, label=label)
        elif trace.type == "histogram":
            xs = np.asarray(trace.x, dtype=float)
            xs = xs[np.isfinite(xs)]
            nbins = getattr(trace, "nbinsx", None) or 30
            histnorm = getattr(trace, "histnorm", None)
            cumulative = getattr(getattr(trace, "cumulative", None), "enabled", False)
            direction = getattr(getattr(trace, "cumulative", None), "direction", "increasing")
            histtype = "step" if getattr(trace, "visible", True) == "legendonly" else "bar"
            ax.hist(xs, bins=nbins, density=histnorm in ("probability density", "density"),
                    cumulative=(-1 if direction == "decreasing" else True) if cumulative else False,
                    histtype=histtype, alpha=.65, color=color, label=label)

    if category_map:
        ax.set_xticks(list(category_map.values()), list(category_map.keys()))
    ax.set_xlabel(getattr(getattr(source.layout.xaxis, "title", None), "text", "") or "")
    ax.set_ylabel(getattr(getattr(source.layout.yaxis, "title", None), "text", "") or "")
    ax.set_title(getattr(getattr(source.layout, "title", None), "text", "") or "")
    ax.grid(bool(getattr(source.layout.xaxis, "showgrid", True)), axis="x")
    ax.grid(bool(getattr(source.layout.yaxis, "showgrid", True)), axis="y")
    for axis, setter in ((source.layout.xaxis, ax.set_xlim), (source.layout.yaxis, ax.set_ylim)):
        bounds = getattr(axis, "range", None)
        if bounds and all(v is not None for v in bounds):
            try:
                setter(*bounds)
            except (ValueError, TypeError):
                pass
    handles, labels = ax.get_legend_handles_labels()
    handles += legend_handles
    labels += legend_texts
    if handles:
        legend = source.layout.legend
        orientation = getattr(legend, "orientation", "v")
        inside = (0 <= float(getattr(legend, "x", 1.02)) <= 1 and
                  0 <= float(getattr(legend, "y", 1)) <= 1)
        loc = "upper right" if inside else "upper left"
        kwargs = {"loc": loc, "title": "Legend", "ncol": max(1, len(labels)) if orientation == "h" else 1}
        if not inside:
            kwargs["bbox_to_anchor"] = (1.01, 1)
        ax.legend(handles, labels, **kwargs)
    fig.tight_layout()
    return fig

