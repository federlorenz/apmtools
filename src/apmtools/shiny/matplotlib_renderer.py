"""Native and converted Matplotlib figures and panel layout."""
from __future__ import annotations
from .profiling import traced
import threading
from datetime import date, datetime, time, timedelta
from collections.abc import Sequence
import numpy as np
import pandas as pd
from ..units import upas_units, lascar_units, purple_units

from .config import VARIABLE_UNITS, PLOT_TYPES, FREQUENCIES, AGGREGATES, _MPL_PALETTES
from .text_escapes import decode_plot_text
from .appearance import _axis_label, _bounded_float, _line_style, _matplotlib_tick_controls, _parse_axis_limit, _parse_label_map, _scatter_marker, _series_style_maps
from .histogram_stats import histogram_heights, histogram_bin_edges, histogram_kde, effective_histogram_bin_mode
from .data import _add_gap_sections, _add_source_sections, _composite_labels
from .line_bands import calculate_line_band


def _matplotlib_defaults(spec, *, palette="Default"):
    """Derive labels and series colours directly from prepared pandas data."""
    kind, df = spec["kind"], spec["data"]
    x, ys, groups = spec["x"], spec["ys"], spec["groups"]
    names = ([str(v) for v in pd.unique(df["__hist_series"])]
             if kind == "histogram" and spec.get("hist_variables") and "__hist_series" in df
             else [str(v) for v in pd.unique(df["__plot_group"])]
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
               "\n".join(_axis_label(y) for y in ys) if len(ys) > 1 else
               _axis_label(ys[0]) if ys else "Count")
    if kind == "scatter" and len(spec.get("x_variables", [])) > 1:
        x_label = "\n".join(_axis_label(c) for c in spec["x_variables"])
    x_label = spec.get("x_label_override") or x_label
    return dict(x_label=x_label, y_label=y_label, names=names, colors=colors)



def _axis_bounds_from_spec(spec, *, bins=30, density=False, stacked=False, cumulative=0,
                           hist_stat=None, hist_arrangement=None, hist_common_bins=True,
                           hist_common_norm=False, hist_edges=None,
                           hist_norm_total=None, bar_stacking=(), hist_bin_mode='count',
                           hist_bin_width=1.0, hist_range_min=None, hist_range_max=None,
                           hist_kde=False, hist_kde_bw=1.0):
    """Compute editable axis bounds from the prepared data, without rendering.

    Datetime X bounds are ISO timestamps; categorical X bounds are intentionally
    omitted. Values are calculated only when a valid plot specification exists.
    """
    if "error" in spec:
        return {key: "" for key in ("x_min", "x_max", "y_min", "y_max")}
    df, kind, x, ys = spec["data"], spec["kind"], spec["x"], spec["ys"]
    # A long-form Y grouping must render its value column, not a source
    # column removed by melt(). Normalize the local spec before faceting.
    if kind == "scatter" and "__axis_y_value" in df.columns and any(y not in df.columns for y in ys):
        ys = ["__axis_y_value"]
        spec = {**spec, "ys": ys}

    # Reactive selection changes can briefly expose a specification whose
    # original Y names are no longer present after a melt/reshape.
    ys = [y for y in ys if y in df.columns]
    if x not in df.columns or (kind != "histogram" and not ys):
        return {key: "" for key in ("x_min", "x_max", "y_min", "y_max")}
    if df.empty:
        return {key: "" for key in ("x_min", "x_max", "y_min", "y_max")}
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
        group_col = ("__hist_series" if spec.get("hist_variables") and "__hist_series" in df
                     else "__plot_group" if spec["groups"] and "__plot_group" in df else None)
        parts = ([part for _, part in df.groupby(group_col, sort=False)]
                 if group_col else [df])
        arrays = [pd.to_numeric(part[x], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                  for part in parts]
        arrangement = hist_arrangement or ("stack" if stacked else "layer")
        edges, heights = histogram_heights(arrays, bins=bins,
            stat=hist_stat or ("density" if density else "count"), arrangement=arrangement,
            common_bins=hist_common_bins, common_norm=hist_common_norm, cumulative=cumulative,
            common_edges=hist_edges, norm_total=hist_norm_total,
            bin_mode=hist_bin_mode, bin_width=hist_bin_width,
            range_min=hist_range_min, range_max=hist_range_max)
        if heights:
            bounds["x_min"] = format(float(min(e[0] for e in edges)), ".12g")
            bounds["x_max"] = format(float(max(e[-1] for e in edges)), ".12g")
            ymax = (np.stack(heights).sum(axis=0).max()
                    if arrangement in ("stack", "fill") else max(h.max() for h in heights))
            if hist_kde and hist_bin_mode != "discrete" and arrangement == "layer" and not cumulative:
                pooled_total = hist_norm_total if hist_common_norm and hist_norm_total is not None else sum(map(len, arrays))
                for group, edge in zip(arrays, edges):
                    finite = group[np.isfinite(group)]
                    if len(finite) < 2:
                        continue
                    grid = np.linspace(edge[0], edge[-1], 256)
                    curve = histogram_kde(finite, grid, bandwidth=hist_kde_bw)
                    scale = float(np.median(np.diff(edge)))
                    denominator = pooled_total if hist_common_norm else len(finite)
                    if hist_stat == "count":
                        curve *= len(finite) * scale
                    elif hist_stat == "frequency":
                        curve *= len(finite)
                    elif hist_stat == "density":
                        curve *= len(finite) / denominator if denominator else 0
                    elif hist_stat == "probability":
                        curve *= scale * len(finite) / denominator if denominator else 0
                    elif hist_stat == "percent":
                        curve *= scale * 100 * len(finite) / denominator if denominator else 0
                    ymax = max(float(ymax), float(curve.max()))
            bounds["y_min"], bounds["y_max"] = "0", format(float(ymax), ".12g")
    elif kind == "bar" and bar_stacking:
        from .bar_stacking import stacked_bar_segments
        segments = stacked_bar_segments(spec, bar_stacking)
        if segments:
            from collections import defaultdict
            totals = defaultdict(float)
            for category, series, stack, value in segments:
                totals[(category, series, value >= 0)] += value
            bounds["y_min"] = format(float(min(0, *totals.values())), ".12g")
            bounds["y_max"] = format(float(max(0, *totals.values())), ".12g")
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
        if spec.get("multi_axis") and spec["groups"] and "__plot_group" in part:
            part = part.copy()
            part["__plot_global_group"] = part["__plot_group"]
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
    # Preserve first-seen order while avoiding a linear scan for every
    # repeated trace name across panels (particularly with many facets).
    names = []
    seen_names = set()
    for _, part, remaining in _panel_partitions(spec, panel_by):
        panel_spec = {**spec, "data": part, "groups": remaining}
        for name in _matplotlib_defaults(panel_spec, palette=palette)["names"]:
            if name not in seen_names:
                seen_names.add(name)
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
    scope = kwargs.pop("hist_panel_norm_scope", "global")
    if scope not in ("global", "panel"):
        raise ValueError(f"Invalid histogram panel normalization scope: {scope}")
    if spec.get("kind") == "histogram":
        global_values = pd.to_numeric(spec["data"][spec["x"]], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan)
        global_values = global_values[np.isfinite(global_values)]
        if len(global_values):
            kwargs = dict(kwargs)
            if (kwargs.get("hist_common_bins", True) or
                    kwargs.get("hist_arrangement", "layer") != "layer"):
                kwargs["hist_edges"] = histogram_bin_edges(global_values, bins=kwargs.get("bins", 30),
                    bin_mode=effective_histogram_bin_mode(global_values, kwargs.get("hist_bin_mode", "count"),
                        range_min=kwargs.get("hist_range_min"), range_max=kwargs.get("hist_range_max")),
                    bin_width=kwargs.get("hist_bin_width", 1.0),
                    range_min=kwargs.get("hist_range_min"),
                    range_max=kwargs.get("hist_range_max"))
            if kwargs.get("hist_common_norm", False) and scope == "global":
                in_domain = global_values
                if kwargs.get("hist_range_min") not in (None, ''):
                    in_domain = in_domain[in_domain >= float(kwargs["hist_range_min"])]
                if kwargs.get("hist_range_max") not in (None, ''):
                    in_domain = in_domain[in_domain <= float(kwargs["hist_range_max"])]
                kwargs["hist_norm_total"] = len(in_domain)
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


def _place_outside_legend(fig, axes, position="right", font_size=12, ncol=1,
                          panel_hspace=0.90, panel_wspace=0.38):
    """Use exactly the requested columns and reserve their measured footprint.

    A single legend is shared across panels. Only one full canvas draw is used
    for measuring; no automatic column search or placement fallback.
    """
    # Entry identity is positional within a panel, not its editable label.
    # This retains two distinct handles renamed to the same displayed text,
    # while collapsing corresponding entries repeated across panels.
    seen = {}
    for ax in axes:
        old = ax.get_legend()
        if old is None:
            continue
        identities = getattr(old, "_apmtools_stack_keys", None)
        for index, (handle, label) in enumerate(zip(old.legend_handles,
                                 [t.get_text() for t in old.get_texts()])):
            # Stacked bar series may differ in number and order across facets.
            # Use their unedited group-combination identity rather than position.
            key = ("stack", identities[index]) if identities is not None else ("position", index)
            if label and key not in seen:
                seen[key] = (handle, label)
        old.set_visible(False)
        old.set_in_layout(False)
    if not seen:
        fig.tight_layout()
        return
    handles = [entry[0] for entry in seen.values()]
    labels = [entry[1] for entry in seen.values()]
    pos = position if position in ("right", "left", "top", "bottom") else "right"
    ncol = max(1, min(int(ncol or 1), len(labels)))
    legend = fig.legend(handles, labels, ncol=ncol, loc="center",
                        bbox_to_anchor=(.5, .5), bbox_transform=fig.transFigure,
                        fontsize=font_size, title=None, title_fontsize=font_size,
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
        # Keep the legend reservation, but reserve more breathing room
        # between facet axes.  This changes only the internal subplot gaps.
        fig.subplots_adjust(wspace=panel_wspace, hspace=panel_hspace)

@traced('_matplotlib_direct_figure', kind="data" if 'shiny/matplotlib_renderer.py' not in ("plot_configs.py", "shiny/matplotlib_renderer.py") else "service")
def _matplotlib_direct_figure(spec, *, title="", x_label="", y_label="",
                              legend_labels="", trace_colors="", palette="Default",
                              x_min="", x_max="", y_min="", y_max="",
                              template="plotly", legend_position="right",
                              legend_inside=False, legend_columns=1, multi_y_color_by="group", show_grid=True, font_size=12,
                              line_width=2, marker_size=7, line_alpha="", line_style="", reference_lines=None,
                              error_bands=False, error_band_method="ci", error_band_confidence=95,
                              error_band_alpha=0.2, error_band_lower="", error_band_upper="",
                              scatter_size="", scatter_marker="", scatter_alpha="", hist_alpha="",
                              box_showfliers=True, box_orientation="vertical", bar_box_width=1.0, bar_stacking=None,
                              title_font_size=14, axis_font_size=12, tick_font_size=10,
                              legend_font_size=12, panel_font_size=14, plot_dpi=100,
                              plot_height_pct=100, bins=30, density=False,
                              stacked=False, histtype="bar", cumulative=0,
                              hist_stat=None, hist_arrangement=None, hist_appearance=None,
                              hist_common_bins=True, hist_common_norm=False,
                              hist_panel_norm_scope="global", hist_bin_mode='count', hist_bin_width=1.0,
                              hist_range_min=None, hist_range_max=None,
                              hist_discrete=False, hist_kde=False, hist_kde_bw=1.0,
                              panel_by=None, independent_panel_limits=True, panel_titles="",
                              panel_hspace=0.90, panel_wspace=0.38, _axes=None,
                              _hist_shared_edges=None, _hist_norm_total=None,
                              x_major_ticks=6, y_major_ticks=6, x_minor_ticks=False, y_minor_ticks=False):
    """Draw directly from the filtered/aggregated DataFrame, without Plotly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    if _axes is None:
        fig, ax = plt.subplots(figsize=(10, 5 * float(plot_height_pct) / 100), dpi=plot_dpi)
    else:
        ax = _axes
        fig = ax.figure
    if "error" in spec:
        ax.text(.5, .5, spec["error"], ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    df, kind, x, ys = spec["data"], spec["kind"], spec["x"], spec["ys"]
    if hist_panel_norm_scope not in ("global", "panel"):
        raise ValueError(f"Invalid histogram panel normalization scope: {hist_panel_norm_scope}")
    # A long-form Y grouping must render its value column, not a source
    # column removed by melt(). Normalize the local spec before faceting.
    if kind == "scatter" and "__axis_y_value" in df.columns and any(y not in df.columns for y in ys):
        ys = ["__axis_y_value"]
        spec = {**spec, "ys": ys}

    # Resolve an automatic datetime intercept ONCE, from the full dataset,
    # before splitting into panels; otherwise each facet gets a different
    # intercept date and identical user settings yield different lines.
    if (kind == "line" and reference_lines and x in df.columns
            and pd.api.types.is_datetime64_any_dtype(df[x]) and _axes is None):
        from .reference_lines import parse_reference_lines
        reference_lines = parse_reference_lines(reference_lines)
        origin_candidates = pd.to_datetime(df[x], errors="coerce").dropna()
        if not origin_candidates.empty:
            default_origin = origin_candidates.min().isoformat()
            reference_lines = [dict(line, origin=line.get("origin") or default_origin)
                               for line in reference_lines]

    # Facet only the native Matplotlib renderer.  Each panel receives the
    # same filtered/aggregated data subset and the same appearance settings.
    if panel_by and _axes is None:
        # A dimension cannot simultaneously define facets and stack segments.
        from .bar_stacking import selected_stack_columns
        blocked = selected_stack_columns(spec.get("groups", ()), bar_stacking) if kind == "bar" else []
        panel_by = [name for name in ([panel_by] if isinstance(panel_by, str) else panel_by)
                    if name not in blocked]
    if panel_by and _axes is None:
        import math
        from .bar_stacking import stacked_bar_segments
        panel_groups = [panel_by] if isinstance(panel_by, str) else list(panel_by)
        palette_spec = {**spec, "groups": [g for g in spec["groups"] if g not in panel_groups]}
        from .bar_stacking import stacked_bar_legend_key
        full_stack_palette = (list(dict.fromkeys(
            stacked_bar_legend_key(item[1], item[2])
            for item in stacked_bar_segments(palette_spec, bar_stacking)))
                              if kind == "bar" and bar_stacking else [])
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
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.4*ncols, 4.7*nrows * float(plot_height_pct) / 100), dpi=plot_dpi,
                                     squeeze=False, sharex=False, sharey=False)
            # Common bins and normalization span all facets, not only the
            # subset of observations shown within an individual panel.
            if kind == "histogram":
                global_values = pd.to_numeric(df[spec["x"]], errors="coerce").to_numpy(
                    dtype=float, na_value=np.nan)
                global_values = global_values[np.isfinite(global_values)]
                if len(global_values):
                    if hist_common_bins or (hist_arrangement or ("stack" if stacked else "layer")) != "layer":
                        _hist_shared_edges = histogram_bin_edges(global_values, bins=bins,
                            bin_mode=effective_histogram_bin_mode(global_values,
                                "discrete" if hist_discrete else hist_bin_mode,
                                range_min=hist_range_min, range_max=hist_range_max),
                            bin_width=hist_bin_width, range_min=hist_range_min,
                            range_max=hist_range_max)
                    in_domain = global_values
                    if hist_range_min not in (None, ''):
                        in_domain = in_domain[in_domain >= float(hist_range_min)]
                    if hist_range_max not in (None, ''):
                        in_domain = in_domain[in_domain <= float(hist_range_max)]
                    _hist_norm_total = len(in_domain) if hist_panel_norm_scope == "global" else None
            for idx, (panel_title, panel_df, other_groups) in enumerate(partitions):
                panel_spec = {**spec, "data": panel_df, "groups": other_groups,
                              "_original_groups": spec["groups"],
                              "_stack_palette": full_stack_palette,
                              "_hide_ungrouped_legend": not bool(other_groups) and not bool(spec.get("hist_variables"))}
                panel_ax = axes.flat[idx]
                _matplotlib_direct_figure(panel_spec, title=panel_title_map.get(panel_title, panel_title), x_label=x_label,
                    y_label=y_label, legend_labels=legend_labels, trace_colors=trace_colors,
                    palette=palette,
                    x_min=_panel_limit_value(x_min, panel_title), x_max=_panel_limit_value(x_max, panel_title),
                    y_min=_panel_limit_value(y_min, panel_title), y_max=_panel_limit_value(y_max, panel_title),
                    template=template, legend_position=legend_position,
                    legend_inside=legend_inside, legend_columns=legend_columns, multi_y_color_by=multi_y_color_by, show_grid=show_grid, font_size=font_size,
                    line_width=line_width, marker_size=marker_size, line_alpha=line_alpha, line_style=line_style, reference_lines=reference_lines,
                    error_bands=error_bands, error_band_method=error_band_method,
                    error_band_confidence=error_band_confidence, error_band_alpha=error_band_alpha,
                    error_band_lower=error_band_lower, error_band_upper=error_band_upper,
                    scatter_size=scatter_size, scatter_marker=scatter_marker, scatter_alpha=scatter_alpha,
                    hist_alpha=hist_alpha, box_showfliers=box_showfliers, box_orientation=box_orientation,
                    bar_box_width=bar_box_width, bar_stacking=bar_stacking,
                    title_font_size=panel_font_size, axis_font_size=axis_font_size,
                    tick_font_size=tick_font_size, legend_font_size=legend_font_size,
                    panel_font_size=panel_font_size, plot_dpi=plot_dpi,
                    plot_height_pct=plot_height_pct,
                    bins=bins, density=density,
                    stacked=stacked, histtype=histtype, cumulative=cumulative,
                    hist_stat=hist_stat, hist_arrangement=hist_arrangement,
                    hist_appearance=hist_appearance, hist_common_bins=hist_common_bins,
                    hist_common_norm=hist_common_norm,
                    hist_panel_norm_scope=hist_panel_norm_scope,
                    hist_bin_mode=hist_bin_mode, hist_bin_width=hist_bin_width,
                    hist_range_min=hist_range_min, hist_range_max=hist_range_max,
                    hist_discrete=hist_discrete, hist_kde=hist_kde, hist_kde_bw=hist_kde_bw,
                    _axes=panel_ax,
                    _hist_shared_edges=_hist_shared_edges, _hist_norm_total=_hist_norm_total,
                    x_major_ticks=x_major_ticks, y_major_ticks=y_major_ticks,
                    x_minor_ticks=x_minor_ticks, y_minor_ticks=y_minor_ticks)
            # Resolve the axes once; reuse the same ordered panel collection for
            # legend layout rather than repeatedly materializing axes.flat.
            flat_axes = axes.flat
            active_axes = [flat_axes[i] for i in range(len(partitions))]
            for i in range(len(partitions), axes.size):
                flat_axes[i].set_visible(False)
            if kind == "bar" and bar_stacking and not independent_panel_limits:
                # Matplotlib independently autoscales each facet because their
                # axes are intentionally created with sharey=False.  Apply the
                # common stacked range *after* drawing all facets, so the
                # checkbox actually determines whether scales are shared.
                # Individual manual panel bounds still take precedence.
                lows, highs = zip(*(panel_ax.get_ylim() for panel_ax in active_axes))
                common_low, common_high = min(lows), max(highs)
                for (panel_title, _, _), panel_ax in zip(partitions, active_axes):
                    lower = _parse_axis_limit(_panel_limit_value(y_min, panel_title))
                    upper = _parse_axis_limit(_panel_limit_value(y_max, panel_title))
                    panel_ax.set_ylim(
                        common_low if lower is None else lower,
                        common_high if upper is None else upper,
                    )
            if legend_inside:
                for panel_ax in active_axes:
                    if panel_ax.get_legend() is not None:
                        panel_ax.get_legend().set_in_layout(False)
                fig.tight_layout()
                # V77 used tight_layout for facets; reserve generous gutters
                # without changing V83's legend behaviour or column control.
                fig.subplots_adjust(wspace=panel_wspace, hspace=panel_hspace)
                _fit_inside_legends(fig, active_axes, legend_font_size, legend_columns)
            else:
                _place_outside_legend(fig, active_axes,
                                      legend_position, legend_font_size, legend_columns,
                                      panel_hspace=panel_hspace, panel_wspace=panel_wspace)
            return fig
    group_col = ("__hist_series" if kind == "histogram" and spec.get("hist_variables")
                 else "__plot_group" if spec["groups"] and "__plot_group" in df.columns else None)
    defaults = _matplotlib_defaults(spec, palette=palette)
    from .legend_identity import canonical_legend_map, legend_label
    aliases = spec.get("legend_aliases", {})
    rename = canonical_legend_map(_parse_label_map(legend_labels), aliases=aliases)
    overrides = _parse_label_map(trace_colors)
    alphas, styles, sizes, markers, marker_alphas = _series_style_maps(
        line_alpha, line_style, scatter_size, scatter_marker, scatter_alpha)
    histogram_alphas = _parse_label_map(hist_alpha)
    palette_values = _MPL_PALETTES.get(palette, _MPL_PALETTES["Default"])
    labels = defaults["names"] if group_col else ([str(y) for y in ys] if len(ys) > 1 else ["All observations"])
    colors = {name: overrides.get(name, defaults["colors"].get(name, palette_values[i % len(palette_values)]))
              for i, name in enumerate(labels)}
    multi_y = kind == "line" and len(ys) > 1
    if multi_y:
        from .multi_y import multi_y_styles
        multi_colors, multi_dashes, variable_names, group_names = multi_y_styles(
            ys, labels if group_col else [], palette_values, multi_y_color_by,
            overrides, styles, aliases=aliases)
    for name, color in list(colors.items()):
        if not mcolors.is_color_like(color):
            colors[name] = defaults["colors"].get(name, palette_values[0])

    def series():
        if group_col:
            for name in labels:
                yield name, df.loc[df[group_col].astype(str) == str(name)]
        else:
            yield "All observations", df

    if kind == "bar" and bar_stacking and any(
            col in df.columns and col in spec["groups"] for col in bar_stacking):
        from .bar_stacking import stacked_bar_segments, stacked_bar_legend_key
        segments = stacked_bar_segments(spec, bar_stacking)
        cats = list(dict.fromkeys(str(v) for v in df[x]))
        series_keys = list(dict.fromkeys(item[1] for item in segments))
        # Colour and legend identities must include non-stacking grouping
        # dimensions too; they determine adjacent bars within each category.
        legend_keys = list(dict.fromkeys(
            stacked_bar_legend_key(item[1], item[2]) for item in segments))
        cat_idx = {cat: i for i, cat in enumerate(cats)}
        series_idx = {item: i for i, item in enumerate(series_keys)}
        bar_width = .82 / max(len(series_keys), 1)
        stacks_positive, stacks_negative = {}, {}
        palette_keys = spec.get("_stack_palette") or legend_keys
        stack_colors = {name: overrides.get(name,
                           palette_values[palette_keys.index(name) % len(palette_values)]
                           if name in palette_keys else palette_values[0])
                        for name in legend_keys}
        for (category, series_key, stack_key, amount) in segments:
            legend_key = stacked_bar_legend_key(series_key, stack_key)
            slot = (category, series_key)
            base_map = stacks_positive if amount >= 0 else stacks_negative
            base = base_map.get(slot, 0.0)
            pos = cat_idx[category] + (series_idx[series_key] - (len(series_keys) - 1) / 2) * bar_width
            ax.bar(pos, amount, bottom=base, width=bar_width * .92 * bar_box_width,
                   color=stack_colors[legend_key],
                   label=legend_label(rename, legend_key, aliases))
            base_map[slot] = base + amount
        ax.set_xticks(range(len(cats)), cats)
        ax.set_xlim(-.5, max(.5, len(cats) - .5))
        # Preserve a separate handle for every actual group combination,
        # even if its user-edited display label equals another entry's.
        if legend_keys and not spec.get("_hide_ungrouped_legend", False):
            stack_legend = ax.legend(handles=[Patch(facecolor=stack_colors[name],
                                     label=legend_label(rename, name, aliases))
                               for name in legend_keys], title=None)
            stack_legend._apmtools_stack_keys = tuple(legend_keys)
    elif spec.get("multi_axis"):
        from .multi_axis import records, style_maps
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        entries = list(records(spec))
        color_map, dash_map, style_groups = style_maps(spec, multi_y_color_by, palette, trace_colors)
        variable_names = list(ys)
        x_names = list(spec.get("x_variables", [x]))
        group_names = style_groups if group_col else []
        # Reuse stable positions for per-variable markers and hatches.
        x_position = {name: i for i, name in enumerate(x_names)}
        y_position = {name: i for i, name in enumerate(ys)}
        group_position = {name: i for i, name in enumerate(group_names)}
        active = set()
        if kind == "scatter":
            scatter_markers = ['o', 's', 'D', 'x', '^', 'v', 'P', '*']
            for key, group, y, xv, category, rows in entries:
                g = group if group_col else None
                active.add((y, g))
                # Use the same editable series keys as the appearance controls.
                # Previously this multi-axis branch ignored all three overrides.
                candidates = (key, f"{g}, {y}" if g is not None else str(y),
                              str(g) if g is not None else str(y), str(y))
                def override(mapping, fallback):
                    return next((mapping[name] for name in candidates if name in mapping), fallback)
                size = _bounded_float({"value": override(sizes, marker_size)}, "value", marker_size, 0.1, 100)
                alpha = _bounded_float({"value": override(marker_alphas, 1)}, "value", 1, 0, 1)
                default_marker = scatter_markers[x_position[xv] % len(scatter_markers)]
                marker = (_scatter_marker({"value": override(markers, "dot")}, "value")
                          if any(name in markers for name in candidates) else default_marker)
                ax.scatter(rows[xv], rows[y], color=color_map[(y, g)],
                           marker=marker, s=size**2, alpha=alpha)
        else:
            cats = list(pd.unique(df[x].astype(str)))
            combinations = [(g, y) for g in (group_names or ['All observations']) for y in ys]
            n = max(1, len(combinations))
            width = .82 / n
            # Resolve categorical positions once rather than scanning both
            # lists for every multi-axis trace. Keep the original ordering.
            combination_index = {item: i for i, item in enumerate(combinations)}
            category_index = {item: i for i, item in enumerate(cats)}
            from collections import defaultdict
            buckets = defaultdict(list)
            for key, group, y, xv, category, rows in entries:
                g = group if group_col else None
                active.add((y, g))
                idx = combination_index[(group, y)]
                pos = category_index[category] + (idx - (n - 1)/2)*width
                values = pd.to_numeric(rows[y], errors='coerce').dropna().to_numpy(dtype=float)
                buckets[(pos, y, g)].extend(values)
            for (pos, y, g), values in buckets.items():
                if not values:
                    continue
                color = color_map[(y, g)]
                hatch_options = ['', '/', '\\', 'x', '.', '+', '-']
                hatch_index = y_position[y] if multi_y_color_by == 'group' else group_position.get(g, 0)
                hatch = '' if multi_y_color_by in ('none', 'both') else hatch_options[hatch_index % len(hatch_options)]
                if kind == 'bar':
                    ax.bar(pos, sum(values), width=width*.92*bar_box_width, color=color, hatch=hatch)
                else:
                    bp = ax.boxplot([values], positions=[pos], widths=width*.92*bar_box_width,
                                    manage_ticks=False, patch_artist=True,
                                    showfliers=bool(box_showfliers),
                                    vert=(box_orientation != 'horizontal'))
                    bp['boxes'][0].set_facecolor(color)
                    bp['boxes'][0].set_hatch(hatch)
                    bp['boxes'][0].set_alpha(.7)
            if kind == 'box' and box_orientation == 'horizontal':
                ax.set_yticks(range(len(cats)), cats)
                ax.set_ylim(-.5, max(.5, len(cats)-.5))
            else:
                ax.set_xticks(range(len(cats)), cats)
                ax.set_xlim(-.5, max(.5, len(cats)-.5))
        handles, legend_names = [], []
        def legend_entry(label, color, marker=None):
            handles.append(Line2D([], [], color=color, marker=marker or None,
                                  linestyle='None' if marker else '-', linewidth=2))
            legend_names.append(legend_label(rename, label, aliases))
        if multi_y_color_by == 'none':
            for g in group_names:
                if any((y, g) in active for y in ys):
                    legend_entry(g, color_map[(ys[0], g)])
        elif multi_y_color_by == 'both':
            for g in (group_names or [None]):
                for y in ys:
                    if (y, g) in active:
                        legend_entry(f'{g}, {y}' if g is not None else y, color_map[(y, g)])
        else:
            for y in ys:
                pairs = [(y, g) for g in (group_names or [None]) if (y, g) in active]
                if pairs:
                    legend_entry(y, color_map[pairs[0]])
            if group_names:
                handles.append(Line2D([], [], color='none'))
                legend_names.append(' ')
                for g in group_names:
                    pairs = [(y, g) for y in ys if (y, g) in active]
                    if pairs:
                        legend_entry(g, color_map[pairs[0]])
        if kind == 'scatter' and len(x_names) > 1:
            if handles:
                handles.append(Line2D([], [], color='none'))
                legend_names.append(' ')
            for i, xv in enumerate(x_names):
                if any(entry[3] == xv for entry in entries):
                    legend_entry(xv, '#666666', ['o', 's', 'D', 'x', '^', 'v', 'P', '*'][i % 8])
        if handles and not spec.get('_hide_ungrouped_legend', False):
            ax.legend(handles, legend_names)
    elif kind == "histogram":
        arrays = []
        for name, rows in series():
            vals = pd.to_numeric(rows[x], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            arrays.append(vals[np.isfinite(vals)])
        arrangement = hist_arrangement or ("stack" if stacked or histtype == "barstacked" else "layer")
        appearance = hist_appearance or {"bar":"bars", "barstacked":"bars", "step":"step",
                                         "stepfilled":"stepfilled"}.get(histtype, "bars")
        stat = hist_stat or ("density" if density else "count")
        edges, heights = histogram_heights(arrays, bins=bins, stat=stat,
            arrangement=arrangement, common_bins=hist_common_bins,
            common_norm=hist_common_norm, cumulative=cumulative,
            common_edges=_hist_shared_edges,
            norm_total=_hist_norm_total if hist_common_norm else None,
            bin_mode=("discrete" if hist_discrete else hist_bin_mode),
            bin_width=hist_bin_width, range_min=hist_range_min, range_max=hist_range_max)
        if heights:
            bottoms = np.zeros_like(heights[0]) if arrangement in ("stack", "fill") else None
            nseries = len(heights)
            for i, (name, edge, height) in enumerate(zip(labels, edges, heights)):
                label = legend_label(rename, name, aliases)
                alpha = _bounded_float(histogram_alphas, name,
                                       1 if appearance == "step" else .65, 0, 1)
                color = colors[name]
                widths = np.diff(edge)
                if arrangement == "dodge":
                    width = widths / nseries
                    left = edge[:-1] + i * width
                else:
                    width = widths
                    left = edge[:-1]
                baseline = bottoms if bottoms is not None else np.zeros_like(height)
                if appearance == "bars":
                    ax.bar(left, height, width=width, align="edge", bottom=baseline,
                           color=color, alpha=alpha, label=label)
                elif appearance in ("step", "stepfilled"):
                    display_edges = (np.r_[left, left[-1] + width[-1]]
                                     if arrangement == "dodge" else edge)
                    ax.stairs(baseline + height, display_edges, baseline=baseline,
                              fill=appearance == "stepfilled", alpha=alpha,
                              color=color, label=label)
                elif appearance == "poly":
                    mids = edge[:-1] + widths / 2
                    ax.plot(mids, baseline + height, color=color, alpha=alpha, label=label)
                else:
                    raise ValueError(f"Unknown histogram appearance: {appearance}")
                if hist_kde and not hist_discrete and arrangement == "layer" and not cumulative:
                    # KDE follows the selected histogram statistic and the
                    # actual bin width; for variable-width bins use median.
                    group = arrays[i]
                    if hist_range_min not in (None, ''):
                        group = group[group >= float(hist_range_min)]
                    if hist_range_max not in (None, ''):
                        group = group[group <= float(hist_range_max)]
                    denom = (_hist_norm_total if hist_common_norm and _hist_norm_total is not None
                             else sum(np.count_nonzero((a >= float(hist_range_min) if hist_range_min not in (None, '') else np.ones(len(a), dtype=bool)) &
                                                       (a <= float(hist_range_max) if hist_range_max not in (None, '') else np.ones(len(a), dtype=bool))) for a in arrays)
                             if hist_common_norm else len(group))
                    if len(group) > 1 and len(edge) > 1:
                        grid = np.linspace(edge[0], edge[-1], 256)
                        kde = histogram_kde(group, grid, bandwidth=hist_kde_bw)
                        width_scale = float(np.median(np.diff(edge)))
                        if stat == "count":
                            kde *= len(group) * width_scale
                        elif stat == "frequency":
                            kde *= len(group)
                        elif stat == "density":
                            kde *= len(group) / denom if denom else 0
                        elif stat == "probability":
                            kde *= (len(group) / denom if denom else 0) * width_scale
                        elif stat == "percent":
                            kde *= (len(group) / denom if denom else 0) * width_scale * 100
                        ax.plot(grid, kde, color=color, linestyle="-", linewidth=1.7,
                                label="_nolegend_")
                if bottoms is not None:
                    bottoms = bottoms + height
        defaults["y_label"] = ("Proportion" if arrangement == "fill" else
            ("Cumulative " if cumulative else "") +
            {"count":"Count", "frequency":"Frequency", "probability":"Probability",
             "percent":"Percent", "density":"Density"}[stat])
    elif kind == "scatter":
        for name, rows in series():
            color = colors.get(name, palette_values[0])
            ax.scatter(rows[x], rows[ys[0]],
                       s=_bounded_float(sizes, name, marker_size, 0.1, 100)**2,
                       marker=_scatter_marker(markers, name),
                       alpha=_bounded_float(marker_alphas, name, 1, 0, 1),
                       color=color, label=legend_label(rename, name, aliases))
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
        # Partition the line data once, not once per Y variable.  This
        # preserves group order and the existing equality semantics while
        # avoiding repeated full-column masks for multi-Y plots.
        grouped_line_rows = ([(n, line_df.loc[line_df[group_col] == n]) for n in labels]
                             if group_col else None)
        # A group can contain many independent traces (e.g. one per identifier
        # or per missing-data section).  Their visual identity is the GROUP,
        # not the individual trace.  Collect only groups actually drawn so the
        # legend and the editable label keys have one entry per group value.
        plotted_line_groups = set()
        band_has_visible_interval = False
        band_max_n = 0
        band_nonempty_points = 0
        for y in ys:
            sets = (grouped_line_rows if grouped_line_rows is not None else
                    [(str(y) if len(ys) > 1 else "All observations", line_df)])
            for name, rows in sets:
                rows = rows.loc[rows[y].notna()]
                if rows.empty:
                    continue
                plotted_line_groups.add(str(name))
                key = (str(y), str(name) if group_col else None)
                color = multi_colors[key] if multi_y else colors.get(name, palette_values[0])
                line_style_value = (_line_style(styles, name) if not multi_y else
                                    _line_style(styles, name) if name in styles and len(ys) == 1 else
                                    multi_dashes[key])
                if error_bands:
                    # Bands are computed over observations at identical X values
                    # inside this plotted group, not on each individual ID trace.
                    # Use the unsegmented prepared dataframe to retain all rows.
                    source = (df.loc[df[group_col].astype(str) == str(name)]
                              if group_col else df)
                    from .line_bands import calculate_resolved_line_band
                    band = calculate_resolved_line_band(
                        source, x, y, method=error_band_method,
                        confidence=error_band_confidence,
                        lower_column=error_band_lower, upper_column=error_band_upper)
                    if band.empty:
                        continue
                    band_max_n = max(band_max_n, int(band["n"].max()))
                    band_nonempty_points += len(band)
                    # When a frequency is selected, split missing X intervals.
                    if spec.get("frequency") is not None:
                        band = _add_gap_sections(band, x_column=x, group_columns=[],
                                                 frequency=spec["frequency"])
                        sections = band.groupby("__line_section", sort=False, dropna=False)
                    else:
                        sections = [(None, band)]
                    for _, part in sections:
                        ax.plot(part[x], part["center"], color=color,
                                linewidth=float(line_width), linestyle=line_style_value,
                                alpha=_bounded_float(alphas, name, 1, 0, 1), label="_nolegend_")
                        # A shaded polygon needs at least two consecutive
                        # finite interval points.  For isolated valid points,
                        # draw a vertical interval instead of an invisible
                        # zero-area fill_between polygon.
                        lo = part["lower"].to_numpy(dtype=float)
                        hi = part["upper"].to_numpy(dtype=float)
                        center = part["center"].to_numpy(dtype=float)
                        finite = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(center)
                        band_has_visible_interval |= bool(np.any(finite))
                        if finite.sum() >= 2:
                            ax.fill_between(part[x], lo, hi, where=finite,
                                            color=color, alpha=float(np.clip(error_band_alpha, 0, 1)),
                                            linewidth=0, label="_nolegend_")
                        # Standalone points (or holes within a band) need
                        # whiskers to make their genuine intervals visible.
                        isolated = finite.copy()
                        if len(finite) > 1:
                            isolated[1:] &= ~finite[:-1]
                            isolated[:-1] &= ~finite[1:]
                        if isolated.any():
                            ax.vlines(part.loc[isolated, x], lo[isolated], hi[isolated],
                                      color=color, linewidth=max(1.0, float(line_width)),
                                      alpha=float(np.clip(error_band_alpha, 0.1, 1)),
                                      label="_nolegend_")
                else:
                    segment_col = section_col or id_col
                    parts = (rows.groupby(segment_col, dropna=False, sort=False)
                             if segment_col else [(None, rows)])
                    for _, part in parts:
                        part = part.sort_values(x)
                        ax.plot(part[x], part[y], color=color,
                                linewidth=float(line_width), linestyle=line_style_value,
                                alpha=_bounded_float(alphas, name, 1, 0, 1), label="_nolegend_")
        if error_bands and not band_has_visible_interval and band_nonempty_points:
            if error_band_method == "custom":
                reason = "No finite lower/upper bound pairs in the selected data."
            else:
                reason = ("Each X value has fewer than two observations within its plotted group "
                          f"(maximum n={band_max_n}).")
            ax.text(0.5, 0.02,
                    "No estimable error bands. " + reason +
                    "\nTry removing identifier from Group by, using unaggregated data, "
                    "or providing custom bounds.",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=9, color="darkred", wrap=True,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9,
                          "edgecolor": "#cc8888"}, zorder=20)
        if group_col and not multi_y and not spec.get("_hide_ungrouped_legend", False):
            from matplotlib.lines import Line2D
            visible_groups = [name for name in labels if name in plotted_line_groups]
            if visible_groups:
                handles = [Line2D([], [], color=colors[name],
                                  linewidth=float(line_width),
                                  linestyle=_line_style(styles, name),
                                  alpha=_bounded_float(alphas, name, 1, 0, 1))
                           for name in visible_groups]
                ax.legend(handles,
                          [legend_label(rename, name, aliases) for name in visible_groups],
                          title=None)
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
                ax.bar(xpos, [totals[pos] for pos in xpos], width=width * .92 * bar_box_width,
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
                                positions=box_positions, widths=width * .92 * bar_box_width,
                                manage_ticks=False, patch_artist=True,
                                showfliers=bool(box_showfliers),
                                vert=(box_orientation != "horizontal"))
                for patch, position in zip(bp["boxes"], box_positions):
                    patch.set_facecolor(bucket_colors[position])
                    patch.set_alpha(.7)
                if box_showfliers:
                    # Explicitly reset numeric limits using all observations;
                    # avoids retaining whisker-only limits after facet changes.
                    all_values = np.asarray(numeric, dtype=float)
                    all_values = all_values[np.isfinite(all_values)]
                    if all_values.size:
                        low, high = float(all_values.min()), float(all_values.max())
                        margin = .05 * (high - low) if high > low else max(abs(low) * .05, .5)
                        if box_orientation == "horizontal":
                            ax.set_xlim(low - margin, high + margin)
                        else:
                            ax.set_ylim(low - margin, high + margin)
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
            ax.legend(handles=[Patch(facecolor=colors[name], label=legend_label(rename, name, aliases))
                               for name in group_names], title=None)
    else:
        raise ValueError(f"Unknown plot type: {kind}")

    if kind not in ("box", "bar") and not spec.get("multi_axis"):
        handles, texts = ax.get_legend_handles_labels()
        if handles and not spec.get("_hide_ungrouped_legend", not bool(spec["groups"])):
            ax.legend(handles, texts, title=None)
    if multi_y:
        from matplotlib.lines import Line2D
        # Independent legend keys for the two dimensions, with explicit headings.
        legend_handles = []
        legend_labels_out = []
        active = {(str(y), str(g) if group_col else None)
                  for y in ys for g in (group_names if group_col else [None])
                  if not df.loc[(df[group_col] == g) if group_col else slice(None), y].dropna().empty}
        if multi_y_color_by == "none":
            for group in group_names if group_col else []:
                if not any((v, group) in active for v in variable_names):
                    continue
                key = (variable_names[0], group)
                legend_handles.append(Line2D([], [], color=multi_colors[key], linestyle=multi_dashes[key], linewidth=line_width))
                legend_labels_out.append(legend_label(rename, group, aliases))
        if multi_y_color_by == "both":
            for group in (group_names or [None]):
                for variable in variable_names:
                    key = (variable, group)
                    # A visible element may be only one of several Y variables
                    # within a group. Never create proxy legend entries for
                    # combinations with no plotted observations.
                    if key not in active:
                        continue
                    label = f"{group}, {variable}" if group is not None else variable
                    legend_handles.append(Line2D([], [], color=multi_colors[key], linestyle=multi_dashes[key], linewidth=line_width))
                    legend_labels_out.append(legend_label(rename, label, aliases))
        added_section = False
        for heading, keys, by_var in (() if multi_y_color_by in ("both", "none") else (("Y variable", variable_names, True), ("Grouping", group_names, False))):
            if not keys or (not by_var and not group_col):
                continue
            if added_section:
                legend_handles.append(Line2D([], [], color="none"))
                legend_labels_out.append(" ")
            added_section = True
            legend_handles.append(Line2D([], [], color="none"))
            legend_labels_out.append(heading)
            for value in keys:
                relevant = [pair for pair in active if (pair[0] == value if by_var else pair[1] == value)]
                if not relevant:
                    continue
                key = relevant[0]
                legend_handles.append(Line2D([], [], color=multi_colors[key], linestyle=multi_dashes[key], linewidth=line_width))
                legend_labels_out.append(legend_label(rename, value, aliases))
        if legend_handles:
            ax.legend(legend_handles, legend_labels_out, title=None)
    # Overlays do not participate in statistical aggregation or alter axis
    # limits. They span the current horizontal plot region, including facets.
    if kind in ("line", "scatter") and reference_lines:
        from .reference_lines import parse_reference_lines
        from matplotlib.lines import Line2D
        from matplotlib import colors as mpl_colors
        displayed = spec.get("_visible_elements")
        legend = ax.get_legend()
        old_handles = list(getattr(legend, "legend_handles", [])) if legend else []
        old_labels = [text.get_text() for text in legend.get_texts()] if legend else []
        # Raw ungrouped line segments intentionally have _nolegend_ labels.
        # When an overlay introduces a legend, include their actual plotted
        # series too, without exposing every identifier/segment as a trace.
        if not old_handles and not group_col and not multi_y and ys:
            if kind == "line" and plotted_line_groups:
                base_name = "All observations"
                old_handles.append(Line2D([], [], color=colors.get(base_name, palette_values[0]),
                                          linewidth=float(line_width),
                                          linestyle=_line_style(styles, base_name),
                                          alpha=_bounded_float(alphas, base_name, 1, 0, 1)))
                old_labels.append(legend_label(rename, str(ys[0]), aliases))
            elif kind == "scatter" and len(df) and df[ys[0]].notna().any():
                base_name = "All observations"
                old_handles.append(Line2D([], [], color=colors.get(base_name, palette_values[0]),
                                          linestyle="None", marker=_scatter_marker(markers, base_name),
                                          markersize=_bounded_float(sizes, base_name, marker_size, 0.1, 100)))
                old_labels.append(legend_label(rename, str(ys[0]), aliases))
        base_count = len(old_handles)
        for index, line in enumerate(parse_reference_lines(reference_lines)):
            if displayed is not None and line["key"] not in displayed:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(df[x])
            is_datetime = (kind == "line" and pd.api.types.is_datetime64_any_dtype(df[x]))
            if not (is_numeric or is_datetime):
                continue
            color = overrides.get(line["key"], palette_values[index % len(palette_values)])
            if not mpl_colors.is_color_like(color):
                color = palette_values[index % len(palette_values)]
            # The reference-line editor owns this label; do not apply the
            # generic legend-label mapping (which would create two editors).
            label = decode_plot_text(line["label"])
            # Matplotlib dates are measured in days. A dated intercept anchors
            # y0 to a real timestamp; the selected unit converts m into y/day.
            if is_datetime:
                import matplotlib.dates as mdates
                origin = line.get("origin", "")
                if not origin:
                    valid = pd.to_datetime(df[x], errors="coerce").dropna()
                    if valid.empty:
                        continue
                    origin = valid.min()
                try:
                    origin = pd.Timestamp(origin)
                    if pd.isna(origin):
                        continue
                    anchor = float(mdates.date2num(origin.to_pydatetime()))
                except (ValueError, TypeError, OverflowError):
                    continue
                per_day = {"second": 86400., "minute": 1440., "hour": 24.,
                           "day": 1., "week": 1. / 7.}
                slope = line["slope"] * per_day[line["time_unit"]]
            else:
                anchor, slope = 0., line["slope"]
            # axline spans the axes without extending data limits and stays
            # clipped when a panel/axis is resized.
            ax.axline((anchor, line["intercept"]), slope=slope,
                      color=color, linewidth=float(line_width), linestyle="--",
                      label="_nolegend_", clip_on=True)
            old_handles.append(Line2D([], [], color=color, linewidth=float(line_width), linestyle="--"))
            old_labels.append(label)
        if old_handles and (legend is not None or len(old_handles) > base_count):
            ax.legend(old_handles, old_labels, title=None)
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
    ax.set_xlabel(decode_plot_text(xlabel), fontsize=axis_font_size)
    ax.set_ylabel(decode_plot_text(ylabel), fontsize=axis_font_size)
    ax.set_title(decode_plot_text(title), fontsize=title_font_size)
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
        kwargs = {"loc": loc, "title": None, "ncol": max(1, len(labels)) if orientation == "h" else 1}
        if not inside:
            kwargs["bbox_to_anchor"] = (1.01, 1)
        ax.legend(handles, labels, **kwargs)
    fig.tight_layout()
    return fig

