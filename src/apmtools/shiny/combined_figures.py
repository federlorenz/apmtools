"""Render independent plot views within one Matplotlib figure.

Ordinary views draw natively into existing Axes. Faceted views retain their
internal panel layout by embedding the complete rendered source figure into one
cell; their pixels are rasterized at export DPI, not silently flattened.
"""
from __future__ import annotations

import math
from io import BytesIO


def normalized_combination(source_ids, available, *, columns=2, hspace=0.55,
                           wspace=0.35, sharex=False, sharey=False):
    """Validate and normalize one combined-view layout without Shiny imports."""
    if not isinstance(source_ids, (list, tuple)):
        raise ValueError('Select at least one plot tab')
    ids = list(dict.fromkeys(source_ids))
    if not ids or any(item not in available for item in ids):
        raise ValueError('Select one or more currently open plot tabs')
    cols = int(columns)
    if not 1 <= cols <= 6:
        raise ValueError('Columns must be between 1 and 6')
    h, w = float(hspace), float(wspace)
    if not (math.isfinite(h) and math.isfinite(w) and 0 <= h <= 3 and 0 <= w <= 3):
        raise ValueError('Spacing must be finite and between 0 and 3')
    return {'sources': ids, 'columns': cols, 'hspace': h, 'wspace': w,
            'sharex': bool(sharex), 'sharey': bool(sharey)}


def combined_figure(source_plots, *, columns=2, hspace=0.55, wspace=0.35,
                    sharex=False, sharey=False, dpi=100, cell_width=5.5,
                    cell_height=4.2):
    """Draw (label, prepared specification, options) tuples in a single Figure.

    Each native Axes retains its own plot styling, legend and titles. Axis
    sharing is opt-in and only appropriate for compatible axes/units. Faceted
    source plots are embedded as complete panel-grid images to avoid replacing
    the parent figure inside the existing panel renderer.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from .matplotlib_renderer import _matplotlib_direct_figure

    if not source_plots:
        raise ValueError('Select at least one plot')
    ncols = max(1, min(int(columns), len(source_plots)))
    nrows = math.ceil(len(source_plots) / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, dpi=dpi,
                             figsize=(cell_width * ncols, cell_height * nrows))
    populated = []
    for index, (label, spec, options) in enumerate(source_plots):
        ax = axes.flat[index]
        options = dict(options)
        options.pop('_axes', None)
        options['plot_dpi'] = dpi
        # Panel rendering internally creates a Figure with multiple Axes.
        # Preserve that nested layout with a complete image in this subplot.
        if options.get('panel_by'):
            nested = _matplotlib_direct_figure(spec, **options)
            try:
                buffer = BytesIO()
                nested.savefig(buffer, format='png', dpi=dpi)
                buffer.seek(0)
                image = plt.imread(buffer, format='png')
                ax.imshow(image, interpolation='nearest', aspect='equal')
                ax.set_axis_off()
            finally:
                plt.close(nested)
        else:
            # The base renderer skips its datetime-origin initialization when
            # supplied an Axes. Resolve the anchor from this full source view.
            frame = spec.get('data')
            x = spec.get('x')
            if (spec.get('kind') == 'line' and frame is not None
                    and x in frame.columns and options.get('reference_lines')):
                import pandas as pd
                if pd.api.types.is_datetime64_any_dtype(frame[x]):
                    from .reference_lines import parse_reference_lines
                    anchors = pd.to_datetime(frame[x], errors='coerce').dropna()
                    if not anchors.empty:
                        origin = anchors.min().isoformat()
                        options['reference_lines'] = [
                            dict(line, origin=line.get('origin') or origin)
                            for line in parse_reference_lines(options['reference_lines'])]
            # The base renderer skips Figure-level legend positioning when
            # supplied an Axes. Keep local legends for independent subplots.
            options['legend_inside'] = True
            _matplotlib_direct_figure(spec, _axes=ax, **options)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_visible(True)
        if not ax.get_title() and label and not options.get('panel_by'):
            ax.set_title(str(label))
        populated.append(ax)
    for ax in list(axes.flat)[len(source_plots):]:
        ax.set_visible(False)
    if sharex or sharey:
        # Axis sharing only applies to ordinary source plots. Date and numeric
        # X axes must not be linked; likewise avoid joining unlike Y labels.
        ordinary = [(ax, spec, opts) for ax, (_, spec, opts) in zip(populated, source_plots)
                    if not opts.get('panel_by')]
        if len(ordinary) > 1:
            import pandas as pd
            root, root_spec, root_opts = ordinary[0]
            root_date = pd.api.types.is_datetime64_any_dtype(root_spec['data'][root_spec['x']])
            for ax, spec, opts in ordinary[1:]:
                date_axis = pd.api.types.is_datetime64_any_dtype(spec['data'][spec['x']])
                if sharex and date_axis != root_date:
                    plt.close(fig)
                    raise ValueError('Cannot share X axes between datetime and numeric plots')
                if sharey and ax.get_ylabel() != root.get_ylabel():
                    plt.close(fig)
                    raise ValueError('Cannot share Y axes with different axis labels/units')
                if sharex:
                    ax.sharex(root)
                if sharey:
                    ax.sharey(root)
    fig.subplots_adjust(hspace=float(hspace), wspace=float(wspace),
                        left=.08, right=.96, top=.94, bottom=.09)
    return fig
