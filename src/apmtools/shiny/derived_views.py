"""Headless, configuration-only table source preparation for Shiny tab actions.

The application creates tables from the owning Summary and ordered view
settings, never from snapshots of already rendered plot/table data.
"""
from __future__ import annotations

import pandas as pd

from .inherited_source import source_frame, source_steps


def prepare_table_source(summary, *, parent_kind=None, parent_settings=None,
                         parent_steps=None, restored_settings=None):
    """Return (display-ready frame, lineage steps) for new/loaded tables.

    ``parent_kind`` is ``'plot'``, ``'table'`` or ``None`` (Info tab).
    For restoration, pass ``restored_settings`` instead; its saved lineage is
    replayed without modification.  No parent DataFrame is copied or cached.
    """
    if restored_settings is not None:
        steps = restored_settings.get('source_steps', [])
    elif parent_kind in ('plot', 'table'):
        if parent_settings is None:
            raise ValueError('Parent view settings must be supplied')
        steps = source_steps(parent_steps or [],
                             parent_settings, parent_kind)
    elif parent_kind is None:
        steps = []
    else:
        raise ValueError(f'Unknown parent view kind: {parent_kind!r}')
    frame = source_frame(summary, steps)
    frame = frame.reset_index(drop=True)
    frame.columns = [str(column) for column in frame.columns]
    return frame, steps
