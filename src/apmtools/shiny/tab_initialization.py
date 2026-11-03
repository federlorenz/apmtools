"""Headless preparation of new and restored Shiny tabs.

The UI still owns Shiny module creation, reactive counters and DOM mounting.
This module resolves the data and settings needed *before* mounting a tab.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any
import pandas as pd

from .derived_views import prepare_table_source


@dataclass(frozen=True)
class PlotInitialization:
    label: str
    settings: dict


@dataclass(frozen=True)
class TableInitialization:
    label: str
    frame: pd.DataFrame
    settings: dict | None
    lineage: list


class TabInitializer:
    """Resolve plot/table initialization without importing Shiny."""

    def __init__(self, workspace, views, summary):
        self.workspace = workspace
        self.views = views
        self.summary = summary

    def new_plot(self, selected, number):
        # A partially initialized input can fail during a snapshot; preserve
        # the existing 'start with defaults' behavior for this operation.
        inherited = {}
        if self.workspace.kind(selected) == 'plot':
            try:
                inherited = copy.deepcopy(self.workspace.plot_snapshots[selected]())
            except Exception:
                inherited = {}
        return PlotInitialization(f'Plot {number}', inherited)

    def restored_plot(self, preset_id, tab_id, number):
        preset = self.views.preset(preset_id)
        inherited = self.views.restore_plot(preset_id, tab_id)
        return PlotInitialization(preset.get('name') or f'Plot {number}', inherited)

    def new_table(self, selected, number):
        kind = self.workspace.kind(selected)
        if kind == 'table':
            frame, steps = prepare_table_source(
                self.summary, parent_kind='table',
                parent_steps=self.workspace.table_lineage.get(selected, []),
                parent_settings=self.workspace.table_settings[selected]())
        elif kind == 'plot':
            frame, steps = prepare_table_source(
                self.summary, parent_kind='plot',
                parent_settings=self.workspace.plot_snapshots[selected]())
        else:
            frame, steps = prepare_table_source(self.summary)
        return TableInitialization(f'Table {number}', frame, None, steps)

    def new_quality(self, selected, number):
        """Create a quality view from the current plot/table's transformed data.

        The stored lineage replays the source transformation on restoration;
        no potentially stale dataframe snapshots are retained in saved views.
        """
        kind = self.workspace.kind(selected)
        if kind == 'table':
            frame, steps = prepare_table_source(
                self.summary, parent_kind='table',
                parent_steps=self.workspace.table_lineage.get(selected, []),
                parent_settings=self.workspace.table_settings[selected]())
        elif kind == 'plot':
            frame, steps = prepare_table_source(
                self.summary, parent_kind='plot',
                parent_settings=self.workspace.plot_snapshots[selected]())
        else:
            frame, steps = prepare_table_source(self.summary)
        return TableInitialization(f'Quality {number}', frame, {'source_steps': steps}, steps)

    def restored_table(self, preset_id, number):
        preset = self.views.preset(preset_id)
        _source, settings = self.views.restore_table(preset_id)
        frame, steps = prepare_table_source(self.summary, restored_settings=settings)
        return TableInitialization(preset.get('name') or f'Table {number}',
                                   frame, settings, steps)
