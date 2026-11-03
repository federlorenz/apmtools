"""Pure-Python registry for active Shiny workspace tabs.

The UI retains ownership of reactive counters and module registration.  This
controller owns bookkeeping, keeping tab lifecycle independently testable.
"""
from __future__ import annotations


class WorkspaceController:
    """Track active plot/table tabs without retaining rendered data snapshots."""

    def __init__(self):
        self.plot_snapshots = {}
        self.table_frames = {}
        self.table_settings = {}
        self.table_lineage = {}
        self.active_tabs = ()

    def kind(self, tab_id):
        if tab_id in self.plot_snapshots:
            return 'plot'
        if tab_id in self.table_frames:
            return 'table'
        return None

    def add_plot(self, tab_id, snapshot):
        self._check_new(tab_id)
        if not callable(snapshot):
            raise TypeError('Plot snapshot must be callable')
        self.plot_snapshots[tab_id] = snapshot
        self.active_tabs += (tab_id,)
        return self.active_tabs

    def add_table(self, tab_id, frame, settings, lineage):
        self._check_new(tab_id)
        if not callable(settings):
            raise TypeError('Table settings must be callable')
        self.table_frames[tab_id] = frame
        self.table_settings[tab_id] = settings
        self.table_lineage[tab_id] = lineage
        self.active_tabs += (tab_id,)
        return self.active_tabs

    def remove(self, tab_id):
        if tab_id not in self.active_tabs:
            return False
        self.plot_snapshots.pop(tab_id, None)
        self.table_frames.pop(tab_id, None)
        self.table_settings.pop(tab_id, None)
        self.table_lineage.pop(tab_id, None)
        self.active_tabs = tuple(t for t in self.active_tabs if t != tab_id)
        return True

    def _check_new(self, tab_id):
        if not isinstance(tab_id, str) or not tab_id:
            raise ValueError('Tab ID must be a nonempty string')
        if tab_id in self.active_tabs or self.kind(tab_id) is not None:
            raise ValueError(f'Already registered tab: {tab_id}')
