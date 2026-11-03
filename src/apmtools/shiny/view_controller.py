"""Framework-independent operations for live and retained plot/table views.

Shiny owns input widgets and tab mounting; this controller owns view selection,
configuration snapshots and retained-view lifecycle.  It deliberately delegates
persistence to PlotConfigManager to preserve the existing saved-view schema.
"""
from __future__ import annotations

import copy
from .view_history import ViewHistory


class ViewController:
    """Coordinate WorkspaceController and PlotConfigManager without Shiny APIs."""

    def __init__(self, workspace, manager):
        self.workspace = workspace
        self.manager = manager
        self.history = ViewHistory()

    def begin_editing(self, tab_id, settings, *, restored=False):
        """Start an isolated editing session; restored snapshots are clean."""
        self.history.register(tab_id, settings, saved=restored)

    def record_edit(self, tab_id, settings):
        """Capture a user-committed configuration, not its source DataFrame."""
        return self.history.record(tab_id, settings)

    def undo_edit(self, tab_id):
        """Return previous settings; UI decides how to rehydrate controls."""
        return self.history.undo(tab_id)

    def redo_edit(self, tab_id):
        """Return next settings; UI decides how to rehydrate controls."""
        return self.history.redo(tab_id)

    def edit_status(self, tab_id):
        return self.history.status(tab_id)

    def close_editing(self, tab_id):
        self.history.discard(tab_id)

    def saved(self):
        """Return independently mutable copies of both saved view types."""
        return [p for p in self.manager.saved() if p.get('type', 'plot') in ('plot', 'table')]

    def choices(self):
        """Stable ID -> human-readable label, independent of the active tab."""
        return {p['id']: f"{'Table' if p.get('type') == 'table' else 'Plot'}: {p.get('name', p['id'])}"
                for p in self.saved()}

    def preset(self, preset_id):
        """Resolve a saved view by its stable ID."""
        for item in self.saved():
            if item['id'] == preset_id:
                return item
        raise KeyError(preset_id)

    def save_active(self, tab_id, name):
        """Snapshot an active plot or table without persisting frame data."""
        kind = self.workspace.kind(tab_id)
        if kind == 'plot':
            self.manager.update(tab_id, self.workspace.plot_snapshots[tab_id]())
            preset_id = self.manager.retain(tab_id, name)
            if tab_id in self.history._tabs:
                self.history.mark_saved(tab_id, self.manager.get(tab_id))
            return preset_id
        if kind == 'table':
            settings = copy.deepcopy(self.workspace.table_settings[tab_id]())
            settings['source_steps'] = copy.deepcopy(self.workspace.table_lineage.get(tab_id, []))
            preset_id = self.manager.retain_table(name, settings)
            if tab_id in self.history._tabs:
                self.history.mark_saved(tab_id, settings)
            return preset_id
        raise ValueError('No active plot or table tab')

    def delete(self, preset_id):
        """Delete a retained view without affecting any open tab."""
        self.manager.delete_saved(preset_id)

    def restore_plot(self, preset_id, tab_id):
        """Populate a fresh live plot draft and return its settings."""
        if self.preset(preset_id).get('type') != 'plot':
            raise ValueError('Saved configuration is not a plot')
        return self.manager.restore(preset_id, tab_id)

    def restore_table(self, preset_id):
        """Return source Summary and saved settings for table reconstruction."""
        if self.preset(preset_id).get('type') != 'table':
            raise ValueError('Saved configuration is not a table')
        return self.manager.restore_table(preset_id)
