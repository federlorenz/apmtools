"""Pure-Python bounded undo/redo and saved-configuration tracking.

Does not retain any DataFrame. A baseline is a distinct saved snapshot, not
an entry in history; restoring a snapshot never automatically creates an edit.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field


def _snapshot(settings):
    if not isinstance(settings, dict):
        raise TypeError('View settings must be a dictionary')
    return json.loads(json.dumps(settings, allow_nan=False))


@dataclass
class _History:
    entries: list[dict] = field(default_factory=list)
    position: int = -1
    baseline: dict | None = None
    limit: int = 50

    @property
    def current(self):
        return copy.deepcopy(self.entries[self.position]) if self.position >= 0 else None

    @property
    def dirty(self):
        return self.current != self.baseline


class ViewHistory:
    """Independent bounded histories keyed by live tab ID.

    For a new unsaved view baseline is None; a restored view's baseline is the
    saved configuration. All returned dictionaries are defensive copies.
    """

    def __init__(self, *, limit=50):
        if not isinstance(limit, int) or limit < 2:
            raise ValueError('History limit must be an integer >= 2')
        self.limit = limit
        self._tabs: dict[str, _History] = {}

    def register(self, tab_id, settings, *, saved=False):
        if tab_id in self._tabs:
            raise ValueError(f'Already registered: {tab_id}')
        initial = _snapshot(settings)
        self._tabs[tab_id] = _History([initial], 0,
                                      copy.deepcopy(initial) if saved else None,
                                      self.limit)

    def record(self, tab_id, settings):
        state = self._tabs[tab_id]
        new = _snapshot(settings)
        if state.entries[state.position] == new:
            return False
        state.entries = state.entries[:state.position + 1]
        state.entries.append(new)
        if len(state.entries) > state.limit:
            state.entries.pop(0)
        state.position = len(state.entries) - 1
        return True

    def undo(self, tab_id):
        state = self._tabs[tab_id]
        if state.position == 0:
            return None
        state.position -= 1
        return state.current

    def redo(self, tab_id):
        state = self._tabs[tab_id]
        if state.position == len(state.entries) - 1:
            return None
        state.position += 1
        return state.current

    def mark_saved(self, tab_id, settings=None):
        state = self._tabs[tab_id]
        # Save commits the actual state being persisted, which may differ from
        # the last observed input during asynchronous Shiny initialization.
        if settings is not None:
            self.record(tab_id, settings)
        state.baseline = state.current

    def status(self, tab_id):
        state = self._tabs[tab_id]
        return {'can_undo': state.position > 0,
                'can_redo': state.position < len(state.entries) - 1,
                'dirty': state.dirty, 'length': len(state.entries)}

    def current(self, tab_id):
        return self._tabs[tab_id].current

    def rekey(self, old_id, new_id):
        """Transfer editing history to a newly mounted Shiny module ID."""
        if new_id in self._tabs:
            raise ValueError(f'Already registered: {new_id}')
        self._tabs[new_id] = self._tabs.pop(old_id)

    def discard(self, tab_id):
        self._tabs.pop(tab_id, None)
