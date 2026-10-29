"""Plot configuration lifecycle: reactive drafts and explicitly retained presets."""
from __future__ import annotations
import copy
import json
import uuid


def _json_ready(value):
    """Require portable, plain JSON-compatible configuration values."""
    return json.loads(json.dumps(value, allow_nan=False))


class PlotConfigManager:
    """Keep live tab drafts separate from explicitly retained Summary.plots.

    All mutations go through methods. When used within Shiny, each tab has
    independent reactive signals for data and appearance changes. Retained
    presets are snapshots and are never modified by draft updates.
    """

    def __init__(self, summary):
        self.summary = summary
        self.drafts = {}
        self.revision = 0  # Backwards-compatible diagnostic counter, not reactive.
        self._signals = {}
        if getattr(summary, 'plots', None) is None:
            summary.plots = []
        if not isinstance(summary.plots, list):
            raise TypeError('Summary.plots must be a list or None')

    # Settings that affect filtering, grouping or aggregation. Everything else
    # is treated as presentation state. Panel layout (panel_by) is
    # presentation-only: it must not invalidate filtering or aggregation. Unknown settings remain in the full
    # snapshot and are conservatively classified as appearance-only until
    # explicitly added here.
    DATA_KEYS = frozenset({
        'plot_type', 'x', 'y', 'numeric_subset_columns',
        'numeric_subset_bounds', 'subset_chain', 'datetime_start',
        'datetime_end', 'group_by', 'frequency', 'aggregate',
        'only_complete', 'flooring_timestamp', 'multi_y_color_by',
    })

    def _signal(self, tab_id, kind):
        # Created only for a tab that is actually used. Shiny is imported
        # lazily so the manager remains usable outside a Shiny session.
        from shiny import reactive
        signals = self._signals.setdefault(tab_id, {})
        if kind not in signals:
            signals[kind] = reactive.value(0)
        return signals[kind]

    def watch(self, tab_id, kind='all'):
        """Declare a per-tab reactive dependency (all/data/appearance/visibility)."""
        if kind not in ('all', 'data', 'appearance', 'visibility'):
            raise ValueError('Unknown configuration signal')
        return self._signal(tab_id, kind).get()

    def update(self, tab_id, config):
        config = _json_ready(config)
        previous = self.drafts.get(tab_id)
        if previous == config:
            return False
        self.drafts[tab_id] = config
        self.revision += 1
        changed = set(config) | set(previous or {})
        changed = {k for k in changed if (previous or {}).get(k) != config.get(k)}
        kinds = {'all'}
        if changed & self.DATA_KEYS:
            kinds.add('data')
        if changed - self.DATA_KEYS:
            kinds.add('appearance')
        if 'visible_elements' in changed:
            kinds.add('visibility')
        # Only notify signals that have subscribers; don't create signals
        # for every tab and category when nobody observes them.
        for kind in kinds:
            signal = self._signals.get(tab_id, {}).get(kind)
            if signal is not None:
                signal.set(signal.get() + 1)
        return True

    def get(self, tab_id, *, watch=False, kind='all'):
        if watch:
            self.watch(tab_id, kind)
        return copy.deepcopy(self.drafts[tab_id])


    def patch(self, tab_id, changes):
        """Apply a partial UI change to the canonical tab configuration.

        Retain unrelated fields (including settings in inactive sidebar tabs).
        Return False for no-op updates; never mutate a retained preset.
        """
        if not isinstance(changes, dict):
            raise TypeError('Configuration changes must be a dictionary')
        current = self.get(tab_id) if tab_id in self.drafts else {}
        current.update(_json_ready(changes))
        return self.update(tab_id, current)

    def value(self, tab_id, key, default=None, *, watch=False):
        """Read a setting from one tab without exposing mutable state."""
        if watch:
            self.watch(tab_id)
        return copy.deepcopy(self.drafts.get(tab_id, {}).get(key, default))

    def discard(self, tab_id):
        if tab_id in self.drafts:
            del self.drafts[tab_id]
            self.revision += 1
            self._signals.pop(tab_id, None)

    def retain(self, tab_id, name, *, preset_id=None):
        """Persist only this explicitly chosen draft in Summary.plots."""
        if tab_id not in self.drafts:
            raise KeyError(f'No live configuration for {tab_id}')
        name = str(name).strip()
        if not name:
            raise ValueError('Configuration name must not be empty')
        preset_id = preset_id or uuid.uuid4().hex
        item = {'schema_version': 1, 'id': preset_id, 'name': name,
                'type': 'plot', 'settings': self.get(tab_id)}
        items = copy.deepcopy(self.summary.plots or [])
        for i, old in enumerate(items):
            if old.get('id') == preset_id:
                items[i] = item
                break
        else:
            items.append(item)
        self.summary.plots = items
        self.revision += 1
        return preset_id

    def saved(self):
        return copy.deepcopy(self.summary.plots or [])

    def restore(self, preset_id, tab_id):
        item = next((p for p in self.saved() if p.get('id') == preset_id), None)
        if item is None:
            raise KeyError(preset_id)
        if item.get('schema_version') != 1 or item.get('type') != 'plot':
            raise ValueError('Unsupported saved configuration')
        self.update(tab_id, item['settings'])
        return self.get(tab_id)
