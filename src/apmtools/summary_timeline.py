"""Session-local navigation of recorded Summary transformations.

The source Summary at the start of the editing session is the replay anchor.
A saved Summary may already have history, but old intermediate data states
cannot be reconstructed without its original, pre-history source dataset.
"""
from __future__ import annotations

from .history_codec import load_history


class SummaryStepTimeline:
    """Navigate recorded edits without storing a full DataFrame per step."""

    def __init__(self, summary):
        self.base = summary
        self.history_prefix = len(getattr(summary, "history", None) or ())
        self.steps = []
        self.cursor = 0
        self.current = summary

    @property
    def count(self):
        return len(self.steps)

    def select(self, index):
        """Reconstruct the state after `index` editing-session steps."""
        index = int(index)
        if not 0 <= index <= self.count:
            raise IndexError("History step is outside the available range")
        if index == self.cursor:
            return self.current
        result = self.base if index == 0 else self.base.apply_history(self.steps[:index])
        self.cursor = index
        self.current = result
        return result

    def append(self, result):
        """Commit a successful recorded transformation, pruning future steps."""
        prior = list(getattr(self.current, "history", None) or ())
        updated = list(getattr(result, "history", None) or ())
        if len(updated) <= len(prior) or updated[:len(prior)] != prior:
            raise ValueError("Editing operation must append recorded history steps")
        # A single sidebar action can legitimately call several recorded
        # Summary methods (e.g. keep columns removes each column in turn,
        # and renaming several columns uses temporary intermediate names).
        # Preserve every recorded transformation so navigation and replay
        # remain consistent with the actual Summary.history sequence.
        steps = load_history(updated[len(prior):])
        self.steps = self.steps[:self.cursor] + steps
        self.cursor += len(steps)
        self.current = result
        return result

    def options(self):
        """Selector labels for the original editing-session state and its edits."""
        labels = {"0": "0. Original editing state"}
        for index, step in enumerate(self.steps, 1):
            kwargs = step.get("args", {})
            method = step["method"]
            detail = kwargs.get("name") or kwargs.get("action") or kwargs.get("column")
            detail = str(detail)[:55] if detail is not None else ""
            labels[str(index)] = f"{index}. {method}" + (f" — {detail}" if detail else "")
        return labels
