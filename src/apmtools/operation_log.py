"""Thread-safe, bounded progress events for one Shiny operation.

The queue is per operation, never process-global. Draining occurs on the Shiny
async task, and only while there is actual work. No monitoring thread/timer.
"""
from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from queue import Empty, Full, Queue


class OperationLog:
    def __init__(self, *, max_pending=2000):
        self._pending = Queue(maxsize=max_pending)

    def emit(self, message):
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        try:
            self._pending.put_nowait(entry)
        except Full:
            try:
                self._pending.get_nowait()  # Preserve newest messages.
            except Empty:
                pass
            try:
                self._pending.put_nowait(entry)
            except Full:
                pass

    def drain(self):
        result = []
        while True:
            try:
                result.append(self._pending.get_nowait())
            except Empty:
                return result


async def stream_operation(operation, log, show_messages, *, interval=0.2):
    """Await a worker coroutine and batch progress messages during execution.

    show_messages is invoked on the event loop, never from a worker thread.
    Worker errors propagate unchanged, after final pending messages are shown.
    """
    task = asyncio.ensure_future(operation)
    try:
        while not task.done():
            updates = log.drain()
            if updates:
                show_messages(updates)
            await asyncio.wait({task}, timeout=interval)
        return await task
    finally:
        updates = log.drain()
        if updates:
            show_messages(updates)
