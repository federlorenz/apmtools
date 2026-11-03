"""Opt-in, low-overhead JSONL tracing for Shiny callbacks and data operations.

Set APMTOOLS_TRACE_PATH to a writable .jsonl path before launching the app.
Records invocations (not Shiny internal invalidation notifications). Never logs
row values or input contents. Each record is one JSON object.
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
from pathlib import Path
import threading
import time
import uuid

_current_span = contextvars.ContextVar('apmtools_trace_span', default=None)
_lock = threading.RLock()
_session = uuid.uuid4().hex[:12]


def enabled():
    return bool(os.environ.get('APMTOOLS_TRACE_PATH'))


def _emit(event):
    path = os.environ.get('APMTOOLS_TRACE_PATH')
    if not path:
        return
    record = {'timestamp_ns': time.time_ns(), 'session': _session, **event}
    try:
        with _lock:
            destination = Path(path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, default=str, separators=(',', ':')) + '\n')
    except OSError:
        # Profiling must never interrupt the Shiny application.
        pass


def traced(label=None, *, kind='function', trigger=None):
    """Instrument a synchronous or asynchronous callable without altering its API."""
    def decorate(func):
        name = label or f'{func.__module__}.{func.__qualname__}'
        if hasattr(func, '__wrapped_by_apmtools_trace__'):
            return func
        if __import__('inspect').iscoroutinefunction(func):
            @functools.wraps(func)
            async def wrapped(*args, **kwargs):
                return await _run_async(func, name, kind, trigger, args, kwargs)
        else:
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                if not enabled():
                    return func(*args, **kwargs)
                parent = _current_span.get()
                span = uuid.uuid4().hex[:12]
                token = _current_span.set(span)
                start = time.perf_counter_ns()
                error = None
                try:
                    return func(*args, **kwargs)
                except BaseException as exc:
                    error = type(exc).__name__
                    raise
                finally:
                    duration = time.perf_counter_ns() - start
                    _current_span.reset(token)
                    _emit({'event': 'execution', 'kind': kind, 'name': name,
                           'trigger': trigger, 'span': span, 'parent': parent,
                           'duration_ms': round(duration / 1e6, 3), 'error': error})
        wrapped.__wrapped_by_apmtools_trace__ = True
        return wrapped
    return decorate


async def _run_async(func, name, kind, trigger, args, kwargs):
    if not enabled():
        return await func(*args, **kwargs)
    parent = _current_span.get()
    span = uuid.uuid4().hex[:12]
    token = _current_span.set(span)
    start = time.perf_counter_ns()
    error = None
    try:
        return await func(*args, **kwargs)
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        duration = time.perf_counter_ns() - start
        _current_span.reset(token)
        _emit({'event': 'execution', 'kind': kind, 'name': name,
               'trigger': trigger, 'span': span, 'parent': parent,
               'duration_ms': round(duration / 1e6, 3), 'error': error})


def marker(name, **metadata):
    """Mark a user interaction or application lifecycle event (no raw data)."""
    if enabled():
        _emit({'event': 'marker', 'name': name, 'parent': _current_span.get(), **metadata})
