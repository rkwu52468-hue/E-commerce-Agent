"""Observability hooks: in-process trace ring buffer, SSE live bridge, and optional Langfuse exporter.

Strictly follows HookRegistry's observation contract:
- Every callback returns None and never mutates control flow or authorization.
- Records metadata only (tool name, validated resource_id, status, duration_ms, token counts),
  never raw passwords, secrets, or unredacted private payload bodies.
- Langfuse export is optional and fail-safe: disabled unless explicitly configured, and any
  exporter failure is silently isolated so user chat turns never fail.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import os
from threading import RLock
from typing import Callable


class LangfuseReporter:
    """Optional Langfuse trace/span bridge (fail-safe; supports injected client for tests)."""

    def __init__(self, client=None):
        self._client = client
        self.enabled = client is not None
        if self._client is None:
            public_key = os.environ.get('LANGFUSE_PUBLIC_KEY', '').strip()
            secret_key = os.environ.get('LANGFUSE_SECRET_KEY', '').strip()
            host = os.environ.get('LANGFUSE_HOST', '').strip() or None
            if public_key and secret_key:
                try:
                    from langfuse import Langfuse
                    kwargs = {'public_key': public_key, 'secret_key': secret_key}
                    if host:
                        kwargs['host'] = host
                    self._client = Langfuse(**kwargs)
                    self.enabled = True
                except Exception:
                    self._client = None
                    self.enabled = False
        self._traces = {}
        self._lock = RLock()

    def on_event(self, event_name: str, payload: dict) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            key = (payload.get('session_id'), payload.get('turn'))
            with self._lock:
                if event_name == 'UserPromptSubmit':
                    trace = self._client.trace(
                        name='customer_service_turn',
                        session_id=payload.get('session_id'),
                        user_id=payload.get('customer_id') or 'guest',
                        metadata={'turn': payload.get('turn')},
                    )
                    self._traces[key] = trace
                elif event_name == 'PostToolUse':
                    trace = self._traces.get(key)
                    if trace is not None and hasattr(trace, 'span'):
                        trace.span(
                            name=str(payload.get('tool') or 'tool'),
                            metadata={
                                'tool_use_id': payload.get('tool_use_id'),
                                'resource_id': payload.get('resource_id'),
                                'ok': payload.get('ok'),
                                'error_code': payload.get('error_code'),
                                'duration_ms': payload.get('duration_ms'),
                            },
                        )
                elif event_name == 'Stop':
                    trace = self._traces.pop(key, None)
                    if trace is not None and hasattr(trace, 'update'):
                        trace.update(
                            metadata={
                                'status': payload.get('status'),
                                'reason': payload.get('reason'),
                                'tool_calls': payload.get('tool_calls'),
                                'steps': payload.get('steps'),
                            }
                        )
                    if hasattr(self._client, 'flush'):
                        self._client.flush()
        except Exception:
            # Never break business replies because of external telemetry faults.
            return


class SessionObserver:
    """Attaches metadata-only hooks to a Harness for UI inspection, SSE live events, and Langfuse."""

    def __init__(self, *, max_entries: int = 80, langfuse: LangfuseReporter | None = None):
        self._entries = deque(maxlen=max_entries)
        self._subscribers: list[Callable[[dict], None]] = []
        self._lock = RLock()
        self.langfuse = langfuse

    def attach(self, harness) -> None:
        for event_name in ('UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop'):
            harness.hooks.register(event_name, self._make_callback(event_name))

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)

        def unsubscribe():
            with self._lock:
                if callback in self._subscribers:
                    self._subscribers.remove(callback)

        return unsubscribe

    def recent_entries(self, limit: int = 30) -> list[dict]:
        with self._lock:
            items = list(self._entries)
        return items[-limit:] if limit > 0 else []

    def _make_callback(self, event_name: str):
        def callback(payload: dict):
            entry = {
                'time_utc': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                'event': event_name,
                'session_id': payload.get('session_id'),
                'customer_id': payload.get('customer_id'),
                'turn': payload.get('turn'),
            }
            for key in ('tool', 'tool_use_id', 'resource_id', 'ok', 'error_code',
                        'duration_ms', 'status', 'reason', 'steps', 'tool_calls'):
                if key in payload and payload[key] is not None:
                    entry[key] = payload[key]
            with self._lock:
                self._entries.append(entry)
                subscribers = list(self._subscribers)
            for sub in subscribers:
                try:
                    sub(dict(entry))
                except Exception:
                    pass
            if self.langfuse is not None:
                self.langfuse.on_event(event_name, payload)
            return None

        return callback
