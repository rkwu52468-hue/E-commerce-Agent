"""s04 adaptation: per-session observation callbacks, never an authorization layer.

The fixed upstream uses ordered callbacks with a non-None short circuit.
This business adaptation keeps the four lifecycle points and registration order,
but requires None returns and isolates failures so observers cannot grant access,
rewrite requests/results, or make Stop issue another model request. These are
trusted project callbacks in one Python process, not a plugin security sandbox.
"""
import copy

EVENTS = ('UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop')


class HookRegistry:
    def __init__(self):
        self._callbacks = {event: [] for event in EVENTS}

    def register(self, event, callback):
        if event not in self._callbacks:
            raise ValueError('Unknown hook event')
        if not callable(callback):
            raise TypeError('Hook callback must be callable')
        self._callbacks[event].append(callback)

    def trigger(self, event, payload):
        if event not in self._callbacks:
            raise ValueError('Unknown hook event')
        if not isinstance(payload, dict):
            raise TypeError('Hook metadata must be a dictionary')
        failures = []
        # Snapshot the registered callbacks and isolate each metadata copy.
        for index, callback in enumerate(tuple(self._callbacks[event])):
            try:
                if callback(copy.deepcopy(payload)) is not None:
                    raise TypeError('Observation callbacks must return None')
            except Exception as error:
                failures.append({'callback_index': index,
                                 'error_type': type(error).__name__})
        return failures
