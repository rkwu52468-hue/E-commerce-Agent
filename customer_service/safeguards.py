"""Production safeguards: retries, tool timeouts, loop breaker, budget caps, and 2-step confirmation."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import queue
import re
import secrets
import threading
import time

from .errors import is_context_overflow

SHANGHAI_TZ = timezone(timedelta(hours=8), name='Asia/Shanghai')

TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504, 529})
TRANSIENT_ERROR_NAMES = frozenset({
    'APIConnectionError',
    'APITimeoutError',
    'RateLimitError',
    'InternalServerError',
    'ServiceUnavailableError',
})

_CONFIRM_ID_RE = re.compile(r'\b(cfm_[0-9a-f]{8})\b', re.IGNORECASE)
_NEGATIVE_PHRASES = (
    '不要取消', '别取消', '不取消', '先不取消', '算了', '放弃', '拒绝', '撤销确认',
    'do not cancel', "don't cancel", 'reject', 'abort',
)
_POSITIVE_PHRASES = (
    '确认取消', '确定取消', '是的取消', '好的取消', '同意取消', '确认执行', '确认操作',
    '确认', '确定', '同意', '是的', '好，取消', '好的，取消',
    'confirm', 'yes, cancel', 'yes cancel', 'proceed', 'approved',
)


@dataclass(frozen=True)
class SafeguardConfig:
    """Opt-in runtime safeguards for action-capable / public-facing sessions."""
    model_max_retries: int = 2
    retry_base_delay: float = 0.5
    retry_max_delay: float = 4.0
    tool_timeout_seconds: float | None = 10.0
    max_identical_tool_calls: int | None = 2
    max_consecutive_tool_failures: int | None = 3
    max_turn_tokens: int | None = 45000
    max_session_tokens: int | None = 200000
    daily_token_budget: int | None = 500000
    budget_store_path: Path | None = None
    confirmation_ttl_seconds: float = 300.0
    allow_chat_confirmation: bool = True
    cache_safe_system: bool = True

    @classmethod
    def production_defaults(cls, budget_store_path: Path | None = None,
                            daily_token_budget: int = 500000,
                            tool_timeout_seconds: float = 10.0,
                            max_turn_tokens: int = 45000,
                            max_session_tokens: int = 200000,
                            cache_safe_system: bool = True) -> 'SafeguardConfig':
        return cls(
            model_max_retries=2,
            retry_base_delay=0.5,
            retry_max_delay=4.0,
            tool_timeout_seconds=tool_timeout_seconds,
            max_identical_tool_calls=2,
            max_consecutive_tool_failures=3,
            max_turn_tokens=max_turn_tokens,
            max_session_tokens=max_session_tokens,
            daily_token_budget=daily_token_budget,
            budget_store_path=budget_store_path,
            confirmation_ttl_seconds=300.0,
            allow_chat_confirmation=True,
            cache_safe_system=cache_safe_system,
        )


def is_transient_model_error(error) -> bool:
    """Return True only for retryable network/rate-limit/5xx model failures."""
    if is_context_overflow(error):
        return False
    status = getattr(error, 'status_code', None)
    if isinstance(status, int):
        return status in TRANSIENT_STATUS_CODES
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    return type(error).__name__ in TRANSIENT_ERROR_NAMES


def call_with_transient_retry(fn, max_retries=2, base_delay=0.5, max_delay=4.0,
                              sleep_fn=None, on_retry=None):
    """Execute fn with bounded exponential backoff on transient model errors only."""
    sleeper = time.sleep if sleep_fn is None else sleep_fn
    attempts = 0
    while True:
        try:
            return fn()
        except Exception as error:
            if attempts >= max_retries or not is_transient_model_error(error):
                raise
            attempts += 1
            delay = min(max_delay, base_delay * (2 ** (attempts - 1)))
            if on_retry is not None:
                on_retry(attempts, delay, error)
            if delay > 0:
                sleeper(delay)


def run_with_timeout(fn, timeout_seconds):
    """Run fn in a daemon worker thread and raise TimeoutError if it exceeds timeout_seconds."""
    if timeout_seconds is None or timeout_seconds <= 0:
        return fn()
    outcome = queue.Queue(maxsize=1)

    def target():
        try:
            outcome.put(('ok', fn()))
        except BaseException as exc:
            outcome.put(('err', exc))

    worker = threading.Thread(target=target, daemon=True, name='tool-timeout-guard')
    worker.start()
    try:
        kind, value = outcome.get(timeout=timeout_seconds)
    except queue.Empty:
        raise TimeoutError(f'Tool execution exceeded {timeout_seconds}s timeout') from None
    if kind == 'err':
        raise value
    return value


def canonical_tool_key(name, args) -> str:
    """Stable key identifying an exact (tool_name, arguments) invocation."""
    try:
        encoded = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    except (TypeError, ValueError):
        encoded = repr(args)
    return f'{name}:{encoded}'


class TurnLoopGuard:
    """Prevents repeated identical tool calls or consecutive tool failure loops within one turn."""

    def __init__(self, max_identical_calls=2, max_consecutive_failures=3):
        self.max_identical_calls = max_identical_calls
        self.max_consecutive_failures = max_consecutive_failures
        self._call_counts = {}
        self._consecutive_failures = {}

    def reset(self):
        self._call_counts.clear()
        self._consecutive_failures.clear()

    def check(self, name, args):
        """Return None if allowed, or (reason, message) if circuit-broken."""
        key = canonical_tool_key(name, args)
        if self.max_identical_calls is not None and self._call_counts.get(key, 0) >= self.max_identical_calls:
            return (
                'identical_call_limit',
                '检测到同一工具与相同参数重复调用已达上限，已触发防死循环熔断。'
                '请基于已有工具结果直接回复客户，或调用 create_support_ticket 转人工处理。',
            )
        if (self.max_consecutive_failures is not None
                and self._consecutive_failures.get(name, 0) >= self.max_consecutive_failures):
            return (
                'consecutive_failure_limit',
                '检测到该工具已连续多次执行失败，已触发防死循环熔断。'
                '请停止重试，向客户说明现状或调用 create_support_ticket 转人工处理。',
            )
        return None

    def record(self, name, args, ok: bool):
        key = canonical_tool_key(name, args)
        self._call_counts[key] = self._call_counts.get(key, 0) + 1
        if ok:
            self._consecutive_failures[name] = 0
        else:
            self._consecutive_failures[name] = self._consecutive_failures.get(name, 0) + 1


def _nonneg_int(value) -> int:
    return value if type(value) is int and value > 0 else 0


def extract_usage_breakdown(usage) -> dict:
    """Extract uncached input, cache read/creation input, output, and total tokens."""
    if not isinstance(usage, dict):
        return {
            'input_tokens': 0,
            'cache_read_input_tokens': 0,
            'cache_creation_input_tokens': 0,
            'total_input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
        }
    inp = _nonneg_int(usage.get('input_tokens'))
    cache_read = _nonneg_int(usage.get('cache_read_input_tokens'))
    cache_create = _nonneg_int(usage.get('cache_creation_input_tokens'))
    out = _nonneg_int(usage.get('output_tokens'))
    total_input = inp + cache_read + cache_create
    return {
        'input_tokens': inp,
        'cache_read_input_tokens': cache_read,
        'cache_creation_input_tokens': cache_create,
        'total_input_tokens': total_input,
        'output_tokens': out,
        'total_tokens': total_input + out,
    }


def extract_usage_tokens(usage) -> tuple[int, int, int]:
    """Return (total_input_tokens, output_tokens, total_tokens) from a usage dict."""
    parts = extract_usage_breakdown(usage)
    return parts['total_input_tokens'], parts['output_tokens'], parts['total_tokens']


def _cache_hit_rate(cache_read: int, total_input: int) -> float | None:
    if total_input <= 0:
        return None
    return round(cache_read / total_input, 4)


class DailyTokenBudgetStore:
    """Thread-safe persistent daily token counter stored as atomic JSON."""

    def __init__(self, path: Path, clock=None):
        self.path = Path(path).resolve()
        self._clock = clock or (lambda: datetime.now(SHANGHAI_TZ))
        self._lock = threading.RLock()

    def _today(self) -> str:
        return self._clock().astimezone(SHANGHAI_TZ).strftime('%Y-%m-%d')

    def _load_locked(self) -> dict:
        if not self.path.exists():
            return {'days': {}}
        try:
            raw = json.loads(self.path.read_text(encoding='utf-8'))
            if isinstance(raw, dict) and isinstance(raw.get('days'), dict):
                return raw
        except (OSError, ValueError):
            pass
        return {'days': {}}

    def _save_locked(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name('.' + self.path.name + '.' + secrets.token_hex(8) + '.tmp')
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(self.path)

    @staticmethod
    def _normalize_entry(raw) -> dict:
        entry = raw if isinstance(raw, dict) else {}
        inp = _nonneg_int(entry.get('input_tokens'))
        cache_read = _nonneg_int(entry.get('cache_read_input_tokens'))
        cache_create = _nonneg_int(entry.get('cache_creation_input_tokens'))
        out = _nonneg_int(entry.get('output_tokens'))
        requests = _nonneg_int(entry.get('requests'))
        total_input = inp + cache_read + cache_create
        return {
            'input_tokens': inp,
            'cache_read_input_tokens': cache_read,
            'cache_creation_input_tokens': cache_create,
            'total_input_tokens': total_input,
            'output_tokens': out,
            'total_tokens': total_input + out,
            'cache_hit_rate': _cache_hit_rate(cache_read, total_input),
            'requests': requests,
        }

    def today_usage(self) -> dict:
        with self._lock:
            day = self._today()
            data = self._load_locked()
            entry = self._normalize_entry(data['days'].get(day))
            return {'date': day, **entry}

    def add(self, input_tokens: int, output_tokens: int,
            cache_read_input_tokens: int = 0, cache_creation_input_tokens: int = 0) -> dict:
        with self._lock:
            day = self._today()
            data = self._load_locked()
            entry = self._normalize_entry(data['days'].get(day))
            entry['input_tokens'] += max(0, int(input_tokens))
            entry['cache_read_input_tokens'] += max(0, int(cache_read_input_tokens))
            entry['cache_creation_input_tokens'] += max(0, int(cache_creation_input_tokens))
            entry['total_input_tokens'] = (
                entry['input_tokens']
                + entry['cache_read_input_tokens']
                + entry['cache_creation_input_tokens']
            )
            entry['output_tokens'] += max(0, int(output_tokens))
            entry['total_tokens'] = entry['total_input_tokens'] + entry['output_tokens']
            entry['cache_hit_rate'] = _cache_hit_rate(
                entry['cache_read_input_tokens'], entry['total_input_tokens']
            )
            entry['requests'] += 1
            data['days'][day] = entry
            self._save_locked(data)
            return {'date': day, **entry}


class TokenBudgetTracker:
    """Enforces per-turn, per-session, and optional global daily token limits."""

    def __init__(self, max_turn_tokens=None, max_session_tokens=None,
                 daily_token_budget=None, budget_store: DailyTokenBudgetStore | None = None):
        self.max_turn_tokens = max_turn_tokens
        self.max_session_tokens = max_session_tokens
        self.daily_token_budget = daily_token_budget
        self.budget_store = budget_store
        self.turn_tokens = 0
        self.session_tokens = 0
        self.session_input_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_creation_tokens = 0
        self.session_output_tokens = 0

    def begin_turn(self):
        self.turn_tokens = 0

    def check(self):
        """Return None if within budget, or (scope, current, limit) if exceeded."""
        if self.max_turn_tokens is not None and self.turn_tokens >= self.max_turn_tokens:
            return ('turn_token_limit', self.turn_tokens, self.max_turn_tokens)
        if self.max_session_tokens is not None and self.session_tokens >= self.max_session_tokens:
            return ('session_token_limit', self.session_tokens, self.max_session_tokens)
        if self.daily_token_budget is not None and self.budget_store is not None:
            daily = self.budget_store.today_usage()['total_tokens']
            if daily >= self.daily_token_budget:
                return ('daily_token_budget', daily, self.daily_token_budget)
        return None

    def record(self, usage):
        parts = extract_usage_breakdown(usage)
        self.turn_tokens += parts['total_tokens']
        self.session_tokens += parts['total_tokens']
        self.session_input_tokens += parts['input_tokens']
        self.session_cache_read_tokens += parts['cache_read_input_tokens']
        self.session_cache_creation_tokens += parts['cache_creation_input_tokens']
        self.session_output_tokens += parts['output_tokens']
        if self.budget_store is not None and parts['total_tokens'] > 0:
            self.budget_store.add(
                parts['input_tokens'],
                parts['output_tokens'],
                cache_read_input_tokens=parts['cache_read_input_tokens'],
                cache_creation_input_tokens=parts['cache_creation_input_tokens'],
            )

    def status(self) -> dict:
        total_input = (
            self.session_input_tokens
            + self.session_cache_read_tokens
            + self.session_cache_creation_tokens
        )
        info = {
            'turn_tokens': self.turn_tokens,
            'max_turn_tokens': self.max_turn_tokens,
            'session_tokens': self.session_tokens,
            'session_input_tokens': self.session_input_tokens,
            'session_cache_read_tokens': self.session_cache_read_tokens,
            'session_cache_creation_tokens': self.session_cache_creation_tokens,
            'session_total_input_tokens': total_input,
            'session_output_tokens': self.session_output_tokens,
            'session_cache_hit_rate': _cache_hit_rate(self.session_cache_read_tokens, total_input),
            'max_session_tokens': self.max_session_tokens,
            'daily_token_budget': self.daily_token_budget,
        }
        if self.budget_store is not None:
            info['daily_usage'] = self.budget_store.today_usage()
        return info


class PendingConfirmationGate:
    """Two-phase programmatic confirmation gate for high-risk write tools.

    A model requesting a write action without an approved token gets a pending
    confirmation_id. The model CANNOT self-approve in the same turn; approval
    must come either from an explicit local command/UI action (approve()) or
    from a subsequent customer turn when allow_chat_confirmation is enabled.
    """

    def __init__(self, session_id: str, customer_id: str | None,
                 ttl_seconds: float = 300.0, allow_chat_confirmation: bool = True,
                 clock=None):
        self.session_id = session_id
        self.customer_id = customer_id
        self.ttl_seconds = float(ttl_seconds)
        self.allow_chat_confirmation = bool(allow_chat_confirmation)
        self._clock = clock or time.monotonic
        self._pending = {}
        self._consumed = set()

    def clear(self):
        self._pending.clear()

    def _prune_expired(self):
        now = self._clock()
        expired = [cid for cid, item in self._pending.items() if now > item['expires_at']]
        for cid in expired:
            del self._pending[cid]

    def create(self, action: str, order_id: str, reason: str, preview: dict, current_turn: int) -> dict:
        self._prune_expired()
        # Reuse existing unconsumed pending confirmation for the exact same action & order.
        for item in self._pending.values():
            if item['action'] == action and item['order_id'] == order_id and item['status'] == 'pending':
                item['reason'] = reason
                item['preview'] = dict(preview)
                item['expires_at'] = self._clock() + self.ttl_seconds
                return dict(item)
        cid = 'cfm_' + secrets.token_hex(4)
        record = {
            'confirmation_id': cid,
            'session_id': self.session_id,
            'customer_id': self.customer_id,
            'action': action,
            'order_id': order_id,
            'reason': reason,
            'preview': dict(preview),
            'created_turn': current_turn,
            'status': 'pending',
            'approved_via': None,
            'expires_at': self._clock() + self.ttl_seconds,
        }
        self._pending[cid] = record
        return dict(record)

    def list_pending(self) -> list[dict]:
        self._prune_expired()
        return [
            {k: v for k, v in item.items() if k != 'expires_at'}
            for item in self._pending.values()
        ]

    def approve(self, confirmation_id: str, via: str = 'explicit_command') -> dict | None:
        self._prune_expired()
        record = self._pending.get(confirmation_id)
        if record is None or record['customer_id'] != self.customer_id:
            return None
        record['status'] = 'approved'
        record['approved_via'] = via
        return {k: v for k, v in record.items() if k != 'expires_at'}

    def reject(self, confirmation_id: str) -> bool:
        self._prune_expired()
        return self._pending.pop(confirmation_id, None) is not None

    def process_user_reply(self, query: str, current_turn: int) -> list[dict]:
        """Inspect a new customer turn to approve/reject prior-turn pending confirmations."""
        self._prune_expired()
        if not self.allow_chat_confirmation or not isinstance(query, str):
            return []
        text = query.strip()
        lower = text.lower()
        prior = [item for item in self._pending.values()
                 if item['status'] == 'pending' and item['created_turn'] < current_turn]
        if not prior:
            return []
        changes = []
        mentioned_ids = [m.group(1).lower() for m in _CONFIRM_ID_RE.finditer(text)]
        targets = [item for item in prior if item['confirmation_id'] in mentioned_ids] if mentioned_ids else (
            prior if len(prior) == 1 else []
        )
        if not targets:
            return []
        if any(phrase in lower for phrase in _NEGATIVE_PHRASES):
            for item in targets:
                self._pending.pop(item['confirmation_id'], None)
                changes.append({'confirmation_id': item['confirmation_id'], 'order_id': item['order_id'], 'decision': 'rejected'})
            return changes
        has_positive = any(phrase in lower for phrase in _POSITIVE_PHRASES) or bool(mentioned_ids)
        if has_positive:
            for item in targets:
                item['status'] = 'approved'
                item['approved_via'] = 'user_chat_reply'
                changes.append({'confirmation_id': item['confirmation_id'], 'order_id': item['order_id'], 'decision': 'approved'})
        return changes

    def consume_if_approved(self, confirmation_id: str, action: str, order_id: str,
                            current_turn: int) -> tuple[bool, str, dict | None]:
        """Validate and atomically consume an approved confirmation token."""
        self._prune_expired()
        if not isinstance(confirmation_id, str) or not confirmation_id.strip():
            return False, 'INVALID_CONFIRMATION_TOKEN', None
        token = confirmation_id.strip().lower()
        if token in self._consumed:
            return False, 'CONFIRMATION_ALREADY_USED', None
        record = self._pending.get(token)
        if record is None:
            return False, 'INVALID_CONFIRMATION_TOKEN', None
        if record['customer_id'] != self.customer_id or record['action'] != action or record['order_id'] != order_id:
            return False, 'CONFIRMATION_MISMATCH', None
        if record['status'] != 'approved':
            if record['created_turn'] == current_turn:
                return False, 'CONFIRMATION_PENDING_USER_REPLY', None
            return False, 'CONFIRMATION_NOT_APPROVED', None
        # Single-use: consume before returning.
        self._pending.pop(token, None)
        self._consumed.add(token)
        return True, 'OK', {k: v for k, v in record.items() if k != 'expires_at'}
