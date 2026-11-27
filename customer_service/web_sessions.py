"""Multi-user Web SessionStore, sliding-window rate limiter, and zero-API interactive demo model."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import secrets
import shutil
from threading import RLock
import time

from .auth import AuthStore
from .live_session import LiveSession
from .observability import LangfuseReporter, SessionObserver
from .safeguards import SafeguardConfig
from .storage import SQLiteStore, initialize_database

_ORDER_RE = re.compile(r'\b([ABC]\d{4})\b', re.IGNORECASE)
_PRODUCT_RE = re.compile(r'\b(P10[0-5])\b', re.IGNORECASE)
_CONFIRM_ID_RE = re.compile(r'\b(cfm_[0-9a-f]{8})\b', re.IGNORECASE)


class SlidingWindowRateLimiter:
    """Thread-safe per-key sliding-window rate limiter (zero external dependencies)."""

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0, clock=None):
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = float(window_seconds)
        self._clock = time.monotonic if clock is None else clock
        self._buckets: dict[str, deque[float]] = {}
        self._lock = RLock()

    def check(self, key: str) -> tuple[bool, int, float]:
        now = float(self._clock())
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(0.1, self.window_seconds - (now - bucket[0]))
                return False, 0, round(retry_after, 2)
            bucket.append(now)
            remaining = max(0, self.max_requests - len(bucket))
            return True, remaining, 0.0


class DemoInteractiveModel:
    """Zero-API interactive model exercising real Shop/SQLite/Hybrid-RAG/Safeguard tools for local UI preview."""
    mode = 'offline-demo'

    def __init__(self):
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f'demo_call_{self._seq}'

    def _latest_tool_results(self, messages: list[dict]) -> list[tuple[str, dict]]:
        if not messages:
            return []
        last = messages[-1]
        content = last.get('content')
        if not isinstance(content, list):
            return []
        out = []
        for block in content:
            if isinstance(block, dict) and block.get('type') == 'tool_result':
                import json
                try:
                    parsed = json.loads(block['content'])
                except (ValueError, TypeError):
                    parsed = {'ok': False}
                out.append((block.get('tool_use_id', ''), parsed))
        return out

    def _current_user_text(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get('role') != 'user':
                continue
            content = msg.get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                if any(isinstance(b, dict) and b.get('type') == 'tool_result' for b in content):
                    continue
                parts = [b.get('text', '') for b in content if isinstance(b, dict) and b.get('type') == 'text']
                if parts:
                    return '\n'.join(parts)
        return ''

    def _find_pending_token(self, messages: list[dict]) -> tuple[str | None, str | None]:
        import json
        for msg in reversed(messages):
            content = msg.get('content')
            if not isinstance(content, list):
                continue
            for block in reversed(content):
                if isinstance(block, dict) and block.get('type') == 'text' and '<confirmation_state>' in block.get('text', ''):
                    text = block['text']
                    m_cid = _CONFIRM_ID_RE.search(text)
                    m_oid = _ORDER_RE.search(text)
                    if '"status": "approved"' in text and m_cid:
                        return m_cid.group(1), (m_oid.group(1).upper() if m_oid else None)
                if isinstance(block, dict) and block.get('type') == 'tool_result':
                    try:
                        parsed = json.loads(block['content'])
                    except (ValueError, TypeError):
                        continue
                    if not parsed.get('ok') and (parsed.get('error') or {}).get('code') == 'CONFIRMATION_REQUIRED':
                        data = parsed.get('data') or {}
                        return data.get('confirmation_id'), data.get('order_id')
        return None, None

    def complete(self, messages, tools, system):
        tool_names = {t['name'] for t in tools}
        latest_results = self._latest_tool_results(messages)
        user_text = self._current_user_text(messages)
        # Strip appended <confirmation_state> block for clean intent matching while keeping raw text
        clean_text = user_text.split('<confirmation_state>')[0].strip()

        if latest_results:
            return {
                'content': [{'type': 'text', 'text': self._format_results(latest_results, clean_text)}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 0, 'output_tokens': 0},
            }

        calls = self._route_tools(clean_text, user_text, messages, tool_names)
        if calls:
            return {
                'content': calls,
                'stop_reason': 'tool_use',
                'usage': {'input_tokens': 0, 'output_tokens': 0},
            }
        return {
            'content': [{'type': 'text', 'text': self._fallback_reply(clean_text)}],
            'stop_reason': 'end_turn',
            'usage': {'input_tokens': 0, 'output_tokens': 0},
        }

    def _route_tools(self, clean_text: str, raw_text: str, messages: list[dict], tool_names: set[str]) -> list[dict]:
        order_match = _ORDER_RE.search(clean_text)
        order_id = order_match.group(1).upper() if order_match else None
        product_match = _PRODUCT_RE.search(clean_text)
        product_id = product_match.group(1).upper() if product_match else None

        # 1. Explicit rejection of pending confirmation
        if any(k in clean_text for k in ('不要取消', '先不取消', '算了', '放弃取消', '拒绝')):
            return []

        # 2. Explicit confirmation of approved token
        if '<confirmation_state>' in raw_text and '"status": "approved"' in raw_text and 'cancel_order' in tool_names:
            cid, pending_oid = self._find_pending_token(messages)
            target_oid = order_id or pending_oid
            if cid and target_oid:
                return [{'type': 'tool_use', 'id': self._next_id(), 'name': 'cancel_order',
                         'input': {'order_id': target_oid, 'reason': '客户在工作台确认取消', 'confirmation_token': cid}}]

        calls = []
        # 3. Support ticket / human escalation
        if any(k in clean_text for k in ('转人工', '人工客服', '人工专员', '建工单', '创建工单', '投诉', '申诉')) and 'create_support_ticket' in tool_names:
            escalate = not any(k in clean_text for k in ('无需人工', '不用升级', '不需要转人工'))
            cat = 'human_escalation' if escalate else 'order_change'
            if '退款' in clean_text:
                cat = 'refund_dispute'
            elif '物流' in clean_text or '延误' in clean_text:
                cat = 'logistics_issue'
            payload = {
                'category': cat,
                'priority': 'high' if escalate else 'normal',
                'summary': f'工作台演示工单：{clean_text[:120]}',
                'escalate_to_human': escalate,
            }
            if order_id:
                payload['order_id'] = order_id
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'create_support_ticket', 'input': payload})
            return calls

        # 4. Cancel order request (step 1)
        if '取消' in clean_text and order_id and 'cancel_order' in tool_names:
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'cancel_order',
                          'input': {'order_id': order_id, 'reason': f'客户申请取消：{clean_text[:80]}'}})
            return calls

        # 5. Refund status
        if '退款' in clean_text and order_id and 'check_refund_status' in tool_names:
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'check_refund_status',
                          'input': {'order_id': order_id}})

        # 6. List tickets / orders
        if any(k in clean_text for k in ('工单列表', '我的工单', '历史工单')) and 'list_my_tickets' in tool_names:
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'list_my_tickets', 'input': {}})
        if any(k in clean_text for k in ('所有订单', '全部订单', '我的订单', '订单列表')) and 'list_my_orders' in tool_names:
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'list_my_orders', 'input': {}})

        # 7. Tracking / specific order
        if any(k in clean_text for k in ('物流', '快递', '签收', '延误')) and order_id and 'get_tracking' in tool_names:
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'get_tracking', 'input': {'order_id': order_id}})
        elif order_id and not calls and 'get_order' in tool_names:
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'get_order', 'input': {'order_id': order_id}})

        # 8. Product lookup
        if product_id and 'get_product' in tool_names:
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'get_product', 'input': {'product_id': product_id}})

        # 9. Policy lookup
        if any(k in clean_text for k in ('政策', '规则', '时效', '升级标准')) and 'get_policy' in tool_names:
            pid = 'returns'
            if '取消' in clean_text:
                pid = 'cancellation'
            elif '退款' in clean_text or '时效' in clean_text:
                pid = 'refund_timeline'
            elif '升级' in clean_text or '人工' in clean_text:
                pid = 'escalation'
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'get_policy', 'input': {'policy_id': pid}})

        # 10. Hybrid RAG knowledge query
        rag_cues = ('清洗', '清洁', '洗碗机', '钢丝球', '研磨粉', '初次', '渗水', '密封圈', '热饮', '微波炉',
                    '破损', '凹陷', '裂痕', '发票', '抬头', '重复扣款', '改地址', '优惠券', '礼盒', '贺卡', '配件', '拆单')
        if ('search_knowledge' in tool_names
                and (any(k in clean_text for k in rag_cues) or (not calls and len(clean_text) >= 4))):
            calls.append({'type': 'tool_use', 'id': self._next_id(), 'name': 'search_knowledge',
                          'input': {'query': clean_text[:200]}})
        return calls

    def _format_results(self, results: list[tuple[str, dict]], clean_text: str) -> str:
        sections = []
        for _, res in results:
            if res.get('ok'):
                data = res.get('data') or {}
                if 'product_id' in data and 'price_cents' in data:
                    sections.append(
                        f"**商品 {data['product_id']} · {data['name']}**\n"
                        f"- 售价：**{data['price_cents'] / 100:.2f} 元（CNY）**（原始值 `{data['price_cents']}` 分）\n"
                        f"- 容量：`{data['capacity_ml']} ml` · 来源：`{data.get('source', '')}`"
                    )
                elif 'policy_id' in data and 'text' in data:
                    sections.append(
                        f"**{data['title']}**（版本 `{data.get('version')}`，来源 `{data.get('source')}`）\n"
                        f"{data['text']}"
                    )
                elif 'orders' in data:
                    lines = [f"- `{o['order_id']}` · {o['status']} · **{o['total_cents'] / 100:.2f} 元**（`{o['product_id']}` × {o['quantity']}）"
                             for o in data['orders']]
                    sections.append('**本人订单列表（模拟 SQLite 实查）**\n' + '\n'.join(lines))
                elif 'order' in data and 'previous_status' in data:
                    o = data['order']
                    msg = f"订单 **`{o['order_id']}`** 已成功取消（`{data['previous_status']}` → `cancelled`）。"
                    rf = data.get('refund_created')
                    if rf:
                        msg += f"\n已自动生成原路退款单 **`{rf['refund_id']}`**，金额 **{rf['amount_cents'] / 100:.2f} 元**（状态 `{rf['status']}`，{rf['estimated_arrival']}）。"
                    sections.append(msg)
                elif 'order_id' in data and 'total_cents' in data:
                    sections.append(
                        f"**订单 `{data['order_id']}`**：状态 `{data['status']}`，商品 `{data['product_id']}` × {data['quantity']}，"
                        f"实付 **{data['total_cents'] / 100:.2f} 元**（`{data['total_cents']}` 分）。"
                    )
                elif 'tracking_id' in data and 'events' in data:
                    ev = '\n'.join(f"- `{e['time']}` · {e['detail']}" for e in data['events'])
                    sections.append(f"**物流单 `{data['tracking_id']}`**（状态 `{data['status']}`）：\n{ev}")
                elif 'refund' in data:
                    rf = data['refund']
                    if rf is None:
                        sections.append(data.get('message') or '该订单当前暂无退款记录。')
                    else:
                        sections.append(
                            f"**退款单 `{rf['refund_id']}`**（订单 `{rf['order_id']}`）\n"
                            f"- 状态：`{rf['status']}` · 金额：**{rf['amount_cents'] / 100:.2f} 元**\n"
                            f"- 预计到账：{rf.get('estimated_arrival')}\n"
                            f"- 审核说明：{rf.get('reviewer_note')}"
                        )
                elif 'ticket' in data:
                    tk = data['ticket']
                    sections.append(
                        f"已创建工单 **`{tk['ticket_id']}`**（队列 `{tk['assigned_queue']}`，"
                        f"人工升级：`{tk['escalate_to_human']}`，优先级 `{tk['priority']}`，状态 `{tk['status']}`）。"
                    )
                elif 'tickets' in data:
                    tks = data['tickets']
                    if not tks:
                        sections.append('当前登录账号暂无工单记录。')
                    else:
                        lines = [f"- `{t['ticket_id']}` · `{t['status']}` / `{t['priority']}` · 队列 `{t['assigned_queue']}` · {t['summary']}"
                                 for t in tks]
                        sections.append('**本人售后工单**\n' + '\n'.join(lines))
                elif 'hits' in data:
                    hits = data.get('hits') or []
                    ctx = data.get('context') or []
                    mode_tag = data.get('retrieval_mode', 'dense')
                    if not hits:
                        sections.append('公开知识库未检索到匹配资料（`no_match`），当前不编造未记录的规则。')
                    else:
                        lines = [f"- [{h['doc_id']}] `{h['chunk_id']}`（{h.get('matched_by', mode_tag)}）：{h['text']}" for h in hits[:2]]
                        if ctx:
                            lines.append(f"- [相邻原文] `{ctx[0]['chunk_id']}`：{ctx[0]['text']}")
                        sections.append(f'**公开知识库检索依据（{mode_tag.upper()} RRF）**\n' + '\n'.join(lines))
            else:
                err = res.get('error') or {}
                code = err.get('code', 'ERROR')
                extra = res.get('data') or {}
                if code == 'CONFIRMATION_REQUIRED':
                    oid = extra.get('order_id') or (extra.get('order') or {}).get('order_id')
                    cid = extra.get('confirmation_id')
                    amt = (extra.get('order') or {}).get('total_cents')
                    amt_txt = f'{amt / 100:.2f} 元' if isinstance(amt, int) else '见订单详情'
                    will_rf = '是（将自动生成原路退款单）' if extra.get('will_create_refund') else '否（待支付订单无需退款）'
                    sections.append(
                        f"**需要二次确认 · 订单 `{oid}`**\n"
                        f"- 确认码：`{cid}` · 订单金额：**{amt_txt}**\n"
                        f"- 是否自动创建退款单：{will_rf}\n"
                        f"- 请点击右侧/下方**「确认取消」**按钮，或直接回复「确认取消 {oid}」完成提交。"
                    )
                elif code == 'AUTHENTICATION_REQUIRED':
                    sections.append('当前为**访客（未登录）**状态，无法查询或修改私人订单。请先在左侧切换/登录模拟账号（如 `alice`、`bob`、`charlie`）。')
                elif code == 'ACCESS_DENIED_OR_NOT_FOUND':
                    sections.append('`ACCESS_DENIED_OR_NOT_FOUND`：该订单不存在或不属于当前登录客户，系统已拦截跨客户访问。')
                elif code == 'ORDER_NOT_CANCELLABLE':
                    sections.append(f"`ORDER_NOT_CANCELLABLE`：{err.get('message')} 若需进一步处理，可让我为您创建人工升级工单。")
                elif code == 'ORDER_ALREADY_CANCELLED':
                    sections.append(f"`ORDER_ALREADY_CANCELLED`：{err.get('message')}")
                else:
                    sections.append(f"`{code}`：{err.get('message')}")
        return '\n\n'.join(sections) if sections else self._fallback_reply(clean_text)

    def _fallback_reply(self, clean_text: str) -> str:
        if any(k in clean_text for k in ('不要取消', '先不取消', '算了', '放弃取消', '拒绝')):
            return '好的，已撤销待确认操作，您的订单状态保持不变。'
        return (
            '当前运行在**离线交互演示模式**（0 API 调用，全部工具均执行真实 SQLite 与本地 Hybrid RAG）。\n'
            '你可以试试问：`P104 多少钱`、`列出我的所有订单`、`取消未发货的 A1003`、`查 B2001`（验证跨客户隔离）、`保温杯能用钢丝球和洗碗机吗` 或 `A1001 延误太久请转人工`。'
        )


@dataclass
class ManagedSession:
    session_id: str
    live_session: LiveSession
    observer: SessionObserver
    lock: RLock
    created_at: float
    last_active: float


class WebSessionStore:
    """Thread-safe multi-user session registry with LRU eviction, TTL, and isolated DB snapshots."""

    def __init__(self, root: Path, model_factory, *, seed_path: Path, database_path: Path,
                 knowledge=None, allow_demo_switch: bool = True,
                 daily_token_budget: int = 500000, tool_timeout_seconds: float = 10.0,
                 max_sessions: int = 64, ttl_seconds: float = 1800.0,
                 langfuse: LangfuseReporter | None = None,
                 enable_subagents: bool = False):
        self.root = Path(root)
        self.model_factory = model_factory
        self.seed_path = Path(seed_path)
        self.database_path = Path(database_path)
        self.knowledge = knowledge
        self.allow_demo_switch = bool(allow_demo_switch)
        self.daily_token_budget = int(daily_token_budget)
        self.tool_timeout_seconds = float(tool_timeout_seconds)
        self.max_sessions = max(1, int(max_sessions))
        self.ttl_seconds = float(ttl_seconds)
        self.langfuse = langfuse
        self.enable_subagents = bool(enable_subagents)
        self.auth_store = AuthStore(self.root / 'data/auth')
        self._sessions: dict[str, ManagedSession] = {}
        self._lock = RLock()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        initialize_database(self.database_path, self.seed_path)

    def _evict_locked(self, now: float) -> None:
        expired = [sid for sid, item in self._sessions.items() if now - item.last_active > self.ttl_seconds]
        for sid in expired:
            item = self._sessions.pop(sid, None)
            if item is not None:
                item.live_session.close()
        while len(self._sessions) >= self.max_sessions:
            oldest_sid = min(self._sessions, key=lambda k: self._sessions[k].last_active)
            item = self._sessions.pop(oldest_sid, None)
            if item is not None:
                item.live_session.close()

    def create_session(self) -> ManagedSession:
        now = time.monotonic()
        with self._lock:
            self._evict_locked(now)
            sid = 'ws_' + secrets.token_hex(12)
            model = self.model_factory()
            safeguards = SafeguardConfig.production_defaults(
                budget_store_path=self.root / 'runtime/web_token_budget.json',
                daily_token_budget=self.daily_token_budget,
                tool_timeout_seconds=self.tool_timeout_seconds,
            )
            live = LiveSession(
                self.root,
                model,
                store='sqlite',
                knowledge=self.knowledge,
                allow_actions=True,
                safeguards=safeguards,
                data_path=self.seed_path,
                database_path=self.database_path,
                enable_subagents=self.enable_subagents,
            )
            observer = SessionObserver(langfuse=self.langfuse)
            observer.attach(live.harness)
            managed = ManagedSession(
                session_id=sid,
                live_session=live,
                observer=observer,
                lock=RLock(),
                created_at=now,
                last_active=now,
            )
            self._sessions[sid] = managed
            return managed

    def get(self, session_id: str) -> ManagedSession | None:
        now = time.monotonic()
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None:
                return None
            if now - item.last_active > self.ttl_seconds:
                self._sessions.pop(session_id, None)
                item.live_session.close()
                return None
            item.last_active = now
            return item

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def login(self, managed: ManagedSession, customer_id: str, password: str | None = None,
              *, demo_switch: bool = False) -> tuple[bool, str]:
        with managed.lock:
            cid = (customer_id or '').strip().lower()
            if cid not in ('alice', 'bob', 'charlie'):
                return False, '仅支持模拟客户账号：alice、bob 或 charlie。'
            if demo_switch:
                if not self.allow_demo_switch:
                    return False, '当前服务端未开启演示一键切换，请提供账号密码登录。'
                verifier = type('_DemoVerifier', (), {'verify': staticmethod(lambda c, _: c if c == cid else None)})()
                ok = managed.live_session.login(cid, '__demo_switch__', verifier)
            else:
                if not password:
                    return False, '请输入账号密码。'
                ok = managed.live_session.login(cid, password, self.auth_store)
            if ok:
                managed.observer.attach(managed.live_session.harness)
                return True, f'已验证并切换为模拟客户 {cid}。'
            return False, '密码校验未通过或该账号尚未通过 --init-auth 初始化本地密码。'

    def logout(self, managed: ManagedSession) -> None:
        with managed.lock:
            managed.live_session.logout()
            managed.observer.attach(managed.live_session.harness)

    def reset_demo_db(self) -> dict:
        """Reset the isolated web SQLite DB back to the clean seed state (never touches CLI shop.db)."""
        with self._lock:
            if self.database_path.name in ('shop.db', 'shop-actions.db'):
                raise RuntimeError('Refusing to reset primary CLI database')
            for suffix in ('', '-wal', '-shm'):
                candidate = Path(str(self.database_path) + suffix)
                if candidate.exists():
                    candidate.unlink()
            info = initialize_database(self.database_path, self.seed_path)
            # Rebuild every active session harness cleanly while preserving its bound identity
            for item in self._sessions.values():
                with item.lock:
                    cid = item.live_session.customer_id
                    item.live_session.logout()
                    if cid is not None and self.allow_demo_switch:
                        verifier = type('_ResetVerifier', (), {'verify': staticmethod(lambda c, _: c)})()
                        item.live_session.login(cid, '__reset__', verifier)
                    item.observer.attach(item.live_session.harness)
            return {'reset': True, 'dataset_version': info['dataset_version']}

    def snapshot_state(self, managed: ManagedSession) -> dict:
        with managed.lock:
            live = managed.live_session
            cid = live.customer_id
            store = SQLiteStore(self.database_path, allow_writes=False)
            products = store.list_products()
            orders = store.list_orders(cid) if cid else []
            refunds = store.list_refunds(cid) if cid else []
            tickets = store.list_tickets(cid) if cid else []
            pending = live.pending_confirmations() if live.harness is not None else []
            budget = live.budget_status()
            return {
                'session_id': managed.session_id,
                'customer_id': cid,
                'is_authenticated': live.is_authenticated,
                'allow_demo_switch': self.allow_demo_switch,
                'model_mode': getattr(live.model, 'mode', 'unknown'),
                'retrieval_mode': getattr(self.knowledge, 'retrieval_mode', None) if self.knowledge is not None else None,
                'dataset_version': store.dataset_version,
                'orders': orders,
                'refunds': refunds,
                'tickets': tickets,
                'products': products,
                'pending_confirmations': pending,
                'budget': budget,
                'traces': managed.observer.recent_entries(30),
                'recorded_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }

    def close_all(self) -> None:
        with self._lock:
            for item in self._sessions.values():
                item.live_session.close()
            self._sessions.clear()
