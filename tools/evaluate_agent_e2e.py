"""100-question End-to-End Agent Evaluation Suite (offline protocol check + live DeepSeek eval).

Measures the 4 core customer-support KPIs overall and across 5 categories:
1. Tool Call Accuracy (expected tools, forbidden tools, error codes, RAG doc hit)
2. Task Resolution Rate (completion + tools + real SQLite DB state + facts + zero violations + escalation)
3. Hallucination / Policy-Violation Count & Rate (forbidden claims & canary leaks)
4. Human Escalation Accuracy / Precision / Recall

Honesty contract:
- `--mode offline` runs a deterministic scripted protocol driver over the REAL Harness,
  Shop, SQLiteStore, Safeguards, and Hybrid KnowledgeSearch with 0 LLM API calls.
  It verifies the dataset, tools, DB state machine, RAG retrieval, and scoring pipeline,
  and MUST NOT be quoted as real LLM task accuracy.
- `--mode live` runs real DeepSeek API calls via AnthropicModel and records true model
  accuracy, latency, token usage, and Prompt Cache hit rate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from time import perf_counter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from customer_service.auth import read_secret
from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.knowledge import KnowledgeSearch
from customer_service.models import AnthropicModel
from customer_service.safeguards import (
    DailyTokenBudgetStore,
    SafeguardConfig,
    extract_usage_breakdown,
)
from customer_service.shop import Shop
from customer_service.storage import SQLiteStore, initialize_database

DATASET_PATH = ROOT / 'data/eval_agent_100.json'
SEED_PATH = ROOT / 'data/shop-action-seed.json'
KNOWLEDGE_DOCS_PATH = ROOT / 'data/knowledge_demo/documents.json'
KNOWLEDGE_INDEX_PATH = ROOT / 'runtime/knowledge_demo'

_ORDER_RE = re.compile(r'\b([ABC]\d{4})\b')
_PRODUCT_RE = re.compile(r'\b(P10[0-5])\b')
_POLICY_KEYWORDS = (
    ('cancellation', ('取消', 'cancel')),
    ('refund_timeline', ('退款时效', '到账', '打款')),
    ('escalation', ('升级标准', '转人工')),
    ('returns', ('退换货', '退货')),
)
_CANARY_RE = re.compile(r'CANARY_[A-Z0-9_]+')


def _extract_order_id(text: str) -> str | None:
    match = _ORDER_RE.search(text)
    return match.group(1) if match else None


def _extract_product_id(text: str) -> str | None:
    match = _PRODUCT_RE.search(text)
    return match.group(1) if match else None


def _extract_policy_id(text: str) -> str:
    for policy_id, keywords in _POLICY_KEYWORDS:
        if any(word in text for word in keywords):
            return policy_id
    return 'returns'


def _latest_tool_results(messages: list[dict]) -> list[tuple[str, dict]]:
    """Return tool results from the most recent user message if it contains tool_result blocks."""
    if not messages:
        return []
    last = messages[-1]
    content = last.get('content')
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if isinstance(block, dict) and block.get('type') == 'tool_result':
            try:
                parsed = json.loads(block['content'])
            except (ValueError, TypeError):
                parsed = {'ok': False, 'raw': block.get('content')}
            out.append((block.get('tool_use_id', ''), parsed))
    return out


def _all_tool_calls_and_results(messages: list[dict]) -> list[dict]:
    """Extract ordered tool call + paired tool result records from harness messages."""
    by_id = {}
    ordered = []
    for msg in messages:
        content = msg.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get('type')
            if btype == 'tool_use':
                rec = {
                    'id': block.get('id'),
                    'name': block.get('name'),
                    'input': block.get('input', {}),
                    'result': None,
                }
                by_id[rec['id']] = rec
                ordered.append(rec)
            elif btype == 'tool_result':
                rec = by_id.get(block.get('tool_use_id'))
                if rec is not None:
                    try:
                        rec['result'] = json.loads(block['content'])
                    except (ValueError, TypeError):
                        rec['result'] = {'ok': False, 'raw': block.get('content')}
    return ordered


def _extract_confirmation_token_from_messages(messages: list[dict]) -> str | None:
    for rec in reversed(_all_tool_calls_and_results(messages)):
        res = rec.get('result') or {}
        if not res.get('ok') and res.get('error', {}).get('code') == 'CONFIRMATION_REQUIRED':
            cid = (res.get('data') or {}).get('confirmation_id')
            if cid:
                return cid
    return None


class OfflineProtocolModel:
    """Deterministic protocol driver exercising the real Harness/Shop/SQLite/RAG stack (0 API calls)."""
    mode = 'offline-protocol-driver'

    def __init__(self, case: dict):
        self.case = case
        self.call_count = 0
        self._turn_index = -1
        self._step_in_turn = 0

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

    def _summarize_tool_results(self, results: list[tuple[str, dict]], user_text: str) -> str:
        parts = []
        for _, res in results:
            if res.get('ok'):
                data = res.get('data', {})
                if 'product_id' in data and 'price_cents' in data:
                    p = data
                    parts.append(f"商品 {p['product_id']}（{p['name']}），售价 {p['price_cents'] / 100:.2f} 元（{p['price_cents']} 分），容量 {p['capacity_ml']} ml。")
                elif 'policy_id' in data and 'text' in data:
                    pol = data
                    parts.append(f"规则《{pol['title']}》：{pol['text']}")
                elif 'orders' in data:
                    lines = [f"{o['order_id']} 状态 {o['status']} 实付 {o['total_cents'] / 100:.2f} 元（{o['total_cents']} 分）"
                             for o in data['orders']]
                    parts.append('本人订单：' + '；'.join(lines))
                elif 'order' in data and 'previous_status' in data:
                    o = data['order']
                    msg = f"订单 {o['order_id']} 已取消（状态 {o['status']}）。"
                    rf = data.get('refund_created')
                    if rf:
                        msg += f" 已自动生成退款单 {rf['refund_id']}，金额 {rf['amount_cents'] / 100:.2f} 元（{rf['amount_cents']} 分）。"
                    parts.append(msg)
                elif 'order_id' in data and 'total_cents' in data:
                    o = data
                    parts.append(f"订单 {o['order_id']} 状态 {o['status']}，实付 {o['total_cents'] / 100:.2f} 元（{o['total_cents']} 分）。")
                elif 'tracking_id' in data and 'events' in data:
                    tr = data
                    events_text = '；'.join(f"{e.get('time', '')} {e.get('detail', '')}" for e in tr.get('events', []))
                    parts.append(f"物流单号 {tr.get('tracking_id')}，状态 {tr.get('status')}：{events_text}。暂无可承诺的送达时间，不能承诺赔付。")
                elif 'refund' in data:
                    rf = data['refund']
                    if rf is None:
                        parts.append(data.get('message') or '该订单当前暂无退款记录。')
                    else:
                        parts.append(
                            f"退款单 {rf['refund_id']}，订单 {rf['order_id']}，状态 {rf['status']}，"
                            f"金额 {rf['amount_cents'] / 100:.2f} 元（{rf['amount_cents']} 分），"
                            f"预计：{rf.get('estimated_arrival', '')}，说明：{rf.get('reviewer_note', '')}"
                        )
                elif 'ticket' in data:
                    tk = data['ticket']
                    parts.append(
                        f"已创建工单 {tk['ticket_id']}（订单 {tk.get('order_id') or '-'}，队列 {tk['assigned_queue']}，"
                        f"人工升级={tk['escalate_to_human']}，状态 {tk['status']}）。"
                    )
                elif 'tickets' in data:
                    tks = data['tickets']
                    if not tks:
                        parts.append('当前账号暂无历史工单记录（0 条）。')
                    else:
                        summary = '；'.join(f"{t['ticket_id']} 订单 {t.get('order_id')} 状态 {t['status']}" for t in tks)
                        parts.append('工单列表：' + summary)
                elif 'hits' in data:
                    snippets = [h['text'] for h in data.get('hits', [])] + [c['text'] for c in data.get('context', [])]
                    if snippets:
                        parts.append('知识库检索依据：' + ' '.join(snippets))
                    else:
                        parts.append('公开知识库未找到对应有效资料，仅以当前有效7天退货公开政策为准。')
            else:
                err = res.get('error', {})
                code = err.get('code', 'ERROR')
                msg = err.get('message', '')
                extra = res.get('data') or {}
                if code == 'CONFIRMATION_REQUIRED':
                    oid = extra.get('order_id') or (extra.get('order') or {}).get('order_id') or _extract_order_id(user_text)
                    status = extra.get('current_status') or (extra.get('order') or {}).get('status')
                    parts.append(
                        f"订单 {oid} 当前状态 {status}，待二次确认，"
                        f"确认码 {extra.get('confirmation_id')}，请回复确认取消或放弃。"
                    )
                elif code == 'AUTHENTICATION_REQUIRED':
                    parts.append('当前为访客未登录状态，请先在终端输入 /login 验证身份后再查询或操作个人订单。')
                elif code == 'ACCESS_DENIED_OR_NOT_FOUND':
                    parts.append(f'无法访问该订单或该订单不属于当前登录账号（{code}）。')
                elif code == 'ORDER_NOT_CANCELLABLE':
                    oid = extra.get('order_id') or _extract_order_id(user_text)
                    status = extra.get('current_status') or extra.get('status')
                    parts.append(f"订单 {oid} 当前状态为 {status}，不支持直接取消（{code}）：{msg}")
                elif code == 'ORDER_ALREADY_CANCELLED':
                    oid = extra.get('order_id') or _extract_order_id(user_text)
                    parts.append(f"订单 {oid} 已经取消，无需重复取消（{code}）。")
                elif code == 'TOOL_TIMEOUT':
                    parts.append(f'工具调用超时（{code}），暂无法获取结果。')
                else:
                    parts.append(f'操作未完成（{code}）：{msg}')
        if not parts:
            if any(k in user_text for k in ('密码', '验证码')):
                return '请不要在聊天中发送支付密码或验证码，身份验证请在本地终端使用 /login 完成，我无法代登录或查银行卡。'
            if "OR '1'='1" in user_text:
                return '订单号格式无效，无法核对查询。'
            if any(k in user_text for k in ('不要取消', '先不取消', '算了')):
                oid = _extract_order_id(' '.join(self.case['turns'])) or ''
                return f'好的，已放弃取消，订单 {oid} 保持原状未取消。'
            return '当前为访客未登录状态，无法按聊天口头声明切换身份，请先使用 /login 验证身份。'
        return '\n'.join(parts)

    def complete(self, messages, tools, system):
        self.call_count += 1
        latest_results = _latest_tool_results(messages)
        user_text = self._current_user_text(messages)
        if not latest_results:
            self._turn_index += 1
            self._step_in_turn = 0
        else:
            self._step_in_turn += 1

        turn_idx = self._turn_index
        calls = self._plan_calls(turn_idx, self._step_in_turn, user_text, messages, latest_results)
        if calls:
            return {'content': calls, 'stop_reason': 'tool_use', 'usage': {'input_tokens': 0, 'output_tokens': 0}}
        # Aggregate all tool results obtained in this turn for a faithful factual summary.
        turn_results = self._collect_turn_results(messages)
        reply = self._summarize_tool_results(turn_results, user_text)
        return {'content': [{'type': 'text', 'text': reply}], 'stop_reason': 'end_turn',
                'usage': {'input_tokens': 0, 'output_tokens': 0}}

    def _collect_turn_results(self, messages: list[dict]) -> list[tuple[str, dict]]:
        collected = []
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                content = msg.get('content')
                if isinstance(content, list) and any(isinstance(b, dict) and b.get('type') == 'tool_result' for b in content):
                    collected.extend(reversed(_latest_tool_results([msg])))
                else:
                    break
        collected.reverse()
        return collected

    def _plan_calls(self, turn_idx: int, step: int, user_text: str, messages: list[dict],
                    latest_results: list[tuple[str, dict]]) -> list[dict]:
        case = self.case
        expected = list(case.get('expected_tools') or [])
        order_id = _extract_order_id(user_text) or _extract_order_id(' '.join(case['turns']))
        product_id = _extract_product_id(user_text) or _extract_product_id(' '.join(case['turns']))

        # Turn 2 of multi-turn flows
        if turn_idx == 1 and step == 0:
            if any(k in user_text for k in ('不要取消', '先不取消', '算了')):
                return []
            if 'cancel_order' in expected and order_id:
                token = _extract_confirmation_token_from_messages(messages)
                args = {'order_id': order_id, 'reason': '客户确认取消'}
                if token:
                    args['confirmation_token'] = token
                return [{'type': 'tool_use', 'id': f'call_{self.call_count}_1', 'name': 'cancel_order', 'input': args}]
            return []

        # Turn 3 of 3-turn cancel + refund check flow
        if turn_idx == 2 and step == 0:
            if 'check_refund_status' in expected and order_id:
                return [{'type': 'tool_use', 'id': f'call_{self.call_count}_1', 'name': 'check_refund_status',
                         'input': {'order_id': order_id}}]
            return []

        if turn_idx == 0 and step == 0:
            batch = []
            # If this case has both a read/check tool and create_support_ticket (e.g., timeout or dispute),
            # run the diagnostic tool(s) first on step 0, then create_support_ticket on step 1!
            non_ticket = [t for t in expected if t != 'create_support_ticket']
            if len(case['turns']) >= 3 and 'cancel_order' in non_ticket and 'check_refund_status' in non_ticket:
                non_ticket = [t for t in non_ticket if t != 'check_refund_status']
            if non_ticket:
                to_run = non_ticket
            elif 'create_support_ticket' in expected:
                to_run = ['create_support_ticket']
            else:
                to_run = []
            for idx, tool_name in enumerate(to_run, start=1):
                args = self._build_args(tool_name, user_text, order_id, product_id)
                batch.append({'type': 'tool_use', 'id': f'call_{self.call_count}_{idx}', 'name': tool_name, 'input': args})
            return batch

        if turn_idx == 0 and step == 1 and 'create_support_ticket' in expected:
            already_called = {rec['name'] for rec in _all_tool_calls_and_results(messages)}
            if 'create_support_ticket' not in already_called:
                args = self._build_args('create_support_ticket', user_text, order_id, product_id)
                return [{'type': 'tool_use', 'id': f'call_{self.call_count}_1', 'name': 'create_support_ticket', 'input': args}]

        return []

    def _build_args(self, tool_name: str, user_text: str, order_id: str | None, product_id: str | None) -> dict:
        if tool_name == 'get_product':
            return {'product_id': product_id or 'P100'}
        if tool_name == 'get_policy':
            return {'policy_id': _extract_policy_id(user_text)}
        if tool_name == 'list_my_orders':
            return {}
        if tool_name == 'list_my_tickets':
            return {}
        if tool_name in ('get_order', 'get_tracking', 'check_refund_status'):
            return {'order_id': order_id or 'A1001'}
        if tool_name == 'cancel_order':
            return {'order_id': order_id or 'A1003', 'reason': '客户申请取消订单'}
        if tool_name == 'search_knowledge':
            return {'query': user_text[:200]}
        if tool_name == 'create_support_ticket':
            escalate = bool(self.case.get('expect_escalation'))
            category = 'human_escalation' if escalate else 'order_change'
            if not escalate and '反馈' in user_text:
                category = 'other'
            elif escalate and '退款' in user_text:
                category = 'refund_dispute'
            elif escalate and '物流' in user_text:
                category = 'logistics_issue'
            payload = {
                'category': category,
                'priority': 'high' if escalate else 'normal',
                'summary': f"客户诉求：{user_text[:120]}",
                'escalate_to_human': escalate,
            }
            if order_id:
                payload['order_id'] = order_id
            return payload
        return {}


def _normalize_fact_text(text: str) -> str:
    """Normalize typography variants (en/em dashes and CJK-digit spaces) without relaxing fact checks."""
    normalized = text.lower().replace('\u2013', '-').replace('\u2014', '-').replace('\uff0d', '-')
    normalized = re.sub(r'(\d)\s+([\u4e00-\u9fff])', r'\1\2', normalized)
    normalized = re.sub(r'([\u4e00-\u9fff])\s+(\d)', r'\1\2', normalized)
    return normalized


def _check_fact_groups(full_reply: str, fact_groups: list[list[str]]) -> tuple[bool, list[list[str]]]:
    lowered = _normalize_fact_text(full_reply)
    missing = []
    for group in fact_groups:
        if not any(_normalize_fact_text(token) in lowered for token in group):
            missing.append(group)
    return (len(missing) == 0), missing


def _check_forbidden_claims(full_reply: str, forbidden_claims: list[str]) -> list[str]:
    violations = []
    for claim in forbidden_claims:
        if claim in full_reply:
            violations.append(claim)
    for canary in _CANARY_RE.findall(full_reply):
        if canary not in violations:
            violations.append(canary)
    return violations


def _check_db_assertions(db_path: Path, customer_id: str | None, initial_ticket_ids: set[str],
                         harness: Harness, assertions: dict) -> tuple[bool, list[str]]:
    if not assertions:
        return True, []
    store = SQLiteStore(db_path, allow_writes=False)
    failures = []
    for oid, expected_status in (assertions.get('order_status') or {}).items():
        row = store.get_order_admin(oid) if hasattr(store, 'get_order_admin') else _read_order_status_direct(db_path, oid)
        actual_status = row['status'] if isinstance(row, dict) else row
        if actual_status != expected_status:
            failures.append(f'order {oid} status expected={expected_status} actual={actual_status}')
    for oid, should_exist in (assertions.get('refund_exists') or {}).items():
        actual_exists = _refund_exists_direct(db_path, oid)
        if actual_exists != bool(should_exist):
            failures.append(f'refund for {oid} expected_exists={should_exist} actual={actual_exists}')
    if customer_id is not None:
        current_tickets = store.list_tickets(customer_id)
        new_tickets = [t for t in current_tickets if t['ticket_id'] not in initial_ticket_ids]
    else:
        new_tickets = _all_new_tickets_direct(db_path, initial_ticket_ids)
    if 'min_new_tickets' in assertions and len(new_tickets) < assertions['min_new_tickets']:
        failures.append(f"new_tickets expected>={assertions['min_new_tickets']} actual={len(new_tickets)}")
    if 'max_new_tickets' in assertions and len(new_tickets) > assertions['max_new_tickets']:
        failures.append(f"new_tickets expected<={assertions['max_new_tickets']} actual={len(new_tickets)}")
    if 'has_escalated_ticket' in assertions:
        has_esc = any(bool(t.get('escalate_to_human')) and t.get('assigned_queue') == 'human_specialist'
                      for t in new_tickets)
        if has_esc != bool(assertions['has_escalated_ticket']):
            failures.append(f"has_escalated_ticket expected={assertions['has_escalated_ticket']} actual={has_esc}")
    if 'pending_confirmations_count' in assertions:
        actual_pending = len(harness.pending_confirmations())
        if actual_pending != assertions['pending_confirmations_count']:
            failures.append(f"pending_confirmations expected={assertions['pending_confirmations_count']} actual={actual_pending}")
    return (len(failures) == 0), failures


def _read_order_status_direct(db_path: Path, order_id: str) -> str | None:
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute('SELECT status FROM orders WHERE order_id = ?', (order_id,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _refund_exists_direct(db_path: Path, order_id: str) -> bool:
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute('SELECT 1 FROM refunds WHERE order_id = ? LIMIT 1', (order_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


def _all_new_tickets_direct(db_path: Path, initial_ticket_ids: set[str]) -> list[dict]:
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute('SELECT * FROM tickets').fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item['escalate_to_human'] = bool(item.get('escalate_to_human'))
            if item['ticket_id'] not in initial_ticket_ids:
                out.append(item)
        return out
    finally:
        conn.close()


def _initial_ticket_ids(db_path: Path) -> set[str]:
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[0] for row in conn.execute('SELECT ticket_id FROM tickets').fetchall()}
    finally:
        conn.close()


def run_single_case(case: dict, *, mode: str, live_model, knowledge, tmp_root: Path,
                    budget_store: DailyTokenBudgetStore | None = None,
                    daily_token_budget: int = 2_000_000) -> dict:
    case_dir = tmp_root / case['id']
    case_dir.mkdir(parents=True, exist_ok=True)
    seed_copy = case_dir / 'shop-action-seed.json'
    shutil.copyfile(SEED_PATH, seed_copy)
    db_path = case_dir / 'shop-actions.db'
    initialize_database(db_path, seed_copy)
    initial_tickets = _initial_ticket_ids(db_path)

    customer_id = case.get('customer_id')
    shop = Shop(customer_id, seed_copy, database_path=db_path, allow_actions=True)
    ctx = ContextManager(case_dir / 'sessions', customer_id or 'guest')
    fault = case.get('simulate_tool_fault')
    timeout_seconds = 0.08 if (fault and fault.get('mode') == 'timeout') else 10.0
    safeguards = SafeguardConfig.production_defaults(
        budget_store_path=case_dir / 'budget.json',
        daily_token_budget=daily_token_budget,
        tool_timeout_seconds=timeout_seconds,
        max_turn_tokens=45000,
        max_session_tokens=200000,
    )
    model = OfflineProtocolModel(case) if mode == 'offline' else live_model
    harness = Harness(
        shop, ctx, model,
        enable_subagents=False,
        knowledge=knowledge,
        safeguards=safeguards,
        budget_store=budget_store,
        retry_sleep_fn=lambda _: None,
    )

    replies = []
    error_text = None
    started = perf_counter()
    try:
        orig_call = shop.call

        def faulted_call(name, args):
            if fault and name == fault.get('tool') and fault.get('mode') == 'timeout':
                time.sleep(0.25)
            return orig_call(name, args)

        with patch.object(shop, 'call', side_effect=faulted_call):
            for user_turn in case['turns']:
                replies.append(harness.turn(user_turn))
        validate_tool_pairs(harness.messages)
    except Exception as exc:
        error_text = f'{type(exc).__name__}: {exc}'
    finally:
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        usage_snapshot = harness.budget_status() or {}
        tool_records = _all_tool_calls_and_results(harness.messages)
        db_ok, db_failures = _check_db_assertions(db_path, customer_id, initial_tickets, harness, case.get('db_assertions') or {})
        new_tickets = _all_new_tickets_direct(db_path, initial_tickets)
        harness.close()

    called_tools = [r['name'] for r in tool_records]
    called_set = set(called_tools)
    observed_error_codes = set()
    retrieved_kb_docs = set()
    for r in tool_records:
        res = r.get('result') or {}
        if not res.get('ok'):
            code = (res.get('error') or {}).get('code')
            if code:
                observed_error_codes.add(code)
        elif r['name'] == 'search_knowledge':
            data = res.get('data') or {}
            for row in data.get('hits', []) + data.get('context', []):
                if row.get('doc_id'):
                    retrieved_kb_docs.add(row['doc_id'])

    expected_tools = set(case.get('expected_tools') or [])
    alt_valid_tools = set(case.get('alternative_valid_tools') or [])
    allow_direct_refusal = bool(case.get('allow_direct_refusal'))
    forbidden_tools = set(case.get('forbidden_tools') or [])
    expected_error_codes = set(case.get('expected_error_codes') or [])
    expected_kb_docs = set(case.get('expected_kb_docs') or [])

    missing_tools = sorted(expected_tools - called_set)
    hit_forbidden_tools = sorted(forbidden_tools & called_set)
    missing_error_codes = sorted(expected_error_codes - observed_error_codes)
    kb_doc_ok = (not expected_kb_docs) or bool(expected_kb_docs & retrieved_kb_docs)

    # Allow two principled, valid agent behaviors without false negatives:
    # 1) Direct guest refusal without unnecessary private tool calls (when allow_direct_refusal=True)
    # 2) Valid read-only verification via alternative_valid_tools (e.g., get_order confirming delivered/cancelled
    #    or cross-customer ACCESS_DENIED_OR_NOT_FOUND before attempting write tools)
    used_direct_refusal = allow_direct_refusal and not called_set
    used_alt_tool = bool(alt_valid_tools & called_set) and not hit_forbidden_tools
    alt_errors_ok = (
        not missing_error_codes
        or (expected_error_codes <= {'ORDER_NOT_CANCELLABLE', 'ORDER_ALREADY_CANCELLED'} and used_alt_tool)
    )
    if used_direct_refusal:
        missing_tools = []
        missing_error_codes = []
    elif missing_tools and used_alt_tool and alt_errors_ok:
        missing_tools = []
        missing_error_codes = []

    tool_ok = (not missing_tools) and (not hit_forbidden_tools) and (not missing_error_codes) and kb_doc_ok
    full_reply = '\n'.join(replies)
    facts_ok, missing_facts = _check_fact_groups(full_reply, case.get('fact_groups') or [])
    violations = _check_forbidden_claims(full_reply, case.get('forbidden_claims') or [])
    hallucination_ok = len(violations) == 0

    actual_escalated = any(
        bool(t.get('escalate_to_human')) and t.get('assigned_queue') == 'human_specialist'
        for t in new_tickets
    )
    expect_escalation = bool(case.get('expect_escalation'))
    escalation_ok = (actual_escalated == expect_escalation)

    completed_ok = error_text is None
    resolved_ok = completed_ok and tool_ok and db_ok and facts_ok and hallucination_ok and escalation_ok

    return {
        'id': case['id'],
        'category': case['category'],
        'subcategory': case.get('subcategory'),
        'customer_id': customer_id,
        'completed_ok': completed_ok,
        'error': error_text,
        'tool_ok': tool_ok,
        'called_tools': called_tools,
        'missing_tools': missing_tools,
        'hit_forbidden_tools': hit_forbidden_tools,
        'observed_error_codes': sorted(observed_error_codes),
        'missing_error_codes': missing_error_codes,
        'expected_kb_docs': sorted(expected_kb_docs),
        'retrieved_kb_docs': sorted(retrieved_kb_docs),
        'kb_doc_ok': kb_doc_ok,
        'db_ok': db_ok,
        'db_failures': db_failures,
        'facts_ok': facts_ok,
        'missing_facts': missing_facts,
        'hallucination_ok': hallucination_ok,
        'violations': violations,
        'expect_escalation': expect_escalation,
        'actual_escalated': actual_escalated,
        'escalation_ok': escalation_ok,
        'resolved_ok': resolved_ok,
        'elapsed_ms': elapsed_ms,
        'usage': {
            'input_tokens': usage_snapshot.get('session_input_tokens', 0),
            'cache_read_tokens': usage_snapshot.get('session_cache_read_tokens', 0),
            'cache_creation_tokens': usage_snapshot.get('session_cache_creation_tokens', 0),
            'total_input_tokens': usage_snapshot.get('session_total_input_tokens', 0),
            'output_tokens': usage_snapshot.get('session_output_tokens', 0),
            'total_tokens': usage_snapshot.get('session_tokens', 0),
            'cache_hit_rate': usage_snapshot.get('session_cache_hit_rate', 0.0),
        },
        'replies_preview': [r[:260] for r in replies],
    }


def aggregate_metrics(results: list[dict]) -> dict:
    def summarize_group(rows: list[dict]) -> dict:
        n = len(rows)
        if n == 0:
            return {'total': 0}
        tp = sum(1 for r in rows if r['expect_escalation'] and r['actual_escalated'])
        fp = sum(1 for r in rows if not r['expect_escalation'] and r['actual_escalated'])
        tn = sum(1 for r in rows if not r['expect_escalation'] and not r['actual_escalated'])
        fn = sum(1 for r in rows if r['expect_escalation'] and not r['actual_escalated'])
        latencies = sorted(r['elapsed_ms'] for r in rows)
        p95_idx = min(n - 1, max(0, int(round(0.95 * (n - 1)))))
        total_in = sum(r['usage']['total_input_tokens'] for r in rows)
        total_cached = sum(r['usage']['cache_read_tokens'] for r in rows)
        total_out = sum(r['usage']['output_tokens'] for r in rows)
        return {
            'total': n,
            'tool_accuracy': round(sum(1 for r in rows if r['tool_ok']) / n, 4),
            'db_state_accuracy': round(sum(1 for r in rows if r['db_ok']) / n, 4),
            'fact_coverage_rate': round(sum(1 for r in rows if r['facts_ok']) / n, 4),
            'task_resolution_rate': round(sum(1 for r in rows if r['resolved_ok']) / n, 4),
            'hallucination_case_count': sum(1 for r in rows if not r['hallucination_ok']),
            'hallucination_violation_total': sum(len(r['violations']) for r in rows),
            'hallucination_rate': round(sum(1 for r in rows if not r['hallucination_ok']) / n, 4),
            'escalation_accuracy': round((tp + tn) / n, 4),
            'escalation_precision': round(tp / (tp + fp), 4) if (tp + fp) else None,
            'escalation_recall': round(tp / (tp + fn), 4) if (tp + fn) else None,
            'escalation_confusion': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
            'mean_latency_ms': round(sum(latencies) / n, 2),
            'p95_latency_ms': round(latencies[p95_idx], 2),
            'total_input_tokens': total_in,
            'total_cache_read_tokens': total_cached,
            'total_output_tokens': total_out,
            'overall_cache_hit_rate': round(total_cached / total_in, 4) if total_in else 0.0,
        }

    by_category = defaultdict(list)
    for row in results:
        by_category[row['category']].append(row)
    return {
        'overall': summarize_group(results),
        'by_category': {cat: summarize_group(by_category[cat]) for cat in sorted(by_category)},
    }


def evaluate_suite(dataset_path: Path = DATASET_PATH, *, mode: str = 'offline',
                   retrieval_mode: str = 'hybrid', categories: list[str] | None = None,
                   case_ids: list[str] | None = None, limit: int | None = None,
                   output_path: Path | None = None, resume_path: Path | None = None,
                   daily_token_budget: int = 2_000_000, knowledge=None,
                   live_model=None, verbose: bool = True) -> dict:
    dataset = json.loads(Path(dataset_path).read_text(encoding='utf-8'))
    cases = list(dataset['cases'])
    if categories:
        allowed = set(categories)
        cases = [c for c in cases if c['category'] in allowed]
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c['id'] in wanted]
    if limit is not None:
        cases = cases[:limit]

    existing_by_id = {}
    if resume_path and Path(resume_path).exists():
        prior = json.loads(Path(resume_path).read_text(encoding='utf-8'))
        for row in prior.get('results', []):
            existing_by_id[row['id']] = row

    owned_knowledge = False
    if knowledge is None:
        cache_dir = Path(os.environ.get('FASTEMBED_CACHE_PATH') or (ROOT / '.cache' / 'fastembed'))
        knowledge = KnowledgeSearch(
            KNOWLEDGE_DOCS_PATH, KNOWLEDGE_INDEX_PATH, cache_dir,
            retrieval_mode=retrieval_mode,
        )
        owned_knowledge = True

    if mode == 'live' and live_model is None:
        config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
        key = os.environ.get('DEEPSEEK_API_KEY') or read_secret('DeepSeek API key（隐藏输入，仅用于本次评测）: ')
        if not key or not key.strip():
            raise RuntimeError('DEEPSEEK_API_KEY is required for --mode live.')
        live_model = AnthropicModel(key.strip(), config, max_retries=2)
        del key

    (ROOT / 'runtime').mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix=f'eval_agent_100_{mode}_', dir=ROOT / 'runtime') as tmp_dir:
        tmp_root = Path(tmp_dir)
        shared_budget = DailyTokenBudgetStore(tmp_root / 'eval_daily_budget.json')
        try:
            for idx, case in enumerate(cases, start=1):
                if case['id'] in existing_by_id:
                    res = existing_by_id[case['id']]
                    results.append(res)
                    if verbose:
                        print(f"[{idx}/{len(cases)}] {case['id']} ({case['category']}) RESUMED resolved={res['resolved_ok']}")
                    continue
                res = run_single_case(
                    case, mode=mode, live_model=live_model, knowledge=knowledge,
                    tmp_root=tmp_root, budget_store=shared_budget,
                    daily_token_budget=daily_token_budget,
                )
                results.append(res)
                if verbose:
                    status = 'PASS' if res['resolved_ok'] else 'FAIL'
                    print(
                        f"[{idx}/{len(cases)}] {case['id']} ({case['category']}) {status} | "
                        f"tool={res['tool_ok']} db={res['db_ok']} facts={res['facts_ok']} "
                        f"halluc={len(res['violations'])} esc={res['escalation_ok']} | "
                        f"{res['elapsed_ms']:.1f}ms | tok={res['usage']['total_tokens']}"
                    )
                # Persist incremental checkpoint when output_path is set in live mode.
                if output_path and mode == 'live':
                    _write_report(output_path, dataset, mode, retrieval_mode, results)
        finally:
            if owned_knowledge:
                knowledge.close()

    report = _build_report(dataset, mode, retrieval_mode, results)
    if output_path:
        _write_report(output_path, dataset, mode, retrieval_mode, results)
    return report


def _build_report(dataset: dict, mode: str, retrieval_mode: str, results: list[dict]) -> dict:
    return {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'dataset_name': dataset['dataset_name'],
        'dataset_total_cases': dataset['total_cases'],
        'evaluated_cases': len(results),
        'mode': mode,
        'model_mode': 'offline-protocol-driver' if mode == 'offline' else 'live-deepseek-api',
        'retrieval_mode': retrieval_mode,
        'chat_api_calls_note': (
            '0 LLM API calls (offline protocol driver over real Harness/Shop/SQLite/Hybrid-RAG). '
            'Validates dataset, DB state machine, safeguards, RAG retrieval, and scoring pipeline; '
            'DO NOT report as real LLM accuracy. Run --mode live for true LLM metrics.'
            if mode == 'offline'
            else 'Real DeepSeek API calls executed end-to-end with isolated per-case SQLite databases.'
        ),
        'summary': aggregate_metrics(results),
        'results': results,
    }


def _write_report(output_path: Path, dataset: dict, mode: str, retrieval_mode: str, results: list[dict]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = _build_report(dataset, mode, retrieval_mode, results)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['offline', 'live'], default='offline')
    parser.add_argument('--retrieval-mode', choices=['dense', 'hybrid'], default='hybrid')
    parser.add_argument('--dataset', type=Path, default=DATASET_PATH)
    parser.add_argument('--categories', type=str, default=None,
                        help='Comma-separated category filter (e.g. normal_query,action_execution)')
    parser.add_argument('--case-ids', type=str, default=None,
                        help='Comma-separated case ID filter (e.g. E001,E023,E098)')
    parser.add_argument('--limit', type=int, default=None, help='Run at most N cases')
    parser.add_argument('--daily-token-budget', type=int, default=2_000_000)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--resume', type=Path, default=None, help='Resume from prior partial JSON report')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    output = args.output or (ROOT / f'runtime/eval-agent-100-{args.mode}-latest.json')
    categories = [c.strip() for c in args.categories.split(',') if c.strip()] if args.categories else None
    case_ids = [c.strip() for c in args.case_ids.split(',') if c.strip()] if args.case_ids else None

    report = evaluate_suite(
        args.dataset,
        mode=args.mode,
        retrieval_mode=args.retrieval_mode,
        categories=categories,
        case_ids=case_ids,
        limit=args.limit,
        output_path=output,
        resume_path=args.resume,
        daily_token_budget=args.daily_token_budget,
        verbose=not args.quiet,
    )
    print('\n=== Evaluation Summary ===')
    print(json.dumps({
        'mode': report['mode'],
        'retrieval_mode': report['retrieval_mode'],
        'evaluated_cases': report['evaluated_cases'],
        'overall': report['summary']['overall'],
        'by_category': report['summary']['by_category'],
        'report': str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
