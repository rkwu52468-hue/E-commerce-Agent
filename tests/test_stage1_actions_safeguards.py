"""Stage 1 acceptance tests: expanded SQLite writes, action tools, 2-step confirmation, and safeguards."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from customer_service.actions_demo import run_actions_demo
from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.live_session import LiveSession
from customer_service.models import AnthropicModel
from customer_service.safeguards import (
    DailyTokenBudgetStore,
    SafeguardConfig,
    is_transient_model_error,
)
from customer_service.shop import ACTION_TOOL_NAMES, MAX_TICKETS_PER_SESSION, Shop
from customer_service.storage import SQLiteStore, StoreError, initialize_database

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / 'data/shop-action-seed.json'
PRIVATE_REASON = 'PRIVATE_CANCEL_REASON_7741'
PRIVATE_SUMMARY = 'PRIVATE_TICKET_SUMMARY_8852'


def text(value):
    return {'type': 'text', 'text': value}


def call(identifier, name, **arguments):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': arguments}


class FakeAPIError(Exception):
    def __init__(self, status_code=429, code='rate_limit_error', message='too many requests'):
        super().__init__(message)
        self.status_code = status_code
        self.body = {'error': {'type': code, 'message': message}}


class ScriptedModel:
    mode = 'scripted-stage1-test'

    def __init__(self, replies, usage=None):
        self.replies = list(replies)
        self.calls = []
        self.default_usage = {'input_tokens': 100, 'output_tokens': 50} if usage is None else usage

    def complete(self, messages, tools, system):
        self.calls.append({'messages': deepcopy(messages), 'tools': deepcopy(tools), 'system': system})
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        content = item(messages, system) if callable(item) else deepcopy(item)
        return {
            'content': content,
            'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in content) else 'end_turn',
            'usage': deepcopy(self.default_usage),
        }


class Stage1ActionAndSafeguardTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='stage1_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)
        self.seed_path = self.runtime / 'shop-action-seed.json'
        shutil.copyfile(SEED_PATH, self.seed_path)
        self.db_path = self.runtime / 'shop-actions.db'
        initialize_database(self.db_path, self.seed_path)

    def make_shop(self, customer='alice', sqlite=True, allow_actions=True):
        return Shop(customer, self.seed_path, database_path=self.db_path if sqlite else None,
                    allow_actions=allow_actions)

    def make_harness(self, replies, customer='alice', sqlite=True, allow_actions=True,
                     safeguards=None, budget_store=None, subagent_model=None, usage=None):
        shop = self.make_shop(customer=customer, sqlite=sqlite, allow_actions=allow_actions)
        ctx = ContextManager(self.runtime / 'sessions', customer or 'guest')
        model = ScriptedModel(replies, usage=usage)
        harness = Harness(
            shop, ctx, model,
            subagent_model=subagent_model,
            safeguards=safeguards,
            budget_store=budget_store,
            retry_sleep_fn=lambda _: None,
        )
        return harness, ctx, model

    def tool_results(self, messages):
        return {
            block['tool_use_id']: json.loads(block['content'])
            for msg in messages if isinstance(msg.get('content'), list)
            for block in msg['content'] if block.get('type') == 'tool_result'
        }

    def test_expanded_seed_initializes_multi_status_orders_refunds_and_tickets(self):
        store = SQLiteStore(self.db_path, allow_writes=True)
        self.assertEqual('action-sqlite-2026-11-26', store.dataset_version)
        for customer in ('alice', 'bob', 'charlie'):
            self.assertTrue(store.has_customer(customer))
        alice_orders = store.list_orders('alice')
        bob_orders = store.list_orders('bob')
        charlie_orders = store.list_orders('charlie')
        self.assertEqual(6, len(alice_orders))
        self.assertEqual(4, len(bob_orders))
        self.assertEqual(2, len(charlie_orders))
        self.assertEqual(
            {'shipped', 'pending_payment', 'paid_unshipped', 'delivered', 'cancelled'},
            {row['status'] for row in alice_orders},
        )
        self.assertEqual('under_review', store.get_refund_by_order('A1004', 'alice')['status'])
        self.assertEqual('refunded', store.get_refund_by_order('A1005', 'alice')['status'])
        self.assertEqual('rejected', store.get_refund_by_order('B2004', 'bob')['status'])
        self.assertEqual(1, len(store.list_tickets('alice')))
        # Read-only SQLiteStore refuses write calls explicitly.
        readonly = SQLiteStore(self.db_path, allow_writes=False)
        with self.assertRaises(StoreError):
            readonly.cancel_order('A1002', 'alice', 'test', 'cfm_12345678')
        with self.assertRaises(StoreError):
            readonly.create_support_ticket('alice', 'other', 'readonly check', 'low')

    def test_guest_and_cross_customer_isolation_on_all_action_tools(self):
        guest = self.make_shop(customer=None)
        for name, args in [
            ('check_refund_status', {'order_id': 'A1004'}),
            ('cancel_order', {'order_id': 'A1002', 'reason': 'guest attempt'}),
            ('create_support_ticket', {'category': 'other', 'priority': 'low', 'summary': 'guest ticket test'}),
            ('list_my_tickets', {}),
        ]:
            with self.subTest(role='guest', tool=name):
                res = guest.call(name, args)
                self.assertFalse(res['ok'])
                self.assertEqual('AUTHENTICATION_REQUIRED', res['error']['code'])

        alice = self.make_shop(customer='alice')
        for name, args in [
            ('check_refund_status', {'order_id': 'B2004'}),
            ('cancel_order', {'order_id': 'B2002', 'reason': 'cross customer attempt'}),
            ('create_support_ticket', {'category': 'refund_dispute', 'priority': 'high',
                                       'order_id': 'B2004', 'summary': 'trying to attach bob order'}),
        ]:
            with self.subTest(role='alice_foreign', tool=name):
                res = alice.call(name, args)
                self.assertFalse(res['ok'])
                self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', res['error']['code'])
                self.assertNotIn('B2004', json.dumps(res, ensure_ascii=False))
                self.assertNotIn('RF-B2004', json.dumps(res, ensure_ascii=False))
        # Direct shop.call('cancel_order') cannot bypass confirmation even for owned cancellable order.
        direct = alice.call('cancel_order', {'order_id': 'A1002', 'reason': 'direct call'})
        self.assertFalse(direct['ok'])
        self.assertEqual('CONFIRMATION_REQUIRED', direct['error']['code'])
        self.assertEqual('pending_payment', SQLiteStore(self.db_path).get_order('A1002', 'alice')['status'])

    def test_subagent_remains_strictly_readonly_even_when_parent_has_actions_enabled(self):
        parent_replies = [
            [call('sub_1', 'task', prompt='检查我的订单列表和退款状态')],
            [text('子调查完成。')],
        ]
        child_replies = [
            [call('c_cancel', 'cancel_order', order_id='A1002', reason='subagent try'),
             call('c_ticket', 'create_support_ticket', category='other', priority='low', summary='subagent ticket'),
             call('c_order', 'get_order', order_id='A1002')],
            [text('只读核对完成。')],
        ]
        child_model = ScriptedModel(child_replies)
        harness, _, parent_model = self.make_harness(parent_replies, subagent_model=child_model)
        self.assertTrue(ACTION_TOOL_NAMES <= {t['name'] for t in harness.tools})
        harness.turn('请用子任务核对我的订单。')
        child_tool_names = {t['name'] for t in child_model.calls[0]['tools']}
        self.assertTrue(ACTION_TOOL_NAMES.isdisjoint(child_tool_names))
        evidence = {e['tool_use_id']: e['result']
                    for e in self.tool_results(parent_model.calls[1]['messages'])['sub_1']['data']['evidence']}
        self.assertEqual('SUBAGENT_TOOL_NOT_ALLOWED', evidence['c_cancel']['error']['code'])
        self.assertEqual('SUBAGENT_TOOL_NOT_ALLOWED', evidence['c_ticket']['error']['code'])
        self.assertTrue(evidence['c_order']['ok'])

    def test_cancel_order_state_machine_two_step_confirmation_refund_and_idempotency(self):
        holder = {}

        def same_turn_self_confirm(messages, system):
            res = self.tool_results(messages)
            self.assertEqual('ORDER_NOT_CANCELLABLE', res['shipped']['error']['code'])
            self.assertEqual('ORDER_ALREADY_CANCELLED', res['already']['error']['code'])
            self.assertEqual('CONFIRMATION_REQUIRED', res['req_paid']['error']['code'])
            self.assertTrue(res['req_paid']['data']['will_create_refund'])
            cid = res['req_paid']['data']['confirmation_id']
            holder['cid'] = cid
            return [call('self_approve', 'cancel_order', order_id='A1003', reason=PRIVATE_REASON, confirmation_token=cid)]

        def after_self_blocked(messages, system):
            res = self.tool_results(messages)
            self.assertEqual('CONFIRMATION_PENDING_USER_REPLY', res['self_approve']['error']['code'])
            return [text('请确认是否取消 A1003。')]

        def exec_with_token(messages, system):
            self.assertNotIn('<pending_confirmations>', system)
            blocks = messages[-1]['content']
            self.assertIsInstance(blocks, list)
            state_text = blocks[-1]['text']
            self.assertIn('<confirmation_state>', state_text)
            self.assertIn('"status": "approved"', state_text)
            return [call('exec_paid', 'cancel_order', order_id='A1003', reason=PRIVATE_REASON, confirmation_token=holder['cid'])]

        def replay_and_verify(messages, system):
            res = self.tool_results(messages)
            self.assertTrue(res['exec_paid']['ok'])
            self.assertEqual('cancelled', res['exec_paid']['data']['status'])
            self.assertEqual('RF-A1003', res['exec_paid']['data']['refund_created']['refund_id'])
            return [
                call('replay_token', 'cancel_order', order_id='A1003', reason=PRIVATE_REASON, confirmation_token=holder['cid']),
                call('repeat_no_token', 'cancel_order', order_id='A1003', reason='again'),
                call('check_rf', 'check_refund_status', order_id='A1003'),
            ]

        harness, ctx, model = self.make_harness([
            [
                call('shipped', 'cancel_order', order_id='A1001', reason='shipped cancel'),
                call('already', 'cancel_order', order_id='A1005', reason='already cancelled'),
                call('req_paid', 'cancel_order', order_id='A1003', reason=PRIVATE_REASON),
            ],
            same_turn_self_confirm,
            after_self_blocked,
            exec_with_token,
            replay_and_verify,
            [text('A1003 已取消并自动生成退款单 RF-A1003。')],
        ])

        harness.turn('帮我取消 A1001、A1005 和 A1003。')
        # Still uncancelled in SQLite before customer replies!
        self.assertEqual('paid_unshipped', SQLiteStore(self.db_path).get_order('A1003', 'alice')['status'])

        harness.turn('我确认取消订单 A1003。')
        # System prompt MUST stay byte-for-byte identical across the entire 2-step confirmation flow (KV prefix cache!).
        self.assertEqual(1, len({entry['system'] for entry in model.calls}))
        results = self.tool_results(harness.messages)
        self.assertEqual('CONFIRMATION_ALREADY_USED', results['replay_token']['error']['code'])
        self.assertEqual('ORDER_ALREADY_CANCELLED', results['repeat_no_token']['error']['code'])
        self.assertEqual('RF-A1003', results['check_rf']['data']['refund']['refund_id'])
        # Persisted durably in SQLite!
        reopened = SQLiteStore(self.db_path)
        self.assertEqual('cancelled', reopened.get_order('A1003', 'alice')['status'])
        self.assertIsNotNone(reopened.get_cancellation('A1003', 'alice'))
        self.assertEqual(22900, reopened.get_refund_by_order('A1003', 'alice')['amount_cents'])
        # Audit stream MUST NOT leak private reason text.
        audit_text = (ctx.session_dir / 'audit.jsonl').read_text(encoding='utf-8')
        self.assertNotIn(PRIVATE_REASON, audit_text)
        validate_tool_pairs(harness.messages)

    def test_pending_payment_cancel_via_explicit_approve_creates_no_refund_and_rejection_clears_token(self):
        harness, _, _ = self.make_harness([
            [call('req_unpaid', 'cancel_order', order_id='A1002', reason='unpaid cancel')],
            [text('请确认取消 A1002。')],
        ])
        harness.turn('取消待支付订单 A1002。')
        pending = harness.pending_confirmations()
        self.assertEqual(1, len(pending))
        cid = pending[0]['confirmation_id']
        # Explicit /confirm execution works directly without another model turn.
        outcome = harness.approve_confirmation(cid, execute_now=True)
        self.assertTrue(outcome['ok'])
        self.assertEqual('cancelled', outcome['data']['status'])
        self.assertIsNone(outcome['data']['refund_created'])
        store = SQLiteStore(self.db_path)
        self.assertEqual('cancelled', store.get_order('A1002', 'alice')['status'])
        self.assertIsNone(store.get_refund_by_order('A1002', 'alice'))

        # Rejection via user chat ("不要取消") invalidates pending confirmation.
        harness2, _, _ = self.make_harness([
            [call('req_1', 'cancel_order', order_id='A1003', reason='maybe')],
            [text('请确认是否取消 A1003。')],
            [text('好的，已放弃取消。')],
        ])
        harness2.turn('帮我取消 A1003。')
        self.assertEqual(1, len(harness2.pending_confirmations()))
        harness2.turn('算了，不要取消了。')
        self.assertEqual([], harness2.pending_confirmations())

    def test_tickets_dedup_escalation_cap_and_audit_privacy(self):
        harness, ctx, _ = self.make_harness([
            [
                call('t1', 'create_support_ticket', category='human_escalation', order_id='A1001',
                     priority='urgent', summary=PRIVATE_SUMMARY),
                call('t2_dup', 'create_support_ticket', category='human_escalation', order_id='A1001',
                     priority='urgent', summary=PRIVATE_SUMMARY),
                call('t_list', 'list_my_tickets'),
            ],
            [text('已为您升级人工工单。')],
        ])
        harness.turn('请转人工处理 A1001。')
        res = self.tool_results(harness.messages)
        self.assertTrue(res['t1']['ok'] and not res['t1']['data']['deduplicated'])
        self.assertEqual('human_specialist', res['t1']['data']['ticket']['assigned_queue'])
        self.assertTrue(res['t1']['data']['ticket']['escalate_to_human'])
        self.assertTrue(res['t2_dup']['ok'] and res['t2_dup']['data']['deduplicated'])
        self.assertEqual(res['t1']['data']['ticket']['ticket_id'], res['t2_dup']['data']['ticket']['ticket_id'])
        self.assertEqual(2, len(res['t_list']['data']['tickets']))
        self.assertNotIn(PRIVATE_SUMMARY, (ctx.session_dir / 'audit.jsonl').read_text(encoding='utf-8'))

        # Per-session ticket creation cap.
        shop = self.make_shop('alice')
        for idx in range(MAX_TICKETS_PER_SESSION):
            out = shop.create_support_ticket('other', f'unique ticket issue number {idx}', 'low')
            self.assertTrue(out['ok'])
        capped = shop.create_support_ticket('other', 'exceeds session ticket limit', 'low')
        self.assertFalse(capped['ok'])
        self.assertEqual('TICKET_SESSION_LIMIT_REACHED', capped['error']['code'])

    def test_safeguards_transient_retry_timeout_loop_breaker_and_token_budget(self):
        # 1) Transient classification
        self.assertTrue(is_transient_model_error(FakeAPIError(429)))
        self.assertTrue(is_transient_model_error(FakeAPIError(503)))
        self.assertTrue(is_transient_model_error(TimeoutError('timed out')))
        self.assertFalse(is_transient_model_error(FakeAPIError(401, 'authentication_error', 'bad key')))
        self.assertFalse(is_transient_model_error(FakeAPIError(400, 'prompt_too_long', 'prompt is too long')))

        # 2) Harness transient retry recovers on 429 + 503; does NOT retry 401
        safeguards = SafeguardConfig(
            model_max_retries=2, retry_base_delay=0.0, retry_max_delay=0.0,
            tool_timeout_seconds=0.15, max_identical_tool_calls=2, max_consecutive_tool_failures=2,
            max_turn_tokens=5000, max_session_tokens=10000, daily_token_budget=50000,
            budget_store_path=self.runtime / 'budget.json',
        )
        harness, ctx, model = self.make_harness(
            [FakeAPIError(429), FakeAPIError(503), [text('重试后成功。')]],
            safeguards=safeguards,
        )
        self.assertEqual('重试后成功。', harness.turn('测试瞬态重试。'))
        self.assertEqual(3, len(model.calls))
        retry_events = [json.loads(line) for line in (ctx.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()
                        if json.loads(line)['event'] == 'model_transient_retry']
        self.assertEqual(2, len(retry_events))

        harness_401, _, model_401 = self.make_harness([FakeAPIError(401, 'authentication_error', 'bad key')], safeguards=safeguards)
        with self.assertRaises(RuntimeError):
            harness_401.turn('401 不应重试。')
        self.assertEqual(1, len(model_401.calls))

        # 3) AnthropicModel built-in transient retry with fake SDK client
        fake_calls = {'n': 0}

        class FakeClient:
            def __init__(self):
                self.messages = self

            def create(self, **kwargs):
                fake_calls['n'] += 1
                if fake_calls['n'] == 1:
                    raise FakeAPIError(502, 'bad_gateway', 'transient')
                return SimpleNamespace(
                    content=[SimpleNamespace(model_dump=lambda **kw: {'type': 'text', 'text': 'ok'})],
                    stop_reason='end_turn',
                    usage=SimpleNamespace(model_dump=lambda **kw: {'input_tokens': 10, 'output_tokens': 5}),
                )

        live_adapter = AnthropicModel('fake', {'model': 'deepseek-flash', 'base_url': 'https://example.invalid'},
                                      max_retries=2, retry_base_delay=0.0, sleep_fn=lambda _: None, client=FakeClient())
        resp = live_adapter.complete([{'role': 'user', 'content': 'hi'}], [], 'sys')
        self.assertEqual('end_turn', resp['stop_reason'])
        self.assertEqual(2, fake_calls['n'])
        self.assertEqual(1, live_adapter.last_retry_count)

        # 4) Tool timeout returns paired TOOL_TIMEOUT cleanly
        t_harness, t_ctx, _ = self.make_harness(
            [[call('slow_tool', 'get_product', product_id='P100')], [text('工具超时后已妥善告知。')]],
            safeguards=safeguards,
        )
        orig_call = t_harness.shop.call

        def slow_call(name, args):
            time.sleep(0.3)
            return orig_call(name, args)

        with patch.object(t_harness.shop, 'call', side_effect=slow_call):
            ans = t_harness.turn('查询商品。')
        self.assertEqual('工具超时后已妥善告知。', ans)
        self.assertEqual('TOOL_TIMEOUT', self.tool_results(t_harness.messages)['slow_tool']['error']['code'])
        validate_tool_pairs(t_harness.messages)

        # 5) Repeat-call & consecutive-failure loop breaker
        l_harness, _, _ = self.make_harness([
            [call('r1', 'get_product', product_id='P100')],
            [call('r2', 'get_product', product_id='P100')],
            [call('r3', 'get_product', product_id='P100')],
            [text('已停止重复调用。')],
        ], safeguards=safeguards)
        with patch.object(l_harness.shop, 'call', wraps=l_harness.shop.call) as spy:
            l_harness.turn('重复查询。')
        self.assertEqual(2, spy.call_count)
        res = self.tool_results(l_harness.messages)
        self.assertEqual('REPEATED_TOOL_CALL_BLOCKED', res['r3']['error']['code'])

        f_harness, _, _ = self.make_harness([
            [call('f1', 'get_order', order_id='NOPE_1')],
            [call('f2', 'get_order', order_id='NOPE_2')],
            [call('f3', 'get_order', order_id='NOPE_3')],
            [text('连续失败后转人工。')],
        ], safeguards=safeguards)
        f_harness.turn('连续失败测试。')
        self.assertEqual('REPEATED_TOOL_CALL_BLOCKED', self.tool_results(f_harness.messages)['f3']['error']['code'])

        # 6) Per-turn and persistent daily token budget circuit breaker
        daily_path = self.runtime / 'daily_budget.json'
        fixed_clock = lambda: datetime(2026, 11, 26, 15, 0, tzinfo=timezone(timedelta(hours=8)))
        daily_store = DailyTokenBudgetStore(daily_path, clock=fixed_clock)
        tight = SafeguardConfig(
            model_max_retries=0, tool_timeout_seconds=1.0,
            max_identical_tool_calls=5, max_consecutive_tool_failures=5,
            max_turn_tokens=200, max_session_tokens=500, daily_token_budget=250,
            budget_store_path=daily_path,
        )
        b_harness, _, b_model = self.make_harness(
            [[call('b1', 'get_product', product_id='P100')], [text('should be blocked')]],
            safeguards=tight, budget_store=daily_store,
            usage={'input_tokens': 160, 'output_tokens': 60},  # 220 >= max_turn_tokens(200)
        )
        turn_reply = b_harness.turn('触发单轮预算上限。')
        self.assertIn('熔断', turn_reply)
        self.assertEqual(1, len(b_model.calls))
        # Second harness sees daily usage 220; after 1 more call (220 -> 440 >= 250), third harness blocked before any API call!
        b2, _, _ = self.make_harness([[text('ok')]], safeguards=tight, budget_store=daily_store,
                                     usage={'input_tokens': 40, 'output_tokens': 10})
        self.assertEqual('ok', b2.turn('consume remaining daily budget'))
        b3, _, b3_model = self.make_harness([[text('never')]], safeguards=tight, budget_store=daily_store)
        daily_reply = b3.turn('每日预算已满。')
        self.assertIn('熔断', daily_reply)
        self.assertEqual(0, len(b3_model.calls))

    def test_live_session_logout_and_switch_clears_pending_confirmations(self):
        root = self.runtime / 'live_root'
        (root / 'data').mkdir(parents=True)
        shutil.copyfile(self.seed_path, root / 'data/shop-action-seed.json')
        initialize_database(root / 'data/shop-actions.db', root / 'data/shop-action-seed.json')
        verifier = SimpleNamespace(verify=lambda cid, pw: cid if pw == 'secret' else None)
        model = ScriptedModel([
            [call('c1', 'cancel_order', order_id='A1003', reason='test logout')],
            [text('请确认取消 A1003。')],
        ])
        session = LiveSession(root, model, allow_actions=True,
                              safeguards=SafeguardConfig.production_defaults(budget_store_path=root / 'runtime/budget.json'))
        self.addCleanup(session.close)
        self.assertTrue(session.login('alice', 'secret', verifier))
        session.turn('取消 A1003。')
        self.assertEqual(1, len(session.pending_confirmations()))
        session.logout()
        self.assertEqual([], session.pending_confirmations())

    def test_cache_read_tokens_and_legacy_budget_file_upgrade_cleanly(self):
        legacy_path = self.runtime / 'legacy_budget.json'
        legacy_path.write_text(json.dumps({
            'days': {'2026-11-26': {'input_tokens': 1000, 'output_tokens': 200, 'total_tokens': 1200, 'requests': 2}}
        }), encoding='utf-8')
        fixed_clock = lambda: datetime(2026, 11, 26, 20, 0, tzinfo=timezone(timedelta(hours=8)))
        store = DailyTokenBudgetStore(legacy_path, clock=fixed_clock)
        before = store.today_usage()
        self.assertEqual(0, before['cache_read_input_tokens'])
        self.assertEqual(1000, before['total_input_tokens'])
        self.assertEqual(0.0, before['cache_hit_rate'])

        safeguards = SafeguardConfig(
            model_max_retries=0, tool_timeout_seconds=1.0,
            max_identical_tool_calls=5, max_consecutive_tool_failures=5,
            max_turn_tokens=50000, max_session_tokens=100000, daily_token_budget=500000,
            budget_store_path=legacy_path,
        )
        harness, _, _ = self.make_harness(
            [[text('cached reply')]],
            safeguards=safeguards,
            budget_store=store,
            usage={'input_tokens': 200, 'cache_read_input_tokens': 3800, 'cache_creation_input_tokens': 0, 'output_tokens': 100},
        )
        harness.turn('test cache accounting')
        status = harness.budget_status()
        self.assertEqual(200, status['session_input_tokens'])
        self.assertEqual(3800, status['session_cache_read_tokens'])
        self.assertEqual(4000, status['session_total_input_tokens'])
        self.assertEqual(4100, status['session_tokens'])
        self.assertEqual(0.95, status['session_cache_hit_rate'])
        daily = status['daily_usage']
        self.assertEqual(1200, daily['input_tokens'])
        self.assertEqual(3800, daily['cache_read_input_tokens'])
        self.assertEqual(5000, daily['total_input_tokens'])
        self.assertEqual(5300, daily['total_tokens'])
        self.assertEqual(0.76, daily['cache_hit_rate'])

    def test_cache_safe_system_keeps_system_prompt_static_across_todo_updates(self):
        safeguards = SafeguardConfig.production_defaults(budget_store_path=self.runtime / 'todo_budget.json')
        harness, _, model = self.make_harness([
            [call('td1', 'todo_write', todos=[
                {'content': '查物流', 'status': 'in_progress'},
                {'content': '建人工工单', 'status': 'pending'},
            ])],
            [call('td2', 'todo_write', todos=[
                {'content': '查物流', 'status': 'completed'},
                {'content': '建人工工单', 'status': 'completed'},
            ])],
            [text('第一轮已完成。')],
            [text('第二轮收到。')],
        ], safeguards=safeguards)
        harness.turn('核实 A1001 延误并建人工工单。')
        harness.turn('继续查一下工单状态。')
        self.assertEqual(4, len(model.calls))
        # System prompt MUST stay byte-for-byte invariant across all todo_write updates!
        self.assertEqual(1, len({c['system'] for c in model.calls}))
        self.assertNotIn('<current_todos>', model.calls[1]['system'])
        self.assertNotIn('<current_todos>', model.calls[2]['system'])
        # Next turn's user message receives <current_todos> append-only at the tail!
        last_user = model.calls[3]['messages'][-1]['content']
        self.assertIsInstance(last_user, list)
        self.assertTrue(any('<current_todos>' in b.get('text', '') for b in last_user))

    def test_offline_actions_demo_passes_all_stage1_checks(self):
        report = run_actions_demo(ROOT)
        self.assertTrue(all(report['checks'].values()), report['checks'])
        self.assertEqual(0, report['api_calls'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
