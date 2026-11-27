"""Stage 1 offline showcase: action tools, 2-step confirmation, and production safeguards."""
import copy
import json
import sqlite3
import time
from pathlib import Path

from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .safeguards import DailyTokenBudgetStore, SafeguardConfig
from .shop import Shop
from .storage import initialize_database


class TransientError(Exception):
    def __init__(self, status_code=429, message='rate limited'):
        super().__init__(message)
        self.status_code = status_code


class DemoModel:
    mode = 'scripted-stage1-demo'

    def __init__(self, replies):
        self.replies = list(replies)
        self.steps = 0
        self.attempts = 0

    def complete(self, messages, tools, system):
        self.attempts += 1
        item = self.replies.pop(0)
        if isinstance(item, BaseException):
            raise item
        self.steps += 1
        if callable(item):
            content = item(messages, system)
        else:
            content = copy.deepcopy(item)
        usage = {'input_tokens': 320, 'output_tokens': 95}
        return {
            'content': content,
            'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in content) else 'end_turn',
            'usage': usage,
        }


def _last_results(messages):
    return {
        block['tool_use_id']: json.loads(block['content'])
        for block in messages[-1]['content']
        if block.get('type') == 'tool_result'
    }


def run_actions_demo(root):
    root = Path(root)
    context = ContextManager(root / 'runtime', 'alice')
    database = context.session_dir / 'shop-actions.db'
    budget_file = context.session_dir / 'token_budget.json'
    seed_path = root / 'data/shop-action-seed.json'
    initialized = initialize_database(database, seed_path)
    shop = Shop('alice', seed_path, database_path=database, allow_actions=True)
    safeguards = SafeguardConfig(
        model_max_retries=2,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        tool_timeout_seconds=0.2,
        max_identical_tool_calls=2,
        max_consecutive_tool_failures=3,
        max_turn_tokens=16000,
        max_session_tokens=60000,
        daily_token_budget=500000,
        budget_store_path=budget_file,
    )
    observed = {}
    token_holder = {}

    def on_list_and_shipped_cancel(messages, system):
        res = _last_results(messages)
        observed['orders'] = res['list_1']
        observed['shipped_cancel'] = res['cancel_shipped']
        assert res['list_1']['ok'] and len(res['list_1']['data']['orders']) == 6
        assert res['cancel_shipped']['error']['code'] == 'ORDER_NOT_CANCELLABLE'
        return [text(
            '已查询到您的 6 笔模拟订单。其中 A1001 当前状态为 shipped（已发货），'
            '系统状态机不允许直接取消，可按需为您查物流或创建人工工单跟进。'
        )]

    def on_preview_and_self_confirm_attempt(messages, system):
        res = _last_results(messages)
        observed['cancel_preview'] = res['cancel_a1003_req']
        assert res['cancel_a1003_req']['error']['code'] == 'CONFIRMATION_REQUIRED'
        cid = res['cancel_a1003_req']['data']['confirmation_id']
        token_holder['cid'] = cid
        # Deliberately try to bypass human confirmation in the same turn!
        return [call('self_confirm_attempt', 'cancel_order',
                     order_id='A1003', reason='买错容量想重拍', confirmation_token=cid)]

    def on_self_confirm_blocked(messages, system):
        res = _last_results(messages)
        observed['self_confirm_blocked'] = res['self_confirm_attempt']
        assert res['self_confirm_attempt']['error']['code'] == 'CONFIRMATION_PENDING_USER_REPLY'
        return [text(
            f'订单 A1003（实付 229.00 元，已付款未发货）取消后将自动生成原路退款单。'
            f'请明确回复确认（确认码 {token_holder["cid"]}）后再为您执行取消。'
        )]

    def call_cancel_with_approved_token(messages, system):
        assert '<pending_confirmations>' not in system
        assert '<confirmation_state>' in json.dumps(messages[-1], ensure_ascii=False)
        return [call('cancel_a1003_exec', 'cancel_order',
                     order_id='A1003', reason='买错容量想重拍', confirmation_token=token_holder['cid'])]

    def on_cancel_executed(messages, system):
        res = _last_results(messages)
        observed['cancel_executed'] = res['cancel_a1003_exec']
        assert res['cancel_a1003_exec']['ok']
        assert res['cancel_a1003_exec']['data']['status'] == 'cancelled'
        assert res['cancel_a1003_exec']['data']['refund_created']['refund_id'] == 'RF-A1003'
        # Immediately try idempotent repeat cancel + refund check + ticket escalation + foreign access!
        return [
            call('cancel_idempotent', 'cancel_order', order_id='A1003', reason='重复取消测试'),
            call('refund_a1003', 'check_refund_status', order_id='A1003'),
            call('refund_a1004', 'check_refund_status', order_id='A1004'),
            call('ticket_escalate', 'create_support_ticket',
                 category='logistics_issue', order_id='A1001', priority='high',
                 summary='A1001 物流网点积压超时，客户申请人工专员核实配送与赔付方案',
                 escalate_to_human=True),
            call('ticket_dedup', 'create_support_ticket',
                 category='logistics_issue', order_id='A1001', priority='high',
                 summary='A1001 物流网点积压超时，客户申请人工专员核实配送与赔付方案',
                 escalate_to_human=True),
            call('foreign_cancel', 'cancel_order', order_id='B2002', reason='试图越权取消他人订单'),
        ]

    def on_post_actions(messages, system):
        res = _last_results(messages)
        observed['cancel_idempotent'] = res['cancel_idempotent']
        observed['refund_a1003'] = res['refund_a1003']
        observed['refund_a1004'] = res['refund_a1004']
        observed['ticket_escalate'] = res['ticket_escalate']
        observed['ticket_dedup'] = res['ticket_dedup']
        observed['foreign_cancel'] = res['foreign_cancel']
        assert res['cancel_idempotent']['error']['code'] == 'ORDER_ALREADY_CANCELLED'
        assert res['refund_a1003']['ok'] and res['refund_a1003']['data']['refund']['refund_id'] == 'RF-A1003'
        assert res['refund_a1004']['ok'] and res['refund_a1004']['data']['refund']['status'] == 'under_review'
        assert res['ticket_escalate']['ok'] and not res['ticket_escalate']['data']['deduplicated']
        assert res['ticket_dedup']['ok'] and res['ticket_dedup']['data']['deduplicated']
        assert res['foreign_cancel']['error']['code'] == 'ACCESS_DENIED_OR_NOT_FOUND'
        tid = res['ticket_escalate']['data']['ticket']['ticket_id']
        return [text(
            f'订单 A1003 已成功取消并自动生成退款单 RF-A1003（229.00 元，预计 1-3 个工作日原路退回）；'
            f'A1004 退款 RF-A1004 当前处于质检审核中；已为延误订单 A1001 创建高优人工升级工单 {tid}；'
            f'他人订单 B2002 已被权限隔离拦截。'
        )]

    def on_loop_guard_done(messages, system):
        res = _last_results(messages)
        observed['loop_blocked'] = res['loop_3']
        assert res['loop_3']['error']['code'] == 'REPEATED_TOOL_CALL_BLOCKED'
        return [text('检测到重复调用同一查询已触发防死循环熔断，直接为您汇总已有结果并建议必要时转人工。')]

    model = DemoModel([
        # Turn 1: Transient 429 error -> auto retry succeeds!
        TransientError(429, 'simulated rate limit'),
        [call('list_1', 'list_my_orders'),
         call('cancel_shipped', 'cancel_order', order_id='A1001', reason='物流太慢想取消')],
        on_list_and_shipped_cancel,
        # Turn 2: Request cancel A1003 -> CONFIRMATION_REQUIRED -> try self-confirm -> BLOCKED!
        [call('cancel_a1003_req', 'cancel_order', order_id='A1003', reason='买错容量想重拍')],
        on_preview_and_self_confirm_attempt,
        on_self_confirm_blocked,
        # Turn 3: Customer confirms in next turn -> execute cancel + auto refund + idempotent + ticket + foreign deny
        call_cancel_with_approved_token,
        on_cancel_executed,
        on_post_actions,
        # Turn 4: Repeated identical call loop guard (calls 1 & 2 succeed, call 3 circuit-broken)
        [call('loop_1', 'get_product', product_id='P100')],
        [call('loop_2', 'get_product', product_id='P100')],
        [call('loop_3', 'get_product', product_id='P100')],
        on_loop_guard_done,
    ])
    harness = Harness(shop, context, model, safeguards=safeguards, retry_sleep_fn=lambda _: None)

    turns = [
        '列出我的订单，并帮我取消已发货的 A1001。',
        '那帮我取消还没发货的 A1003，原因是我买错容量了。',
        '确认取消 A1003，顺便查下 A1003 和 A1004 的退款进度，再帮我把延误的 A1001 建个高优人工工单，另外帮我取消 B2002。',
        '再反复查一下 P100 商品信息测试防死循环护栏。',
    ]
    for idx, query in enumerate(turns, 1):
        print(f'\n[Turn {idx}] 客户：{query}')
        print(f'[Turn {idx}] 客服（离线脚本）：' + harness.turn(query))
    validate_tool_pairs(harness.messages)

    # Verify tool timeout guard cleanly on a dedicated slow call.
    timeout_ctx = ContextManager(root / 'runtime', 'alice')
    timeout_shop = Shop('alice', seed_path, database_path=database, allow_actions=True)
    orig_call = timeout_shop.call

    def slow_call(name, args):
        if name == 'get_product':
            time.sleep(0.35)
        return orig_call(name, args)

    timeout_shop.call = slow_call
    timeout_model = DemoModel([
        [call('slow_1', 'get_product', product_id='P100')],
        [text('商品查询超时，已为您记录并建议稍后重试或转人工。')],
    ])
    timeout_harness = Harness(timeout_shop, timeout_ctx, timeout_model, safeguards=safeguards, retry_sleep_fn=lambda _: None)
    print('\n[护栏演示：工具超时] ' + timeout_harness.turn('测试工具超时保护。'))
    timeout_result = json.loads(timeout_harness.messages[2]['content'][0]['content'])
    observed['tool_timeout'] = timeout_result
    validate_tool_pairs(timeout_harness.messages)

    # Verify daily token budget circuit breaker cleanly.
    budget_ctx = ContextManager(root / 'runtime', 'alice')
    budget_store = DailyTokenBudgetStore(budget_file)
    tight_safeguards = SafeguardConfig(
        model_max_retries=0,
        tool_timeout_seconds=1.0,
        max_identical_tool_calls=2,
        max_consecutive_tool_failures=3,
        max_turn_tokens=100,
        max_session_tokens=100,
        daily_token_budget=100,
        budget_store_path=budget_file,
    )
    budget_model = DemoModel([[text('Should not run')]])
    budget_harness = Harness(shop, budget_ctx, budget_model, safeguards=tight_safeguards, budget_store=budget_store)
    budget_reply = budget_harness.turn('测试每日 Token 预算熔断。')
    print('[护栏演示：预算熔断] ' + budget_reply)
    observed['budget_exceeded_reply'] = budget_reply

    # Reopen SQLite from scratch to verify durability of writes!
    reopened = Shop('alice', seed_path, database_path=database, allow_actions=True)
    a1003_now = reopened.call('get_order', {'order_id': 'A1003'})
    a1003_refund = reopened.call('check_refund_status', {'order_id': 'A1003'})
    alice_tickets = reopened.call('list_my_tickets', {})
    events = [json.loads(line) for line in (context.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]

    checks = {
        'sqlite_real_file': database.read_bytes().startswith(b'SQLite format 3\x00'),
        'six_alice_orders_loaded': len(observed['orders']['data']['orders']) == 6,
        'shipped_cancel_rejected_by_state_machine': observed['shipped_cancel']['error']['code'] == 'ORDER_NOT_CANCELLABLE',
        'two_step_confirmation_requested': observed['cancel_preview']['error']['code'] == 'CONFIRMATION_REQUIRED',
        'same_turn_self_confirm_blocked': observed['self_confirm_blocked']['error']['code'] == 'CONFIRMATION_PENDING_USER_REPLY',
        'cancel_persisted_in_sqlite': a1003_now['ok'] and a1003_now['data']['status'] == 'cancelled',
        'auto_refund_created_for_paid_order': a1003_refund['ok'] and a1003_refund['data']['refund']['refund_id'] == 'RF-A1003',
        'repeat_cancel_is_idempotent': observed['cancel_idempotent']['error']['code'] == 'ORDER_ALREADY_CANCELLED',
        'ticket_created_and_deduplicated': observed['ticket_escalate']['ok'] and observed['ticket_dedup']['data']['deduplicated'],
        'tickets_persisted_in_sqlite': alice_tickets['ok'] and len(alice_tickets['data']['tickets']) == 2,
        'foreign_order_action_denied': observed['foreign_cancel']['error']['code'] == 'ACCESS_DENIED_OR_NOT_FOUND',
        'transient_429_retried_and_recovered': any(e['event'] == 'model_transient_retry' for e in events),
        'repeat_tool_loop_circuit_broken': observed['loop_blocked']['error']['code'] == 'REPEATED_TOOL_CALL_BLOCKED',
        'tool_timeout_caught_cleanly': timeout_result['error']['code'] == 'TOOL_TIMEOUT',
        'daily_token_budget_circuit_broken': budget_model.attempts == 0 and '熔断' in budget_reply,
    }
    assert all(checks.values()), checks

    report = {
        'mode': model.mode,
        'api_calls': 0,
        'database_path': str(database),
        'database_initialization': initialized,
        'budget_status': harness.budget_status(),
        'checks': checks,
        'session_dir': str(context.session_dir),
        'limitation': 'Offline scripted Stage 1 demonstration over a real isolated SQLite database; zero live LLM API calls.',
    }
    print('\n=== 阶段 1 验收项汇总 ===')
    for name, passed in checks.items():
        print(f'  [{"PASS" if passed else "FAIL"}] {name}')
    print('Token 统计：' + json.dumps(harness.budget_status(), ensure_ascii=False))
    return save_report(root, 'actions-demo-latest.json', report)
