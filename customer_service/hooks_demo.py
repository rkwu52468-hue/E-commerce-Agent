"""s04 observation hooks over real SQLite and scripted responses; zero API."""
import copy
import json
from pathlib import Path
import re
from unittest.mock import patch
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop
from .storage import initialize_database

HOOK_EVENTS = ('UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop')


def _read_audit(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def _run_case(root, customer_id):
    label = 'guest' if customer_id is None else customer_id
    context = ContextManager(root / 'runtime', label)
    database = context.session_dir / 'shop.db'
    initialize_database(database, root / 'data/shop-sqlite-seed.json')
    shop = Shop(customer_id, root / 'data/shop.json', database_path=database)
    query = ('HOOKS-' + label.upper() + '-PROMPT-ONLY：'
             + ('访客查询A1001订单，同时查询公开商品P100。' if customer_id is None
                else '按已由课堂程序绑定的alice查询A1001和B2001订单。'))
    plan = ([call('hooks_guest_private', 'get_order', order_id='A1001'),
             call('hooks_guest_public', 'get_product', product_id='P100')]
            if customer_id is None else
            [call('hooks_alice_own', 'get_order', order_id='A1001'),
             call('hooks_alice_foreign', 'get_order', order_id='B2001')])
    expected_ids = [block['id'] for block in plan]
    expected_resources = {block['id']: next(iter(block['input'].values())) for block in plan}
    observed, observer_order, dispatch_trace, sql_order_calls = [], [], [], []
    captured = {}
    audit_path = context.session_dir / 'audit.jsonl'

    def explain_actual_results(messages):
        validate_tool_pairs(messages)
        batch = messages[-1]['content']
        captured['next_model_batch'] = copy.deepcopy(batch)
        assert [block['tool_use_id'] for block in batch] == expected_ids
        outputs = {block['tool_use_id']: json.loads(block['content']) for block in batch}
        captured['actual_results'] = copy.deepcopy(outputs)
        lines = []
        for identifier in expected_ids:
            output = outputs[identifier]
            if not output['ok']:
                lines.append(identifier + '返回' + output['error']['code'])
            elif 'price_cents' in output['data']:
                data = output['data']
                lines.append(data['product_id'] + '现价' + format(data['price_cents'] / 100, '.2f')
                             + '元，来源' + data['source'])
            else:
                data = output['data']
                lines.append(data['order_id'] + '历史金额' + format(data['total_cents'] / 100, '.2f')
                             + '元，来源' + data['source'])
        return [text('；'.join(lines) + '。本轮已解释查询结果，其中仍包含权限拒绝。')]

    model = ScriptedModel([plan, explain_actual_results])
    harness = Harness(shop, context, model)

    def observe(metadata):
        # Static trusted observer: copies metadata and returns None; no business mutation.
        observed.append(copy.deepcopy(metadata))
        rows = _read_audit(audit_path)
        latest = rows[-1] if rows else {}
        observer_order.append(latest.get('event') == metadata['event']
                              and latest.get('tool_use_id') == metadata.get('tool_use_id')
                              and latest.get('turn') == metadata['turn'])
        return None

    for event in HOOK_EVENTS:
        harness.hooks.register(event, observe)

    original_dispatch = shop.call
    original_sql_order = shop.store.get_order

    def dispatch_spy(name, arguments):
        # Observe the default Pre record before entering the unchanged Shop.call.
        rows = _read_audit(audit_path)
        latest = rows[-1] if rows else {}
        dispatch_trace.append({'tool': name, 'audit_event_before_dispatch': latest.get('event'),
                               'tool_use_id': latest.get('tool_use_id'),
                               'pre_written_before_business_check': latest.get('event') == 'PreToolUse'
                                   and latest.get('tool') == name})
        return original_dispatch(name, arguments)

    def sql_order_spy(order_id, bound_customer):
        sql_order_calls.append({'order_id': order_id, 'customer_id': bound_customer})
        return original_sql_order(order_id, bound_customer)

    with patch.object(shop, 'call', side_effect=dispatch_spy), patch.object(shop.store, 'get_order', side_effect=sql_order_spy):
        answer = harness.turn(query)
    validate_tool_pairs(harness.messages)
    rows = _read_audit(audit_path)
    pre = {row['tool_use_id']: row for row in rows if row['event'] == 'PreToolUse'}
    post = {row['tool_use_id']: row for row in rows if row['event'] == 'PostToolUse'}
    stop = next(row for row in rows if row['event'] == 'Stop')
    outputs = captured['actual_results']
    expected_codes = {identifier: ('OK' if output['ok'] else output['error']['code'])
                      for identifier, output in outputs.items()}
    required = {'event', 'time_utc', 'mode', 'session_id', 'customer_id', 'turn'}
    denied_ids = [identifier for identifier in expected_ids if not outputs[identifier]['ok']]
    checks = {
        'real_isolated_sqlite_file': database.read_bytes().startswith(b'SQLite format 3\x00'),
        'complete_batch_in_next_scripted_request': [b['tool_use_id'] for b in captured['next_model_batch']] == expected_ids,
        'six_audit_records_in_lifecycle_order': [r['event'] for r in rows] == ['UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PreToolUse', 'PostToolUse', 'Stop'],
        'registered_observer_really_ran_six_times': len(observed) == 6,
        'default_audit_precedes_custom_observer': len(observer_order) == 6 and all(observer_order),
        'pre_audit_precedes_unchanged_business_dispatch': len(dispatch_trace) == 2 and all(r['pre_written_before_business_check'] for r in dispatch_trace),
        'pre_post_preserve_every_tool_id': set(pre) == set(post) == set(expected_ids),
        'resource_ids_match_valid_business_inputs': all(pre[i]['resource_id'] == post[i]['resource_id'] == expected_resources[i] and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', pre[i]['resource_id']) is not None for i in expected_ids),
        'post_codes_match_actual_results': all(post[i]['result_code'] == expected_codes[i] for i in expected_ids),
        'denials_are_audited': len(denied_ids) == 1 and all(post[i]['status'] == 'denied' for i in denied_ids),
        'post_duration_is_nonnegative_attempt_time': all(type(post[i]['duration_ms']) in (int, float) and post[i]['duration_ms'] >= 0 for i in expected_ids),
        'handler_entered_does_not_mean_sql_ran': all(post[i]['handler_called'] is True for i in expected_ids),
        'metadata_has_identity_session_turn_and_mode': all(required <= set(r) and r['customer_id'] == customer_id and r['session_id'] == context.session_id and r['turn'] == 1 and r['mode'] == model.mode for r in rows),
        'audit_omits_raw_query_arguments_and_results': all(not ({'query', 'arguments', 'result', 'messages', 'password'} & set(r)) for r in rows) and query not in audit_path.read_text(encoding='utf-8'),
        'stop_completed_even_with_one_denial': stop['status'] == 'completed' and stop['denied'] == 1 and stop['succeeded'] == 1,
        'stop_did_not_request_another_model_response': model.steps == 2,
        'sql_permission_boundary_verified': (not sql_order_calls if customer_id is None else len(sql_order_calls) == 2 and all(r['customer_id'] == 'alice' for r in sql_order_calls)),
        'tool_pairs_remain_complete': True,
    }
    if customer_id is None:
        checks['guest_private_denied_before_store_query'] = outputs[expected_ids[0]]['error']['code'] == 'AUTHENTICATION_REQUIRED' and not sql_order_calls
        checks['guest_public_product_allowed'] = outputs[expected_ids[1]]['ok'] and outputs[expected_ids[1]]['data']['price_cents'] == 19900
    else:
        checks['alice_own_order_allowed'] = outputs[expected_ids[0]]['ok'] and outputs[expected_ids[0]]['data']['total_cents'] == 16900
        checks['alice_foreign_order_denied'] = outputs[expected_ids[1]]['error']['code'] == 'ACCESS_DENIED_OR_NOT_FOUND'
    assert all(checks.values()), checks
    report = {'mode': model.mode, 'api_calls': 0, 'customer_id': customer_id,
              'identity_source': 'guest' if customer_id is None else 'explicit simulated fixture, not password login',
              'scripted_response_steps': model.steps, 'answer': answer, 'checks': checks,
              'audit_rows': rows, 'observer_metadata': observed, 'dispatch_trace': dispatch_trace,
              'store_get_order_calls': sql_order_calls, 'actual_results': outputs,
              'database_path': str(database), 'audit_path': str(audit_path), 'session_dir': str(context.session_dir)}
    (context.session_dir / 'hooks-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def run_hooks(root):
    root = Path(root)
    guest = _run_case(root, None)
    alice = _run_case(root, 'alice')
    checks = {**{'guest_' + key: value for key, value in guest['checks'].items()},
              **{'alice_' + key: value for key, value in alice['checks'].items()},
              'independent_identity_sessions': guest['session_dir'] != alice['session_dir'],
              'four_total_scripted_responses': guest['scripted_response_steps'] + alice['scripted_response_steps'] == 4}
    assert all(checks.values()), checks
    print('s04 Hooks课堂：四个观察事件均已实际执行；默认审计先记录，再调用自有观察回调。')
    print('访客：A1001需登录、P100公开可查；Store.get_order实际调用0次。')
    print('模拟alice：A1001历史169元可查、B2001被拒；此身份由课堂程序绑定，不是登录实验。')
    print('两组各6条audit，Pre/Post使用同一tool_use_id，拒绝也留痕，Post含非负尝试耗时。')
    print('Stop.status=completed表示到达正常结束观察点；本课堂回调均成功，但每组仍有1次权限拒绝，并非全部工具成功。')
    print('脚本模型响应合计4次；真实API0。审计文件不含原始query/arguments/result。')
    print('访客audit：' + guest['audit_path'])
    print('alice audit：' + alice['audit_path'])
    report = {'mode': 'scripted-offline', 'api_calls': 0, 'scripted_response_steps': 4,
              'cases': {'guest': guest, 'alice': alice}, 'checks': checks,
              'session_dir': guest['session_dir'], 'session_dirs': [guest['session_dir'], alice['session_dir']],
              'limitation': 'Trusted observation callbacks, real SQLite synthetic records and scripted model choices. No API call, real password login, production authentication, SQL timing measurement or untrusted-plugin sandbox claim.'}
    return save_report(root, 'hooks-latest.json', report)
