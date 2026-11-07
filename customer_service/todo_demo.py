"""s05 model-reported progress with one injected tracking failure; zero API."""
import copy
import json
from pathlib import Path
import re
from unittest.mock import patch
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop, failure
from .storage import initialize_database

INITIAL_TODOS = [
    {'content': '查询A1001订单状态', 'status': 'in_progress'},
    {'content': '查询A1001物流事实', 'status': 'pending'},
    {'content': '核对delivery模拟政策', 'status': 'pending'},
    {'content': '根据核验结果提供有限建议并说明缺口', 'status': 'pending'},
]


def _projection(system):
    match = re.search(r'<current_todos>\s*(.*?)\s*</current_todos>', system, re.S)
    return json.loads(match.group(1)) if match else None


def _last_reminders(messages):
    content = messages[-1].get('content')
    return [block['text'] for block in content if block.get('type') == 'text' and '<reminder>' in block.get('text', '')] if isinstance(content, list) else []


class RecordingScriptedModel(ScriptedModel):
    def __init__(self, replies):
        super().__init__(replies)
        self.requests = []

    def complete(self, messages, tools, system):
        self.requests.append({'messages': copy.deepcopy(messages), 'system': system,
                              'current_todos': _projection(system),
                              'last_batch_reminders': _last_reminders(messages)})
        return super().complete(messages, tools, system)


def run_todo(root):
    root = Path(root)
    context = ContextManager(root / 'runtime', 'alice')
    database = context.session_dir / 'shop.db'
    initialize_database(database, root / 'data/shop-sqlite-seed.json')
    fixture = json.loads((root / 'data/shop.json').read_text(encoding='utf-8'))
    fixture['dataset_version'] = 'todo-fixture-2026-11-04'
    fixture['policies']['delivery'] = {
        'policy_id': 'delivery', 'version': 'todo-demo-2026-11-04',
        'title': '课堂模拟配送信息核验流程',
        'text': '课堂模拟流程：物流信息获取失败时，记录资料缺口并建议人工核验；不能据此判断延迟原因或承诺送达时间。',
        'source': 'todo-fixture.json#policies.delivery',
    }
    fixture_path = context.session_dir / 'todo-fixture.json'
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding='utf-8')
    shop = Shop('alice', fixture_path, database_path=database)
    captured = {}
    injected = []

    def last_result(messages, identifier):
        validate_tool_pairs(messages)
        blocks = [block for block in messages[-1]['content'] if block.get('type') == 'tool_result']
        assert len(blocks) == 1 and blocks[0]['tool_use_id'] == identifier
        output = json.loads(blocks[0]['content'])
        captured[identifier] = copy.deepcopy(output)
        return output

    def after_plan(messages):
        output = last_result(messages, 'todo_initial_plan')
        assert output['ok'] and output['data']['todos'] == INITIAL_TODOS
        captured['initial_rendered'] = output['data']['rendered']
        return [call('todo_order_lookup', 'get_order', order_id='A1001')]

    def after_order(messages):
        output = last_result(messages, 'todo_order_lookup')
        assert output['ok'] and output['data']['status'] == 'shipped'
        return [call('todo_tracking_lookup', 'get_tracking', order_id='A1001')]

    def after_tracking_failure(messages):
        output = last_result(messages, 'todo_tracking_lookup')
        assert not output['ok'] and output['error']['code'] == 'DEMO_TRACKING_UNAVAILABLE'
        return [call('todo_policy_lookup', 'get_policy', policy_id='delivery')]

    def update_progress_after_policy(messages):
        output = last_result(messages, 'todo_policy_lookup')
        assert output['ok'] and output['data']['policy_id'] == 'delivery'
        reminder = _last_reminders(messages)
        assert len(reminder) == 1
        captured['reminder'] = reminder[0]
        assert harness.todo.items == INITIAL_TODOS
        order = captured['todo_order_lookup']['data']
        code = captured['todo_tracking_lookup']['error']['code']
        policy = output['data']
        final_todos = [
            {'content': '查询' + order['order_id'] + '订单：已取得' + order['status'] + '状态与来源', 'status': 'completed'},
            {'content': '查询A1001物流：' + code + '，查询失败，尚未取得物流事实', 'status': 'in_progress'},
            {'content': '核对delivery模拟政策：已取得' + policy['version'] + '版本和来源', 'status': 'completed'},
            {'content': '提供有限建议并说明缺口；交付状态仍待随后显式更新', 'status': 'pending'},
        ]
        captured['final_todos'] = copy.deepcopy(final_todos)
        return [call('todo_progress_update', 'todo_write', todos=final_todos)]

    def explain_verified_gap(messages):
        output = last_result(messages, 'todo_progress_update')
        assert output['ok'] and output['data']['todos'] == captured['final_todos']
        assert output['data']['completed'] == 2 and output['data']['total'] == 4
        captured['final_rendered'] = output['data']['rendered']
        order = captured['todo_order_lookup']['data']
        problem = captured['todo_tracking_lookup']['error']['code']
        policy = captured['todo_policy_lookup']['data']
        answer = (order['order_id'] + '订单状态为' + order['status'] + '（已发货），来源：'
                  + order['source'] + '。物流查询返回' + problem + '，尚未取得物流事实，'
                  + '不能判断延迟原因，也不能承诺送达时间。'
                  + '依据课堂模拟政策' + policy['version'] + '（来源：' + policy['source']
                  + '），当前建议人工核验物流信息并记录资料缺口。'
                  + '清单当前2/4完成，物流仍在进行中，建议项仍为pending；'
                  + '本次已给出有限建议，但尚需随后显式更新其交付状态，不能声称整个任务已完成。')
        return [text(answer)]

    model = RecordingScriptedModel([
        [call('todo_initial_plan', 'todo_write', todos=copy.deepcopy(INITIAL_TODOS))],
        after_plan, after_order, after_tracking_failure, update_progress_after_policy, explain_verified_gap,
    ])
    harness = Harness(shop, context, model)
    real_dispatch = shop.call

    def dispatch_with_demo_failure(name, args):
        if name == 'get_tracking' and args == {'order_id': 'A1001'} and shop.customer_id == 'alice':
            output = failure('DEMO_TRACKING_UNAVAILABLE', '课堂模拟：物流查询暂不可用，未取得物流事实。')
            injected.append(copy.deepcopy(output))
            return output
        return real_dispatch(name, args)

    query = '请按多步骤核对A1001订单、物流及delivery模拟政策，再给出有依据的建议；查询失败时说明缺口，不要退款或创建售后单。'
    with patch.object(shop, 'call', side_effect=dispatch_with_demo_failure):
        answer = harness.turn(query)
    validate_tool_pairs(harness.messages)
    snapshot = json.loads((context.session_dir / 'history.json').read_text(encoding='utf-8'))
    events = [json.loads(line) for line in (context.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    audit = [json.loads(line) for line in (context.session_dir / 'audit.jsonl').read_text(encoding='utf-8').splitlines()]
    reminder_events = [event for event in events if event['event'] == 'todo_reminder']
    todos = harness.todo.items
    projections = [request['current_todos'] for request in model.requests]
    expected_ids = ['todo_initial_plan', 'todo_order_lookup', 'todo_tracking_lookup', 'todo_policy_lookup', 'todo_progress_update']
    result_ids = [block['tool_use_id'] for message in harness.messages if isinstance(message.get('content'), list)
                  for block in message['content'] if block.get('type') == 'tool_result']
    other_context = ContextManager(root / 'runtime', 'guest')
    other = Harness(Shop(None, fixture_path, database_path=database), other_context, ScriptedModel([]))
    checks = {
        'six_scripted_model_responses': model.steps == len(model.requests) == 6,
        'real_session_sqlite_and_policy_fixture': database.read_bytes().startswith(b'SQLite format 3\x00') and fixture_path.is_file(),
        'initial_update_returns_exact_whole_plan': captured['todo_initial_plan']['data']['todos'] == INITIAL_TODOS,
        'real_order_lookup_succeeded': captured['todo_order_lookup']['ok'] and captured['todo_order_lookup']['data']['total_cents'] == 16900,
        'single_recognizable_tracking_failure_injected': len(injected) == 1 and captured['todo_tracking_lookup'] == injected[0] and 'data' not in injected[0],
        'policy_really_loaded_from_session_fixture': captured['todo_policy_lookup']['data'] == fixture['policies']['delivery'],
        'one_reminder_after_three_non_update_batches': len(reminder_events) == 1 and [bool(request['last_batch_reminders']) for request in model.requests] == [False, False, False, False, True, False],
        'reminder_in_same_batch_after_policy_tool_result': model.requests[4]['messages'][-1]['content'][-1]['type'] == 'text' and model.requests[4]['messages'][-1]['content'][0]['tool_use_id'] == 'todo_policy_lookup',
        'reminder_did_not_block_policy_or_add_request': captured['todo_policy_lookup']['ok'] and model.steps == 6,
        'complete_expected_tool_ids_preserved': result_ids == expected_ids,
        'first_request_has_no_plan_projection': projections[0] is None,
        'latest_initial_plan_projected_until_explicit_update': all(projection['todos'] == INITIAL_TODOS for projection in projections[1:5]),
        'next_system_contains_updated_manager_plan': projections[5]['todos'] == todos == captured['final_todos'],
        'projection_reference_and_status_source_explicit': all(p['reference_only'] is True and p['status_source'] == 'model_reported' for p in projections[1:]),
        'final_snapshot_matches_manager': snapshot['todos'] == todos,
        'tracking_failure_remains_in_progress_with_reason': todos[1]['status'] == 'in_progress' and 'DEMO_TRACKING_UNAVAILABLE' in todos[1]['content'] and '未取得物流事实' in todos[1]['content'],
        'advice_not_precompleted_before_answer': todos[3]['status'] == 'pending',
        'two_of_four_model_reported_complete': sum(item['status'] == 'completed' for item in todos) == 2 and len(todos) == 4,
        'render_has_actual_newlines': '\n' in captured['initial_rendered'] and '\n' in captured['final_rendered'],
        'todo_tool_updates_covered_by_audit': all(sum(row['event'] == event and row.get('tool_use_id') == identifier and row.get('tool') == 'todo_write' for row in audit) == 1 for event in ('PreToolUse', 'PostToolUse') for identifier in ('todo_initial_plan', 'todo_progress_update')),
        'managers_are_instance_isolated': other.todo is not harness.todo and other.todo.items == [],
        'final_answer_names_gap_and_limited_advice': 'DEMO_TRACKING_UNAVAILABLE' in answer and '尚未取得物流事实' in answer and '人工核验' in answer and '2/4' in answer,
        'final_answer_does_not_invent_hidden_tracking_facts': '配送网点积压' not in answer and '明天送达' not in answer,
        'final_answer_preserves_actual_sources': captured['todo_order_lookup']['data']['source'] in answer and captured['todo_policy_lookup']['data']['source'] in answer,
        'all_tool_pairs_valid': True,
    }
    assert all(checks.values()), checks
    (context.session_dir / 'todo-model-requests.json').write_text(json.dumps(model.requests, ensure_ascii=False, indent=2), encoding='utf-8')
    print('s05课堂初始清单：\n' + captured['initial_rendered'])
    print('实际批末提醒：' + captured['reminder'])
    print('s05课堂最终清单：\n' + captured['final_rendered'])
    print('客服（离线脚本）：' + answer)
    print('物流失败仍in_progress并注明原因；建议仍pending，不把清单状态当成事实验收。')
    print('提醒在3个未成功更新批次后出现1次，不拦工具、不额外请求；脚本响应6次、真实API0。')
    report = {'mode': model.mode, 'api_calls': 0, 'scripted_response_steps': model.steps,
              'identity_source': 'explicit simulated alice fixture, not password login',
              'initial_todos': INITIAL_TODOS, 'final_todos': todos,
              'initial_rendered': captured['initial_rendered'], 'final_rendered': captured['final_rendered'],
              'completed': 2, 'total': 4, 'actual_results': captured,
              'reminder_events': reminder_events, 'reminder_text': captured['reminder'],
              'system_plan_projections': projections, 'answer': answer, 'checks': checks,
              'database_path': str(database), 'policy_fixture_path': str(fixture_path),
              'session_dir': str(context.session_dir), 'other_empty_manager_session': str(other_context.session_dir),
              'limitation': 'Real isolated SQLite and simulated delivery policy, one explicitly injected tracking failure and scripted planning/tool choices. No real API outage, password login, automatic fact verification, cross-process restore or model-quality evidence.'}
    (context.session_dir / 'todo-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return save_report(root, 'todo-latest.json', report)
