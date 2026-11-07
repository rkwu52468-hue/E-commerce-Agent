"""s06 synchronous read-only child context, actual evidence and scripted models."""
import copy
import json
from pathlib import Path
import uuid
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop
from .storage import initialize_database

TASK_PROMPT = ('请只读调查当前程序绑定客户的A1001订单和物流，核对returns政策来源，'
               '并检查B2001是否可见。只查资料，不退款、不创建售后单；'
               '依据实际工具结果写简短报告，拒绝时不提供他人数据，不承诺送达时间。')
READ_ONLY_TOOLS = {'get_product', 'get_policy', 'get_order', 'get_tracking',
                   'list_my_orders', 'query_archive', 'query_tool_output', 'load_skill'}


class RecordingScriptedModel(ScriptedModel):
    def __init__(self, replies, label, request_order):
        super().__init__(replies)
        self.label, self.request_order = label, request_order
        self.requests = []

    def complete(self, messages, tools, system):
        self.requests.append({'messages': copy.deepcopy(messages), 'tools': copy.deepcopy(tools), 'system': system})
        self.request_order.append({'model': self.label, 'request': len(self.requests)})
        return super().complete(messages, tools, system)


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def _tool_results(messages):
    return {block['tool_use_id']: json.loads(block['content'])
            for message in messages if isinstance(message.get('content'), list)
            for block in message['content'] if block.get('type') == 'tool_result'}


def run_subagent_demo(root):
    root = Path(root)
    context = ContextManager(root / 'runtime', 'alice')
    database = context.session_dir / 'shop.db'
    initialize_database(database, root / 'data/shop-sqlite-seed.json')
    shop = Shop('alice', root / 'data/shop.json', database_path=database)
    history_canary = 'PARENT-HISTORY-CANARY-' + uuid.uuid4().hex[:12]
    todo_canary = 'PARENT-TODO-CANARY-' + uuid.uuid4().hex[:12]
    child_canary = 'CHILD-INTERMEDIATE-CANARY-' + uuid.uuid4().hex[:12]
    captured, request_order = {}, []
    child_plan = [
        call('sub_own_order', 'get_order', order_id='A1001'),
        call('sub_tracking', 'get_tracking', order_id='A1001'),
        call('sub_returns', 'get_policy', policy_id='returns'),
        call('sub_foreign_order', 'get_order', order_id='B2001'),
    ]

    def child_first_request(messages):
        assert messages == [{'role': 'user', 'content': TASK_PROMPT}]
        assert {tool['name'] for tool in child_model.requests[0]['tools']} == READ_ONLY_TOOLS
        assert history_canary not in json.dumps(messages, ensure_ascii=False)
        assert todo_canary not in child_model.requests[0]['system']
        return [text(child_canary), *copy.deepcopy(child_plan)]

    def child_report_from_actual_results(messages):
        validate_tool_pairs(messages)
        outputs = _tool_results(messages)
        assert set(outputs) == {block['id'] for block in child_plan}
        own, tracking, policy, foreign = [outputs[key] for key in
            ('sub_own_order', 'sub_tracking', 'sub_returns', 'sub_foreign_order')]
        assert own['ok'] and tracking['ok'] and policy['ok']
        assert not foreign['ok'] and 'data' not in foreign
        captured['child_actual_results'] = copy.deepcopy(outputs)
        order = own['data']
        parcel = tracking['data']
        rule = policy['data']
        report = (order['order_id'] + '订单状态' + order['status'] + '，历史金额'
                  + format(order['total_cents'] / 100, '.2f') + '元，来源' + order['source']
                  + '；物流状态' + parcel['status'] + '，最新记录为'
                  + parcel['events'][-1]['detail'] + '，来源' + parcel['source']
                  + '；已核对returns政策' + rule['version'] + '，来源' + rule['source']
                  + '；B2001返回' + foreign['error']['code'] + '，未取得该订单数据。'
                  + '仅完成只读调查，不办理退款或售后，不承诺送达时间。')
        captured['child_report'] = report
        return [text(report)]

    child_model = RecordingScriptedModel([child_first_request, child_report_from_actual_results], 'child', request_order)

    def parent_answer_from_task(messages):
        validate_tool_pairs(messages)
        output = json.loads(messages[-1]['content'][0]['content'])
        assert messages[-1]['content'][0]['tool_use_id'] == 'parent_readonly_task'
        assert output['ok'] and output['data']['reference_only'] is True
        data = output['data']
        evidence = {item['tool_use_id']: item['result'] for item in data['evidence']}
        assert evidence == captured['child_actual_results']
        assert data['report'] == captured['child_report']
        captured['task_result'] = copy.deepcopy(output)
        own = evidence['sub_own_order']['data']
        tracking = evidence['sub_tracking']['data']
        policy = evidence['sub_returns']['data']
        denied = evidence['sub_foreign_order']
        answer = ('根据子任务参考报告及实际工具证据：' + own['order_id'] + '已发货，历史金额'
                  + format(own['total_cents'] / 100, '.2f') + '元（' + own['source'] + '）。'
                  + '物流为' + tracking['status'] + '，最新记录：' + tracking['events'][-1]['detail']
                  + '（' + tracking['source'] + '）。returns政策版本' + policy['version']
                  + '（' + policy['source'] + '）。B2001返回' + denied['error']['code']
                  + '，没有他人订单数据可提供。继续只读，不退款、不创建售后单，也不承诺送达时间。')
        return [text(answer)]

    parent_model = RecordingScriptedModel([
        [call('parent_readonly_task', 'task', prompt=TASK_PROMPT)],
        parent_answer_from_task,
    ], 'parent', request_order)
    parent = Harness(shop, context, parent_model, subagent_model=child_model)
    parent.messages = [
        {'role': 'user', 'content': history_canary},
        {'role': 'assistant', 'content': [text('这是仅供父会话的历史夹具。')]},
    ]
    parent.todo.call({'todos': [{'content': todo_canary, 'status': 'in_progress'}]})
    answer = parent.turn('请把A1001的只读资料核验交给独立子任务，核对B2001权限后按证据汇总。')
    data = captured['task_result']['data']
    child_dir = Path(data['child_session_dir'])
    child_snapshot = json.loads((child_dir / 'history.json').read_text(encoding='utf-8'))
    child_messages = child_snapshot['messages']
    validate_tool_pairs(parent.messages)
    validate_tool_pairs(child_messages)
    parent_audit = _read_lines(context.session_dir / 'audit.jsonl')
    child_audit = _read_lines(child_dir / 'audit.jsonl')
    runs = copy.deepcopy(parent.subagents.runs)
    evidence = data['evidence']
    parent_result_ids = list(_tool_results(parent.messages))
    actual_child_results = _tool_results(child_messages)
    child_tools = {tool['name'] for tool in child_model.requests[0]['tools']}
    checks = {
        'parent_two_and_child_two_scripted_responses': parent_model.steps == child_model.steps == 2,
        'synchronous_request_order': request_order == [{'model': 'parent', 'request': 1}, {'model': 'child', 'request': 1}, {'model': 'child', 'request': 2}, {'model': 'parent', 'request': 2}],
        'real_isolated_sqlite_seed': database.read_bytes().startswith(b'SQLite format 3\x00'),
        'child_first_messages_only_explicit_prompt': child_model.requests[0]['messages'] == [{'role': 'user', 'content': TASK_PROMPT}],
        'parent_history_canary_really_present': history_canary in json.dumps(parent_model.requests[0]['messages'], ensure_ascii=False),
        'parent_todo_canary_really_projected': todo_canary in parent_model.requests[0]['system'],
        'parent_history_not_automatically_in_child': all(history_canary not in json.dumps(request, ensure_ascii=False) for request in child_model.requests),
        'parent_todo_not_automatically_in_child': all(todo_canary not in json.dumps(request, ensure_ascii=False) for request in child_model.requests),
        'child_plan_starts_and_remains_empty': child_snapshot['todos'] == [] and '<current_todos>' not in child_model.requests[0]['system'],
        'parent_plan_not_changed_by_child': parent.todo.items == [{'content': todo_canary, 'status': 'in_progress'}],
        'child_advertises_exact_read_only_allowlist': child_tools == READ_ONLY_TOOLS,
        'child_has_no_task_todo_compact_or_file_sql_tool': not child_tools & {'task', 'todo_write', 'compact', 'read_file', 'write_file', 'bash', 'sql'},
        'child_same_program_bound_alice_identity': len(runs) == 1 and runs[0]['customer_id'] == 'alice' and all(row['customer_id'] == 'alice' for row in child_audit),
        'own_order_success_with_actual_amount': actual_child_results['sub_own_order']['ok'] and actual_child_results['sub_own_order']['data']['total_cents'] == 16900,
        'tracking_delay_is_actual_returned_state': actual_child_results['sub_tracking']['ok'] and actual_child_results['sub_tracking']['data']['status'] == 'delayed',
        'returns_policy_has_actual_source': actual_child_results['sub_returns']['ok'] and actual_child_results['sub_returns']['data']['source'] == 'shop.json#policies.returns',
        'foreign_order_truly_denied_without_data': actual_child_results['sub_foreign_order']['error']['code'] == 'ACCESS_DENIED_OR_NOT_FOUND' and 'data' not in actual_child_results['sub_foreign_order'],
        'child_middle_assistant_text_not_copied_to_parent': child_canary in json.dumps(child_messages, ensure_ascii=False) and child_canary not in json.dumps(parent.messages, ensure_ascii=False),
        'parent_only_has_task_top_level_result': parent_result_ids == ['parent_readonly_task'],
        'four_actual_tool_results_in_reference_evidence': len(evidence) == 4 and {e['tool_use_id']: e['result'] for e in evidence} == actual_child_results,
        'evidence_preserves_tool_names_and_ids': {e['tool_use_id']: e['tool'] for e in evidence} == {block['id']: block['name'] for block in child_plan},
        'bounded_report_and_evidence': len(data['report']) <= 3000 and len(json.dumps(evidence, ensure_ascii=False)) <= 6000,
        'reference_only_task_receipt': data['reference_only'] is True,
        'receipt_links_child_session_and_two_requests': data['child_session_id'] == child_dir.name and child_dir != context.session_dir and data['model_requests'] == 2,
        'run_receipt_completed_once': len(runs) == 1 and runs[0]['status'] == 'completed' and runs[0]['child_session_id'] == data['child_session_id'],
        'fake_usage_records_remain_unknown': len(data['usage']['records']) == 2 and data['usage']['complete'] is False,
        'parent_task_pre_post_audit_link': len(parent_audit) == 4 and sum(row['event'] == 'PreToolUse' and row.get('tool_use_id') == 'parent_readonly_task' and row.get('tool') == 'task' for row in parent_audit) == 1 and sum(row['event'] == 'PostToolUse' and row.get('tool_use_id') == 'parent_readonly_task' and row.get('result_code') == 'OK' for row in parent_audit) == 1,
        'child_independent_ten_audit_rows': len(child_audit) == 10 and all(row['session_id'] == data['child_session_id'] for row in child_audit),
        'child_foreign_denial_audited': any(row['event'] == 'PostToolUse' and row.get('tool_use_id') == 'sub_foreign_order' and row.get('status') == 'denied' for row in child_audit),
        'final_answer_uses_actual_evidence_sources': all(actual_child_results[key]['data']['source'] in answer for key in ('sub_own_order', 'sub_tracking', 'sub_returns')),
        'all_parent_child_tool_pairs_valid': True,
    }
    assert all(checks.values()), checks
    for directory, name, requests in [(context.session_dir, 'subagent-parent-requests.json', parent_model.requests),
                                       (child_dir, 'subagent-child-requests.json', child_model.requests)]:
        (directory / name).write_text(json.dumps(requests, ensure_ascii=False, indent=2), encoding='utf-8')
    print('s06课堂：沿用教师推荐的模拟alice；这是程序身份夹具，不是登录或学员独立选择验收。')
    print('父2次、子2次脚本响应，同步顺序父→子→子→父，真实API0。')
    print('子首请求仅含明确任务prompt；父历史/计划未自动进入，子中间消息未回填父messages。')
    print('子只读工具实际查A1001/物流/returns并拒绝B2001，未返回他人订单数据。')
    print('父收到有界参考报告及4项实际证据；父4条/子10条audit可由task回执关联。')
    print('父客服（离线脚本）：' + answer)
    print('子会话：' + str(child_dir))
    report = {'mode': 'scripted-offline', 'api_calls': 0, 'parent_scripted_responses': 2,
              'child_scripted_responses': 2, 'total_scripted_responses': 4,
              'identity_source': 'teacher-recommended simulated alice fixture, not password login or independent student answer',
              'task_prompt': TASK_PROMPT, 'task_result': captured['task_result'],
              'run_receipts': runs, 'request_order': request_order, 'answer': answer,
              'parent_audit_count': len(parent_audit), 'child_audit_count': len(child_audit),
              'child_session_dir': str(child_dir), 'checks': checks, 'session_dir': str(context.session_dir),
              'limitation': 'Real isolated SQLite synthetic data and scripted parent/child models. Fresh prompt and plan isolation, actual foreign-order denial and actual evidence checked. Cross-registry/hidden-tool hard rejects belong to separate automated tests, not this classroom. No real API, login, autonomous delegation, concurrency or OS sandbox evidence.'}
    (context.session_dir / 'subagent-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return save_report(root, 'subagent-latest.json', report)
