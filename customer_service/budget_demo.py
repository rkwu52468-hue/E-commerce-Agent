"""Front tool-result budget in the actual harness; synthetic fixture, zero API."""
import copy
import json
from pathlib import Path
import re
import uuid
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop

ACTIVE_REQUEST = '查询A1001这份课程合成长物流结果，核对BUDGET-TAIL尾部编号，不要根据开头预览猜测尾部。'


def below_batch_gate(root):
    context = ContextManager(root / 'runtime', 'alice')
    body = 'x' * 35001
    use = call('budget_below_gate', 'get_tracking', order_id='A1001')
    before = [
        {'role': 'user', 'content': '课堂预算边界：单条超过30000，批次未超过200000。'},
        {'role': 'assistant', 'content': [use]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': use['id'], 'content': body}]},
    ]
    after = context.prepare(copy.deepcopy(before), before[0]['content'])
    validate_tool_pairs(after)
    checks = {
        'single_exceeds_30000': len(body) > 30000,
        'batch_under_200000': len(body) <= 200000,
        'messages_unchanged': after == before,
        'no_budget_event_below_batch_gate': not context.budget_events,
        'no_persisted_output_authorized': not context.tool_outputs,
        'no_micro_fit_or_summary': not context.micro_events and not context.fit_events and not context.summary_events,
    }
    assert all(checks.values()), checks
    report = {'single_result_chars': len(body), 'batch_chars': len(body),
              'context_json_chars': context.snipper.estimate_chars(before),
              'budget_events': context.budget_events, 'checks': checks,
              'session_dir': str(context.session_dir), 'api_calls': 0}
    (context.session_dir / 'budget-below-gate.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def run_budget(root):
    root = Path(root)
    boundary = below_batch_gate(root)
    context = ContextManager(root / 'runtime', 'alice')
    code = 'BUDGET-TAIL-' + uuid.uuid4().hex[:12].upper()
    fixture = json.loads((root / 'data/shop.json').read_text(encoding='utf-8'))
    fixture['dataset_version'] = 'budget-fixture-2026-11-03'
    order = fixture['orders']['A1001']
    order['source'] = 'budget-fixture.json#orders.A1001'
    unit = '课程合成物流扫描数据；仅用于观察前置大结果预算，不代表新的业务事实。'
    order['tracking']['events'] = [{'status': 'SIMULATED_VERBOSE_SCAN',
                                     'detail': unit * (210000 // len(unit) + 1)}]
    order['tracking']['tail_note'] = {'code': code, 'detail': '尾部核对编号只存在于本次完整合成工具结果中。'}
    fixture_path = context.session_dir / 'budget-fixture.json'
    fixture_path.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding='utf-8')
    shop = Shop('alice', fixture_path)
    original_output = shop.call('get_tracking', {'order_id': 'A1001'})
    original_body = json.dumps(original_output, ensure_ascii=False)
    assert len(original_body) > 200000
    captured = {}

    def lookup_after_budget(messages):
        captured['after'] = copy.deepcopy(messages)
        result = messages[-1]['content'][0]
        assert result['tool_use_id'] == 'budget_tracking_new'
        marker = result['content']
        assert marker.startswith('<persisted-output>\n')
        preview = marker.split('Preview:\n', 1)[1].removesuffix('\n</persisted-output>')
        assert preview == original_body[:2000]
        assert code not in json.dumps(messages, ensure_ascii=False)
        saved = Path(context.snipper.persisted_output_path(marker))
        assert saved.stem in context.tool_outputs
        assert saved.read_text(encoding='utf-8') == original_body
        captured['saved_path'] = saved
        captured['preview'] = preview
        reconstructed = copy.deepcopy(messages)
        reconstructed[-1]['content'][0]['content'] = original_body
        captured['before'] = reconstructed
        return [call('budget_tail_lookup', 'query_tool_output', output_id=saved.stem, query='BUDGET-TAIL')]

    def quote_tail(messages):
        output = json.loads(messages[-1]['content'][0]['content'])
        assert output['ok']
        snippet = next(item['content'] for item in output['data']['matches'] if code in item['content'])
        recovered = re.search(r'BUDGET-TAIL-[0-9A-F]{12}', snippet).group(0)
        captured['lookup_result'] = copy.deepcopy(output)
        return [text('已从完整存盘物流结果查回尾部编号：' + recovered + '。这是课程模拟核对，不承诺送达时间。')]

    model = ScriptedModel([
        [call('budget_tracking_new', 'get_tracking', order_id='A1001')],
        lookup_after_budget,
        quote_tail,
    ])
    harness = Harness(shop, context, model)
    answer = harness.turn(ACTIVE_REQUEST)
    validate_tool_pairs(harness.messages)
    before, after = captured['before'], captured['after']
    event = context.budget_events[0]
    events = [json.loads(line) for line in (context.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    tracking_event = next(item for item in events if item['event'] == 'tool_result' and item['tool'] == 'get_tracking')
    checks = {
        'below_gate_scenario_passed': all(boundary['checks'].values()),
        'batch_and_single_exceed_separate_thresholds': len(original_body) > 200000 and len(original_body) > 30000,
        'one_front_budget_event_applied': len(context.budget_events) == 1 and event['status'] == 'applied',
        'event_uses_default_batch_limit': event['batch_limit'] == 200000,
        'event_uses_default_single_limit': event['large_result_limit'] == 30000,
        'event_uses_2000_preview': event['preview_chars'] == 2000,
        'event_batch_sum_is_body_length': event['batch_chars_before'] == len(original_body),
        'event_shortened_exact_new_result': event['shortened_tool_use_ids'] == ['budget_tracking_new'],
        'event_context_before_matches_actual_batch': event['context_chars_before'] == context.snipper.estimate_chars(before),
        'event_context_after_matches_model_prompt': event['context_chars_after'] == context.snipper.estimate_chars(after),
        'next_model_receives_under_50000': context.snipper.estimate_chars(after) < 50000,
        'no_micro_fit_or_summary': not context.micro_events and not context.fit_events and not context.summary_events,
        'same_message_count_and_tool_id': len(before) == len(after) == 3 and before[-1]['content'][0]['tool_use_id'] == after[-1]['content'][0]['tool_use_id'],
        'preview_is_exact_first_2000_chars': captured['preview'] == original_body[:2000],
        'full_original_body_saved_exactly': captured['saved_path'].read_text(encoding='utf-8') == original_body,
        'actual_tool_event_matches_saved_source': tracking_event['result'] == original_output,
        'tail_absent_from_preview': code not in captured['preview'],
        'tail_recovered_from_authorized_file': code in answer,
        'three_scripted_task_responses': model.steps == 3,
        'all_tool_pairs_valid': True,
    }
    assert all(checks.values()), checks
    for name, value in [('budget-before.json', before), ('budget-after.json', after)]:
        (context.session_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    print('前置预算课堂A：单条35001 > 30000，但批总35001 <= 200000，不落盘、不触micro或fit。')
    print('前置预算课堂B：新结果正文批总' + str(event['batch_chars_before']) + ' -> ' + str(event['batch_chars_after']) + '字符。')
    print('整份messages JSON估算：' + str(event['context_chars_before']) + ' -> ' + str(event['context_chars_after']) + '字符；消息仍为3条、ID不变。')
    print('模型实际收到全文路径+准确前2000字符预览，尾部编号经query_tool_output查回。')
    print('本次只触发前置tool_result_budget；micro/fit/摘要均未执行。脚本响应3次、真实API0。')
    report = {'mode': model.mode, 'api_calls': 0, 'scripted_response_steps': model.steps,
              'thresholds': {'tool_result_batch_chars': 200000, 'large_single_result_chars': 30000, 'preview_chars': 2000},
              'below_gate_scenario': boundary, 'budget_events': context.budget_events,
              'message_count_before': len(before), 'message_count_after': len(after),
              'batch_chars_before': event['batch_chars_before'], 'batch_chars_after': event['batch_chars_after'],
              'context_chars_before': event['context_chars_before'], 'context_chars_after': event['context_chars_after'],
              'output_id': captured['saved_path'].stem, 'checks': checks,
              'session_dir': str(context.session_dir),
              'limitation': 'Session-local synthetic JSON logistics fixture and scripted tool choices/answer. Actual harness budget, file contents and authorized retrieval checked. No real API, model token measurement or production identity evidence.'}
    (context.session_dir / 'budget-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return save_report(root, 'budget-latest.json', report)
