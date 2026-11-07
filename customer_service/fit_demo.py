"""Offline demonstration: micro protects four results, fit shortens the largest."""
import copy
import json
from pathlib import Path
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop


def fit_history():
    messages, bodies = [], []
    for label, target in zip('ABCD', [6000, 7000, 8000, 30000]):
        payload = {'ok': True, 'simulated': True, 'data': {
            'order_id': 'A1001', 'snapshot': label, 'events': []}}
        while len(json.dumps(payload, ensure_ascii=False)) < target:
            number = len(payload['data']['events'])
            payload['data']['events'].append({'sequence': number, 'status': 'SIMULATED_SCAN',
                'detail': '课程合成物流节点，仅用于验证正文预算、预览与完整结果查回。'})
        if label == 'D':
            payload['data']['tail_error'] = {'code': 'FIT-TAIL-001',
                'reason': '模拟配送回传延迟，需等待下一次状态更新；没有可承诺的送达时间。'}
        body = json.dumps(payload, ensure_ascii=False)
        bodies.append(body)
        block = call('fit_' + label, 'get_tracking', order_id='A1001')
        messages.extend([{'role': 'user', 'content': '查询模拟物流快照' + label + '。'},
            {'role': 'assistant', 'content': [block]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': block['id'], 'content': body}]}])
        if label != 'D':
            messages.append({'role': 'assistant', 'content': [text('已按该份合成快照作答。')]})
    assert len(messages) == 15
    validate_tool_pairs(messages)
    return messages, bodies


def run_fit(root):
    shop = Shop('alice', root / 'data/shop.json')
    context = ContextManager(root / 'runtime', 'alice')
    messages, bodies = fit_history()
    before = copy.deepcopy(messages)
    chars_before = context.snipper.estimate_chars(messages)
    assert chars_before > context.snipper.CONTEXT_CHAR_LIMIT
    assert context.snipper.unseen_tool_result_positions(messages) == {(14, 0)}
    after = context.prepare(messages, '查询模拟物流快照D。')
    blocks = [b for m in after if m['role'] == 'user' and isinstance(m['content'], list)
              for b in m['content'] if b.get('type') == 'tool_result']
    chars_after = context.snipper.estimate_chars(after)
    assert not context.micro_events
    event = context.fit_events[-1]
    assert event['chars_before'] == chars_before
    assert event['shortened_tool_use_ids'] == ['fit_D']
    assert [b['content'] for b in blocks[:3]] == bodies[:3]
    marker = blocks[3]['content']
    assert marker.startswith('<persisted-output>\n')
    preview = marker.split('Preview:\n', 1)[1].removesuffix('\n</persisted-output>')
    assert preview == bodies[3][:1000]
    assert 'FIT-TAIL-001' not in json.dumps(after, ensure_ascii=False)
    saved = Path(context.snipper.persisted_output_path(marker))
    assert saved.read_text(encoding='utf-8') == bodies[3]
    assert saved.stem in context.tool_outputs
    assert len(before) == len(after) == 15
    for name, value in [('fit-before.json', before), ('fit-after.json', after)]:
        (context.session_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

    def answer_from_result(messages):
        result = json.loads(messages[-1]['content'][0]['content'])
        snippet = next(item['content'] for item in result['data']['matches'] if 'FIT-TAIL-001' in item['content'])
        return [text('已从完整存盘结果查回尾部异常原文：' + snippet)]

    model = ScriptedModel([[call('fit_lookup_1', 'query_tool_output', output_id=saved.stem, query='FIT-TAIL-001')], answer_from_result])
    harness = Harness(shop, context, model)
    harness.messages = copy.deepcopy(after)
    answer = harness.turn('请从存盘物流快照D查回尾部FIT-TAIL-001的原文。')
    validate_tool_pairs(harness.messages)
    checks = {
        'micro_protected_three_consumed_and_latest': not context.micro_events,
        'fit_selected_largest_latest_D': event['shortened_tool_use_ids'] == ['fit_D'],
        'same_count_and_valid_tool_ids': len(before) == len(after) == 15,
        'first_1000_characters_preserved': preview == bodies[3][:1000],
        'full_output_saved_exactly': saved.read_text(encoding='utf-8') == bodies[3],
        'tail_fact_absent_from_preview': 'FIT-TAIL-001' not in marker,
        'authorized_query_recovered_tail_fact': 'FIT-TAIL-001' in answer,
        'under_target': chars_after <= 40000,
    }
    assert all(checks.values())
    print('离线fit课堂：A、B、C为最近3个已提供结果；D刚返回，且正文最大。')
    print('micro_compact：无可缩短旧结果，字符数仍为' + str(chars_before))
    print('fit_tool_results：' + str(chars_before) + ' -> ' + str(chars_after) + '字符；缩短D。')
    print('消息条数：15 -> 15；调用ID不变。')
    print('D保留1000字符预览，全文核对一致；尾部事实不在预览中。')
    print('query_tool_output：已从完整文件查回FIT-TAIL-001。')
    return save_report(root, 'fit-latest.json', {'mode': model.mode, 'api_calls': 0,
        'scripted_response_steps': model.steps, 'message_count_before': 15, 'message_count_after': 15,
        'chars_before': chars_before, 'chars_after_micro': event['chars_before'], 'chars_after_fit': chars_after,
        'fit_events': context.fit_events, 'checks': checks, 'session_dir': str(context.session_dir),
        'limitation': 'Synthetic logistics and scripted lookup. No real API, model-generated summary, or real-model retrieval performance test.'})
