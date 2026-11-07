"""Deterministic micro-compaction lesson over synthetic logistics results."""
import copy
import json
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop


def micro_history():
    messages = []
    bodies = []
    for index, label in enumerate('ABCDEF'):
        payload = {'ok': True, 'simulated': True, 'data': {
            'order_id': 'A1001', 'snapshot': label,
            'historical_note': '历史节点编号 LEGACY-A-001，模拟早期分拣扫描。' if label == 'A' else '模拟快照',
            'events': []}}
        sequence = 0
        while len(json.dumps(payload, ensure_ascii=False)) < 10000:
            payload['data']['events'].append({'sequence': sequence, 'snapshot': label,
                'status': 'SIMULATED_SCAN', 'detail': '课程合成仓配轨迹：节点扫描与分拣记录，仅用于检验旧工具结果正文缩短。'})
            sequence += 1
        body = json.dumps(payload, ensure_ascii=False)
        bodies.append(body)
        block = call('tracking_' + label, 'get_tracking', order_id='A1001')
        messages.extend([{'role': 'user', 'content': '查询A1001的第' + label + '份模拟物流轨迹。'},
                         {'role': 'assistant', 'content': [block]},
                         {'role': 'user', 'content': [{'type': 'tool_result',
                          'tool_use_id': block['id'], 'content': body}]}])
        if label != 'F':
            messages.append({'role': 'assistant', 'content': [text('已根据该份模拟轨迹作答。')]})
    validate_tool_pairs(messages)
    return messages, bodies


def run_micro(root):
    shop = Shop('alice', root / 'data/shop.json')
    context = ContextManager(root / 'runtime', 'alice')
    messages, bodies = micro_history()
    before = copy.deepcopy(messages)
    active_request = '查询A1001的第F份模拟物流轨迹。'
    chars_before = context.snipper.estimate_chars(messages)
    after = context.prepare(messages, active_request)
    blocks = [b for m in after if m['role'] == 'user' and isinstance(m['content'], list)
              for b in m['content'] if b.get('type') == 'tool_result']
    chars_after = context.snipper.estimate_chars(after)
    assert len(before) == len(after) == 23
    assert context.micro_events[-1]['shortened_tool_use_ids'] == ['tracking_A', 'tracking_B']
    assert [b['content'] for b in blocks[2:]] == bodies[2:]
    paths = [context.snipper.persisted_output_path(blocks[i]['content']) for i in range(2)]
    from pathlib import Path
    assert all(Path(path).read_text(encoding='utf-8') == bodies[i] for i, path in enumerate(paths))
    output_id = Path(paths[0]).stem
    for name, data in [('micro-before.json', before), ('micro-after.json', after)]:
        (context.session_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def answer_from_result(messages):
        output = json.loads(messages[-1]['content'][0]['content'])
        snippet = output['data']['matches'][0]['content']
        assert 'LEGACY-A-001' in snippet
        return [text('查回旧结果A中的原文片段：' + snippet)]

    model = ScriptedModel([[call('output_lookup_1', 'query_tool_output', output_id=output_id, query='LEGACY-A-001')], answer_from_result])
    harness = Harness(shop, context, model)
    harness.messages = copy.deepcopy(after)
    answer = harness.turn('请从保存的旧物流结果中查回LEGACY-A-001的原文。')
    validate_tool_pairs(harness.messages)
    checks = {
        'same_message_count': len(before) == len(after),
        'only_A_B_shortened': context.micro_events[-1]['shortened_tool_use_ids'] == ['tracking_A', 'tracking_B'],
        'recent_C_D_E_and_unseen_F_unchanged': [b['content'] for b in blocks[2:]] == bodies[2:],
        'old_bodies_saved_exactly': True,
        'same_tool_ids_and_valid_pairs': True,
        'characters_reduced_below_limit': chars_after <= 50000 < chars_before,
        'old_fact_recovered_via_authorized_tool': 'LEGACY-A-001' in answer,
    }
    assert all(checks.values())
    print('离线micro课堂：23条合成消息；A至E已提供，F刚返回。')
    print('消息条数：' + str(len(before)) + ' -> ' + str(len(after)))
    print('JSON估算字符：' + str(chars_before) + ' -> ' + str(chars_after))
    print('已缩短：A、B；保留正文：C、D、E及新结果F。')
    print('旧结果全文核对一致；query_tool_output已查回LEGACY-A-001。')
    return save_report(root, 'micro-latest.json', {'mode': model.mode, 'api_calls': 0,
        'scripted_response_steps': model.steps, 'message_count_before': len(before),
        'message_count_after': len(after), 'chars_before': chars_before, 'chars_after': chars_after,
        'checks': checks, 'micro_events': context.micro_events,
        'session_dir': str(context.session_dir),
        'limitation': 'Synthetic logistics snapshots and scripted lookup; not a real-model behavioral test.'})
