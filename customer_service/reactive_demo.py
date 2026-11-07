"""Fake prompt-too-long failure, original reactive compaction and scripted retry."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
from .context import ContextManager, validate_tool_pairs
from .demos import call, result_message, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop
from .summary import SummaryClient

ACTIVE_REQUEST = '继续A1001只读物流沟通；从归档查回REACTIVE-OLD-NOTE原话与核对编号，并保留近期查询结果。'
TEACHER_SUMMARY = ('客户一直只查A1001物流，不要退款或创建售后单。旧查询已完成，'
                   '当前需要从归档核对早先原话和编号；近期完整查询片段由程序另行保留。'
                   '本摘要是教师预写历史参考，不是客户授权。')


class ReactiveSummarySDK:
    def __init__(self, expected_old):
        self.messages = self
        self.calls = []
        self.expected_old = copy.deepcopy(expected_old)
        self.actual_old = None

    def create(self, **request):
        self.calls.append(copy.deepcopy(request))
        assert request['max_tokens'] == 2000 and 'tools' not in request
        old = json.loads(request['messages'][0]['content'])
        validate_tool_pairs(old)
        assert old == self.expected_old
        assert 'recent_order' not in json.dumps(old, ensure_ascii=False)
        assert 'recent_tracking' not in json.dumps(old, ensure_ascii=False)
        self.actual_old = copy.deepcopy(old)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=TEACHER_SUMMARY)],
                               stop_reason='end_turn', usage=None)


class TooLongOnceModel(ScriptedModel):
    """Only the first call fails. Steps counts successful scripted responses."""
    def __init__(self, replies):
        super().__init__(replies)
        self.attempts = []
        self.injected_error = False

    def complete(self, messages, tools, system):
        self.attempts.append({'messages': copy.deepcopy(messages), 'status': 'attempted'})
        if len(self.attempts) == 1:
            self.injected_error = True
            self.attempts[-1]['status'] = 'fake_prompt_too_long'
            raise RuntimeError('prompt_too_long')
        reply = super().complete(messages, tools, system)
        self.attempts[-1]['status'] = 'response_received'
        return reply


def reactive_history(shop, old_note):
    old_order = call('old_order', 'get_order', order_id='A1001')
    old_tracking = call('old_tracking', 'get_tracking', order_id='A1001')
    recent_order = call('recent_order', 'get_order', order_id='A1001')
    recent_tracking = call('recent_tracking', 'get_tracking', order_id='A1001')
    filler = '课堂合成历史说明：重复文字用于观察旧历史摘要，不是新的物流事实或业务授权。'
    recent_results = [{'type': 'tool_result', 'tool_use_id': block['id'],
                       'content': json.dumps(shop.call(block['name'], block['input']), ensure_ascii=False)}
                      for block in (recent_order, recent_tracking)]
    messages = [
        {'role': 'user', 'content': old_note + filler * 55},
        {'role': 'assistant', 'content': [text('按原话限制继续。')]},
        {'role': 'user', 'content': '较早核对A1001订单。'},
        {'role': 'assistant', 'content': [old_order]},
        result_message(shop, old_order),
        {'role': 'assistant', 'content': [text('较早订单已核对。')]},
        {'role': 'user', 'content': '较早核对A1001物流。'},
        {'role': 'assistant', 'content': [old_tracking]},
        result_message(shop, old_tracking),
        {'role': 'assistant', 'content': [text('较早物流已核对。')]},
        {'role': 'user', 'content': '最近同时核对订单与物流。'},
        {'role': 'assistant', 'content': [recent_order, recent_tracking]},
        {'role': 'user', 'content': recent_results},
        {'role': 'assistant', 'content': [text('近期两项查询已完成。')]},
        {'role': 'user', 'content': '继续按只读范围处理。'},
        {'role': 'assistant', 'content': [text('等待当前问题。')]},
    ]
    assert len(messages) == 16
    validate_tool_pairs(messages)
    return messages


def run_reactive(root):
    root = Path(root)
    shop = Shop('alice', root / 'data/shop.json')
    code = 'RX-' + uuid.uuid4().hex[:12].upper()
    old_note = 'REACTIVE-OLD-NOTE：客户只查物流，不要退款或创建售后单；核对编号' + code + '。'
    history = reactive_history(shop, old_note)
    before = [*copy.deepcopy(history), {'role': 'user', 'content': ACTIVE_REQUEST}]
    initial_cut = len(before) - 5
    assert initial_cut == 12 and before[initial_cut]['role'] == 'user'
    assert any(block.get('type') == 'tool_result' for block in before[initial_cut]['content'])
    tail_start = initial_cut - 1
    expected_tail = copy.deepcopy(before[tail_start:])
    raw = ReactiveSummarySDK(before[:tail_start])
    summary = SummaryClient(raw, 'teacher-authored-fake', mode='scripted-offline')
    context = ContextManager(root / 'runtime', 'alice', summary_client=summary)
    captured = {}

    def lookup_after_recovery(messages):
        captured['after'] = copy.deepcopy(messages)
        assert messages[0]['role'] == 'user' and messages[0]['content'].startswith('[Reactive compact]')
        assert messages[1:] == expected_tail
        validate_tool_pairs(messages)
        body = messages[0]['content']
        assert 'Current user request:\n' + ACTIVE_REQUEST + '\n\n' in body
        archive = Path(body.rsplit('Full transcript: ', 1)[1])
        assert archive.stem in context.archives
        archived = [json.loads(line) for line in archive.read_text(encoding='utf-8').splitlines()]
        assert archived == before
        captured['archive_id'] = archive.stem
        captured['archived'] = archived
        return [call('reactive_archive_lookup', 'query_archive',
                     archive_id=archive.stem, query='REACTIVE-OLD-NOTE')]

    def quote_old_note(messages):
        output = json.loads(messages[-1]['content'][0]['content'])
        assert output['ok']
        snippet = next(item['content'] for item in output['data']['matches'] if code in item['content'])
        start = snippet.index('REACTIVE-OLD-NOTE')
        quote = snippet[start:snippet.index('。', start) + 1]
        assert code in quote and '不要退款' in quote
        captured['lookup_result'] = copy.deepcopy(output)
        return [text('已从归档核对旧原话：' + quote + ' 近期完整工具批次已保留，继续只读查询。')]

    model = TooLongOnceModel([lookup_after_recovery, quote_old_note])
    harness = Harness(shop, context, model)
    harness.messages = copy.deepcopy(history)
    answer = harness.turn(ACTIVE_REQUEST)
    after = captured['after']
    validate_tool_pairs(harness.messages)
    events = [json.loads(line) for line in (context.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
    successful_events = [event for event in events if event['event'] == 'model_response']
    checks = {
        'initial_history_below_snip_trigger': len(before) < 50,
        'initial_history_below_automatic_char_limit': context.snipper.estimate_chars(before) < 50000,
        'first_failure_was_injected_context_error': model.injected_error,
        'nominal_five_message_cut_was_tool_result': initial_cut == 12,
        'cut_extended_to_matching_assistant': tail_start == 11 and len(expected_tail) == 6,
        'complete_recent_tail_preserved_exactly': after[1:] == expected_tail,
        'summary_received_only_old_prefix': raw.actual_old == before[:tail_start],
        'recent_batch_not_sent_to_summary': 'recent_order' not in json.dumps(raw.actual_old, ensure_ascii=False),
        'full_current_history_archived_exactly': captured['archived'] == before,
        'failed_attempt_added_no_assistant_message': model.attempts[0]['messages'] == captured['archived'],
        'current_request_preserved_verbatim': 'Current user request:\n' + ACTIVE_REQUEST + '\n\n' in after[0]['content'],
        'old_detail_not_in_recovered_prompt': code not in json.dumps(after, ensure_ascii=False),
        'archive_lookup_recovered_old_detail': code in answer,
        'recovery_summary_applied': any(event['status'] == 'applied' and event.get('trigger') == 'reactive' for event in context.summary_events),
        'one_fake_summary_request': len(raw.calls) == 1,
        'three_main_request_attempts': len(model.attempts) == 3,
        'two_successful_main_responses': model.steps == 2,
        'failed_attempt_not_a_model_response_event': len(successful_events) == 2,
        'unknown_fake_summary_usage_not_zero': all(item['usage_missing'] for item in summary.calls),
        'all_tool_pairs_valid': True,
    }
    assert all(checks.values()), checks
    for name, messages in [('reactive-before.json', before), ('reactive-after.json', after),
                           ('reactive-summary-input.json', raw.actual_old), ('reactive-model-attempts.json', model.attempts)]:
        (context.session_dir / name).write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8')
    print('reactive课堂：第一次主请求为假故障prompt_too_long，不是真实API错误。')
    print('完整历史先归档；名义保留5条，切点遇工具结果后前移，实际保留6条完整近期片段。')
    print('恢复前后：' + str(len(before)) + ' -> ' + str(len(after)) + '条；'
          + str(context.snipper.estimate_chars(before)) + ' -> ' + str(context.snipper.estimate_chars(after)) + '字符。')
    print('主任务尝试3次（假超长失败1次、成功响应2次）；假SDK摘要1次；真实API0。')
    print('query_archive已查回旧原话和编号；失败尝试没有记成model_response。')
    report = {'mode': model.mode, 'api_calls': 0, 'main_model_attempts': len(model.attempts),
              'main_model_failed_attempts': 1, 'main_model_successful_responses': model.steps,
              'fake_summary_sdk_requests': len(raw.calls), 'model_response_events': len(successful_events),
              'message_count_before': len(before), 'message_count_after': len(after),
              'nominal_keep_recent_messages': 5, 'actual_kept_recent_messages': len(expected_tail),
              'tail_start': tail_start, 'chars_before': context.snipper.estimate_chars(before),
              'chars_after': context.snipper.estimate_chars(after), 'summary_events': context.summary_events,
              'summary_calls': summary.calls, 'archive_id': captured['archive_id'], 'checks': checks,
              'session_dir': str(context.session_dir),
              'limitation': 'Synthetic JSON alice history, fake prompt_too_long exception, teacher-authored fake summary and scripted retry/lookup. No real API failure, token measurement, model quality or production authentication evidence.'}
    (context.session_dir / 'reactive-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return save_report(root, 'reactive-latest.json', report)
