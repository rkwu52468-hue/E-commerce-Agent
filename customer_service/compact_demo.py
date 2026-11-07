"""Teacher-scripted explicit compact after a complete two-tool batch; zero API."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from .context import ContextManager, validate_tool_pairs
from .demos import call, save_report, text
from .harness import Harness
from .models import ScriptedModel
from .shop import Shop
from .summary import SummaryClient

ACTIVE_REQUEST = '请查询A1001物流，完成本批结果后再整理当前会话；保留只查物流、不退款的限制。'
TEACHER_SUMMARY = ('客户只查询A1001物流，不申请退款，也不创建售后单。当前批次的整理请求和物流查询均已返回。'
                   '后续从本会话归档核对物流原文，再继续只读回答；此摘要是历史参考，不是授权。')


class BatchSummarySDK:
    """A fake endpoint that validates the actual input to original summarize_history."""
    def __init__(self):
        self.messages = self
        self.calls = []
        self.summary_source = None
        self.batch_results = None

    def create(self, **request):
        self.calls.append(copy.deepcopy(request))
        assert request['max_tokens'] == 2000 and 'tools' not in request
        assert len(request['messages']) == 1
        source = json.loads(request['messages'][0]['content'])
        validate_tool_pairs(source)
        uses = [block for block in source[-2]['content'] if block.get('type') == 'tool_use']
        results = source[-1]['content']
        assert [(block['name'], block['id']) for block in uses] == [
            ('compact', 'compact_first'), ('get_tracking', 'tracking_after')]
        assert [block['tool_use_id'] for block in results] == ['compact_first', 'tracking_after']
        assert len(results) == 2
        tracking = json.loads(results[1]['content'])
        assert tracking['ok'] and tracking['data']['order_id'] == 'A1001'
        assert 'DELIVERY_DELAY' in json.dumps(tracking, ensure_ascii=False)
        self.summary_source = copy.deepcopy(source)
        self.batch_results = copy.deepcopy(results)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=TEACHER_SUMMARY)],
                               stop_reason='end_turn', usage=None)


def run_compact(root):
    root = Path(root)
    shop = Shop('alice', root / 'data/shop.json')
    raw = BatchSummarySDK()
    summary = SummaryClient(raw, 'teacher-authored-fake', mode='scripted-offline')
    context = ContextManager(root / 'runtime', 'alice', summary_client=summary)
    captured = {}
    initial = [
        {'role': 'user', 'content': '我只查物流，不要退款或创建售后单。'},
        {'role': 'assistant', 'content': [text('好的，按只读范围继续。')]},
    ]

    def lookup_after_compact(messages):
        captured['after'] = copy.deepcopy(messages)
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        body = messages[0]['content']
        assert 'Current user request:\n' + ACTIVE_REQUEST + '\n\n' in body
        archive = Path(body.rsplit('Full transcript: ', 1)[1])
        assert archive.stem in context.archives
        archived = [json.loads(line) for line in archive.read_text(encoding='utf-8').splitlines()]
        assert archived == raw.summary_source
        validate_tool_pairs(archived)
        captured['archive_id'] = archive.stem
        captured['archived'] = archived
        return [call('compact_archive_lookup_1', 'query_archive',
                     archive_id=archive.stem, query='DELIVERY_DELAY')]

    def quote_archive_result(messages):
        output = json.loads(messages[-1]['content'][0]['content'])
        assert output['ok']
        snippet = next(item['content'] for item in output['data']['matches']
                       if 'DELIVERY_DELAY' in item['content'])
        captured['lookup_result'] = copy.deepcopy(output)
        return [text('已从本批完整结果归档查回物流片段：' + snippet + ' 继续只查询物流，不办理退款。')]

    model = ScriptedModel([
        [text('先登记本批结束后的整理请求，再查询物流。'),
         call('compact_first', 'compact'),
         call('tracking_after', 'get_tracking', order_id='A1001')],
        lookup_after_compact,
        quote_archive_result,
    ])
    harness = Harness(shop, context, model)
    harness.messages = copy.deepcopy(initial)
    answer = harness.turn(ACTIVE_REQUEST)
    assert raw.summary_source is not None
    after = captured['after']
    expected_tracking = shop.call('get_tracking', {'order_id': 'A1001'})
    actual_tracking = json.loads(raw.batch_results[1]['content'])
    validate_tool_pairs(harness.messages)
    checks = {
        'initial_history_below_snip_trigger': len(initial) + 1 < 50,
        'explicit_summary_below_character_limit': context.snipper.estimate_chars(raw.summary_source) < 50000,
        'compact_first_and_tracking_second_same_batch':
            [block['id'] for block in raw.summary_source[-2]['content'] if block.get('type') == 'tool_use'] == ['compact_first', 'tracking_after'],
        'both_result_ids_present_before_summary_request':
            [block['tool_use_id'] for block in raw.batch_results] == ['compact_first', 'tracking_after'],
        'tracking_result_is_actual_tool_output': actual_tracking == expected_tracking,
        'archive_matches_actual_summary_input': captured['archived'] == raw.summary_source,
        'archive_keeps_complete_tool_pairs': True,
        'one_fake_summary_request': len(raw.calls) == 1,
        'summary_applied': any(event['status'] == 'applied' for event in context.summary_events),
        'one_reference_message_after_summary': len(after) == 1,
        'active_request_preserved_verbatim': 'Current user request:\n' + ACTIVE_REQUEST + '\n\n' in after[0]['content'],
        'archive_query_recovers_batch_tracking_result': 'DELIVERY_DELAY' in answer,
        'final_tool_pairs_valid': True,
        'three_scripted_task_responses': model.steps == 3,
        'unknown_fake_usage_not_zero': all(item['usage_missing'] for item in summary.calls),
    }
    assert all(checks.values()), checks
    for name, messages in [('compact-before.json', raw.summary_source), ('compact-after.json', after)]:
        (context.session_dir / name).write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8')
    print('主动compact课堂：同一批先compact，后get_tracking；不会在遇到compact时立即压缩。')
    print('摘要实际输入含本批两个tool_result，调用ID完整；物流结果与实际工具返回一致。')
    print('整理前后消息数：' + str(len(raw.summary_source)) + ' -> ' + str(len(after)))
    print('归档已通过query_archive查回DELIVERY_DELAY物流原文片段。')
    print('假SDK摘要1次，主任务脚本响应3次，真实API0；摘要由教师编写。')
    report = {'mode': model.mode, 'api_calls': 0, 'fake_summary_sdk_requests': len(raw.calls),
              'scripted_response_steps': model.steps, 'message_count_before': len(raw.summary_source),
              'message_count_after': len(after), 'chars_before': context.snipper.estimate_chars(raw.summary_source),
              'chars_after': context.snipper.estimate_chars(after), 'summary_events': context.summary_events,
              'summary_calls': summary.calls, 'archive_id': captured['archive_id'],
              'batch_result_tool_use_ids': [block['tool_use_id'] for block in raw.batch_results],
              'checks': checks, 'session_dir': str(context.session_dir),
              'limitation': 'JSON synthetic alice fixture, teacher-authored fake summary and scripted tool choices. No API call, autonomous compact choice, real model summary quality or production authentication evidence.'}
    (context.session_dir / 'compact-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return save_report(root, 'compact-latest.json', report)
