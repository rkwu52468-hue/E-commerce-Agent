"""Original summary path with a teacher-authored fake response; zero API calls."""
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

CONSTRAINT = '历史约束 SUMMARY-CONSTRAINT-001：我只查物流，不要退款或创建售后单。'
ACTIVE_REQUEST = '继续处理A1001的物流查询，请先核对我之前对退款和售后的限制，引用原话。'
TEACHER_SUMMARY = ('客户正在查询A1001物流，尚需核对先前的操作范围限制。'
                   '请从本会话归档查回客户原话，再继续只读查询。'
                   '这是教师编写、故意没有包含完整限制的摘要示例；不是客户授权。')


class TeacherSummarySDK:
    """A fake messages.create endpoint; returns fixed text, never makes a network call."""
    def __init__(self):
        self.messages = self
        self.calls = []

    def create(self, **request):
        self.calls.append(copy.deepcopy(request))
        assert request['max_tokens'] == 2000
        assert 'tools' not in request
        assert len(request['messages']) == 1
        assert CONSTRAINT in request['messages'][0]['content']
        return SimpleNamespace(content=[SimpleNamespace(type='text', text=TEACHER_SUMMARY)],
                               stop_reason='end_turn', usage=None)


def summary_history():
    detail = '合成会话详情：反复核对包裹运输节点，不代表新的业务状态或操作授权。'
    messages = [
        {'role': 'user', 'content': '请只查询我的A1001订单物流。'},
        {'role': 'assistant', 'content': [text('接下来核对查询范围。')]},
        {'role': 'user', 'content': CONSTRAINT},
        {'role': 'assistant', 'content': [text('已收到该限制。')]},
        {'role': 'user', 'content': detail * (55000 // len(detail) + 1)},
        {'role': 'assistant', 'content': [text('上述内容为课堂合成历史，尚待核对客户原话。')]},
    ]
    validate_tool_pairs(messages)
    return messages


def run_summary(root):
    raw = TeacherSummarySDK()
    summary = SummaryClient(raw, 'teacher-authored-fake', mode='scripted-offline')
    context = ContextManager(root / 'runtime', 'alice', summary_client=summary)
    before = copy.deepcopy(summary_history())
    expected_archive = [*copy.deepcopy(before), {'role': 'user', 'content': ACTIVE_REQUEST}]
    captured = {}

    def lookup_from_summary(messages):
        captured['after'] = copy.deepcopy(messages)
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        body = messages[0]['content']
        assert 'Current user request:\n' + ACTIVE_REQUEST + '\n\n' in body
        assert TEACHER_SUMMARY in body
        assert '不要退款' not in body
        archive_path = Path(body.rsplit('Full transcript: ', 1)[1])
        assert archive_path.stem in context.archives
        captured['archive_id'] = archive_path.stem
        archived = [json.loads(line) for line in archive_path.read_text(encoding='utf-8').splitlines()]
        assert archived == expected_archive
        captured['archived'] = archived
        return [call('summary_archive_lookup_1', 'query_archive',
                     archive_id=archive_path.stem, query='SUMMARY-CONSTRAINT-001')]

    def answer_from_result(messages):
        output = json.loads(messages[-1]['content'][0]['content'])
        assert output['ok']
        original = next(item['content'] for item in output['data']['matches']
                        if 'SUMMARY-CONSTRAINT-001' in item['content'])
        assert '不要退款' in original
        return [text('已通过query_archive查回客户原话：' + original + ' 因此继续只查询物流。')]

    model = ScriptedModel([lookup_from_summary, answer_from_result])
    harness = Harness(Shop('alice', root / 'data/shop.json'), context, model)
    harness.messages = copy.deepcopy(before)
    answer = harness.turn(ACTIVE_REQUEST)
    after = captured['after']
    validate_tool_pairs(harness.messages)
    for name, messages in [('summary-before.json', expected_archive), ('summary-after.json', after)]:
        (context.session_dir / name).write_text(
            json.dumps(messages, ensure_ascii=False, indent=2), encoding='utf-8')
    checks = {
        'below_snip_message_trigger': len(expected_archive) < 50,
        'before_over_default_50000': context.snipper.estimate_chars(expected_archive) > 50000,
        'no_tool_result_can_be_reduced': not any(isinstance(m['content'], list) and
             any(b.get('type') == 'tool_result' for b in m['content']) for m in expected_archive),
        'one_fake_summary_request': len(raw.calls) == 1,
        'current_request_preserved_verbatim': 'Current user request:\n' + ACTIVE_REQUEST + '\n\n' in after[0]['content'],
        'current_messages_archived_exactly': captured['archived'] == expected_archive,
        'single_summary_user_message': len(after) == 1 and after[0]['role'] == 'user',
        'summary_under_default_limit': context.snipper.estimate_chars(after) <= 50000,
        'teacher_summary_omitted_constraint': '不要退款' not in after[0]['content'],
        'archive_tool_recovered_constraint': '不要退款' in answer,
        'two_scripted_task_responses': model.steps == 2,
        'tool_pairs_valid': True,
        'missing_usage_is_not_reported_as_zero': all(c.get('usage_missing') for c in summary.calls),
    }
    assert all(checks.values()), checks
    print('离线摘要课堂：普通聊天正文过长，micro与fit无法缩短，进入原版compact_history。')
    print('摘要前后消息数：' + str(len(expected_archive)) + ' -> ' + str(len(after)))
    print('JSON估算字符数：' + str(context.snipper.estimate_chars(expected_archive)) + ' -> ' + str(context.snipper.estimate_chars(after)))
    print('假SDK摘要请求：1次；主任务脚本响应：2次；真实API：0次。')
    print('当前请求逐字保留；摘要故意漏掉的限制，已通过query_archive查回并放入messages。')
    print('摘要文本由教师编写，不是模型真实生成；无真实token用量或质量结论。')
    return save_report(root, 'summary-latest.json', {
        'mode': model.mode, 'api_calls': 0, 'fake_summary_sdk_requests': len(raw.calls),
        'scripted_response_steps': model.steps, 'message_count_before': len(expected_archive),
        'message_count_after': len(after), 'chars_before': context.snipper.estimate_chars(expected_archive),
        'chars_after': context.snipper.estimate_chars(after), 'summary_events': context.summary_events,
        'summary_calls': summary.calls, 'checks': checks, 'session_dir': str(context.session_dir),
        'limitation': 'Synthetic history, teacher-authored fake summary and scripted archive lookup. No API call, actual token measurement, live summary quality or autonomous recall evidence.'})
