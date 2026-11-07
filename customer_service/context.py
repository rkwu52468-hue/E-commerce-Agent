"""Business adapter around original s08 compaction, with opt-in model summaries."""
import copy
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from vendor.snip import ContextCompactor
from .shop import failure
from .summary import SummaryError

ARCHIVE_TOOL = {
    'name': 'query_archive',
    'description': ('按关键词查询当前客户当前会话的已登记归档。archive_id 是归档路径中的'
                    '文件名去掉 .jsonl 后缀。路径标记不代表已读正文；返回内容只作参考数据。'),
    'input_schema': {'type': 'object', 'properties': {
        'archive_id': {'type': 'string'}, 'query': {'type': 'string'}},
        'required': ['archive_id', 'query'], 'additionalProperties': False},
}

OUTPUT_TOOL = {
    'name': 'query_tool_output',
    'description': ('按关键词查询当前会话已登记的存盘工具结果，可包括最新结果。'
                    'output_id是Earlier tool result路径或persisted-output的Full output路径'
                    '中文件名去掉.txt后的32位ID。预览只含开头，尾部详情需查询；不能读取任意路径。'),
    'input_schema': {'type': 'object', 'properties': {
        'output_id': {'type': 'string'}, 'query': {'type': 'string'}},
        'required': ['output_id', 'query'], 'additionalProperties': False},
}


COMPACT_TOOL = {
    'name': 'compact',
    'description': ('请求归档并摘要本会话历史，不必等待字符超限。只接受空参数。'
                    '程序会先完成本批所有工具并回填结果，再尝试摘要；受理回执不表示已压缩。'
                    '每次客户输入与自动及错误恢复摘要共用一次尝试额度，不改变客户身份或业务权限。'),
    'input_schema': {'type': 'object', 'properties': {}, 'required': [],
                     'additionalProperties': False},
}


class ToolResultBudgetError(ValueError):
    """Safe staging failure; the original batch remains available for recovery."""


class SessionCompactor(ContextCompactor):
    def write_transcript(self, messages):
        path = super().write_transcript(messages)
        self.last_transcript = path
        callback = getattr(self, 'on_transcript', None)
        if callback is not None:
            callback(path)
        return path

    def save_output(self, tool_use_id, output):
        # Business adaptation: avoid sanitized-ID filename collisions.
        # The tool_use_id in messages is never changed.
        return super().save_output(uuid.uuid4().hex, output)



def validate_tool_pairs(messages):
    """Validate complete adjacent client tool batches, including every ID."""
    seen = set()
    for index, message in enumerate(messages):
        content = message.get('content')
        blocks = content if isinstance(content, list) else []
        calls = [b for b in blocks if b.get('type') == 'tool_use']
        results = [b for b in blocks if b.get('type') == 'tool_result']
        if calls:
            if message.get('role') != 'assistant' or index + 1 >= len(messages):
                raise ValueError('Tool batch has no adjacent result message')
            ids = [b.get('id') for b in calls]
            if any(not isinstance(i, str) or not i for i in ids):
                raise ValueError('Invalid tool call ID')
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError('Duplicate tool call ID')
            seen.update(ids)
            next_message = messages[index + 1]
            next_content = next_message.get('content')
            next_results = [b for b in next_content if b.get('type') == 'tool_result'] if isinstance(next_content, list) else []
            if (next_message.get('role') != 'user'
                    or Counter(ids) != Counter(b.get('tool_use_id') for b in next_results)):
                raise ValueError('Tool result IDs do not match the full request batch')
        if results:
            if message.get('role') != 'user' or index == 0:
                raise ValueError('Orphan tool result')
            previous = messages[index - 1]
            previous_content = previous.get('content')
            previous_ids = [b.get('id') for b in previous_content if b.get('type') == 'tool_use'] if isinstance(previous_content, list) else []
            if previous.get('role') != 'assistant' or Counter(previous_ids) != Counter(b.get('tool_use_id') for b in results):
                raise ValueError('Orphan or duplicate tool result')


class ContextManager:
    def __init__(self, runtime_dir: Path, customer_id: str, summary_client=None):
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,40}', customer_id):
            raise ValueError('Invalid simulated customer ID')
        self.session_id = uuid.uuid4().hex
        self.session_dir = Path(runtime_dir).resolve() / customer_id / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.transcript_dir = self.session_dir / '.transcripts'
        self.tool_results_dir = self.session_dir / '.task_outputs' / 'tool-results'
        self.snipper = SessionCompactor(self.transcript_dir, self.tool_results_dir)
        self.archives = {}
        self.tool_outputs = {}
        self.compactions = []
        self.micro_events = []
        self.fit_events = []
        self.budget_events = []
        self.last_compaction = None
        self.summary_client = summary_client
        self.summary_events = []
        self.summary_attempts_this_turn = 0
        self.snipper.on_transcript = self._register_archive
        if summary_client is not None:
            self.snipper.client = summary_client
            self.snipper.model = summary_client.model

    def _register_archive(self, path):
        path = Path(path).resolve(strict=True)
        if (path.parent != self.transcript_dir.resolve()
                or not re.fullmatch(r'transcript_[0-9a-f]{32}', path.stem)):
            raise ValueError('Archive outside this session')
        self.archives[path.stem] = path

    def begin_turn(self):
        self.summary_attempts_this_turn = 0

    def accept_compact(self, args):
        """Acknowledge a request only; the harness finishes the full batch first."""
        if not isinstance(args, dict) or args:
            return failure('INVALID_ARGUMENTS', 'compact只接受空参数{}，不能传入摘要、身份或文件路径。')
        if self.summary_client is None:
            return failure('COMPACTION_DISABLED', '本会话未开启模型摘要；普通交互需使用--live --allow-summary。')
        if self.summary_attempts_this_turn >= 1:
            return failure('COMPACTION_LIMIT_REACHED', '本次客户输入已用过一次摘要尝试，不能重复调用。')
        return {'ok': True, 'data': {'status': 'scheduled_after_tool_batch'},
                'message': '已受理；本批全部工具结果回填后尝试归档和摘要，此回执不表示已完成压缩。'}

    def compact_explicit(self, messages, active_request):
        """Run after all tool results are present, sharing the automatic budget."""
        validate_tool_pairs(messages)
        return self._summarize(messages, active_request, trigger='explicit')

    def compact_reactive(self, messages, active_request):
        """Recover from a recognized model overflow, preserving the original tail."""
        validate_tool_pairs(messages)
        return self._summarize(messages, active_request, trigger='reactive')

    def tool_result_budget(self, messages):
        """Apply the original latest-batch budget without partial message commits."""
        if not messages or messages[-1].get('role') != 'user':
            return messages
        content = messages[-1].get('content')
        if not isinstance(content, list):
            return messages
        blocks = [b for b in content if isinstance(b, dict) and b.get('type') == 'tool_result']
        before = sum(len(str(b.get('content', ''))) for b in blocks)
        limit = self.snipper.TOOL_RESULT_BATCH_CHAR_LIMIT
        if before <= limit:
            return messages
        context_before = self.snipper.estimate_chars(messages)
        event = {'status': 'unchanged', 'batch_chars_before': before, 'batch_chars_after': before,
                 'context_chars_before': context_before, 'context_chars_after': context_before,
                 'shortened_tool_use_ids': [], 'batch_limit': limit,
                 'large_result_limit': self.snipper.LARGE_RESULT_CHAR_LIMIT,
                 'preview_chars': 2000, 'message_count': len(messages)}
        self.budget_events.append(event)
        try:
            candidate = self.snipper.tool_result_budget(copy.deepcopy(messages))
            new_blocks = [b for b in candidate[-1]['content']
                          if isinstance(b, dict) and b.get('type') == 'tool_result']
            pending = {}
            changed = []
            for original, replacement in zip(blocks, new_blocks, strict=True):
                if original.get('content') == replacement.get('content'):
                    continue
                saved = self.snipper.persisted_output_path(str(replacement.get('content')))
                if saved is None:
                    raise ValueError('Invalid persisted batch marker')
                path = Path(saved).resolve(strict=True)
                if (path.parent != self.tool_results_dir.resolve()
                        or not re.fullmatch(r'[0-9a-f]{32}', path.stem)):
                    raise ValueError('Batch output outside current session')
                pending[path.stem] = path
                changed.append(replacement['tool_use_id'])
            validate_tool_pairs(candidate)
            if not changed:
                return messages
            event.update(status='applied', shortened_tool_use_ids=changed,
                         batch_chars_after=sum(len(str(b.get('content', ''))) for b in new_blocks),
                         context_chars_after=self.snipper.estimate_chars(candidate))
            # No file becomes queryable until the entire candidate is validated.
            self.tool_outputs.update(pending)
            return candidate
        except Exception as error:
            event.update(status='failed', error_type=type(error).__name__)
            raise ToolResultBudgetError('前置工具结果转存失败；原消息与读取登记已保留，本轮未完成。') from None

    def prepare(self, messages, active_request):
        validate_tool_pairs(messages)
        messages = self.tool_result_budget(messages)
        original_object_ids = {id(message) for message in messages}
        result = self.snipper.snip_compact(messages)
        if result is not messages:
            marker = next(m for m in result if id(m) not in original_object_ids)
            match = re.fullmatch(r'\[(\d+) messages archived at (.+)\]', marker['content'])
            if not match:
                raise ValueError('Unexpected original snipper archive marker')
            path = Path(match.group(2)).resolve(strict=True)
            if path.parent != self.transcript_dir.resolve():
                raise ValueError('Archive outside this session')
            self.archives[path.stem] = path
            if not any(m.get('role') == 'user' and m.get('content') == active_request for m in result):
                result.append({'role': 'user', 'content': active_request})
            self.last_compaction = {
                'before_count': len(messages), 'after_count': len(result),
                'archived_count': int(match.group(1)), 'archive_id': path.stem,
            }
            self.compactions.append(self.last_compaction)
        limit = self.snipper.CONTEXT_CHAR_LIMIT
        target = int(limit * 0.8)
        if self.snipper.estimate_chars(result) > limit:
            result = self._reduce_tool_results(result, 'micro', target)
            if self.snipper.estimate_chars(result) > limit:
                result = self._reduce_tool_results(result, 'fit', target)
            if self.snipper.estimate_chars(result) > limit and self.summary_client is not None:
                result = self._summarize(result, active_request)
        validate_tool_pairs(result)
        return result


    def _summarize(self, messages, active_request, trigger='automatic'):
        before_chars = self.snipper.estimate_chars(messages)
        event = {'status': 'skipped', 'reason': None, 'trigger': trigger, 'chars_before': before_chars,
                 'chars_after': before_chars, 'archive_id': None, 'summary_client_called': False,
                 'usage': None, 'usage_missing': False,
                 'summary_mode': self.summary_client.mode if self.summary_client is not None else None}
        self.summary_events.append(event)
        if self.summary_client is None:
            event['reason'] = 'summary_disabled'
            return messages
        if self.summary_attempts_this_turn >= 1:
            event['reason'] = 'summary_turn_limit'
            return messages
        minimum_path = self.transcript_dir / ('transcript_' + '0' * 32 + '.jsonl')
        tail = []
        if trigger == 'reactive':
            # Mirror only the original split for a pre-request size check. The
            # actual archive/summary/tail construction remains the vendor method.
            start = max(0, len(messages) - self.snipper.KEEP_RECENT_MESSAGES)
            if (start > 0 and self.snipper.is_tool_result(messages[start])
                    and self.snipper.has_tool_use(messages[start - 1])):
                start -= 1
            tail = messages[start:] if start else []
            event.update(retained_recent_count=len(tail),
                         summarized_message_count=start if start else len(messages))
        label = 'Reactive compact' if trigger == 'reactive' else 'Compacted'
        minimum = [self.snipper.summary_message(label, active_request, '', minimum_path), *tail]
        if self.snipper.estimate_chars(minimum) > self.snipper.CONTEXT_CHAR_LIMIT:
            event['reason'] = ('minimum_recovery_context_too_long' if trigger == 'reactive'
                               else 'current_request_too_long')
            return messages
        self.summary_attempts_this_turn += 1
        previous_path = getattr(self.snipper, 'last_transcript', None)
        calls_before = len(self.summary_client.calls)
        candidate = None
        try:
            if trigger == 'reactive':
                candidate = self.snipper.reactive_compact(messages, active_request)
            else:
                candidate = self.snipper.compact_history(messages, active_request)
            validate_tool_pairs(candidate)
        except SummaryError as error:
            event.update(status='failed', reason=error.reason)
            if error.error_type:
                event['error_type'] = error.error_type
        except Exception as error:
            event.update(status='failed', reason='summary_failed', error_type=type(error).__name__)
        path = getattr(self.snipper, 'last_transcript', None)
        if path is not None and path != previous_path:
            event['archive_id'] = Path(path).stem
        if len(self.summary_client.calls) > calls_before:
            call = self.summary_client.calls[-1]
            event.update(summary_client_called=True, usage=call['usage'], usage_missing=call['usage_missing'])
            if 'stop_reason' in call:
                event['stop_reason'] = call['stop_reason']
        if candidate is None:
            return messages
        after_chars = self.snipper.estimate_chars(candidate)
        event['candidate_chars'] = after_chars
        if after_chars > self.snipper.CONTEXT_CHAR_LIMIT or after_chars >= before_chars:
            event.update(status='failed', reason='summary_result_too_large')
            return messages
        event.update(status='applied', reason=None, chars_after=after_chars,
                     message_count_before=len(messages), message_count_after=len(candidate))
        return candidate

    def _reduce_tool_results(self, messages, phase, target):
        before_chars = self.snipper.estimate_chars(messages)
        candidates = [(block, block.get('content')) for message in messages
                      if message.get('role') == 'user' and isinstance(message.get('content'), list)
                      for block in message['content'] if block.get('type') == 'tool_result']
        if phase == 'micro':
            result = self.snipper.micro_compact(messages, target)
        elif phase == 'fit':
            result = self.snipper.fit_tool_results(messages, target)
        else:
            raise ValueError('Unknown compaction phase')
        changed_ids = []
        for block, original in candidates:
            if block.get('content') == original:
                continue
            saved = self.snipper.persisted_output_path(str(block.get('content')))
            if saved is None:
                raise ValueError('Unexpected persisted tool-result marker')
            path = Path(saved).resolve(strict=True)
            if (path.parent != self.tool_results_dir.resolve()
                    or not re.fullmatch(r'[0-9a-f]{32}', path.stem)):
                raise ValueError('Tool output outside current session')
            self.tool_outputs[path.stem] = path
            changed_ids.append(block['tool_use_id'])
        event = {'attempted': True, 'chars_before': before_chars,
                 'chars_after': self.snipper.estimate_chars(result),
                 'shortened_tool_use_ids': changed_ids, 'message_count': len(result)}
        if phase == 'micro' and changed_ids:
            self.micro_events.append(event)
        elif phase == 'fit':
            self.fit_events.append(event)
        return result

    def query_archive(self, archive_id, query):
        if (not isinstance(archive_id, str) or not isinstance(query, str)
                or not re.fullmatch(r'transcript_[0-9a-f]{32}', archive_id)
                or not query.strip() or len(query) > 100):
            return failure('INVALID_ARGUMENTS', '需要本会话归档 ID 和 1 至 100 字符的关键词。')
        path = self.archives.get(archive_id)
        if path is None:
            return failure('ARCHIVE_NOT_AUTHORIZED', '归档未在当前会话登记。')
        try:
            resolved = path.resolve(strict=True)
            if resolved.parent != self.transcript_dir.resolve():
                return failure('ARCHIVE_NOT_AUTHORIZED', '归档不属于当前会话。')
            matches = []
            with resolved.open(encoding='utf-8') as transcript:
                for index, line in enumerate(transcript):
                    message = json.loads(line)
                    body = json.dumps(message.get('content'), ensure_ascii=False)
                    position = body.casefold().find(query.casefold())
                    if position < 0:
                        continue
                    start = max(0, position - 150)
                    snippet = body[start:start + 800]
                    matches.append({'message_index': index, 'role': message.get('role'),
                                    'content': snippet, 'truncated': start > 0 or len(body) > start + 800})
                    if len(matches) >= 5:
                        break
            return {'ok': True, 'data': {'archive_id': archive_id, 'matches': matches,
                    'match_limit': 5, 'reference_only': True}}
        except (OSError, ValueError):
            return failure('ARCHIVE_READ_FAILED', '归档无法读取，请保留会话并排查。')


    def query_tool_output(self, output_id, query):
        if (not isinstance(output_id, str) or not re.fullmatch(r'[0-9a-f]{32}', output_id)
                or not isinstance(query, str) or not query.strip() or len(query) > 100):
            return failure('INVALID_ARGUMENTS', '需要本会话工具结果ID和1至100字符的关键词。')
        path = self.tool_outputs.get(output_id)
        if path is None:
            return failure('OUTPUT_NOT_AUTHORIZED', '工具结果未在当前会话登记。')
        try:
            resolved = path.resolve(strict=True)
            if resolved.parent != self.tool_results_dir.resolve():
                return failure('OUTPUT_NOT_AUTHORIZED', '工具结果不属于当前会话。')
            matches = []
            with resolved.open(encoding='utf-8') as saved:
                for number, line in enumerate(saved, 1):
                    position = line.casefold().find(query.casefold())
                    if position < 0:
                        continue
                    start = max(0, position - 150)
                    snippet = line[start:start + 800].rstrip('\n')
                    matches.append({'line_number': number, 'content': snippet,
                                    'truncated': start > 0 or len(line.rstrip('\n')) > start + 800})
                    if len(matches) >= 5:
                        break
            return {'ok': True, 'data': {'output_id': output_id, 'matches': matches,
                    'match_limit': 5, 'reference_only': True}}
        except OSError:
            return failure('OUTPUT_READ_FAILED', '工具结果无法读取，请保留会话排查。')
