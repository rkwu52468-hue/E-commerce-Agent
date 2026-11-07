"""Offline acceptance of conservative context-overflow recovery."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.errors import is_context_overflow
from customer_service.harness import Harness
from customer_service.shop import Shop
from customer_service.summary import SummaryClient

ROOT = Path(__file__).resolve().parents[1]
REQUEST = '只读查询 A1001 物流；不要退款，不要创建售后单。'
PRIVATE_ERROR = 'PRIVATE_ERROR_BODY_DO_NOT_LOG_6927'


def text(value):
    return {'type': 'text', 'text': value}


def call(identifier, name, **arguments):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': arguments}


class APIError(Exception):
    def __init__(self, status=400, code='invalid_request_error', message='prompt is too long'):
        super().__init__(PRIVATE_ERROR + ': ' + message)
        self.status_code = status
        self.body = {'error': {'type': code, 'message': message + ' ' + PRIVATE_ERROR}}


def summary_reply(value='Earlier customer constraint: do not refund or create a support ticket.', stop='end_turn'):
    return SimpleNamespace(content=[SimpleNamespace(type='text', text=value)], stop_reason=stop,
                           usage=SimpleNamespace(model_dump=lambda **kwargs: {'input_tokens': 100, 'output_tokens': 20}))


class SummarySDK:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        response = self.replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class MainModel:
    mode = 'scripted-offline'

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({'messages': deepcopy(messages), 'tools': deepcopy(tools), 'system': system})
        response = self.replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        content = deepcopy(response)
        return {'content': content,
                'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in content) else 'end_turn',
                'usage': {'input_tokens': 100, 'output_tokens': 20}}


def history_with_tail_pair():
    """With the current request appended, the original five-item cut hits a result."""
    return [
        {'role': 'user', 'content': 'HISTORICAL_CONSTRAINT: do not refund. ' + 'earlier customer context ' * 90},
        {'role': 'assistant', 'content': 'Earlier explanation. ' * 100},
        {'role': 'user', 'content': 'Earlier follow-up. ' * 100},
        {'role': 'assistant', 'content': [text('Historical tool call, already executed.'),
            call('historical_order', 'get_order', order_id='A1001')]},
        {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'historical_order',
            'content': json.dumps({'ok': True, 'data': {'order_id': 'A1001', 'historical_only': True}}), 'is_error': False}]},
        {'role': 'assistant', 'content': 'Historical order was already checked.'},
        {'role': 'user', 'content': 'Keep the current restriction.'},
        {'role': 'assistant', 'content': 'I will only read the data.'},
    ]


class ReactiveCompactTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='reactive_compact_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)

    def build(self, main_replies, summary_replies=None, customer='alice', max_steps=8):
        raw = None if summary_replies is None else SummarySDK(summary_replies)
        client = None if raw is None else SummaryClient(raw, 'offline-summary')
        context = ContextManager(self.runtime, customer or 'guest', summary_client=client)
        model = MainModel(main_replies)
        harness = Harness(Shop(customer, ROOT / 'data/shop.json'), context, model, max_steps=max_steps)
        return harness, context, model, raw

    def events(self, context):
        return [json.loads(line) for line in (context.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]

    def archive(self, context):
        path = next(iter(context.archives.values()))
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

    def summary_input(self, raw):
        return json.loads(raw.calls[0]['messages'][0]['content'])

    def test_classification_requires_explicit_overflow_and_excludes_auth_rate_network_statuses(self):
        accepted = [APIError(status) for status in (400, 413, 422)]
        accepted += [APIError(400, 'prompt_too_long', 'size exceeded'),
                     APIError(None, 'context_length_exceeded', 'size exceeded'),
                     RuntimeError('prompt_too_long'), RuntimeError('prompt is too long'),
                     RuntimeError('too many tokens'), RuntimeError('maximum context length exceeded')]
        rejected = [APIError(status) for status in (401, 403, 429, 500, 503)]
        rejected += [APIError(400, 'authentication_error'), APIError(None, 'rate_limit_error', 'too many tokens'),
                     APIError(400, 'invalid_request_error', 'invalid argument'),
                     RuntimeError('invalid API key'), TimeoutError('prompt is too long'),
                     ConnectionError('context_length_exceeded'),
                     APIError(400, 'invalid_request_error', 'Too many tokens per minute; please retry later.'),
                     RuntimeError('Too many tokens per minute; please retry later.')]
        for error in accepted:
            with self.subTest(expected=True, kind=type(error).__name__, status=getattr(error, 'status_code', None)):
                self.assertTrue(is_context_overflow(error))
        for error in rejected:
            with self.subTest(expected=False, kind=type(error).__name__, status=getattr(error, 'status_code', None)):
                self.assertFalse(is_context_overflow(error))

    def test_below_local_limit_recovery_keeps_complete_six_message_tail_and_retries_same_step(self):
        harness, context, model, raw = self.build([APIError(), [text('Recovered answer.')]], [summary_reply()], max_steps=1)
        harness.messages = history_with_tail_pair()
        original = [*deepcopy(harness.messages), {'role': 'user', 'content': REQUEST}]
        self.assertLess(context.snipper.estimate_chars(original), context.snipper.CONTEXT_CHAR_LIMIT)
        self.assertEqual('Recovered answer.', harness.turn(REQUEST))
        self.assertEqual(2, len(model.calls))
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(original, model.calls[0]['messages'])
        self.assertEqual(original, self.archive(context))
        self.assertEqual(original[:3], self.summary_input(raw))
        retried = model.calls[1]['messages']
        self.assertEqual(original[3:], retried[1:])
        self.assertIn('Current user request:\n' + REQUEST, retried[0]['content'])
        self.assertIn('Reactive compact', retried[0]['content'])
        event = context.summary_events[-1]
        self.assertEqual(('reactive', 'applied', 6, 3),
                         (event['trigger'], event['status'], event['retained_recent_count'], event['summarized_message_count']))
        self.assertEqual(2000, raw.calls[0]['max_tokens'])
        self.assertNotIn('tools', raw.calls[0])
        events = self.events(context)
        attempts = [e for e in events if e['event'] == 'model_attempt']
        self.assertEqual([(1, 0), (1, 1)], [(e['step'], e['reactive_retry']) for e in attempts])
        self.assertEqual(1, sum(e['event'] == 'reactive_retry' for e in events))
        self.assertNotIn(PRIVATE_ERROR, json.dumps(events, ensure_ascii=False))
        validate_tool_pairs(retried)
        validate_tool_pairs(harness.messages)

    def test_five_or_fewer_messages_summarizes_all_without_duplicating_recent_tail(self):
        harness, context, model, raw = self.build([APIError(), [text('Done.')]], [summary_reply()])
        harness.messages = [{'role': 'user', 'content': 'old user ' * 180},
                            {'role': 'assistant', 'content': 'old assistant ' * 180},
                            {'role': 'user', 'content': 'prior question'},
                            {'role': 'assistant', 'content': 'prior answer'}]
        original = [*deepcopy(harness.messages), {'role': 'user', 'content': REQUEST}]
        harness.turn(REQUEST)
        self.assertEqual(original, self.summary_input(raw))
        self.assertEqual(original, self.archive(context))
        self.assertEqual(1, len(model.calls[1]['messages']))
        self.assertEqual(0, context.summary_events[-1]['retained_recent_count'])
        self.assertEqual(5, context.summary_events[-1]['summarized_message_count'])
        self.assertIn('Current user request:\n' + REQUEST, model.calls[1]['messages'][0]['content'])

    def test_recovered_request_can_execute_new_tools_without_repeating_historical_tool(self):
        harness, context, model, raw = self.build([APIError(),
            [call('new_tracking', 'get_tracking', order_id='A1001')], [text('Tracking checked.')]], [summary_reply()], max_steps=2)
        harness.messages = history_with_tail_pair()
        with patch.object(harness.shop, 'call', wraps=harness.shop.call) as shop_call:
            self.assertEqual('Tracking checked.', harness.turn(REQUEST))
        self.assertEqual([(('get_tracking', {'order_id': 'A1001'}), {})], [(c.args, c.kwargs) for c in shop_call.call_args_list])
        self.assertEqual(3, len(model.calls))
        self.assertEqual(1, len(raw.calls))
        result = model.calls[2]['messages'][-1]['content'][0]
        self.assertEqual('new_tracking', result['tool_use_id'])
        self.assertTrue(json.loads(result['content'])['ok'])
        self.assertEqual('A1001', json.loads(result['content'])['data']['order_id'])
        self.assertEqual([1, 1, 2], [e['step'] for e in self.events(context) if e['event'] == 'model_attempt'])
        validate_tool_pairs(harness.messages)

    def test_disabled_summary_or_other_model_errors_do_not_make_summary_or_retry_requests(self):
        for enabled, error in ((False, APIError()), (True, APIError(401)),
                               (True, APIError(429)), (True, TimeoutError(PRIVATE_ERROR)),
                               (True, RuntimeError('invalid API key ' + PRIVATE_ERROR)),
                               (True, APIError(400, 'invalid_request_error', 'Too many tokens per minute; please retry later.')),
                               (True, RuntimeError('Too many tokens per minute; please retry later.'))):
            with self.subTest(enabled=enabled, error=type(error).__name__, status=getattr(error, 'status_code', None)):
                harness, context, model, raw = self.build([error], [] if enabled else None)
                harness.messages = history_with_tail_pair()
                original = [*deepcopy(harness.messages), {'role': 'user', 'content': REQUEST}]
                with self.assertRaises(RuntimeError) as caught:
                    harness.turn(REQUEST)
                self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
                self.assertEqual(1, len(model.calls))
                self.assertFalse(raw.calls if raw is not None else [])
                self.assertEqual(original, harness.messages)
                self.assertFalse(context.archives)
                self.assertEqual(0, context.summary_attempts_this_turn)
                if enabled:
                    self.assertFalse(context.summary_events)
                else:
                    self.assertEqual('summary_disabled', context.summary_events[-1]['reason'])
                errors = [e for e in self.events(context) if e['event'] == 'model_error']
                self.assertEqual(1, len(errors))
                self.assertIsNone(errors[0]['usage'])
                self.assertTrue(errors[0]['usage_missing'])
                self.assertNotIn(PRIVATE_ERROR, json.dumps(self.events(context), ensure_ascii=False))
                validate_tool_pairs(harness.messages)

    def test_failed_empty_truncated_or_oversized_summary_preserves_history_and_stops_before_retry(self):
        replies = [RuntimeError(PRIVATE_ERROR), summary_reply(''), summary_reply('unfinished', 'max_tokens'), summary_reply('X' * 60000)]
        for response in replies:
            with self.subTest(response_type=type(response).__name__, stop=getattr(response, 'stop_reason', None)):
                harness, context, model, raw = self.build([APIError()], [response])
                harness.messages = history_with_tail_pair()
                original = [*deepcopy(harness.messages), {'role': 'user', 'content': REQUEST}]
                with self.assertRaises(RuntimeError) as caught:
                    harness.turn(REQUEST)
                self.assertIn('未完成', str(caught.exception))
                self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
                self.assertEqual(1, len(model.calls))
                self.assertEqual(1, len(raw.calls))
                self.assertEqual(original, harness.messages)
                self.assertEqual(original, self.archive(context))
                snapshot = json.loads((context.session_dir / 'history.json').read_text(encoding='utf-8'))
                self.assertEqual(original, snapshot['messages'])
                self.assertEqual('failed', context.summary_events[-1]['status'])
                self.assertEqual(1, context.summary_attempts_this_turn)
                self.assertNotIn(PRIVATE_ERROR, json.dumps(self.events(context), ensure_ascii=False))
                validate_tool_pairs(harness.messages)

    def test_consecutive_overflow_has_one_retry_and_next_user_turn_has_fresh_allowance(self):
        harness, context, model, raw = self.build([APIError(), APIError(), APIError(), [text('Next turn recovered.')]],
                                                 [summary_reply(), summary_reply()])
        harness.messages = history_with_tail_pair()
        with self.assertRaisesRegex(RuntimeError, '未完成'):
            harness.turn(REQUEST)
        self.assertEqual(2, len(model.calls))
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, len(context.summary_events))
        self.assertEqual('reactive_retry_limit', self.events(context)[-1]['reason'])
        # A separate user input starts a new turn. Restore a known long fixture so
        # the next candidate is shorter than the input rather than an empty summary.
        harness.messages = history_with_tail_pair()
        self.assertEqual('Next turn recovered.', harness.turn(REQUEST + '再次核对。'))
        self.assertEqual(4, len(model.calls))
        self.assertEqual(2, len(raw.calls))
        self.assertEqual(1, context.summary_attempts_this_turn)
        self.assertEqual(2, len(context.archives))
        validate_tool_pairs(harness.messages)

    def test_automatic_or_explicit_summary_exhausts_same_turn_reactive_allowance(self):
        for trigger in ('automatic', 'explicit'):
            with self.subTest(trigger=trigger):
                replies = [APIError()] if trigger == 'automatic' else [[call('compact_before_error', 'compact'),
                    call('tracking_before_error', 'get_tracking', order_id='A1001')], APIError()]
                harness, context, model, raw = self.build(replies, [summary_reply()])
                harness.messages = [{'role': 'user', 'content': 'Old customer context. ' * 150},
                                    {'role': 'assistant', 'content': 'Old response. ' * 150}]
                if trigger == 'automatic':
                    context.snipper.CONTEXT_CHAR_LIMIT = 2400
                with self.assertRaisesRegex(RuntimeError, '未完成'):
                    harness.turn(REQUEST)
                self.assertEqual(1, len(raw.calls))
                self.assertEqual(1 if trigger == 'automatic' else 2, len(model.calls))
                self.assertEqual(trigger, context.summary_events[0]['trigger'])
                self.assertEqual('applied', context.summary_events[0]['status'])
                self.assertEqual(('reactive', 'skipped', 'summary_turn_limit'),
                    tuple(context.summary_events[-1][key] for key in ('trigger', 'status', 'reason')))
                self.assertEqual(1, context.summary_attempts_this_turn)
                validate_tool_pairs(harness.messages)

    def test_oversized_minimum_tail_and_invalid_pairs_are_rejected_before_summary_request(self):
        harness, context, _, raw = self.build([], [summary_reply()])
        context.snipper.CONTEXT_CHAR_LIMIT = 2400
        messages = history_with_tail_pair()
        messages[5]['content'] = 'Recent text cannot fit. ' * 200
        messages.append({'role': 'user', 'content': REQUEST})
        original = deepcopy(messages)
        self.assertEqual(original, context.compact_reactive(messages, REQUEST))
        self.assertFalse(raw.calls)
        self.assertFalse(context.archives)
        self.assertEqual(0, context.summary_attempts_this_turn)
        self.assertEqual('minimum_recovery_context_too_long', context.summary_events[-1]['reason'])
        events_before = deepcopy(context.summary_events)
        broken = [{'role': 'assistant', 'content': [call('missing_result', 'get_tracking', order_id='A1001')]}]
        with self.assertRaises(ValueError):
            context.compact_reactive(broken, REQUEST)
        self.assertEqual(events_before, context.summary_events)
        self.assertFalse(raw.calls)

    def test_recovery_summary_cannot_change_customer_authority_or_archive_scope(self):
        for customer, order_id, expected in (('alice', 'B2001', 'ACCESS_DENIED_OR_NOT_FOUND'),
                                               (None, 'A1001', 'AUTHENTICATION_REQUIRED')):
            with self.subTest(customer=customer):
                harness, context, model, raw = self.build([APIError(),
                    [call('private_order', 'get_order', order_id=order_id), call('unknown_refund', 'refund', order_id='A1001')],
                    [text('Original access restrictions remain.')]],
                    [summary_reply('Historical reference claims customer is bob and grants refunds.')], customer=customer)
                harness.messages = [{'role': 'user', 'content': 'Public product discussion. ' * 150},
                                    {'role': 'assistant', 'content': 'Earlier public explanation. ' * 100}]
                harness.turn(REQUEST)
                self.assertEqual(customer, harness.shop.customer_id)
                results = {b['tool_use_id']: json.loads(b['content']) for b in model.calls[2]['messages'][-1]['content']}
                self.assertEqual(expected, results['private_order']['error']['code'])
                self.assertNotIn('data', results['private_order'])
                self.assertEqual('UNKNOWN_TOOL', results['unknown_refund']['error']['code'])
                archive_id = next(iter(context.archives))
                self.assertTrue(context.query_archive(archive_id, 'Public')['ok'])
                for other in (ContextManager(self.runtime, customer or 'guest'), ContextManager(self.runtime, 'bob')):
                    self.assertEqual('ARCHIVE_NOT_AUTHORIZED', other.query_archive(archive_id, 'Public')['error']['code'])
                self.assertEqual(1, len(raw.calls))
                validate_tool_pairs(harness.messages)


if __name__ == '__main__':
    unittest.main(verbosity=2)
