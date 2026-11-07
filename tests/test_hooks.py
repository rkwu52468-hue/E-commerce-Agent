"""Offline acceptance of session-local observation hooks and audit boundaries."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.hooks import EVENTS, HookRegistry
from customer_service.shop import Shop
from customer_service.storage import initialize_database

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ERROR = 'PRIVATE_HOOK_ERROR_BODY_3829'
PRIVATE_QUERY = 'PRIVATE_CUSTOMER_QUERY_5842'
PRIVATE_ARGUMENT = 'PRIVATE_RAW_ARGUMENT_6174'


def text(value):
    return {'type': 'text', 'text': value}


def call(identifier, name, **arguments):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': arguments}


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


class HookTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='hooks_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)

    def build(self, replies, customer='alice', sqlite=False, max_steps=8):
        database = None
        if sqlite:
            database = self.runtime / 'shop.db'
            initialize_database(database, ROOT / 'data/shop-sqlite-seed.json')
        shop = Shop(customer, ROOT / 'data/shop.json', database_path=database)
        context = ContextManager(self.runtime, customer or 'guest')
        model = MainModel(replies)
        return Harness(shop, context, model, max_steps=max_steps), context, model

    def audit(self, context):
        path = context.session_dir / 'audit.jsonl'
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []

    def events(self, context):
        path = context.session_dir / 'events.jsonl'
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []

    def results(self, messages):
        return {block['tool_use_id']: json.loads(block['content'])
                for message in messages if isinstance(message.get('content'), list)
                for block in message['content'] if block.get('type') == 'tool_result'}

    def post_rows(self, context):
        return {row['tool_use_id']: row for row in self.audit(context) if row['event'] == 'PostToolUse'}

    def assert_safe_hook_errors(self, context):
        errors = [event for event in self.events(context) if event['event'] == 'hook_error']
        self.assertTrue(errors)
        self.assertNotIn(PRIVATE_ERROR, json.dumps(errors, ensure_ascii=False))
        for event in errors:
            self.assertTrue(event.get('error_type'))
            self.assertNotIn('message', event)
            self.assertNotIn('payload', event)
            self.assertNotIn('arguments', event)
            self.assertNotIn('result', event)

    def test_registry_validates_events_and_callbacks_and_instances_are_independent(self):
        self.assertEqual(('UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop'), EVENTS)
        registry = HookRegistry()
        invoked = []
        registry.register('PreToolUse', lambda payload: invoked.append(payload['tool']))
        self.assertEqual([], registry.trigger('PreToolUse', {'tool': 'get_product'}))
        self.assertEqual(['get_product'], invoked)
        self.assertEqual([], HookRegistry().trigger('PreToolUse', {'tool': 'get_order'}))
        for event in ('UnknownEvent', '', None):
            with self.subTest(event=event):
                with self.assertRaises((ValueError, TypeError)):
                    registry.register(event, lambda payload: None)
        for callback in (None, 'not callable', {}):
            with self.subTest(callback=callback):
                with self.assertRaises((ValueError, TypeError)):
                    registry.register('Stop', callback)

    def test_registry_callbacks_receive_independent_deepcopies_and_all_errors_are_collected(self):
        registry = HookRegistry()
        original = {'nested': {'customer_id': 'alice'}, 'tool': 'get_order'}
        order, snapshots = [], []

        def mutate(payload):
            order.append(0)
            payload['nested']['customer_id'] = 'bob'
            payload['added'] = PRIVATE_ARGUMENT

        def explode(payload):
            order.append(1)
            snapshots.append(deepcopy(payload))
            raise RuntimeError(PRIVATE_ERROR)

        def non_none(payload):
            order.append(2)
            return False

        def last(payload):
            order.append(3)
            snapshots.append(deepcopy(payload))

        for callback in (mutate, explode, non_none, last):
            registry.register('PreToolUse', callback)
        errors = registry.trigger('PreToolUse', original)
        self.assertEqual([0, 1, 2, 3], order)
        self.assertEqual([original, original], snapshots)
        self.assertEqual({'nested': {'customer_id': 'alice'}, 'tool': 'get_order'}, original)
        self.assertEqual([1, 2], [entry['callback_index'] for entry in errors])
        self.assertTrue(all(set(entry) == {'callback_index', 'error_type'} and isinstance(entry['error_type'], str) for entry in errors))
        self.assertNotIn(PRIVATE_ERROR, json.dumps(errors))
        self.assertNotIn(PRIVATE_ARGUMENT, json.dumps(errors))

    def test_guest_private_batch_is_audited_but_never_queries_sql_orders(self):
        calls = [call('guest_order', 'get_order', order_id='A1001'),
                 call('guest_tracking', 'get_tracking', order_id='A1001'), call('guest_list', 'list_my_orders')]
        harness, context, model = self.build([calls, [text('Please log in locally.')]], customer=None, sqlite=True)
        with patch.object(harness.shop, '_call_database', side_effect=AssertionError('Private dispatch must stop before SQL')) as dispatch, \
             patch.object(harness.shop.store, 'get_order', side_effect=AssertionError('No private SQL lookup')) as order_query, \
             patch.object(harness.shop.store, 'list_orders', side_effect=AssertionError('No private SQL list')) as list_query:
            self.assertEqual('Please log in locally.', harness.turn('Read my order and tracking.'))
        dispatch.assert_not_called()
        order_query.assert_not_called()
        list_query.assert_not_called()
        actual = self.results(model.calls[1]['messages'])
        self.assertEqual({c['id'] for c in calls}, set(actual))
        self.assertTrue(all(result['error']['code'] == 'AUTHENTICATION_REQUIRED' and 'data' not in result for result in actual.values()))
        rows = self.audit(context)
        self.assertEqual(['UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PreToolUse', 'PostToolUse',
                          'PreToolUse', 'PostToolUse', 'Stop'], [row['event'] for row in rows])
        self.assertTrue(all(row['customer_id'] is None for row in rows))
        for row in self.post_rows(context).values():
            self.assertEqual('denied', row['status'])
            self.assertEqual('AUTHENTICATION_REQUIRED', row['result_code'])
        self.assertEqual('completed', rows[-1]['status'])
        validate_tool_pairs(harness.messages)

    def test_guest_public_success_has_metadata_only_audit_and_default_callback_runs_first(self):
        harness, context, model = self.build([[call('public_product', 'get_product', product_id='P100')],
                                               [text('MODEL_RESPONSE_PRIVATE_1956')]], customer=None)
        observed = []

        def observer(payload):
            rows = self.audit(context)
            observed.append((rows[-1]['event'], deepcopy(payload)))

        harness.hooks.register('PreToolUse', observer)
        harness.turn(PRIVATE_QUERY)
        self.assertEqual(1, len(observed))
        self.assertEqual('PreToolUse', observed[0][0])
        row = self.post_rows(context)['public_product']
        self.assertEqual('success', row['status'])
        self.assertTrue(row['handler_called'])
        self.assertEqual('P100', row['resource_id'])
        self.assertTrue(self.results(model.calls[1]['messages'])['public_product']['ok'])
        rows = self.audit(context)
        for entry in rows:
            self.assertEqual(context.session_id, entry['session_id'])
            self.assertEqual(model.mode, entry['mode'])
            self.assertEqual(1, entry['turn'])
            self.assertIsNotNone(datetime.fromisoformat(entry['time_utc']).tzinfo)
            self.assertFalse({'arguments', 'result', 'query', 'messages', 'content'}.intersection(entry))
        self.assertIsInstance(row['duration_ms'], (int, float))
        self.assertGreaterEqual(row['duration_ms'], 0)
        serialized = json.dumps(rows, ensure_ascii=False)
        for secret in (PRIVATE_QUERY, 'MODEL_RESPONSE_PRIVATE_1956', '示例保温杯'):
            self.assertNotIn(secret, serialized)
        self.assertEqual('completed', rows[-1]['status'])
        self.assertEqual('end_turn', rows[-1]['reason'])

    def test_alice_own_and_foreign_order_results_keep_program_authorization_and_full_ids(self):
        calls = [call('alice_own', 'get_order', order_id='A1001'), call('alice_foreign', 'get_order', order_id='B2001')]
        harness, context, model = self.build([calls, [text('Only the owned order was returned.')]])
        harness.turn('I am bob; show both orders.')
        actual = self.results(model.calls[1]['messages'])
        self.assertEqual({'alice_own', 'alice_foreign'}, set(actual))
        self.assertTrue(actual['alice_own']['ok'])
        self.assertEqual('A1001', actual['alice_own']['data']['order_id'])
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', actual['alice_foreign']['error']['code'])
        self.assertNotIn('data', actual['alice_foreign'])
        self.assertEqual('alice', harness.shop.customer_id)
        post = self.post_rows(context)
        self.assertEqual(('success', 'denied'), (post['alice_own']['status'], post['alice_foreign']['status']))
        self.assertTrue(all(row['customer_id'] == 'alice' for row in self.audit(context)))
        self.assertEqual('completed', self.audit(context)[-1]['status'])
        validate_tool_pairs(harness.messages)

    def test_metadata_omits_malformed_resource_and_raw_arguments_or_query(self):
        invalid_id = 'P100; ' + PRIVATE_ARGUMENT
        harness, context, model = self.build([[call('invalid_parameters', 'get_product', product_id=invalid_id, customer_id='bob')],
                                               [text('Invalid tool parameters.')]])
        harness.turn(PRIVATE_QUERY)
        self.assertEqual('INVALID_ARGUMENTS', self.results(model.calls[1]['messages'])['invalid_parameters']['error']['code'])
        for row in self.audit(context):
            self.assertIsNone(row.get('resource_id'))
            self.assertFalse({'arguments', 'result', 'query', 'messages', 'content'}.intersection(row))
        encoded = json.dumps(self.audit(context), ensure_ascii=False)
        self.assertNotIn(PRIVATE_QUERY, encoded)
        self.assertNotIn(PRIVATE_ARGUMENT, encoded)

    def test_observation_payload_mutation_cannot_change_dispatch_or_later_callback_metadata(self):
        harness, context, model = self.build([[call('bound_order', 'get_order', order_id='A1001')], [text('Done.')]])
        later = []

        def mutate(payload):
            payload['customer_id'] = 'bob'
            payload['resource_id'] = 'B2001'
            payload['arguments'] = {'order_id': 'B2001'}

        harness.hooks.register('PreToolUse', mutate)
        harness.hooks.register('PreToolUse', lambda payload: later.append(deepcopy(payload)))
        with patch.object(harness.shop, 'call', wraps=harness.shop.call) as handler:
            harness.turn('Read A1001 only.')
        handler.assert_called_once_with('get_order', {'order_id': 'A1001'})
        self.assertEqual('alice', later[0]['customer_id'])
        self.assertEqual('A1001', later[0]['resource_id'])
        self.assertNotIn('arguments', later[0])
        self.assertEqual('A1001', self.results(model.calls[1]['messages'])['bound_order']['data']['order_id'])
        self.assertEqual('alice', self.post_rows(context)['bound_order']['customer_id'])

    def test_pre_hook_exception_or_non_none_return_skips_handlers_and_closes_whole_batch(self):
        for kind in ('exception', 'non_none'):
            with self.subTest(kind=kind):
                calls = [call('pre_first', 'get_order', order_id='A1001'), call('pre_later', 'get_tracking', order_id='A1001')]
                harness, context, model = self.build([calls])
                observed = []

                def fail(payload):
                    if kind == 'exception':
                        raise RuntimeError(PRIVATE_ERROR)
                    return {'attempted_replacement': PRIVATE_ARGUMENT}

                harness.hooks.register('PreToolUse', fail)
                harness.hooks.register('PreToolUse', lambda payload: observed.append(payload['tool_use_id']))
                with patch.object(harness, 'call_tool', side_effect=AssertionError('Pre failure cannot call handler')) as handler:
                    answer = harness.turn('Read my order.')
                handler.assert_not_called()
                self.assertIsInstance(answer, str)
                self.assertEqual(1, len(model.calls))
                self.assertEqual(['pre_first', 'pre_later'], observed)
                self.assertEqual({'pre_first', 'pre_later'}, set(self.results(harness.messages)))
                self.assertTrue(all(v['error']['code'] == 'HOOK_FAILED' for v in self.results(harness.messages).values()))
                post = self.post_rows(context)
                self.assertEqual({'pre_first', 'pre_later'}, set(post))
                self.assertTrue(all(row['status'] == 'error' and not row['handler_called'] for row in post.values()))
                self.assertEqual('incomplete', self.audit(context)[-1]['status'])
                self.assert_safe_hook_errors(context)
                validate_tool_pairs(harness.messages)

    def test_post_hook_failure_preserves_actual_result_skips_later_handler_and_stops_model(self):
        calls = [call('post_first', 'get_order', order_id='A1001'), call('post_later', 'get_tracking', order_id='A1001')]
        harness, context, model = self.build([calls])
        observed = []

        def fail(payload):
            if payload['tool_use_id'] == 'post_first':
                raise ValueError(PRIVATE_ERROR)

        harness.hooks.register('PostToolUse', fail)
        harness.hooks.register('PostToolUse', lambda payload: observed.append(payload['tool_use_id']))
        with patch.object(harness.shop, 'call', wraps=harness.shop.call) as handler:
            answer = harness.turn('Read my order and tracking.')
        handler.assert_called_once_with('get_order', {'order_id': 'A1001'})
        self.assertIsInstance(answer, str)
        self.assertEqual(1, len(model.calls))
        actual = self.results(harness.messages)
        self.assertEqual({'post_first', 'post_later'}, set(actual))
        self.assertTrue(actual['post_first']['ok'])
        self.assertEqual('A1001', actual['post_first']['data']['order_id'])
        self.assertEqual('HOOK_FAILED', actual['post_later']['error']['code'])
        self.assertEqual(['post_first', 'post_later'], observed)
        post = self.post_rows(context)
        self.assertEqual('success', post['post_first']['status'])
        self.assertTrue(post['post_first']['handler_called'])
        self.assertEqual('error', post['post_later']['status'])
        self.assertFalse(post['post_later']['handler_called'])
        self.assertEqual('incomplete', self.audit(context)[-1]['status'])
        self.assert_safe_hook_errors(context)
        validate_tool_pairs(harness.messages)

    def test_handler_exception_becomes_safe_paired_tool_error_and_post_audit(self):
        harness, context, model = self.build([[call('broken_handler', 'get_order', order_id='A1001')]], max_steps=1)
        with patch.object(harness.shop, 'call', side_effect=RuntimeError(PRIVATE_ERROR)) as handler:
            harness.turn('Read my order.')
        handler.assert_called_once_with('get_order', {'order_id': 'A1001'})
        actual = self.results(harness.messages)['broken_handler']
        self.assertEqual('TOOL_EXECUTION_ERROR', actual['error']['code'])
        self.assertNotIn(PRIVATE_ERROR, json.dumps(actual, ensure_ascii=False))
        post = self.post_rows(context)['broken_handler']
        self.assertEqual('error', post['status'])
        self.assertEqual('TOOL_EXECUTION_ERROR', post['result_code'])
        self.assertTrue(post['handler_called'])
        self.assertGreaterEqual(post['duration_ms'], 0)
        self.assertEqual(1, len(model.calls))
        self.assertEqual('incomplete', self.audit(context)[-1]['status'])
        self.assertNotIn(PRIVATE_ERROR, json.dumps(self.events(context), ensure_ascii=False))
        validate_tool_pairs(harness.messages)

    def test_model_exception_still_fires_one_stop_with_error_status_and_safe_reason(self):
        harness, context, model = self.build([RuntimeError(PRIVATE_ERROR)])
        with self.assertRaises(RuntimeError) as caught:
            harness.turn(PRIVATE_QUERY)
        self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
        rows = self.audit(context)
        self.assertEqual(['UserPromptSubmit', 'Stop'], [row['event'] for row in rows])
        self.assertEqual('error', rows[-1]['status'])
        self.assertEqual('exception', rows[-1]['reason'])
        self.assertEqual(1, len(model.calls))
        self.assertNotIn(PRIVATE_ERROR, json.dumps(rows, ensure_ascii=False))
        self.assertNotIn(PRIVATE_QUERY, json.dumps(rows, ensure_ascii=False))

    def test_character_budget_stops_before_model_and_still_emits_incomplete_stop(self):
        harness, context, model = self.build([])
        answer = harness.turn('User text cannot be shortened by tool result compactors. ' * 1500)
        self.assertIsInstance(answer, str)
        self.assertEqual([], model.calls)
        rows = self.audit(context)
        self.assertEqual(['UserPromptSubmit', 'Stop'], [row['event'] for row in rows])
        self.assertEqual('incomplete', rows[-1]['status'])

    def test_harness_registry_isolation_turn_counters_and_direct_dispatch_without_hooks(self):
        first, context_one, _ = self.build([[text('First answer.')], [text('Second answer.')]])
        second, context_two, _ = self.build([[text('Other customer answer.')]], customer='bob')
        self.assertIsNot(first.hooks, second.hooks)
        seen = []
        first.hooks.register('UserPromptSubmit', lambda payload: seen.append(payload['turn']))
        self.assertTrue(first.call_tool('get_product', {'product_id': 'P100'})['ok'])
        self.assertFalse(self.audit(context_one))
        first.turn('Question one.')
        second.turn('Other question.')
        first.turn('Question two.')
        self.assertEqual([1, 2], seen)
        self.assertEqual([1, 2], [row['turn'] for row in self.audit(context_one) if row['event'] == 'Stop'])
        self.assertEqual([1], [row['turn'] for row in self.audit(context_two) if row['event'] == 'Stop'])
        self.assertTrue(all(row['customer_id'] == 'alice' for row in self.audit(context_one)))
        self.assertTrue(all(row['customer_id'] == 'bob' for row in self.audit(context_two)))


    def test_stop_callback_failure_appends_incomplete_outcome_without_retriggering_stop_or_tools(self):
        for kind in ('exception', 'non_none'):
            with self.subTest(kind=kind):
                calls = [call('stop_own', 'get_order', order_id='A1001'),
                         call('stop_foreign', 'get_order', order_id='B2001')]
                harness, context, model = self.build([calls, [text('Original model answer.')]])
                stop_calls, later_stop_calls = [], []

                def fail(payload):
                    stop_calls.append(deepcopy(payload))
                    if kind == 'exception':
                        raise RuntimeError(PRIVATE_ERROR)
                    return False

                harness.hooks.register('Stop', fail)
                harness.hooks.register('Stop', lambda payload: later_stop_calls.append(deepcopy(payload)))
                with patch.object(harness.shop, 'call', wraps=harness.shop.call) as handler:
                    answer = harness.turn('Read only my order; refuse the foreign order.')
                self.assertIsInstance(answer, str)
                self.assertEqual(2, len(model.calls))
                self.assertEqual([('get_order', {'order_id': 'A1001'}), ('get_order', {'order_id': 'B2001'})],
                                 [entry.args for entry in handler.call_args_list])
                self.assertEqual(1, len(stop_calls))
                self.assertEqual(1, len(later_stop_calls))
                actual = self.results(harness.messages)
                self.assertEqual({'stop_own', 'stop_foreign'}, set(actual))
                self.assertTrue(actual['stop_own']['ok'])
                self.assertEqual('A1001', actual['stop_own']['data']['order_id'])
                self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', actual['stop_foreign']['error']['code'])
                rows = self.audit(context)
                self.assertEqual(7, len(rows))
                self.assertEqual(['UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'PreToolUse', 'PostToolUse',
                                  'Stop', 'TurnOutcome'], [row['event'] for row in rows])
                self.assertEqual('completed', rows[-2]['status'])
                self.assertEqual('incomplete', rows[-1]['status'])
                self.assertEqual('hook_failure', rows[-1]['reason'])
                self.assertEqual(rows[-2]['turn'], rows[-1]['turn'])
                self.assertEqual(context.session_id, rows[-1]['session_id'])
                self.assert_safe_hook_errors(context)
                self.assertNotIn(PRIVATE_ERROR, json.dumps(rows, ensure_ascii=False))
                validate_tool_pairs(harness.messages)


if __name__ == '__main__':
    unittest.main(verbosity=2)
