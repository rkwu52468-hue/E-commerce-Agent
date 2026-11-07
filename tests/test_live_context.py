"""Offline acceptance of the default-50000 micro+fit live trial."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from customer_service.context import OUTPUT_TOOL, ContextManager, validate_tool_pairs
import customer_service.live_context as live_context
from vendor.snip import ContextCompactor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'model': 'deepseek-flash', 'base_url': 'https://api.deepseek.com/anthropic'}
PRIVATE_ERROR = 'PRIVATE_LIVE_CONTEXT_ERROR_7142'
_DEFAULT = object()


def text(value):
    return {'type': 'text', 'text': value}


def response(content, stop='end_turn', usage=_DEFAULT):
    if usage is _DEFAULT:
        usage = {'input_tokens': 100, 'output_tokens': 10}
    return {'content': content, 'stop_reason': stop, 'usage': usage}


class SDKObject:
    def __init__(self, value):
        self.value = deepcopy(value)

    def model_dump(self, **kwargs):
        return deepcopy(self.value)

    def __getattr__(self, name):
        try:
            return self.value[name]
        except KeyError:
            raise AttributeError(name) from None


class SDKResponse:
    def __init__(self, value):
        self.content = [SDKObject(block) for block in value['content']]
        self.stop_reason = value['stop_reason']
        self.usage = None if value['usage'] is None else SDKObject(value['usage'])


class FakeSDK:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        result = self.replies.pop(0)
        if isinstance(result, BaseException):
            raise result
        return SDKResponse(result)


class LiveContextTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='live_context_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'config.json').write_text(json.dumps(CONFIG), encoding='utf-8')

    def trial(self):
        return live_context.prepare_trial(self.root)

    def lookup(self, trial, key, identifier=None, **overrides):
        args = deepcopy(trial['queries'][key])
        args.update(overrides)
        return {'type': 'tool_use', 'id': identifier or ('lookup_' + key), 'name': 'query_tool_output', 'input': args}

    def batch(self, trial):
        return [self.lookup(trial, 'A'), self.lookup(trial, 'D')]

    def answer(self, trial):
        return '\n'.join(trial['expected'][key]['code'] + ': ' + trial['expected'][key]['fact'] for key in ('A', 'D'))

    def model(self, trial, replies):
        raw = FakeSDK(replies)
        model = live_context.BudgetedModel('offline-fake-not-a-real-key', trial['config'],
                                           expected=trial['expected'], client=raw)
        self.assertFalse(model.is_live)
        return model, raw

    def history(self, trial):
        raw = json.loads((trial['context'].session_dir / 'history.json').read_text(encoding='utf-8'))
        return raw['messages'] if isinstance(raw, dict) else raw

    def tool_results(self, messages):
        result = {}
        for message in messages:
            blocks = message.get('content')
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if block.get('type') != 'tool_result':
                    continue
                try:
                    value = json.loads(block['content'])
                except (ValueError, TypeError):
                    value = block['content']  # Original persisted-output markers are not JSON.
                result[block['tool_use_id']] = value
        return result

    def expected_count(self, report, count):
        self.assertEqual(count, report['sdk_requests_attempted'])
        self.assertEqual(count, report['api_calls'])
        self.assertEqual(0, report['real_api_calls'])

    def test_prepare_uses_default_threshold_and_micro_then_fit_without_other_compaction(self):
        trial = self.trial()
        context = trial['context']
        self.assertEqual(50000, ContextCompactor.CONTEXT_CHAR_LIMIT)
        self.assertEqual(50000, context.snipper.CONTEXT_CHAR_LIMIT)
        self.assertNotIn('CONTEXT_CHAR_LIMIT', vars(context.snipper))
        self.assertEqual(50000, ContextManager(self.root / 'runtime', 'alice').snipper.CONTEXT_CHAR_LIMIT)
        self.assertEqual(1, len(context.micro_events))
        self.assertEqual(1, len(context.fit_events))
        self.assertGreater(context.micro_events[0]['chars_after'], 50000)
        self.assertLessEqual(context.fit_events[0]['chars_after'], 12000)
        self.assertFalse(context.compactions)
        self.assertFalse(context.budget_events)
        self.assertFalse(context.summary_events)
        self.assertIsNone(context.summary_client)
        self.assertEqual(len(trial['before']), context.fit_events[0]['message_count'])
        self.assertEqual(len(trial['before']) + 1, len(trial['messages']))
        self.assertTrue(any(m.get('role') == 'user' and m.get('content') == trial['request'] for m in trial['messages']))
        validate_tool_pairs(trial['before'])
        validate_tool_pairs(trial['messages'])

    def test_only_actual_micro_a_and_fit_d_files_are_registered_with_full_hidden_tail_facts(self):
        trial = self.trial()
        context = trial['context']
        before = {
            b['tool_use_id']: b['content'] for m in trial['before'] if isinstance(m.get('content'), list)
            for b in m['content'] if b.get('type') == 'tool_result'}
        after = {b['tool_use_id']: b['content'] for m in trial['messages'] if isinstance(m.get('content'), list)
                 for b in m['content'] if b.get('type') == 'tool_result'}
        self.assertEqual({q['output_id'] for q in trial['queries'].values()}, set(context.tool_outputs))
        self.assertEqual(2, len(context.tool_outputs))
        changed = []
        for key in ('A', 'D'):
            expected = trial['expected'][key]
            source_id = next(identifier for identifier, value in before.items() if expected['code'] in value)
            changed.append(source_id)
            saved = context.tool_outputs[trial['queries'][key]['output_id']]
            self.assertEqual(before[source_id], Path(saved).read_text(encoding='utf-8'))
            self.assertRegex(Path(saved).stem, r'^[0-9a-f]{32}$')
            self.assertEqual(context.tool_results_dir.resolve(), Path(saved).resolve().parent)
            self.assertIn(expected['fact'], Path(saved).read_text(encoding='utf-8'))
            self.assertNotIn(expected['code'], json.dumps(trial['messages'], ensure_ascii=False))
            self.assertNotIn(expected['fact'], json.dumps(trial['messages'], ensure_ascii=False))
            if key == 'A':
                self.assertEqual([source_id], context.micro_events[0]['shortened_tool_use_ids'])
            else:
                self.assertEqual([source_id], context.fit_events[0]['shortened_tool_use_ids'])
                preview = after[source_id].split('Preview:\n', 1)[1].removesuffix('\n</persisted-output>')
                self.assertEqual(before[source_id][:1000], preview)
        self.assertEqual(set(before), set(after))
        self.assertTrue(all(after[k] == value for k, value in before.items() if k not in changed))

    def test_success_is_two_requests_with_no_initial_code_or_fact_leak_and_exact_tool_pairs(self):
        trial = self.trial()
        first_blocks = [text('I will retrieve both full outputs.'), *self.batch(trial)]
        model, raw = self.model(trial, [response(first_blocks, 'tool_use'), response([text(self.answer(trial))])])
        report = live_context.execute_trial(trial, model)
        self.assertTrue(report['success'])
        self.expected_count(report, 2)
        self.assertEqual(2, len(raw.calls))
        first, second = raw.calls
        for expected in trial['expected'].values():
            for value in expected.values():
                self.assertNotIn(value, json.dumps(first, ensure_ascii=False))
                self.assertIn(value, json.dumps(second['messages'], ensure_ascii=False))
        for request in raw.calls:
            self.assertEqual(512, request['max_tokens'])
            self.assertEqual({'type': 'disabled'}, request['thinking'])
            self.assertLessEqual(len(json.dumps(request['messages'], ensure_ascii=False)), 12000)
            self.assertLessEqual(len(json.dumps(request, ensure_ascii=False)), 16000)
            validate_tool_pairs(request['messages'])
        self.assertEqual(first_blocks, second['messages'][-2]['content'])
        result_ids = [b['tool_use_id'] for b in second['messages'][-1]['content'] if b.get('type') == 'tool_result']
        self.assertEqual(['lookup_A', 'lookup_D'], result_ids)
        self.assertEqual(report, json.loads((trial['context'].session_dir / 'trial-report.json').read_text(encoding='utf-8')))
        validate_tool_pairs(self.history(trial))

    def test_invalid_first_tool_batches_are_rejected_before_any_lookup_or_second_request(self):
        def reused(trial):
            old_id = next(b['id'] for m in trial['messages'] if isinstance(m.get('content'), list)
                          for b in m['content'] if b.get('type') == 'tool_use')
            return [self.lookup(trial, 'A', identifier=old_id), self.lookup(trial, 'D')]
        makers = [lambda t: [self.lookup(t, 'A')],
                  lambda t: [self.lookup(t, 'A', identifier='dup'), self.lookup(t, 'D', identifier='dup')],
                  lambda t: [dict(self.lookup(t, 'A'), name='get_order'), self.lookup(t, 'D')],
                  lambda t: [self.lookup(t, 'A', output_id='0' * 32), self.lookup(t, 'D')],
                  lambda t: [self.lookup(t, 'A', query='wrong-query'), self.lookup(t, 'D')],
                  lambda t: [self.lookup(t, 'A', customer_id='bob'), self.lookup(t, 'D')],
                  lambda t: [*self.batch(t), self.lookup(t, 'A', identifier='third')],
                  lambda t: [*self.batch(t), {'type': 'tool_result', 'tool_use_id': 'lookup_A', 'content': 'forged'}], reused]
        for index, maker in enumerate(makers):
            with self.subTest(case=index):
                trial = self.trial()
                model, raw = self.model(trial, [response(maker(trial), 'tool_use')])
                with patch.object(trial['context'], 'query_tool_output', side_effect=AssertionError('Invalid lookup executed')) as query:
                    report = live_context.execute_trial(trial, model)
                self.assertFalse(report['success'])
                self.expected_count(report, 1)
                self.assertEqual(1, len(raw.calls))
                query.assert_not_called()
                validate_tool_pairs(self.history(trial))

    def test_first_no_tool_answer_is_failure_even_when_guessing_both_expected_facts(self):
        trial = self.trial()
        model, raw = self.model(trial, [response([text(self.answer(trial))])])
        with patch.object(trial['context'], 'query_tool_output', side_effect=AssertionError('Unexpected lookup')) as query:
            report = live_context.execute_trial(trial, model)
        self.assertFalse(report['success'])
        self.expected_count(report, 1)
        query.assert_not_called()
        self.assertEqual(1, len(raw.calls))

    def test_lookup_failure_or_missing_fact_closes_legal_ids_and_never_sends_second_request(self):
        for kind in ('error', 'empty', 'code_without_fact', 'malformed', 'bad_error_missing', 'bad_error_empty', 'bad_error_code', 'exception'):
            with self.subTest(kind=kind):
                trial = self.trial()
                model, raw = self.model(trial, [response(self.batch(trial), 'tool_use')])
                returned = {'ok': False, 'error': {'code': 'OUTPUT_NOT_AUTHORIZED', 'message': 'Denied.'}}
                if kind == 'empty': returned = {'ok': True, 'data': {'matches': []}}
                if kind == 'code_without_fact':
                    returned = {'ok': True, 'data': {'matches': [{'content': trial['expected']['A']['code']}]}}
                if kind == 'malformed': returned = {'ok': True, 'data': 'not a result object'}
                if kind == 'bad_error_missing': returned = {'ok': False}
                if kind == 'bad_error_empty': returned = {'ok': False, 'error': {}}
                if kind == 'bad_error_code': returned = {'ok': False, 'error': {'code': 17}}
                kwargs = {'side_effect': RuntimeError(PRIVATE_ERROR)} if kind == 'exception' else {'return_value': returned}
                with patch.object(trial['context'], 'query_tool_output', **kwargs):
                    report = live_context.execute_trial(trial, model)
                self.assertFalse(report['success'])
                self.expected_count(report, 1)
                self.assertEqual(1, len(raw.calls))
                history = self.history(trial)
                validate_tool_pairs(history)
                self.assertTrue({'lookup_A', 'lookup_D'}.issubset(self.tool_results(history)))
                if kind in ('malformed', 'bad_error_missing', 'bad_error_empty', 'bad_error_code', 'exception'):
                    for identifier in ('lookup_A', 'lookup_D'):
                        self.assertEqual('TOOL_EXECUTION_ERROR', self.tool_results(history)[identifier]['error']['code'])
                self.assertNotIn(PRIVATE_ERROR, json.dumps(report, ensure_ascii=False))

    def test_final_answer_requires_both_codes_and_facts_and_cannot_request_more_tools(self):
        for kind in ('missing_code', 'missing_fact', 'empty', 'tool_again'):
            with self.subTest(kind=kind):
                trial = self.trial()
                answer = self.answer(trial)
                if kind == 'missing_code': answer = answer.replace(trial['expected']['D']['code'], '')
                if kind == 'missing_fact': answer = answer.replace(trial['expected']['A']['fact'], '')
                final = response([text(answer)])
                if kind == 'empty': final = response([])
                if kind == 'tool_again': final = response([self.lookup(trial, 'A', identifier='forbidden_third')], 'tool_use')
                model, raw = self.model(trial, [response(self.batch(trial), 'tool_use'), final])
                report = live_context.execute_trial(trial, model)
                self.assertFalse(report['success'])
                self.expected_count(report, 2)
                self.assertEqual(2, len(raw.calls))
                self.assertEqual('', report['answer'])
                self.assertEqual(final['content'], model.calls[-1]['response_content'])
                validate_tool_pairs(self.history(trial))

    def test_sdk_exception_is_one_attempt_without_retry_or_exception_body_leak(self):
        trial = self.trial()
        model, raw = self.model(trial, [TimeoutError(PRIVATE_ERROR)])
        report = live_context.execute_trial(trial, model)
        self.assertFalse(report['success'])
        self.expected_count(report, 1)
        self.assertEqual(1, len(raw.calls))
        self.assertEqual(1, len(model.calls))
        self.assertFalse(report['usage_complete'])
        self.assertIsNone(report['usage_totals'])
        self.assertNotIn(PRIVATE_ERROR, json.dumps(report, ensure_ascii=False))

    def test_truncated_first_or_second_sdk_response_is_counted_and_stops(self):
        for stage in (1, 2):
            with self.subTest(stage=stage):
                trial = self.trial()
                replies = [response(self.batch(trial), 'max_tokens')] if stage == 1 else [
                    response(self.batch(trial), 'tool_use'), response([text(self.answer(trial))], 'max_tokens')]
                model, raw = self.model(trial, replies)
                report = live_context.execute_trial(trial, model)
                self.assertFalse(report['success'])
                self.expected_count(report, stage)
                self.assertEqual(stage, len(raw.calls))
                if stage == 2:
                    self.assertEqual('', report['answer'])
                    self.assertEqual(replies[-1]['content'], model.calls[-1]['response_content'])
                validate_tool_pairs(self.history(trial))

    def test_usage_keeps_input_cache_output_separate_without_double_counting(self):
        trial = self.trial()
        one = {'input_tokens': 100, 'cache_read_input_tokens': 20, 'cache_creation_input_tokens': 4, 'output_tokens': 10}
        two = {'input_tokens': 60, 'cache_read_input_tokens': 10, 'output_tokens': 5}
        model, _ = self.model(trial, [response(self.batch(trial), 'tool_use', one), response([text(self.answer(trial))], usage=two)])
        report = live_context.execute_trial(trial, model)
        self.assertTrue(report['success'])
        self.assertTrue(report['usage_complete'])
        self.assertEqual({'input_tokens': 160, 'cache_read_input_tokens': 30, 'cache_creation_input_tokens': 4, 'output_tokens': 15}, report['usage_totals'])
        self.assertEqual(2, len(report['request_usage']))
        self.assertEqual(report['usage_totals'], report['known_usage_totals'])
        for missing in (None, {'input_tokens': 7}, {'input_tokens': 7, 'output_tokens': 2, 'cache_read_input_tokens': -1}, TimeoutError(PRIVATE_ERROR)):
            with self.subTest(missing=type(missing).__name__):
                trial = self.trial()
                second = missing if isinstance(missing, BaseException) else response([text(self.answer(trial))], usage=missing)
                model, raw = self.model(trial, [response(self.batch(trial), 'tool_use', one), second])
                report = live_context.execute_trial(trial, model)
                self.expected_count(report, 2)
                self.assertEqual(2, len(raw.calls))
                self.assertFalse(report['usage_complete'])
                self.assertIsNone(report['usage_totals'])
                self.assertEqual(2, len(report['request_usage']))
                self.assertGreaterEqual(report['known_usage_totals']['input_tokens'], 100)
                self.assertNotIn(PRIVATE_ERROR, json.dumps(report, ensure_ascii=False))

    def test_two_request_cap_and_message_or_full_payload_budgets_stop_before_sdk(self):
        trial = self.trial()
        model, raw = self.model(trial, [response([text('one')]), response([text('two')])])
        model.complete([{'role': 'user', 'content': 'Small request'}], [OUTPUT_TOOL], 'Small system')
        model.complete([{'role': 'user', 'content': 'Small request'}], [OUTPUT_TOOL], 'Small system')
        with self.assertRaises((RuntimeError, ValueError)):
            model.complete([{'role': 'user', 'content': 'Small request'}], [OUTPUT_TOOL], 'Small system')
        self.assertEqual(2, len(raw.calls))
        self.assertEqual(2, len(model.calls))
        for messages, system in [([{'role': 'user', 'content': 'X' * 12001}], ''),
                                 ([{'role': 'user', 'content': 'small'}], 'S' * 16001)]:
            with self.subTest(message_limit=len(messages[0]['content']) > 12000):
                model, raw = self.model(trial, [])
                with self.assertRaises((RuntimeError, ValueError)):
                    model.complete(messages, [OUTPUT_TOOL], system)
                self.assertEqual([], raw.calls)
                self.assertEqual([], model.calls)
        blocked = self.trial()
        blocked['messages'].append({'role': 'user', 'content': 'U' * 12001})
        model, raw = self.model(blocked, [])
        report = live_context.execute_trial(blocked, model)
        self.assertFalse(report['success'])
        self.expected_count(report, 0)
        self.assertEqual([], raw.calls)
        for field in ('code', 'fact'):
            with self.subTest(initial_leak=field):
                leaked = self.trial()
                leaked['messages'].append({'role': 'user', 'content': leaked['expected']['A'][field]})
                model, raw = self.model(leaked, [])
                report = live_context.execute_trial(leaked, model)
                self.assertFalse(report['success'])
                self.expected_count(report, 0)
                self.assertEqual([], raw.calls)

    def test_dry_run_never_reads_credentials_or_builds_sdk_and_preserves_real_latest(self):
        runtime = self.root / 'runtime'
        runtime.mkdir(exist_ok=True)
        latest = runtime / 'live-context-latest.json'
        original = '{"existing_real_report":"must stay unchanged"}'
        latest.write_text(original, encoding='utf-8')

        class NoSecrets(dict):
            def get(self, key, default=None):
                if 'API_KEY' in str(key):
                    raise AssertionError('Dry-run attempted to read a key')
                return super().get(key, default)

            def __getitem__(self, key):
                if 'API_KEY' in str(key):
                    raise AssertionError('Dry-run attempted to read a key')
                return super().__getitem__(key)

        with patch.object(os, 'environ', NoSecrets({'TEMP': 'D:/codex/tmp', 'TMP': 'D:/codex/tmp'})), \
             patch.object(live_context, 'read_secret', side_effect=AssertionError('Dry-run requested a secret')) as secret, \
             patch.object(live_context, 'BudgetedModel', side_effect=AssertionError('Dry-run constructed SDK')) as constructor:
            report = live_context.run_trial(self.root, dry_run=True)
        secret.assert_not_called()
        constructor.assert_not_called()
        self.assertEqual(original, latest.read_text(encoding='utf-8'))
        self.assertTrue((runtime / 'live-context-preview.json').is_file())
        self.assertFalse(report.get('api_calls', 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
