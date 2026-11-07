"""Offline acceptance of bounded, identity-scoped read-only subagents."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.shop import Shop

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ERROR = 'PRIVATE_CHILD_EXCEPTION_6472'
ALLOWED = {'get_product', 'get_policy', 'get_order', 'get_tracking', 'list_my_orders', 'query_archive', 'query_tool_output', 'load_skill'}


def text(value):
    return {'type': 'text', 'text': value}


def call(identifier, name='task', **arguments):
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
        if isinstance(response, dict):
            return deepcopy(response)
        content = deepcopy(response)
        return {'content': content, 'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in content) else 'end_turn',
                'usage': {'input_tokens': 100, 'output_tokens': 20}}


class SubagentTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='subagent_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)

    def build(self, parent_replies, child_replies=None, customer='alice', enabled=True):
        context = ContextManager(self.runtime, customer or 'guest')
        parent = MainModel(parent_replies)
        child = None if child_replies is None else MainModel(child_replies)
        harness = Harness(Shop(customer, ROOT / 'data/shop.json'), context, parent,
                          subagent_model=child, enable_subagents=enabled)
        return harness, context, parent, child

    def results(self, messages):
        return {b['tool_use_id']: json.loads(b['content']) for m in messages if isinstance(m.get('content'), list)
                for b in m['content'] if b.get('type') == 'tool_result'}

    def load_child(self, harness, index=-1):
        run = harness.subagents.runs[index]
        directory = Path(run.get('child_session_dir') or run['session_dir'])
        return directory, json.loads((directory / 'history.json').read_text(encoding='utf-8'))

    def audit(self, directory):
        return [json.loads(x) for x in (directory / 'audit.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_task_schema_and_invalid_arguments_do_not_spend_the_valid_delegation(self):
        invalid = [{}, {'prompt': ''}, {'prompt': ' '}, {'prompt': 17}, {'prompt': 'x' * 2001},
                   {'prompt': 'Read order', 'customer_id': 'bob'}]
        batch = [call(f'invalid_{i}', **args) for i, args in enumerate(invalid)]
        batch.append(call('valid_after_invalid', prompt='Read public information.'))
        harness, _, parent, child = self.build([batch, [text('Finished.')]], [[text('Reference report.')]])
        schema = next(tool for tool in harness.tools if tool['name'] == 'task')['input_schema']
        self.assertEqual({'prompt'}, set(schema['properties']))
        self.assertEqual(['prompt'], schema['required'])
        self.assertFalse(schema['additionalProperties'])
        harness.turn('Run one bounded investigation.')
        actual = self.results(parent.calls[1]['messages'])
        for i in range(len(invalid)):
            self.assertEqual('INVALID_ARGUMENTS', actual[f'invalid_{i}']['error']['code'])
        self.assertTrue(actual['valid_after_invalid']['ok'])
        self.assertEqual(1, len(child.calls))
        self.assertEqual(1, len(harness.subagents.runs))
        validate_tool_pairs(harness.messages)

    def test_child_starts_with_only_prompt_and_keeps_complete_blocks_out_of_parent_history(self):
        prompt = 'Read A1001 without changing anything.'
        child_first = [{'type': 'thinking', 'thinking': 'CHILD_PRIVATE_THOUGHT_719', 'signature': 'offline-signature'},
                       text('CHILD_INTERMEDIATE_TEXT_719'), call('child_order', 'get_order', order_id='A1001')]
        parent_first = [text('PARENT_TEXT_MUST_SURVIVE'), call('delegation', prompt=prompt)]
        harness, context, parent, child = self.build([parent_first, [text('Parent answer.')]],
                                                     [child_first, [text('A1001 was read from the tool.')]])
        harness.messages = [{'role': 'user', 'content': 'PARENT_HISTORY_SECRET_719'},
                            {'role': 'assistant', 'content': 'Parent history response.'}]
        harness.todo.update([{'content': 'PARENT_PLAN_SECRET_719', 'status': 'in_progress'}])
        harness.turn('Delegate this narrow read.')
        self.assertEqual([{'role': 'user', 'content': prompt}], child.calls[0]['messages'])
        self.assertNotIn('PARENT_HISTORY_SECRET_719', json.dumps(child.calls))
        self.assertNotIn('PARENT_PLAN_SECRET_719', json.dumps(child.calls))
        result = self.results(parent.calls[1]['messages'])['delegation']
        self.assertTrue(result['ok'])
        data = result['data']
        self.assertTrue(data['reference_only'])
        self.assertEqual('A1001 was read from the tool.', data['report'])
        self.assertEqual(2, data['model_requests'])
        directory, history = self.load_child(harness)
        self.assertTrue(directory.resolve().is_relative_to((context.session_dir / 'subagents').resolve()))
        self.assertEqual(directory.name, data['child_session_id'])
        self.assertEqual(child_first, history['messages'][1]['content'])
        self.assertTrue(any(m.get('content') == parent_first for m in harness.messages))
        parent_assistant_blocks = [b for m in harness.messages if m['role'] == 'assistant' and isinstance(m['content'], list) for b in m['content']]
        self.assertNotIn(child_first[0], parent_assistant_blocks)
        self.assertNotIn(child_first[1], parent_assistant_blocks)
        self.assertEqual([{'content': 'PARENT_PLAN_SECRET_719', 'status': 'in_progress'}], harness.todo.items)
        validate_tool_pairs(history['messages'])
        validate_tool_pairs(harness.messages)

    def test_alice_bob_and_guest_child_tools_enforce_the_same_parent_principal(self):
        for customer, allowed, foreign, denied in [('alice', 'A1001', 'B2001', 'ACCESS_DENIED_OR_NOT_FOUND'),
                                                   ('bob', 'B2001', 'A1001', 'ACCESS_DENIED_OR_NOT_FOUND'),
                                                   (None, 'A1001', 'B2001', 'AUTHENTICATION_REQUIRED')]:
            with self.subTest(customer=customer):
                tools = [call('child_own', 'get_order', order_id=allowed),
                         call('child_foreign', 'get_order', order_id=foreign),
                         call('child_public', 'get_product', product_id='P100')]
                harness, _, parent, child = self.build([[call('scope_task', prompt='Check the bound customer only.')], [text('Reviewed.')]],
                                                       [tools, [text('Use only the returned records.')]], customer=customer)
                harness.turn('Investigate safely.')
                data = self.results(parent.calls[1]['messages'])['scope_task']['data']
                evidence = {entry['tool_use_id']: entry['result'] for entry in data['evidence']}
                self.assertTrue(evidence['child_public']['ok'])
                if customer is None:
                    self.assertEqual('AUTHENTICATION_REQUIRED', evidence['child_own']['error']['code'])
                else:
                    self.assertTrue(evidence['child_own']['ok'])
                    self.assertEqual(allowed, evidence['child_own']['data']['order_id'])
                self.assertEqual(denied, evidence['child_foreign']['error']['code'])
                self.assertNotIn('data', evidence['child_foreign'])
                self.assertEqual(customer, harness.shop.customer_id)
                directory, history = self.load_child(harness)
                self.assertEqual(customer, history['simulated_customer'])
                self.assertTrue(all(row['customer_id'] == customer for row in self.audit(directory)))
                self.assertEqual(2, len(child.calls))

    def test_child_hard_allowlist_rejects_task_todo_compact_and_file_tools(self):
        forbidden = [call('nested_task', 'task', prompt='Recurse'), call('child_todo', 'todo_write', todos=[]),
                     call('child_compact', 'compact'), call('child_file', 'read_file', path='private.txt')]
        harness, _, parent, child = self.build([[call('restricted', prompt='Only observe public facts.')], [text('Reviewed.')]],
                                               [forbidden, [text('Forbidden tools were rejected.')]])
        harness.todo.update([{'content': 'Parent plan must remain', 'status': 'in_progress'}])
        harness.turn('Check the child boundary.')
        self.assertEqual(ALLOWED, {tool['name'] for tool in child.calls[0]['tools']})
        _, history = self.load_child(harness)
        denied = self.results(history['messages'])
        self.assertEqual({entry['id'] for entry in forbidden}, set(denied))
        self.assertTrue(all(not result['ok'] for result in denied.values()))
        self.assertEqual([{'content': 'Parent plan must remain', 'status': 'in_progress'}], harness.todo.items)
        self.assertEqual(1, len(harness.subagents.runs))
        self.assertTrue(self.results(parent.calls[1]['messages'])['restricted']['ok'])
        validate_tool_pairs(history['messages'])

    def test_second_valid_task_in_one_turn_is_rejected_without_another_child_request(self):
        harness, _, parent, child = self.build([[call('one', prompt='First read.'), call('two', prompt='Second read.')], [text('One task only.')]],
                                               [[text('First bounded report.')]])
        harness.turn('Try two delegations.')
        actual = self.results(parent.calls[1]['messages'])
        self.assertTrue(actual['one']['ok'])
        self.assertEqual('SUBAGENT_LIMIT_REACHED', actual['two']['error']['code'])
        self.assertEqual(1, len(child.calls))
        self.assertEqual(1, len(harness.subagents.runs))
        validate_tool_pairs(harness.messages)

    def test_valid_failed_child_spends_allowance_and_next_user_turn_resets_it(self):
        replies = [[call('failed_task', prompt='A valid narrow request.'), call('same_turn_again', prompt='Try again.')], [text('Child failed safely.')],
                   [call('new_turn_task', prompt='A fresh request.')], [text('Fresh turn completed.')]]
        harness, _, parent, child = self.build(replies, [RuntimeError(PRIVATE_ERROR), [text('Recovered on the new turn.')]])
        harness.turn('First input.')
        first = self.results(parent.calls[1]['messages'])
        self.assertEqual('SUBAGENT_FAILED', first['failed_task']['error']['code'])
        self.assertEqual('SUBAGENT_LIMIT_REACHED', first['same_turn_again']['error']['code'])
        self.assertEqual(1, len(child.calls))
        self.assertNotIn(PRIVATE_ERROR, json.dumps(first))
        harness.turn('Second input.')
        self.assertTrue(self.results(parent.calls[3]['messages'])['new_turn_task']['ok'])
        self.assertEqual(2, len(child.calls))
        self.assertEqual(2, len(harness.subagents.runs))
        directory, history = self.load_child(harness, 0)
        self.assertNotIn(PRIVATE_ERROR, (directory / 'events.jsonl').read_text(encoding='utf-8'))
        self.assertNotIn(PRIVATE_ERROR, json.dumps(history))
        validate_tool_pairs(harness.messages)

    def test_child_step_limit_has_three_requests_and_no_tool_reexecution_or_fake_success(self):
        child_replies = [[call(f'bounded_{i}', 'get_product', product_id='P100')] for i in range(3)]
        harness, _, parent, child = self.build([[call('bounded_task', prompt='Read within the limit.')], [text('Task incomplete.')]], child_replies)
        with patch.object(harness.shop, 'call', wraps=harness.shop.call) as handler:
            harness.turn('Use the bounded child.')
        self.assertEqual(3, len(child.calls))
        self.assertEqual(3, handler.call_count)
        result = self.results(parent.calls[1]['messages'])['bounded_task']
        self.assertEqual('SUBAGENT_INCOMPLETE', result['error']['code'])
        directory, history = self.load_child(harness)
        self.assertEqual({'bounded_0', 'bounded_1', 'bounded_2'}, set(self.results(history['messages'])))
        self.assertEqual('incomplete', self.audit(directory)[-1]['status'])
        validate_tool_pairs(history['messages'])
        validate_tool_pairs(harness.messages)

    def test_empty_and_over_3000_character_child_reports_are_explicit_task_errors(self):
        for blocks, expected in [([], 'SUBAGENT_EMPTY_REPORT'), ([text('X' * 3001)], 'SUBAGENT_REPORT_TOO_LARGE')]:
            with self.subTest(expected=expected):
                harness, _, parent, child = self.build([[call('invalid_report', prompt='Return a short report.')], [text('No accepted report.')]], [blocks])
                harness.turn('Delegate one read.')
                result = self.results(parent.calls[1]['messages'])['invalid_report']
                self.assertFalse(result['ok'])
                self.assertEqual(expected, result['error']['code'])
                self.assertEqual('', result['data']['report'])
                self.assertTrue(result['data']['child_session_id'])
                self.assertIn('usage', result['data'])
                self.assertEqual(1, len(child.calls))
                self.assertEqual(1, len(harness.subagents.runs))
                validate_tool_pairs(harness.messages)

    def test_large_evidence_is_explicitly_shortened_while_full_child_results_stay_on_disk(self):
        tools = [call('large_a', 'get_product', product_id='P100'), call('large_b', 'get_policy', policy_id='returns')]
        full = [{'ok': True, 'data': {'source': f'synthetic.json#record.{i}', 'detail': f'LONG_FACT_{i}:' + 'Z' * 4500}, 'simulated': True}
                for i in range(2)]
        harness, _, parent, _ = self.build([[call('evidence_task', prompt='Read both records.')], [text('Reviewed short evidence.')]],
                                           [tools, [text('Two synthetic facts checked.')]])
        with patch.object(harness.shop, 'call', side_effect=full):
            harness.turn('Use bounded evidence.')
        data = self.results(parent.calls[1]['messages'])['evidence_task']['data']
        encoded = json.dumps(data['evidence'], ensure_ascii=False)
        self.assertLessEqual(len(encoded), 6000)
        self.assertTrue(data['evidence_omitted_count'] > 0 or '"details_omitted": true' in encoded)
        for entry in data['evidence']:
            omitted = entry.get('details_omitted') or entry['result'].get('details_omitted')
            if omitted:
                self.assertNotIn('Z' * 4500, json.dumps(entry))
            else:
                expected = full[0 if entry['tool_use_id'] == 'large_a' else 1]
                self.assertEqual(expected, entry['result'])
        _, history = self.load_child(harness)
        actual = self.results(history['messages'])
        self.assertEqual(full, [actual['large_a'], actual['large_b']])
        validate_tool_pairs(history['messages'])

    def test_parent_archive_and_old_output_registration_are_not_inherited_by_child(self):
        archive_id = 'transcript_' + '1' * 32
        output_id = '2' * 32
        tools = [call('foreign_archive', 'query_archive', archive_id=archive_id, query='PARENT_ONLY'),
                 call('foreign_output', 'query_tool_output', output_id=output_id, query='PARENT_ONLY')]
        harness, context, parent, child = self.build([[call('separate_scope', prompt='Try the provided IDs only.')], [text('No inherited file access.')]],
                                                     [tools, [text('These IDs were not authorized here.')]])
        context.transcript_dir.mkdir(parents=True)
        archive = context.transcript_dir / (archive_id + '.jsonl')
        archive.write_text(json.dumps({'role': 'user', 'content': 'PARENT_ONLY_ARCHIVE_SECRET'}) + '\n', encoding='utf-8')
        context.archives[archive_id] = archive
        context.tool_results_dir.mkdir(parents=True)
        output = context.tool_results_dir / (output_id + '.txt')
        output.write_text('PARENT_ONLY_OUTPUT_SECRET', encoding='utf-8')
        context.tool_outputs[output_id] = output
        harness.turn('Delegate without passing old state.')
        self.assertNotIn('PARENT_ONLY_ARCHIVE_SECRET', json.dumps(child.calls))
        self.assertNotIn('PARENT_ONLY_OUTPUT_SECRET', json.dumps(child.calls))
        data = self.results(parent.calls[1]['messages'])['separate_scope']['data']
        evidence = {entry['tool_use_id']: entry['result'] for entry in data['evidence']}
        self.assertEqual('ARCHIVE_NOT_AUTHORIZED', evidence['foreign_archive']['error']['code'])
        self.assertEqual('OUTPUT_NOT_AUTHORIZED', evidence['foreign_output']['error']['code'])
        self.assertEqual({archive_id}, set(context.archives))
        self.assertEqual({output_id}, set(context.tool_outputs))

    def test_parent_child_audits_receipts_and_usage_are_separate_and_missing_usage_stays_unknown(self):
        missing = {'content': [text('Short child report.')], 'stop_reason': 'end_turn', 'usage': None}
        harness, context, parent, child = self.build([[call('audited_task', prompt='Check P100.')], [text('Parent explained.')]],
                                                     [[call('audited_product', 'get_product', product_id='P100')], missing])
        harness.turn('Run an audited read.')
        data = self.results(parent.calls[1]['messages'])['audited_task']['data']
        self.assertEqual(2, data['model_requests'])
        self.assertFalse(data['usage']['complete'])
        self.assertEqual(2, len(data['usage']['records']))
        directory, _ = self.load_child(harness)
        parent_audit, child_audit = self.audit(context.session_dir), self.audit(directory)
        self.assertEqual(['audited_task', 'audited_task'], [r['tool_use_id'] for r in parent_audit if 'tool_use_id' in r])
        self.assertEqual(['audited_product', 'audited_product'], [r['tool_use_id'] for r in child_audit if 'tool_use_id' in r])
        self.assertTrue(all(r['session_id'] == context.session_id for r in parent_audit))
        self.assertTrue(all(r['session_id'] == data['child_session_id'] for r in child_audit))
        snapshot = json.loads((context.session_dir / 'history.json').read_text(encoding='utf-8'))
        self.assertEqual(harness.subagents.runs, snapshot['subagent_runs'])
        self.assertEqual(1, len(snapshot['subagent_runs']))
        self.assertEqual(2, snapshot['subagent_runs'][0]['model_requests'])
        self.assertEqual(2, len(child.calls))

    def test_disabled_runner_rejects_unadvertised_task_and_default_runner_can_share_stateless_model(self):
        disabled, _, parent, child = self.build([[call('disabled_task', prompt='No delegation.')], [text('No child.')]],
                                               [[text('Must never run.')]], enabled=False)
        self.assertNotIn('task', {t['name'] for t in disabled.tools})
        disabled.turn('Try an unexpected task call.')
        self.assertFalse(self.results(parent.calls[1]['messages'])['disabled_task']['ok'])
        self.assertEqual([], child.calls)
        # One stateless model instance is used for parent and child requests;
        # their tools and histories still alternate between independent scopes.
        harness, _, shared, _ = self.build([[call('shared_task', prompt='Read public P100.')],
                                           [call('shared_product', 'get_product', product_id='P100')],
                                           [text('Child report.')], [text('Parent answer.')]])
        self.assertEqual('Parent answer.', harness.turn('Delegate using the default model.'))
        self.assertEqual(4, len(shared.calls))
        self.assertIn('task', {t['name'] for t in shared.calls[0]['tools']})
        self.assertEqual(ALLOWED, {t['name'] for t in shared.calls[1]['tools']})
        self.assertEqual(ALLOWED, {t['name'] for t in shared.calls[2]['tools']})
        self.assertIn('task', {t['name'] for t in shared.calls[3]['tools']})
        self.assertTrue(self.results(shared.calls[3]['messages'])['shared_task']['ok'])
        validate_tool_pairs(harness.messages)


    def test_trace_failure_at_start_or_end_closes_parent_batch_without_replaying_child(self):
        for stage in ('subagent_start', 'subagent_end'):
            with self.subTest(stage=stage):
                child_tools = [call('trace_child_order', 'get_order', order_id='A1001')]
                parent_batch = [call('trace_task', prompt='Read A1001 once.'),
                                call('trace_later_business', 'get_product', product_id='P100')]
                harness, context, parent, child = self.build([parent_batch],
                    [child_tools, [text('Trace child obtained A1001.')]])
                original_event = harness.event
                failed_writes = []

                def fail_trace(kind, **fields):
                    if kind == stage:
                        failed_writes.append(kind)
                        raise OSError(PRIVATE_ERROR)
                    return original_event(kind, **fields)

                with patch.object(harness, 'event', side_effect=fail_trace), \
                     patch.object(harness.shop, 'call', wraps=harness.shop.call) as handler:
                    answer = harness.turn('Delegate once, then read the public product.')
                self.assertIsInstance(answer, str)
                self.assertEqual([stage], failed_writes)
                self.assertEqual(1, len(parent.calls))
                actual = self.results(harness.messages)
                self.assertEqual({'trace_task', 'trace_later_business'}, set(actual))
                self.assertEqual('HOOK_FAILED', actual['trace_later_business']['error']['code'])
                self.assertTrue(harness._hooks_failed)
                if stage == 'subagent_start':
                    self.assertEqual('SUBAGENT_FAILED', actual['trace_task']['error']['code'])
                    self.assertEqual([], child.calls)
                    handler.assert_not_called()
                else:
                    self.assertTrue(actual['trace_task']['ok'])
                    self.assertEqual('Trace child obtained A1001.', actual['trace_task']['data']['report'])
                    evidence = actual['trace_task']['data']['evidence']
                    self.assertEqual('trace_child_order', evidence[0]['tool_use_id'])
                    self.assertTrue(evidence[0]['result']['ok'])
                    self.assertEqual('A1001', evidence[0]['result']['data']['order_id'])
                    self.assertEqual(2, len(child.calls))
                    handler.assert_called_once_with('get_order', {'order_id': 'A1001'})
                    _, child_history = self.load_child(harness)
                    validate_tool_pairs(child_history['messages'])
                # Direct dispatch does not begin another user turn. A valid
                # delegation remains spent even when trace persistence failed.
                rejected = harness.call_tool('task', {'prompt': 'No second delegation in this turn.'})
                self.assertEqual('SUBAGENT_LIMIT_REACHED', rejected['error']['code'])
                self.assertEqual(0 if stage == 'subagent_start' else 2, len(child.calls))
                self.assertEqual('incomplete', self.audit(context.session_dir)[-1]['status'])
                event_text = (context.session_dir / 'events.jsonl').read_text(encoding='utf-8')
                self.assertNotIn(PRIVATE_ERROR, event_text)
                self.assertNotIn(PRIVATE_ERROR, json.dumps(actual))
                validate_tool_pairs(harness.messages)

    def test_usage_optional_cache_fields_are_validated_without_fabricating_missing_counters(self):
        cases = [({'input_tokens': 31, 'output_tokens': 7}, True)]
        for field in ('cache_read_input_tokens', 'cache_creation_input_tokens'):
            for invalid in (-1, True, None):
                cases.append(({'input_tokens': 31, 'output_tokens': 7, field: invalid}, False))
        for raw_usage, expected_complete in cases:
            with self.subTest(usage=raw_usage):
                child_reply = {'content': [text('Short usage report.')], 'stop_reason': 'end_turn',
                               'usage': deepcopy(raw_usage)}
                harness, _, parent, child = self.build([[call('usage_task', prompt='Return a short reference.')],
                                                        [text('Parent response.')]], [child_reply])
                harness.turn('Delegate once.')
                result = self.results(parent.calls[1]['messages'])['usage_task']
                self.assertTrue(result['ok'])
                usage = result['data']['usage']
                self.assertEqual(expected_complete, usage['complete'])
                self.assertEqual(1, len(usage['records']))
                self.assertEqual('received', usage['records'][0]['status'])
                self.assertEqual(raw_usage, usage['records'][0]['usage'])
                self.assertEqual(set(raw_usage), set(usage['records'][0]['usage']))
                self.assertEqual(1, result['data']['model_requests'])
                self.assertEqual(1, len(child.calls))


if __name__ == '__main__':
    unittest.main(verbosity=2)
