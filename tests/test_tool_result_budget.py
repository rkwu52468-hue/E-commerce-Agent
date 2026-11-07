"""Offline acceptance of the original front tool-result batch budget."""
from copy import deepcopy
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from customer_service.context import ContextManager, ToolResultBudgetError, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.shop import Shop

ROOT = Path(__file__).resolve().parents[1]
REQUEST = '只读查询 A1001 物流；不要退款，不要创建售后单。'
PRIVATE_ERROR = 'PRIVATE_BUDGET_DISK_ERROR_7148'


def body(length, label='simulated'):
    unit = label + ': 模拟物流检查点。\n'
    return (unit * (length // len(unit) + 1))[:length]


def result_blocks(messages):
    return [block for message in messages if isinstance(message.get('content'), list)
            for block in message['content'] if block.get('type') == 'tool_result']


def shape(messages):
    result = deepcopy(messages)
    for block in result_blocks(result):
        block['content'] = '<result body>'
    return result


def batch(bodies, identifiers=None):
    identifiers = identifiers or [f'budget_{i}' for i in range(len(bodies))]
    calls = [{'type': 'tool_use', 'id': identifier, 'name': 'get_tracking', 'input': {'order_id': 'A1001'}}
             for identifier in identifiers]
    results = [{'type': 'tool_result', 'tool_use_id': identifier, 'content': value, 'is_error': False}
               for identifier, value in zip(identifiers, bodies)]
    return [{'role': 'user', 'content': REQUEST}, {'role': 'assistant', 'content': calls},
            {'role': 'user', 'content': results}]


def preview(marker):
    return marker.split('Preview:\n', 1)[1].removesuffix('\n</persisted-output>')


class MainModel:
    mode = 'scripted-offline'

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append(deepcopy(messages))
        content = deepcopy(self.replies.pop(0))
        return {'content': content, 'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in content) else 'end_turn',
                'usage': {'input_tokens': 100, 'output_tokens': 20}}


class ToolResultBudgetTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='tool_result_budget_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)

    def context(self, customer='alice'):
        return ContextManager(self.runtime, customer)

    def test_single_output_helper_has_30000_boundary_and_exact_2000_character_preview(self):
        context = self.context()
        at_limit = body(30000)
        self.assertEqual(at_limit, context.snipper.persist_large_output('exact_boundary', at_limit))
        self.assertFalse(context.tool_results_dir.exists())
        output = body(30001 - len('TAIL_FACT_30001')) + 'TAIL_FACT_30001'
        self.assertEqual(30001, len(output))
        marker = context.snipper.persist_large_output('large_single', output)
        self.assertTrue(marker.startswith('<persisted-output>\n'))
        self.assertEqual(output[:2000], preview(marker))
        saved = Path(context.snipper.persisted_output_path(marker))
        self.assertEqual(output, saved.read_text(encoding='utf-8'))
        self.assertEqual(context.tool_results_dir.resolve(), saved.resolve().parent)
        self.assertRegex(saved.stem, r'^[0-9a-f]{32}$')
        self.assertTrue(output.endswith('TAIL_FACT_30001'))
        # Direct vendor helpers do not grant business access to a saved file.
        self.assertFalse(context.tool_outputs)

    def test_single_30001_result_below_batch_budget_is_not_changed_by_prepare(self):
        context = self.context()
        messages = batch([body(30001)])
        original = deepcopy(messages)
        prepared = context.prepare(messages, REQUEST)
        self.assertIs(messages, prepared)
        self.assertEqual(original, prepared)
        self.assertFalse(context.budget_events)
        self.assertFalse(context.tool_outputs)
        self.assertFalse(context.tool_results_dir.exists())
        self.assertFalse(context.micro_events)
        self.assertFalse(context.fit_events)
        self.assertEqual(50000, context.snipper.CONTEXT_CHAR_LIMIT)
        self.assertEqual(200000, context.snipper.TOOL_RESULT_BATCH_CHAR_LIMIT)
        self.assertEqual(30000, context.snipper.LARGE_RESULT_CHAR_LIMIT)

    def test_exact_200000_body_characters_are_unchanged_even_with_larger_json_envelope(self):
        context = self.context()
        messages = batch([body(100000, 'A'), body(100000, 'B')])
        original = deepcopy(messages)
        self.assertGreater(context.snipper.estimate_chars(messages), 200000)
        self.assertIs(messages, context.tool_result_budget(messages))
        self.assertEqual(original, messages)
        self.assertFalse(context.budget_events)
        self.assertFalse(context.tool_outputs)
        self.assertFalse(context.tool_results_dir.exists())
        over = batch([body(100001, 'A'), body(100000, 'B')])
        reduced = context.tool_result_budget(over)
        self.assertNotEqual(over, reduced)
        self.assertEqual(200001, context.budget_events[-1]['batch_chars_before'])
        self.assertEqual(['budget_0'], context.budget_events[-1]['shortened_tool_use_ids'])

    def test_over_budget_latest_batch_shortens_largest_first_and_stops_as_soon_as_total_fits(self):
        context = self.context()
        # Raise only this test instance's context threshold to observe the front
        # 200000 budget without later 50000 micro/fit stages masking its choice.
        context.snipper.CONTEXT_CHAR_LIMIT = 400000
        bodies = [body(60000, 'small'), body(95000, 'largest'), body(52000, 'third')]
        messages = batch(bodies)
        original = deepcopy(messages)
        prepared = context.prepare(messages, REQUEST)
        self.assertEqual(original, messages)
        self.assertEqual(shape(original), shape(prepared))
        blocks = result_blocks(prepared)
        self.assertEqual(bodies[0], blocks[0]['content'])
        self.assertTrue(blocks[1]['content'].startswith('<persisted-output>\n'))
        self.assertEqual(bodies[2], blocks[2]['content'])
        self.assertEqual(bodies[1][:2000], preview(blocks[1]['content']))
        event = context.budget_events[-1]
        self.assertEqual('applied', event['status'])
        self.assertEqual(207000, event['batch_chars_before'])
        self.assertEqual(sum(len(b['content']) for b in blocks), event['batch_chars_after'])
        self.assertLessEqual(event['batch_chars_after'], 200000)
        self.assertEqual(context.snipper.estimate_chars(original), event['context_chars_before'])
        self.assertEqual(context.snipper.estimate_chars(prepared), event['context_chars_after'])
        self.assertEqual(['budget_1'], event['shortened_tool_use_ids'])
        self.assertEqual((200000, 30000, 2000, 3), tuple(event[k] for k in
            ('batch_limit', 'large_result_limit', 'preview_chars', 'message_count')))
        self.assertEqual(1, len(context.tool_outputs))
        self.assertFalse(context.micro_events)
        self.assertFalse(context.fit_events)
        validate_tool_pairs(prepared)

    def test_batch_total_counts_result_blocks_and_cannot_shorten_candidates_at_30000(self):
        context = self.context()
        messages = batch([body(30000, str(i)) for i in range(7)])
        messages[-1]['content'].append({'type': 'text', 'text': 'Extra text is not a result block.' * 100})
        original = deepcopy(messages)
        result = context.tool_result_budget(messages)
        self.assertIs(messages, result)
        self.assertEqual(original, result)
        event = context.budget_events[-1]
        self.assertEqual('unchanged', event['status'])
        self.assertEqual(210000, event['batch_chars_before'])
        self.assertEqual(210000, event['batch_chars_after'])
        self.assertEqual([], event['shortened_tool_use_ids'])
        self.assertFalse(context.tool_outputs)
        self.assertFalse(context.tool_results_dir.exists())
        validate_tool_pairs(result)

    def test_only_latest_user_result_batch_is_considered_not_old_large_batches(self):
        context = self.context()
        old = batch([body(220000, 'old')], ['old_batch_result'])
        current = batch([body(31000, 'latest')], ['current_batch_result'])
        messages = [*old, {'role': 'assistant', 'content': 'Old batch was already read.'},
                    {'role': 'user', 'content': 'A new read-only question.'}, *current[1:]]
        original = deepcopy(messages)
        self.assertIs(messages, context.tool_result_budget(messages))
        self.assertEqual(original, messages)
        for non_batch in ([], [{'role': 'user', 'content': 'ordinary text'}],
                          [*messages, {'role': 'assistant', 'content': 'already consumed'}]):
            self.assertIs(non_batch, context.tool_result_budget(non_batch))
        self.assertFalse(context.budget_events)
        self.assertFalse(context.tool_results_dir.exists())
        validate_tool_pairs(messages)

    def test_existing_persisted_marker_is_reused_without_nested_preview_or_extra_file(self):
        context = self.context()
        context.snipper.CONTEXT_CHAR_LIMIT = 400000
        bodies = [body(120001, 'large'), body(100000, 'untouched')]
        first = context.prepare(batch(bodies), REQUEST)
        saved_paths = dict(context.tool_outputs)
        first_marker = result_blocks(first)[0]['content']
        saved = next(iter(saved_paths.values()))
        self.assertEqual(bodies[0], saved.read_text(encoding='utf-8'))
        second = context.prepare(first, REQUEST)
        self.assertIs(first, second)
        self.assertEqual(first_marker, result_blocks(second)[0]['content'])
        self.assertEqual(saved_paths, context.tool_outputs)
        self.assertEqual(1, len(context.budget_events))
        self.assertEqual([saved], list(context.tool_results_dir.glob('*.txt')))
        self.assertEqual(1, first_marker.count('<persisted-output>'))
        self.assertEqual(bodies[0][:2000], preview(first_marker))
        validate_tool_pairs(second)

    def test_saved_output_ids_are_unique_and_only_registered_in_current_customer_session(self):
        context = self.context()
        context.snipper.CONTEXT_CHAR_LIMIT = 600000
        bodies = [body(160000 - len('TAIL_A_FACT'), 'A') + 'TAIL_A_FACT',
                  body(150000 - len('TAIL_B_FACT'), 'B') + 'TAIL_B_FACT', body(140000, 'C')]
        identifiers = ['collision/a', 'collision?a', 'third']
        original = batch(bodies, identifiers)
        result = context.prepare(original, REQUEST)
        self.assertEqual(shape(original), shape(result))
        self.assertEqual(2, len(context.tool_outputs))
        self.assertTrue(all(re.fullmatch(r'[0-9a-f]{32}', key) for key in context.tool_outputs))
        paths = [Path(context.snipper.persisted_output_path(b['content'])) for b in result_blocks(result)[:2]]
        self.assertNotEqual(paths[0], paths[1])
        for path, value, keyword in zip(paths, bodies, ('TAIL_A_FACT', 'TAIL_B_FACT')):
            self.assertEqual(value, path.read_text(encoding='utf-8'))
            output_id = path.stem
            found = context.query_tool_output(output_id, keyword)
            self.assertTrue(found['ok'])
            self.assertTrue(any(keyword in match['content'] for match in found['data']['matches']))
            for other in (self.context('alice'), self.context('bob')):
                self.assertEqual('OUTPUT_NOT_AUTHORIZED', other.query_tool_output(output_id, keyword)['error']['code'])
        self.assertEqual('INVALID_ARGUMENTS', context.query_tool_output('../' + paths[0].name, 'TAIL')['error']['code'])
        self.assertEqual(REQUEST, result[0]['content'])
        self.assertEqual(identifiers, [b['tool_use_id'] for b in result_blocks(result)])
        validate_tool_pairs(result)

    def test_harness_front_budget_runs_before_snip_and_keeps_all_tool_ids_and_current_request(self):
        context = self.context()
        bodies = [body(60000, 'A'), body(95000, 'B'), body(52000, 'C')]
        calls = batch(bodies)[1]['content']
        model = MainModel([calls, [{'type': 'text', 'text': 'Completed synthetic read-only task.'}]])
        harness = Harness(Shop('alice', ROOT / 'data/shop.json'), context, model)
        observed = []
        original_snip = context.snipper.snip_compact

        def observe_snip(messages):
            observed.append(deepcopy(messages))
            return original_snip(messages)

        outputs = iter(bodies)
        with patch.object(harness.shop, 'call', side_effect=lambda name, args: {'ok': True, 'data': {'synthetic_body': next(outputs)}}), \
             patch.object(context.snipper, 'snip_compact', side_effect=observe_snip):
            self.assertEqual('Completed synthetic read-only task.', harness.turn(REQUEST))
        self.assertEqual(2, len(model.calls))
        at_snip = next(m for m in observed if result_blocks(m))
        at_snip_blocks = result_blocks(at_snip)
        self.assertFalse(at_snip_blocks[0]['content'].startswith('<persisted-output>'))
        self.assertTrue(at_snip_blocks[1]['content'].startswith('<persisted-output>'))
        self.assertFalse(at_snip_blocks[2]['content'].startswith('<persisted-output>'))
        self.assertEqual(['budget_1'], context.budget_events[0]['shortened_tool_use_ids'])
        self.assertEqual(['budget_0', 'budget_1', 'budget_2'], [b['tool_use_id'] for b in result_blocks(model.calls[1])])
        self.assertTrue(any(m['role'] == 'user' and m['content'] == REQUEST for m in model.calls[1]))
        self.assertLessEqual(context.snipper.estimate_chars(model.calls[1]), 50000)
        self.assertTrue(context.fit_events)
        events = [json.loads(line) for line in (context.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        front_index = next(i for i, e in enumerate(events) if e['event'] == 'tool_result_budget')
        fit_index = next(i for i, e in enumerate(events) if e['event'] == 'fit_tool_results')
        self.assertLess(front_index, fit_index)
        self.assertEqual('applied', events[front_index]['status'])
        validate_tool_pairs(harness.messages)

    def test_partial_disk_or_outside_path_failure_is_atomic_and_stops_before_next_model(self):
        for failure in ('second_save', 'outside_session'):
            with self.subTest(failure=failure):
                context = self.context()
                bodies = [body(160000, 'A'), body(150000, 'B'), body(140000, 'C')]
                model = MainModel([batch(bodies)[1]['content']])
                harness = Harness(Shop('alice', ROOT / 'data/shop.json'), context, model)
                context.tool_results_dir.mkdir(parents=True)
                prior = context.tool_results_dir / ('0' * 32 + '.txt')
                prior.write_text('previous authorized result', encoding='utf-8')
                context.tool_outputs[prior.stem] = prior
                before_registry = dict(context.tool_outputs)
                outside = self.runtime / 'outside.txt'
                outside.write_text('existing file outside session', encoding='utf-8')
                save_count = 0
                original_save = context.snipper.save_output

                def failing_save(identifier, output):
                    nonlocal save_count
                    save_count += 1
                    if failure == 'outside_session':
                        return outside
                    if save_count == 2:
                        raise OSError(PRIVATE_ERROR)
                    return original_save(identifier, output)

                outputs = iter(bodies)
                with patch.object(harness.shop, 'call', side_effect=lambda name, args: {'ok': True, 'data': {'synthetic_body': next(outputs)}}), \
                     patch.object(context.snipper, 'save_output', side_effect=failing_save):
                    with self.assertRaises(ToolResultBudgetError) as caught:
                        harness.turn(REQUEST)
                self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
                self.assertEqual(1, len(model.calls))
                self.assertEqual(before_registry, context.tool_outputs)
                results = result_blocks(harness.messages)
                self.assertEqual(bodies, [json.loads(b['content'])['data']['synthetic_body'] for b in results])
                self.assertEqual('failed', context.budget_events[-1]['status'])
                snapshot = json.loads((context.session_dir / 'history.json').read_text(encoding='utf-8'))
                self.assertEqual(harness.messages, snapshot['messages'])
                events = [json.loads(line) for line in (context.session_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertTrue(any(e['event'] == 'tool_result_budget' and e['status'] == 'failed' for e in events))
                self.assertEqual('tool_result_budget_failed', events[-1]['reason'])
                self.assertNotIn(PRIVATE_ERROR, json.dumps(events, ensure_ascii=False))
                orphans = [p for p in context.tool_results_dir.glob('*.txt') if p != prior]
                if failure == 'second_save':
                    self.assertEqual(1, len(orphans))
                    self.assertEqual('OUTPUT_NOT_AUTHORIZED', context.query_tool_output(orphans[0].stem, 'A')['error']['code'])
                validate_tool_pairs(harness.messages)


if __name__ == '__main__':
    unittest.main(verbosity=2)
