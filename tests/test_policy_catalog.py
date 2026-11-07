"""Offline regressions for shop-local discoverable policy IDs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.shop import Shop, TOOL_DEFINITIONS

ROOT = Path(__file__).resolve().parents[1]
CLAUSE = 'POLICY_CLAUSE_ONLY_AVAILABLE_FROM_TOOL_8194'
PRIVATE = 'PRIVATE_FOREIGN_ORDER_NOTE_6284'


def call(identifier, name, **arguments):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': arguments}


def text(value):
    return {'type': 'text', 'text': value}


class Model:
    mode = 'scripted-offline'

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({'messages': deepcopy(messages), 'tools': deepcopy(tools), 'system': system})
        blocks = deepcopy(self.replies.pop(0))
        return {'content': blocks, 'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in blocks) else 'end_turn',
                'usage': {'input_tokens': 10, 'output_tokens': 2}}


def policy_tool(definitions):
    return next(tool for tool in definitions if tool['name'] == 'get_policy')


def policy_ids(definitions):
    return policy_tool(definitions)['input_schema']['properties']['policy_id'].get('enum')


class PolicyCatalogTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='policy_catalog_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.runtime = Path(self.temporary.name)
        self.base = json.loads((ROOT / 'data/shop.json').read_text(encoding='utf-8'))
        self.delivery = {'policy_id': 'delivery', 'title': '配送信息核验目录标题', 'version': 'catalog-fixture-v1',
                         'text': CLAUSE, 'source': 'catalog-fixture.json#policies.delivery'}

    def shop(self, name, policies, customer='alice'):
        fixture = deepcopy(self.base)
        fixture['policies'] = deepcopy(policies)
        fixture['orders']['B2001']['internal_note'] = PRIVATE
        path = self.runtime / (name + '.json')
        path.write_text(json.dumps(fixture, ensure_ascii=False), encoding='utf-8')
        return Shop(customer, path)

    def test_catalog_uses_actual_sorted_ids_and_titles_without_exposing_policy_clauses(self):
        policies = {'returns': self.base['policies']['returns'], 'delivery': self.delivery}
        shop = self.shop('two-policies', policies)
        definitions = shop.tool_definitions()
        self.assertEqual(['delivery', 'returns'], policy_ids(definitions))
        self.assertEqual({tool['name'] for tool in TOOL_DEFINITIONS}, {tool['name'] for tool in definitions})
        definition = policy_tool(definitions)
        for identifier, record in policies.items():
            self.assertIn(identifier, definition['description'])
            self.assertIn(record['title'], definition['description'])
            result = shop.call('get_policy', {'policy_id': identifier})
            self.assertTrue(result['ok'])
            self.assertEqual(record, result['data'])
        self.assertNotIn(CLAUSE, json.dumps(definitions, ensure_ascii=False))
        self.assertNotIn(PRIVATE, json.dumps(definitions, ensure_ascii=False))
        self.assertEqual(['policy_id'], definition['input_schema']['required'])
        self.assertFalse(definition['input_schema']['additionalProperties'])

    def test_shop_instances_and_returned_nested_definitions_do_not_pollute_each_other_or_template(self):
        template = deepcopy(TOOL_DEFINITIONS)
        first = self.shop('returns-only', {'returns': self.base['policies']['returns']})
        second = self.shop('delivery-and-returns', {'returns': self.base['policies']['returns'], 'delivery': self.delivery})
        one, two = first.tool_definitions(), second.tool_definitions()
        self.assertEqual(['returns'], policy_ids(one))
        self.assertEqual(['delivery', 'returns'], policy_ids(two))
        policy_ids(two).append('injected_id')
        policy_tool(two)['description'] = 'Mutated caller copy'
        policy_tool(two)['input_schema']['required'].append('customer_id')
        self.assertEqual(['returns'], policy_ids(first.tool_definitions()))
        self.assertEqual(['delivery', 'returns'], policy_ids(second.tool_definitions()))
        self.assertNotEqual('Mutated caller copy', policy_tool(second.tool_definitions())['description'])
        self.assertEqual(['policy_id'], policy_tool(second.tool_definitions())['input_schema']['required'])
        self.assertEqual(template, TOOL_DEFINITIONS)

    def test_empty_catalog_does_not_emit_empty_enum_and_unknown_policy_is_still_not_found(self):
        shop = self.shop('no-policies', {}, customer=None)
        definition = policy_tool(shop.tool_definitions())
        self.assertNotIn('enum', definition['input_schema']['properties']['policy_id'])
        description = definition['description'].lower()
        self.assertTrue(any(word in description for word in ('无', '没有', '暂无', '未登记', '为空', 'no policies', 'no policy')))
        for identifier in ('returns', 'shipping', 'delivery'):
            with self.subTest(identifier=identifier):
                result = shop.call('get_policy', {'policy_id': identifier})
                self.assertFalse(result['ok'])
                self.assertEqual('NOT_FOUND', result['error']['code'])
                self.assertNotIn('data', result)

    def test_parent_and_actual_child_receive_same_catalog_and_delivery_result_enters_task_evidence(self):
        shop = self.shop('parent-child-catalog', {'returns': self.base['policies']['returns'], 'delivery': self.delivery})
        context = ContextManager(self.runtime, 'alice')
        parent = Model([[call('policy_task', 'task', prompt='查询实际目录中的 delivery 政策并报告来源。')],
                        [text('Parent cites the child evidence.')]])
        child = Model([[call('child_delivery', 'get_policy', policy_id='delivery')], [text('Delivery policy was read with its source.')]])
        harness = Harness(shop, context, parent, subagent_model=child)
        harness.turn('Delegate the policy check.')
        self.assertEqual(['delivery', 'returns'], policy_ids(parent.calls[0]['tools']))
        self.assertEqual(policy_tool(parent.calls[0]['tools']), policy_tool(child.calls[0]['tools']))
        self.assertEqual(8, len(child.calls[0]['tools']))
        self.assertNotIn('task', {tool['name'] for tool in child.calls[0]['tools']})
        result_block = next(b for m in parent.calls[1]['messages'] if isinstance(m.get('content'), list)
                            for b in m['content'] if b.get('type') == 'tool_result' and b.get('tool_use_id') == 'policy_task')
        result = json.loads(result_block['content'])
        self.assertTrue(result['ok'])
        evidence = next(entry for entry in result['data']['evidence'] if entry['tool_use_id'] == 'child_delivery')
        self.assertTrue(evidence['result']['ok'])
        self.assertEqual(self.delivery, evidence['result']['data'])
        self.assertNotIn(PRIVATE, json.dumps([parent.calls, child.calls, result], ensure_ascii=False))
        self.assertNotIn(CLAUSE, json.dumps(parent.calls[0]['tools'], ensure_ascii=False))
        self.assertNotIn(CLAUSE, json.dumps(child.calls[0]['tools'], ensure_ascii=False))
        self.assertEqual(2, len(parent.calls))
        self.assertEqual(2, len(child.calls))
        validate_tool_pairs(harness.messages)
        child_directory = Path(result['data']['child_session_dir'])
        child_history = json.loads((child_directory / 'history.json').read_text(encoding='utf-8'))
        validate_tool_pairs(child_history['messages'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
