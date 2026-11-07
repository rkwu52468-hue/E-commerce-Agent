"""Offline acceptance of explicit memory commands and parent-only recall."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.live_session import LiveSession
from customer_service.memory import MemoryStore, MemoryStoreError
from customer_service.shop import Shop
from customer_service.summary import SummaryClient
from test_memory import RecordingSelector, PRIVATE_ERROR

ROOT = Path(__file__).resolve().parents[1]


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
        result = self.replies.pop(0)
        if isinstance(result, BaseException):
            raise result
        blocks = deepcopy(result)
        return {'content': blocks, 'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in blocks) else 'end_turn',
                'usage': {'input_tokens': 10, 'output_tokens': 2}}


class FakeAuth:
    def verify(self, customer_id, password):
        return customer_id if customer_id in ('alice', 'bob') and password == 'offline-password' else None


class MemorySessionTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='memory_session_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'data').mkdir()
        shutil.copyfile(ROOT / 'data/shop.json', self.root / 'data/shop.json')
        self.store = MemoryStore(self.root / 'memory', 'alice')
        self.store.set_preference('reply_detail', 'concise')

    def build(self, replies, selector, store=None, customer='alice', summary=None, child=None):
        context = ContextManager(self.root / 'runtime', customer or 'guest', summary_client=summary)
        model = Model(replies)
        harness = Harness(Shop(customer, self.root / 'data/shop.json'), context, model,
                          subagent_model=child, memory_store=self.store if store is None else store,
                          memory_selector=selector)
        return harness, context, model

    def events(self, context):
        path = context.session_dir / 'events.jsonl'
        return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []

    def test_one_recall_per_parent_turn_not_per_tool_continuation_and_current_query_does_not_save(self):
        selector = RecordingSelector()
        replies = [[call('memory_order', 'get_order', order_id='A1001')], [text('Detailed explanation requested this time.')],
                   [text('Second turn answer.')]]
        harness, context, model = self.build(replies, selector)
        before = {p.name: p.read_bytes() for p in self.store.directory.iterdir() if p.is_file()}
        harness.turn('This time explain in detail; keep my saved concise preference unchanged.')
        self.assertEqual(1, len(selector.calls))
        self.assertTrue(harness.memory_context)
        for request in model.calls:
            self.assertIn(harness.memory_context, request['system'])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.store.directory.iterdir() if p.is_file()})
        self.assertFalse({'remember', 'forget', 'save_memory', 'memory_write', 'load_memory'}.intersection(t['name'] for t in harness.tools))
        harness.turn('Another question.')
        self.assertEqual(2, len(selector.calls))
        self.assertEqual(3, len(model.calls))
        validate_tool_pairs(harness.messages)

    def test_broken_store_or_selector_clears_previous_context_and_wrong_principal_never_reaches_catalog(self):
        selector = RecordingSelector()
        harness, context, model = self.build([[text('One.')], [text('Two.')], [text('Three.')]], selector)
        harness.turn('First recall.')
        previous = harness.memory_context
        self.assertTrue(previous)
        with patch.object(self.store, 'list_records', side_effect=MemoryStoreError(PRIVATE_ERROR)):
            harness.turn('Store unavailable now.')
        self.assertEqual('', harness.memory_context)
        self.assertNotIn(previous, model.calls[1]['system'])
        self.assertEqual(1, len(selector.calls))
        with patch.object(selector, 'select', side_effect=RuntimeError(PRIVATE_ERROR)):
            harness.turn('Selector unavailable now.')
        self.assertEqual('', harness.memory_context)
        self.assertNotIn(previous, model.calls[2]['system'])
        self.assertNotIn(PRIVATE_ERROR, json.dumps(self.events(context), ensure_ascii=False))
        other_selector = RecordingSelector()
        try:
            mismatched, _, _ = self.build([[text('No foreign preference.')]], other_selector, customer='bob')
        except (ValueError, MemoryStoreError):
            self.assertEqual([], other_selector.calls)
        else:
            mismatched.turn('Bob question.')
            self.assertEqual('', mismatched.memory_context)
            self.assertEqual([], other_selector.calls)
        # Rebinding a live Harness must be rejected before enumerating records,
        # even though this foreign store is valid for its own principal.
        foreign_store = MemoryStore(self.root / 'memory', 'bob')
        foreign_store.set_preference('reply_detail', 'detailed')
        harness.memory_store = foreign_store
        model.replies.append([text('No foreign preference was loaded.')])
        with patch.object(foreign_store, 'list_records', side_effect=AssertionError('Foreign records enumerated')) as listing:
            harness.turn('After accidental runtime rebinding.')
        listing.assert_not_called()
        self.assertEqual('', harness.memory_context)
        self.assertEqual(1, len(selector.calls))
        last_recall = [e for e in self.events(context) if e['event'] == 'memory_recall'][-1]
        self.assertEqual('principal_mismatch', last_recall['status'])
        self.assertEqual(0, last_recall['api_calls'])

    def test_local_budget_preflight_skips_selector_and_reactive_retry_does_not_select_twice(self):
        selector = RecordingSelector()
        harness, _, model = self.build([], selector)
        harness.turn('Unshortenable user content. ' * 2500)
        self.assertEqual([], model.calls)
        self.assertEqual([], selector.calls)

        class RawSummary:
            def __init__(self):
                self.calls = []
                self.messages = self

            def create(self, **kwargs):
                self.calls.append(deepcopy(kwargs))
                return SimpleNamespace(content=[SimpleNamespace(type='text', text='Earlier discussion summarized as reference.')],
                                       stop_reason='end_turn',
                                       usage=SimpleNamespace(model_dump=lambda **kwargs: {'input_tokens': 10, 'output_tokens': 2}))

        raw = RawSummary()
        summary = SummaryClient(raw, 'offline-summary')
        selector = RecordingSelector()
        harness, context, model = self.build([RuntimeError('prompt_too_long'), [text('Recovered answer.')]], selector, summary=summary)
        harness.messages = [{'role': 'user', 'content': 'Old user discussion. ' * 200},
                            {'role': 'assistant', 'content': 'Old answer discussion. ' * 200}]
        self.assertEqual('Recovered answer.', harness.turn('Continue the current query.'))
        self.assertEqual(1, len(selector.calls))
        self.assertEqual(2, len(model.calls))
        self.assertEqual(1, len(raw.calls))
        self.assertEqual('reactive', context.summary_events[-1]['trigger'])
        self.assertTrue(all(harness.memory_context in request['system'] for request in model.calls))

    def test_memory_commands_are_explicit_local_disabled_guest_or_invalid_inputs_are_rejected(self):
        model = Model([])
        disabled = LiveSession(self.root, model, store='json', allow_memory=False)
        self.assertFalse(disabled.memory_command('/remember reply_detail concise')['ok'])
        session = LiveSession(self.root, model, store='json', allow_memory=True, memory_selector=RecordingSelector())
        self.assertIsNone(session.harness.memory_store)
        self.assertFalse(session.memory_command('/remember reply_detail concise')['ok'])
        self.assertTrue(session.login('alice', 'offline-password', FakeAuth()))
        for command in ('/remember order_id A1001', '/remember reply_detail concise bob',
                        '/remember reply_detail unknown', '/forget ../bob/reply_detail', '/memories extra'):
            with self.subTest(command=command):
                self.assertFalse(session.memory_command(command)['ok'])
        self.assertEqual([], session.harness.messages)
        self.assertEqual([], model.calls)
        audit_path = session.context.session_dir / 'audit.jsonl'
        if audit_path.exists():
            audit = [json.loads(x) for x in audit_path.read_text(encoding='utf-8').splitlines()]
            encoded = json.dumps(audit, ensure_ascii=False)
            self.assertNotIn('/remember', encoded)
            self.assertNotIn('A1001', encoded)
            self.assertFalse(any({'value', 'body', 'command', 'raw_command'}.intersection(row) for row in audit))

    def test_login_logout_and_bob_switch_isolate_messages_but_alice_preferences_persist_until_forget(self):
        model = Model([])
        session = LiveSession(self.root, model, store='json', allow_memory=True, memory_selector=RecordingSelector())
        auth = FakeAuth()
        self.assertTrue(session.login('alice', 'offline-password', auth))
        self.assertTrue(session.memory_command('/remember reply_detail concise')['ok'])
        self.assertTrue(session.memory_command('/remember reply_language zh-CN')['ok'])
        self.assertTrue(session.memory_command('/memories')['ok'])
        alice_store = session.harness.memory_store
        self.assertEqual({'reply_detail': 'concise', 'reply_language': 'zh-CN'}, {r['key']: r['value'] for r in alice_store.list_records()})
        alice_session = session.context.session_id
        audit = [json.loads(x) for x in (session.context.session_dir / 'audit.jsonl').read_text(encoding='utf-8').splitlines()]
        encoded = json.dumps(audit, ensure_ascii=False)
        self.assertNotIn('concise', encoded)
        self.assertNotIn('zh-CN', encoded)
        self.assertNotIn('/remember', encoded)
        session.logout()
        self.assertIsNone(session.harness.memory_store)
        self.assertNotEqual(alice_session, session.context.session_id)
        self.assertFalse(session.memory_command('/memories')['ok'])
        self.assertTrue(session.login('bob', 'offline-password', auth))
        self.assertEqual([], session.harness.memory_store.list_records())
        self.assertEqual([], session.harness.messages)
        self.assertTrue(session.login('alice', 'offline-password', auth))
        self.assertEqual(2, len(session.harness.memory_store.list_records()))
        self.assertTrue(session.memory_command('/forget reply_detail')['ok'])
        self.assertEqual(['reply_language'], [r['key'] for r in session.harness.memory_store.list_records()])
        self.assertEqual([], session.harness.messages)
        self.assertEqual([], model.calls)

    def test_child_has_no_memory_store_or_selector_inheritance_and_still_only_eight_read_only_tools(self):
        selector = RecordingSelector()
        child = Model([[call('child_without_memory', 'get_product', product_id='P100')], [text('Public fact checked.')]])
        harness, context, model = self.build([[call('memory_parent_task', 'task', prompt='Read P100 only.')], [text('Parent reply.')]], selector, child=child)
        harness.turn('Delegate one fact lookup.')
        self.assertEqual(1, len(selector.calls))
        self.assertTrue(harness.memory_context)
        self.assertEqual(2, len(child.calls))
        self.assertEqual(8, len(child.calls[0]['tools']))
        self.assertNotIn(harness.memory_context, child.calls[0]['system'])
        self.assertFalse({'remember', 'forget', 'save_memory', 'memory_write', 'load_memory'}.intersection(t['name'] for t in child.calls[0]['tools']))
        task = next(json.loads(b['content']) for m in model.calls[1]['messages'] if isinstance(m.get('content'), list)
                    for b in m['content'] if b.get('type') == 'tool_result' and b.get('tool_use_id') == 'memory_parent_task')
        child_dir = Path(task['data']['child_session_dir'])
        events = [json.loads(x) for x in (child_dir / 'events.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertFalse(any(e['event'] == 'memory_recall' for e in events))
        self.assertEqual(1, len(task['data']['evidence']))
        validate_tool_pairs(harness.messages)


if __name__ == '__main__':
    unittest.main(verbosity=2)
