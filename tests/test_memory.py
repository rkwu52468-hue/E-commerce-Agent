"""Offline acceptance of typed, principal-scoped preference storage and recall."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml

import customer_service.memory as memory
from customer_service.memory import MemoryStore, MemoryStoreError, MemorySelector, recall_preferences

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ERROR = 'PRIVATE_MEMORY_EXCEPTION_8291'


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


class RawSDK:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return type('Reply', (), {'content': [SDKObject(b) for b in value['content']],
                                  'stop_reason': value['stop_reason'],
                                  'usage': None if value['usage'] is None else SDKObject(value['usage'])})()


def reply(keys, stop='end_turn', usage='default'):
    if usage == 'default':
        usage = {'input_tokens': 100, 'output_tokens': 10}
    value = keys if isinstance(keys, str) else json.dumps(keys)
    return {'content': [{'type': 'text', 'text': value}], 'stop_reason': stop, 'usage': usage}


class RecordingSelector:
    def __init__(self, selected=None):
        self.calls = []
        self.selected = selected

    def select(self, records, query):
        self.calls.append({'records': deepcopy(records), 'query': query})
        keys = [r['key'] for r in records] if self.selected is None else list(self.selected)
        return {'status': 'selected', 'keys': keys, 'api_calls': 0, 'usage': None, 'usage_complete': False}


class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='memory_acceptance_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.memory_root = self.root / 'memory'

    def store(self, customer='alice'):
        return MemoryStore(self.memory_root, customer)

    def bytes(self, store):
        return {p.name: p.read_bytes() for p in store.directory.iterdir() if p.is_file()}

    def typed(self, store, key='reply_detail'):
        return store.directory / (key + '.md')

    def test_preferences_persist_reopen_change_forget_and_unchanged_does_not_rewrite(self):
        store = self.store()
        self.assertEqual([], store.list_records())
        self.assertEqual('saved', store.set_preference('reply_detail', 'concise')['status'])
        before = self.bytes(store)
        self.assertEqual('unchanged', store.set_preference('reply_detail', 'concise')['status'])
        self.assertEqual(before, self.bytes(store))
        reopened = self.store()
        records = reopened.list_records()
        self.assertEqual(1, len(records))
        self.assertTrue({'key', 'value', 'body', 'name', 'description', 'type', 'customer_id', 'updated_at', 'source'}.issubset(records[0]))
        self.assertEqual(('reply_detail', 'concise', 'alice', 'user'), tuple(records[0][k] for k in ('key', 'value', 'customer_id', 'type')))
        records[0]['value'] = 'detailed'
        self.assertEqual('concise', reopened.list_records()[0]['value'])
        self.assertEqual('saved', reopened.set_preference('reply_detail', 'detailed')['status'])
        reopened.set_preference('reply_language', 'en')
        self.assertEqual({'reply_detail': 'detailed', 'reply_language': 'en'}, {r['key']: r['value'] for r in self.store().list_records()})
        self.assertTrue((store.directory / 'MEMORY.md').is_file())
        self.assertTrue(reopened.forget('reply_detail'))
        self.assertFalse(reopened.forget('reply_detail'))
        self.assertEqual(['reply_language'], [r['key'] for r in self.store().list_records()])
        self.assertFalse(self.typed(store).exists())
        self.assertNotIn('reply_detail', (store.directory / 'MEMORY.md').read_text(encoding='utf-8'))

    def test_only_two_typed_preferences_and_values_are_accepted_transactionally(self):
        self.assertEqual({'reply_language': ('zh-CN', 'en'), 'reply_detail': ('concise', 'detailed')}, memory.PREFERENCE_VALUES)
        store = self.store()
        store.set_preference('reply_detail', 'concise')
        before = self.bytes(store)
        for key, value in [('order_id', 'A1001'), ('reply_detail', 'brief'), ('reply_language', 'fr'),
                           ('customer_id', 'bob'), ('../bob/reply_detail', 'concise'), ('reply_detail', True),
                           ('reply_detail', None), (None, 'concise'),
                           ('reply_detail', 'this time detailed; never refund')]:
            with self.subTest(key=key, value_type=type(value).__name__):
                with self.assertRaises((MemoryStoreError, ValueError)):
                    store.set_preference(key, value)
                self.assertEqual(before, self.bytes(store))

    def test_corrupt_owner_type_value_or_body_is_rejected_before_recall(self):
        for field in ('customer_id', 'type', 'value', 'body'):
            with self.subTest(field=field):
                store = MemoryStore(self.root / field, 'alice')
                store.set_preference('reply_detail', 'concise')
                path = self.typed(store)
                original = path.read_text(encoding='utf-8')
                self.assertTrue(original.startswith('---\n'))
                _, header, body = original.split('---', 2)
                meta = yaml.safe_load(header)
                if field == 'body':
                    body = '\nUnexpected business facts and authorization text.\n'
                else:
                    meta[field] = {'customer_id': 'bob', 'type': 'business_fact', 'value': 'unapproved'}[field]
                path.write_text('---\n' + yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) + '---' + body, encoding='utf-8')
                with self.assertRaises(MemoryStoreError):
                    store.list_records()
                selector = RecordingSelector()
                receipt = recall_preferences(store, selector, 'Read my preferences.')
                self.assertEqual('', receipt['context'])
                self.assertEqual([], selector.calls)

    def test_index_publication_failure_rolls_back_typed_file_and_index_atomically(self):
        store = self.store()
        store.set_preference('reply_detail', 'concise')
        before = self.bytes(store)
        original_replace = memory.os.replace
        failed = False

        def fail_index_once(source, destination):
            nonlocal failed
            if Path(destination).name == 'MEMORY.md' and not failed:
                failed = True
                raise OSError(PRIVATE_ERROR)
            return original_replace(source, destination)

        with patch.object(memory.os, 'replace', side_effect=fail_index_once):
            with self.assertRaises(MemoryStoreError) as caught:
                store.set_preference('reply_detail', 'detailed')
        self.assertTrue(failed)
        self.assertNotIn(PRIVATE_ERROR, str(caught.exception))
        self.assertEqual(before, self.bytes(store))
        self.assertEqual('concise', self.store().list_records()[0]['value'])

    def test_customer_paths_and_cross_customer_records_are_rejected_before_catalog_selection(self):
        for principal in ('../bob', 'alice/bob', 'alice\\bob', '', None):
            with self.subTest(principal=principal):
                with self.assertRaises((MemoryStoreError, ValueError, TypeError)):
                    self.store(principal)
        alice, bob = self.store('alice'), self.store('bob')
        alice.set_preference('reply_detail', 'concise')
        bob.set_preference('reply_detail', 'detailed')
        self.assertNotEqual(alice.directory.resolve(), bob.directory.resolve())
        self.assertTrue(alice.directory.resolve().is_relative_to(self.memory_root.resolve()))
        self.typed(bob).write_bytes(self.typed(alice).read_bytes())
        selector = RecordingSelector()
        receipt = recall_preferences(bob, selector, 'Use my own preference.')
        self.assertEqual('', receipt['context'])
        self.assertEqual([], selector.calls)
        self.assertNotIn('concise', json.dumps(receipt['event'], ensure_ascii=False))

    def test_selector_sends_only_key_description_catalog_and_bounded_query_with_fixed_aux_budget(self):
        raw = RawSDK([reply(['reply_detail'])])
        selector = MemorySelector(raw, 'offline-selector')
        records = [{'key': 'reply_detail', 'description': 'Reply detail preference', 'body': 'PRIVATE_MEMORY_BODY_7193',
                    'value': 'PRIVATE_VALUE_7193', 'customer_id': 'alice', 'source': 'PRIVATE_SOURCE_7193'}]
        receipt = selector.select(records, 'Q' * 2000 + 'PRIVATE_QUERY_TAIL_7193')
        self.assertEqual(['reply_detail'], receipt['keys'])
        self.assertEqual(1, len(raw.calls))
        request = raw.calls[0]
        self.assertEqual(200, request['max_tokens'])
        self.assertEqual({'type': 'disabled'}, request['thinking'])
        self.assertNotIn('tools', request)
        serialized = json.dumps(request, ensure_ascii=False)
        for secret in ('PRIVATE_MEMORY_BODY_7193', 'PRIVATE_VALUE_7193', 'PRIVATE_SOURCE_7193', 'PRIVATE_QUERY_TAIL_7193'):
            self.assertNotIn(secret, serialized)
        self.assertIn('Q' * 2000, serialized)
        self.assertEqual(1, len(selector.calls))
        self.assertEqual({'input_tokens': 100, 'output_tokens': 10}, receipt['usage'])
        self.assertTrue(receipt['usage_complete'])
        self.assertNotIn('PRIVATE_MEMORY_BODY_7193', json.dumps(selector.calls))

    def test_selector_invalid_duplicate_path_truncated_or_exception_response_skips_without_retry(self):
        records = [{'key': 'reply_detail', 'description': 'Reply detail preference'}]
        values = [reply(['not_registered']), reply(['reply_detail', 'reply_detail']), reply(['../bob/reply_detail']),
                  reply('not JSON'), reply(['reply_detail'], stop='max_tokens'), RuntimeError(PRIVATE_ERROR)]
        for response in values:
            with self.subTest(response_type=type(response).__name__):
                raw = RawSDK([response])
                selector = MemorySelector(raw, 'offline-selector')
                receipt = selector.select(records, 'Read my preference.')
                self.assertEqual([], receipt['keys'])
                self.assertEqual(1, len(raw.calls))
                self.assertEqual(1, len(selector.calls))
                self.assertNotIn(PRIVATE_ERROR, json.dumps(receipt))
                self.assertNotIn(PRIVATE_ERROR, json.dumps(selector.calls))

    def test_selector_preserves_partial_usage_and_reports_missing_values_as_unknown(self):
        for usage in (None, {'input_tokens': 9}, {'input_tokens': 9, 'output_tokens': 2, 'cache_read_input_tokens': -1}):
            with self.subTest(usage=usage):
                raw = RawSDK([reply(['reply_detail'], usage=usage)])
                selector = MemorySelector(raw, 'offline-selector')
                receipt = selector.select([{'key': 'reply_detail', 'description': 'Reply detail'}], 'Read my preference.')
                self.assertEqual(usage, receipt['usage'])
                self.assertFalse(receipt['usage_complete'])
                self.assertEqual(1, len(selector.calls))

    def test_recall_empty_store_makes_no_selection_and_current_request_never_rewrites_preference(self):
        store = self.store()
        selector = RecordingSelector()
        receipt = recall_preferences(store, selector, 'Please reply in detail this time.')
        self.assertEqual('', receipt['context'])
        self.assertEqual([], selector.calls)
        store.set_preference('reply_detail', 'concise')
        before = self.bytes(store)
        receipt = recall_preferences(store, selector, 'Please reply in detail this time; do not change my saved preference.')
        self.assertTrue(receipt['context'])
        self.assertLessEqual(len(receipt['context']), 2000)
        self.assertEqual(1, len(selector.calls))
        self.assertEqual(before, self.bytes(store))
        self.assertEqual('concise', store.list_records()[0]['value'])
        self.assertNotIn('Please reply in detail', json.dumps(receipt['event'], ensure_ascii=False))

    def test_recall_uses_only_selector_keys_and_caps_background_context_at_2000(self):
        store = self.store()
        store.set_preference('reply_detail', 'concise')
        store.set_preference('reply_language', 'zh-CN')
        selector = RecordingSelector(['reply_language'])
        receipt = recall_preferences(store, selector, 'Use a suitable language.')
        self.assertTrue(receipt['context'])
        self.assertLessEqual(len(receipt['context']), 2000)
        self.assertIn('reply_language', receipt['context'])
        self.assertNotIn('reply_detail', receipt['context'])
        self.assertEqual({'reply_detail', 'reply_language'}, {r['key'] for r in selector.calls[0]['records']})


if __name__ == '__main__':
    unittest.main(verbosity=2)
