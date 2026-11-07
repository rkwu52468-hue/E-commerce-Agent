"""Offline integration checks; fake retrieval proves message and access flow only."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.live_session import LiveSession
from customer_service.shop import Shop
from customer_service.skills import SkillLoader
from customer_service.team_worker import TeamWorker


CLAUSE = 'KNOWLEDGE_TOOL_ONLY_CLAUSE_52741'
ROOT = Path(__file__).resolve().parents[1]


def call(identifier, name='search_knowledge', **arguments):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': arguments}


def text(value):
    return {'type': 'text', 'text': value}


class Model:
    mode = 'scripted-offline'

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools, system):
        self.calls.append({'messages': deepcopy(messages), 'tools': deepcopy(tools), 'system': system})
        if not self.replies:
            raise AssertionError('Unexpected scripted model request')
        content = deepcopy(self.replies.pop(0))
        return {'content': content, 'stop_reason': 'tool_use' if any(b.get('type') == 'tool_use' for b in content) else 'end_turn',
                'usage': {'input_tokens': 10, 'output_tokens': 2}}


class Knowledge:
    def __init__(self):
        self.queries = []
        self.close_count = 0
        self.result = {'ok': True, 'data': {'matches': [
            {'chunk_id': 'returns-1', 'text': CLAUSE, 'source': 'knowledge-fixture/returns.md',
             'version': 'fixture-v1', 'score': 0.8}]}, 'simulated': True}

    def search(self, query):
        if self.close_count:
            raise AssertionError('Closed index must not be queried')
        self.queries.append(query)
        return deepcopy(self.result)

    def close(self):
        self.close_count += 1


class KnowledgeIntegrationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='knowledge_integration_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'data').mkdir()
        fixture = {'dataset_version': 'knowledge-integration-fixture',
                   'customers': {'alice': {}, 'bob': {}}, 'products': {}, 'orders': {},
                   'policies': {'returns': {'policy_id': 'returns', 'title': '模拟退货规则',
                       'text': '商品未使用并保留包装，申请由人工审核。',
                       'source': 'integration-fixture#returns', 'version': 'fixture-v1'}}}
        self.data_path = self.root / 'data/shop.json'
        self.data_path.write_text(json.dumps(fixture, ensure_ascii=False), encoding='utf-8')

    def harness(self, model=None, knowledge=None, **kwargs):
        return Harness(Shop(None, self.data_path), ContextManager(self.root / 'runtime', 'guest'),
                       model or Model(), knowledge=knowledge, **kwargs)

    def test_default_tool_names_unchanged_and_disabled_does_not_query(self):
        harness = self.harness()
        expected = {tool['name'] for tool in harness.shop.tool_definitions()}
        expected.update({'query_archive', 'query_tool_output', 'todo_write', 'load_skill', 'task'})
        self.assertEqual(expected, {tool['name'] for tool in harness.tools})
        self.assertEqual('KNOWLEDGE_DISABLED', harness.call_tool('search_knowledge', {'query': '退货'})['error']['code'])

    def test_public_result_enters_next_messages_and_saved_history_with_audit(self):
        knowledge = Knowledge()
        model = Model([[call('search-1', query='盒子拆了还能退吗')], [text('按引用说明并保留资料缺口。')]])
        harness = self.harness(model, knowledge)
        with redirect_stdout(io.StringIO()):
            harness.turn('盒子拆了还能退吗')
        self.assertEqual(['盒子拆了还能退吗'], knowledge.queries)
        self.assertIsNone(harness.shop.customer_id)
        self.assertIn('search_knowledge', {tool['name'] for tool in model.calls[0]['tools']})
        self.assertNotIn(CLAUSE, json.dumps(model.calls[0], ensure_ascii=False))
        returned = [b for message in model.calls[1]['messages'] if isinstance(message.get('content'), list)
                    for b in message['content'] if b.get('type') == 'tool_result']
        self.assertEqual(1, len(returned))
        self.assertEqual('search-1', returned[0]['tool_use_id'])
        self.assertEqual(knowledge.result, json.loads(returned[0]['content']))
        validate_tool_pairs(harness.messages)
        history = json.loads((harness.context.session_dir / 'history.json').read_text(encoding='utf-8'))
        validate_tool_pairs(history['messages'])
        self.assertIn(CLAUSE, json.dumps(history, ensure_ascii=False))
        audit_text = (harness.context.session_dir / 'audit.jsonl').read_text(encoding='utf-8')
        audit = [json.loads(line) for line in audit_text.splitlines()]
        events = [entry['event'] for entry in audit]
        self.assertEqual(['UserPromptSubmit', 'PreToolUse', 'PostToolUse', 'Stop'], events)
        post = next(entry for entry in audit if entry['event'] == 'PostToolUse')
        self.assertEqual('search_knowledge', post['tool'])
        self.assertEqual('OK', post['result_code'])
        self.assertNotIn(CLAUSE, audit_text)
        self.assertNotIn('盒子拆了还能退吗', audit_text)

    def test_scope_identity_and_invalid_query_arguments_never_reach_retriever(self):
        knowledge = Knowledge()
        harness = self.harness(knowledge=knowledge)
        invalid = [None, [], {}, {'query': ''}, {'query': '  '}, {'query': 3}, {'query': 'x' * 501},
                   {'query': '退货', 'customer_id': 'bob'}, {'query': '退货', 'store_id': 'foreign'},
                   {'query': '退货', 'path': 'outside'}, {'query': '退货', 'top_k': 99}]
        for args in invalid:
            with self.subTest(args=args):
                self.assertEqual('INVALID_ARGUMENTS', harness.call_tool('search_knowledge', args)['error']['code'])
        self.assertEqual([], knowledge.queries)
        self.assertTrue(harness.call_tool('search_knowledge', {'query': 'x' * 500})['ok'])

    def test_backend_failures_and_malformed_results_return_safe_error(self):
        secret = 'backend-secret-that-must-not-leak'
        for value in (RuntimeError(secret), None, [], {'ok': 'true'}, {'ok': False, 'error': {}},
                      {'ok': True, 'data': object()}):
            with self.subTest(value=type(value).__name__):
                knowledge = SimpleNamespace(search=Mock(side_effect=value) if isinstance(value, Exception)
                                            else Mock(return_value=value))
                result = self.harness(knowledge=knowledge).call_tool('search_knowledge', {'query': '退货'})
                self.assertEqual('KNOWLEDGE_UNAVAILABLE', result['error']['code'])
                self.assertNotIn(secret, json.dumps(result))

    def test_empty_hit_and_explicit_backend_error_remain_distinct(self):
        knowledge = Knowledge()
        harness = self.harness(knowledge=knowledge)
        for result in ({'ok': True, 'data': {'matches': [], 'status': 'no_match'}},
                       {'ok': False, 'error': {'code': 'KNOWLEDGE_INDEX_STALE', 'message': '本地索引需要重建。'}}):
            knowledge.result = result
            self.assertEqual(result, harness.call_tool('search_knowledge', {'query': '不在资料内的问题'}))

    def test_harness_close_rejects_queries_without_closing_shared_resource(self):
        knowledge = Knowledge()
        harness = self.harness(knowledge=knowledge)
        harness.close()
        self.assertEqual('KNOWLEDGE_SESSION_CLOSED', harness.call_tool('search_knowledge', {'query': '退货'})['error']['code'])
        self.assertEqual([], knowledge.queries)
        self.assertEqual(0, knowledge.close_count)

    def test_login_logout_replace_history_and_keep_shared_index_alive_until_final_close(self):
        knowledge = Knowledge()
        session = LiveSession(self.root, Model(), store='json', knowledge=knowledge)
        verifier = SimpleNamespace(verify=lambda identity, password: identity if password == 'offline-password' else None)
        previous = session.harness
        old_paths = {previous.context.session_dir}
        for identity in ('alice', 'bob', None):
            previous.messages = [{'role': 'user', 'content': 'private marker for old context'}]
            if identity is None:
                session.logout()
            else:
                self.assertTrue(session.login(identity, 'offline-password', verifier))
            self.assertEqual([], session.harness.messages)
            self.assertNotIn(session.context.session_dir, old_paths)
            old_paths.add(session.context.session_dir)
            self.assertEqual(identity, session.customer_id)
            self.assertIs(knowledge, session.harness.knowledge)
            self.assertEqual(0, knowledge.close_count)
            self.assertTrue(session.harness.call_tool('search_knowledge', {'query': '退货'})['ok'])
            self.assertEqual('KNOWLEDGE_SESSION_CLOSED', previous.call_tool('search_knowledge', {'query': '退货'})['error']['code'])
            previous = session.harness
        session.close()
        session.close()
        self.assertEqual(1, knowledge.close_count)
        self.assertEqual('KNOWLEDGE_SESSION_CLOSED', session.harness.call_tool('search_knowledge', {'query': '退货'})['error']['code'])
        self.assertEqual(3, len(knowledge.queries))

    def test_cleanup_runs_once_even_when_harness_or_index_close_fails(self):
        for owner in ('harness', 'index'):
            with self.subTest(owner=owner):
                knowledge = Knowledge()
                session = LiveSession(self.root, Model(), store='json', knowledge=knowledge)
                if owner == 'harness':
                    session.harness.close = Mock(side_effect=RuntimeError('offline cleanup failure'))
                else:
                    knowledge.close = Mock(side_effect=RuntimeError('offline cleanup failure'))
                with self.assertRaises(RuntimeError):
                    session.close()
                session.close()
                self.assertTrue(session._closed)
                if owner == 'index':
                    self.assertEqual(1, knowledge.close.call_count)
                else:
                    self.assertEqual(1, knowledge.close_count)

    def test_subagent_cannot_inherit_or_call_main_knowledge_tool(self):
        knowledge = Knowledge()
        child_model = Model([[call('child-guess', query='退货')], [text('当前子任务不支持此工具。')]])
        model = Model([[call('delegate-1', 'task', prompt='核对公开资料')], [text('说明子任务资料缺口。')]])
        harness = self.harness(model, knowledge, subagent_model=child_model)
        with redirect_stdout(io.StringIO()):
            harness.turn('核对公开资料')
        self.assertEqual([], knowledge.queries)
        self.assertNotIn('search_knowledge', {tool['name'] for tool in child_model.calls[0]['tools']})
        child_results = [b for m in child_model.calls[1]['messages'] if isinstance(m.get('content'), list)
                         for b in m['content'] if b.get('type') == 'tool_result']
        self.assertEqual('SUBAGENT_TOOL_NOT_ALLOWED', json.loads(child_results[0]['content'])['error']['code'])

    def test_policy_teammate_keeps_original_role_allowlist(self):
        shop = Shop('alice', self.data_path)
        parent = ContextManager(self.root / 'runtime', 'alice')
        worker = TeamWorker('policy', shop, parent.session_dir, Model(), SkillLoader())
        self.addCleanup(worker.close)
        self.assertIsNone(worker.harness.knowledge)
        self.assertEqual({'get_policy', 'load_skill'}, {tool['name'] for tool in worker.harness.tools})
        result = worker.harness.call_tool('search_knowledge', {'query': '退货'})
        self.assertEqual('TEAM_TOOL_NOT_ALLOWED', result['error']['code'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
