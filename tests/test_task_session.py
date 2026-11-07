"""Offline s10 acceptance of session-bound persistent investigation tasks."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import app
from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.live_session import LiveSession
from customer_service.shop import Shop
from customer_service.task_system import CustomerTaskStore, TaskStoreError
from test_subagent import MainModel

ROOT = Path(__file__).resolve().parents[1]
TASK_NAMES = {'create_task', 'update_task', 'list_tasks', 'get_task', 'claim_task', 'complete_task'}
_AUTO = object()


def text(value):
    return {'type': 'text', 'text': value}


def call(identifier, name, **arguments):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': arguments}


def result_map(messages):
    return {block['tool_use_id']: json.loads(block['content'])
            for message in messages if isinstance(message.get('content'), list)
            for block in message['content'] if block.get('type') == 'tool_result'}


class FakeAuth:
    def verify(self, customer_id, password):
        return customer_id if customer_id in ('alice', 'bob') and password == 'offline-password' else None


class TaskSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='task_session_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'data').mkdir()
        shutil.copyfile(ROOT / 'data/shop.json', self.root / 'data/shop.json')
        shutil.copyfile(ROOT / 'config.json', self.root / 'config.json')
        self.base = self.root / 'data/tasks'

    def build(self, replies=None, *, customer='alice', enabled=True, store=_AUTO, child=None, model=None):
        if store is _AUTO:
            store = CustomerTaskStore(self.base, customer) if customer is not None and enabled else None
        context = ContextManager(self.root / 'runtime', customer or 'guest')
        model = MainModel(replies or []) if model is None else model
        h = Harness(Shop(customer, self.root / 'data/shop.json'), context, model,
                    enable_tasks=enabled, task_store=store, subagent_model=child)
        return h, context, model

    def create(self, store, subject='核对本人订单'):
        result = store.call('create_task', {'subject': subject, 'description': '内部只读调查步骤'})
        self.assertTrue(result['ok'], result)
        return result['data']['task']

    def rows(self, context, name='audit.jsonl'):
        return [json.loads(line) for line in (context.session_dir / name).read_text(encoding='utf-8').splitlines()]

    def test_default_harness_does_not_advertise_tasks_or_add_auxiliary_requests(self):
        context = ContextManager(self.root / 'runtime', 'alice')
        model = MainModel([[text('P100的一般说明。')]])
        h = Harness(Shop('alice', self.root / 'data/shop.json'), context, model)
        self.assertFalse(TASK_NAMES.intersection(t['name'] for t in h.tools))
        with redirect_stdout(io.StringIO()):
            self.assertEqual('P100的一般说明。', h.turn('说明P100即可。'))
        self.assertEqual(1, len(model.calls))
        self.assertFalse(self.base.exists())
        for name, args in [('create_task', {'subject': '不得写入'}), ('list_tasks', {})]:
            denied = h.call_tool(name, args)
            self.assertEqual('TASK_SYSTEM_DISABLED', denied['error']['code'])
        self.assertFalse(self.base.exists())

    def test_parent_uses_returned_random_ids_to_link_claim_complete_and_unlock_in_later_batches(self):
        case = self

        class Investigation(MainModel):
            def __init__(self):
                super().__init__([])
                self.ids = None
                self.first = [text('先建立两个调查节点，再根据返回编号添加依赖。'),
                              call('create_order', 'create_task', subject='核对A1001订单', description='读取本人订单金额与来源'),
                              call('create_tracking', 'create_task', subject='查询A1001物流', description='核对运输状态')]

            def complete(self, messages, tools, system):
                self.calls.append({'messages': deepcopy(messages), 'tools': deepcopy(tools), 'system': system})
                step = len(self.calls)
                received = result_map(messages)
                if step == 1:
                    blocks = self.first
                elif step == 2:
                    a, b = [received[name]['data']['task'] for name in ('create_order', 'create_tracking')]
                    for task in (a, b):
                        case.assertRegex(task['id'], r'^task_[0-9a-f]{8}$')
                        case.assertEqual('pending', task['status'])
                        case.assertIsNone(task['owner'])
                        case.assertEqual([], task['blockedBy'])
                    case.assertNotEqual(a['id'], b['id'])
                    self.ids = (a['id'], b['id'])
                    blocks = [call('link_dependency', 'update_task', task_id=b['id'], addBlockedBy=[a['id']])]
                elif step == 3:
                    a, b = self.ids
                    case.assertEqual([a], received['link_dependency']['data']['task']['blockedBy'])
                    blocks = [call('claim_order', 'claim_task', task_id=a), call('too_early_tracking', 'claim_task', task_id=b)]
                elif step == 4:
                    case.assertEqual('in_progress', received['claim_order']['data']['task']['status'])
                    case.assertEqual('customer_service_main', received['claim_order']['data']['task']['owner'])
                    case.assertEqual('TASK_BLOCKED', received['too_early_tracking']['error']['code'])
                    blocks = [call('business_order', 'get_order', order_id='A1001'),
                              call('finish_order', 'complete_task', task_id=self.ids[0])]
                elif step == 5:
                    case.assertTrue(received['business_order']['ok'])
                    case.assertEqual([self.ids[1]], received['finish_order']['data']['newly_unblocked'])
                    blocks = [call('claim_tracking', 'claim_task', task_id=self.ids[1]),
                              call('business_tracking', 'get_tracking', order_id='A1001')]
                elif step == 6:
                    case.assertEqual('in_progress', received['claim_tracking']['data']['task']['status'])
                    case.assertTrue(received['business_tracking']['ok'])
                    blocks = [call('finish_tracking', 'complete_task', task_id=self.ids[1]), call('final_list', 'list_tasks')]
                elif step == 7:
                    tasks = received['final_list']['data']['tasks']
                    case.assertEqual(set(self.ids), {task['id'] for task in tasks})
                    case.assertTrue(all(task['status'] == 'completed' for task in tasks))
                    blocks = [text('两项内部只读调查已完成。')]
                else:
                    raise AssertionError('Unexpected extra model request')
                return {'content': deepcopy(blocks), 'stop_reason': 'end_turn' if step == 7 else 'tool_use',
                        'usage': {'input_tokens': 10, 'output_tokens': 2}}

        model = Investigation()
        h, context, _ = self.build(model=model)
        with redirect_stdout(io.StringIO()):
            self.assertEqual('两项内部只读调查已完成。', h.turn('核对我的A1001订单和物流，记录调查依赖与进度。'))
        self.assertEqual(7, len(model.calls))
        self.assertIn({'role': 'assistant', 'content': model.first}, h.messages)
        expected_ids = {'create_order', 'create_tracking', 'link_dependency', 'claim_order', 'too_early_tracking',
                        'business_order', 'finish_order', 'claim_tracking', 'business_tracking', 'finish_tracking', 'final_list'}
        self.assertEqual(expected_ids, set(result_map(h.messages)))
        audit = self.rows(context)
        for identifier in expected_ids:
            before = [r for r in audit if r['event'] == 'PreToolUse' and r['tool_use_id'] == identifier]
            after = [r for r in audit if r['event'] == 'PostToolUse' and r['tool_use_id'] == identifier]
            self.assertEqual(1, len(before)); self.assertEqual(1, len(after))
        blocked = next(r for r in audit if r.get('tool_use_id') == 'too_early_tracking' and r['event'] == 'PostToolUse')
        self.assertEqual('TASK_BLOCKED', blocked['result_code'])
        self.assertNotEqual('success', blocked['status'])
        self.assertEqual('completed', audit[-1]['status'])
        self.assertNotIn('读取本人订单金额与来源', json.dumps(audit, ensure_ascii=False))
        validate_tool_pairs(h.messages)
        snapshot = json.loads((context.session_dir / 'history.json').read_text(encoding='utf-8'))
        self.assertEqual(h.messages, snapshot['messages'])

    def test_enabled_guest_and_authenticated_missing_store_fail_before_any_task_access(self):
        cases = [(None, 'AUTHENTICATION_REQUIRED'), ('alice', 'TASK_STORE_UNAVAILABLE')]
        for customer, code in cases:
            with self.subTest(customer=customer):
                batch = [call('guard_create', 'create_task', subject='不得写入'), call('guard_list', 'list_tasks'),
                         call('guard_get', 'get_task', task_id='task_00000000'),
                         call('guard_claim', 'claim_task', task_id='task_00000000'),
                         call('guard_complete', 'complete_task', task_id='task_00000000'),
                         call('guard_update', 'update_task', task_id='task_00000000', addBlockedBy=[])]
                h, context, model = self.build([batch, [text('当前无法访问任务。')]], customer=customer, store=None)
                self.assertEqual(TASK_NAMES, TASK_NAMES.intersection(t['name'] for t in h.tools))
                with redirect_stdout(io.StringIO()):
                    h.turn('列出并维护调查任务。')
                self.assertEqual(2, len(model.calls))
                self.assertTrue(all(r['error']['code'] == code for r in result_map(h.messages).values()))
                self.assertFalse(self.base.exists())
                audit = self.rows(context)
                self.assertEqual(6, sum(r['event'] == 'PreToolUse' for r in audit))
                self.assertEqual(6, sum(r['event'] == 'PostToolUse' for r in audit))
                self.assertEqual('completed', audit[-1]['status'])
                validate_tool_pairs(h.messages)

    def test_mismatched_store_is_rejected_at_construction_and_runtime_without_calling_it(self):
        foreign = CustomerTaskStore(self.base, 'bob')
        self.create(foreign, 'BOB_PRIVATE_TASK_817')
        context = ContextManager(self.root / 'runtime', 'alice')
        with patch.object(foreign, 'call', side_effect=AssertionError('Foreign store was consulted')) as spy:
            with self.assertRaises(ValueError):
                Harness(Shop('alice', self.root / 'data/shop.json'), context, MainModel([]),
                        task_store=foreign, enable_tasks=True)
            spy.assert_not_called()
        h, _, model = self.build([[call('runtime_wrong_scope', 'list_tasks')], [text('拒绝错配存储。')]])
        h.task_store = foreign
        with patch.object(foreign, 'call', side_effect=AssertionError('Runtime foreign store was consulted')) as spy:
            with redirect_stdout(io.StringIO()):
                h.turn('列出我的任务。')
            spy.assert_not_called()
        result = result_map(h.messages)['runtime_wrong_scope']
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', result['error']['code'])
        self.assertNotIn('BOB_PRIVATE_TASK_817', json.dumps(model.calls, ensure_ascii=False))
        validate_tool_pairs(h.messages)

    def test_task_schemas_and_router_reject_identity_owner_overrides_and_foreign_task_ids(self):
        foreign = CustomerTaskStore(self.base, 'bob')
        secret = self.create(foreign, 'BOB_TASK_PRIVATE_194')
        h, _, _ = self.build()
        own = self.create(h.task_store)
        args = {'create_task': {'subject': '不得创建的伪造任务'}, 'update_task': {'task_id': own['id'], 'addBlockedBy': []},
                'list_tasks': {}, 'get_task': {'task_id': own['id']}, 'claim_task': {'task_id': own['id']},
                'complete_task': {'task_id': own['id']}}
        batch = [call('foreign_task', 'get_task', task_id=secret['id'])]
        for name, arguments in args.items():
            schema = next(t for t in h.tools if t['name'] == name)['input_schema']
            self.assertFalse(schema['additionalProperties'])
            self.assertFalse({'customer_id', 'owner', 'path'}.intersection(schema['properties']))
            batch.extend([call(name + '_identity', name, **arguments, customer_id='bob'),
                          call(name + '_owner', name, **arguments, owner='other_agent')])
        h.model = MainModel([batch, [text('伪造参数与外部任务均被拒绝。')]])
        with redirect_stdout(io.StringIO()):
            h.turn('检查任务访问参数。')
        received = result_map(h.messages)
        self.assertEqual('TASK_NOT_FOUND', received['foreign_task']['error']['code'])
        for identifier, result in received.items():
            if identifier != 'foreign_task':
                self.assertEqual('INVALID_ARGUMENTS', result['error']['code'])
        self.assertEqual('pending', h.task_store.call('get_task', {'task_id': own['id']})['data']['task']['status'])
        self.assertEqual(1, len(h.task_store.call('list_tasks', {})['data']['tasks']))
        self.assertNotIn('BOB_TASK_PRIVATE_194', json.dumps(h.model.calls, ensure_ascii=False))
        validate_tool_pairs(h.messages)

    def test_hook_failures_close_entire_task_batch_and_do_not_repeat_or_undo_known_writes(self):
        for phase in ('PreToolUse', 'PostToolUse'):
            with self.subTest(phase=phase):
                store = CustomerTaskStore(self.root / phase, 'alice')
                batch = [call('hook_task_first', 'create_task', subject='首个调查步骤'),
                         call('hook_task_second', 'create_task', subject='后续不得执行')]
                h, context, model = self.build([batch], store=store)
                def fail(payload):
                    raise RuntimeError('PRIVATE_TASK_HOOK_EXCEPTION_438')
                h.hooks.register(phase, fail)
                with redirect_stdout(io.StringIO()):
                    answer = h.turn('建立两项调查。')
                actual = result_map(h.messages)
                self.assertEqual({'hook_task_first', 'hook_task_second'}, set(actual))
                self.assertEqual(1, len(model.calls))
                self.assertIn('未完成', answer)
                records = store.call('list_tasks', {})['data']['tasks']
                self.assertEqual(0 if phase == 'PreToolUse' else 1, len(records))
                if phase == 'PreToolUse':
                    self.assertEqual('HOOK_FAILED', actual['hook_task_first']['error']['code'])
                else:
                    self.assertTrue(actual['hook_task_first']['ok'])
                    self.assertEqual(actual['hook_task_first']['data']['task']['id'], records[0]['id'])
                self.assertEqual('HOOK_FAILED', actual['hook_task_second']['error']['code'])
                audit = self.rows(context)
                self.assertEqual(2, sum(r['event'] == 'PreToolUse' for r in audit))
                self.assertEqual(2, sum(r['event'] == 'PostToolUse' for r in audit))
                self.assertNotIn('PRIVATE_TASK_HOOK_EXCEPTION_438', json.dumps(self.rows(context, 'events.jsonl')))
                validate_tool_pairs(h.messages)

    def test_logout_new_session_and_bob_switch_keep_tasks_persistent_but_histories_and_todos_fresh(self):
        session = LiveSession(self.root, MainModel([]), store='json', allow_tasks=True)
        self.assertIsNone(session.harness.task_store)
        auth = FakeAuth()
        self.assertTrue(session.login('alice', 'offline-password', auth))
        old = session.harness
        task = self.create(old.task_store)
        claimed = old.call_tool('claim_task', {'task_id': task['id']})
        self.assertTrue(claimed['ok'])
        self.assertEqual('customer_service_main', claimed['data']['task']['owner'])
        note = '物流查询失败，缺物流依据，稍后重试'
        with self.subTest(progress_note_preserves_in_progress=True):
            self.assertEqual('', task['note'])
            updated = old.call_tool('update_task', {'task_id': task['id'], 'note': note})
            self.assertTrue(updated['ok'], updated)
            self.assertEqual('in_progress', updated['data']['task']['status'])
            self.assertEqual('customer_service_main', updated['data']['task']['owner'])
            self.assertEqual(note, updated['data']['task']['note'])
        old.messages = [{'role': 'user', 'content': 'ALICE_OLD_HISTORY_738'}]
        old.todo.update([{'content': 'OLD_TODO_738', 'status': 'in_progress'}])
        old_id = old.context.session_id
        self.assertEqual((self.base / 'alice').resolve(), old.task_store.directory.resolve())
        session.logout()
        self.assertIsNone(session.harness.task_store)
        self.assertEqual([], session.harness.messages)
        self.assertEqual([], session.harness.todo.items)
        self.assertEqual('AUTHENTICATION_REQUIRED', session.harness.call_tool('list_tasks', {})['error']['code'])
        self.assertTrue(session.login('bob', 'offline-password', auth))
        self.assertEqual([], session.harness.call_tool('list_tasks', {})['data']['tasks'])
        restarted = LiveSession(self.root, MainModel([]), store='json', allow_tasks=True)
        self.assertTrue(restarted.login('alice', 'offline-password', auth))
        self.assertNotEqual(old_id, restarted.context.session_id)
        self.assertEqual([], restarted.harness.messages)
        self.assertEqual([], restarted.harness.todo.items)
        self.assertEqual({}, restarted.context.archives)
        self.assertEqual({}, restarted.context.tool_outputs)
        loaded = restarted.harness.call_tool('list_tasks', {})['data']['tasks']
        self.assertEqual([task['id']], [r['id'] for r in loaded])
        self.assertEqual('in_progress', loaded[0]['status'])
        self.assertEqual('customer_service_main', loaded[0]['owner'])
        with self.subTest(progress_note_survives_fresh_session=True):
            resumed = restarted.harness.call_tool('get_task', {'task_id': task['id']})
            self.assertTrue(resumed['ok'], resumed)
            self.assertEqual(note, resumed['data']['task']['note'])
            self.assertEqual(note, loaded[0]['note'])
            self.assertEqual('in_progress', resumed['data']['task']['status'])
            self.assertEqual('customer_service_main', resumed['data']['task']['owner'])
        self.assertEqual([], restarted.model.calls)

    def test_read_only_child_does_not_receive_task_store_or_the_six_persistent_tools(self):
        child_batch = [call('child_' + name, name, **({'subject': '不得创建'} if name == 'create_task'
                       else {} if name == 'list_tasks' else {'task_id': 'task_00000000', 'addBlockedBy': []}
                       if name == 'update_task' else {'task_id': 'task_00000000'})) for name in sorted(TASK_NAMES)]
        child = MainModel([child_batch, [text('未执行任何持久任务变更。')]])
        h, _, parent = self.build([[call('delegate_readonly', 'task', prompt='仅做只读调查，不管理持久任务。')], [text('父答复完成。')]], child=child)
        seen = []
        original = Harness.__init__
        def capture_child(instance, *args, **kwargs):
            original(instance, *args, **kwargs)
            seen.append(instance)
        with patch.object(Harness, '__init__', capture_child), redirect_stdout(io.StringIO()):
            h.turn('委派一次只读调查。')
        self.assertEqual(1, len(seen))
        self.assertIsNone(seen[0].task_store)
        self.assertFalse(TASK_NAMES.intersection(t['name'] for t in child.calls[0]['tools']))
        self.assertEqual(8, len(child.calls[0]['tools']))
        self.assertEqual(2, len(child.calls))
        self.assertEqual(2, len(parent.calls))
        self.assertTrue(all(r['error']['code'] == 'SUBAGENT_TOOL_NOT_ALLOWED' for r in result_map(seen[0].messages).values()))
        self.assertEqual([], h.task_store.call('list_tasks', {})['data']['tasks'])
        validate_tool_pairs(seen[0].messages)
        validate_tool_pairs(h.messages)

    def test_s08_history_snip_preserves_external_task_state_current_request_and_tool_pairs(self):
        h, context, model = self.build([[call('post_snip_list', 'list_tasks')], [text('任务仍在本人持久存储中。')]])
        task = self.create(h.task_store)
        h.messages = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': '旧对话第' + str(i) + '轮'} for i in range(60)]
        query = '列出我已经创建的调查任务。'
        with redirect_stdout(io.StringIO()):
            h.turn(query)
        self.assertTrue(context.compactions)
        self.assertTrue(context.archives)
        self.assertEqual(query, model.calls[0]['messages'][-1]['content'])
        received = result_map(model.calls[1]['messages'])['post_snip_list']
        self.assertEqual([task['id']], [r['id'] for r in received['data']['tasks']])
        self.assertEqual('pending', received['data']['tasks'][0]['status'])
        self.assertEqual(2, len(model.calls))
        self.assertEqual('pending', h.task_store.call('get_task', {'task_id': task['id']})['data']['task']['status'])
        validate_tool_pairs(h.messages)
        for request in model.calls:
            validate_tool_pairs(request['messages'])

    def test_cli_tasks_flag_is_live_only_rejected_before_key_and_opt_in_adds_no_sdk_calls(self):
        invalid = [['--allow-tasks'], ['--demo', '--allow-tasks'], ['--lesson', 's08', '--allow-tasks'],
                   ['--live-context', '--dry-run', '--allow-tasks'], ['--init-db', '--allow-tasks'],
                   ['--live', '--allow-tasks', '--customer', 'alice']]
        for args in invalid:
            with self.subTest(args=args), patch.object(app, 'ROOT', self.root), \
                 patch('sys.argv', ['app.py', *args]), \
                 patch.object(app, 'read_secret', side_effect=AssertionError('Read key before invalid flag rejection')) as secret, \
                 patch.object(app, 'AnthropicModel', side_effect=AssertionError('Created provider before invalid flag rejection')) as sdk, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    app.main()
                self.assertEqual(2, error.exception.code)
                secret.assert_not_called(); sdk.assert_not_called()
        for enabled in (False, True):
            with self.subTest(legal_live_enabled=enabled):
                fake = MainModel([])
                sessions = []
                def factory(*args, **kwargs):
                    value = LiveSession(*args, **kwargs)
                    sessions.append(value)
                    return value
                argv = ['app.py', '--live', '--store', 'json'] + (['--allow-tasks'] if enabled else [])
                with patch.object(app, 'ROOT', self.root), patch('sys.argv', argv), \
                     patch.object(app, 'AnthropicModel', return_value=fake), \
                     patch.object(app, 'LiveSession', side_effect=factory), patch.object(app, 'AuthStore'), \
                     patch.object(app, 'MemorySelector', side_effect=AssertionError('Unexpected auxiliary client')), \
                     patch.object(app, 'MemoryExtractor', side_effect=AssertionError('Unexpected auxiliary client')), \
                     patch.object(app, 'SummaryClient', side_effect=AssertionError('Unexpected summary client')), \
                     patch.object(app, 'read_secret', side_effect=AssertionError('Fake key should be used')), \
                     patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-no-provider-key'}), \
                     patch('builtins.input', return_value='q'), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    app.main()
                self.assertEqual(1, len(sessions))
                self.assertIsNone(sessions[0].customer_id)
                self.assertIsNone(sessions[0].harness.task_store)
                actual = TASK_NAMES.intersection(t['name'] for t in sessions[0].harness.tools)
                self.assertEqual(TASK_NAMES if enabled else set(), actual)
                self.assertEqual([], fake.calls)

        with self.subTest(logout_task_store_rebuild_failure=True):
            fake = MainModel([])
            sessions = []
            def failing_factory(*args, **kwargs):
                session = LiveSession(*args, **kwargs)
                self.assertTrue(session.login('alice', 'offline-password', FakeAuth()))
                def fail_rebuild(customer_id):
                    raise TaskStoreError('TASK_STORE_UNAVAILABLE', 'PRIVATE_TASK_LOGOUT_FAILURE_582')
                session._new_harness = fail_rebuild
                sessions.append(session)
                return session
            with patch.object(app, 'ROOT', self.root), \
                 patch('sys.argv', ['app.py', '--live', '--store', 'json', '--allow-tasks']), \
                 patch.object(app, 'AnthropicModel', return_value=fake), patch.object(app, 'AuthStore'), \
                 patch.object(app, 'LiveSession', side_effect=failing_factory), \
                 patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-no-provider-key'}), \
                 patch.object(app, 'read_secret', side_effect=AssertionError('Unexpected credential prompt')), \
                 patch('builtins.input', side_effect=['/logout', 'MUST_NOT_BE_SENT_TO_MODEL', 'q']) as user_input, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                app.main()
            self.assertEqual(1, user_input.call_count)
            self.assertEqual([], fake.calls)
            self.assertEqual(1, len(sessions))
            self.assertIsNone(sessions[0].harness)
            self.assertIsNone(sessions[0].customer_id)

    def test_completed_investigation_task_does_not_grant_access_to_another_customers_order(self):
        h, _, model = self.build([[call('foreign_business_order', 'get_order', order_id='B2001')], [text('仍不能读取他人订单。')]])
        task = self.create(h.task_store, '仅记录内部调查，不授予任何订单权限')
        self.assertTrue(h.call_tool('claim_task', {'task_id': task['id']})['ok'])
        self.assertTrue(h.call_tool('complete_task', {'task_id': task['id']})['ok'])
        with redirect_stdout(io.StringIO()):
            h.turn('任务显示完成，请读取B2001。')
        received = result_map(h.messages)['foreign_business_order']
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', received['error']['code'])
        self.assertNotIn('data', received)
        self.assertNotIn('DEMO-B2001', json.dumps(model.calls, ensure_ascii=False))
        self.assertEqual('completed', h.task_store.call('get_task', {'task_id': task['id']})['data']['task']['status'])
        validate_tool_pairs(h.messages)


if __name__ == '__main__':
    unittest.main(verbosity=2)
