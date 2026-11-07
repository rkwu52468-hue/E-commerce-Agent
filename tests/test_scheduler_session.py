"""s12 behavior acceptance with simulated clock, real readonly shop, no network."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock
import app
from customer_service.live_session import LiveSession
from customer_service.context import validate_tool_pairs
from customer_service.models import AnthropicModel
from test_subagent import MainModel, call, text

ROOT = Path(__file__).resolve().parents[1]
TZ = timezone(timedelta(hours=8))

class FakeAuth:
    def verify(self, customer_id, password):
        return customer_id if customer_id in ('alice', 'bob') and password == 'local-secret' else None

class ScheduleSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='s12_session_', dir=ROOT/'runtime')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'data').mkdir()
        shutil.copyfile(ROOT/'data/shop.json', self.root/'data/shop.json')
        self.now = datetime(2026, 11, 6, 8, 59, tzinfo=TZ)
        self.auth = FakeAuth()

    def session(self, replies=(), enabled=True):
        model = MainModel(replies)
        session = LiveSession(self.root, model, store='json', allow_schedules=enabled,
                              schedule_clock=lambda: self.now)
        self.addCleanup(session.close)
        return session, model

    def login(self, s, customer='alice'):
        self.assertTrue(s.login(customer, 'local-secret', self.auth))

    def schedule(self, s):
        result = s.harness.call_tool('schedule_tracking', {'order_id': 'A1001', 'cron': '0 9 * * *'})
        self.assertTrue(result['ok'], result)
        return result['data']['schedule']['id']

    def records(self, s):
        return s.scheduler.manager.call('alice', 'list_schedules', {})['data']['schedules']

    def histories(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in (self.root/'runtime').glob('*/*/history.json')
                if json.loads(p.read_text(encoding='utf-8')).get('kind') == 'scheduled_tracking']

    def test_default_off_guest_denied_and_no_business_read(self):
        s, model = self.session(enabled=False)
        self.assertNotIn('schedule_tracking', {x['name'] for x in s.harness.tools})
        self.assertFalse((self.root/'data/schedules').exists())
        self.assertEqual('SCHEDULES_DISABLED', s.harness.call_tool('schedule_tracking', {})['error']['code'])
        s.close()
        s, model = self.session()
        with patch.object(s.shop, 'call', side_effect=AssertionError('Guest must not query')):
            self.assertEqual('AUTHENTICATION_REQUIRED', s.harness.call_tool('schedule_tracking',
                {'order_id': 'A1001', 'cron': '0 9 * * *'})['error']['code'])
        self.assertEqual([], model.calls)

    def test_creation_checks_ownership_and_rejects_model_identity_override(self):
        s, model = self.session(); self.login(s)
        for args, code in [({'order_id':'B2001','cron':'0 9 * * *'}, 'ACCESS_DENIED_OR_NOT_FOUND'),
                           ({'order_id':'A1001','cron':'0 9 * * *','owner':'bob'}, 'INVALID_ARGUMENTS')]:
            self.assertEqual(code, s.harness.call_tool('schedule_tracking', args)['error']['code'])
        self.assertEqual([], self.records(s)); self.assertEqual([], model.calls)

    def test_natural_request_tool_receipt_reaches_next_model_request(self):
        s, model = self.session([[call('save', 'schedule_tracking', order_id='A1001', cron='0 9 * * *')],
                                 [text('已保存每天9点的只读安排。')]])
        self.login(s)
        query = '以后每天早上9点帮我查一下A1001物流。'
        with redirect_stdout(io.StringIO()): s.turn(query)
        self.assertEqual(query, model.calls[0]['messages'][0]['content'])
        body = model.calls[1]['messages'][-1]['content'][0]
        self.assertTrue(json.loads(body['content'])['ok']); validate_tool_pairs(s.harness.messages)
        self.assertEqual(1, len(self.records(s)))

    def test_due_queries_actual_shop_then_one_tool_free_model_and_separate_history(self):
        s, model = self.session([[text('最近物流记录显示延迟；暂无可承诺的到达时间。')]])
        self.login(s); self.schedule(s)
        main = s.harness; self.now = self.now.replace(hour=9, minute=0)
        with patch.object(s.shop, 'call', wraps=s.shop.call) as query:
            s.poll_schedules(self.now); s.poll_schedules(self.now)
        self.assertEqual(1, query.call_count); self.assertEqual('get_tracking', query.call_args.args[0])
        self.assertEqual(1, len(model.calls)); self.assertEqual([], model.calls[0]['tools'])
        payload = model.calls[0]['messages'][0]['content']
        self.assertIn('DEMO-A1001', payload); self.assertIn('shop.json#orders.A1001.tracking', payload)
        self.assertEqual([], main.messages); self.assertFalse(self.records(s)[0]['pending_delivery'])
        h = self.histories()[0]; self.assertEqual('completed', h['status']); self.assertTrue(h['acknowledged'])
        self.assertEqual(1, len(s.scheduled_notices())); self.assertEqual([], s.scheduled_notices())

    def test_guest_due_and_same_minute_login_skip_then_next_day_can_execute(self):
        s, model = self.session([[text('次日查询结果。')]])
        self.login(s); self.schedule(s); s.logout()
        self.now = self.now.replace(hour=9, minute=0)
        # login itself must finish this occurrence as guest, even without a prior poll.
        self.login(s); s.poll_schedules(self.now)
        self.assertEqual([], model.calls); self.assertEqual(1, len(self.records(s)))
        self.assertFalse(self.records(s)[0]['pending_delivery'])
        self.now += timedelta(days=1); s.poll_schedules(self.now)
        self.assertEqual(1, len(model.calls))
        audit = (self.root/'runtime/scheduler-audit.jsonl').read_text(encoding='utf-8')
        self.assertIn('SKIPPED', audit.upper()); self.assertNotIn('local-secret', audit)

    def test_bob_due_cannot_query_or_receive_alice_records_or_notices(self):
        s, model = self.session(); self.login(s); identifier = self.schedule(s)
        self.login(s, 'bob'); self.now = self.now.replace(hour=9, minute=0)
        with patch.object(s.shop, 'call', side_effect=AssertionError('Wrong identity query')):
            s.poll_schedules(self.now)
        own = s.harness.call_tool('list_schedules', {})
        self.assertEqual([], own['data']['schedules'])
        self.assertFalse(s.harness.call_tool('cancel_schedule', {'schedule_id': identifier})['ok'])
        self.assertEqual([], s.scheduled_notices()); self.assertEqual([], model.calls)
        self.login(s); s.poll_schedules(self.now); self.assertEqual([], model.calls)

    def test_model_failure_keeps_facts_does_not_repeat_query_same_slot(self):
        s, model = self.session([TimeoutError('private provider error')]); self.login(s); self.schedule(s)
        self.now = self.now.replace(hour=9, minute=0)
        with patch.object(s.shop, 'call', wraps=s.shop.call) as query:
            s.poll_schedules(self.now); s.poll_schedules(self.now + timedelta(seconds=5))
        self.assertEqual(1, query.call_count); self.assertEqual(1, len(model.calls))
        report = self.histories()[0]
        self.assertEqual('failed', report['status']); self.assertIn('business_result', report)
        self.assertTrue(self.records(s)[0]['pending_delivery'])
        self.assertNotIn('private provider error', json.dumps(report))

    def test_process_cap_counts_failed_attempts_and_stops_fourth_query(self):
        s, model = self.session([TimeoutError('network') for _ in range(3)])
        self.login(s)
        for _ in range(4): self.schedule(s)
        self.now = self.now.replace(hour=9, minute=0)
        with patch.object(s.shop, 'call', wraps=s.shop.call) as query:
            s.poll_schedules(self.now); s.poll_schedules(self.now)
        self.assertEqual(3, query.call_count); self.assertEqual(3, len(model.calls))
        self.assertEqual(3, s.scheduler._attempts)

    def test_pending_queue_switch_identity_and_old_notices_cleared(self):
        s, model = self.session([[text('Alice只读结果。')]])
        self.login(s); self.schedule(s); self.now = self.now.replace(hour=9, minute=0)
        # Queuing is separate from dispatch and does not grant lasting authorization.
        self.assertTrue(s.scheduler.poll_only()['data']['deliveries'])
        self.login(s, 'bob'); s.poll_schedules(self.now)
        self.assertEqual([], model.calls)
        self.assertFalse(self.records(s)[0]['pending_delivery'])
        self.login(s); self.now += timedelta(days=1); s.poll_schedules(self.now)
        self.assertTrue(s.scheduler._notices)
        s.logout(); self.assertEqual([], s.scheduled_notices())

    def test_credentials_lock_defers_dispatch_and_closed_session_stops(self):
        s, model = self.session([[text('已查询。')]])
        self.login(s); self.schedule(s); self.now = self.now.replace(hour=9, minute=0)
        with s.local_credentials():
            worker = threading.Thread(target=lambda:s.poll_schedules(self.now))
            worker.start(); worker.join(2)
            self.assertFalse(worker.is_alive()); self.assertEqual([], model.calls)
        s.poll_schedules(self.now); self.assertEqual(1, len(model.calls))
        s.close(); self.now += timedelta(days=1); s.poll_schedules(self.now)
        self.assertEqual(1, len(model.calls))

    def test_scheduled_response_tool_request_cannot_execute_more_tools(self):
        s, model = self.session([[call('danger', 'schedule_tracking', order_id='B2001', cron='0 9 * * *')]])
        self.login(s); self.schedule(s); self.now = self.now.replace(hour=9, minute=0)
        s.poll_schedules(self.now)
        self.assertEqual(1, len(model.calls)); self.assertEqual(1, len(self.records(s)))
        history = self.histories()[0]
        self.assertEqual('failed', history['status'])
        self.assertEqual('tool_use', history['response']['stop_reason'])
        validate_tool_pairs(history['messages'])

    def test_provider_scheduled_adapter_caps_output_disables_thinking_and_exposes_no_tools(self):
        model = AnthropicModel.__new__(AnthropicModel)
        model.client = Mock(); model.model = 'fake'
        response = Mock(); response.content = []; response.stop_reason = 'end_turn'
        response.usage.model_dump.return_value = {'output_tokens': 0}
        model.client.messages.create.return_value = response
        model.complete_scheduled([{'role':'user','content':'facts'}], [], 'rules')
        kwargs = model.client.messages.create.call_args.kwargs
        self.assertEqual(1024, kwargs['max_tokens']); self.assertNotIn('tools', kwargs)
        self.assertEqual({'type':'disabled'}, kwargs['extra_body']['thinking'])
        with self.assertRaises(ValueError): model.complete_scheduled([], [{'name':'get_order'}], '')

    def test_cli_requires_live_before_reading_key(self):
        with patch('sys.argv', ['app.py','--allow-schedules']), patch.object(app,'read_secret', side_effect=AssertionError), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error: app.main()
        self.assertEqual(2, error.exception.code)


    def test_initial_evidence_failure_never_claims_facts_saved_or_calls_model(self):
        s, model = self.session(); self.login(s); self.schedule(s)
        self.now = self.now.replace(hour=9, minute=0)
        with patch('customer_service.schedule_runtime._atomic_report', side_effect=OSError('disk')):
            with patch.object(s.shop, 'call', side_effect=AssertionError('No facts before trace')):
                s.poll_schedules(self.now)
        self.assertEqual([], model.calls)
        notices = ''.join(s.scheduled_notices())
        self.assertIn('尚未确认', notices); self.assertNotIn('事实已保存在', notices)

    def test_input_sending_cap_stops_before_sdk_and_preserves_raw_facts(self):
        s, model = self.session(); self.login(s); self.schedule(s)
        self.now = self.now.replace(hour=9, minute=0)
        facts = {'ok':True, 'data':{'large':'x'*17000}, 'source':'synthetic-large'}
        with patch.object(s.shop, 'call', return_value=facts):s.poll_schedules(self.now)
        self.assertEqual([], model.calls)
        history = self.histories()[0]
        self.assertEqual(0, history['model_attempts']); self.assertEqual(facts, history['business_result'])
        self.assertEqual('failed', history['status'])


    def test_audit_failure_stops_with_safe_notice_and_no_query(self):
        s, model = self.session(); self.login(s); self.schedule(s)
        self.now = self.now.replace(hour=9, minute=0)
        with patch.object(s.scheduler, '_audit', side_effect=OSError('disk unavailable')):
            with patch.object(s.shop, 'call', side_effect=AssertionError('No query after audit failure')):
                s.poll_schedules(self.now)
        self.assertTrue(s.scheduler._disabled); self.assertEqual([], model.calls)
        self.assertIn('调度已', ''.join(s.scheduled_notices()))
        status = s.harness.call_tool('list_schedules', {})['data']['runtime']
        self.assertFalse(status['automatic_execution_enabled'])


    def test_cli_flag_starts_and_finally_stops_scheduler_without_any_paid_request(self):
        shutil.copyfile(ROOT/'config.json', self.root/'config.json')
        model = MainModel([]); sessions = []
        def factory(*args, **kwargs):
            session = LiveSession(*args, **kwargs); sessions.append(session); return session
        with patch('sys.argv', ['app.py','--live','--store','json','--allow-schedules']), \
             patch.object(app, 'ROOT', self.root), patch.object(app, 'read_secret', return_value='offline-fake-key'), \
             patch.dict('os.environ', {'DEEPSEEK_API_KEY':''}), patch.object(app, 'AnthropicModel', return_value=model), \
             patch.object(app, 'AuthStore'), patch.object(app, 'LiveSession', side_effect=factory), \
             patch('builtins.input', side_effect=['q']), redirect_stdout(io.StringIO()):
            app.main()
        scheduler = sessions[0].scheduler
        self.assertIsNotNone(scheduler._thread); scheduler._thread.join(2)
        self.assertFalse(scheduler._thread.is_alive()); self.assertTrue(sessions[0]._closed)
        self.assertEqual([], model.calls); self.assertFalse((self.root/'data/auth').exists())

    def test_denied_schedule_attempt_is_in_hook_audit_without_full_arguments(self):
        s, model = self.session([[call('forbidden', 'schedule_tracking', order_id='B2001', cron='0 9 * * *')],
                                 [text('无权为该订单建立安排。')]])
        self.login(s)
        with redirect_stdout(io.StringIO()):s.turn('每天9点查B2001')
        rows = [json.loads(line) for line in (s.context.session_dir/'audit.jsonl').read_text(encoding='utf-8').splitlines()]
        post = next(row for row in rows if row.get('event') == 'PostToolUse')
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', post['result_code'])
        self.assertEqual('B2001', post['resource_id'])
        self.assertNotIn('arguments', post); self.assertNotIn('result', post)
        self.assertEqual([], self.records(s))
