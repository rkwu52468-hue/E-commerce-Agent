"""Stage 3 acceptance tests: FastAPI + SSE streaming, multi-user isolation, 2-step UI gate, rate limiter, and Langfuse hook."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from customer_service.observability import LangfuseReporter, SessionObserver
from customer_service.web_sessions import SlidingWindowRateLimiter

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / 'data/shop-action-seed.json'
HAS_FASTAPI = bool(importlib.util.find_spec('fastapi') and importlib.util.find_spec('httpx'))


def _parse_sse_events(raw_text: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw_text.split('\n\n'):
        block = block.strip()
        if not block or block.startswith(':'):
            continue
        ev_name = 'message'
        data_str = ''
        for line in block.split('\n'):
            if line.startswith('event:'):
                ev_name = line[6:].strip()
            elif line.startswith('data:'):
                data_str += line[5:].strip()
        if data_str:
            events.append((ev_name, json.loads(data_str)))
    return events


class RateLimiterAndObservabilityUnitTests(unittest.TestCase):
    def test_sliding_window_rate_limiter_enforces_limit_and_recovers_after_window(self):
        now = {'t': 100.0}
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0, clock=lambda: now['t'])
        ok1, rem1, _ = limiter.check('client_a')
        ok2, rem2, _ = limiter.check('client_a')
        ok3, rem3, retry3 = limiter.check('client_a')
        ok_b, rem_b, _ = limiter.check('client_b')
        self.assertTrue(ok1 and rem1 == 1)
        self.assertTrue(ok2 and rem2 == 0)
        self.assertFalse(ok3)
        self.assertEqual(0, rem3)
        self.assertGreater(retry3, 0.0)
        self.assertTrue(ok_b and rem_b == 1)
        # Advance past window -> recovered
        now['t'] = 161.0
        ok4, rem4, _ = limiter.check('client_a')
        self.assertTrue(ok4 and rem4 == 1)

    def test_langfuse_reporter_emits_spans_and_isolates_failures(self):
        recorded = {'traces': [], 'spans': [], 'updates': [], 'flushed': 0}

        class FakeTrace:
            def span(self, **kwargs):
                recorded['spans'].append(kwargs)

            def update(self, **kwargs):
                recorded['updates'].append(kwargs)

        class FakeLangfuse:
            def trace(self, **kwargs):
                recorded['traces'].append(kwargs)
                return FakeTrace()

            def flush(self):
                recorded['flushed'] += 1

        reporter = LangfuseReporter(client=FakeLangfuse())
        self.assertTrue(reporter.enabled)
        reporter.on_event('UserPromptSubmit', {'session_id': 's1', 'customer_id': 'alice', 'turn': 1})
        reporter.on_event('PostToolUse', {'session_id': 's1', 'turn': 1, 'tool': 'get_order', 'resource_id': 'A1001', 'ok': True, 'duration_ms': 4.2})
        reporter.on_event('Stop', {'session_id': 's1', 'turn': 1, 'status': 'completed', 'tool_calls': 1, 'steps': 2})
        self.assertEqual(1, len(recorded['traces']))
        self.assertEqual('alice', recorded['traces'][0]['user_id'])
        self.assertEqual(1, len(recorded['spans']))
        self.assertEqual('get_order', recorded['spans'][0]['name'])
        self.assertEqual(1, len(recorded['updates']))
        self.assertEqual(1, recorded['flushed'])

        # Broken client must NEVER raise
        class BrokenLangfuse:
            def trace(self, **kwargs):
                raise RuntimeError('langfuse unreachable')

        broken = LangfuseReporter(client=BrokenLangfuse())
        broken.on_event('UserPromptSubmit', {'session_id': 's2', 'customer_id': 'bob', 'turn': 1})


@unittest.skipUnless(HAS_FASTAPI, 'optional web dependencies (fastapi/httpx) are not installed')
class Stage3FastApiWorkbenchTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from customer_service.web_app import create_app

        (ROOT / 'runtime').mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='stage3_web_test_', dir=ROOT / 'runtime')
        self.addCleanup(self.temporary.cleanup)
        self.tmp = Path(self.temporary.name)
        self.seed_copy = self.tmp / 'shop-action-seed.json'
        shutil.copyfile(SEED_PATH, self.seed_copy)
        self.db_path = self.tmp / 'web-test-actions.db'

        self.lf_events = []

        class SpyTrace:
            def __init__(self, sink):
                self.sink = sink

            def span(self, **kw):
                self.sink.append(('span', kw))

            def update(self, **kw):
                self.sink.append(('update', kw))

        class SpyLangfuse:
            def __init__(self, sink):
                self.sink = sink

            def trace(self, **kw):
                self.sink.append(('trace', kw))
                return SpyTrace(self.sink)

            def flush(self):
                self.sink.append(('flush', {}))

        self.app = create_app(
            root=ROOT,
            seed_path=self.seed_copy,
            database_path=self.db_path,
            allow_demo_switch=True,
            rate_limit_rpm=20,
            langfuse=LangfuseReporter(client=SpyLangfuse(self.lf_events)),
        )
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def test_serves_static_ui_and_health_endpoint(self):
        idx = self.client.get('/')
        self.assertEqual(200, idx.status_code)
        self.assertIn('desk.cs', idx.text)
        css = self.client.get('/static/styles.css')
        self.assertEqual(200, css.status_code)
        self.assertIn('--tan-light', css.text)
        js = self.client.get('/static/app.js')
        self.assertEqual(200, js.status_code)
        self.assertIn('punchBinderHoles', js.text)

        health = self.client.get('/api/health').json()
        self.assertEqual('ok', health['status'])
        self.assertEqual('offline-demo', health['model_mode'])
        self.assertTrue(health['langfuse_enabled'])
        self.assertEqual(20, health['rate_limit_rpm'])
        # Web workbench disables redundant subagent delegation by default for low latency & high cache hit
        sid = self.client.post('/api/sessions').json()['session_id']
        managed = self.app.state.session_store.get(sid)
        self.assertNotIn('task', {t['name'] for t in managed.live_session.harness.tools})

    def test_multi_user_session_isolation_and_cross_customer_block(self):
        s_alice = self.client.post('/api/sessions').json()['session_id']
        s_bob = self.client.post('/api/sessions').json()['session_id']
        self.assertNotEqual(s_alice, s_bob)

        r_a = self.client.post(f'/api/sessions/{s_alice}/login', json={'customer_id': 'alice', 'demo_switch': True})
        r_b = self.client.post(f'/api/sessions/{s_bob}/login', json={'customer_id': 'bob', 'demo_switch': True})
        self.assertEqual(200, r_a.status_code)
        self.assertEqual(200, r_b.status_code)
        self.assertEqual(6, len(r_a.json()['state']['orders']))
        self.assertEqual(4, len(r_b.json()['state']['orders']))
        self.assertTrue(all(o['order_id'].startswith('A') for o in r_a.json()['state']['orders']))
        self.assertTrue(all(o['order_id'].startswith('B') for o in r_b.json()['state']['orders']))

        # Alice attempting to query Bob's order B2001 gets denied without leaking Bob's data
        chat = self.client.post(f'/api/sessions/{s_alice}/chat', json={'message': '帮我查一下订单 B2001 的金额。'})
        self.assertEqual(200, chat.status_code)
        reply = chat.json()['reply']
        self.assertIn('ACCESS_DENIED_OR_NOT_FOUND', reply)
        self.assertNotIn('338', reply)

    def test_sse_stream_two_step_confirmation_gate_auto_refund_ticket_and_reset(self):
        sid = self.client.post('/api/sessions').json()['session_id']
        self.client.post(f'/api/sessions/{sid}/login', json={'customer_id': 'alice', 'demo_switch': True})

        # 1) Stream cancel_order step 1 over SSE
        resp = self.client.post(f'/api/sessions/{sid}/chat/stream',
                                json={'message': '帮我取消还没发货的 A1003，买错容量。'})
        self.assertEqual(200, resp.status_code)
        self.assertIn('text/event-stream', resp.headers['content-type'])
        events = _parse_sse_events(resp.text)
        names = [e[0] for e in events]
        self.assertIn('start', names)
        self.assertIn('tool_start', names)
        self.assertIn('tool_end', names)
        self.assertIn('delta', names)
        self.assertIn('state', names)
        self.assertIn('done', names)

        state_ev = [p for n, p in events if n == 'state'][-1]
        self.assertEqual(1, len(state_ev['pending_confirmations']))
        cid = state_ev['pending_confirmations'][0]['confirmation_id']
        a1003_before = next(o for o in state_ev['orders'] if o['order_id'] == 'A1003')
        self.assertEqual('paid_unshipped', a1003_before['status'])

        # 2) Approve via visual confirmation gate endpoint -> DB updated + auto refund RF-A1003 created!
        approved = self.client.post(f'/api/sessions/{sid}/confirmations/{cid}/approve').json()
        self.assertTrue(approved['result']['ok'])
        self.assertEqual('RF-A1003', approved['result']['data']['refund_created']['refund_id'])
        a1003_after = next(o for o in approved['state']['orders'] if o['order_id'] == 'A1003')
        self.assertEqual('cancelled', a1003_after['status'])
        self.assertTrue(any(r['refund_id'] == 'RF-A1003' for r in approved['state']['refunds']))
        self.assertEqual([], approved['state']['pending_confirmations'])

        # 3) Test rejection path on A1002
        self.client.post(f'/api/sessions/{sid}/chat', json={'message': '帮我取消待支付订单 A1002。'})
        st2 = self.client.get(f'/api/sessions/{sid}/state').json()
        cid2 = st2['pending_confirmations'][0]['confirmation_id']
        rejected = self.client.post(f'/api/sessions/{sid}/confirmations/{cid2}/reject').json()
        self.assertTrue(rejected['result']['ok'])
        a1002_after = next(o for o in rejected['state']['orders'] if o['order_id'] == 'A1002')
        self.assertEqual('pending_payment', a1002_after['status'])

        # 4) Escalation ticket creation updates live ledger & Langfuse captured trace/spans
        tk_resp = self.client.post(f'/api/sessions/{sid}/chat', json={'message': 'A1001 物流严重延误，请转人工专员建工单处理。'}).json()
        self.assertTrue(any(t['escalate_to_human'] and t['assigned_queue'] == 'human_specialist'
                            for t in tk_resp['state']['tickets']))
        self.assertTrue(any(kind == 'trace' for kind, _ in self.lf_events))
        self.assertTrue(any(kind == 'span' for kind, _ in self.lf_events))

        # 5) Reset demo DB cleanly restores A1003 to paid_unshipped
        reset = self.client.post(f'/api/sessions/{sid}/reset-demo-db').json()
        self.assertTrue(reset['ok'])
        a1003_reset = next(o for o in reset['state']['orders'] if o['order_id'] == 'A1003')
        self.assertEqual('paid_unshipped', a1003_reset['status'])

    def test_chat_endpoint_enforces_rate_limit_429(self):
        from fastapi.testclient import TestClient
        from customer_service.web_app import create_app

        tight_db = self.tmp / 'web-ratelimit.db'
        tight_app = create_app(
            root=ROOT,
            seed_path=self.seed_copy,
            database_path=tight_db,
            rate_limit_rpm=2,
        )
        with TestClient(tight_app) as client:
            sid = client.post('/api/sessions').json()['session_id']
            r1 = client.post(f'/api/sessions/{sid}/chat', json={'message': 'P100 多少钱？'})
            r2 = client.post(f'/api/sessions/{sid}/chat', json={'message': 'P101 多少钱？'})
            r3 = client.post(f'/api/sessions/{sid}/chat', json={'message': 'P102 多少钱？'})
            self.assertEqual(200, r1.status_code)
            self.assertEqual(200, r2.status_code)
            self.assertEqual(429, r3.status_code)
            self.assertIn('Retry-After', r3.headers)


class Stage4DeploymentConfigTests(unittest.TestCase):
    def test_web_server_env_resolution_supports_auto_live_demo_and_port(self):
        import web_server

        # APP_MODE=auto without key -> demo on container port 7860
        s_demo = web_server.resolve_runtime_settings(
            argv=[],
            environ={'APP_MODE': 'auto', 'HOST': '0.0.0.0', 'PORT': '7860', 'RATE_LIMIT_RPM': '20'},
        )
        self.assertEqual('demo', s_demo.effective_mode)
        self.assertEqual('0.0.0.0', s_demo.host)
        self.assertEqual(7860, s_demo.port)
        self.assertEqual(20, s_demo.rate_limit)

        # APP_MODE=auto with DEEPSEEK_API_KEY -> live
        s_live = web_server.resolve_runtime_settings(
            argv=[],
            environ={'APP_MODE': 'auto', 'DEEPSEEK_API_KEY': 'sk-test-placeholder', 'STRICT_AUTH': 'true'},
        )
        self.assertEqual('live', s_live.effective_mode)
        self.assertTrue(s_live.strict_auth)

        # Explicit CLI flag overrides env
        s_override = web_server.resolve_runtime_settings(
            argv=['--mode', 'demo', '--port', '9001'],
            environ={'APP_MODE': 'live', 'DEEPSEEK_API_KEY': 'sk-test-placeholder', 'PORT': '7860'},
        )
        self.assertEqual('demo', s_override.effective_mode)
        self.assertEqual(9001, s_override.port)

    def test_deployment_artifacts_exist_and_follow_container_security_hygiene(self):
        dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
        dockerignore = (ROOT / '.dockerignore').read_text(encoding='utf-8')
        compose = (ROOT / 'docker-compose.yml').read_text(encoding='utf-8')
        env_example = (ROOT / '.env.example').read_text(encoding='utf-8')
        gitignore = (ROOT / '.gitignore').read_text(encoding='utf-8')
        deploy_doc = (ROOT / 'docs/deployment.md').read_text(encoding='utf-8')

        self.assertIn('COPY vendor ./vendor', dockerfile)
        self.assertIn('import web_server', dockerfile)
        self.assertIn('USER appuser', dockerfile)
        self.assertIn('/api/health', dockerfile)
        self.assertIn('EXPOSE 7860', dockerfile)
        self.assertIn('data/auth/', dockerignore)
        self.assertIn('*.db', dockerignore)
        self.assertIn('DEEPSEEK_API_KEY', compose)
        self.assertIn('DEEPSEEK_API_KEY=', env_example)
        self.assertNotIn('sk-', env_example)
        self.assertIn('!.env.example', gitignore)
        self.assertIn('Hugging Face Spaces', deploy_doc)


if __name__ == '__main__':
    unittest.main(verbosity=2)
