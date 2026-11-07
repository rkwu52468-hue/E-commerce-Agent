"""Fake SDK plus real local threads for the independent planning experiment."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from customer_service import live_background_plan as experiment
from customer_service.background import TrackingBackgroundManager
from customer_service.context import validate_tool_pairs
from customer_service.shop import Shop, failure
from test_memory import RawSDK

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ERROR = 'PRIVATE_PLANNING_SDK_ERROR_691'


def text(value):
    return {'type': 'text', 'text': value}


def plan(serial=False, mode='background'):
    return {'steps': [
        {'id': 'policy', 'tool': 'get_policy', 'arguments': {'policy_id': 'returns'},
         'depends_on': ['tracking'] if serial else [], 'mode': 'sync', 'reason': '核对现有规则依据'},
        {'id': 'tracking', 'tool': 'get_tracking', 'arguments': {'order_id': 'A1001'},
         'depends_on': [], 'mode': mode, 'reason': '核对当前物流记录'}], 'answer_after': ['tracking', 'policy']}


def tool(value, identifier='plan_call', name='submit_query_plan'):
    return {'type': 'tool_use', 'id': identifier, 'name': name, 'input': deepcopy(value)}


def response(content, stop='end_turn', usage='default'):
    if usage == 'default': usage = {'input_tokens': 100, 'output_tokens': 10}
    return {'content': deepcopy(content), 'stop_reason': stop, 'usage': deepcopy(usage)}


class LiveBackgroundPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='live_background_plan_', dir=ROOT / 'runtime')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'data').mkdir()
        shutil.copyfile(ROOT / 'data/shop.json', self.root / 'data/shop.json')
        shutil.copyfile(ROOT / 'config.json', self.root / 'config.json')
        timer = patch.object(experiment, 'SIMULATED_TRACKING_DELAY_SECONDS', 0)
        timer.start(); self.addCleanup(timer.stop)

    def perform(self, value=None, replies=None):
        trial = experiment.prepare_trial(self.root)
        replies = replies if replies is not None else [response([tool(plan() if value is None else value)], 'tool_use'),
                                                      response([text('依据实际只读查询说明物流与政策；不办理退款或售后。')])]
        raw = RawSDK(replies)
        model = experiment.BudgetedModel('offline-no-provider-key', trial['config'], client=raw)
        with redirect_stdout(io.StringIO()):
            report = experiment.execute_trial(trial, model)
        history = json.loads((trial['context'].session_dir / 'history.json').read_text(encoding='utf-8'))['messages']
        validate_tool_pairs(history)
        return trial, raw, model, report, history

    def test_prepare_exposes_real_catalog_and_fair_capability_metadata_without_plan_or_business_answers(self):
        with patch.object(Shop, 'call', side_effect=AssertionError('prepare queried business data')), \
             patch.object(experiment, 'TrackingBackgroundManager', side_effect=AssertionError('prepare created workers')):
            trial = experiment.prepare_trial(self.root)
        self.assertEqual([{'role': 'user', 'content': experiment.REQUEST}], trial['messages'])
        self.assertEqual(['submit_query_plan'], [t['name'] for t in trial['tools']])
        first = json.loads((trial['context'].session_dir / 'first-payload.json').read_text(encoding='utf-8'))
        self.assertEqual(1024, first['max_tokens'])
        self.assertEqual({'type': 'disabled'}, first['thinking'])
        self.assertIn('sync_tracking_delay_seconds', first['system'])
        self.assertIn('background_tracking_delay_seconds', first['system'])
        self.assertIn('shared_synthetic_fixture', first['system'])
        self.assertIn('returns', first['system'])
        self.assertNotIn('answer_after', first['messages'][0]['content'])
        self.assertNotIn('"status": "delayed"', json.dumps(first, ensure_ascii=False))
        self.assertNotIn('shop.json#orders.A1001', json.dumps(first, ensure_ascii=False))
        self.assertNotIn('16900', first['system'])
        self.assertEqual(0, trial['limits']['manager_extra_delay_seconds'])
        self.assertFalse(trial['limits']['memory'])
        self.assertFalse(trial['limits']['summary'])
        self.assertFalse(trial['limits']['subagent'])
        self.assertFalse(trial['limits']['persistent_tasks'])

    def test_independent_background_plan_really_overlaps_and_full_results_reach_second_request(self):
        first_blocks = [{'type': 'thinking', 'thinking': 'offline-only', 'signature': 'fixture-signature'},
                        text('先提交只读计划。'), tool(plan())]
        with patch.object(experiment, 'SIMULATED_TRACKING_DELAY_SECONDS', 0.06):
            trial, raw, model, report, history = self.perform(replies=[response(first_blocks, 'tool_use'), response([text('根据实际结果答复。')])])
        self.assertTrue(report['success'])
        self.assertTrue(report['execution_success'])
        self.assertTrue(report['answer_requires_review'])
        for key in ('plan_valid', 'required_checks_covered', 'independence_identified', 'model_selected_background', 'overlap_observed', 'answer_after_satisfied'):
            self.assertIs(report[key], True, key)
        self.assertEqual('ready_background_before_sync', report['program_scheduler'])
        self.assertEqual(plan(), report['query_plan'])
        self.assertEqual(2, len(raw.calls))
        self.assertEqual(0, report['real_api_calls'])
        self.assertNotIn('tools', raw.calls[1])
        self.assertEqual(first_blocks, history[1]['content'])
        returned = json.loads(raw.calls[1]['messages'][-1]['content'][0]['content'])
        self.assertEqual('plan_call', raw.calls[1]['messages'][-1]['content'][0]['tool_use_id'])
        for check in returned['data']['checks']:
            self.assertEqual(trial['shop'].call(check['tool'], check['arguments']), check['result'])
        intervals = {t['tool']: t for t in report['trace']}
        a, b = intervals['get_tracking'], intervals['get_policy']
        self.assertLess(max(a['start_seconds'], b['start_seconds']), min(a['end_seconds'], b['end_seconds']))
        self.assertGreaterEqual(a['duration_seconds'], 0.05)
        self.assertEqual({'input_tokens': 200, 'output_tokens': 20}, report['usage_totals'])
        validate_tool_pairs(raw.calls[1]['messages'])

    def test_safe_serial_plan_is_not_rewritten_and_does_not_pass_independence_or_overlap(self):
        submitted = plan(serial=True, mode='sync')
        _, _, _, report, _ = self.perform(submitted)
        self.assertTrue(report['success'])
        self.assertTrue(report['plan_valid'])
        self.assertTrue(report['answer_after_satisfied'])
        self.assertFalse(report['independence_identified'])
        self.assertFalse(report['model_selected_background'])
        self.assertFalse(report['overlap_observed'])
        self.assertEqual(submitted, report['query_plan'])
        intervals = {t['tool']: t for t in report['trace']}
        self.assertGreaterEqual(intervals['get_policy']['start_seconds'], intervals['get_tracking']['end_seconds'])

        with self.subTest(unrelated_foreign_tracking_does_not_count_toward_target_overlap=True):
            extra = plan(serial=True, mode='sync')
            extra['steps'].append({'id': 'other_order', 'tool': 'get_tracking', 'arguments': {'order_id': 'B2001'},
                                   'depends_on': [], 'mode': 'background', 'reason': '合成评估边界用例'})
            checks = [{'step_id': s['id'], 'status': 'failed' if s['id'] == 'other_order' else 'completed',
                       'result': failure('ACCESS_DENIED_OR_NOT_FOUND', '合成拒绝')} for s in extra['steps']]
            trace = [{'step_id': key, 'tool': name, 'phase': 'business_query', 'start_seconds': start, 'end_seconds': end}
                     for key, name, start, end in [('tracking','get_tracking',0,1), ('policy','get_policy',2,3),
                                                   ('other_order','get_tracking',2,4)]]
            with patch.object(experiment, '_execute_plan', return_value=(checks, trace, [])):
                _, _, _, report, _ = self.perform(extra)
            self.assertFalse(report['overlap_observed'])
            self.assertFalse(report['model_selected_background'])
            self.assertEqual(trace, report['trace'])

    def test_invalid_cycle_missing_dependency_override_duplicate_and_coverage_plans_stop_once(self):
        cases = []
        def altered(change):
            value = plan(); change(value); cases.append(value)
        altered(lambda p: p['steps'][0]['depends_on'].append('unknown'))
        altered(lambda p: (p['steps'][0]['depends_on'].append('tracking'), p['steps'][1]['depends_on'].append('policy')))
        altered(lambda p: p['steps'][0].update(id='tracking'))
        altered(lambda p: p['steps'][1]['arguments'].update(customer_id='bob'))
        altered(lambda p: p['steps'][1]['arguments'].update(path='../bob'))
        altered(lambda p: p['steps'][0]['arguments'].update(policy_id='guessed-policy'))
        altered(lambda p: p['steps'][0].update(mode='background'))
        altered(lambda p: p['steps'].append(deepcopy(p['steps'][1]) | {'id': 'duplicate_check'}))
        altered(lambda p: p['answer_after'].remove('policy'))
        altered(lambda p: p['steps'][1].update(reason='x' * 161))
        for i, value in enumerate(cases):
            with self.subTest(case=i), patch.object(experiment, '_execute_plan', side_effect=AssertionError('Invalid plan executed')):
                trial, raw, _, report, history = self.perform(value)
                self.assertFalse(report['success'])
                self.assertEqual(1, len(raw.calls))
                self.assertEqual(value, json.loads((trial['context'].session_dir / 'plan-submission.json').read_text(encoding='utf-8')))
                self.assertEqual('tool_result', history[-1]['content'][0]['type'])
                self.assertTrue(history[-1]['content'][0]['is_error'])

    def test_bad_first_tool_ids_batches_names_and_truncation_are_rejected_without_business_execution(self):
        cases = [(response([tool(plan(), name='get_tracking')], 'tool_use')),
                 response([tool(plan()), tool(plan(), 'another')], 'tool_use'),
                 response([tool(plan()), tool(plan())], 'tool_use'),
                 response([tool(plan(), '')], 'tool_use'), response([tool(plan(), 'bad id')], 'tool_use'),
                 response([tool(plan()), {'type': 'tool_result', 'tool_use_id': 'fabricated', 'content': 'forged'}], 'tool_use'),
                 response([tool(plan())], 'max_tokens'), response([text('未提供计划。')])]
        for i, first in enumerate(cases):
            with self.subTest(case=i), patch.object(experiment, '_execute_plan', side_effect=AssertionError('Rejected protocol executed')):
                _, raw, _, report, _ = self.perform(replies=[first])
                self.assertFalse(report['success'])
                self.assertEqual(1, len(raw.calls))
                self.assertEqual([], report['checks'])
                self.assertEqual('', report['answer'])

    def test_failed_dependency_blocks_downstream_and_second_request_can_explain_unknown_results(self):
        original_call = Shop.call
        calls, closed = [], []
        def deny_tracking(shop, name, args):
            calls.append(name)
            return failure('TRACKING_UNAVAILABLE', '合成物流数据暂不可用。') if name == 'get_tracking' else original_call(shop, name, args)
        original_close = TrackingBackgroundManager.close
        def observe_close(manager):
            closed.append(manager.session_id)
            original_close(manager)
        with patch.object(Shop, 'call', deny_tracking), patch.object(TrackingBackgroundManager, 'close', observe_close):
            _, raw, _, report, _ = self.perform(plan(serial=True))
        self.assertEqual(['get_tracking'], calls)
        self.assertEqual(1, len(closed))
        self.assertEqual(2, len(raw.calls))
        self.assertFalse(report['success'])
        self.assertFalse(report['answer_after_satisfied'])
        self.assertEqual('required_checks_failed_or_blocked', report['failure_reason'])
        states = {c['step_id']: c['status'] for c in report['checks']}
        self.assertEqual({'policy': 'blocked', 'tracking': 'failed'}, states)
        self.assertTrue(report['answer'])
        self.assertIn('DEPENDENCY_FAILED', json.dumps(raw.calls[1], ensure_ascii=False))

    def test_sdk_failure_truncation_and_incomplete_usage_count_attempts_without_retry_or_raw_errors(self):
        cases = [([RuntimeError(PRIVATE_ERROR)], 1),
                 ([response([tool(plan())], 'tool_use'), TimeoutError(PRIVATE_ERROR)], 2),
                 ([response([tool(plan())], 'max_tokens')], 1),
                 ([response([tool(plan())], 'tool_use'), response([text('半截答复')], 'max_tokens')], 2)]
        for replies, attempts in cases:
            with self.subTest(attempts=attempts, error=type(replies[-1]).__name__):
                _, raw, model, report, _ = self.perform(replies=replies)
                self.assertFalse(report['success'])
                self.assertEqual(attempts, len(raw.calls))
                self.assertEqual(attempts, len(model.calls))
                self.assertEqual('', report['answer'])
                self.assertNotIn(PRIVATE_ERROR, json.dumps(report, ensure_ascii=False))
                if any(isinstance(value, BaseException) for value in replies):
                    self.assertFalse(report['usage_complete'])
                    self.assertIsNone(report['usage_totals'])
        for usage in (None, {'input_tokens': 3}, {'input_tokens': 3, 'output_tokens': 1, 'cache_read_input_tokens': True}):
            with self.subTest(usage=usage):
                _, _, model, report, _ = self.perform(replies=[response([tool(plan())], 'tool_use', usage), response([text('已说明缺口。')])])
                self.assertFalse(report['usage_complete'])
                self.assertIsNone(report['usage_totals'])
                self.assertEqual(usage, model.calls[0]['usage'])
        usage = {'input_tokens': 3, 'output_tokens': 2, 'cache_read_input_tokens': 7}
        _, _, _, report, _ = self.perform(replies=[response([tool(plan())], 'tool_use', usage), response([text('答复。')], usage=usage)])
        self.assertEqual({'input_tokens': 6, 'output_tokens': 4, 'cache_read_input_tokens': 14}, report['usage_totals'])

    def test_payload_caps_are_checked_before_sdk_and_global_attempt_limit_is_two(self):
        trial = experiment.prepare_trial(self.root)
        raw = RawSDK([])
        model = experiment.BudgetedModel('offline', trial['config'], client=raw)
        trial['system'] += 'x' * experiment.PAYLOAD_CHAR_CAP
        report = experiment.execute_trial(trial, model)
        self.assertEqual(0, len(raw.calls))
        self.assertEqual(0, report['sdk_requests_attempted'])
        _, raw, _, report, _ = self.perform(replies=[response([text('x' * experiment.PAYLOAD_CHAR_CAP), tool(plan())], 'tool_use')])
        self.assertEqual(1, len(raw.calls))
        self.assertEqual('payload_character_cap', report['failure_reason'])
        trial, raw, model, report, history = self.perform()
        with self.assertRaises(experiment.TrialError):
            model.complete(history, [], trial['system'])
        self.assertEqual(2, len(raw.calls))
        for call in raw.calls:
            self.assertEqual(1024, call['max_tokens'])
            self.assertEqual({'type': 'disabled'}, call['thinking'])
            self.assertLessEqual(len(json.dumps(call, ensure_ascii=False)), 24000)

    def test_second_response_tools_reused_ids_and_empty_text_stop_without_extra_business_calls(self):
        for content in ([tool(plan(), 'extra')], [tool(plan(), 'plan_call')], [text('  ')]):
            with self.subTest(second_content=content):
                _, raw, model, report, _ = self.perform(replies=[response([tool(plan())], 'tool_use'), response(content, 'tool_use' if content[0]['type'] == 'tool_use' else 'end_turn')])
                self.assertFalse(report['success'])
                self.assertEqual('', report['answer'])
                self.assertEqual(2, len(raw.calls))
                self.assertEqual(2, len(report['checks']))
                self.assertEqual(content, model.calls[-1]['response_content'])

    def test_finite_background_deadline_reports_unfinished_and_closes_manager(self):
        with patch.object(experiment, 'SIMULATED_TRACKING_DELAY_SECONDS', 0.2), \
             patch.object(experiment, 'EXECUTION_TIMEOUT_SECONDS', 0.01):
            _, raw, _, report, _ = self.perform()
        self.assertEqual(2, len(raw.calls))
        self.assertFalse(report['success'])
        self.assertFalse(report['answer_after_satisfied'])
        tracking = next(c for c in report['checks'] if c['step_id'] == 'tracking')
        self.assertEqual('failed', tracking['status'])
        self.assertEqual('EXECUTION_DEADLINE', tracking['result']['error']['code'])
        self.assertIn({'event': 'execution_deadline'}, report['events'])

    def test_dry_run_reads_no_key_creates_no_sdk_or_worker_and_preserves_real_latest(self):
        (self.root / 'runtime').mkdir()
        latest = self.root / 'runtime/live-background-plan-latest.json'
        latest.write_bytes(b'{"existing_live_report":true}')
        before = latest.read_bytes()
        with patch.object(experiment, 'read_secret', side_effect=AssertionError('dry run read secret')), \
             patch.object(experiment, 'BudgetedModel', side_effect=AssertionError('dry run created SDK')), \
             patch.object(experiment, 'TrackingBackgroundManager', side_effect=AssertionError('dry run made workers')), \
             patch.object(Shop, 'call', side_effect=AssertionError('dry run queried data')), \
             patch.object(os.environ, 'get', side_effect=AssertionError('dry run read env')), redirect_stdout(io.StringIO()):
            preview = experiment.run_plan_trial(self.root, dry_run=True)
        self.assertEqual(0, preview['api_calls'])
        self.assertEqual(0, preview['real_api_calls'])
        self.assertEqual(before, latest.read_bytes())
        self.assertTrue((self.root / 'runtime/live-background-plan-preview.json').is_file())

    def test_plan_evidence_write_failure_closes_paid_tool_id_and_preserves_request_ledger_when_possible(self):
        original = experiment._write
        def fail_plan_file(path, value):
            if Path(path).name == 'plan-submission.json':
                raise OSError(PRIVATE_ERROR)
            return original(path, value)
        with patch.object(experiment, '_write', side_effect=fail_plan_file), \
             patch.object(experiment, '_execute_plan', side_effect=AssertionError('Failed evidence write executed queries')):
            trial, raw, model, report, history = self.perform()
        self.assertEqual(1, len(raw.calls))
        self.assertFalse(report['success'])
        self.assertFalse(report['evidence_saved'])
        self.assertIn({'file': 'plan-submission.json', 'error_type': 'OSError'}, report['evidence_write_errors'])
        self.assertEqual('trial_error_OSError', report['failure_reason'])
        self.assertEqual('plan_call', history[-1]['content'][0]['tool_use_id'])
        self.assertTrue(history[-1]['content'][0]['is_error'])
        ledger = json.loads((trial['context'].session_dir / 'requests.json').read_text(encoding='utf-8'))
        self.assertEqual(model.calls, ledger)
        self.assertNotIn(PRIVATE_ERROR, json.dumps(report, ensure_ascii=False))
        validate_tool_pairs(history)
        # If all final evidence writes fail, retain an honest in-memory outcome and safe error notice.
        trial = experiment.prepare_trial(self.root)
        raw = RawSDK([response([tool(plan())], 'tool_use')])
        model = experiment.BudgetedModel('offline', trial['config'], client=raw)
        output = io.StringIO()
        with patch.object(experiment, '_write', side_effect=OSError(PRIVATE_ERROR)), redirect_stdout(output):
            report = experiment.execute_trial(trial, model)
        self.assertEqual(1, len(raw.calls))
        self.assertFalse(report['evidence_saved'])
        self.assertEqual({'plan-submission.json', 'history.json', 'requests.json', 'trial-report.json'}, {r['file'] for r in report['evidence_write_errors']})
        self.assertIn('保存未完成', output.getvalue())
        self.assertNotIn(PRIVATE_ERROR, output.getvalue())
        self.assertFalse((trial['context'].session_dir / 'requests.json').exists())

        with self.subTest(execution_evidence_write_failure=True):
            def fail_execution_file(path, value):
                if Path(path).name == 'execution-results.json':
                    raise OSError(PRIVATE_ERROR)
                return original(path, value)
            with patch.object(experiment, '_write', side_effect=fail_execution_file):
                trial, raw, _, report, history = self.perform()
            self.assertEqual(1, len(raw.calls))
            self.assertFalse(report['evidence_saved'])
            self.assertIn({'file': 'execution-results.json', 'error_type': 'OSError'}, report['evidence_write_errors'])
            self.assertTrue(report['session_report_saved'])
            self.assertTrue((trial['context'].session_dir / 'requests.json').is_file())
            validate_tool_pairs(history)
        with self.subTest(latest_publication_failure_preserves_previous_report=True):
            latest = self.root / 'runtime/live-background-plan-latest.json'
            latest.write_bytes(b'{"existing_live_report":true}')
            before = latest.read_bytes()
            raw = RawSDK([response([tool(plan(mode='sync'))], 'tool_use'), response([text('只读结果答复。')])])
            original_model = experiment.BudgetedModel
            def fake_model(api_key, config):
                return original_model(api_key, config, client=raw)
            def fail_latest(path, value):
                if Path(path).name == latest.name:
                    self.assertIs(value['latest_saved'], True)
                    raise OSError(PRIVATE_ERROR)
                return original(path, value)
            output = io.StringIO()
            with patch.object(experiment, '_write', side_effect=fail_latest), \
                 patch.object(experiment, 'BudgetedModel', side_effect=fake_model), \
                 patch.object(experiment, 'read_secret', return_value='offline-test-key'), \
                 patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''}), redirect_stdout(output):
                report = experiment.run_plan_trial(self.root)
            self.assertFalse(report['latest_saved'])
            self.assertEqual({'file': latest.name, 'error_type': 'OSError'}, report['latest_write_error'])
            self.assertEqual(before, latest.read_bytes())
            self.assertNotIn(str(latest), output.getvalue())
            self.assertIn('当前会话报告（latest未保存）', output.getvalue())
            self.assertEqual(str(Path(report['session_dir']) / 'trial-report.json'), report['available_report_path'])
            self.assertTrue(Path(report['available_report_path']).is_file())
            self.assertNotIn(PRIVATE_ERROR, output.getvalue())

    def test_run_with_injected_fake_sdk_prints_each_dimension_and_writes_evidence_without_key(self):
        raw = RawSDK([response([tool(plan(serial=True, mode='sync'))], 'tool_use'), response([text('只读结果答复。')])])
        original = experiment.BudgetedModel
        def fake_model(api_key, config):
            self.assertEqual('offline-hidden-test-key', api_key)
            return original(api_key, config, client=raw)
        output = io.StringIO()
        with patch.object(experiment, 'BudgetedModel', side_effect=fake_model), \
             patch.object(experiment, 'read_secret', return_value='offline-hidden-test-key'), \
             patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''}), redirect_stdout(output):
            report = experiment.run_plan_trial(self.root)
        for key in ('plan_valid', 'required_checks_covered', 'independence_identified', 'model_selected_background', 'overlap_observed', 'answer_after_satisfied'):
            self.assertIn(key + '=' + str(report[key]), output.getvalue())
        self.assertEqual(0, report['real_api_calls'])
        self.assertTrue(report['latest_saved'])
        self.assertTrue((self.root / 'runtime/live-background-plan-latest.json').is_file())
        directory = Path(report['session_dir'])
        for filename in ('prepare.json', 'first-payload.json', 'plan-submission.json', 'execution-results.json', 'history.json', 'requests.json', 'trial-report.json'):
            self.assertTrue((directory / filename).is_file(), filename)
        self.assertNotIn('offline-hidden-test-key', output.getvalue())
        self.assertNotIn('offline-hidden-test-key', json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    unittest.main(verbosity=2)
