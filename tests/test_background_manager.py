"""Event-controlled offline acceptance of bounded tracking background workers."""
from copy import deepcopy
import json
import threading
from time import monotonic
import unittest
from unittest.mock import Mock, patch
import uuid

from customer_service import background
from customer_service.background import (TrackingBackgroundManager, START_TRACKING_TOOL,
    BACKGROUND_SYSTEM, MAX_BACKGROUND_RESULT_CHARS, MAX_ACTIVE_BACKGROUND_WORKERS)

PRIVATE_EXCEPTION = 'PRIVATE_BACKGROUND_EXCEPTION_941'


def success(order='A1001'):
    return {'ok': True, 'data': {'order_id': order, 'status': 'delayed',
            'events': [{'note': '完整尾部事实：仍在运输途中'}], 'source': 'synthetic.json#tracking'},
            'dataset_version': 'offline-only', 'simulated': True}


class BackgroundManagerTests(unittest.TestCase):
    def setUp(self):
        self.managers = []
        self.gates = []
        self.threads = []
        self.real_thread = threading.Thread
        patcher = patch.object(background.threading, 'Thread', side_effect=self.spawn)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.finish)

    def spawn(self, *args, **kwargs):
        thread = self.real_thread(*args, **kwargs)
        self.threads.append(thread)
        return thread

    def finish(self):
        for manager in self.managers:
            manager.close()
        for event in self.gates:
            event.set()
        for thread in self.threads:
            if thread.ident is not None:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive(), 'A controlled test worker did not exit')

    def event(self):
        event = threading.Event()
        self.gates.append(event)
        return event

    def manager(self, reader=None, customer='alice', delay=0.0):
        result = TrackingBackgroundManager(customer, uuid.uuid4().hex,
            (lambda order: success(order)) if reader is None else reader, delay_seconds=delay)
        self.managers.append(result)
        return result

    def started(self, manager, order='A1001'):
        result = manager.start(order)
        self.assertTrue(result['ok'], result)
        data = result['data']
        self.assertEqual({'background_id', 'order_id', 'status', 'kind', 'result_pending'}, set(data))
        self.assertRegex(data['background_id'], r'^bg_[0-9a-f]{16}$')
        self.assertEqual(order, data['order_id'])
        self.assertEqual('submitted', data['status'])
        self.assertEqual('tracking_query', data['kind'])
        self.assertIs(data['result_pending'], True)
        self.assertNotIn('result', data)
        return data['background_id']

    def ready(self, manager, count=1):
        deadline = monotonic() + 2
        while monotonic() < deadline:
            ready = manager.peek_ready()
            if len(ready) >= count:
                return ready
            threading.Event().wait(0.002)
        self.fail('Controlled worker did not publish its expected result')

    def test_tool_schema_and_constructor_preserve_bound_scope_and_finite_delay(self):
        self.assertEqual('start_tracking_query', START_TRACKING_TOOL['name'])
        schema = START_TRACKING_TOOL['input_schema']
        self.assertEqual(['order_id'], schema['required'])
        self.assertEqual({'order_id': {'type': 'string', 'minLength': 1, 'maxLength': 80}}, schema['properties'])
        self.assertFalse(schema['additionalProperties'])
        self.assertIn('get_policy', BACKGROUND_SYSTEM)
        self.assertIn('get_tracking', BACKGROUND_SYSTEM)
        self.assertIn('task_notification', BACKGROUND_SYSTEM)
        reader = Mock()
        manager = self.manager(reader, delay=10)
        self.assertEqual('alice', manager.customer_id)
        self.assertRegex(manager.session_id, r'^[0-9a-f]{32}$')
        with self.assertRaises(AttributeError): manager.customer_id = 'bob'
        with self.assertRaises(AttributeError): manager.session_id = 'other-session'
        for delay in (-0.1, 11, float('inf'), float('-inf'), float('nan'), True, None, '0', 10 ** 1000):
            with self.subTest(delay_type=type(delay).__name__), self.assertRaises(ValueError):
                self.manager(reader, delay=delay)
        with self.assertRaises(ValueError):
            TrackingBackgroundManager('alice', 'offline-session', None)
        reader.assert_not_called()
        self.assertEqual([], self.threads)

    def test_submission_returns_while_reader_is_blocked_and_independent_work_can_continue(self):
        entered, release = self.event(), self.event()
        calls = []
        def slow(order):
            calls.append(order)
            entered.set()
            release.wait(5)
            return success(order)
        manager = self.manager(slow)
        identifier = self.started(manager)
        self.assertTrue(entered.wait(1))
        self.assertFalse(release.is_set())
        independent = [item for item in ('returns-policy', 'public-product')]
        self.assertEqual(2, len(independent))
        self.assertEqual([], manager.peek_ready())
        pending = manager.pending()
        self.assertEqual([identifier], [r['background_id'] for r in pending])
        self.assertEqual({'background_id', 'order_id', 'status'}, set(pending[0]))
        self.assertEqual('running', pending[0]['status'])
        self.assertTrue(self.threads[0].daemon)
        release.set()
        self.assertEqual('completed', self.ready(manager)[0]['status'])
        self.assertEqual(['A1001'], calls)

    def test_full_result_peek_copies_do_not_consume_and_acknowledgement_is_exactly_once(self):
        original = success()
        expected = deepcopy(original)
        reader = Mock(return_value=original)
        manager = self.manager(reader)
        identifier = self.started(manager)
        first = self.ready(manager)
        self.assertEqual(expected, first[0]['result'])
        self.assertEqual({'background_id', 'order_id', 'kind', 'status', 'result'}, set(first[0]))
        first[0]['result']['data']['events'][0]['note'] = 'caller mutation'
        first.clear()
        original['data']['events'][0]['note'] = 'external reader mutation'
        pending = manager.pending(); pending[0]['order_id'] = 'forged'
        self.assertEqual(expected, manager.peek_ready()[0]['result'])
        self.assertEqual('A1001', manager.pending()[0]['order_id'])
        self.assertEqual(1, reader.call_count)
        manager.acknowledge([identifier])
        self.assertEqual([], manager.peek_ready())
        self.assertEqual([], manager.pending())
        with self.assertRaises(ValueError): manager.acknowledge([identifier])
        manager.acknowledge([])
        # Once consumed, an explicit new request may query that order again.
        self.started(manager)
        self.ready(manager)
        self.assertEqual(2, reader.call_count)

    def test_duplicate_and_two_unconsumed_job_limit_include_ready_results(self):
        release = self.event()
        def slow(order):
            release.wait(5)
            return success(order)
        manager = self.manager(slow)
        first = self.started(manager, 'A1001')
        self.started(manager, 'A1002')
        self.assertEqual('BACKGROUND_DUPLICATE', manager.start('A1001')['error']['code'])
        self.assertEqual('BACKGROUND_LIMIT_REACHED', manager.start('A1003')['error']['code'])
        release.set()
        self.ready(manager, 2)
        self.assertEqual('BACKGROUND_LIMIT_REACHED', manager.start('A1003')['error']['code'])
        self.assertEqual('BACKGROUND_DUPLICATE', manager.start('A1002')['error']['code'])
        manager.acknowledge([first])
        self.started(manager, 'A1003')
        self.ready(manager, 2)

    def test_guest_bad_parameters_and_closed_manager_never_invoke_reader_or_launch_thread(self):
        reader = Mock(return_value=success())
        guest = self.manager(reader, customer=None)
        self.assertEqual('AUTHENTICATION_REQUIRED', guest.start('A1001')['error']['code'])
        manager = self.manager(reader)
        for order in (None, False, [], {}, {'order_id': 'A1001', 'customer_id': 'bob'}, '', ' ', 'x' * 81, 'A1001\n'):
            with self.subTest(order_type=type(order).__name__):
                self.assertEqual('INVALID_ARGUMENTS', manager.start(order)['error']['code'])
        manager.close(); manager.close()
        self.assertEqual('BACKGROUND_CLOSED', manager.start('A1001')['error']['code'])
        self.assertEqual([], manager.pending())
        self.assertEqual([], manager.peek_ready())
        reader.assert_not_called()
        self.assertEqual([], self.threads)

    def test_business_denial_is_full_failed_result_and_query_exception_is_safe_without_retry(self):
        denied = {'ok': False, 'error': {'code': 'ACCESS_DENIED_OR_NOT_FOUND', 'message': '该订单不属于当前身份。'},
                  'simulated': True, 'dataset_version': 'offline-only'}
        for response in (denied, RuntimeError(PRIVATE_EXCEPTION)):
            with self.subTest(response_type=type(response).__name__):
                reader = Mock(side_effect=response) if isinstance(response, BaseException) else Mock(return_value=response)
                manager = self.manager(reader)
                self.started(manager)
                item = self.ready(manager)[0]
                self.assertEqual('failed', item['status'])
                self.assertFalse(item['result']['ok'])
                if isinstance(response, dict):
                    self.assertEqual(denied, item['result'])
                else:
                    self.assertEqual('BACKGROUND_QUERY_FAILED', item['result']['error']['code'])
                self.assertNotIn(PRIVATE_EXCEPTION, json.dumps(item))
                self.assertEqual(1, reader.call_count)
                manager.close()

    def test_malformed_non_json_and_nonfinite_results_fail_safely_without_body_or_retry(self):
        invalid = [None, [], {'data': {}}, {'ok': 1}, {'ok': False}, {'ok': False, 'error': {}},
                   {'ok': False, 'error': {'code': 1}}, {'ok': False, 'error': {'code': 'bad\n' + PRIVATE_EXCEPTION}},
                   {'ok': True, 'data': {'not_json': object()}}, {'ok': True, 'data': {'nonfinite': float('nan')}}]
        for index, response in enumerate(invalid):
            with self.subTest(case=index):
                reader = Mock(return_value=response)
                manager = self.manager(reader)
                self.started(manager)
                item = self.ready(manager)[0]
                self.assertEqual('failed', item['status'])
                self.assertEqual('BACKGROUND_QUERY_FAILED', item['result']['error']['code'])
                self.assertNotIn(PRIVATE_EXCEPTION, json.dumps(item))
                self.assertEqual(1, reader.call_count)
                manager.close()

    def test_full_serialized_result_limit_accepts_boundary_and_rejects_oversize_without_truncation(self):
        base = {'ok': True, 'data': {'details': ''}, 'simulated': True, 'dataset_version': 'offline'}
        overhead = len(json.dumps(base, ensure_ascii=False))
        tail = '完整尾部依据'
        for extra in (0, 1):
            with self.subTest(extra=extra):
                response = deepcopy(base)
                response['data']['details'] = 'x' * (MAX_BACKGROUND_RESULT_CHARS - overhead - len(tail) + extra) + tail
                self.assertEqual(MAX_BACKGROUND_RESULT_CHARS + extra, len(json.dumps(response, ensure_ascii=False)))
                manager = self.manager(lambda order, value=response: value)
                self.started(manager)
                item = self.ready(manager)[0]
                if extra:
                    self.assertEqual('failed', item['status'])
                    self.assertEqual('BACKGROUND_RESULT_TOO_LARGE', item['result']['error']['code'])
                    self.assertNotIn('data', item['result'])
                else:
                    self.assertEqual('completed', item['status'])
                    self.assertEqual(response, item['result'])
                    self.assertTrue(item['result']['data']['details'].endswith(tail))
                manager.close()

    def test_thread_construction_or_start_failure_cleans_jobs_and_process_slots(self):
        manager = self.manager()
        for stage in ('constructor', 'start'):
            with self.subTest(stage=stage):
                for _ in range(MAX_ACTIVE_BACKGROUND_WORKERS + 1):
                    if stage == 'constructor':
                        with patch.object(background.threading, 'Thread', side_effect=OSError(PRIVATE_EXCEPTION)):
                            failed = manager.start('A1001')
                    else:
                        with patch.object(self.real_thread, 'start', side_effect=RuntimeError(PRIVATE_EXCEPTION)):
                            failed = manager.start('A1001')
                    self.assertEqual('BACKGROUND_START_FAILED', failed['error']['code'])
                    self.assertNotIn(PRIVATE_EXCEPTION, json.dumps(failed))
                    self.assertEqual([], manager.pending())
                    self.assertEqual([], manager.peek_ready())
        self.started(manager)
        self.assertEqual('completed', self.ready(manager)[0]['status'])

    def test_close_cancels_cooperative_delay_without_calling_reader(self):
        reader = Mock(return_value=success())
        manager = self.manager(reader, delay=10)
        self.started(manager)
        manager.close()
        self.threads[-1].join(timeout=1)
        self.assertFalse(self.threads[-1].is_alive())
        reader.assert_not_called()
        self.assertEqual([], manager.peek_ready())
        self.assertEqual([], manager.pending())
        self.assertEqual('BACKGROUND_CLOSED', manager.start('A1002')['error']['code'])

    def test_close_does_not_claim_to_kill_running_reader_and_discards_late_result(self):
        entered, release = self.event(), self.event()
        calls = []
        def slow(order):
            calls.append(order)
            entered.set()
            release.wait(5)
            return success(order)
        manager = self.manager(slow)
        self.started(manager)
        self.assertTrue(entered.wait(1))
        worker = self.threads[-1]
        manager.close(); manager.close()
        self.assertTrue(worker.is_alive())
        self.assertEqual([], manager.pending())
        self.assertEqual([], manager.peek_ready())
        release.set()
        worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.assertEqual([], manager.peek_ready())
        self.assertEqual(['A1001'], calls)

    def test_revoked_blocking_workers_still_count_toward_global_capacity_until_real_exit(self):
        slots = threading.BoundedSemaphore(MAX_ACTIVE_BACKGROUND_WORKERS)
        with patch.object(background, '_WORKER_SLOTS', slots):
            old = []
            for i in range(MAX_ACTIVE_BACKGROUND_WORKERS):
                entered, release = self.event(), self.event()
                def slow(order, begun=entered, gate=release):
                    begun.set()
                    gate.wait(5)
                    return success(order)
                manager = self.manager(slow)
                self.started(manager, 'A' + str(i))
                self.assertTrue(entered.wait(1))
                worker = self.threads[-1]
                manager.close()
                old.append((manager, release, worker))
            newcomer = self.manager()
            self.assertEqual('BACKGROUND_WORKER_LIMIT_REACHED', newcomer.start('NEW_ORDER')['error']['code'])
            self.assertEqual([], newcomer.pending())
            old[0][1].set(); old[0][2].join(timeout=1)
            self.assertFalse(old[0][2].is_alive())
            self.started(newcomer, 'NEW_ORDER')
            self.ready(newcomer)
            for manager, release, worker in old:
                release.set(); worker.join(timeout=1)
                self.assertFalse(worker.is_alive())
                self.assertEqual([], manager.peek_ready())

    def test_acknowledgement_validates_whole_batch_and_rejects_other_session_or_pending_ids(self):
        release = self.event()
        def mixed(order):
            if order == 'A1001': release.wait(5)
            return success(order)
        manager = self.manager(mixed)
        pending_id = self.started(manager, 'A1001')
        ready_id = self.started(manager, 'A1002')
        self.assertEqual([ready_id], [r['background_id'] for r in self.ready(manager)])
        other = self.manager()
        foreign_id = self.started(other, 'A1003')
        self.ready(other)
        for ids in ([ready_id, pending_id], [ready_id, foreign_id], [ready_id, ready_id],
                    ['bg_' + '0' * 16], '../foreign', [None]):
            with self.subTest(ids_type=type(ids).__name__), self.assertRaises(ValueError):
                manager.acknowledge(ids)
            self.assertEqual([ready_id], [r['background_id'] for r in manager.peek_ready()])
        self.assertEqual([foreign_id], [r['background_id'] for r in other.peek_ready()])
        release.set()
        self.ready(manager, 2)
        manager.acknowledge([pending_id, ready_id])
        self.assertEqual([], manager.pending())
        self.assertEqual([], manager.peek_ready())


if __name__ == '__main__':
    unittest.main(verbosity=2)
