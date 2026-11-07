"""Offline s12 durable scheduling state tests; no sleep, model, network or auth."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from customer_service import scheduler
from customer_service.scheduler import ScheduleManager, ScheduleStoreError, SCHEDULE_TOOLS, SHANGHAI, MAX_STORE_BYTES

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ERROR = 'PRIVATE_SCHEDULE_STORE_ERROR_492'


def at(day=6, hour=9, minute=0, second=0):
    return datetime(2026, 11, day, hour, minute, second, tzinfo=SHANGHAI)


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='scheduler_', dir=ROOT / 'runtime')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.base = self.root / 'schedules'
        self.manager = ScheduleManager(self.base)

    def ok(self, response):
        self.assertTrue(response['ok'], response)
        return response['data']

    def create(self, owner='alice', order='A1001', cron='0 9 * * *', manager=None):
        return self.ok((manager or self.manager).call(owner, 'schedule_tracking', {'order_id': order, 'cron': cron}))['schedule']

    def listed(self, owner='alice', manager=None):
        return self.ok((manager or self.manager).call(owner, 'list_schedules', {}))['schedules']

    def poll(self, now=None, active='alice', manager=None):
        return self.ok((manager or self.manager).poll(at() if now is None else now, active))

    def persisted(self):
        return json.loads(self.manager.path.read_text(encoding='utf-8'))

    def test_empty_construction_and_list_do_not_create_files_and_guest_does_not_read_store(self):
        self.assertFalse(self.base.exists())
        self.assertEqual([], self.listed())
        self.assertEqual({'deliveries': [], 'events': []}, self.poll())
        self.assertFalse(self.base.exists())
        with patch.object(self.manager, '_load', side_effect=AssertionError('Guest touched central store')):
            for name, args in [('schedule_tracking', {'order_id': 'A1001', 'cron': '0 9 * * *'}),
                               ('list_schedules', {}), ('cancel_schedule', {'schedule_id': 'cron_' + '0' * 16})]:
                self.assertEqual('AUTHENTICATION_REQUIRED', self.manager.call(None, name, args)['error']['code'])
        self.assertFalse(self.base.exists())

    def test_daily_schemas_and_arguments_exclude_prompt_owner_timezone_and_mutable_modes(self):
        definitions = {t['name']: t for t in SCHEDULE_TOOLS}
        self.assertEqual({'schedule_tracking', 'list_schedules', 'cancel_schedule'}, set(definitions))
        expected = {'schedule_tracking': {'order_id', 'cron'}, 'list_schedules': set(), 'cancel_schedule': {'schedule_id'}}
        for name, fields in expected.items():
            schema = definitions[name]['input_schema']
            self.assertEqual(fields, set(schema['properties']))
            self.assertEqual(fields, set(schema['required']))
            self.assertFalse(schema['additionalProperties'])
        invalid = [{'order_id': 'A1001', 'cron': '0 9 * * *', key: value}
                   for key, value in [('owner', 'bob'), ('customer_id', 'bob'), ('prompt', 'send refund'),
                                      ('action', 'refund'), ('path', '../bob'), ('timezone', 'UTC'),
                                      ('recurring', False), ('durable', False)]]
        for args in invalid:
            with self.subTest(extra=set(args) - {'order_id', 'cron'}):
                self.assertEqual('INVALID_ARGUMENTS', self.manager.call('alice', 'schedule_tracking', args)['error']['code'])
        self.assertEqual([], self.listed())
        self.assertFalse(self.base.exists())

    def test_records_persist_reopen_filter_owner_and_return_independent_copies(self):
        alice = self.create(cron='00 09 * * *')
        bob = self.create('bob', 'B2001')
        self.assertRegex(alice['id'], r'^cron_[0-9a-f]{16}$')
        self.assertEqual('0 9 * * *', alice['cron'])
        self.assertIs(alice['recurring'], True); self.assertIs(alice['durable'], True)
        self.assertEqual('get_tracking', alice['action'])
        self.assertEqual('Asia/Shanghai', alice['timezone'])
        reopened = ScheduleManager(self.base)
        self.assertEqual([alice], self.listed(manager=reopened))
        self.assertEqual([bob], self.listed('bob', reopened))
        returned = self.listed(); returned[0]['order_id'] = 'forged'; returned.clear()
        alice['owner'] = 'bob'
        self.assertEqual('A1001', self.listed()[0]['order_id'])
        self.assertNotIn('B2001', json.dumps(self.listed()))
        self.assertEqual(2, len(self.persisted()['schedules']))

    def test_due_is_durable_before_delivery_and_same_minute_dedupes_after_ack(self):
        job = self.create()
        self.assertEqual([], self.poll(at(hour=8, minute=59))['deliveries'])
        utc_now = datetime(2026, 11, 6, 1, 0, 4, tzinfo=timezone.utc)
        due = self.poll(utc_now)
        self.assertEqual([job['id']], [j['id'] for j in due['deliveries']])
        disk = self.persisted()['schedules'][0]
        self.assertTrue(disk['pending_delivery'])
        self.assertEqual('2026-11-06 09:00', disk['last_fired'])
        self.assertEqual(disk, self.ok(self.manager.begin_delivery(job['id'], 'alice'))['schedule'])
        self.assertEqual('SCHEDULE_IN_FLIGHT', self.manager.begin_delivery(job['id'], 'alice')['error']['code'])
        self.assertEqual([], self.poll(at(second=20))['deliveries'])
        ack = self.ok(self.manager.acknowledge(job['id']))
        self.assertIs(ack['delivery_only'], True)
        self.assertFalse(self.persisted()['schedules'][0]['pending_delivery'])
        self.assertEqual([], self.poll(at(second=59))['deliveries'])
        self.assertEqual([], self.poll(at(day=5))['deliveries'])
        self.assertEqual([job['id']], [j['id'] for j in self.poll(at(day=7))['deliveries']])

    def test_inactive_due_is_skipped_and_login_later_in_same_minute_does_not_catch_up(self):
        job = self.create()
        skipped = self.poll(active=None)
        self.assertEqual([], skipped['deliveries'])
        self.assertEqual([{'owner': 'alice', 'id': job['id'], 'status': 'skipped', 'code': 'OWNER_INACTIVE'}], skipped['events'])
        self.assertEqual('2026-11-06 09:00', self.listed()[0]['last_fired'])
        self.assertFalse(self.listed()[0]['pending_delivery'])
        reopened = ScheduleManager(self.base)
        self.assertEqual([], self.poll(at(second=30), manager=reopened)['deliveries'])
        self.assertEqual([], self.poll(at(hour=10), manager=reopened)['deliveries'])
        self.assertEqual(1, len(self.listed(manager=reopened)))
        self.assertEqual([job['id']], [j['id'] for j in self.poll(at(day=7), manager=reopened)['deliveries']])

    def test_offline_missed_time_is_not_enqueued_on_a_later_poll(self):
        job = self.create()
        restarted = ScheduleManager(self.base)
        self.assertEqual([], self.poll(at(hour=11), manager=restarted)['deliveries'])
        self.assertIsNone(self.listed(manager=restarted)[0]['last_fired'])
        self.assertEqual([job['id']], [j['id'] for j in self.poll(at(day=7), manager=restarted)['deliveries']])

    def test_pending_can_recover_in_fresh_process_but_shared_process_suppression_prevents_retry(self):
        job = self.create()
        self.poll(); self.ok(self.manager.begin_delivery(job['id'], 'alice'))
        self.ok(self.manager.restore_delivery(job['id']))
        self.assertEqual([], self.poll(at(second=15))['deliveries'])
        another = ScheduleManager(self.base)
        self.assertEqual([], self.poll(at(second=16), manager=another)['deliveries'])
        program = ('import json,sys; from datetime import datetime; '
                   'from customer_service.scheduler import ScheduleManager,SHANGHAI; '
                   'm=ScheduleManager(sys.argv[1]); p=m.poll(datetime(2026,11,6,9,0,17,tzinfo=SHANGHAI),"alice"); '
                   'j=p["data"]["deliveries"][0]; b=m.begin_delivery(j["id"],"alice"); '
                   'print(json.dumps({"poll":p,"begin":b}))')
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT)
        completed = subprocess.run([sys.executable, '-B', '-c', program, str(self.base)],
                                   cwd=str(ROOT), env=env, text=True, encoding='utf-8', capture_output=True, timeout=10)
        self.assertEqual(0, completed.returncode, completed.stderr)
        observed = json.loads(completed.stdout)
        self.assertEqual(job['id'], observed['poll']['data']['deliveries'][0]['id'])
        self.assertTrue(observed['begin']['ok'])
        self.assertTrue(self.listed()[0]['pending_delivery'])

    def test_wrong_owner_cannot_begin_and_poll_discards_old_pending_without_canceling_definition(self):
        job = self.create()
        self.poll()
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', self.manager.begin_delivery(job['id'], 'bob')['error']['code'])
        self.assertEqual('AUTHENTICATION_REQUIRED', self.manager.begin_delivery(job['id'], None)['error']['code'])
        self.assertTrue(self.listed()[0]['pending_delivery'])
        skipped = self.poll(at(hour=9, minute=1), active='bob')
        self.assertEqual([], skipped['deliveries'])
        self.assertEqual('OWNER_INACTIVE', skipped['events'][0]['code'])
        self.assertEqual({'owner', 'id', 'status', 'code'}, set(skipped['events'][0]))
        self.assertFalse(self.listed()[0]['pending_delivery'])
        self.assertEqual('2026-11-06 09:00', self.listed()[0]['last_fired'])
        self.assertEqual([], self.poll(at(hour=10))['deliveries'])
        self.assertEqual(1, len(self.listed()))

    def test_restore_is_suppressed_until_next_daily_tick_and_does_not_rewrite_pending(self):
        job = self.create()
        self.poll(); self.ok(self.manager.begin_delivery(job['id'], 'alice'))
        before = self.manager.path.read_bytes()
        restored = self.ok(self.manager.restore_delivery(job['id']))
        self.assertTrue(restored['pending_delivery']); self.assertTrue(restored['retry_suppressed'])
        self.assertEqual(before, self.manager.path.read_bytes())
        self.assertEqual('SCHEDULE_DELIVERY_SUPPRESSED', self.manager.begin_delivery(job['id'], 'alice')['error']['code'])
        self.assertEqual([], self.poll(at(second=40))['deliveries'])
        self.assertEqual([], self.poll(at(hour=20))['deliveries'])
        due = self.poll(at(day=7))
        self.assertEqual([job['id']], [j['id'] for j in due['deliveries']])
        self.assertEqual('2026-11-07 09:00', due['deliveries'][0]['last_fired'])
        self.ok(self.manager.begin_delivery(job['id'], 'alice'))
        self.ok(self.manager.acknowledge(job['id']))
        self.assertFalse(self.listed()[0]['pending_delivery'])

    def test_cancel_checks_owner_removes_unbegun_pending_and_cannot_kill_inflight(self):
        job = self.create()
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', self.manager.call('bob', 'cancel_schedule', {'schedule_id': job['id']})['error']['code'])
        self.poll(); self.ok(self.manager.begin_delivery(job['id'], 'alice'))
        denied = self.manager.call('alice', 'cancel_schedule', {'schedule_id': job['id']})
        self.assertEqual('SCHEDULE_IN_FLIGHT', denied['error']['code'])
        self.ok(self.manager.restore_delivery(job['id']))
        self.ok(self.manager.call('alice', 'cancel_schedule', {'schedule_id': job['id']}))
        self.assertEqual([], self.listed())
        self.assertEqual([], self.poll(at(day=7))['deliveries'])
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND', self.manager.begin_delivery(job['id'], 'alice')['error']['code'])

    def test_capacity_limits_are_per_customer_and_global_without_overwriting_existing_jobs(self):
        for owner in ('alice', 'bob', 'charlie', 'dora'):
            for i in range(5):
                self.create(owner, f'{owner}_order_{i}')
            rejected = self.manager.call(owner, 'schedule_tracking', {'order_id': 'EXTRA', 'cron': '0 10 * * *'})
            self.assertEqual('SCHEDULE_LIMIT_REACHED', rejected['error']['code'])
        before = self.manager.path.read_bytes()
        rejected = self.manager.call('eve', 'schedule_tracking', {'order_id': 'EXTRA', 'cron': '0 10 * * *'})
        self.assertEqual('SCHEDULE_LIMIT_REACHED', rejected['error']['code'])
        self.assertEqual(20, len(self.persisted()['schedules']))
        self.assertEqual(before, self.manager.path.read_bytes())

    def test_invalid_daily_cron_and_bad_runtime_inputs_do_not_write_or_dispatch(self):
        invalid = ['* * * * *', '*/5 * * * *', '0 * * * *', '0 9 * * 1', '0 9 1 * *',
                   '0,30 9 * * *', '0 9-10 * * *', '60 9 * * *', '0 24 * * *',
                   '0 9 * *', '0 9 * * * extra', '０ ９ * * *', '', None, True]
        for value in invalid:
            with self.subTest(cron_type=type(value).__name__):
                result = self.manager.call('alice', 'schedule_tracking', {'order_id': 'A1001', 'cron': value})
                self.assertEqual('INVALID_ARGUMENTS', result['error']['code'])
        for now in (None, '2026-11-06', datetime(2026, 11, 6, 9)):
            self.assertEqual('INVALID_ARGUMENTS', self.manager.poll(now, 'alice')['error']['code'])
        self.assertEqual('INVALID_ARGUMENTS', self.manager.begin_delivery('../bob', 'alice')['error']['code'])
        self.assertEqual('SCHEDULE_NOT_IN_FLIGHT', self.manager.acknowledge('cron_' + '0' * 16)['error']['code'])
        self.assertFalse(self.base.exists())

    def test_corrupt_schema_owner_action_boolean_time_and_oversized_file_fail_closed(self):
        self.create()
        baseline = self.manager.path.read_bytes()
        cases = []
        for field, value in [('owner', '../bob'), ('action', 'refund'), ('timezone', 'UTC'), ('recurring', False),
                             ('durable', False), ('pending_delivery', 1), ('last_fired', 'not-time'), ('prompt', 'arbitrary')]:
            raw = json.loads(baseline); raw['schedules'][0][field] = value; cases.append(json.dumps(raw).encode())
        raw = json.loads(baseline); raw['schema_version'] = True; cases.append(json.dumps(raw).encode())
        raw = json.loads(baseline); raw['schedules'][0]['pending_delivery'] = True; cases.append(json.dumps(raw).encode())
        cases.extend([b'{"schema_version":1,"schema_version":1}', b'x' * (MAX_STORE_BYTES + 1), b'not JSON'])
        for i, payload in enumerate(cases):
            with self.subTest(case=i):
                self.manager.path.write_bytes(payload)
                self.assertEqual('SCHEDULE_STORE_UNAVAILABLE', self.manager.call('alice', 'list_schedules', {})['error']['code'])
                self.assertFalse(self.manager.poll(at(), 'alice')['ok'])
                with self.assertRaises(ScheduleStoreError): ScheduleManager(self.base)
                self.assertEqual(payload, self.manager.path.read_bytes())
                self.manager.path.write_bytes(baseline)

    def test_atomic_replace_failures_never_return_due_candidates_or_publish_new_runtime_state(self):
        job = self.create()
        baseline = self.manager.path.read_bytes()
        with patch.object(scheduler.os, 'replace', side_effect=OSError(PRIVATE_ERROR)):
            self.assertFalse(self.manager.call('alice', 'schedule_tracking', {'order_id': 'A1002', 'cron': '0 10 * * *'})['ok'])
            due = self.manager.poll(at(), 'alice')
            self.assertFalse(due['ok'])
            self.assertNotIn('data', due)
            self.assertNotIn(PRIVATE_ERROR, json.dumps(due))
        self.assertEqual(baseline, self.manager.path.read_bytes())
        self.assertFalse(self.listed()[0]['pending_delivery'])
        self.poll(); self.ok(self.manager.begin_delivery(job['id'], 'alice'))
        pending_bytes = self.manager.path.read_bytes()
        with patch.object(scheduler.os, 'replace', side_effect=OSError(PRIVATE_ERROR)):
            self.assertFalse(self.manager.acknowledge(job['id'])['ok'])
        self.assertEqual(pending_bytes, self.manager.path.read_bytes())
        self.assertEqual('SCHEDULE_IN_FLIGHT', self.manager.begin_delivery(job['id'], 'alice')['error']['code'])
        self.ok(self.manager.acknowledge(job['id']))
        acknowledged_bytes = self.manager.path.read_bytes()
        with patch.object(scheduler.os, 'replace', side_effect=OSError(PRIVATE_ERROR)):
            self.assertFalse(self.manager.call('alice', 'cancel_schedule', {'schedule_id': job['id']})['ok'])
        self.assertEqual(acknowledged_bytes, self.manager.path.read_bytes())
        self.assertEqual(1, len(self.listed()))
        self.assertEqual([], list(self.base.glob('.schedule_*.tmp')))

    def test_instances_share_inflight_and_observe_fresh_disk_and_busy_lock_safely(self):
        other = ScheduleManager(self.base)
        job = self.create()
        self.assertEqual([job], self.listed(manager=other))
        self.poll(); self.ok(other.begin_delivery(job['id'], 'alice'))
        self.assertEqual('SCHEDULE_IN_FLIGHT', self.manager.begin_delivery(job['id'], 'alice')['error']['code'])
        self.ok(self.manager.acknowledge(job['id']))
        lock = self.base / '.schedules.lock'
        with lock.open('r+b') as held:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(held.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                result = other.call('alice', 'list_schedules', {})
                self.assertEqual('SCHEDULE_STORE_BUSY', result['error']['code'])
                self.assertNotIn('data', result)
            finally:
                if os.name == 'nt':
                    held.seek(0); msvcrt.locking(held.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(held.fileno(), fcntl.LOCK_UN)
        self.assertTrue(lock.is_file())
        self.assertFalse(self.listed()[0]['pending_delivery'])

    def test_process_crash_releases_native_lock_without_deleting_lock_file_and_pending_recovers(self):
        job = self.create()
        self.poll()
        lock = self.base / '.schedules.lock'
        program = (
            "import os,sys\n"
            "f=open(sys.argv[1],'r+b')\n"
            "if os.name == 'nt':\n"
            " import msvcrt; msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)\n"
            "else:\n"
            " import fcntl; fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            "print('LOCKED',flush=True)\n"
            "sys.stdin.readline()\n")
        process = subprocess.Popen([sys.executable, '-B', '-u', '-c', program, str(lock)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding='utf-8', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            self.assertEqual('LOCKED', process.stdout.readline().strip())
            self.assertEqual('SCHEDULE_STORE_BUSY', self.manager.call('alice', 'list_schedules', {})['error']['code'])
            process.terminate()
            process.communicate(timeout=5)
        finally:
            if process.poll() is None:
                process.kill(); process.communicate(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None: stream.close()
        self.assertTrue(lock.is_file())
        reopened = ScheduleManager(self.base)
        self.assertEqual([job['id']], [j['id'] for j in self.poll(at(second=20), manager=reopened)['deliveries']])
        self.ok(reopened.begin_delivery(job['id'], 'alice'))
        self.ok(reopened.acknowledge(job['id']))

    def test_confirmed_replace_has_no_redundant_temp_unlink_and_post_publish_unlock_error_is_unknown(self):
        real_unlink = Path.unlink
        def refuse_temp_cleanup(path, *args, **kwargs):
            if path.name.startswith('.schedule_'):
                raise OSError(PRIVATE_ERROR)
            return real_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', refuse_temp_cleanup):
            job = self.create()
        self.assertEqual([job], self.listed())
        self.assertEqual([], list(self.base.glob('.schedule_*.tmp')))
        if os.name == 'nt':
            import msvcrt
            native_lock = msvcrt.locking
            def fail_unlock(fd, mode, length):
                if mode == msvcrt.LK_UNLCK: raise OSError(PRIVATE_ERROR)
                return native_lock(fd, mode, length)
            injection = patch.object(msvcrt, 'locking', side_effect=fail_unlock)
        else:
            import fcntl
            native_lock = fcntl.flock
            def fail_unlock(fd, operation):
                if operation == fcntl.LOCK_UN: raise OSError(PRIVATE_ERROR)
                return native_lock(fd, operation)
            injection = patch.object(fcntl, 'flock', side_effect=fail_unlock)
        with injection:
            result = self.manager.call('alice', 'schedule_tracking', {'order_id': 'A1002', 'cron': '0 10 * * *'})
        self.assertFalse(result['ok'])
        self.assertEqual('SCHEDULE_WRITE_UNKNOWN', result['error']['code'])
        self.assertNotIn(PRIVATE_ERROR, json.dumps(result))
        # The definition really was published: unknown must not be described as not saved.
        self.assertEqual({'A1001', 'A1002'}, {j['order_id'] for j in self.listed()})

    def test_root_relocation_and_external_directory_link_cannot_escape_the_bound_storage(self):
        self.create()
        original_directory = self.manager.directory
        self.manager.directory = self.root / 'outside'
        self.assertEqual('SCHEDULE_STORE_UNAVAILABLE', self.manager.call('alice', 'list_schedules', {})['error']['code'])
        self.assertFalse((self.root / 'outside').exists())
        self.manager.directory = original_directory
        target = self.root / 'external'; target.mkdir()
        (target / 'keep.txt').write_text('must survive link cleanup', encoding='utf-8')
        link = self.root / 'linked_store'
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            quote = lambda value: "'" + str(value).replace("'", "''") + "'"
            command = 'New-Item -ItemType Junction -Path ' + quote(link) + ' -Target ' + quote(target) + ' -ErrorAction Stop | Out-Null'
            created = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                                     text=True, capture_output=True, timeout=10)
            if created.returncode != 0:
                self.skipTest('Neither symlink nor fixture-local Windows junction is available')
        try:
            with self.assertRaises(ScheduleStoreError): ScheduleManager(link)
        finally:
            # Only unlink this fixture-local directory link, never recursively delete its target.
            self.assertEqual(self.root, link.parent)
            self.assertEqual(target, link.resolve())
            link.rmdir()
        self.assertEqual('must survive link cleanup', (target / 'keep.txt').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
