"""Offline state/concurrency acceptance of session-only s13 teams."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
from time import perf_counter
import unittest
from unittest.mock import patch
import uuid

from customer_service import teams
from customer_service.teams import TeamManager, TeamError, TEAM_TOOLS, TEAM_NAMES

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_ERROR='PRIVATE_TEAM_PROVIDER_ERROR_658'


def receipt(ok=True, **updates):
    value={'ok':ok,'report':'依据本次真实工具结果形成的合成报告。' if ok else '',
           'evidence':[{'tool':'get_tracking','tool_use_id':'fixture_lookup','result':{'ok':ok,'data':{'source':'synthetic#tracking'}}}],
           'evidence_ok':ok,'provider_ledger':[{'status':'received','usage':{'input_tokens':10,'output_tokens':2},'usage_complete':True}],
           'worker_session_id':'a'*32,'error_code':None if ok else 'QUERY_FAILED'}
    value.update(updates)
    return value


class Runner:
    def __init__(self, callback=None, cleanup_gate=None):
        self.callback=callback; self.jobs=[]; self.closed=threading.Event(); self.cleanup_gate=cleanup_gate
    def execute(self,job):
        self.jobs.append(deepcopy(job))
        return self.callback(job) if self.callback else receipt()
    def close(self):
        self.closed.set()
        if self.cleanup_gate: self.cleanup_gate.wait(3)


class TeamManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='teams_',dir=ROOT/'runtime')
        self.root=Path(self.tmp.name); self.managers=[]; self.gates=[]; self.runners={}; self.factory_calls=[]
        self.addCleanup(self.finish)

    def finish(self):
        for gate in self.gates: gate.set()
        for manager in self.managers: manager.close()
        for manager in self.managers:
            for worker in manager._workers.values():
                thread=worker['thread']
                if thread and thread.ident is not None:
                    thread.join(timeout=2)
                    self.assertFalse(thread.is_alive(),'Controlled fixture worker did not exit')
        self.tmp.cleanup()

    def event(self):
        gate=threading.Event(); self.gates.append(gate); return gate

    def manager(self,callbacks=None,customer='alice',factory=None):
        if factory is None:
            def factory(role):
                self.factory_calls.append(role)
                runner=Runner((callbacks or {}).get(role)); self.runners.setdefault(role,[]).append(runner)
                return runner
        manager=TeamManager(customer,uuid.uuid4().hex,self.root/uuid.uuid4().hex,factory)
        self.managers.append(manager); return manager

    def ok(self,result):
        self.assertTrue(result['ok'],result); return result['data']

    def start(self,manager,role='logistics',resource='A1001'):
        return self.ok(manager.start(role,resource,'只核对指定资源并保留依据。'))

    def ack(self,manager,envelopes):
        return self.ok(manager.acknowledge([e['envelope_id'] for e in envelopes]))

    def until(self,predicate):
        deadline=perf_counter()+2
        while perf_counter()<deadline:
            if predicate(): return
            threading.Event().wait(.002)
        self.fail('Controlled fixture did not reach expected state')

    def test_strict_tool_schemas_guest_and_scope_never_allow_identity_overrides(self):
        expected={'start_team_task':{'role','resource_id','instruction'},'resume_team_task':{'step_id','instruction'},
                  'list_team':set(),'shutdown_teammate':{'role'}}
        self.assertEqual(('logistics','policy'),TEAM_NAMES)
        for definition in TEAM_TOOLS:
            schema=definition['input_schema']; self.assertEqual(expected[definition['name']],set(schema['properties']))
            self.assertFalse(schema['additionalProperties'])
        guest=self.manager(customer=None)
        self.assertEqual('AUTHENTICATION_REQUIRED',guest.start('logistics','A1001','查物流')['error']['code'])
        self.assertEqual('AUTHENTICATION_REQUIRED',guest.resume('team_'+'0'*16,'重试')['error']['code'])
        manager=self.manager()
        for extra in ('owner','customer_id','session_id','path'):
            args={'role':'logistics','resource_id':'A1001','instruction':'只读',extra:'bob'}
            self.assertEqual('INVALID_ARGUMENTS',manager.call('start_team_task',args)['error']['code'])
        self.assertEqual([],self.factory_calls)
        with self.assertRaises(AttributeError): manager.customer_id='bob'
        self.assertEqual([],manager.list_state()['steps'])

    def test_different_roles_run_concurrently_but_only_ack_can_complete_cards_and_release_owners(self):
        entered={r:self.event() for r in TEAM_NAMES}; release=self.event()
        def callback(role):
            def execute(job):
                entered[role].set(); release.wait(3)
                return receipt(report=role+'独立报告')
            return execute
        manager=self.manager({r:callback(r) for r in TEAM_NAMES})
        jobs=[self.start(manager,'logistics'),self.start(manager,'policy','returns')]
        self.assertTrue(all(e.wait(1) for e in entered.values()))
        self.assertEqual([],manager.peek_ready(0))
        self.assertTrue(all(s['status']=='in_progress' for s in manager.list_state()['steps']))
        self.assertEqual(2,len({w['thread'].ident for w in manager._workers.values()}))
        self.assertTrue(all(w['thread'].daemon for w in manager._workers.values()))
        release.set(); envelopes=manager.peek_ready(1)
        self.assertEqual(2,sum(e['type']=='result' for e in envelopes))
        self.assertEqual(2,sum(e['type']=='idle_notification' for e in envelopes))
        self.assertTrue(all(s['status']=='in_progress' for s in manager.list_state()['steps']))
        saved=deepcopy(envelopes)
        envelopes[0]['receipt']['report']='caller mutation'
        self.assertEqual(saved,manager.peek_ready(0))
        self.ack(manager,saved)
        state=manager.list_state()
        self.assertTrue(all(s['status']=='completed' and s['owner'] is None for s in state['steps']))
        self.assertEqual(state['steps'],json.loads(manager.path.read_text(encoding='utf-8'))['steps'])
        self.ack(manager,saved)  # Idempotent ACK, no second execution.
        self.assertEqual([],manager.peek_ready(0))

    def test_two_callers_competing_for_same_role_have_one_atomic_claim_winner(self):
        manager=self.manager(); barrier=threading.Barrier(3); results=[]
        def compete():
            barrier.wait(); results.append(manager.start('logistics','A1001','核对物流'))
        callers=[threading.Thread(target=compete) for _ in range(2)]
        for t in callers:t.start()
        barrier.wait()
        for t in callers:t.join(timeout=1)
        self.assertEqual(1,sum(r['ok'] for r in results))
        self.assertEqual('TEAM_ROLE_BUSY',next(r for r in results if not r['ok'])['error']['code'])
        self.assertEqual(1,manager.list_state()['accepted_assignments'])
        self.assertEqual(1,len(manager.list_state()['steps']))
        envelopes=manager.peek_ready(1); self.ack(manager,envelopes)
        self.assertEqual(1,len(self.runners['logistics'][0].jobs))

    def test_successful_next_card_reuses_same_worker_runner_and_history(self):
        manager=self.manager()
        first=self.start(manager); self.ack(manager,manager.peek_ready(1))
        thread=manager._workers['logistics']['thread']
        second=self.start(manager,resource='A1002'); self.ack(manager,manager.peek_ready(1))
        self.assertEqual(1,first['step']['work_version'])
        self.assertEqual(3,second['step']['work_version'])
        self.assertEqual(4,manager.list_state()['work_versions']['logistics'])
        self.assertIs(thread,manager._workers['logistics']['thread'])
        self.assertEqual(['logistics'],self.factory_calls)
        jobs=self.runners['logistics'][0].jobs
        self.assertEqual([first['step']['step_id'],second['step']['step_id']],[j['step_id'] for j in jobs])
        self.assertNotEqual(jobs[0]['request_id'],jobs[1]['request_id'])
        state=manager.list_state(); state['steps'].clear()
        self.assertEqual(2,len(manager.list_state()['steps']))

    def test_failed_or_evidence_missing_result_keeps_owner_and_requires_explicit_resume_same_card(self):
        for bad in (receipt(False),receipt(evidence_ok=False),receipt(report=' '),receipt(evidence=[])):
            with self.subTest(evidence_ok=bad['evidence_ok'],report_chars=len(bad['report'])):
                responses=[bad,receipt()]
                manager=self.manager({'logistics':lambda job:responses.pop(0)})
                first=self.start(manager); envelopes=manager.peek_ready(1)
                self.assertEqual('TEAM_RESULT_UNACKNOWLEDGED',manager.resume(first['step']['step_id'],'显式重试')['error']['code'])
                self.ack(manager,envelopes)
                card=manager.list_state()['steps'][0]
                self.assertEqual(('in_progress','logistics'),(card['status'],card['owner']))
                self.assertEqual('delivered',card['delivery_status'])
                self.assertTrue(card['last_error_code'])
                self.assertIn(card['last_error_code'],card['note'])
                self.assertIsNone(card['completion_proof'])
                self.assertEqual(card,json.loads(manager.path.read_text(encoding='utf-8'))['steps'][0])
                self.assertEqual('TEAM_ROLE_BUSY',manager.start('logistics','A1002','另一卡')['error']['code'])
                resumed=self.ok(manager.resume(card['step_id'],'现在显式重查原资源'))
                self.assertEqual(card['step_id'],resumed['step']['step_id'])
                self.assertEqual(2,resumed['step']['work_version'])
                self.assertNotEqual(first['request_id'],resumed['request_id'])
                self.ack(manager,manager.peek_ready(1))
                self.assertEqual('completed',manager.list_state()['steps'][0]['status'])
                manager.close()
                self.until(lambda:manager._workers['logistics']['state']=='stopped')

    def test_timeout_does_not_fake_idle_and_late_receipt_cannot_complete_explicit_new_version(self):
        entered,release=self.event(),self.event(); calls=[]
        def execute(job):
            calls.append(job['request_id'])
            if len(calls)==1:entered.set(); release.wait(3)
            return receipt()
        manager=self.manager({'logistics':execute}); first=self.start(manager)
        self.assertTrue(entered.wait(1))
        expired=manager.peek_ready(.02)
        self.assertEqual(['result'],[e['type'] for e in expired])
        self.assertEqual('TEAM_TIMEOUT',expired[0]['receipt']['error_code'])
        self.ack(manager,expired)
        self.assertEqual('TEAM_WORKER_BUSY',manager.resume(first['step']['step_id'],'过早重试')['error']['code'])
        release.set(); self.until(lambda:manager._workers['logistics']['state']=='idle')
        late=manager.peek_ready(0)
        self.assertFalse(any(e['type']=='result' and e['request_id']==first['request_id'] for e in late))
        self.ack(manager,late)
        second=self.ok(manager.resume(first['step']['step_id'],'明确重试'))
        current=manager.peek_ready(1)
        self.ack(manager,expired)  # Old ACK cannot touch the new version.
        self.assertEqual('in_progress',manager.list_state()['steps'][0]['status'])
        self.assertEqual(second['request_id'],manager.list_state()['steps'][0]['request_id'])
        self.ack(manager,current)
        self.assertEqual('completed',manager.list_state()['steps'][0]['status'])

    def test_shutdown_requested_then_ack_precedes_actual_cleanup_and_stopped(self):
        cleanup=self.event(); runner=Runner(cleanup_gate=cleanup)
        manager=self.manager(factory=lambda role:runner)
        self.start(manager); self.ack(manager,manager.peek_ready(1))
        requested=self.ok(manager.shutdown('logistics'))
        self.assertEqual('REQUESTED',requested['status'])
        envelopes=manager.peek_ready(1)
        self.assertTrue(any(e['type']=='shutdown_ack' and e['shutdown_request_id']==requested['request_id'] for e in envelopes))
        self.assertFalse(any(e['type']=='stopped' for e in envelopes))
        self.assertTrue(runner.closed.wait(1))
        self.assertTrue(manager._workers['logistics']['thread'].is_alive())
        cleanup.set(); self.until(lambda:manager._workers['logistics']['state']=='stopped')
        self.assertTrue(any(e['type']=='stopped' for e in manager.peek_ready(0)))
        self.ack(manager,manager.peek_ready(0))

    def test_closed_blocking_workers_hold_global_slots_and_never_publish_to_new_session(self):
        with patch.object(teams,'_TEAM_SLOTS',threading.BoundedSemaphore(4)):
            old=[]
            for i in range(4):
                entered,release=self.event(),self.event()
                def execute(job,begun=entered,gate=release):
                    begun.set(); gate.wait(3); return receipt()
                manager=self.manager({'logistics':execute}); self.start(manager)
                self.assertTrue(entered.wait(1)); runner=self.runners['logistics'][-1]
                manager.close(); self.assertTrue(runner.closed.is_set())
                self.assertTrue(manager._workers['logistics']['thread'].is_alive())
                old.append((manager,release))
            current=self.manager()
            self.assertEqual('TEAM_WORKER_LIMIT_REACHED',current.start('logistics','A1001','只读')['error']['code'])
            old[0][1].set(); self.until(lambda:old[0][0]._workers['logistics']['state']=='stopped')
            self.start(current); self.ack(current,current.peek_ready(1))
            for manager,gate in old:
                gate.set(); self.until(lambda m=manager:m._workers['logistics']['state']=='stopped')
                self.assertEqual([],manager.peek_ready(0))
                self.assertTrue(manager.list_state()['closed'])

    def test_save_failure_prevents_queue_and_failed_ack_preserves_result_without_reexecuting(self):
        manager=self.manager()
        with patch.object(manager,'_save',side_effect=OSError(PRIVATE_ERROR)):
            denied=manager.start('logistics','A1001','只读')
        self.assertFalse(denied['ok']); self.assertNotIn(PRIVATE_ERROR,json.dumps(denied))
        self.assertEqual(0,manager.list_state()['accepted_assignments'])
        self.assertEqual([],manager.list_state()['steps'])
        self.assertEqual([],self.factory_calls)
        self.start(manager); envelopes=manager.peek_ready(1)
        with patch.object(manager,'_save',side_effect=OSError(PRIVATE_ERROR)):
            denied=manager.acknowledge([e['envelope_id'] for e in envelopes])
        self.assertFalse(denied['ok'])
        self.assertEqual('in_progress',manager.list_state()['steps'][0]['status'])
        self.assertEqual(envelopes,manager.peek_ready(0))
        self.ack(manager,envelopes)
        self.assertEqual(1,len(self.runners['logistics'][0].jobs))

    def test_ack_checks_all_request_step_versions_before_changing_or_consuming_any_envelope(self):
        manager=self.manager(); self.start(manager)
        envelopes=manager.peek_ready(1); result=next(e for e in envelopes if e['type']=='result')
        for field,value in (('request_id','req_'+'0'*16),('step_id','team_'+'0'*16),('work_version',99),('work_version',True),('customer_id','bob'),('session_id','b'*32)):
            with self.subTest(field=field):
                original=manager._ready[result['envelope_id']][field]
                manager._ready[result['envelope_id']][field]=value
                denied=manager.acknowledge([e['envelope_id'] for e in envelopes])
                self.assertEqual('TEAM_STALE_RESULT',denied['error']['code'])
                self.assertEqual('in_progress',manager.list_state()['steps'][0]['status'])
                self.assertEqual(len(envelopes),len(manager.peek_ready(0)))
                manager._ready[result['envelope_id']][field]=original
        self.assertEqual('TEAM_INVALID_ACK',manager.acknowledge([result['envelope_id'],'foreign-envelope'])['error']['code'])
        self.ack(manager,envelopes)

    def test_six_accepted_assignments_bound_resume_and_new_cards_without_extra_factory_calls(self):
        manager=self.manager()
        for i in range(6):
            self.start(manager,resource='A'+str(i)); self.ack(manager,manager.peek_ready(1))
        self.assertEqual(6,manager.list_state()['accepted_assignments'])
        self.assertEqual('TEAM_LIMIT_REACHED',manager.start('logistics','A7','只读')['error']['code'])
        self.assertEqual(1,len(self.runners['logistics']))
        self.assertEqual(6,len(self.runners['logistics'][0].jobs))

    def test_malformed_oversized_and_exception_receipts_never_complete_or_leak_raw_errors(self):
        values=[None,receipt(report='x'*3001),receipt(evidence=[{'body':'x'*25000}]),receipt(provider_ledger=[{}]*4),RuntimeError(PRIVATE_ERROR)]
        for value in values:
            with self.subTest(value_type=type(value).__name__):
                def execute(job,value=value):
                    if isinstance(value,BaseException):raise value
                    return value
                manager=self.manager({'logistics':execute}); self.start(manager)
                envelopes=manager.peek_ready(1)
                result=next(e for e in envelopes if e['type']=='result')
                self.assertFalse(result['receipt']['ok'])
                self.assertNotIn(PRIVATE_ERROR,json.dumps(envelopes))
                self.ack(manager,envelopes)
                self.assertEqual('in_progress',manager.list_state()['steps'][0]['status'])
                manager.close(); self.until(lambda:manager._workers['logistics']['state']=='stopped')

    def test_factory_return_after_close_never_executes_and_role_epoch_is_revoked(self):
        entered,release=self.event(),self.event(); runner=Runner()
        def factory(role):
            entered.set(); release.wait(3); return runner
        manager=self.manager(factory=factory)
        self.start(manager)
        self.assertTrue(entered.wait(1))
        previous=manager.list_state()['work_versions']['logistics']
        manager.close()
        self.assertGreater(manager.list_state()['work_versions']['logistics'],previous)
        release.set(); self.until(lambda:manager._workers['logistics']['state']=='stopped')
        self.assertTrue(runner.closed.is_set())
        self.assertEqual([],runner.jobs)
        self.assertEqual([],manager.peek_ready(0))

    def test_provider_attempts_and_usage_remain_unknown_when_failed_wrapper_has_no_safe_ledger(self):
        known=receipt(provider_attempts_known=True,provider_attempts=1,usage_complete=True)
        variants=[receipt(report='x'*3001,provider_ledger=[]),
                  dict(known,report='x'*3001),known,
                  receipt(report='x'*3001,provider_ledger=[{'raw_exception':PRIVATE_ERROR}])]
        for i,value in enumerate(variants):
            with self.subTest(case=i):
                manager=self.manager({'logistics':lambda job,value=value:deepcopy(value)})
                self.start(manager)
                envelopes=manager.peek_ready(1)
                result=next(e['receipt'] for e in envelopes if e['type']=='result')
                if i in (0,3):
                    self.assertEqual([],result['provider_ledger'])
                    self.assertFalse(result['provider_attempts_known'])
                    self.assertIsNone(result['provider_attempts'])
                    self.assertFalse(result['usage_complete'])
                else:
                    self.assertEqual(value['provider_ledger'],result['provider_ledger'])
                    self.assertTrue(result['provider_attempts_known'])
                    self.assertEqual(1,result['provider_attempts'])
                    self.assertTrue(result['usage_complete'])
                if i==2:
                    self.assertEqual(value,result)
                else:
                    self.assertFalse(result['ok'])
                self.assertNotIn(PRIVATE_ERROR,json.dumps(result))
                self.ack(manager,envelopes)
                manager.close(); self.until(lambda:manager._workers['logistics']['state']=='stopped')

    def test_thread_start_failure_and_unsafe_or_preexisting_board_do_not_publish_assignment(self):
        manager=self.manager()
        with patch.object(threading.Thread,'start',side_effect=RuntimeError(PRIVATE_ERROR)):
            for _ in range(5):
                denied=manager.start('logistics','A1001','只读')
                self.assertEqual('TEAM_START_FAILED',denied['error']['code'])
        self.assertEqual(0,manager.list_state()['accepted_assignments'])
        self.start(manager); self.ack(manager,manager.peek_ready(1))
        before=manager.path.read_bytes()
        with self.assertRaises(TeamError):TeamManager('alice',manager.session_id,manager.directory,lambda role:Runner())
        self.assertEqual(before,manager.path.read_bytes())
        original=manager.directory; manager.directory=self.root/'outside'
        denied=manager.start('policy','returns','只读')
        self.assertFalse(denied['ok'])
        self.assertFalse((self.root/'outside').exists())
        manager.directory=original


if __name__=='__main__':unittest.main(verbosity=2)
