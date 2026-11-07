"""Controlled local team-trial integration, fake models and genuine worker threads."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

from customer_service import team_trial
from customer_service.context import validate_tool_pairs
from customer_service.live_session import LiveSession
from test_subagent import call, text
from test_team_session import FakeAuth, RoleModel, notices

ROOT=Path(__file__).resolve().parents[1]
PRIVATE_ERROR='PRIVATE_TEAM_TRIAL_ERROR_509'


class TrialModel:
    mode='offline-team-trial'
    def __init__(self,child=None,summary_error=None):
        self.worker=child or RoleModel(); self.summary_calls=[]; self.normal_calls=[]; self.normal_replies=[]
        self.summary_error=summary_error; self.summary_observer=None
    def complete(self,messages,tools,system):
        if not self.normal_replies:
            raise AssertionError('Trial unexpectedly requested a planning/main model turn')
        self.normal_calls.append({'messages':deepcopy(messages),'tools':deepcopy(tools),'system':system})
        blocks=deepcopy(self.normal_replies.pop(0))
        return {'content':blocks,'stop_reason':'tool_use' if any(b['type']=='tool_use' for b in blocks) else 'end_turn',
                'usage':{'input_tokens':25,'output_tokens':11,'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}
    def complete_teammate(self,messages,tools,system):
        if tools:
            return self.worker.complete(messages,tools,system)
        self.summary_calls.append({'messages':deepcopy(messages),'tools':deepcopy(tools),'system':system})
        if self.summary_observer:self.summary_observer()
        if self.summary_error:raise self.summary_error
        return {'content':[text('本轮只读查询依据已汇总；包裹问题仍需跟进，未办理退款或售后。')],
                'stop_reason':'end_turn','usage':{'input_tokens':25,'output_tokens':11,
                                               'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}


class TeamTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='team_trial_',dir=ROOT/'runtime')
        self.root=Path(self.temp.name);(self.root/'data').mkdir()
        shutil.copyfile(ROOT/'data/shop.json',self.root/'data/shop.json')
        self.sessions=[];self.auth=FakeAuth();self.addCleanup(self.finish)
    def finish(self):
        for session in self.sessions:
            session.close()
        for session in self.sessions:
            manager=session.harness.team if session.harness is not None else None
            if manager:
                for worker in manager._workers.values():
                    if worker['thread'].ident is not None:worker['thread'].join(2)
        self.temp.cleanup()
    def session(self,model=None,enabled=True,customer='alice'):
        model=model or TrialModel()
        session=LiveSession(self.root,model,store='json',allow_teams=enabled)
        self.sessions.append(session)
        if customer is not None:self.assertTrue(session.login(customer,'local-secret',self.auth))
        return session,model
    def report(self,result):
        self.assertTrue(result['report_path'])
        return json.loads(Path(result['report_path']).read_text(encoding='utf-8'))
    def invoke(self,session):
        return team_trial.run_team_trial(session)
    def assert_adapter_restored(self,h,original):
        model,tools,steps,system=original
        self.assertIs(model,h.model);self.assertEqual(tools,h.tools);self.assertEqual(steps,h.max_steps);self.assertEqual(system,h.system)
    def counts(self,model):return (len(model.worker.calls),len(model.summary_calls),len(model.normal_calls))

    def test_disabled_guest_and_bob_are_rejected_before_any_worker_or_model_request(self):
        for enabled,customer,code in ((False,'alice','TEAMS_DISABLED'),(True,None,'AUTHENTICATION_REQUIRED'),
                                      (True,'bob','ACCESS_DENIED_OR_NOT_FOUND')):
            with self.subTest(enabled=enabled,customer=customer):
                session,model=self.session(enabled=enabled,customer=customer)
                result=self.invoke(session)
                self.assertFalse(result['ok']);self.assertEqual(code,result['status'])
                self.assertEqual((0,0,0),self.counts(model))
                state=getattr(session.harness,'_team_trial_state',None)
                self.assertTrue(state is None or state['rounds']==0)

    def test_auxiliary_flags_and_summary_are_refused_without_creating_auxiliary_clients(self):
        for flag in ('memory','tasks','background','schedules','summary'):
            with self.subTest(flag=flag):
                session,model=self.session()
                if flag=='summary':session.harness.context.summary_client=object()
                else:setattr(session,'allow_'+flag,True)
                result=self.invoke(session)
                self.assertEqual('TEAM_TRIAL_INCOMPATIBLE_FLAGS',result['status'])
                self.assertEqual((0,0,0),self.counts(model))
                self.assertEqual([],session.harness.team.list_state()['steps'])

    def test_two_rounds_reuse_both_workers_but_require_four_fresh_assignments_and_bounded_summary_inputs(self):
        session,model=self.session();h=session.harness
        h.messages=[{'role':'user','content':'UNRELATED_OLD_MAIN_HISTORY_SECRET'},
                    {'role':'assistant','content':[text('旧主会话回答。')]}]
        case_path=self.root/'data/tasks/alice/existing-case.json';case_path.parent.mkdir(parents=True)
        case_path.write_bytes(b'{"status":"in_progress","sentinel":"preserve"}');case_before=case_path.read_bytes()
        original=(h.model,deepcopy(h.tools),h.max_steps,h.system)
        first=self.invoke(session);self.assertTrue(first['ok'],first);r1=self.report(first)
        workers={role:(id(worker['runner']),worker['thread'].ident) for role,worker in h.team._workers.items()}
        second=self.invoke(session);self.assertTrue(second['ok'],second);r2=self.report(second)
        self.assertEqual(workers,{role:(id(worker['runner']),worker['thread'].ident) for role,worker in h.team._workers.items()})
        self.assertEqual((8,2,0),self.counts(model))
        self.assertEqual(4,h.team.list_state()['accepted_assignments'])
        self.assertIsNone(first['metrics']['worker_context_reused'])
        self.assertIs(second['metrics']['worker_context_reused'],True)
        old_requests={j['request_id'] for j in r1['jobs']};new_requests={j['request_id'] for j in r2['jobs']}
        self.assertFalse(old_requests & new_requests)
        for result,report in ((first,r1),(second,r2)):
            self.assertEqual('completed',result['status'])
            self.assertTrue(report['program_assigned']);self.assertFalse(report['model_selected_team'])
            self.assertEqual(2,len(report['jobs']));self.assertEqual(2,len(report['notifications']))
            self.assertTrue(all(s['status']=='completed' and s['completion_proof'] and s['delivery_status']=='delivered' for s in report['steps']))
            self.assertTrue(all(n['receipt']['ok'] and n['receipt']['evidence_ok'] for n in report['notifications']))
            metrics=result['metrics']
            self.assertEqual(4,metrics['child_provider_attempts'])
            self.assertEqual(1,metrics['summary_provider_attempts'])
            self.assertEqual(5,metrics['total_provider_attempts'])
            self.assertTrue(metrics['usage_complete'])
            self.assertEqual(345,metrics['known_usage_subtotal']['input_tokens'])
            self.assertEqual(131,metrics['known_usage_subtotal']['output_tokens'])
            self.assertLessEqual(len(json.dumps(report['summary_input'],ensure_ascii=False)),24000)
        for request in model.summary_calls:
            self.assertEqual([],request['tools'])
            self.assertNotIn('UNRELATED_OLD_MAIN_HISTORY_SECRET',json.dumps(request,ensure_ascii=False))
            self.assertEqual(1,len(request['messages']))
        second_view=json.loads(model.summary_calls[1]['messages'][0]['content'])
        self.assertEqual(2,len(second_view['previous_evidence_reference_only']))
        self.assertEqual(new_requests,{n['request_id'] for n in second_view['current_reports']})
        for role in ('logistics','policy'):
            calls=[c for c in model.worker.calls if c['job']['role']==role]
            self.assertEqual(4,len(calls))
            self.assertGreater(len(calls[2]['messages']),len(calls[0]['messages']))
        self.assertTrue(all(count==2 for count in model.worker.counts.values()))
        self.assertEqual(case_before,case_path.read_bytes())
        self.assert_adapter_restored(h,original)
        self.assertFalse(any(b.get('type')=='tool_use' for m in h.messages if isinstance(m.get('content'),list) for b in m['content']))
        validate_tool_pairs(h.messages)

    def test_third_trial_is_rejected_before_worker_or_summary_calls(self):
        session,model=self.session()
        self.assertTrue(self.invoke(session)['ok']);self.assertTrue(self.invoke(session)['ok'])
        before=self.counts(model);assignments=session.harness.team.list_state()['accepted_assignments']
        denied=self.invoke(session)
        self.assertEqual('TEAM_TRIAL_LIMIT',denied['status'])
        self.assertEqual(before,self.counts(model))
        self.assertEqual(assignments,session.harness.team.list_state()['accepted_assignments'])
        self.assertEqual(2,session.harness._team_trial_state['rounds'])

    def test_failure_without_fresh_evidence_never_summarizes_and_blocks_later_trial(self):
        session,model=self.session(model=TrialModel(RoleModel(direct=True)))
        result=self.invoke(session)
        self.assertEqual('TEAM_TRIAL_EVIDENCE_FAILED',result['status'])
        self.assertEqual(0,len(model.summary_calls))
        self.assertEqual(2,len(model.worker.calls))
        self.assertTrue(all(s['status']=='in_progress' and s['owner'] for s in session.harness.team.list_state()['steps']))
        self.assertTrue(session.harness._team_trial_state['blocked'])
        before=self.counts(model)
        self.assertEqual('TEAM_TRIAL_BLOCKED',self.invoke(session)['status'])
        self.assertEqual(before,self.counts(model))
        self.assertEqual([],self.report(result)['summary_ledger'])

    def test_scaffold_or_parent_snapshot_save_failure_does_not_request_summary_or_retry(self):
        with self.subTest(failure='scaffold_before_any_provider'):
            session,model=self.session()
            with patch.object(team_trial,'_write_report',side_effect=OSError(PRIVATE_ERROR)):
                result=self.invoke(session)
            self.assertEqual('TEAM_TRIAL_STORE_FAILED',result['status'])
            self.assertEqual((0,0,0),self.counts(model))
            self.assertEqual(0,session.harness.team.list_state()['accepted_assignments'])
            self.assertNotIn(PRIVATE_ERROR,json.dumps(result))
        with self.subTest(failure='parent_snapshot_after_workers_started'):
            session,model=self.session()
            original=(session.harness.model,deepcopy(session.harness.tools),session.harness.max_steps,session.harness.system)
            with patch.object(session.harness,'snapshot',side_effect=OSError(PRIVATE_ERROR)):
                result=self.invoke(session)
            self.assertEqual('TEAM_TRIAL_DELIVERY_FAILED',result['status'])
            self.assertEqual(0,len(model.summary_calls))
            self.assertTrue(all(s['status']=='in_progress' for s in session.harness.team.list_state()['steps']))
            self.assert_adapter_restored(session.harness,original)
            self.assertTrue(session.harness._team_trial_state['blocked'])
            self.assertNotIn(PRIVATE_ERROR,json.dumps(self.report(result),ensure_ascii=False))

        with self.subTest(failure='final_report_after_actual_summary'):
            session,model=self.session();h=session.harness
            original_writer=team_trial._write_report
            original_adapter=(h.model,deepcopy(h.tools),h.max_steps,h.system)
            def fail_final(path,value):
                if value.get('status')=='completed':raise OSError(PRIVATE_ERROR)
                return original_writer(path,value)
            with patch.object(team_trial,'_write_report',side_effect=fail_final):
                result=self.invoke(session)
            self.assertEqual('TEAM_TRIAL_STORE_FAILED',result['status'])
            self.assertFalse(result['ok']);self.assertIsNone(result['report_path']);self.assertEqual('',result['answer'])
            self.assertEqual((4,1,0),self.counts(model))
            self.assertEqual(5,result['metrics']['total_provider_attempts'])
            self.assertTrue(all(s['status']=='completed' and s['completion_proof'] for s in h.team.list_state()['steps']))
            self.assert_adapter_restored(h,original_adapter)
            saved=json.loads((h.context.session_dir/'team-trial/round-1.json').read_text(encoding='utf-8'))
            self.assertNotEqual('completed',saved['status'])
            before=self.counts(model)
            self.assertEqual('TEAM_TRIAL_BLOCKED',self.invoke(session)['status'])
            self.assertEqual(before,self.counts(model))
            self.assertNotIn(PRIVATE_ERROR,json.dumps(result,ensure_ascii=False))

    def test_summary_exception_preserves_completed_cards_and_restores_normal_adapter(self):
        session,model=self.session(model=TrialModel(summary_error=TimeoutError(PRIVATE_ERROR)))
        h=session.harness;original=(h.model,deepcopy(h.tools),h.max_steps,h.system)
        result=self.invoke(session)
        self.assertEqual('TEAM_TRIAL_SUMMARY_FAILED',result['status'])
        self.assertEqual((4,1,0),self.counts(model))
        self.assertTrue(all(s['status']=='completed' and s['completion_proof'] for s in h.team.list_state()['steps']))
        self.assert_adapter_restored(h,original)
        report=self.report(result)
        self.assertEqual(1,len(report['summary_ledger']))
        self.assertFalse(result['metrics']['usage_complete'])
        self.assertEqual(1,result['metrics']['summary_provider_attempts'])
        self.assertNotIn(PRIVATE_ERROR,json.dumps(report,ensure_ascii=False))
        before=self.counts(model)
        self.assertEqual('TEAM_TRIAL_BLOCKED',self.invoke(session)['status'])
        self.assertEqual(before,self.counts(model))

    def test_normal_team_assignments_share_original_six_job_budget_with_trials(self):
        session,model=self.session();h=session.harness
        for number in range(2):
            model.normal_replies.extend([[call('normal_log_'+str(number),'start_team_task',role='logistics',resource_id='A1001',instruction='核对本次物流'),
                                          call('normal_policy_'+str(number),'start_team_task',role='policy',resource_id='returns',instruction='核对政策')],
                                         [text('普通团队调查报告已收到。')]])
            h.turn('普通自然需求：核对A1001物流和现有规则。')
        self.assertEqual(4,h.team.list_state()['accepted_assignments'])
        result=self.invoke(session);self.assertTrue(result['ok'],result)
        self.assertEqual(6,h.team.list_state()['accepted_assignments'])
        before=self.counts(model)
        self.assertEqual('TEAM_LIMIT_REACHED',self.invoke(session)['status'])
        self.assertEqual(before,self.counts(model))
        self.assertEqual(4,result['metrics']['child_provider_attempts'])
        self.assertEqual(1,result['metrics']['summary_provider_attempts'])

    def test_previous_completed_cards_cannot_stand_in_for_current_round_delivery(self):
        session,model=self.session();self.assertTrue(self.invoke(session)['ok'])
        h=session.harness;previous={s['step_id'] for s in h.team.list_state()['steps']}
        with patch.object(h,'_collect_team_results',return_value=True):
            result=self.invoke(session)
        self.assertEqual('TEAM_TRIAL_EVIDENCE_FAILED',result['status'])
        self.assertEqual(1,len(model.summary_calls))
        report=self.report(result)
        self.assertFalse(previous & {j['step_id'] for j in report['jobs']})
        self.assertTrue(all(s['status']=='in_progress' for s in report['steps']))
        self.assertEqual([],report['summary_ledger'])

    def test_operation_lock_is_held_through_summary_and_relogin_gets_fresh_trial_state(self):
        session,model=self.session()
        class ObservedLock:
            def __init__(self):self.lock=threading.RLock();self.held=False;self.entries=0
            def __enter__(self):self.lock.acquire();self.held=True;self.entries+=1;return self
            def __exit__(self,*args):self.held=False;self.lock.release()
        observed=ObservedLock();session._operation_lock=observed
        model.summary_observer=lambda:self.assertTrue(observed.held)
        old=session.harness
        self.assertTrue(self.invoke(session)['ok'])
        self.assertEqual(1,observed.entries);self.assertFalse(observed.held)
        self.assertTrue(session.login('alice','local-secret',self.auth))
        self.assertTrue(old.team.list_state()['closed'])
        self.assertFalse(hasattr(session.harness,'_team_trial_state'))
        self.assertTrue(self.invoke(session)['ok'])
        self.assertEqual(1,session.harness._team_trial_state['rounds'])
        self.assertEqual(2,session.harness.team.list_state()['accepted_assignments'])

    def test_post_start_audit_failure_retains_started_job_and_does_not_report_zero_child_attempts(self):
        session,model=self.session();h=session.harness;original=h._write_audit
        def fail_assignment(event,payload):
            if event=='ProgramTeamAssignment':raise OSError(PRIVATE_ERROR)
            return original(event,payload)
        with patch.object(h,'_write_audit',side_effect=fail_assignment):
            result=self.invoke(session)
        self.assertEqual('TEAM_TRIAL_START_FAILED',result['status'])
        self.assertEqual(0,len(model.summary_calls))
        report=self.report(result)
        self.assertEqual(1,len(report['jobs']))
        self.assertEqual(1,h.team.list_state()['accepted_assignments'])
        self.assertNotEqual(0,result['metrics']['child_provider_attempts'])
        self.assertNotIn(PRIVATE_ERROR,json.dumps(report,ensure_ascii=False))
        self.assertTrue(h._team_trial_state['blocked'])


if __name__=='__main__':unittest.main(verbosity=2)
