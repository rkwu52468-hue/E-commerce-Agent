"""s13 integration acceptance: genuine threads/shop, bounded scripted models."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch, Mock

import app
from customer_service.context import ContextManager, validate_tool_pairs
from customer_service.harness import Harness
from customer_service.live_session import LiveSession
from customer_service.models import AnthropicModel
from customer_service.shop import Shop, failure
from customer_service.skills import SkillLoader
from customer_service.storage import initialize_database
from customer_service.task_system import CustomerTaskStore
from customer_service.team_worker import TeamWorker, TeamModel
from test_subagent import MainModel, call, text

ROOT = Path(__file__).resolve().parents[1]


def notices(messages):
    rows=[]
    for msg in messages:
        if not isinstance(msg.get('content'), list): continue
        for block in msg['content']:
            value=block.get('text','')
            if block.get('type')=='text' and value.startswith('<team_notification>\n'):
                rows.append(json.loads(value[len('<team_notification>\n'):-len('\n</team_notification>')]))
    return rows


class RoleModel:
    mode='scripted-team-offline'
    def __init__(self, *, direct=False, wrong=False, forever=False, forbidden=False):
        self.calls=[]; self.counts={}; self.lock=threading.Lock()
        self.direct,self.wrong,self.forever,self.forbidden=direct,wrong,forever,forbidden
    def complete(self,messages,tools,system):
        prompt=next(m['content'] for m in reversed(messages) if m['role']=='user' and isinstance(m['content'],str)
                    and m['content'].startswith('当前团队步骤由程序登记：'))
        job=json.loads(prompt.split('：',1)[1].split('\n',1)[0]); request=job['request_id']
        with self.lock:
            self.counts[request]=self.counts.get(request,0)+1; number=self.counts[request]
            self.calls.append({'messages':deepcopy(messages),'tools':deepcopy(tools),'system':system,'job':deepcopy(job)})
        if self.direct or number>1 and not self.forever:
            content=[text('参考报告：本次资料已核对；最近物流可能延迟，具体以工具证据为准，不能承诺送达。')]
        else:
            name='get_tracking' if job['role']=='logistics' else 'get_policy'
            field='order_id' if job['role']=='logistics' else 'policy_id'
            resource='B2001' if self.wrong and field=='order_id' else job['resource_id']
            content=[call('child_'+request+'_'+str(number),name,**{field:resource})]
            if self.forbidden:content.insert(0,call('write_'+request+'_'+str(number),'write_file',path='forbidden.txt',content='no'))
        return {'content':content,'stop_reason':'tool_use' if any(x['type']=='tool_use' for x in content) else 'end_turn',
                'usage':{'input_tokens':80,'output_tokens':30,'cache_read_input_tokens':0,'cache_creation_input_tokens':0}}


class FakeAuth:
    def verify(self,name,password):return name if name in ('alice','bob') and password=='local-secret' else None


class TeamSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='team_session_',dir=ROOT/'runtime');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'data').mkdir()
        shutil.copyfile(ROOT/'data/shop.json',self.root/'data/shop.json')
        self.auth=FakeAuth()
    def clean(self,h):
        h.close()
        if h.team is not None:
            for w in list(h.team._workers.values()):
                if w.get('thread') is not None:w['thread'].join(2)
    def build(self,replies=(),*,child=None,enabled=True,customer='alice',persistent=False,wait=2):
        model=MainModel(replies);child=RoleModel() if child is None else child
        shop=Shop(customer,self.root/'data/shop.json');context=ContextManager(self.root/'runtime',customer or 'guest')
        store=CustomerTaskStore(self.root/'data/tasks',customer) if persistent and customer else None
        h=Harness(shop,context,model,enable_teams=enabled,team_model=child,team_wait_seconds=wait,
                  enable_tasks=persistent,task_store=store)
        self.addCleanup(lambda:self.clean(h));return h,model,child
    def start(self,h,role='logistics',resource='A1001'):
        out=h.call_tool('start_team_task',{'role':role,'resource_id':resource,'instruction':'核对本次事实并交回来源和缺口。'})
        self.assertTrue(out['ok'],out);return out['data']['step']['step_id']
    def steps(self,h):return h.team.list_state()['steps']
    def collect(self,h):
        with redirect_stdout(io.StringIO()):self.assertTrue(h._collect_team_results())
    def two_starts(self,suffix=""):
        # IDs stay unique across all turns in one provider conversation.
        return [call('log'+suffix,'start_team_task',role='logistics',resource_id='A1001',instruction='核对物流'),
                call('pol'+suffix,'start_team_task',role='policy',resource_id='returns',instruction='核对现有退货规则')]

    def test_default_off_and_guest_refusal_without_worker_requests(self):
        h,m,c=self.build(enabled=False)
        self.assertNotIn('start_team_task',{x['name'] for x in h.tools})
        self.assertEqual('TEAMS_DISABLED',h.call_tool('start_team_task',{})['error']['code'])
        h,m,c=self.build(customer=None)
        self.assertEqual('AUTHENTICATION_REQUIRED',h.call_tool('start_team_task',
            {'role':'logistics','resource_id':'A1001','instruction':'查询'})['error']['code'])
        self.assertEqual([],c.calls);self.assertEqual([],h.team.list_state()['workers'])

    def test_foreign_order_rejected_before_any_worker_or_provider_call(self):
        h,m,c=self.build()
        out=h.call_tool('start_team_task',{'role':'logistics','resource_id':'B2001','instruction':'查询'})
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND',out['error']['code'])
        self.assertEqual([],self.steps(h));self.assertEqual([],c.calls)

    def test_natural_request_runs_two_roles_and_closed_batch_precedes_reports(self):
        h,m,c=self.build([self.two_starts(),[text('已汇总事实和缺口，整体未收货问题仍需跟进。')]])
        query='A1001还没收到，请核对物流和店铺规则，说明依据和缺口，只查询。'
        with redirect_stdout(io.StringIO()):h.turn(query)
        self.assertEqual(query,m.calls[0]['messages'][0]['content'])
        self.assertEqual(2,len(m.calls));validate_tool_pairs(h.messages)
        delivered=notices(m.calls[1]['messages']);results=[x for x in delivered if x['type']=='result']
        self.assertEqual(2,len(results));self.assertEqual({'logistics','policy'},{x['role'] for x in results})
        self.assertTrue(all(x['receipt']['evidence_ok'] for x in results))
        self.assertTrue(all(x['status']=='completed' for x in self.steps(h)))
        self.assertEqual(4,len(c.calls));self.assertIn('<current_team_state>',m.calls[1]['system'])
        rows=[json.loads(x) for x in (h.context.session_dir/'audit.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(x['event']=='TeamEvent' for x in rows))

    def test_step_success_does_not_modify_persistent_customer_case(self):
        h,m,c=self.build([[call('start','start_team_task',role='logistics',resource_id='A1001',instruction='核对本次物流')],[text('报告已交接，包裹仍未到。')]],persistent=True)
        case=h.call_tool('create_task',{'subject':'持续跟进A1001未收货问题'})['data']['task']['id']
        self.assertTrue(h.call_tool('claim_task',{'task_id':case})['ok'])
        path=self.root/'data/tasks/alice'/(case+'.json');before=path.read_bytes()
        with redirect_stdout(io.StringIO()):h.turn('核对本次物流即可')
        self.assertEqual('completed',self.steps(h)[0]['status']);self.assertEqual(before,path.read_bytes())
        self.assertEqual('in_progress',json.loads(path.read_text(encoding='utf-8'))['status'])

    def test_query_failure_keeps_card_and_explicit_next_turn_resume_reuses_context(self):
        h,m,c=self.build([[call('first','start_team_task',role='logistics',resource_id='A1001',instruction='核对物流')],[text('查询失败，保留资料缺口。')]])
        original=Shop.call
        def fail_tracking(shop,name,args):
            return failure('DEMO_TRACKING_UNAVAILABLE','模拟失败') if name=='get_tracking' else original(shop,name,args)
        with patch.object(Shop,'call',new=fail_tracking),redirect_stdout(io.StringIO()):h.turn('核对物流')
        card=self.steps(h)[0];self.assertEqual('in_progress',card['status']);self.assertEqual('logistics',card['owner'])
        runner=h.team._workers['logistics']['runner'];old_session=runner.context.session_dir.name
        self.assertEqual('TEAM_ROLE_BUSY',h.team.call('start_team_task',{'role':'logistics','resource_id':'A1001','instruction':'另开卡'})['error']['code'])
        m.replies.extend([[call('resume','resume_team_task',step_id=card['step_id'],instruction='现在重试物流核对')],[text('重试取得资料，报告已交接。')]])
        with redirect_stdout(io.StringIO()):h.turn('重试上次失败的物流步骤')
        current=self.steps(h)[0]
        self.assertEqual('completed',current['status']);self.assertEqual(card['step_id'],current['step_id'])
        self.assertNotEqual(card['request_id'],current['request_id']);self.assertGreater(current['work_version'],card['work_version'])
        self.assertIs(runner,h.team._workers['logistics']['runner']);self.assertEqual(old_session,runner.context.session_dir.name)
        self.assertEqual(4,len(c.calls))

    def test_worker_cannot_use_old_history_as_fresh_evidence(self):
        child=RoleModel();h,m,c=self.build(child=child)
        self.start(h);self.collect(h);self.assertEqual('completed',self.steps(h)[0]['status'])
        child.direct=True
        m.replies.extend([[call('again','start_team_task',role='logistics',resource_id='A1001',instruction='再核对一次')],[text('本次没有新的查询依据，保留进行中。')]])
        with redirect_stdout(io.StringIO()):h.turn('重新核对物流')
        self.assertEqual('in_progress',self.steps(h)[1]['status'])
        result=[x for x in notices(h.messages) if x['type']=='result'][-1]['receipt']
        self.assertFalse(result['evidence_ok']);self.assertEqual('TEAM_FRESH_EVIDENCE_REQUIRED',result['error_code'])

    def test_resource_and_tool_allowlists_are_enforced_beyond_schema(self):
        for mode in ('wrong','forbidden'):
            with self.subTest(mode=mode):
                child=RoleModel(wrong=mode=='wrong',forbidden=mode=='forbidden')
                h,m,c=self.build(child=child);self.start(h);self.collect(h)
                self.assertEqual('in_progress',self.steps(h)[0]['status'])
                receipt=next(x['receipt'] for x in notices(h.messages) if x['type']=='result')
                codes=[x['result'].get('error',{}).get('code') for x in receipt['evidence']]
                self.assertIn('TEAM_RESOURCE_MISMATCH' if mode=='wrong' else 'TEAM_TOOL_NOT_ALLOWED',codes)
                self.assertFalse(list(self.root.rglob('forbidden.txt')))
                self.clean(h)

    def test_each_assignment_stops_at_three_provider_attempts(self):
        h,m,c=self.build(child=RoleModel(forever=True));self.start(h);self.collect(h)
        self.assertEqual(3,len(c.calls));self.assertEqual('in_progress',self.steps(h)[0]['status'])
        receipt=next(x['receipt'] for x in notices(h.messages) if x['type']=='result')
        self.assertEqual(3,len(receipt['provider_ledger']));self.assertFalse(receipt['ok'])

    def test_repeated_turns_reuse_workers_until_session_assignment_cap(self):
        h,m,c=self.build();identities=None
        for i in range(4):
            m.replies.extend([self.two_starts(str(i)),[text('仅说明本次结果。')]])
            with redirect_stdout(io.StringIO()):h.turn('综合核对订单物流和规则 '+str(i))
            if i==0:identities={role:id(w['runner']) for role,w in h.team._workers.items()}
            self.assertEqual(identities,{role:id(w['runner']) for role,w in h.team._workers.items()})
        self.assertEqual(6,h.team.list_state()['accepted_assignments']);self.assertEqual(12,len(c.calls))
        self.assertTrue(all(x['status']=='completed' for x in self.steps(h)))

    def test_same_role_and_s06_mixed_delegation_denied_in_one_turn(self):
        h,m,c=self.build();self.start(h)
        self.assertEqual('TEAM_TURN_LIMIT',h.call_tool('start_team_task',{'role':'logistics','resource_id':'A1001','instruction':'重复'})['error']['code'])
        self.assertEqual('DELEGATION_MODE_CONFLICT',h.call_tool('task',{'prompt':'重复调查'})['error']['code'])
        self.collect(h)
        h,m,c=self.build();h._delegations_this_turn=1
        self.assertEqual('DELEGATION_MODE_CONFLICT',h.call_tool('start_team_task',{'role':'policy','resource_id':'returns','instruction':'重复'})['error']['code'])
        self.assertEqual([],c.calls)

    def test_snapshot_failure_preserves_ready_and_staged_ids_then_no_duplicate_delivery(self):
        h,m,c=self.build();self.start(h)
        with patch.object(h,'snapshot',side_effect=OSError('disk')):
            self.assertFalse(h._collect_team_results())
        self.assertEqual('in_progress',self.steps(h)[0]['status']);self.assertTrue(h.team.peek_ready(0))
        count=len(notices(h.messages));self.assertGreater(count,0)
        self.collect(h)
        self.assertEqual(count,len(notices(h.messages)));self.assertEqual('completed',self.steps(h)[0]['status'])

    def test_post_ack_snapshot_failure_preserves_completed_state_and_truthful_receipt(self):
        h,m,c=self.build();self.start(h);original=h.snapshot;count=[0]
        def save():
            count[0]+=1
            if count[0]==2:raise OSError('post ack')
            return original()
        with patch.object(h,'snapshot',side_effect=save):self.assertFalse(h._collect_team_results())
        self.assertEqual('completed',self.steps(h)[0]['status']);self.assertFalse(h.team.peek_ready(0))
        self.assertIn('已保存并完成确认',h._team_delivery_notice)

    def test_ready_needs_parent_snapshot_before_completion(self):
        h,m,c=self.build();self.start(h)
        ready=h.team.peek_ready(2);self.assertTrue(ready)
        self.assertEqual('in_progress',self.steps(h)[0]['status'])
        self.collect(h)
        stored=json.loads((h.context.session_dir/'history.json').read_text(encoding='utf-8'))
        self.assertTrue(notices(stored['messages']));self.assertEqual('completed',stored['team_state']['steps'][0]['status'])

    def test_identity_tampered_notification_is_not_injected(self):
        h,m,c=self.build();self.start(h);ready=h.team.peek_ready(2)
        with h.team._lock:h.team._ready[ready[0]['envelope_id']]['customer_id']='bob'
        self.assertFalse(h._collect_team_results());self.assertFalse(notices(h.messages));self.assertTrue(h.team.list_state()['closed'])

    def test_login_logout_even_same_alice_does_not_reuse_old_workers(self):
        s=LiveSession(self.root,RoleModel(),store='json',allow_teams=True);self.addCleanup(s.close)
        self.assertTrue(s.login('alice','local-secret',self.auth));old=s.harness;self.addCleanup(lambda:self.clean(old))
        self.start(old);self.collect(old);old_worker=old.team._workers['logistics']['runner']
        self.assertTrue(s.login('alice','local-secret',self.auth));self.addCleanup(lambda:self.clean(s.harness))
        self.assertTrue(old.team.list_state()['closed']);self.assertEqual([],s.harness.messages)
        self.assertEqual([],s.harness.team.list_state()['steps'])
        s.logout();self.assertIsNone(s.customer_id);self.assertTrue(old_worker.closed.is_set())

    def test_team_envelope_stays_durable_when_later_parent_model_fails(self):
        h,m,c=self.build([[call('start','start_team_task',role='logistics',resource_id='A1001',instruction='查询')],RuntimeError('parent failed')])
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):h.turn('查询物流')
        self.assertEqual(2,len(c.calls));self.assertEqual('completed',self.steps(h)[0]['status'])
        stored=json.loads((h.context.session_dir/'history.json').read_text(encoding='utf-8'))
        self.assertTrue(notices(stored['messages']));validate_tool_pairs(stored['messages'])

    def test_shutdown_request_is_followed_by_distinct_ack_and_stopped(self):
        h,m,c=self.build();self.start(h);self.collect(h)
        out=h.call_tool('shutdown_teammate',{'role':'logistics'});self.assertTrue(out['ok']);self.assertEqual('REQUESTED',out['data']['status'])
        self.collect(h);events=notices(h.messages)
        self.assertTrue(any(x['type']=='shutdown_ack' for x in events));self.assertTrue(any(x['type']=='stopped' for x in events))

    def test_provider_adapter_has_real_call_output_timeout_and_thinking_bounds(self):
        model=AnthropicModel.__new__(AnthropicModel);model.client=Mock();model.model='fake'
        response=Mock();response.content=[];response.stop_reason='end_turn';response.usage.model_dump.return_value={'output_tokens':0}
        model.client.messages.create.return_value=response
        model.complete_teammate([],[],'rules');params=model.client.messages.create.call_args.kwargs
        self.assertEqual(1024,params['max_tokens']);self.assertEqual(20,params['timeout'])
        self.assertEqual({'type':'disabled'},params['extra_body']['thinking'])

    def test_cli_flag_requires_live_before_reading_credentials(self):
        with patch('sys.argv',['app.py','--allow-teams']),patch.object(app,'read_secret',side_effect=AssertionError),redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:app.main()
        self.assertEqual(2,e.exception.code)

    def test_cli_passes_flag_and_finally_closes_without_provider_call(self):
        shutil.copyfile(ROOT/'config.json',self.root/'config.json');sessions=[];model=MainModel([])
        def factory(*args,**kwargs):
            value=LiveSession(*args,**kwargs);sessions.append(value);return value
        with patch('sys.argv',['app.py','--live','--store','json','--allow-teams']),patch.object(app,'ROOT',self.root), \
             patch.object(app,'read_secret',return_value='offline-fake-key'),patch.dict('os.environ',{'DEEPSEEK_API_KEY':''}), \
             patch.object(app,'AnthropicModel',return_value=model),patch.object(app,'AuthStore'), \
             patch.object(app,'LiveSession',side_effect=factory),patch('builtins.input',side_effect=['q']),redirect_stdout(io.StringIO()):app.main()
        self.assertTrue(sessions[0].allow_teams);self.assertTrue(sessions[0].harness.team.list_state()['closed'])
        self.assertEqual([],model.calls);self.assertFalse((self.root/'data/auth').exists())

    def test_full_input_cap_denies_before_provider_call_and_records_known_zero(self):
        model=RoleModel();wrapped=TeamModel(model,threading.Event())
        with self.assertRaisesRegex(ValueError,'TEAM_INPUT_LIMIT'):
            wrapped.complete([{'role':'user','content':'x'*24001}],[],'rules')
        self.assertEqual([],model.calls);self.assertEqual([],wrapped.records)

    def test_failed_provider_attempt_has_known_call_count_but_unknown_usage(self):
        class FailingModel:
            mode='scripted-failure'
            def __init__(self):self.calls=0
            def complete(self,*args,**kwargs):
                self.calls+=1;raise OSError('simulated unavailable')
        child=FailingModel();h,m,c=self.build(child=child);self.start(h);self.collect(h)
        receipt=next(x['receipt'] for x in notices(h.messages) if x['type']=='result')
        self.assertEqual(1,child.calls);self.assertTrue(receipt['provider_attempts_known'])
        self.assertEqual(1,receipt['provider_attempts']);self.assertFalse(receipt['usage_complete'])
        self.assertIsNone(receipt['provider_ledger'][0]['usage'])
        self.assertEqual('in_progress',self.steps(h)[0]['status'])

    def test_default_sqlite_login_path_runs_team_threads_without_modifying_database(self):
        database=self.root/'data/shop.db';initialize_database(database,self.root/'data/shop.json')
        before=hashlib.sha256(database.read_bytes()).hexdigest()
        parent=MainModel([self.two_starts(),[text('本次报告已交接，整体未收货仍需跟进。')]])
        child=RoleModel();session=LiveSession(self.root,parent,allow_teams=True)
        self.addCleanup(session.close);self.assertTrue(session.login('alice','local-secret',self.auth))
        h=session.harness;self.addCleanup(lambda:self.clean(h))
        h.team._factory=lambda role:TeamWorker(role,deepcopy(h.shop),h.context.session_dir,child,h.skills)
        with redirect_stdout(io.StringIO()):h.turn('核对A1001物流和现有规则')
        self.assertTrue(all(x['status']=='completed' for x in self.steps(h)))
        self.assertEqual('sqlite',h.shop.call('get_order',{'order_id':'A1001'})['storage'])
        self.assertEqual('ACCESS_DENIED_OR_NOT_FOUND',h.shop.call('get_order',{'order_id':'B2001'})['error']['code'])
        self.assertEqual(before,hashlib.sha256(database.read_bytes()).hexdigest())
        self.assertEqual(4,len(child.calls))
