"""Local diagnostic CLI is distinct from customer/model input."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import app
from customer_service.live_session import LiveSession
from test_subagent import MainModel
from test_team_session import FakeAuth

ROOT=Path(__file__).resolve().parents[1]


class TeamTrialCliTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='team_trial_cli_',dir=ROOT/'runtime')
        self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'data').mkdir();shutil.copyfile(ROOT/'data/shop.json',self.root/'data/shop.json')
        shutil.copyfile(ROOT/'config.json',self.root/'config.json')

    def run_cli(self, enabled, inputs, *, trial=None):
        sessions=[];model=MainModel([]);output=io.StringIO()
        def factory(*args,**kwargs):
            value=LiveSession(*args,**kwargs);sessions.append(value);return value
        argv=['app.py','--live','--store','json']+(['--allow-teams'] if enabled else [])
        with patch('sys.argv',argv),patch.object(app,'ROOT',self.root), \
             patch.object(app,'read_secret',return_value='local-secret'), \
             patch.dict('os.environ',{'DEEPSEEK_API_KEY':''}), \
             patch.object(app,'AnthropicModel',return_value=model), \
             patch.object(app,'AuthStore',return_value=FakeAuth()), \
             patch.object(app,'LiveSession',side_effect=factory), \
             patch('builtins.input',side_effect=inputs),redirect_stdout(output):
            if trial is None:app.main()
            else:
                with patch.object(app,'run_team_trial',return_value=trial) as invoke:
                    app.main();self.assertEqual(1,invoke.call_count)
                    self.assertEqual('alice',invoke.call_args.args[0].customer_id)
        return sessions[0],model,output.getvalue()

    def test_local_command_refuses_disabled_guest_and_parameterized_forms_without_provider_calls(self):
        for enabled,query,code in [(False,'/team-trial','TEAMS_DISABLED'),
                                   (True,'/team-trial','AUTHENTICATION_REQUIRED'),
                                   (True,'/team-trial A1001','未知本地命令')]:
            with self.subTest(query=query,enabled=enabled):
                session,model,output=self.run_cli(enabled,[query,'q'])
                self.assertEqual([],model.calls);self.assertEqual([],session.harness.messages)
                self.assertIn(code,output);self.assertTrue(session.harness._closed)

    def test_logged_in_diagnostic_is_dispatched_locally_and_not_forwarded_to_model(self):
        trial={'ok':True,'status':'completed','message':'program assigned',
               'answer':'参考汇总','metrics':{'total_provider_attempts':5},'report_path':'fixture-report.json'}
        session,model,output=self.run_cli(True,['/login','alice','/team-trial','q'],trial=trial)
        self.assertEqual([],model.calls);self.assertEqual([],session.harness.messages)
        self.assertIn('参考汇总',output);self.assertIn('fixture-report.json',output)
        self.assertTrue(session.harness.team.list_state()['closed'])
