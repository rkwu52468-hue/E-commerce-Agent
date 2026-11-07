"""Dispatch/secret-boundary checks for the opt-in two-request planning trial."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import app

ROOT = Path(__file__).resolve().parents[1]


class BackgroundPlanCLITests(unittest.TestCase):
    def test_dry_run_dispatches_to_trial_before_normal_live_key_or_model(self):
        with patch('sys.argv', ['app.py', '--live-background-plan', '--dry-run']), \
             patch.object(app, 'run_plan_trial', return_value={'real_api_calls': 0}) as trial, \
             patch.object(app, 'read_secret', side_effect=AssertionError('No normal key read')) as secret, \
             patch.object(app, 'AnthropicModel', side_effect=AssertionError('No normal SDK')) as model:
            app.main()
        trial.assert_called_once_with(ROOT, dry_run=True)
        secret.assert_not_called();model.assert_not_called()

    def test_trial_dispatches_with_false_preview_only_after_explicit_mode(self):
        with patch('sys.argv', ['app.py', '--live-background-plan']), \
             patch.object(app, 'run_plan_trial') as trial:
            app.main()
        trial.assert_called_once_with(ROOT, dry_run=False)

    def test_feature_mix_and_other_identity_fail_before_trial_or_secret(self):
        cases = [['--live-background-plan', '--customer', 'bob'],
                 ['--live-background-plan', '--allow-background'],
                 ['--live-background-plan', '--allow-tasks'],
                 ['--live-background-plan', '--allow-memory'],
                 ['--live-background-plan', '--allow-summary'],
                 ['--live-background-plan', '--store', 'json'],
                 ['--live-background-plan', '--live']]
        for args in cases:
            with self.subTest(args=args), patch('sys.argv', ['app.py', *args]), \
                 patch.object(app, 'run_plan_trial') as trial, patch.object(app, 'read_secret') as secret, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:app.main()
                self.assertEqual(2, caught.exception.code)
                trial.assert_not_called();secret.assert_not_called()
