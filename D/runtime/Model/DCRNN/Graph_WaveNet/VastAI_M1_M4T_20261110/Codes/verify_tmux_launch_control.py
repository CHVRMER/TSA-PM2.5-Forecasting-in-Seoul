"""Mock tmux control-plane boundaries; no GPU or cloud resource is started."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
CODES = ROOT if (ROOT/'full_run_control.py').is_file() else ROOT / 'staging/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/Codes'
spec = importlib.util.spec_from_file_location('control', CODES / 'full_run_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
TMP = ROOT / 'test_tmp'
TMP.mkdir(exist_ok=True)


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TMP)
        self.root = Path(self.temp.name)
        self.config = self.root / 'default.json'
        self.config.write_text(json.dumps({'execution': {'conditions': ['M1'], 'seeds': [11]}}))
        self.args = argparse.Namespace(project_root=self.root, output_root=self.root / 'Result',
                                       config=self.config, conditions=None, seeds=None,
                                       final_test=False, fresh=False, detached=True)
        self.plan = {'action': 'launch', 'resume_source': None, 'condition_sources': {},
                     'checkpoint_states': {}}
        self.patches = [patch.object(control, 'is_linux_server', return_value=True),
                        patch.object(control, 'ensure_tmux'),
                        patch.object(control, 'build_resume_plan', return_value=self.plan),
                        patch.object(control, 'attach')]
        self.mocks = [item.start() for item in self.patches]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def result(self, stdout='', returncode=0):
        return subprocess.CompletedProcess([], returncode, stdout, '')

    def test_legacy_running_outside_tmux_does_not_launch(self):
        with patch.object(control, 'running_experiments', return_value=[{'pid': 7}]), \
             patch.object(control, 'matching_session', return_value=None), \
             patch.object(control, 'tmux_call') as tmux:
            self.assertEqual(control.start(self.args), 0)
            tmux.assert_not_called()

    def test_existing_running_session_attaches_without_launch(self):
        with patch.object(control, 'running_experiments', return_value=[{'pid': 7}]), \
             patch.object(control, 'matching_session', return_value='old-manual-session'), \
             patch.object(control, 'tmux_call') as tmux:
            self.assertEqual(control.start(self.args), 0)
            self.assertFalse(any(call.args[0] == 'new-session' for call in tmux.call_args_list))
            self.mocks[-1].assert_called_once_with('old-manual-session', True)

    def test_live_supervisor_before_runner_starts_is_not_duplicated(self):
        output = self.args.output_root.resolve()
        control.atomic_json(control.control_dir(output) / 'tmux_status.json', {
            'session': 'owned-suffixed-session', 'output_root': str(output), 'state': 'launching'})
        def tmux(*args, **kwargs):
            if args[0] == 'has-session':
                return self.result()
            if args[0] == 'show-options':
                return self.result(str(output))
            if args[0] == 'list-panes':
                return self.result('0')
            if args[0] == 'set-option':
                return self.result()
            self.fail('Unexpected tmux mutation: ' + str(args))
        with patch.object(control, 'running_experiments', return_value=[]), \
             patch.object(control, 'tmux_call', side_effect=tmux):
            self.assertEqual(control.start(self.args), 0)
            self.mocks[-1].assert_called_once_with('owned-suffixed-session', True)

    def test_new_launch_preserves_environment_and_uses_one_tmux(self):
        calls = []
        def tmux(*args, **kwargs):
            calls.append(args)
            return self.result(returncode=1 if args[0] == 'has-session' else 0)
        with patch.object(control, 'running_experiments', return_value=[]), \
             patch.object(control, 'tmux_call', side_effect=tmux), \
             patch.dict(os.environ, {'RCLONE_CONFIG': '/custom/drive.conf',
                                     'TSA_GWN_PYTHON': '/custom/python'}):
            self.assertEqual(control.start(self.args), 0)
        self.assertEqual(sum(call[0] == 'new-session' for call in calls), 1)
        request = next((self.args.output_root / 'Control/Invocations').glob('*/request.json'))
        value = control.read_json(request)
        self.assertEqual(value['environment']['RCLONE_CONFIG'], '/custom/drive.conf')
        self.assertEqual(value['environment']['TSA_GWN_PYTHON'], '/custom/python')
        self.assertNotIn('RCLONE_CONFIG_PASS', value['environment'])

    def test_completed_does_not_create_session(self):
        self.plan['action'] = 'completed'
        with patch.object(control, 'running_experiments', return_value=[]), \
             patch.object(control, 'tmux_call', return_value=self.result(returncode=1)) as tmux:
            self.assertEqual(control.start(self.args), 0)
            self.assertFalse(any(call.args[0] == 'new-session' for call in tmux.call_args_list))

    def test_runner_lease_prevents_duplicate_before_torch_import(self):
        args = ['--project-root', str(self.root), '--output-root', str(self.args.output_root), '--mode', 'full']
        with control.runner_lease(args) as first:
            self.assertTrue(first)
            with control.runner_lease(args) as second:
                self.assertFalse(second)
        with control.runner_lease(args) as third:
            self.assertTrue(third)

    def test_supervisor_passes_resume_and_current_environment(self):
        request_path = self.root / 'request.json'
        request = {'output_root': str(self.args.output_root), 'project_root': str(self.root),
                   'config': str(self.config), 'conditions': ['M1'], 'seeds': [11], 'fresh': False,
                   'final_test': True, 'status_path': str(self.root / 'status.json'),
                   'environment': {'RCLONE_CONFIG': '/custom/drive.conf'}}
        control.atomic_json(request_path, request)
        self.plan['resume_source'] = str(self.root / 'previous full')
        with patch.object(control, 'wait_for_training', return_value=(0, False)) as run:
            self.assertEqual(control.supervise(request_path), 0)
        invocation = run.call_args
        self.assertIn('--resume-source', invocation.args[0])
        self.assertIn('--final-test', invocation.args[0])
        self.assertEqual(invocation.args[2]['RCLONE_CONFIG'], '/custom/drive.conf')
        self.assertEqual(control.read_json(self.root / 'status.json')['state'], 'finished')


if __name__ == '__main__':
    unittest.main(verbosity=2)
