"""Independent signal/child-process regression tests; never signal the real console."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
CODES = (HERE if (HERE / 'full_run_control.py').is_file() else
         HERE / 'staging/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/Codes')
spec = importlib.util.spec_from_file_location('interrupt_control', CODES / 'full_run_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
TMP = HERE / 'test_tmp'
TMP.mkdir(exist_ok=True)


class FakeChild:
    pid = 500

    def __init__(self, actions):
        self.actions = iter(actions)
        self.wait_calls = []
        self.returncode = None

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        action = next(self.actions)
        if callable(action):
            action = action()
        if isinstance(action, BaseException):
            raise action
        self.returncode = action
        return action

    def poll(self):
        return self.returncode


class InterruptTests(unittest.TestCase):
    def setUp(self):
        self.handlers = {signal.SIGINT: Mock(name='old_sigint'),
                         signal.SIGTERM: Mock(name='old_sigterm')}
        self.original = dict(self.handlers)
        self.patches = [
            patch.object(control.signal, 'getsignal', side_effect=lambda sig: self.handlers[sig]),
            patch.object(control.signal, 'signal', side_effect=self.set_handler),
            patch.object(control.os, 'kill'),
            patch.object(control, 'running_experiments', return_value=[]),
            patch.object(control, 'process_ancestors', side_effect=lambda pid: {pid, 500} if pid == 600 else {pid, 999}),
        ]
        self.mocks = [p.start() for p in self.patches]
        self.kill = self.mocks[2]
        self.runners = self.mocks[3]

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def set_handler(self, sig, handler):
        old = self.handlers[sig]
        self.handlers[sig] = handler
        return old

    def interrupt(self, sig=signal.SIGINT):
        self.handlers[sig](sig, None)
        return subprocess.TimeoutExpired(['fake-training'], .5)

    def wait_for(self, child, callback=None):
        with patch.object(control.subprocess, 'Popen', return_value=child) as popen:
            result = control.wait_for_training(['bash', 'run_full.sh'], CODES, {'PATH': 'fake'},
                                               Path('/workspace/TSA'), on_stopping=callback)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(popen.call_args.kwargs['env'], {'PATH': 'fake'})
        self.assertEqual(self.handlers, self.original, 'Signal handlers must be restored')
        return result

    def test_success_restores_handlers_and_never_signals_any_process(self):
        self.assertEqual(self.wait_for(FakeChild([0])), (0, False))
        self.kill.assert_not_called()

    def test_nonzero_exit_without_interrupt_is_not_mislabelled_as_interrupted(self):
        self.assertEqual(self.wait_for(FakeChild([7])), (7, False))
        self.kill.assert_not_called()

    def test_ctrl_c_signals_only_owned_runner_not_other_runner_or_wrapper(self):
        self.runners.return_value = [{'pid': 600}, {'pid': 700}]
        callback = Mock()
        code, stopped = self.wait_for(FakeChild([self.interrupt, 130]), callback)
        self.assertTrue(stopped)
        self.assertEqual(code, 130)
        self.kill.assert_called_once_with(600, signal.SIGTERM)
        callback.assert_called_once()

    def test_interrupt_before_runner_spawn_stays_pending_until_owned_runner_appears(self):
        self.runners.side_effect = [[], [{'pid': 600}]]
        timeout = subprocess.TimeoutExpired(['fake-training'], .5)
        code, stopped = self.wait_for(FakeChild([self.interrupt, timeout, 130]))
        self.assertTrue(stopped)
        self.assertEqual(code, 130)
        self.kill.assert_called_once_with(600, signal.SIGTERM)

    def test_repeated_ctrl_c_does_not_redeliver_signal_or_force_kill(self):
        self.runners.return_value = [{'pid': 600}]
        callback = Mock()
        code, stopped = self.wait_for(FakeChild([self.interrupt, self.interrupt, 130]), callback)
        self.assertTrue(stopped)
        self.assertEqual(code, 130)
        self.kill.assert_called_once_with(600, signal.SIGTERM)
        callback.assert_called_once()

    def test_supervisor_sigterm_uses_the_same_graceful_runner_request(self):
        self.runners.return_value = [{'pid': 600}]
        result = self.wait_for(FakeChild([lambda: self.interrupt(signal.SIGTERM), 130]))
        self.assertTrue(result[1])
        self.kill.assert_called_once_with(600, signal.SIGTERM)


class StateAndPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TMP)
        self.root = Path(self.temp.name)
        self.settings = self.root / 'config.json'
        self.settings.write_text('{}', encoding='utf-8')
        self.status = self.root / 'status.json'
        self.request = self.root / 'request.json'
        control.atomic_json(self.request, {
            'config': str(self.settings), 'output_root': str(self.root / 'Result'),
            'project_root': str(self.root), 'conditions': ['M1'], 'seeds': [11],
            'final_test': False, 'fresh': False, 'status_path': str(self.status),
            'environment': {},
        })

    def tearDown(self):
        self.temp.cleanup()

    def test_interrupted_status_written_after_wait_finishes(self):
        plan = {'action': 'launch', 'resume_source': None, 'condition_sources': {}, 'checkpoint_states': {}}
        def wait(*args, **kwargs):
            self.assertEqual(control.read_json(self.status)['state'], 'running')
            callback = kwargs.get('on_stopping')
            if callback:
                callback()
            return 130, True
        with patch.object(control, 'build_resume_plan', return_value=plan), \
             patch.object(control, 'wait_for_training', side_effect=wait):
            self.assertEqual(control.supervise(self.request), 130)
        saved = control.read_json(self.status)
        self.assertEqual(saved['state'], 'interrupted')
        self.assertEqual(saved['exit_code'], 130)

    def test_interactive_worker_returns_to_bash_prompt_after_supervisor(self):
        with patch.object(control, 'supervise', return_value=130), \
             patch.object(control.sys.stdin, 'isatty', return_value=True), \
             patch.object(control.sys.stdout, 'isatty', return_value=True), \
             patch.object(control.os, 'execvpe') as execute:
            control.interactive_worker(self.request)
        self.assertEqual(execute.call_args.args[0], '/bin/bash')
        self.assertIn('-i', execute.call_args.args[1])

    def test_noninteractive_worker_does_not_spawn_a_shell(self):
        with patch.object(control, 'supervise', return_value=0), \
             patch.object(control.sys.stdin, 'isatty', return_value=False), \
             patch.object(control.sys.stdout, 'isatty', return_value=False), \
             patch.object(control.os, 'execvpe') as execute:
            self.assertEqual(control.interactive_worker(self.request), 0)
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
