"""Independent stdlib checks for automatic resume and real process locking."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TEST_TMP = HERE / 'test_tmp'
TEST_TMP.mkdir(exist_ok=True)
CODES = next(path for path in (
    HERE, HERE / 'Codes',
    HERE / 'staging/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112/Codes'
) if (path / 'full_run_control.py').is_file())
spec = importlib.util.spec_from_file_location('full_run_control', CODES / 'full_run_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n', encoding='utf-8')


class ResumeTests(unittest.TestCase):
    def test_previous_regional_checkpoint_rejected_before_training(self):
        batch=self.batch()
        run,_,_=self.checkpoint(batch)
        write_json(run/'experiment_config.json',{'training':{'input_condition':'M1','mode':'full'}})
        with self.assertRaisesRegex(RuntimeError,'different Graph WaveNet experiment bundle'):
            self.plan()

    def test_old_graph_wavenet_checkpoint_rejected_before_training(self):
        batch=self.batch()
        run,seed,_=self.checkpoint(batch)
        foreign=seed.parent.parent/'GraphWaveNet_distance_fixed_plus_adaptive'
        seed.parent.rename(foreign)
        with self.assertRaisesRegex(RuntimeError,'model family differs'):
            self.plan()

    def test_schema_mismatch_cannot_silently_reuse_full(self):
        batch=self.batch()
        self.checkpoint(batch)
        settings=control.read_json(batch/'resolved_configuration.json')
        settings['schema_version']='gwn-vast-m1-m4t-checkpoint-v1'
        write_json(batch/'resolved_configuration.json',settings)
        with self.assertRaisesRegex(RuntimeError,'different experiment schema'):
            self.plan()
    def test_stopping_changes_rejected_even_when_full_completed(self):
        batch = self.batch(status='passed')
        self.checkpoint(batch, 'M1')
        self.checkpoint(batch, 'M2')
        record = control.read_json(batch/'batch_manifest.json')
        record['conditions'] = {'M1': {'status': 'passed'}, 'M2': {'status': 'passed'}}
        score = batch/'scores';score.mkdir()
        for name in control.STANDARD_OUTPUTS:(score/name).write_bytes(b'ok')
        record['combined_scoring'] = {'validation': str(score)}
        write_json(batch/'batch_manifest.json',record)
        self.config['training'].update(max_epochs=25,patience=5)
        with self.assertRaisesRegex(RuntimeError,'different training configuration'):
            self.plan()

    def test_condition_run_parent_lineage_resolves_only_its_own_arm(self):
        old = self.batch('20261110_000001')
        run, _, _ = self.checkpoint(old, 'M1')
        new = self.batch('20261110_000002', parent=run)
        plan = self.plan()
        self.assertEqual(Path(plan['resume_source']), new)
        self.assertEqual(Path(plan['condition_sources']['M1']), run)
        self.assertNotIn('M2', plan['condition_sources'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.root = Path(self.temp.name)
        self.output = self.root / 'Result'
        self.config = {
            'schema_version': control.CONFIG_SCHEMA,
            'data': {'lookback_hours': 48, 'trajectory_radius_km': 75},
            'model': {'architecture_profile': 'project48', 'dropout': 0.3},
            'training': {'max_epochs': 75, 'patience': 10},
            'execution': {'conditions': control.CONDITIONS, 'seeds': [11]},
        }
        self.contract = {'version':control.TRAINING_VERSION,'model':control.MODEL,
                         'architecture':{'horizons':control.HORIZONS,'output_steps':48,'implementation':'gwn-seoul-h48-rf61-v1','receptive_field_hours':61},
                         'training':{'mode':'full','seed':11,'experiment_bundle':control.EXPERIMENT_SCHEMA},
                         'signature': 'same-data'}

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, name='20261110_000001', *, parent=None, status='interrupted',
              mode='full', validate_only=False):
        path = self.output / ('pilot' if mode == 'pilot' else 'full') / name
        record = {'mode': mode, 'status': status, 'validate_only': validate_only,
                  'resume_source': str(parent) if parent else None,
                  'seeds': [11], 'final_test': False, 'conditions': {}}
        write_json(path / 'batch_manifest.json', record)
        write_json(path / 'resolved_configuration.json', dict(self.config, seeds_resolved=[11]))
        return path

    def checkpoint(self, batch, condition='M1', *, epoch=1, run_name='run_001',
                   generation_suffix='a', contract=None):
        run = batch / 'Conditions' / condition / run_name
        write_json(run / 'experiment_config.json', {'training': {'input_condition': condition, 'mode': 'full',
                                                               'experiment_bundle':control.EXPERIMENT_SCHEMA}})
        seed = run / 'Models' / control.MODEL / 'seed11'
        expected = self.contract if contract is None else contract
        write_json(seed / 'contract.json', expected)
        name = f'epoch_{epoch:06d}_{generation_suffix}'
        folder = seed / 'Checkpoints' / name
        folder.mkdir(parents=True, exist_ok=True)
        state = ('synthetic trusted bytes ' + name).encode()
        (folder / 'state.pt').write_bytes(state)
        manifest = {'schema_version': control.SCHEMA, 'generation': name,
                    'epoch': epoch - 1, 'completed_epochs': epoch, 'best_epoch': epoch,
                    'best_score': 1.0, 'contract_sha256': control.canonical_hash(expected),
                    'state_bytes': len(state), 'state_sha256': hashlib.sha256(state).hexdigest()}
        write_json(folder / 'manifest.json', manifest)
        write_json(folder / 'COMMITTED.json', {'schema_version': control.SCHEMA,
                                             'generation': name,
                                             'manifest_sha256': control.sha256(folder / 'manifest.json')})
        return run, seed, folder

    def plan(self, **kwargs):
        return control.build_resume_plan(self.output, self.config, ['M1', 'M2'], [11], **kwargs)

    def snapshot(self):
        return {str(path.relative_to(self.root)): control.sha256(path)
                for path in self.root.rglob('*') if path.is_file()}

    def test_no_full_source_starts_fresh(self):
        result = self.plan()
        self.assertEqual(result['action'], 'launch')
        self.assertIsNone(result['resume_source'])

    def test_pilot_and_validation_only_never_selected(self):
        pilot = self.batch('20261110_999999', mode='pilot')
        self.checkpoint(pilot)
        val = self.batch('20261110_999998', validate_only=True)
        self.checkpoint(val)
        self.assertIsNone(self.plan()['resume_source'])

    def test_empty_newer_preflight_batch_skips_to_existing(self):
        old = self.batch('20261110_000001')
        self.checkpoint(old)
        self.batch('20261110_000002', status='failed')
        result = self.plan()
        self.assertEqual(Path(result['resume_source']), old)
        self.assertEqual(len(result['skipped_batches']), 1)

    def test_latest_batch_condition_lineage_preserves_later_arm(self):
        old = self.batch('20261110_000001')
        old_m1, _, _ = self.checkpoint(old, 'M1', epoch=5)
        old_m2, _, _ = self.checkpoint(old, 'M2', epoch=4)
        new = self.batch('20261110_000002', parent=old)
        new_m1, _, _ = self.checkpoint(new, 'M1', epoch=6)
        before = self.snapshot()
        result = self.plan()
        self.assertEqual(Path(result['resume_source']), new)
        self.assertEqual(Path(result['condition_sources']['M1']), new_m1)
        self.assertEqual(Path(result['condition_sources']['M2']), old_m2)
        self.assertNotEqual(Path(result['condition_sources']['M1']), old_m1)
        self.assertEqual(before, self.snapshot(), 'Discovery must never alter source results')

    def test_empty_latest_arm_falls_back_to_lineage(self):
        old = self.batch('20261110_000001')
        old_m1, _, _ = self.checkpoint(old)
        new = self.batch('20261110_000002', parent=old)
        (new / 'Conditions/M1/run_002/Models').mkdir(parents=True)
        result = self.plan()
        self.assertEqual(Path(result['condition_sources']['M1']), old_m1)

    def test_valid_previous_generation_survives_latest_corruption(self):
        batch = self.batch()
        _, seed, old = self.checkpoint(batch, epoch=4)
        _, _, newest = self.checkpoint(batch, epoch=5)
        (newest / 'state.pt').write_bytes(b'corrupt')
        result = control.validate_seed_checkpoint(seed)
        self.assertEqual(result['generation'], old.name)
        self.assertEqual(len(result['rejected_generations']), 1)

    def test_all_corrupt_latest_batch_does_not_fallback_other_batch(self):
        old = self.batch('20261110_000001')
        self.checkpoint(old)
        new = self.batch('20261110_000002')
        _, _, generation = self.checkpoint(new)
        (generation / 'state.pt').write_bytes(b'broken')
        with self.assertRaisesRegex(RuntimeError, 'no valid committed'):
            self.plan()

    def test_committed_contract_mismatch_is_fatal(self):
        batch = self.batch()
        _, seed, generation = self.checkpoint(batch)
        write_json(seed / 'contract.json', {'different': 'contract'})
        with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
            control.validate_seed_checkpoint(seed)

    def test_uncommitted_manifest_corruption_falls_back(self):
        batch = self.batch()
        _, seed, old = self.checkpoint(batch, epoch=4)
        _, _, new = self.checkpoint(batch, epoch=5)
        record = control.read_json(new / 'manifest.json')
        record['contract_sha256'] = 'corrupted contract bytes'
        write_json(new / 'manifest.json', record)
        result = control.validate_seed_checkpoint(seed)
        self.assertEqual(result['generation'], old.name)
        self.assertEqual(len(result['rejected_generations']), 1)

    def test_pending_and_legacy_states_do_not_silently_fresh_start(self):
        seed = self.root / 'seed11'
        (seed / 'Checkpoints/.pending_stale').mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, 'no valid committed'):
            control.validate_seed_checkpoint(seed)
        other = self.root / 'seed22'
        other.mkdir()
        (other / 'last.pt').write_bytes(b'legacy')
        with self.assertRaisesRegex(RuntimeError, 'no valid committed'):
            control.validate_seed_checkpoint(other)

    def test_missing_manifest_with_checkpoint_is_fatal(self):
        batch = self.batch()
        self.checkpoint(batch)
        (batch / 'batch_manifest.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'manifest is missing'):
            self.plan()

    def test_missing_lineage_source_is_fatal(self):
        batch = self.batch(parent=self.root / 'does_not_exist')
        with self.assertRaisesRegex(RuntimeError, 'resume batch is missing'):
            self.plan()

    def test_lineage_cycle_is_fatal(self):
        first = self.batch('20261110_000001')
        second = self.batch('20261110_000002', parent=first)
        record = control.read_json(first / 'batch_manifest.json')
        record['resume_source'] = str(second)
        write_json(first / 'batch_manifest.json', record)
        with self.assertRaisesRegex(RuntimeError, 'lineage cycle'):
            self.plan()

    def test_missing_absolute_old_root_relocated_by_batch_name(self):
        old = self.batch('20261110_000001')
        run, _, _ = self.checkpoint(old)
        new = self.batch('20261110_000002', parent=self.root / 'old_server' / old.name)
        self.assertEqual(control.resolve_condition_source(new, 'M1', 2), run)

    def test_real_training_configuration_change_rejected(self):
        batch = self.batch()
        self.checkpoint(batch)
        self.config['training']['max_epochs'] = 101
        with self.assertRaisesRegex(RuntimeError, 'different training configuration'):
            self.plan()

    def test_non_training_settings_may_change(self):
        batch = self.batch()
        self.checkpoint(batch)
        self.config['backup'] = {'remote': 'different-remote'}
        self.config['execution']['cache_root'] = '/somewhere/new'
        self.assertEqual(Path(self.plan()['resume_source']), batch)

    def test_seed_change_requires_intentional_new_experiment(self):
        batch = self.batch()
        self.checkpoint(batch)
        with self.assertRaisesRegex(RuntimeError, 'different seeds'):
            control.build_resume_plan(self.output, self.config, ['M1'], [22])

    def test_explicit_fresh_does_not_consult_corrupt_previous(self):
        batch = self.batch()
        _, _, generation = self.checkpoint(batch)
        (generation / 'state.pt').write_bytes(b'broken')
        self.assertIsNone(self.plan(fresh=True)['resume_source'])

    def test_completed_is_noop_until_final_test_requested(self):
        batch = self.batch(status='passed')
        for condition in ['M1', 'M2']:
            self.checkpoint(batch, condition)
        score = self.root / 'combined_scores'
        score.mkdir()
        for name in control.STANDARD_OUTPUTS:
            (score / name).write_text('fixture', encoding='utf-8')
        record = control.read_json(batch / 'batch_manifest.json')
        record['conditions'] = {c: {'status': 'passed'} for c in ['M1', 'M2']}
        record['combined_scoring'] = {'validation': str(score)}
        write_json(batch / 'batch_manifest.json', record)
        self.assertEqual(self.plan()['action'], 'completed')
        self.assertEqual(self.plan(final_test=True)['action'], 'launch')
        (score / control.STANDARD_OUTPUTS[0]).unlink()
        self.assertEqual(self.plan()['action'], 'launch')

    def test_direct_condition_source_must_match_condition(self):
        batch = self.batch()
        run, _, _ = self.checkpoint(batch)
        with self.assertRaisesRegex(ValueError, 'condition mismatch'):
            control.resolve_condition_source(run, 'M2', 1)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            control.resolve_condition_source(run, 'M1', 2)

    def test_only_this_bundle_current_runner_detected_not_graph_wavenet(self):
        proc = self.root / 'proc'
        project = self.root / 'TSA'
        script=str((CODES/'run_experiments.py').resolve())
        gwn=str(self.root/'Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/Codes/run_experiments.py')
        old_dcrnn=str(self.root/'Model/DCRNN/VastAI_M1_M4T_H48_20261111/Codes/run_experiments.py')
        for pid, args in [(101, ['python', script, '--project-root', str(project), '--mode', 'full']),
                          (102, ['python', script, '--project-root', str(project), '--mode', 'pilot']),
                          (103, ['python', script, '--project-root', str(self.root / 'Other')]),
                          (104, ['python', 'other_script.py', '--project-root', str(project)]),
                          (105, ['python', gwn, '--project-root', str(project), '--mode', 'full']),
                          (106, ['python','run_experiments.py','--project-root',str(project),'--mode','full']),
                          (107, ['python',old_dcrnn,'--project-root',str(project),'--mode','full'])]:
            directory = proc / str(pid)
            directory.mkdir(parents=True)
            (directory / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in args) + b'\0')
        found = control.running_experiments(project, proc)
        self.assertEqual({row['pid'] for row in found}, {101, 102})
        self.assertTrue(all(set(row) == {'pid', 'mode'} for row in found))

    def test_old_four_horizon_contract_rejected_even_in_renamed_model_folder(self):
        batch=self.batch()
        contract=copy.deepcopy(self.contract)
        contract['architecture'].update(horizons=[3,6,12,24],output_steps=24)
        self.checkpoint(batch,contract=contract)
        with self.assertRaisesRegex(RuntimeError,'H48 checkpoint contract mismatch'):
            self.plan()

    def test_old_four_horizon_version_rejected_even_with_six_labels(self):
        batch=self.batch()
        contract=copy.deepcopy(self.contract)
        contract['version']='seoul-gwn-training-v1.0.0'
        self.checkpoint(batch,contract=contract)
        with self.assertRaisesRegex(RuntimeError,'H48 checkpoint contract mismatch'):
            self.plan()


    def test_relative_runner_path_resolved_from_process_cwd(self):
        proc=self.root/'proc'; folder=proc/'101';cwd=folder/'cwd'
        cwd.mkdir(parents=True)
        (folder/'cmdline').write_bytes(b'python\0run_experiments.py\0--project-root\0'+str(self.root).encode()+b'\0--mode\0full\0')
        with patch.object(control,'HERE',cwd):
            found=control.running_experiments(self.root,proc)
        self.assertEqual([v['pid'] for v in found],[101])


class LockTests(unittest.TestCase):
    def test_lock_denies_other_process_and_releases_after_crash(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            path = Path(folder) / 'training.lock'
            code = (
                'import sys;sys.path.insert(0,sys.argv[1]);'
                'from full_run_control import FileLease;'
                'lease=FileLease(sys.argv[2]);'
                'ok=lease.acquire();print("ACQUIRED" if ok else "BLOCKED",flush=True);'
                'sys.stdin.read() if ok else None'
            )
            holder = subprocess.Popen([sys.executable, '-u', '-c', code, str(CODES), str(path)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(holder.stdout.readline().strip(), 'ACQUIRED')
                contender = subprocess.run([sys.executable, '-u', '-c', code, str(CODES), str(path)],
                                           input='', capture_output=True, text=True, timeout=10)
                self.assertEqual(contender.returncode, 0, contender.stderr)
                self.assertEqual(contender.stdout.strip(), 'BLOCKED')
                holder.kill()
                holder.communicate(timeout=10)
                lease = control.FileLease(path)
                self.assertTrue(lease.acquire(), 'OS must release lock after the owner dies')
                lease.release()
                self.assertTrue(path.exists(), 'Never remove/recreate a live lock inode')
            finally:
                if holder.poll() is None:
                    holder.kill()
                    holder.communicate(timeout=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
