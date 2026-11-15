"""Independent stopping-limit revision checks using fake checkpoint stores."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CODES = HERE if (HERE / 'stopping_policy.py').is_file() else HERE / 'staging/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/Codes'
spec = importlib.util.spec_from_file_location('stopping_policy', CODES / 'stopping_policy.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
TEST_TMP = HERE / 'test_tmp'
TEST_TMP.mkdir(exist_ok=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, default=str) + '\n', encoding='utf-8')


class FakeStore:
    stores = {}

    def __init__(self, folder, snapshot_every=5, keep_recent=2):
        self.seed_folder = Path(folder)
        self.root = self.seed_folder / 'Checkpoints'
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_every = snapshot_every
        self.keep_recent = keep_recent
        self.states = {}
        self.rejected = set()
        self.saved = []
        self.copied_to = None
        FakeStore.stores[str(self.seed_folder)] = self

    def add(self, state):
        name = 'epoch_' + str(state['epoch'] + 1).zfill(6) + '_fixture'
        folder = self.root / name
        folder.mkdir(exist_ok=True)
        state = copy.deepcopy(state)
        self.states[name] = state
        write_json(folder / 'state.json', state)
        return folder

    def _candidates(self):
        return sorted((self.root / name for name in self.states), reverse=True)

    def _validate_directory(self, folder, expected_contract=None, **kwargs):
        name = Path(folder).name
        if name in self.rejected:
            raise ValueError('Synthetic damaged generation')
        state = copy.deepcopy(self.states[name])
        if expected_contract is not None and state['contract'] != expected_contract:
            raise RuntimeError('Synthetic checkpoint contract mismatch')
        manifest = {'generation': name, 'epoch': state['epoch'],
                    'completed_epochs': state['epoch'] + 1, 'best_epoch': state['best_epoch'],
                    'state_sha256': hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()}
        return state, manifest

    def load_latest(self, expected_contract=None, **kwargs):
        for folder in self._candidates():
            if folder.name in self.rejected:
                continue
            state, manifest = self._validate_directory(folder, expected_contract)
            return state, dict(manifest, generation_dir=str(folder), state_path=str(folder / 'state.json'))
        return None, {'status': 'fresh'}

    def save(self, payload, pin=False):
        self.saved.append(copy.deepcopy(payload))
        folder = self.add(payload)
        return {'generation': folder.name, 'completed_epochs': payload['epoch'] + 1,
                'state_path': str(folder / 'state.json'), 'pinned': pin}

    def copy_to(self, destination):
        self.copied_to = str(destination)
        target = FakeStore(destination, self.snapshot_every, self.keep_recent)
        for state in self.states.values():
            target.add(state)
        return {'source_files_modified': False, 'copied_generations': list(self.states)}


class StoppingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.root = Path(self.temporary.name)
        self.old_training = {'max_epochs': 100, 'patience': 20, 'learning_rate': 0.001,
                             'optimizer': 'adam', 'mode': 'full', 'seed': 11,
                             'library_versions': {'python': '3.12.0', 'torch': '2.x'}}
        self.new_training = dict(self.old_training, max_epochs=50, patience=10)
        self.source = FakeStore(self.root / 'source')
        self.destination = self.root / 'destination'
        self.curves = []

    def tearDown(self):
        self.temporary.cleanup()
        FakeStore.stores.clear()

    def state(self, epochs, bad=1, best_epoch=None):
        best_epoch = best_epoch or max(1, epochs - bad)
        history = [{'epoch': i, 'global_step': i * 8, 'validation_rmse_h6': float(100 - i),
                    'bad_epochs': bad if i == epochs else 0} for i in range(1, epochs + 1)]
        return {'contract': {'version': 'fixture', 'training': copy.deepcopy(self.old_training),
                             'data': 'fixed-data', 'graph': 'fixed-graph', 'preprocessing': 'fixed-preprocessing'},
                'epoch': epochs - 1, 'best_epoch': best_epoch,
                'best': history[best_epoch - 1]['validation_rmse_h6'], 'bad_epochs': bad,
                'model': {'weights': [float(epochs), 2.0]},
                'best_model': {'weights': [float(best_epoch), 2.0]},
                'optimizer': {'state': {'momentum': [1, 2]}, 'step': epochs * 8},
                'scheduler': None, 'scaler': {}, 'global_step': epochs * 8,
                'rng': {'python': 'python-rng', 'numpy': 'numpy-rng', 'torch': 'torch-rng', 'cuda': 'cuda-rng'},
                'history': history}

    def write_curve(self, path, history):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(str(row['epoch']) for row in history), encoding='utf-8')
        self.curves.append(copy.deepcopy(history))

    def import_state(self, config=None):
        return policy.import_seed_checkpoint(self.source, self.destination,
                    self.new_training if config is None else config,
                    FakeStore, write_json, self.write_curve)

    def source_snapshot(self):
        return {str(path.relative_to(self.source.seed_folder)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.source.seed_folder.rglob('*') if path.is_file()}

    def test_exact_limits_reduction_is_allowed_without_mutation(self):
        before = copy.deepcopy((self.old_training, self.new_training))
        changed = policy.stopping_reduction(self.old_training, self.new_training)
        self.assertEqual(set(changed), {'max_epochs', 'patience'})
        self.assertEqual((self.old_training, self.new_training), before)

    def test_same_training_has_no_revision(self):
        self.assertEqual(policy.stopping_reduction(self.old_training, copy.deepcopy(self.old_training)), {})

    def test_unchanged_policy_uses_normal_verified_copy(self):
        original = self.state(30)
        self.source.add(original)
        write_json(self.source.seed_folder / 'contract.json', original['contract'])
        write_json(self.source.seed_folder / 'preprocessing.json', {'mean': 1.0, 'std': 2.0})
        before = self.source_snapshot()
        self.import_state(self.old_training)
        self.assertEqual(self.source.copied_to, str(self.destination))
        copied = FakeStore.stores[str(self.destination)]
        self.assertEqual(list(copied.states.values()), [original])
        self.assertEqual(self.source_snapshot(), before)

    def test_empty_source_explicitly_records_no_completed_epoch(self):
        report = self.import_state()
        self.assertEqual(report['status'], 'fresh_restart_no_completed_epoch')
        self.assertFalse(self.destination.exists())

    def test_optimizer_or_learning_rate_changes_are_rejected(self):
        for key, value in [('learning_rate', 0.002), ('optimizer', 'adamw'),
                           ('seed', 22), ('library_versions', {'python': '3.13.0', 'torch': '2.x'})]:
            with self.subTest(key=key), self.assertRaises((ValueError, RuntimeError)):
                policy.stopping_reduction(self.old_training, dict(self.new_training, **{key: value}))

    def test_limit_increases_are_rejected(self):
        for config in [dict(self.new_training, max_epochs=101), dict(self.new_training, patience=21)]:
            with self.assertRaises((ValueError, RuntimeError)):
                policy.stopping_reduction(self.old_training, config)

    def test_nonpositive_boolean_or_fractional_limits_are_rejected(self):
        for key in ('max_epochs', 'patience'):
            for value in (0, -1, True, 10.5):
                with self.subTest(key=key, value=value), self.assertRaises((ValueError, RuntimeError, TypeError)):
                    policy.stopping_reduction(self.old_training, dict(self.new_training, **{key: value}))

    def test_revision_preserves_weights_optimizer_rng_history_and_source(self):
        source_state = self.state(30, bad=1, best_epoch=29)
        self.source.add(source_state)
        write_json(self.source.seed_folder / 'preprocessing.json', {'mean': 1.0, 'std': 2.0})
        before = self.source_snapshot()
        report = self.import_state()
        self.assertIsInstance(report, dict)
        target = FakeStore.stores[str(self.destination)]
        imported = target.saved[-1]
        self.assertEqual(imported['contract']['training'], self.new_training)
        for key in source_state.keys() - {'contract', '_checkpoint_store'}:
            self.assertEqual(imported[key], source_state[key], key)
        self.assertEqual(self.source_snapshot(), before)
        self.assertEqual((self.source.seed_folder / 'preprocessing.json').read_bytes(),
                         (self.destination / 'preprocessing.json').read_bytes())
        self.assertIsNone(self.source.copied_to, 'Changed contracts must not be mixed with copied old generations')

    def test_bad_epoch_counter_is_not_reset_when_already_at_new_patience(self):
        self.source.add(self.state(30, bad=12, best_epoch=18))
        self.import_state()
        imported = FakeStore.stores[str(self.destination)].saved[-1]
        self.assertEqual(imported['bad_epochs'], 12)

    def test_over_cap_selects_latest_verified_state_at_or_below_fifty(self):
        self.source.add(self.state(45, bad=2, best_epoch=43))
        self.source.add(self.state(50, bad=1, best_epoch=49))
        self.source.add(self.state(60, bad=2, best_epoch=58))
        before = self.source_snapshot()
        report = self.import_state()
        imported = FakeStore.stores[str(self.destination)].saved[-1]
        self.assertEqual(imported['epoch'] + 1, 50)
        self.assertEqual(imported['best_epoch'], 49)
        self.assertEqual(len(imported['history']), 50)
        self.assertEqual(self.source_snapshot(), before)
        revision = report['stopping_policy_revision']
        self.assertEqual(revision['source_completed_epochs'], 60)
        self.assertEqual(revision['selected_completed_epochs'], 50)
        self.assertTrue(revision['bounded_by_new_epoch_limit'])
        self.assertTrue(revision['patience_is_prospective_not_retroactive_replay'])
        if self.curves:
            self.assertEqual(len(self.curves[-1]), 50)

    def test_no_verified_checkpoint_below_new_cap_fails_explicitly(self):
        self.source.add(self.state(60))
        with self.assertRaises((ValueError, RuntimeError)):
            self.import_state()

    def test_damaged_fifty_snapshot_can_fall_back_to_forty_five(self):
        self.source.add(self.state(45))
        damaged = self.source.add(self.state(50))
        self.source.rejected.add(damaged.name)
        self.source.add(self.state(60))
        self.import_state()
        imported = FakeStore.stores[str(self.destination)].saved[-1]
        self.assertEqual(imported['epoch'] + 1, 45)


if __name__ == '__main__':
    unittest.main(verbosity=2)
