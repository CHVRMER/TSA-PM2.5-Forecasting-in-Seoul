"""Read-only structural checks for the preserved DCRNN model/trainer.

This script requires only the Python standard library. It does not claim that
PyTorch forward/backward or GPU training has been executed; run the separate
verify_dcrnn_runtime.py on the prepared server for those numerical checks.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
MODEL = HERE/'vendor/dcrnn_model.py'
TRAINER = HERE/'vendor/neural_training.py'
STORE = HERE/'checkpoint_store.py'
RUNNER = HERE/'run_experiments.py'
HORIZONS = (3, 6, 12, 24, 36, 48)
PRESERVED_MODEL_SHA256 = '480cc1bb9b88a90918b4dac81ccf060b58a1d90d644e79380a48ee94c84f1a7e'


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))


def function(node, name):
    return next(item for item in ast.walk(node)
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)


def call_names(node):
    return [ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)]


class DCRNNTrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tree(MODEL)
        cls.trainer = tree(TRAINER)
        cls.store = tree(STORE)
        cls.runner = tree(RUNNER)

    def test_preserved_model_source_is_exact(self):
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(), PRESERVED_MODEL_SHA256)

    def test_restore_and_trainer_versions_match(self):
        def version(path):
            for node in tree(path).body:
                if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAINING_VERSION' for t in node.targets):
                    return ast.literal_eval(node.value)
            raise AssertionError('Missing TRAINING_VERSION: '+str(path))
        self.assertEqual(version(TRAINER),version(HERE/'restore_verified_backup.py'))
        self.assertEqual(version(TRAINER),'seoul-dcrnn-m1-m4t-h48-training-20261111-v1')

    def test_runner_uses_six_horizon_override_and_h48_namespace(self):
        constants = {node.targets[0].id: ast.literal_eval(node.value)
            for node in self.runner.body if isinstance(node,ast.Assign)
            and isinstance(node.targets[0],ast.Name)
            and node.targets[0].id in {'HORIZONS','MODEL_NAME','EXPERIMENT_BUNDLE'}}
        self.assertEqual(constants['HORIZONS'],HORIZONS)
        self.assertEqual(max(constants['HORIZONS']),48)
        self.assertEqual(constants['MODEL_NAME'],'DCRNN_trajectory_directed_H48')
        self.assertEqual(constants['EXPERIMENT_BUNDLE'],'dcrnn-vast-m1-m4t-h48-v1')
        calls=[node for node in ast.walk(self.runner) if isinstance(node,ast.Call)
               and ast.unparse(node.func)=='DCRNNMultiHorizon']
        self.assertEqual(len(calls),1)
        keywords={keyword.arg:ast.unparse(keyword.value) for keyword in calls[0].keywords}
        self.assertEqual(keywords['horizons'],'HORIZONS')
        self.assertEqual(keywords['context_dim'],'context_dim')
        self.assertIn(None,keywords)  # Preserved backbone settings are explicit **settings.
        settings=function(self.runner,'model_settings')
        assignment=next(node for node in ast.walk(settings) if isinstance(node,ast.Assign)
            and any(isinstance(target,ast.Name) and target.id=='settings' for target in node.targets))
        defaults={keyword.arg:ast.literal_eval(keyword.value) for keyword in assignment.value.keywords}
        self.assertEqual(defaults['cl_decay_steps'],200.)
        self.assertEqual((defaults['hidden_dim'],defaults['num_layers'],defaults['diffusion_steps']),(64,2,2))

    def test_inference_forward_has_no_label_arguments(self):
        forward = function(self.model, 'forward')
        self.assertEqual([item.arg for item in forward.args.args], ['self', 'x', 'supports'])
        # The first forward belongs to DiffusionLinear; the public forecaster is
        # located by class, not by the incidental source order of helper cells.
        cls = next(n for n in self.model.body if isinstance(n, ast.ClassDef) and n.name == 'DCRNNMultiHorizon')
        public = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        self.assertEqual([item.arg for item in public.args.args], ['self', 'x', 'context'])
        calls = [n for n in ast.walk(public) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'self._run']
        self.assertEqual(len(calls), 1)
        self.assertEqual([ast.unparse(n) for n in calls[0].args], ['x', 'context'])
        self.assertFalse(calls[0].keywords)

    def test_original_stacked_encoder_decoder_defaults(self):
        cls = next(n for n in self.model.body if isinstance(n, ast.ClassDef) and n.name == 'DCRNNMultiHorizon')
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        positional = init.args.args[-len(init.args.defaults):]
        defaults = {key.arg: ast.literal_eval(value) for key, value in zip(positional, init.args.defaults)}
        self.assertEqual(defaults['hidden_dim'], 64)
        self.assertEqual(defaults['num_layers'], 2)
        self.assertEqual(defaults['diffusion_steps'], 2)
        self.assertEqual(defaults['horizons'], (3, 6, 12, 24))
        self.assertEqual(defaults['dropout'], 0.0)
        self.assertEqual(defaults['cl_decay_steps'], 3000.0)

    def test_train_only_sparse_sampling_rejects_eval(self):
        forward = function(self.model, 'forward_train')
        first_condition = next(n for n in forward.body if isinstance(n, ast.If))
        self.assertEqual(ast.unparse(first_condition.test), 'not self.training')
        self.assertTrue(any(isinstance(n, ast.Raise) for n in ast.walk(first_condition)))
        body = ast.unparse(forward)
        self.assertIn("targets.shape != (x.shape[0], len(self.horizons))", body)
        self.assertIn('not torch.isfinite(targets[mask]).all()', body)

    def test_decoder_feedback_only_uses_available_junggu_label(self):
        run = function(self.model, '_run')
        body = ast.unparse(run)
        self.assertIn('range(1, max(self.horizons) + 1)', body)
        self.assertIn('lead < max(self.horizons)', body)
        self.assertIn('next_input[:, 0, 0]', body)
        self.assertIn('torch.where(available, targets[:, column]', body)
        self.assertNotIn('targets[:, :,', body)

    def test_bidirectional_receiver_by_sender_support_orientation(self):
        body = ast.unparse(function(self.model, 'diffusion_supports'))
        self.assertIn('(adjacency, adjacency.T)', body)
        self.assertIn('probability.T.contiguous()', body)
        self.assertIn('torch.eye', body)

    def test_inference_loader_targets_are_not_model_inputs(self):
        inference = function(self.trainer, 'predict_neural')
        model_calls = [n for n in ast.walk(inference) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'model']
        self.assertEqual(len(model_calls), 1)
        self.assertEqual([ast.unparse(n) for n in model_calls[0].args], ['x.to(device)', 'context.to(device)'])
        self.assertNotIn('model.forward_train', call_names(inference))
        self.assertIn('model.eval', call_names(inference))

    def test_curriculum_entry_is_restricted_to_training_update(self):
        train = function(self.trainer, 'train_one_neural_run')
        update = function(train, 'update')
        branches = [n for n in ast.walk(update) if isinstance(n, ast.If) and 'use_curriculum_learning' in ast.unparse(n.test)]
        self.assertEqual(len(branches), 1)
        self.assertIn('model.forward_train', call_names(branches[0]))
        self.assertIn('model.train', call_names(train))

    def test_nonfinite_original_unit_loss_fails_without_clipping(self):
        loss = function(self.trainer, 'masked_loss_sums')
        body = ast.unparse(loss)
        self.assertIn('Nonfinite original-unit training loss; no silent clipping.', body)
        self.assertNotIn('torch.clamp', call_names(loss))
        self.assertNotIn('torch.nan_to_num', call_names(loss))
        self.assertIn('torch.where(mask, target, torch.zeros_like(target))', body)

    def test_full_mode_cannot_accept_pilot_sample_restrictions(self):
        body = ast.unparse(function(self.trainer, 'train_one_neural_run'))
        self.assertIn("mode == 'full'", body)
        self.assertIn('Full mode cannot use pilot sample limits.', body)

    def test_current_and_best_weights_optimizer_scheduler_rng_all_saved(self):
        train = function(self.trainer, 'train_one_neural_run')
        payload = next(n.value for n in ast.walk(train) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'payload' for t in n.targets))
        self.assertIsInstance(payload, ast.Dict)
        fields = {ast.literal_eval(key) for key in payload.keys}
        self.assertTrue({'contract','model','best_model','optimizer','scheduler','scaler','epoch',
                         'global_step','best','bad_epochs','history','rng','best_epoch'} <= fields)

    def test_saved_state_restores_optimizer_scheduler_and_rng(self):
        body = ast.unparse(function(self.trainer, 'train_one_neural_run'))
        self.assertIn("model.load_state_dict(state['model'])", body)
        self.assertIn("optimizer.load_state_dict(state['optimizer'])", body)
        self.assertIn("scheduler.load_state_dict(state['scheduler'])", body)
        self.assertIn("_restore_rng(state['rng'], device)", body)
        self.assertIn("store.load_latest(expected_contract=contract", body)

    def test_checkpoint_contract_mismatch_cannot_be_silent_fallback(self):
        load = function(self.store, 'load_latest')
        handlers = [n for n in ast.walk(load) if isinstance(n, ast.ExceptHandler)
                    and n.type is not None and ast.unparse(n.type) == 'CheckpointContractMismatch']
        self.assertEqual(len(handlers), 1)
        self.assertTrue(any(isinstance(n, ast.Raise) and n.exc is None for n in handlers[0].body))
        validated = ast.unparse(function(self.store, '_validate_directory'))
        self.assertLess(validated.index('manifest.get(\'contract_sha256\')'), validated.index('torch.load'))

    def test_validation_selection_defaults_h6_allows_declared_h24_and_test_is_explicit(self):
        train = ast.unparse(function(self.trainer, 'train_one_neural_run'))
        experiment = ast.unparse(function(self.trainer, '_run_neural_experiment'))
        self.assertIn("cfg.get('selection_horizon', 6)", train)
        self.assertIn("values.loc[values.horizon.eq(selection_horizon), 'RMSE']", train)
        self.assertIn('selection_horizon not in HORIZONS',train)
        self.assertIn('step_scheduler(optimizer, scheduler, cfg, score)',train)
        default=json.loads((HERE.parent/'configs/default.json').read_text(encoding='utf-8'))
        separate=json.loads((HERE.parent/'configs/selection24_separate.json').read_text(encoding='utf-8'))
        self.assertEqual(default['training']['selection_horizon'],6)
        self.assertEqual(separate['training']['selection_horizon'],24)
        self.assertEqual(default['execution']['forecast_horizons'],list(HORIZONS))
        self.assertEqual(default['model']['cl_decay_steps'],200.)
        self.assertIn("cfg.get('run_final_test', False)", experiment)
        self.assertIn('Only a full model may infer test.', experiment)
        self.assertIn('Pilot cannot evaluate test.', experiment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DCRNNTrainingContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'passed' if result.wasSuccessful() else 'failed',
              'static_checks': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors),
              'model_sha256': hashlib.sha256(MODEL.read_bytes()).hexdigest(),
              'runtime_horizons':list(HORIZONS),'runtime_decoder_steps':48,
              'vendor_original_defaults_preserved':True,
              'numerical_forward_backward_tested': False,
              'checkpoint_tensor_roundtrip_tested': False,
              'gpu_training_tested': False,
              'next_step': 'Run verify_dcrnn_runtime.py in the prepared PyTorch environment.'}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
