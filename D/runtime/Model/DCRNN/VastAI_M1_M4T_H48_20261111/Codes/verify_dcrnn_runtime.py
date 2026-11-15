"""Small numerical DCRNN checks in an installed PyTorch environment.

Uses synthetic observations only: 3 nodes, 750 context channels, a 48-hour
encoder window, and the six-horizon/48-step autoregressive decoder. It checks
label-free inference, finite gradients, an interrupted/resumed two-epoch run,
optimizer/scheduler/RNG recovery, safe model-contract rejection, and Parquet
prediction export. It does not train the real 635-node experiment.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
HORIZONS = (3, 6, 12, 24, 36, 48)


def dependencies():
    missing = [name for name in ('torch', 'numpy', 'pandas', 'pyarrow')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit('Numerical verification was NOT executed. Missing packages: '
                         + ', '.join(missing)
                         + '. Run setup_vast.sh and retry in its DCRNN environment.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE/'vendor'))
    import numpy as np
    import pandas as pd
    import torch
    import dcrnn_model as model
    import neural_training as trainer
    from checkpoint_store import CheckpointStore, CheckpointContractMismatch
    return np, pd, torch, model, trainer, CheckpointStore, CheckpointContractMismatch


def synthetic_problem(np, pd, torch):
    rng = np.random.default_rng(2701)
    history = 48
    origins = 8
    values = rng.normal(0, 0.15, size=(90, 3)).astype(np.float32)
    observed = np.zeros_like(values)
    observed[2:4, 1] = 1
    values[observed.astype(bool)] = 0
    context = rng.normal(0, 0.1, size=(origins, 750)).astype(np.float32)
    context[:, 375:] = 0
    labels = rng.normal(0, 0.1, size=(origins, len(HORIZONS))).astype(np.float32)
    eligible = np.ones((origins, len(HORIZONS)), dtype=bool)
    eligible[0, 2] = False
    raw = np.expm1(labels.astype(float) * .12 + 2.5)
    raw[~eligible] = np.nan
    positions = history-1+3*np.arange(origins)
    splits = ['train']*4+['validation']*2+['test']*2
    bundle = {
        'nodes': pd.DataFrame({'station_key': ['KR:111121', 'KR:SYNTH1', 'CN:SYNTH2']}),
        'origins': pd.DataFrame({'origin_time_kst': pd.date_range('2021-01-04', periods=origins,
                                                                freq='3h', tz='Asia/Seoul'),
                                 'origin_split': splits}),
        'horizons': HORIZONS, 'lookback_hours': history,
        'pm_values': values.copy(), 'context_values': context[:, :375].copy(),
        'y_raw': raw, 'eligible': eligible, 'origin_positions': positions,
    }
    arrays = {'hourly_x': np.stack([values, observed], axis=-1),
              'context_x': context, 'y_scaled': labels, 'eligible': eligible}
    preprocessing = {'pm_mean': [2.5]*3, 'pm_std': [.12]*3}

    class SyntheticOriginDataset(torch.utils.data.Dataset):
        def __init__(self, active_bundle, active_arrays, ids, lookback_hours):
            self.bundle = active_bundle
            self.arrays = active_arrays
            self.ids = np.asarray(ids, dtype=int)
            self.lookback = lookback_hours
        def __len__(self):
            return len(self.ids)
        def __getitem__(self, index):
            item = int(self.ids[index])
            end = int(self.bundle['origin_positions'][item])+1
            return (torch.as_tensor(self.arrays['hourly_x'][end-self.lookback:end]),
                    torch.as_tensor(self.arrays['context_x'][item]),
                    torch.as_tensor(self.arrays['y_scaled'][item]),
                    torch.as_tensor(self.arrays['eligible'][item]),
                    torch.as_tensor(item, dtype=torch.long))

    def split_indices(active_bundle, split, require_any_label):
        keep = active_bundle['origins'].origin_split.eq(split).to_numpy()
        if require_any_label:
            keep &= active_bundle['eligible'].any(axis=1)
        return np.flatnonzero(keep)

    api = {'DCRNNOriginDataset': SyntheticOriginDataset,
           'split_origin_indices': split_indices}
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], np.float32)
    return bundle, arrays, preprocessing, api, adjacency


def sha_tree(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def suite_for(deps, device, workdir):
    np, pd, torch, model_module, trainer, Store, ContractMismatch = deps
    bundle, arrays, preprocessing, api, adjacency = synthetic_problem(np, pd, torch)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    selected_device = torch.device(device)
    spec = {'model': 'DCRNN_trajectory_directed_H48', 'graph': 'trajectory_directed'}

    def factory(graph, _spec, context_dim):
        return model_module.DCRNNMultiHorizon(graph, hidden_dim=4, num_layers=2,
                    context_dim=context_dim, horizons=HORIZONS, dropout=0.0,
                    diffusion_steps=2, use_curriculum_learning=True, cl_decay_steps=200.)

    config = {'seed': 11, 'mode': 'pilot', 'input_condition': 'M1',
              'max_epochs': 2, 'patience': 10, 'micro_batch_size': 2,
              'effective_batch_size': 4, 'eval_batch_size': 2,
              'optimizer': 'adam', 'learning_rate': .001, 'optimizer_epsilon': .001,
              'weight_decay': 0., 'scheduler': 'plateau', 'lr_patience': 3,
              'lr_gamma': .5, 'min_learning_rate': 1e-5, 'max_grad_norm': 5.,
              'loss': 'mae_original', 'device': device, 'use_amp': False,
              'data_signature': trainer.neural_data_signature(bundle),
              'resume': True, 'checkpoint_policy': {'snapshot_every_epochs': 5, 'keep_recent': 2}}

    def state_for(run):
        folder = run/'Models'/spec['model']/'seed11'
        contract = json.loads((folder/'contract.json').read_text(encoding='utf-8'))
        state, _ = Store(folder, snapshot_every=5, keep_recent=2).load_latest(contract)
        return folder, state

    def assert_nested_equal(test, left, right):
        if isinstance(left, torch.Tensor):
            test.assertTrue(torch.equal(left.cpu(), right.cpu()))
        elif isinstance(left, np.ndarray):
            test.assertTrue(np.array_equal(left, right, equal_nan=True))
        elif isinstance(left, dict):
            test.assertEqual(set(left), set(right))
            for key in left:
                assert_nested_equal(test, left[key], right[key])
        elif isinstance(left, (tuple, list)):
            test.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                assert_nested_equal(test, a, b)
        else:
            test.assertEqual(left, right)

    class NumericalChecks(unittest.TestCase):
        def tearDown(self):
            trainer.CHECKPOINT_EVENT_CALLBACK = None

        def test_01_750_context_forward_backward_and_sparse_teacher_forcing(self):
            trainer.set_neural_seed(11)
            model = factory(adjacency, spec, 750).to(selected_device)
            dataset = api['DCRNNOriginDataset'](bundle, arrays, [0, 1], 48)
            loader = torch.utils.data.DataLoader(dataset, batch_size=2)
            x, c, y, mask, _ = next(iter(loader))
            x, c, y, mask = (v.to(selected_device) for v in (x, c, y, mask))
            model.train()
            prediction = model.forward_train(x, c, y, mask, batches_seen=0)
            self.assertEqual(tuple(prediction.shape), (2, 6))
            self.assertTrue(bool(torch.isfinite(prediction).all()))
            sums = trainer.masked_loss_sums(prediction, y, mask, preprocessing, 'mae_original')
            (sums/mask.sum(0).clamp_min(1)).mean().backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertEqual(model.architecture_config['decoder_steps'], 48)
            self.assertEqual(model.architecture_config['horizons'],list(HORIZONS))
            self.assertEqual(model.architecture_config['cl_decay_steps'],200.)
            self.assertEqual(bundle['context_values'].shape,(8,375))
            self.assertEqual(model.num_layers, 2)
            self.assertEqual(model.context_dim, 750)
            model.eval()
            with self.assertRaises(RuntimeError):
                model.forward_train(x, c, y, mask, batches_seen=0)

        def test_02_forward_reverse_support_direction_and_probability(self):
            supports = model_module.diffusion_supports(adjacency)
            signal = torch.tensor([[[1.], [0.], [0.]]])
            reached = torch.matmul(supports[0], signal)
            self.assertGreater(float(reached[0, 1, 0]), 0)
            self.assertEqual(float(reached[0, 2, 0]), 0.)
            self.assertTrue(bool(torch.isfinite(supports).all()))
            self.assertAlmostEqual(model_module.scheduled_sampling_probability(0,200.),200/201)
            self.assertAlmostEqual(factory(adjacency,spec,750).sampling_probability(0),200/201)
            self.assertEqual(model_module.scheduled_sampling_probability(1e12), 0.)

        def test_03_inference_ignores_loader_future_targets(self):
            trainer.set_neural_seed(11)
            model = factory(adjacency, spec, 750).to(selected_device)
            ids = api['split_origin_indices'](bundle, 'validation', False)
            clean = trainer._loader(api, bundle, arrays, ids, 2, False, selected_device)
            poisoned_arrays = dict(arrays, y_scaled=np.full_like(arrays['y_scaled'], np.nan),
                                   eligible=np.zeros_like(arrays['eligible']))
            poisoned = trainer._loader(api, bundle, poisoned_arrays, ids, 2, False, selected_device)
            kwargs = (bundle, preprocessing, selected_device, spec['model'], 'M1', 11, 'synthetic', 'pilot')
            before, _ = trainer.predict_neural(model, clean, *kwargs)
            after, _ = trainer.predict_neural(model, poisoned, *kwargs)
            self.assertTrue(np.array_equal(before.y_hat.to_numpy(), after.y_hat.to_numpy()))
            self.assertEqual(len(before), 12)
            self.assertEqual(set(before.horizon),set(HORIZONS))

        def test_04_interruption_resume_matches_uninterrupted_optimizer_scheduler_rng(self):
            reference = workdir/'uninterrupted'
            reference.mkdir()
            trainer.train_one_neural_run(bundle, arrays, preprocessing, adjacency, spec,
                                         config, reference, factory, api)
            _, complete = state_for(reference)
            source = workdir/'interrupted'
            source.mkdir()

            def stop_after_first_commit(event):
                if event.get('epoch') == 1 and not event.get('final') and not event.get('resumed'):
                    raise KeyboardInterrupt('Synthetic epoch-boundary interruption')
            trainer.CHECKPOINT_EVENT_CALLBACK = stop_after_first_commit
            with self.assertRaises(KeyboardInterrupt):
                trainer.train_one_neural_run(bundle, arrays, preprocessing, adjacency, spec,
                                             config, source, factory, api)
            trainer.CHECKPOINT_EVENT_CALLBACK = None
            source_folder, partial = state_for(source)
            self.assertEqual(partial['epoch'], 0)
            source_before = sha_tree(source)
            resumed = workdir/'resumed_new_execution'
            destination_seed = resumed/'Models'/spec['model']/'seed11'
            Store(source_folder, 5, 2).copy_to(destination_seed)
            model, summary, predictions = trainer.train_one_neural_run(
                bundle, arrays, preprocessing, adjacency, spec, config, resumed, factory, api)
            _, restored = state_for(resumed)
            self.assertEqual(sha_tree(source), source_before)
            self.assertEqual(restored['epoch'], 1)
            for key in ('model', 'best_model', 'optimizer', 'scheduler', 'scaler', 'rng',
                        'epoch', 'best_epoch', 'best', 'bad_epochs', 'global_step'):
                assert_nested_equal(self, complete[key], restored[key])
            for actual, expected in zip(restored['history'], complete['history']):
                assert_nested_equal(self, {k: v for k, v in actual.items() if k != 'seconds'},
                                     {k: v for k, v in expected.items() if k != 'seconds'})
            self.assertEqual(summary['trained_epochs'], 2)
            self.assertEqual(summary['best_epoch'], complete['best_epoch'])
            self.assertEqual(summary['selection_horizon'],6)
            for row in restored['history']:
                self.assertEqual(row['selection_horizon'],6)
                self.assertEqual(row['validation_selection_score'],row['validation_rmse_h6'])
            output = trainer.export_neural_predictions([predictions], resumed, 'validation',
                                                       {'synthetic_verification': True})
            exported = pd.read_parquet(output)
            self.assertEqual(len(exported), 12)
            self.assertEqual(set(exported.horizon),set(HORIZONS))
            self.assertFalse(exported.duplicated(['origin_time_kst', 'horizon', 'seed']).any())
            self.assertTrue(np.isfinite(exported.y_hat).all())
            self.assertTrue(output.with_suffix('.meta.json').is_file())
            self.assertTrue((resumed/'Exports/validation_predictions_ensemble_raw_mean.parquet').is_file())
            # Loading the selected model again does not pass future observations.
            loader = trainer._loader(api, bundle, arrays, [6, 7], 2, False, selected_device)
            test, _ = trainer.predict_neural(model, loader, bundle, preprocessing, selected_device,
                              spec['model'], 'M1', 11, 'synthetic', 'pilot')
            self.assertEqual(len(test), 12)
            self.assertTrue(test.split.eq('test').all())

        def test_05_original_h4_and_graphwavenet_checkpoint_contracts_are_rejected(self):
            # The payload is generated locally by this verifier, never loaded
            # from a downloaded external checkpoint.
            trainer.set_neural_seed(91)
            model = factory(adjacency, spec, 750)
            optimizer, _ = trainer.optimizer_and_scheduler(model, config)
            contract = {'version':trainer.TRAINING_VERSION,
                        'model': spec['model'], 'architecture': model.architecture_config,
                        'data': 'synthetic-model-family-contract-check'}
            weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            foreign = {'contract': dict(contract, model='GraphWaveNet_distance_fixed_plus_adaptive'),
                       'model': weights, 'best_model': copy.deepcopy(weights),
                       'optimizer': optimizer.state_dict(), 'scheduler': None, 'scaler': {},
                       'epoch': 0, 'best_epoch': 1, 'best': 1., 'bad_epochs': 0,
                       'global_step': 1, 'history': [{'epoch': 1, 'validation_rmse_h6': 1.,
                                                    'global_step': 1}],
                       'rng': trainer._rng_state()}
            legacy=model_module.DCRNNMultiHorizon(adjacency,hidden_dim=4,num_layers=2,
                context_dim=750,horizons=(3,6,12,24),cl_decay_steps=3000.)
            old_weights={k:v.detach().cpu().clone() for k,v in legacy.state_dict().items()}
            old_contract=dict(contract,version='seoul-dcrnn-m1-m4t-training-20261111-v1',
                model='DCRNN_trajectory_directed',architecture=legacy.architecture_config)
            old_payload=dict(foreign,contract=old_contract,model=old_weights,
                             best_model=copy.deepcopy(old_weights))
            for family,payload in (('gwn',foreign),('h4',old_payload)):
                with self.subTest(family=family):
                    store=Store(workdir/('foreign_'+family+'_store'),5,2)
                    store.save(payload)
                    with self.assertRaises(ContractMismatch):
                        store.load_latest(contract)

        def test_06_nonfinite_loss_is_reported_without_silent_cleaning(self):
            prediction = torch.full((1, len(HORIZONS)), 1e8, device=selected_device)
            target = torch.zeros_like(prediction)
            mask = torch.ones_like(prediction, dtype=torch.bool)
            with self.assertRaises(FloatingPointError):
                trainer.masked_loss_sums(prediction, target, mask, preprocessing, 'mae_original')
            # A missing label does not invalidate other observed horizons.
            target[0, 2] = float('nan')
            mask[0, 2] = False
            sums = trainer.masked_loss_sums(torch.zeros_like(prediction), target, mask,
                                             preprocessing, 'mae_original')
            self.assertTrue(bool(torch.isfinite(sums).all()))
            self.assertEqual(float(sums[2]), 0.)

        def test_07_optional_h24_checkpoint_selection_uses_selected_metric(self):
            root=workdir/'separate_h24_selection'
            root.mkdir()
            config24=dict(config,selection_horizon=24)
            scheduler_scores=[]
            original_step=trainer.step_scheduler
            def record_step(optimizer,scheduler,cfg,score):
                scheduler_scores.append(float(score))
                return original_step(optimizer,scheduler,cfg,score)
            trainer.step_scheduler=record_step
            try:
                _,summary,predictions=trainer.train_one_neural_run(bundle,arrays,preprocessing,
                    adjacency,spec,config24,root,factory,api)
            finally:
                trainer.step_scheduler=original_step
            _,saved=state_for(root)
            self.assertEqual(summary['selection_horizon'],24)
            self.assertEqual(saved['contract']['training']['selection_horizon'],24)
            self.assertEqual(scheduler_scores,[row['validation_selection_score'] for row in saved['history']])
            self.assertTrue(all(row['selection_horizon']==24 for row in saved['history']))
            best=min(saved['history'],key=lambda row:row['validation_selection_score'])
            self.assertEqual(summary['best_epoch'],best['epoch'])
            h24=trainer.prediction_metrics(predictions).set_index('horizon').loc[24,'RMSE']
            self.assertAlmostEqual(summary['best_validation_selection_score'],float(h24),places=10)

    return unittest.defaultTestLoader.loadTestsFromTestCase(NumericalChecks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    deps = dependencies()
    torch = deps[2]
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA verification requested but no CUDA GPU is available.')
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dcrnn_numerical_', dir=args.work_dir) as folder:
        result = unittest.TextTestRunner(verbosity=2).run(suite_for(deps, args.device, Path(folder)))
    report = {'status': 'passed' if result.wasSuccessful() else 'failed',
              'numerical_checks': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'device': args.device, 'torch': torch.__version__,
              'real_dataset_used': False, 'real_635_node_training_tested': False,
              'synthetic_3_node_750_context_model': True,
              'horizons':list(HORIZONS),'decoder_steps':48,'raw_context_channels':375,
              'two_origin_prediction_rows':12,'curriculum_decay_steps':200.,
              'original_h4_and_gwn_checkpoints_rejected':True,
              'default_h6_and_optional_h24_checkpoint_selection_checked':True,
              'interrupted_resume_checks_optimizer_scheduler_rng': True,
              'prediction_export_parquet_checked': True}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
