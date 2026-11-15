"""Read-only Graph WaveNet H48 structural checks; no Torch/GPU is claimed.

The separate verify_gwn_runtime.py checks actual tensor calculations on server.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
MODEL=HERE/'vendor/graph_wavenet_model.py'
TRAINER=HERE/'vendor/neural_training.py'
RUNNER=HERE/'run_experiments.py'
STORE=HERE/'checkpoint_store.py'
HORIZONS=(3,6,12,24,36,48)

def tree(path):return ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))

def function(node,name):
    return next(n for n in ast.walk(node) if isinstance(n,ast.FunctionDef) and n.name==name)

def constant(path,name):
    for node in tree(path).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError('Missing constant '+name+': '+str(path))

class GraphWaveNetH48Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=tree(MODEL);cls.trainer=tree(TRAINER);cls.runner=tree(RUNNER);cls.store=tree(STORE)
        cls.forecaster=next(n for n in cls.model.body if isinstance(n,ast.ClassDef) and n.name=='GraphWaveNetMultiHorizon')
        cls.init=function(cls.forecaster,'__init__')
        cls.defaults={key.arg:ast.literal_eval(value) for key,value in zip(cls.init.args.kwonlyargs,cls.init.args.kw_defaults)}

    def test_model_six_horizon_direct_forecast_defaults(self):
        d=self.defaults
        self.assertEqual(d['horizons'],HORIZONS)
        self.assertEqual(d['output_steps'],48)
        self.assertEqual((d['blocks'],d['layers'],d['kernel_size']),(4,4,2))
        self.assertEqual(d['architecture_profile'],'project48')
        self.assertEqual((d['residual_channels'],d['dilation_channels'],d['skip_channels'],d['end_channels']),(32,32,256,512))
        self.assertEqual((d['dropout'],d['context_dropout'],d['embedding_dim']),(.3,.2,10))
        self.assertEqual(d['graph_mode'],'fixed_plus_adaptive')

    def test_rf61_uses_four_dilations_per_block(self):
        body=ast.unparse(self.init)
        self.assertIn('dilation = 2 ** layer',body)
        self.assertIn('range(blocks)',body)
        self.assertIn('range(layers)',body)
        self.assertIn('1 + (kernel_size - 1) * sum(dilations)',body)
        d=self.defaults
        self.assertEqual(1+(d['kernel_size']-1)*d['blocks']*(2**d['layers']-1),61)

    def test_padding_mask_is_one_and_pm_is_zero_without_changing_observations(self):
        body=ast.unparse(function(self.forecaster,'prepare_temporal_input'))
        self.assertIn('padding = z.new_zeros',body)
        self.assertIn('padding[:, 1, :, :] = 1.0',body)
        self.assertIn('torch.cat((padding, z), dim=-1)',body)
        self.assertNotIn('x[',body.replace('x.shape',''))
        self.assertEqual(self.defaults['padding_policy'],'scaled_pm_zero_missing_one')

    def test_context_dropout_is_in_head_after_activation(self):
        calls=[n for n in ast.walk(self.init) if isinstance(n,ast.Assign)
               and any(ast.unparse(t)=='self.context_head' for t in n.targets)]
        self.assertEqual(len(calls),1)
        body=ast.unparse(calls[0])
        self.assertIn('nn.Linear(context_dim, residual_channels), nn.ReLU(), nn.Dropout(context_dropout)',body)

    def test_no_teacher_forcing_or_future_target_interface(self):
        public=function(self.forecaster,'forward')
        self.assertEqual([a.arg for a in public.args.args],['self','x','context'])
        self.assertIsNone(public.args.kwarg)
        methods={n.name for n in self.forecaster.body if isinstance(n,ast.FunctionDef)}
        self.assertNotIn('forward_train',methods)
        self.assertNotIn('use_curriculum_learning',ast.unparse(self.forecaster))

    def test_forecast_reads_final_position_and_requested_hour_channels(self):
        body=ast.unparse(function(self.forecaster,'forward_all'))
        self.assertIn('self.forward_sequence(x)[..., -1]',body)
        public=ast.unparse(function(self.forecaster,'forward'))
        self.assertIn('[h - 1 for h in self.horizons]',public)
        self.assertIn('index_select(1, leads)',public)

    def test_graph_support_orientation_and_adaptive_softmax(self):
        message=ast.unparse(function(self.model,'wave_message'))
        self.assertIn("torch.einsum('bcnt,nm->bcmt', x, support)",message)
        body=ast.unparse(function(self.forecaster,'adaptive_adjacency'))
        self.assertIn('torch.softmax(torch.relu(self.nodevec1 @ self.nodevec2), dim=1)',body)

    def test_conv2d_gates_and_new_implementation_identity(self):
        body=ast.unparse(self.init)
        self.assertIn("'implementation': 'gwn-seoul-h48-rf61-v1'",body)
        self.assertNotIn('nn.Conv1d',body)
        self.assertIn("'context_dropout': float(context_dropout)",body)
        self.assertIn("'padding_policy': padding_policy",body)
        self.assertFalse((HERE/'vendor/dcrnn_model.py').exists())

    def test_runner_identity_and_explicit_horizon_output_overrides(self):
        self.assertEqual(constant(RUNNER,'HORIZONS'),HORIZONS)
        self.assertEqual(constant(RUNNER,'MODEL_NAME'),'GraphWaveNet_distance_fixed_plus_adaptive_H48')
        self.assertEqual(constant(RUNNER,'EXPERIMENT_BUNDLE'),'gwn-vast-m1-m4t-h48-v1')
        calls=[n for n in ast.walk(self.runner) if isinstance(n,ast.Call) and ast.unparse(n.func)=='GraphWaveNetMultiHorizon']
        self.assertEqual(len(calls),1)
        keys={k.arg:ast.unparse(k.value) for k in calls[0].keywords}
        self.assertEqual(keys['horizons'],'HORIZONS')
        self.assertEqual(keys['output_steps'],'48')
        self.assertEqual(keys['context_dim'],'context_dim')

    def test_train_restore_versions_match_only_new_graphwavenet(self):
        version=constant(TRAINER,'TRAINING_VERSION')
        self.assertEqual(version,'seoul-gwn-m1-m4t-h48-training-20261112-v1')
        self.assertEqual(version,constant(HERE/'restore_verified_backup.py','TRAINING_VERSION'))
        self.assertEqual(constant(TRAINER,'HORIZONS'),HORIZONS)

    def test_inference_uses_inputs_without_loader_targets(self):
        body=ast.unparse(function(self.trainer,'predict_neural'))
        self.assertIn('model.eval()',body)
        self.assertIn('model(x.to(device), context.to(device))',body)
        self.assertNotIn('forward_train',body)

    def test_model_optimizer_scheduler_rng_saved_and_restored(self):
        body=ast.unparse(function(self.trainer,'train_one_neural_run'))
        for text in ("model.load_state_dict(state['model'])","optimizer.load_state_dict(state['optimizer'])",
                     "scheduler.load_state_dict(state['scheduler'])","_restore_rng(state['rng'], device)",
                     "'best_model'","'bad_epochs'","'global_step'","'rng'"):
            self.assertIn(text,body)
        load=function(self.store,'load_latest')
        handlers=[n for n in ast.walk(load) if isinstance(n,ast.ExceptHandler) and n.type is not None
                  and ast.unparse(n.type)=='CheckpointContractMismatch']
        self.assertEqual(len(handlers),1)
        self.assertTrue(any(isinstance(n,ast.Raise) and n.exc is None for n in handlers[0].body))

    def test_validation_selection_and_nonfinite_errors_are_explicit(self):
        train=ast.unparse(function(self.trainer,'train_one_neural_run'))
        loss=ast.unparse(function(self.trainer,'masked_loss_sums'))
        self.assertIn("cfg.get('selection_horizon', 6)",train)
        self.assertIn('step_scheduler(optimizer, scheduler, cfg, score)',train)
        self.assertIn('Nonfinite original-unit training loss; no silent clipping.',loss)
        self.assertNotIn('torch.nan_to_num',loss)

    def test_configuration_75epochs_patience10_six_horizons_and_plateau(self):
        c=json.loads((HERE.parent/'configs/default.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(c['execution']['forecast_horizons'],list(HORIZONS))
        t=c['training'];m=c['model']
        self.assertEqual((t['max_epochs'],t['patience']),(75,10))
        self.assertEqual((t['learning_rate'],t['weight_decay'],t['optimizer_epsilon']),(.001,1e-4,1e-8))
        self.assertEqual((t['scheduler'],t['lr_gamma'],t['lr_patience'],t['min_learning_rate']),('plateau',.5,3,1e-5))
        self.assertEqual(t['selection_horizon'],6)
        self.assertEqual((m['blocks'],m['layers'],m['context_dropout']),(4,4,.2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GraphWaveNetH48Contracts))
    report={'status':'passed' if result.wasSuccessful() else 'failed','static_checks':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'model_sha256':hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        'horizons':list(HORIZONS),'receptive_field_hours':61,'output_steps':48,
        'numerical_forward_backward_tested':False,'checkpoint_tensor_roundtrip_tested':False,'gpu_training_tested':False,
        'next_step':'Run verify_gwn_runtime.py in the prepared server PyTorch environment.'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return 0 if result.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
