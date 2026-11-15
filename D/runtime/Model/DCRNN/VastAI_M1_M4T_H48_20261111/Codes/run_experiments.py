"""Run six frozen DCRNN information conditions on a persistent GPU-cloud disk.

The preserved model backbone is trained for six forecast horizons. The private
scorer retains the established output format. Every invocation creates a new
batch folder; the controller finds the latest compatible H48 checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import signal
import sys
import tempfile
import time
import traceback
from typing import Any
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TARGET = 'KR:111121'
CONDITIONS = ('M1', 'M2', 'M3', 'M4', 'M3T', 'M4T')
CONTRASTS = (('M2','M1'), ('M3','M2'), ('M3','M1'), ('M4','M3'), ('M4','M1'), ('M3T','M3'), ('M4T','M4'))
MODEL_NAME = 'DCRNN_trajectory_directed_H48'
EXPERIMENT_BUNDLE = 'dcrnn-vast-m1-m4t-h48-v1'
HORIZONS = (3,6,12,24,36,48)
STANDARD_OUTPUTS = ('metrics.csv','seed_metric_summary.csv','tests.csv','conditions.csv',
                    'score_rows.parquet','scoring_audit.json')


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_config(path: Path | None) -> dict:
    if path is None:
        path = HERE.parent/'configs/default.json'
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() in ('.yaml','.yml'):
        import yaml
        result = yaml.safe_load(text)
    else:
        result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('The experiment configuration must be an object/mapping.')
    if result.get('schema_version') != 'dcrnn-vast-m1-m4t-h48-checkpoint-v1':
        raise ValueError('Use this DCRNN M1-M4T bundle configuration, not a GWN/regional configuration.')
    if tuple(result.get('execution',{}).get('forecast_horizons',())) != HORIZONS:
        raise ValueError('This H48 package requires exactly3/6/12/24/36/48h.')
    return result


def normalized_tokens(values, allowed=None):
    result = [token.strip() for value in values for token in str(value).split(',') if token.strip()]
    if not result or len(set(result)) != len(result):
        raise ValueError('Declare a nonempty list without duplicate entries.')
    if allowed is not None and not set(result).issubset(allowed):
        raise ValueError('Unknown entries: ' + ', '.join(sorted(set(result)-set(allowed))))
    return result


def model_settings(config):
    model = dict(config.get('model', {}))
    profile = model.pop('architecture_profile', 'preserved_seq2seq')
    if profile != 'preserved_seq2seq':
        raise ValueError('This package preserves the stacked DCRNN seq2seq backbone.')
    settings = dict(hidden_dim=64, diffusion_steps=2, dropout=0., num_layers=2,
                    use_curriculum_learning=True, cl_decay_steps=200., decoder_start='zeros')
    for key, value in model.items():
        if key == 'graph':
            continue
        if key not in settings:
            raise ValueError('Unknown DCRNN model setting: '+key)
        settings[key] = value
    if model.get('graph','trajectory_directed') != 'trajectory_directed':
        raise ValueError('All six arms freeze the same train-only trajectory_directed graph.')
    if (settings['hidden_dim'],settings['num_layers'],settings['diffusion_steps']) != (64,2,2):
        raise ValueError('The preserved DCRNN backbone requires hidden_dim64, layers2, diffusion2.')
    if not 0 <= float(settings['dropout']) < 1 or float(settings['cl_decay_steps']) <= 0:
        raise ValueError('Invalid dropout/curriculum settings.')
    if type(settings['use_curriculum_learning']) is not bool:
        raise TypeError('use_curriculum_learning must be a JSON boolean.')
    return settings


def comparison_run_pairs(selected, seeds, model_name=MODEL_NAME):
    """Exact common-scorer run IDs; same seed is paired across input conditions."""
    selected = set(selected)
    contrasts = [(a,b) for a,b in CONTRASTS if a in selected and b in selected]
    pairs=[]
    for candidate, reference in contrasts:
        for seed in seeds:
            pairs.append((f'{model_name}|{candidate}|seed|seed={seed}',
                          f'{model_name}|{reference}|seed|seed={seed}'))
        if len(seeds)>=2:
            pairs.append((f'{model_name}|{candidate}|ensemble',
                          f'{model_name}|{reference}|ensemble'))
    return pairs


def checkpoint_run_from_source(source, condition, selected_count):
    """Resolve the same condition across resumed-batch lineage, never another arm."""
    from full_run_control import resolve_condition_source
    return resolve_condition_source(source, condition, selected_count)


def latest_training_run(condition_base):
    options=sorted(p for p in Path(condition_base).glob('*') if p.is_dir() and (p/'Models').is_dir())
    return options[-1] if options else None


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, text):
        self.original.write(text); self.log.write(text); self.log.flush()
        return len(text)
    def flush(self):
        self.original.flush(); self.log.flush()
    def isatty(self):
        return False
    @property
    def encoding(self):
        return 'utf-8'


def common_modules(project_root):
    directory = Path(project_root)/'Model/Common/DCRNN_H48_20261111'
    if not all((directory/name).is_file() for name in ('score.py','grid.py','neural_evaluation.py')):
        raise FileNotFoundError('Upload the private Model/Common/DCRNN_H48_20261111 folder from this ZIP.')
    sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('_dcrnn_h48_neural_evaluation',directory/'neural_evaluation.py')
    evaluation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    scorer,grid_module,hashes=evaluation._load_common(project_root,directory)
    return directory,evaluation,scorer,grid_module,hashes


def combined_score(project_root, completed, selected, seeds, batch_root, config, *, final, grid_path):
    """Keep common output names/column order and correct global Holm family."""
    common,evaluation,scorer,grid_module,hashes=common_modules(project_root)
    grid_path=Path(grid_path)
    grid=grid_module.read_frame(grid_path)
    pairs=comparison_run_pairs(selected,seeds)
    scoring=dict(config.get('scoring',{}))
    outputs={}
    for split in (('validation','test') if final else ('validation',)):
        frames=[];provenance=[]
        for condition in selected:
            run=Path(completed[condition])
            source=run/'Exports'/f'{split}_predictions_by_seed.parquet'
            if not source.is_file():
                raise FileNotFoundError('Missing complete condition predictions: '+str(source))
            frame=pd.read_parquet(source)
            if not frame.input_condition.eq(condition).all():
                raise ValueError('Prediction source has the wrong condition: '+str(source))
            if not frame.prediction_kind.eq('seed').all():
                raise ValueError('Pass raw seed forecasts only; the common scorer generates ensembles.')
            frames.append(frame)
            sidecar=source.with_suffix('.meta.json')
            provenance.append({'path':str(source),'sha256':scorer.sha256(source),
                               'sidecar':json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.is_file() else None})
        frame=pd.concat(frames,ignore_index=True)
        directory=evaluation._unique_invocation_directory(Path(batch_root)/'Common_Scoring/Combined'/split)
        result=scorer.score_frames(grid,frame,split=split,pairs=pairs,final=final and split=='test',
                                   bootstrap_reps=int(scoring.get('bootstrap_reps',1000)),
                                   bootstrap_seed=int(scoring.get('bootstrap_seed',0)),
                                   hac_lag=scoring.get('hac_lag'),make_ensembles=True)
        evaluation._output_tables(result,directory)
        audit=dict(result['audit'],written_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                   grid_path=str(grid_path.resolve()),grid_sha256=scorer.sha256(grid_path),
                   prediction_sources=provenance)
        atomic_json(directory/'scoring_audit.json',audit)
        extra=directory/'Additional_Analysis'
        atomic_json(extra/'input_condition_comparison_plan.json',{
            'selected_conditions':list(selected),'seeds':list(seeds),'predeclared_pairs':[list(p) for p in pairs],
            'comparison_sign':'candidate RMSE minus reference RMSE; negative is better',
            'global_holm_family':'all included horizons, runs, baseline and input-condition contrasts in this invocation',
            'validation_is_used_for_early_stopping':True,'test_2024_previously_inspected':True,
            'common_code_sha256':hashes,'standard_outputs_unchanged':list(STANDARD_OUTPUTS)})
        for name in STANDARD_OUTPUTS:
            if not (directory/name).is_file():raise RuntimeError('Missing standard scoring output: '+name)
        outputs[split]=str(directory)
        print(f'[combined score] {split}: {directory}',flush=True)
    return outputs


def self_test():
    assert normalized_tokens(['M1,M2','M3'],CONDITIONS)==['M1','M2','M3']
    assert model_settings({})['num_layers']==2
    assert model_settings({})['hidden_dim']==64
    try:model_settings({'model':{'graph':'distance'}})
    except ValueError:pass
    else:raise AssertionError('Graph changes are separate experiments.')
    pairs=comparison_run_pairs(CONDITIONS,[11])
    assert len(pairs)==7 and all('|seed|seed=11' in p[0] for p in pairs)
    assert (f'{MODEL_NAME}|M3|seed|seed=11', f'{MODEL_NAME}|M1|seed|seed=11') in pairs
    assert len(comparison_run_pairs(CONDITIONS,[11,22,33]))==28
    assert not comparison_run_pairs(['M4T'],[11])
    with tempfile.TemporaryDirectory() as name:
        batch=Path(name);atomic_json(batch/'batch_manifest.json',{'mode':'full'})
        assert checkpoint_run_from_source(batch,'M1',6) is None
        run=batch/'Conditions/M2/20260101_000000_a';(run/'Models').mkdir(parents=True)
        atomic_json(run/'experiment_config.json',{'training':{'input_condition':'M2'}})
        assert checkpoint_run_from_source(batch,'M2',6) is None  # No completed epoch exists yet.
        try:checkpoint_run_from_source(run,'M1',1)
        except ValueError:pass
        else:raise AssertionError('Cross-condition resume must fail.')
    print(json.dumps({'runner_helper_tests':'passed','full_training_tested':False}),flush=True)


def _main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=('pilot','full'),default='pilot')
    ap.add_argument('--conditions',nargs='+',default=None)
    ap.add_argument('--seeds',nargs='+',default=None)
    ap.add_argument('--final-test',action='store_true')
    ap.add_argument('--resume-source',type=Path)
    ap.add_argument('--project-root',type=Path,default=Path('/workspace/TSA'))
    ap.add_argument('--output-root',type=Path)
    ap.add_argument('--config',type=Path)
    ap.add_argument('--continue-on-error',action='store_true')
    ap.add_argument('--allow-cpu-full',action='store_true')
    ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args(argv)
    def interrupted_signal(_number,_frame):
        raise KeyboardInterrupt('Termination requested; recover from the latest completed epoch.')
    signal.signal(signal.SIGTERM,interrupted_signal)
    if args.self_test:self_test();return 0
    if args.mode=='pilot' and args.final_test:
        ap.error('--final-test is allowed only with --mode full.')
    config=read_config(args.config)
    execution=dict(config.get('execution',{}))
    selected=normalized_tokens(args.conditions or execution.get('conditions',CONDITIONS),CONDITIONS)
    seeds=[int(s) for s in normalized_tokens(args.seeds or execution.get('seeds',[11]))]
    settings=model_settings(config)
    project=args.project_root.resolve()
    output=(args.output_root or project/'Result/DCRNN/VastAI_H48').resolve()
    configured_cache=execution.get('cache_root')
    cache=Path(configured_cache).expanduser() if configured_cache else project/'Model/DCRNN/VastAI_M1_M4T_H48_20261111/Cache'
    if not cache.is_absolute():cache=project/cache
    cache=cache.resolve()
    sys.path.insert(0,str(HERE/'vendor'))
    import torch
    import neural_training as trainer
    from dcrnn_model import DCRNNMultiHorizon
    from data_experiments import prepare_universe,prepare_condition
    from drive_backup import RcloneDriveBackup,DriveBackupError
    common,_,_,_,_=common_modules(project)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode=='full' and device=='cpu' and not args.allow_cpu_full and not args.validate_only:
        raise RuntimeError('Full training requires a CUDA GPU. Use pilot/validate-only, or explicitly --allow-cpu-full.')
    batch=trainer.reserve_execution_directory(output/args.mode)
    state={'status':'running','mode':args.mode,'model_family':'DCRNN','experiment_schema':EXPERIMENT_BUNDLE,
           'conditions':{},'selected_conditions':selected,
           'seeds':seeds,'final_test':args.final_test,'validate_only':args.validate_only,
           'resume_source':str(args.resume_source.resolve()) if args.resume_source else None,
           'batch_root':str(batch),'cache_root':str(cache),'started_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}
    atomic_json(batch/'batch_manifest.json',state)
    atomic_json(batch/'resolved_configuration.json',dict(config,model_resolved=settings,selected_conditions=selected,seeds_resolved=seeds))
    atomic_json(batch/'predeclared_comparisons.json',{'pairs':[list(p) for p in comparison_run_pairs(selected,seeds)],
                'test_requires_explicit_flag':True,'same_graph_for_all_conditions':'2021-2022 trajectory_directed',
                'data_variant_resume_must_match':True})
    begin=time.perf_counter();completed={};failed=[]
    with (batch/'console.log').open('x',encoding='utf-8',buffering=1) as log:
      with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
        print(f'Batch: {batch}\nDevice: {device}; conditions={selected}; seeds={seeds}\nPersistent data cache: {cache}',flush=True)
        print('Architecture: stacked DCGRU encoder2/decoder2, hidden64, diffusion2; 48h encoder and 48-step decoder; horizons='+str(HORIZONS),flush=True)
        configured_training=config.get('training',{})
        print('Training profile='+str(configured_training.get('training_profile','stabilized_lr'))+
              '; LR='+str(configured_training.get('learning_rate',.001))+
              '; max_epochs='+str(configured_training.get('max_epochs',50))+
              '; patience='+str(configured_training.get('patience',10))+
              '; scheduler='+str(configured_training.get('scheduler','plateau'))+
              '; validation selection horizon='+str(configured_training.get('selection_horizon',6)),flush=True)
        print('H48 training: curriculum tau='+str(settings['cl_decay_steps'])+'; checkpoint compatibility is restricted to this six-horizon version.',flush=True)
        print('Curriculum labels are training-only; inference never receives future targets. M4 is an empirical information test, not a presumed negative control.',flush=True)
        backup=RcloneDriveBackup(config,batch,args.mode)
        try:
            if not args.validate_only:
                preflight=backup.preflight()
                atomic_json(batch/'backup_preflight.json',preflight)
                print('[Drive backup] preflight completed; configured='+str(preflight),flush=True)
                # Verify actual write permission before spending time on full training.
                backup.backup('initial_pretraining_write_check',force=True)
            def checkpoint_backup(event):
                epoch=int(event['epoch'])
                interval=int(config.get('checkpoint',{}).get('snapshot_every_epochs',5))
                if event.get('final') or event.get('resumed') or epoch==1 or epoch%interval==0:
                    return backup.backup('checkpoint_epoch_'+str(epoch)+('_final' if event.get('final') else ''),force=True)
            trainer.CHECKPOINT_EVENT_CALLBACK=checkpoint_backup
            universe=prepare_universe(project,config,cache,build_trajectory=any(c.endswith('T') for c in selected))
            from horizon_contract import ensure_h48_grid
            grid_path=ensure_h48_grid(project,cache/'evaluation',universe['bundle'])
            atomic_json(batch/'evaluation_grid_contract.json',{'path':str(grid_path),'sha256':hashlib.sha256(Path(grid_path).read_bytes()).hexdigest(),
                'horizons':list(HORIZONS),'seasonal_baseline':'latest previous daily phase at or before origin; 36h=t-12h,48h=t',
                'original_four_horizons_preserved':True})
            atomic_json(batch/'source_contract.json',universe['source_contract'])
            available=set(universe['conditions'])
            if not set(selected).issubset(available):raise ValueError('Data experiment conditions unavailable.')
            adjacency=np.asarray(universe['adjacency'],dtype=np.float32)
            if adjacency.ndim!=2 or adjacency.shape[0]!=adjacency.shape[1] or not np.isfinite(adjacency).all() or (adjacency<0).any():
                raise ValueError('A finite nonnegative square train-only transport adjacency is required.')
            node_order=universe['bundle']['nodes'].station_key.astype(str).tolist()
            if not node_order or node_order[0]!=TARGET:raise ValueError('Target must remain node index zero.')
            for number,condition in enumerate(selected,1):
                started=time.perf_counter();condition_base=batch/'Conditions'/condition
                state['conditions'][condition]={'status':'running','condition_base_dir':str(condition_base)}
                atomic_json(batch/'batch_manifest.json',state)
                print(f'\n[{number}/{len(selected)}] {condition} starting; total elapsed={(started-begin)/60:.1f} min',flush=True)
                try:
                    prepared=prepare_condition(universe,condition,config)
                    bundle,preprocessing,arrays=prepared['bundle'],prepared['preprocessing'],prepared['arrays']
                    if bundle['nodes'].station_key.astype(str).tolist()!=node_order:
                        raise ValueError('Node order must be fixed across information conditions.')
                    if adjacency.shape!=(len(node_order),len(node_order)):
                        raise ValueError('Fixed graph and node universe differ.')
                    atomic_json(batch/'Condition_Contracts'/f'{condition}.json',prepared['contract'])
                    if args.validate_only:
                        state['conditions'][condition]={'status':'validated_only','contract_path':str(batch/'Condition_Contracts'/f'{condition}.json')}
                        atomic_json(batch/'batch_manifest.json',state)
                        print(f'[{condition}] data validation completed; no training/scoring.',flush=True)
                        continue
                    training=dict(config.get('training',{}))
                    cfg=dict(model_family='DCRNN',experiment_bundle=EXPERIMENT_BUNDLE,
                             input_condition=condition,mode=args.mode,seeds=seeds,
                             max_epochs=50,patience=10,micro_batch_size=8,effective_batch_size=64,eval_batch_size=8,
                             optimizer='adam',learning_rate=.001,optimizer_epsilon=.001,weight_decay=0.,
                             exclude_bias_norm_from_decay=False,scheduler='plateau',lr_patience=3,
                             lr_gamma=.5,min_learning_rate=1e-5,max_grad_norm=5.,selection_horizon=6,
                             loss='mae_original',min_delta=0.,resume=True,run_final_test=args.final_test,
                             device=device,use_amp=False,model_configuration=settings,comparison_pairs=[],
                             architecture_profile='preserved_seq2seq',bootstrap_reps=int(config.get('scoring',{}).get('bootstrap_reps',1000)))
                    protected={'model_family','experiment_bundle','input_condition','mode','seeds','run_final_test','device','data_signature','model_configuration','comparison_pairs','evaluation_grid_path','evaluation_grid_sha256'}
                    if protected & set(training):raise ValueError('Training config cannot override frozen orchestration keys: '+','.join(sorted(protected & set(training))))
                    cfg.update(training)
                    cfg['evaluation_grid_path']=str(grid_path)
                    cfg['evaluation_grid_sha256']=hashlib.sha256(Path(grid_path).read_bytes()).hexdigest()
                    cfg['checkpoint_policy']=dict(config.get('checkpoint',{}))
                    cfg['library_versions']={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__}
                    cfg['scoring_output_root']=str(batch/'Common_Scoring'/condition)
                    source=checkpoint_run_from_source(args.resume_source,condition,len(selected))
                    cfg['resume_source_run']=str(source) if source else None
                    if args.mode=='pilot':
                        pilot=config.get('pilot',{})
                        cfg.update(max_epochs=int(pilot.get('epochs',2)),max_train_origins=int(pilot.get('max_train_origins',64)),
                                   max_val_origins=int(pilot.get('max_val_origins',32)),max_train_batches=int(pilot.get('max_train_batches',8)))
                    if source:print(f'[{condition}] resuming compatible checkpoints from {source}',flush=True)
                    spec={'model':MODEL_NAME,'graph':'trajectory_directed','adaptive_mode':'none',
                          'architecture_profile':'preserved_seq2seq','model_settings':settings}
                    def factory(a,spec,context_dim):
                        return DCRNNMultiHorizon(a,input_dim=2,context_dim=context_dim,horizons=HORIZONS,**settings)
                    summaries=trainer.run_neural_experiment(bundle,{'trajectory_directed':adjacency},preprocessing,arrays,[spec],cfg,
                                                            condition_base,factory,universe['api'],project,common_dir=common)
                    run=Path(summaries[0]['experiment_run_root'])
                    completed[condition]=str(run)
                    backup.backup('condition_'+condition+'_inference_scoring_complete',force=True)
                    state['conditions'][condition]={'status':'passed','run_root':str(run),
                        'scoring_locations':summaries[0]['scoring_locations'],'seconds':time.perf_counter()-started}
                    atomic_json(batch/'batch_manifest.json',state)
                    outcome='finished + inferred/exported; pilot partial-grid common scoring skipped' if args.mode=='pilot' else 'finished + inferred/exported/scored'
                    print(f'[{condition}] {outcome}; elapsed={(time.perf_counter()-started)/60:.1f} min; total={(time.perf_counter()-begin)/60:.1f} min\n  {run}',flush=True)
                    del prepared,bundle,preprocessing,arrays
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                except KeyboardInterrupt:
                    run=latest_training_run(condition_base)
                    state['conditions'][condition]={'status':'interrupted','run_root':str(run) if run else None}
                    state['status']='interrupted';atomic_json(batch/'batch_manifest.json',state)
                    try:backup.backup('interrupted_latest_committed_state',force=True)
                    except Exception as backup_error:print('[Drive backup FAILED on interruption] '+str(backup_error),flush=True)
                    print(f'Interrupted. Resume source: {batch}',flush=True)
                    return 130
                except Exception as error:
                    traceback.print_exc();failed.append(condition)
                    run=latest_training_run(condition_base)
                    state['conditions'][condition]={'status':'failed','error_type':type(error).__name__,'error':str(error),
                                                   'run_root':str(run) if run else None}
                    atomic_json(batch/'batch_manifest.json',state)
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                    if isinstance(error,DriveBackupError) or not args.continue_on_error:break
            if not failed and not args.validate_only and args.mode=='full':
                state['combined_scoring']=combined_score(project,completed,selected,seeds,batch,config,final=args.final_test,grid_path=grid_path)
            elif failed:
                state['combined_scoring_skipped']='Incomplete condition family; per-condition completed scores remain available. Do not shrink Holm family silently.'
            state['status']='failed' if failed else ('validated_only' if args.validate_only else 'passed')
            state['failed_conditions']=failed;state['elapsed_seconds']=time.perf_counter()-begin
            atomic_json(batch/'batch_manifest.json',state)
            if not args.validate_only and not failed:
                backup.backup('batch_complete',force=True)
            print(f'\nBatch {state["status"]}; elapsed={state["elapsed_seconds"]/60:.1f} min; results={batch}',flush=True)
            return 1 if failed else 0
        except KeyboardInterrupt:
            state['status']='interrupted';atomic_json(batch/'batch_manifest.json',state);return 130
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-begin)
            atomic_json(batch/'batch_manifest.json',state);traceback.print_exc();return 1


def main(argv=None):
    from full_run_control import runner_lease
    arguments = list(sys.argv[1:] if argv is None else argv)
    with runner_lease(arguments) as acquired:
        return _main(arguments) if acquired else 0


if __name__=='__main__':
    raise SystemExit(main())
