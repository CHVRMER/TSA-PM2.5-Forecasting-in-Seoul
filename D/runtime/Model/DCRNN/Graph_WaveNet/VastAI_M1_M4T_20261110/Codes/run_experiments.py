"""Run six frozen Graph WaveNet information conditions on a persistent Vast.ai disk.

The existing model, neural trainer and common six-output scorer are reused.
Every invocation creates a new batch folder. Resume explicitly names an older
batch/run; checkpoints are copied by the preserved trainer, never overwritten.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib
import importlib.metadata
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
CONTRASTS = (('M2','M1'), ('M3','M2'), ('M3','M1'), ('M4','M3'), ('M3T','M3'), ('M4T','M4'))
MODEL_NAME = 'GraphWaveNet_distance_fixed_plus_adaptive'
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
        return {}
    text = path.read_text(encoding='utf-8-sig')
    if path.suffix.lower() in ('.yaml','.yml'):
        import yaml
        result = yaml.safe_load(text)
    else:
        result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError('The experiment configuration must be an object/mapping.')
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
    profile = model.pop('architecture_profile', 'paper')
    if profile not in ('paper','project48'):
        raise ValueError('architecture_profile must be paper or project48.')
    settings = dict(residual_channels=32, dilation_channels=32, skip_channels=256,
                    end_channels=512, kernel_size=2, blocks=4,
                    layers=2 if profile=='paper' else 4, diffusion_steps=2,
                    dropout=.3, embedding_dim=10, embedding_init='normal',
                    architecture_profile=profile)
    for key, value in model.items():
        if key in ('graph_mode','graph'):
            continue
        if key not in settings:
            raise ValueError('Unknown Graph WaveNet model setting: '+key)
        settings[key] = value
    if model.get('graph_mode','fixed_plus_adaptive') != 'fixed_plus_adaptive':
        raise ValueError('This information experiment freezes fixed_plus_adaptive for every condition.')
    if model.get('graph','distance') != 'distance':
        raise ValueError('This information experiment freezes the distance graph, including M3T/M4T.')
    expected_layers = 2 if profile=='paper' else 4
    if (settings['blocks'], settings['layers'], settings['kernel_size']) != (4,expected_layers,2):
        raise ValueError('Architecture profile and temporal-convolution settings disagree.')
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
    directory = Path(project_root)/'Model/Common'
    if not all((directory/name).is_file() for name in ('score.py','grid.py','neural_evaluation.py')):
        raise FileNotFoundError('Upload the unchanged Model/Common score.py, grid.py and neural_evaluation.py.')
    sys.path.insert(0,str(directory))
    evaluation=importlib.import_module('neural_evaluation')
    scorer,grid_module,hashes=evaluation._load_common(project_root,directory)
    return directory,evaluation,scorer,grid_module,hashes


def combined_score(project_root, completed, selected, seeds, batch_root, config, *, final):
    """Keep common output names/column order and correct global Holm family."""
    common,evaluation,scorer,grid_module,hashes=common_modules(project_root)
    grid_path=Path(project_root)/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
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
    assert model_settings({})['layers']==2
    assert model_settings({'model':{'architecture_profile':'project48'}})['layers']==4
    pairs=comparison_run_pairs(CONDITIONS,[11])
    assert len(pairs)==6 and all('|seed|seed=11' in p[0] for p in pairs)
    assert (f'{MODEL_NAME}|M3|seed|seed=11', f'{MODEL_NAME}|M1|seed|seed=11') in pairs
    assert len(comparison_run_pairs(CONDITIONS,[11,22,33]))==24
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
    output=(args.output_root or project/'Result/GraphWaveNet/VastAI').resolve()
    configured_cache=execution.get('cache_root')
    cache=Path(configured_cache).expanduser() if configured_cache else project/'Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/Cache'
    if not cache.is_absolute():cache=project/cache
    cache=cache.resolve()
    sys.path.insert(0,str(HERE/'vendor'))
    import torch
    import neural_training as trainer
    from graph_wavenet_model import GraphWaveNetMultiHorizon
    from data_experiments import prepare_universe,prepare_condition
    from drive_backup import RcloneDriveBackup,DriveBackupError
    common,_,_,_,_=common_modules(project)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode=='full' and device=='cpu' and not args.allow_cpu_full and not args.validate_only:
        raise RuntimeError('Full training requires a CUDA GPU. Use pilot/validate-only, or explicitly --allow-cpu-full.')
    batch=trainer.reserve_execution_directory(output/args.mode)
    state={'status':'running','mode':args.mode,'conditions':{},'selected_conditions':selected,
           'seeds':seeds,'final_test':args.final_test,'validate_only':args.validate_only,
           'resume_source':str(args.resume_source.resolve()) if args.resume_source else None,
           'batch_root':str(batch),'cache_root':str(cache),'started_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}
    atomic_json(batch/'batch_manifest.json',state)
    atomic_json(batch/'resolved_configuration.json',dict(config,model_resolved=settings,selected_conditions=selected,seeds_resolved=seeds))
    atomic_json(batch/'predeclared_comparisons.json',{'pairs':[list(p) for p in comparison_run_pairs(selected,seeds)],
                'test_requires_explicit_flag':True,'same_graph_for_all_conditions':'distance + adaptive',
                'data_variant_resume_must_match':True})
    begin=time.perf_counter();completed={};failed=[]
    with (batch/'console.log').open('x',encoding='utf-8',buffering=1) as log:
      with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
        print(f'Batch: {batch}\nDevice: {device}; conditions={selected}; seeds={seeds}\nPersistent data cache: {cache}',flush=True)
        print(f'Architecture: {settings["architecture_profile"]}; RF={13 if settings["layers"]==2 else 61} hours. Input history=48h.',flush=True)
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
            atomic_json(batch/'source_contract.json',universe['source_contract'])
            available=set(universe['conditions'])
            if not set(selected).issubset(available):raise ValueError('Data experiment conditions unavailable.')
            adjacency=np.asarray(universe['adjacency'],dtype=np.float32)
            if adjacency.ndim!=2 or adjacency.shape[0]!=adjacency.shape[1] or not np.isfinite(adjacency).all() or (adjacency<0).any():
                raise ValueError('A finite nonnegative square distance adjacency is required.')
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
                    cfg=dict(model_family='GraphWaveNet',input_condition=condition,mode=args.mode,seeds=seeds,
                             max_epochs=50,patience=10,micro_batch_size=8,effective_batch_size=64,eval_batch_size=8,
                             optimizer='adam',learning_rate=.001,optimizer_epsilon=1e-8,weight_decay=1e-4,
                             exclude_bias_norm_from_decay=False,scheduler='none',min_learning_rate=0.,max_grad_norm=5.,
                             loss='mae_original',min_delta=0.,resume=True,run_final_test=args.final_test,
                             device=device,use_amp=False,model_configuration=settings,comparison_pairs=[],
                             architecture_profile=settings['architecture_profile'],bootstrap_reps=int(config.get('scoring',{}).get('bootstrap_reps',1000)))
                    protected={'model_family','input_condition','mode','seeds','run_final_test','device','data_signature','model_configuration','comparison_pairs'}
                    if protected & set(training):raise ValueError('Training config cannot override frozen orchestration keys: '+','.join(sorted(protected & set(training))))
                    cfg.update(training)
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
                    spec={'model':MODEL_NAME,'graph':'distance','adaptive_mode':'fixed_plus_adaptive',
                          'architecture_profile':settings['architecture_profile'],'model_settings':settings}
                    def factory(a,spec,context_dim):
                        return GraphWaveNetMultiHorizon(a,num_nodes=len(node_order),input_dim=2,context_dim=context_dim,
                                      graph_mode='fixed_plus_adaptive',horizons=(3,6,12,24),output_steps=24,**settings)
                    summaries=trainer.run_neural_experiment(bundle,{'distance':adjacency},preprocessing,arrays,[spec],cfg,
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
                state['combined_scoring']=combined_score(project,completed,selected,seeds,batch,config,final=args.final_test)
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
