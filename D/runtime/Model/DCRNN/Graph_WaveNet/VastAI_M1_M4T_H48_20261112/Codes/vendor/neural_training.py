"""Shared target-only experiment runner for DCRNN and Graph WaveNet.

The project data, graph formulas, origin grid and targets remain unchanged.
Only training code can call a curriculum-learning entry point. Inference never
receives targets. Every checkpoint includes optimizer/scheduler/RNG state.
"""
from __future__ import annotations
from pathlib import Path
import copy, hashlib, inspect, json, math, os, random, time
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from checkpoint_store import CheckpointStore

CHECKPOINT_EVENT_CALLBACK=None

TRAINING_VERSION = 'seoul-gwn-m1-m4t-h48-training-20261112-v1'
HORIZONS = (3,6,12,24,36,48)
EXECUTION_ONLY_KEYS = ('device','resume','run_final_test','scoring_output_root','resume_source_run','execution_id','checkpoint_policy','evaluation_grid_path')


def reserve_execution_directory(base):
    base=Path(base).resolve();base.mkdir(parents=True,exist_ok=True)
    while True:
        identifier=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]
        candidate=base/identifier
        try:candidate.mkdir(exist_ok=False)
        except FileExistsError:continue
        return candidate


def canonical_json(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,default=str,allow_nan=False)


def json_digest(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as source:
        for piece in iter(lambda:source.read(4*1024*1024),b''):result.update(piece)
    return result.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def save_checkpoint(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    torch.save(payload,temporary);os.replace(temporary,path)


def save_learning_curve(path,history):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as stream:
        pd.DataFrame(history).to_csv(stream,index=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def neural_data_signature(bundle):
    result=hashlib.sha256()
    for name in ('pm_values','context_values','y_raw','eligible','origin_positions'):
        array=np.ascontiguousarray(bundle[name])
        result.update(name.encode());result.update(str(array.dtype).encode());result.update(str(array.shape).encode())
        result.update(memoryview(array).cast('B'))
    for values in (bundle['nodes'].station_key,bundle['origins'].origin_time_kst):
        result.update('|'.join(values.astype(str)).encode('utf-8'))
    return result.hexdigest()


def set_neural_seed(seed):
    random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed))
    if torch.cuda.is_available():torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def optimizer_and_scheduler(model,config):
    kind=config.get('optimizer','adam').lower()
    if kind not in ('adam','adamw'):raise ValueError('optimizer must be adam or adamw')
    wd=float(config.get('weight_decay',0.0))
    if config.get('exclude_bias_norm_from_decay',False) and wd:
        decay=[];no_decay=[]
        for _,parameter in model.named_parameters():
            if parameter.requires_grad:(decay if parameter.ndim>1 else no_decay).append(parameter)
        parameters=[{'params':decay,'weight_decay':wd},{'params':no_decay,'weight_decay':0.0}]
    else:parameters=model.parameters()
    cls=torch.optim.Adam if kind=='adam' else torch.optim.AdamW
    optimizer=cls(parameters,lr=float(config['learning_rate']),weight_decay=wd,
                  eps=float(config.get('optimizer_epsilon',1e-8)),betas=tuple(config.get('betas',(.9,.999))))
    scheduler_kind=config.get('scheduler','none')
    if scheduler_kind=='multistep':
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=list(config.get('lr_milestones',[20,30,40,50])),gamma=float(config.get('lr_gamma',.1)))
    elif scheduler_kind=='plateau':
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=float(config.get('lr_gamma',.5)),patience=int(config.get('lr_patience',5)),min_lr=float(config.get('min_learning_rate',1e-5)))
    elif scheduler_kind=='none':scheduler=None
    else:raise ValueError('scheduler must be none, multistep or plateau')
    return optimizer,scheduler


def step_scheduler(optimizer,scheduler,config,validation_score):
    if scheduler is not None:
        if config.get('scheduler')=='plateau':scheduler.step(validation_score)
        else:scheduler.step()
    floor=float(config.get('min_learning_rate',0.0))
    for group in optimizer.param_groups:group['lr']=max(floor,float(group['lr']))


def inverse_neural(values,preprocessing):
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.expm1(np.asarray(values,dtype=np.float64)*float(preprocessing['pm_std'][0])+float(preprocessing['pm_mean'][0]))
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite inverse prediction: lower learning rate in a NEW run; no rows were removed.')
    return result


def masked_loss_sums(prediction,target,mask,preprocessing,loss_kind):
    if prediction.shape!=target.shape or target.shape!=mask.shape:raise ValueError('Prediction/target/mask shape mismatch.')
    if mask.dtype!=torch.bool:raise ValueError('Target availability mask must be boolean.')
    if not torch.isfinite(target[mask]).all():raise ValueError('Available targets must be finite.')
    target=torch.where(mask,target,torch.zeros_like(target))
    if not torch.isfinite(prediction).all():raise FloatingPointError('Nonfinite model output.')
    if loss_kind=='mae_original':
        mean=float(preprocessing['pm_mean'][0]);std=float(preprocessing['pm_std'][0])
        p=torch.expm1(prediction*std+mean);y=torch.expm1(target*std+mean)
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite original-unit training loss; no silent clipping.')
        error=(p-y).abs()
    elif loss_kind=='mse_log_z':error=(prediction-target).square()
    else:raise ValueError('Unknown training loss.')
    return torch.where(mask,error,torch.zeros_like(error)).sum(dim=0)


@torch.no_grad()
def predict_neural(model,loader,bundle,preprocessing,device,model_name,input_condition,seed,data_version,mode):
    model.eval();predictions=[];indices=[]
    for x,context,_target,_mask,index in loader:
        value=model(x.to(device),context.to(device))
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite inference output.')
        predictions.append(value.cpu().numpy());indices.extend(index.cpu().numpy().tolist())
    if not predictions:raise ValueError('Empty evaluation origin list.')
    indices=np.asarray(indices,dtype=int)
    if len(indices)!=len(np.unique(indices)):raise ValueError('Duplicate evaluation origins.')
    raw=inverse_neural(np.concatenate(predictions),preprocessing)
    records=[]
    for row,index in enumerate(indices):
        origin=bundle['origins'].iloc[index]
        for j,h in enumerate(bundle['horizons']):
            records.append({'origin_time_kst':origin.origin_time_kst,'target_time_kst':origin.origin_time_kst+pd.Timedelta(hours=int(h)),
                'horizon':int(h),'model':model_name,'input_condition':input_condition,'prediction_kind':'seed','seed':str(seed),
                'split':origin.origin_split,'y_true':float(bundle['y_raw'][index,j]),'y_hat':float(raw[row,j]),
                'eligible':bool(bundle['eligible'][index,j]),'data_version':data_version,'experiment_mode':mode,
                'common_export_unclipped':True})
    return pd.DataFrame(records),indices


def prediction_metrics(frame):
    rows=[]
    for h,part in frame.loc[frame.eligible].groupby('horizon'):
        y=part.y_true.to_numpy(float);p=np.maximum(part.y_hat.to_numpy(float),0)
        if not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Nonfinite eligible metric input.')
        error=p-y;sst=float(np.sum((y-y.mean())**2))
        rows.append({'horizon':int(h),'n':len(y),'RMSE':float(np.sqrt(np.mean(error**2))),
            'MAE':float(np.mean(np.abs(error))),'MBE':float(error.mean()),
            'R2':float(1-np.sum(error**2)/sst) if sst else None})
    return pd.DataFrame(rows)


def _rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if device.type=='cuda' and state.get('cuda') is not None:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def _loader(api,bundle,arrays,ids,batch_size,shuffle,device):
    dataset=api['DCRNNOriginDataset'](bundle,arrays,ids,bundle['lookback_hours'])
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=0,pin_memory=device.type=='cuda',drop_last=False)


def train_one_neural_run(bundle,arrays,preprocessing,adjacency,spec,train_config,run_root,model_factory,api):
    """One graph/seed. A full run cannot contain pilot sample restrictions."""
    cfg=copy.deepcopy(train_config);seed=int(cfg['seed']);set_neural_seed(seed)
    mode=cfg['mode'];model_name=spec['model'];condition=cfg['input_condition']
    if mode not in ('pilot','full'):raise ValueError('mode must be pilot or full')
    if mode=='full' and any(cfg.get(k) for k in ('max_train_origins','max_val_origins','max_train_batches')):
        raise ValueError('Full mode cannot use pilot sample limits.')
    device=torch.device(cfg.get('device','cuda' if torch.cuda.is_available() else 'cpu'))
    model=model_factory(adjacency,spec,arrays['context_x'].shape[1]).to(device)
    architecture=copy.deepcopy(model.architecture_config)
    signature=cfg['data_signature'];graph_hash=hashlib.sha256(np.ascontiguousarray(adjacency,dtype=np.float32).tobytes()).hexdigest() if adjacency is not None else 'adaptive_only'
    contract={'version':TRAINING_VERSION,'data':signature,'graph':graph_hash,'model':model_name,'architecture':architecture,
              'training':{k:v for k,v in cfg.items() if k not in EXECUTION_ONLY_KEYS},'preprocessing':json_digest(preprocessing)}
    folder=Path(run_root)/'Models'/model_name/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    contract_path=folder/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text(encoding='utf-8'))!=json.loads(canonical_json(contract)):
        raise ValueError('Existing run uses different model/data/training settings. Choose a NEW RUN_NAME.')
    save_json(contract_path,contract);save_json(folder/'preprocessing.json',preprocessing)
    optimizer,scheduler=optimizer_and_scheduler(model,cfg)
    use_amp=bool(cfg.get('use_amp',False)) and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    state_path=folder/'last.pt';best_path=folder/'best.pt'
    policy=cfg.get('checkpoint_policy',{})
    store=CheckpointStore(folder,snapshot_every=int(policy.get('snapshot_every_epochs',5)),
                          keep_recent=int(policy.get('keep_recent',2)))
    best=float('inf');bad=0;start_epoch=0;global_step=0;history=[];best_model=None;best_epoch=None
    state,recovery=store.load_latest(expected_contract=contract,map_location=device)
    if state is None and state_path.exists():
        raise ValueError('Legacy last.pt has no verified generation. Use its original runner or an explicitly verified migration; do not silently import it.')
    if state is not None:
        if not cfg.get('resume',True):raise FileExistsError('Committed checkpoint exists and RESUME is False; choose a new run.')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        if scheduler is not None:scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler']);best=state['best'];bad=state['bad_epochs'];start_epoch=state['epoch']+1
        history=state['history'];global_step=state['global_step'];_restore_rng(state['rng'],device)
        best_model={k:v.detach().cpu().clone() for k,v in state['best_model'].items()}
        best_epoch=int(state['best_epoch'])
        save_learning_curve(folder/'learning_curve.csv',history)
        save_json(folder/'checkpoint_recovery.json',recovery)
        print('[checkpoint resume] epoch='+str(start_epoch)+'; '+str(recovery),flush=True)
        if callable(CHECKPOINT_EVENT_CALLBACK):CHECKPOINT_EVENT_CALLBACK({'epoch':start_epoch,'resumed':True,'seed_folder':str(folder)})
    train_ids=api['split_origin_indices'](bundle,'train',require_any_label=True)
    val_ids=api['split_origin_indices'](bundle,'validation',require_any_label=False)
    if mode=='pilot':
        train_ids=train_ids[:int(cfg.get('max_train_origins',64))];val_ids=val_ids[:int(cfg.get('max_val_origins',32))]
    micro=int(cfg['micro_batch_size']);effective=int(cfg['effective_batch_size'])
    if micro<1 or effective<micro or effective%micro:raise ValueError('Effective batch must be an integer multiple of micro batch.')
    train_loader=_loader(api,bundle,arrays,train_ids,micro,True,device)
    val_loader=_loader(api,bundle,arrays,val_ids,int(cfg.get('eval_batch_size',micro)),False,device)
    accumulation=effective//micro;begin=time.perf_counter()
    for epoch in range(start_epoch,int(cfg['max_epochs'])):
        if bad>=int(cfg['patience']):break
        model.train();epoch_begin=time.perf_counter();window=[];window_losses=[];used_lr=float(optimizer.param_groups[0]['lr'])
        def update(batches):
            nonlocal global_step
            counts=sum((batch[3].sum(dim=0) for batch in batches)).to(device)
            active=counts>0
            if not bool(active.any()):return
            optimizer.zero_grad(set_to_none=True);total=0.0
            for x,c,y,m,_ in batches:
                x,c,y,m=x.to(device),c.to(device),y.to(device),m.to(device)
                with torch.autocast(device_type=device.type,enabled=use_amp):
                    if getattr(model,'use_curriculum_learning',False):
                        prediction=model.forward_train(x,c,y,m,batches_seen=global_step)
                    else:prediction=model(x,c)
                    sums=masked_loss_sums(prediction,y,m,preprocessing,cfg.get('loss','mae_original'))
                    loss=(sums[active]/counts[active]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; reduce LR in a new run. No observations were dropped.')
                scaler.scale(loss).backward();total+=float(loss.detach())
            scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get('max_grad_norm',5.)))
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient norm.')
            scaler.step(optimizer);scaler.update();global_step+=1;window_losses.append(total)
        for number,batch in enumerate(train_loader):
            if cfg.get('max_train_batches') is not None and number>=int(cfg['max_train_batches']):break
            window.append(batch)
            if len(window)==accumulation:update(window);window=[]
        if window:update(window)
        if not window_losses:raise ValueError('No supervised training updates.')
        validation,produced=predict_neural(model,val_loader,bundle,preprocessing,device,model_name,condition,seed,signature,mode)
        if not np.array_equal(produced,val_ids):raise ValueError('Validation origin coverage mismatch.')
        values=prediction_metrics(validation)
        selection_horizon=int(cfg.get('selection_horizon',6))
        if selection_horizon not in HORIZONS:raise ValueError('Selection horizon must be one of the six declared forecasts.')
        selected=values.loc[values.horizon.eq(selection_horizon),'RMSE']
        if selected.empty:raise ValueError('No eligible validation targets for selection horizon '+str(selection_horizon))
        score=float(selected.iloc[0]);improved=score<best-float(cfg.get('min_delta',0.))
        if improved:
            best=score;bad=0;best_epoch=epoch+1
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:bad+=1
        step_scheduler(optimizer,scheduler,cfg,score)
        row={'epoch':epoch+1,'train_loss':float(np.mean(window_losses)),'loss_kind':cfg.get('loss','mae_original'),
             'validation_rmse_h6':float(values.loc[values.horizon.eq(6),'RMSE'].iloc[0]),
             'validation_selection_score':score,'selection_horizon':selection_horizon,
             'learning_rate_used':used_lr,'learning_rate_next':float(optimizer.param_groups[0]['lr']),
             'bad_epochs':bad,'global_step':global_step,'seconds':time.perf_counter()-epoch_begin}
        if getattr(model,'use_curriculum_learning',False):row['teacher_forcing_probability']=float(model.sampling_probability(global_step))
        history.append(row)
        payload={'contract':contract,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict() if scheduler is not None else None,'scaler':scaler.state_dict(),
            'epoch':epoch,'global_step':global_step,'best':best,'bad_epochs':bad,'history':history,'rng':_rng_state(),
            'best_model':best_model,'best_epoch':best_epoch}
        committed=store.save(payload)
        # Compatibility exports are not the authority for recovery or best-model inference.
        # The single committed generation contains both current and best weights.
        if improved:save_checkpoint(best_path,dict(payload,model=best_model))
        save_checkpoint(state_path,payload)
        save_learning_curve(folder/'learning_curve.csv',history)
        if callable(CHECKPOINT_EVENT_CALLBACK):
            CHECKPOINT_EVENT_CALLBACK({'epoch':epoch+1,'final':False,'seed_folder':str(folder),'commit':committed})
        print(f'[{model_name} seed{seed}] epoch={epoch+1}/{cfg["max_epochs"]} train={row["train_loss"]:.4f} val6hRMSE={row["validation_rmse_h6"]:.4f} selected_h={selection_horizon} selectedRMSE={score:.4f} lr={used_lr:.3g} next={row["learning_rate_next"]:.3g} elapsed={time.perf_counter()-begin:.1f}s',flush=True)
    best_state,final_recovery=store.load_latest(expected_contract=contract,map_location=device)
    if best_state is None:raise RuntimeError('No verified current/best checkpoint exists.')
    save_learning_curve(folder/'learning_curve.csv',best_state['history'])
    final_commit=store.save(best_state,pin=True)
    model.load_state_dict(best_state['best_model']);model.eval()
    save_checkpoint(best_path,dict(best_state,model=best_state['best_model']))
    if callable(CHECKPOINT_EVENT_CALLBACK):
        CHECKPOINT_EVENT_CALLBACK({'epoch':int(best_state['epoch'])+1,'final':True,'seed_folder':str(folder),'commit':final_commit})
    validation,produced=predict_neural(model,val_loader,bundle,preprocessing,device,model_name,condition,seed,signature,mode)
    validation.to_parquet(folder/'validation_predictions.parquet',index=False)
    prediction_metrics(validation).to_csv(folder/'validation_metrics.csv',index=False)
    if hasattr(model,'export_adaptive_adjacency'):
        adaptive=model.export_adaptive_adjacency()
        if adaptive is not None:
            np.save(folder/'adaptive_adjacency.npy',adaptive)
            bundle['nodes'].to_csv(folder/'adaptive_node_order.csv',index=False,encoding='utf-8-sig')
    best_metrics=prediction_metrics(validation)
    summary={'model':model_name,'graph':spec['graph'],'adaptive_mode':spec.get('adaptive_mode','none'),'seed':seed,'mode':mode,
        'best_epoch':int(best_state['best_epoch']),'trained_epochs':len(best_state['history']),
        'validation_rmse_h6':float(best_metrics.loc[best_metrics.horizon.eq(6),'RMSE'].iloc[0]),
        'best_validation_selection_score':float(best_state['best']),'selection_horizon':int(cfg.get('selection_horizon',6)),
        'architecture':architecture,'checkpoint':str(best_path),'checkpoint_sha256':file_digest(best_path),
        'data_signature':signature,'graph_signature':graph_hash,'parameter_count':sum(p.numel() for p in model.parameters()),
        'session_training_seconds':time.perf_counter()-begin,'device':str(device),'test_scored':False,
        'sampling_labels_used_only_in_training':bool(getattr(model,'use_curriculum_learning',False)),
        'effective_batch_size':effective,'micro_batch_size':micro,'gradient_accumulation_steps':accumulation,
        'checkpoint_storage_schema':'self-contained-verified-generations',
        'checkpoint_policy':policy,'checkpoint_recovery':final_recovery,
        'batchnorm_microbatch_statistics_note':'BatchNorm, if present, observes micro batch, not effective batch.'}
    revision_path=folder/'stopping_policy_revision.json'
    if revision_path.is_file():
        summary['stopping_policy_revision']=json.loads(revision_path.read_text(encoding='utf-8'))
    summary['effective_stopping_limits']={'max_epochs':int(cfg['max_epochs']),'patience':int(cfg['patience'])}
    save_json(folder/'run_summary.json',summary)
    return model,summary,validation


def export_neural_predictions(tables,run_root,split,metadata):
    frame=pd.concat(tables,ignore_index=True)
    keys=['model','input_condition','seed','origin_time_kst','horizon']
    if frame.duplicated(keys).any():raise ValueError('Duplicate seed prediction keys.')
    folder=Path(run_root)/'Exports';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{split}_predictions_by_seed.parquet';frame.to_parquet(path,index=False)
    save_json(path.with_suffix('.meta.json'),dict(metadata,prediction_kind='seed',prediction_sha256=file_digest(path),split=split))
    ensemble=frame.groupby(['model','input_condition','origin_time_kst','horizon'],sort=False,as_index=False).agg(
        y_hat=('y_hat','mean'),y_true=('y_true','first'),target_time_kst=('target_time_kst','first'),split=('split','first'),
        eligible=('eligible','first'),data_version=('data_version','first'),n_seeds=('seed','nunique'))
    expected=frame.groupby(['model','input_condition']).seed.nunique()
    for _,row in ensemble.iterrows():
        if row.n_seeds!=expected.loc[(row.model,row.input_condition)]:raise ValueError('Missing seed member in ensemble.')
    ensemble['seed']='';ensemble['prediction_kind']='ensemble'
    ep=folder/f'{split}_predictions_ensemble_raw_mean.parquet';ensemble.to_parquet(ep,index=False)
    save_json(ep.with_suffix('.meta.json'),dict(metadata,prediction_kind='ensemble',prediction_sha256=file_digest(ep),split=split,ensemble_rule='raw unit arithmetic mean before clipping'))
    return path


def _run_neural_experiment(bundle,graphs,preprocessing,arrays,model_specs,config,run_root,model_factory,api,project_root,common_dir=None):
    """Run every declared graph/seed, export, then automatically common-score full runs."""
    import sys
    import importlib.util
    common=Path(common_dir) if common_dir else Path(project_root)/'Model/Common/GraphWaveNet_H48_20261112'
    sys.path.insert(0,str(common))
    evaluation_spec=importlib.util.spec_from_file_location('_gwn_h48_neural_evaluation_training',common/'neural_evaluation.py')
    evaluation=importlib.util.module_from_spec(evaluation_spec)
    evaluation_spec.loader.exec_module(evaluation)
    write_neural_scoring_plan=evaluation.write_neural_scoring_plan
    score_neural_run=evaluation.score_neural_run
    root=Path(run_root);root.mkdir(parents=True,exist_ok=True)
    signature=neural_data_signature(bundle);cfg=copy.deepcopy(config);cfg['data_signature']=signature
    for name in ('score.py','grid.py','neural_evaluation.py'):
        expected=cfg.get('source_code_hashes',{}).get('Model/Common/'+name)
        if expected is not None and file_digest(common/name)!=expected:
            raise ValueError('Common code differs from this notebook version: '+name)
    if tuple(bundle['horizons'])!=HORIZONS:raise ValueError('H48 requires exactly3/6/12/24/36/48h.')
    if cfg['mode']=='pilot' and cfg.get('run_final_test',False):raise ValueError('Pilot cannot evaluate test.')
    frozen_cfg={k:v for k,v in cfg.items() if k not in EXECUTION_ONLY_KEYS}
    training_meta={'target_station_key':'KR:111121','horizons':list(HORIZONS),'lookback_hours':bundle['lookback_hours'],
        'target_transform':'log1p + train-only standardization','data_signature':signature,
        'preprocessing_signature':json_digest(preprocessing),'train_split':'2021-2022','validation_split':'2023',
        'prediction_unit':'ug/m3','training_configuration':frozen_cfg,'model_configuration':cfg.get('model_configuration',{}),
        'preprocessing_profile':'legacy_inputs_six_horizon_targets','test2024_previously_inspected':True,
        'evaluation_grid_path':cfg['evaluation_grid_path'],'evaluation_grid_sha256':cfg['evaluation_grid_sha256'],
        'selection_horizon':int(cfg.get('selection_horizon',6)),
        'seasonal_baseline_policy':'last24h_phase_at_or_before_origin'}
    write_neural_scoring_plan(root,cfg['model_family'],cfg['input_condition'],model_specs=model_specs,seeds=cfg['seeds'],training_metadata=training_meta,pair_specs=cfg.get('comparison_pairs'))
    run_contract={'training':frozen_cfg,'model_specs':model_specs,'data':signature,'preprocessing':json_digest(preprocessing),'version':TRAINING_VERSION}
    contract_file=root/'experiment_config.json'
    if contract_file.exists() and json.loads(contract_file.read_text(encoding='utf-8'))!=json.loads(canonical_json(run_contract)):
        raise ValueError('RUN_NAME has a different frozen configuration.')
    save_json(contract_file,run_contract);save_json(root/'preprocessing.json',preprocessing)
    save_json(root/'execution_protocol.json',{'mode':cfg['mode'],'resume':cfg.get('resume',True),'run_final_test':cfg.get('run_final_test',False),
        'test2024_previously_inspected':True,'device':cfg.get('device'),'training_settings_unchanged_on_test_toggle':True,
        'execution_id':cfg.get('execution_id'),'resume_source_run':cfg.get('resume_source_run')})
    bundle['nodes'].to_csv(root/'node_order.csv',index=False,encoding='utf-8-sig')
    graph_dir=root/'Graphs';graph_dir.mkdir(exist_ok=True)
    for key,value in graphs.items():np.save(graph_dir/f'{key}.npy',value)
    summaries=[];validation_tables=[];test_tables=[]
    for spec in model_specs:
        adjacency=None if spec.get('adaptive_mode')=='adaptive_only' else graphs[spec['graph']]
        for seed in cfg['seeds']:
            local=dict(cfg,seed=int(seed))
            model,summary,validation=train_one_neural_run(bundle,arrays,preprocessing,adjacency,spec,local,root,model_factory,api)
            summaries.append(summary);validation_tables.append(validation)
            if cfg.get('run_final_test',False):
                if cfg['mode']!='full':raise ValueError('Only a full model may infer test.')
                ids=api['split_origin_indices'](bundle,'test',require_any_label=False)
                device=next(model.parameters()).device
                loader=_loader(api,bundle,arrays,ids,int(cfg.get('eval_batch_size',cfg['micro_batch_size'])),False,device)
                table,produced=predict_neural(model,loader,bundle,preprocessing,device,spec['model'],cfg['input_condition'],int(seed),signature,'full')
                if not np.array_equal(produced,ids):raise ValueError('Test origin coverage mismatch.')
                test_tables.append(table)
            del model
            if torch.cuda.is_available():torch.cuda.empty_cache()
    save_json(root/'run_summaries.json',summaries)
    meta={'model_family':cfg['model_family'],'input_condition':cfg['input_condition'],'seeds':cfg['seeds'],
          'data_version':signature,'training_configuration':cfg,'experiment_mode':cfg['mode'],
          'validation_selects_best_epoch':True,'validation_not_independent_final_test':True}
    export_neural_predictions(validation_tables,root,'validation',meta)
    scoring_root=Path(cfg.get('scoring_output_root',Path(project_root)/'Result/Common_Scoring'/cfg['model_family']/cfg['input_condition']/root.name))
    scoring_locations={}
    if cfg['mode']=='full':
        evaluation=score_neural_run(project_root,root,cfg['model_family'],cfg['input_condition'],split='validation',experiment_mode='full',common_dir=common,bootstrap_reps=int(cfg.get('bootstrap_reps',1000)),output_root=scoring_root)
        scoring_locations['validation']=evaluation['output_dir']
        if test_tables:
            export_neural_predictions(test_tables,root,'test',meta)
            evaluation=score_neural_run(project_root,root,cfg['model_family'],cfg['input_condition'],split='test',experiment_mode='full',final=True,common_dir=common,bootstrap_reps=int(cfg.get('bootstrap_reps',1000)),output_root=scoring_root)
            scoring_locations['test']=evaluation['output_dir']
    else:
        pilot_folder=reserve_execution_directory(scoring_root/'pilot')
        save_json(pilot_folder/'pilot_skipped.json',{'status':'skipped','reason':'Pilot has a partial origin grid; full fixed-row scoring requires mode full.'})
        scoring_locations['pilot']=str(pilot_folder)
    save_json(root/'scoring_locations.json',scoring_locations)
    save_json(root/'completion.json',{'status':'passed','mode':cfg['mode'],'runs':len(summaries),'automatic_common_scoring':cfg['mode']=='full',
                                   'test_scored':bool(test_tables),'training_version':TRAINING_VERSION,'scoring_locations':scoring_locations})
    for summary in summaries:
        summary['experiment_run_root']=str(root);summary['scoring_locations']=scoring_locations
        summary['test_scored']=bool('test' in scoring_locations)
        summary['validation_scored']=bool('validation' in scoring_locations)
        summary_path=root/'Models'/summary['model']/f"seed{int(summary['seed'])}"/'run_summary.json'
        save_json(summary_path,summary)
    save_json(root/'run_summaries.json',summaries)
    return summaries


def run_neural_experiment(bundle,graphs,preprocessing,arrays,model_specs,config,run_root,model_factory,api,project_root,common_dir=None):
    """Track this invocation separately from a previous successful completion."""
    root=reserve_execution_directory(run_root)
    local_config=copy.deepcopy(config);local_config['execution_id']=root.name
    scoring_base=Path(config.get('scoring_output_root',Path(project_root)/'Result/Common_Scoring'/config['model_family']/config['input_condition']))
    local_config['scoring_output_root']=str(scoring_base/root.name)
    source=config.get('resume_source_run')
    if source:
        source=Path(source).resolve()
        if source==root or source.is_relative_to(root):raise ValueError('Resume source must be a previous independent run.')
        model_dir=source/'Models'
        if not model_dir.is_dir():raise FileNotFoundError('Selected resume run has no Models checkpoints: '+str(source))
        copied=[]
        # Copy only fully committed immutable generations, then copy informational
        # contracts. Independent best.pt/last.pt are regenerated from verified state.
        seed_folders=sorted({p.parent for p in model_dir.rglob('Checkpoints') if p.is_dir()})
        if not seed_folders:
            raise ValueError('Resume source has no verified VastAI generations. Original Runpod last.pt requires its original runner.')
        policy=config.get('checkpoint_policy',{})
        for source_seed in seed_folders:
            destination=root/'Models'/source_seed.relative_to(model_dir)
            source_store=CheckpointStore(source_seed,
                snapshot_every=int(policy.get('snapshot_every_epochs',5)),
                keep_recent=int(policy.get('keep_recent',2)))
            from stopping_policy import import_seed_checkpoint
            report=import_seed_checkpoint(source_store,destination,local_config,
                CheckpointStore,save_json,save_learning_curve)
            copied.append(report)
            if report.get('stopping_policy_revision'):
                revision=report['stopping_policy_revision']
                print('[stopping policy revised] '+str(revision),flush=True)
        save_json(root/'resume_copy_manifest.json',{'source_run':str(source),'source_files_modified':False,'copied':copied})
        local_config['resume']=True
    state={'status':'running','mode':config.get('mode'),'run_final_test':bool(config.get('run_final_test',False)),
           'previous_completion_retained':(root/'completion.json').is_file(),'training_version':TRAINING_VERSION}
    save_json(root/'execution_status.json',state)
    try:
        summaries=_run_neural_experiment(bundle,graphs,preprocessing,arrays,model_specs,local_config,root,model_factory,api,project_root,common_dir)
    except BaseException as error:
        save_json(root/'execution_status.json',dict(state,status='failed',error_type=type(error).__name__,error_message=str(error)))
        raise
    save_json(root/'execution_status.json',dict(state,status='passed',completed_runs=len(summaries)))
    return summaries
