"""Strict cross-condition regional comparisons using the unchanged core scorer.

This module scores complete, predeclared regional jobs only. The changed
observation masks and per-node PM normalizers are permitted differences; fixed
nodes/graphs, targets, context, evaluation calendar, seeds and the within-family
model/training protocol must match. It never trains a model or weakens the
single-condition neural evaluator's guards.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HORIZONS=(3,6,12,24)
INVARIANT_CONTRACT_KEYS=('universe_id','node_order_sha256','graph_sha256',
    'target_observations_sha256','context_values_sha256','context_names_sha256','origin_grid_sha256',
    'y_raw_sha256','eligible_sha256','origin_positions_sha256','source_universe_signature',
    'target_context_preprocessing_sha256','config_sha256')
SCHEMA_VERSION='regional-neural-evaluation-v1.0.0'


def _canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def canonical_sha256(value):return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def array_sha256(array):
    array=np.ascontiguousarray(array)
    digest=hashlib.sha256()
    digest.update(str(array.dtype).encode());digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def node_order_sha256(keys):
    return hashlib.sha256(json.dumps(list(keys),ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


def target_context_preprocessing_sha256(preprocessing):
    p=preprocessing
    value={key:p[key] for key in ('fit_start_inclusive','fit_end_exclusive','context_mean','context_std',
                                  'context_valid_count','context_names')}
    value.update(target_pm_mean=p['pm_mean'][0],target_pm_std=p['pm_std'][0],
                 target_pm_valid_count=p['pm_valid_count'][0],target_key='KR:111121')
    return canonical_sha256(value)


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def _unique(base):
    base=Path(base).resolve();base.mkdir(parents=True,exist_ok=True)
    for _ in range(20):
        ident=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8]
        path=base/ident
        try:path.mkdir(exist_ok=False)
        except FileExistsError:continue
        return path
    raise RuntimeError('Could not allocate a fresh regional-scoring directory.')


def _load_common(project_root,common_dir):
    directory=Path(common_dir) if common_dir else Path(project_root)/'Model/Common'
    paths={name:directory/(name+'.py') for name in ('grid','score')}
    for path in paths.values():
        if not path.is_file():raise FileNotFoundError(path)
    spec=importlib.util.spec_from_file_location('_regional_fixed_grid',paths['grid'])
    grid=importlib.util.module_from_spec(spec);spec.loader.exec_module(grid)
    previous=sys.modules.get('grid')
    try:
        sys.modules['grid']=grid
        spec=importlib.util.spec_from_file_location('_regional_fixed_score',paths['score'])
        scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    finally:
        if previous is None:sys.modules.pop('grid',None)
        else:sys.modules['grid']=previous
    return scorer,grid,{str(path.resolve()):_sha(path) for path in paths.values()}


def _configuration(config):
    protected={}
    if isinstance(config,(str,Path)):
        directory=Path(config)
        if directory.is_file():directory=directory.parent
        raw={name:_read(directory/(name+'.json')) for name in ('experiments','regions','comparisons')}
        protected={str((directory/(name+'.json')).resolve()):_sha(directory/(name+'.json')) for name in raw}
    else:
        raw=dict(config)
        if not set(('experiments','regions','comparisons')).issubset(raw):
            raise ValueError('Pass config_dir or {experiments, regions, comparisons} JSON objects.')
        raw={key:raw[key] for key in ('experiments','regions','comparisons')}
    conditions=raw['experiments']['conditions']
    comparisons=raw['comparisons'].get('comparisons',raw['comparisons'])
    if not isinstance(conditions,dict) or not conditions or not isinstance(comparisons,list) or not comparisons:
        raise ValueError('Declare nonempty regional conditions and comparisons.')
    if 'D67' in conditions:raise ValueError('D67 was removed from the regional protocol.')
    if not {'DN','F'}.issubset(conditions):raise ValueError('Regional protocol requires DN and F.')
    ids=[];pairs=[]
    for item in comparisons:
        ident=item.get('comparison_id',item.get('id'))
        if not ident or item.get('kind') not in ('add','drop','total','foreign_all'):
            raise ValueError('Declare comparison_id and kind add/drop/total for every contrast.')
        candidate,reference=item['candidate'],item['reference']
        if candidate==reference or candidate not in conditions or reference not in conditions:
            raise ValueError('A regional comparison references undeclared or identical conditions.')
        if item['kind']=='add' and reference!='DN':raise ValueError('Individual additions must use the same DN reference.')
        if item['kind']=='drop' and candidate!='F':raise ValueError('Drop comparisons must use F as the candidate and F-minus-region as reference.')
        if item['kind'] in ('total','foreign_all') and (candidate,reference)!=('F','DN'):
            raise ValueError('The total contrast must be F versus DN.')
        ids.append(ident);pairs.append((candidate,reference))
    if len(ids)!=len(set(ids)) or len(pairs)!=len(set(pairs)):
        raise ValueError('Duplicate comparison IDs or condition pairs.')
    return raw,conditions,comparisons,protected


def _training_protocol(plan):
    metadata=dict(plan['training_metadata'])
    # These are exactly the deliberately varied regional inputs/normalizers.
    metadata.pop('data_signature',None);metadata.pop('preprocessing_signature',None)
    configuration=dict(metadata.get('training_configuration',{}))
    for key in ('input_condition','data_signature'):configuration.pop(key,None)
    metadata['training_configuration']=configuration
    return {'model_specs':plan['model_specs'],'seeds':plan['seeds'],'metadata':metadata}


def _normalizer_contract(path,contract,keys,selected):
    p=_read(path)
    if p['node_order']!=keys:raise ValueError('Preprocessing node order differs from the fixed graph nodes.')
    excluded=[i for i,key in enumerate(keys) if key!='KR:111121' and key not in selected]
    counts=np.asarray(p['pm_valid_count'])
    if len(counts)!=len(keys) or (excluded and np.any(counts[excluded]!=0)):
        raise ValueError('Excluded-region observations reached the training normalizer.')
    if target_context_preprocessing_sha256(p)!=contract['target_context_preprocessing_sha256']:
        raise ValueError('Target/context preprocessing differs from the predeclared regional contract.')


def _validate_job(record,condition,spec,scorer,grid_file,config_digest,split):
    root=Path(record['run_root']).resolve()
    contract_path=Path(record.get('regional_contract_path',root/'regional_contract.json')).resolve()
    paths={name:root/name for name in ('scoring_plan.json','node_order.csv','preprocessing.json','completion.json','execution_status.json')}
    for path in [contract_path,*paths.values()]:
        if not path.is_file():raise FileNotFoundError(path)
    contract=_read(contract_path);plan=_read(paths['scoring_plan.json'])
    for name in INVARIANT_CONTRACT_KEYS:
        if name not in contract:raise ValueError('Incomplete regional contract: '+name)
    for name in ('observation_mask_applied_before_fit','excluded_values_are_normalized_zero_and_missing_one',
                 'fixed_node_order_and_graphs','target_context_labels_preserved','latent_relay_nodes_retained'):
        if contract.get(name) is not True:raise ValueError('Regional input contract must guarantee: '+name)
    if contract.get('condition_id')!=condition or contract['config_sha256']!=config_digest:
        raise ValueError('Regional condition or configuration digest differs from the predeclared plan.')
    selected=set(spec['station_keys'])
    if set(contract['allowed_station_keys'])!=selected:
        raise ValueError('Allowed regional station list differs from the condition definition.')
    nodes=pd.read_csv(paths['node_order.csv'],dtype={'station_key':str})
    keys=nodes.station_key.tolist()
    if not keys or keys[0]!='KR:111121' or len(set(keys))!=len(keys) or not selected.issubset(keys):
        raise ValueError('Fixed nodes must be unique, contain the target first, and represent all allowed observations.')
    if node_order_sha256(keys)!=contract['node_order_sha256']:
        raise ValueError('Actual graph node order differs from the regional contract.')
    excluded=set(keys)-selected-{'KR:111121'}
    if set(contract['excluded_station_keys'])!=excluded:
        raise ValueError('Excluded station list does not exactly complement the selected observations.')
    mask=np.asarray([key=='KR:111121' or key in selected for key in keys],dtype=bool)
    if array_sha256(mask)!=contract['observation_mask_sha256']:
        raise ValueError('Regional observation mask differs from its declared station selection.')
    graph_paths=sorted((root/'Graphs').glob('*.npy'))
    graphs={path.stem:array_sha256(np.load(path,allow_pickle=False)) for path in graph_paths}
    if graphs!=contract['graph_sha256']:raise ValueError('Actual graph supports differ from the fixed regional graph.')
    _normalizer_contract(paths['preprocessing.json'],contract,keys,selected)
    completion=_read(paths['completion.json']);state=_read(paths['execution_status.json'])
    if completion.get('status')!='passed' or completion.get('mode')!='full' or state.get('status')!='passed':
        raise ValueError('Regional scoring requires a completed full model job, not pilot/failed/partial results.')
    if plan['model_family']!=record['model_family'] or plan['input_condition']!=condition:
        raise ValueError('Job identity differs from the frozen neural scoring plan.')
    if tuple(plan['training_metadata']['horizons'])!=HORIZONS:
        raise ValueError('The regional protocol requires all four fixed forecast horizons.')
    if len(plan['model_specs'])!=1:
        raise ValueError('Regional primary analysis requires exactly one fixed graph/model configuration per family.')
    prediction_path=root/'Exports'/f'{split}_predictions_by_seed.parquet'
    if not prediction_path.is_file():raise FileNotFoundError(prediction_path)
    frame=scorer.prepare_predictions(pd.read_parquet(prediction_path))
    if set(frame.prediction_kind)!={'seed'} or not frame.input_condition.eq(condition).all():
        raise ValueError('Regional inputs must be only raw seed exports with their true condition IDs.')
    if 'split' in frame and not frame.split.eq(split).all():raise ValueError('Prediction export split mismatch.')
    if not np.isfinite(frame.y_hat.to_numpy(float)).all():raise ValueError('Nonfinite regional forecasts; no rows may be removed.')
    if 'experiment_mode' in frame and not frame.experiment_mode.eq('full').all():raise ValueError('Pilot exports are not regional results.')
    if 'common_export_unclipped' not in frame or not frame.common_export_unclipped.eq(True).all():
        raise ValueError('Regional forecasts must be explicitly raw/unclipped original units.')
    names={item['model'] for item in plan['model_specs']}
    if set(frame.model)!=names or set(frame.seed)!=set(plan['seeds']):
        raise ValueError('Exported model or seed set differs from its predeclared neural plan.')
    if set(frame.data_version)!={plan['training_metadata']['data_signature']}:
        raise ValueError('Prediction data version differs from the fitted masked-observation data signature.')
    for seed,part in frame.groupby('seed'):
        if set(part.horizon)!=set(HORIZONS):raise ValueError('Each seed must contain all four horizons.')
    if 'target_time_kst' in frame:
        actual=pd.DatetimeIndex(pd.to_datetime(frame.target_time_kst,utc=True)).tz_convert('Asia/Seoul').as_unit('ns')
        expected=pd.DatetimeIndex(frame.origin_time_kst+pd.to_timedelta(frame.horizon,unit='h')).as_unit('ns')
        if not np.array_equal(actual.asi8,expected.asi8):raise ValueError('Regional target timestamps differ from origin+horizon.')
    protected={str(path.resolve()):_sha(path) for path in [contract_path,prediction_path,*paths.values(),*graph_paths]}
    sidecar=prediction_path.with_suffix('.meta.json')
    if sidecar.is_file():protected[str(sidecar.resolve())]=_sha(sidecar)
    provenance={'path':str(prediction_path),'sha256':_sha(prediction_path),'sidecar':_read(sidecar) if sidecar.is_file() else None}
    return {'record':dict(record),'contract':contract,'plan':plan,'frame':frame,'keys':keys,
            'model':next(iter(names)),'protected':protected,'provenance':provenance}


def _resolved_comparisons(jobs,comparisons,families,seeds):
    by={(job['record']['model_family'],job['record']['condition']):job for job in jobs}
    declarations=[];pairs=[]
    for family in families:
        for spec in comparisons:
            candidate=by[(family,spec['candidate'])]['model'];reference=by[(family,spec['reference'])]['model']
            kinds=[('seed',seed) for seed in seeds]+([('ensemble','')] if len(seeds)>=2 else [])
            for kind,seed in kinds:
                a=f"{candidate}|{spec['candidate']}|{kind}"+(f'|seed={seed}' if seed else '')
                b=f"{reference}|{spec['reference']}|{kind}"+(f'|seed={seed}' if seed else '')
                pairs.append((a,b));declarations.append(dict(spec,model_family=family,model=candidate,
                    comparison_id=spec.get('comparison_id',spec.get('id')),prediction_kind=kind,seed=seed,
                    candidate_run_id=a,reference_run_id=b))
    return declarations,pairs


def _regional_gains(result,declarations):
    metrics=result['metrics'].set_index(['run_id','horizon'])
    tests=result['tests'].set_index(['run_id','versus_run_id','horizon'])
    rows=[]
    for spec in declarations:
        a,b=spec['candidate_run_id'],spec['reference_run_id']
        for h in HORIZONS:
            candidate=metrics.loc[(a,h)];reference=metrics.loc[(b,h)];test=tests.loc[(a,b,h)]
            denominator=float(candidate.RMSE if spec['kind']=='drop' else reference.RMSE)
            gain=float(reference.RMSE-candidate.RMSE)
            rows.append({**spec,'horizon':h,'primary_horizon':h==6,'n':int(candidate.n),
                'candidate_RMSE':float(candidate.RMSE),'reference_RMSE':float(reference.RMSE),
                'RMSE_gain':gain,'RMSE_gain_pct':100*gain/denominator if denominator>0 else np.nan,
                'percentage_denominator':'RMSE(F)' if spec['kind']=='drop' else 'RMSE(reference)',
                'MAE_gain':float(reference.MAE-candidate.MAE),
                'gain_ci95_low':-float(test.ci95_high),'gain_ci95_high':-float(test.ci95_low),
                'dm_p':float(test.p),'p_holm':float(test.p_holm),'improved':bool(test.improved),
                'interpretation':'Predictive observation-information value; not pollutant mass/source contribution.'})
    return pd.DataFrame(rows)


def _plots(gains,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure_dir=directory/'Figures';figure_dir.mkdir(exist_ok=True)
    for family,part in gains.groupby('model_family',sort=False):
        selected=part.loc[part.prediction_kind.eq('ensemble')]
        if selected.empty:
            selected=part.loc[part.prediction_kind.eq('seed')].groupby(['kind','region_id','horizon'],dropna=False,as_index=False).RMSE_gain.mean()
            suffix='mean seed gains (no ensemble available)'
        else:suffix='prediction ensemble'
        for kind in ('add','drop'):
            subset=selected.loc[selected.kind.eq(kind)]
            if subset.empty:continue
            table=subset.pivot(index='region_id',columns='horizon',values='RMSE_gain').reindex(columns=HORIZONS)
            limit=max(float(np.nanmax(np.abs(table.to_numpy(float)))),1e-8)
            fig,ax=plt.subplots(figsize=(7.8,max(4.,.42*len(table)+1.8)))
            image=ax.imshow(table.to_numpy(float),aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
            ax.set_xticks(range(4),HORIZONS);ax.set_yticks(range(len(table)),table.index)
            ax.set_xlabel('Forecast horizon (hours)');ax.set_ylabel('Declared region')
            ax.set_title(f'{family}: regional {kind} value | {suffix}\nPositive gain = lower forecast error',fontsize=10)
            for i in range(len(table)):
                for j in range(4):
                    value=float(table.iloc[i,j]);color='white' if abs(value)>.6*limit else '#202020'
                    ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,color=color)
            fig.colorbar(image,ax=ax,label='RMSE gain (ug/m3)');fig.tight_layout()
            fig.savefig(figure_dir/(family+'_'+kind+'_rmse_gain.png'),dpi=170);plt.close(fig)


def score_regional_runs(project_root,completed_runs,experiment_config,output_root,split='validation',
                        final=False,common_dir=None,grid_path=None,bootstrap_reps=1000,bootstrap_seed=0,
                        hac_lag=None,make_plots=True,expected_families=('DCRNN','GraphWaveNet')):
    """Score every predeclared condition/family jointly; never select successful rows.

    completed_runs: {condition, model_family, run_root, regional_contract_path?}.
    experiment_config: directory with experiments/regions/comparisons.json or a
    mapping containing those full JSON objects. Contract may live in an outer
    job directory and is read-only. One invocation defines the global Holm family.
    """
    if split not in ('validation','test'):raise ValueError('Only validation or test is supported.')
    if split=='test' and not final:raise ValueError('2024 test requires final=True before reading any test predictions.')
    raw,conditions,comparisons,config_sources=_configuration(experiment_config)
    digest=canonical_sha256(raw)
    families=list(expected_families)
    if not families or len(families)!=len(set(families)):raise ValueError('Declare unique nonempty model families.')
    records=[dict(row) for row in completed_runs]
    for row in records:
        if 'condition' not in row and 'condition_id' in row:row['condition']=row['condition_id']
    identities=[(row['model_family'],row['condition']) for row in records]
    wanted={(family,condition) for family in families for condition in conditions}
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate family/condition completed jobs.')
    if set(identities)!=wanted:
        raise ValueError('Regional comparison requires all predeclared jobs; missing='+str(sorted(wanted-set(identities)))+
                         '; unexpected='+str(sorted(set(identities)-wanted)))
    scorer,grid_module,common_hashes=_load_common(project_root,common_dir)
    grid_file=Path(grid_path).resolve() if grid_path else Path(project_root)/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
    if not grid_file.is_file():raise FileNotFoundError(grid_file)
    protected={**common_hashes,**config_sources,str(grid_file.resolve()):_sha(grid_file)}
    jobs=[];invariant=None;protocols={};seeds=None;family_models={}
    for record in records:
        job=_validate_job(record,record['condition'],conditions[record['condition']],scorer,grid_file,digest,split)
        expected_universe=raw['experiments'].get('universe_id')
        expected_nodes=raw['experiments'].get('expected_node_count')
        if expected_universe is not None and job['contract']['universe_id']!=expected_universe:
            raise ValueError('Job uses a different universe from the declared regional configuration.')
        if expected_nodes is not None and len(job['keys'])!=expected_nodes:
            raise ValueError('Job node count differs from the declared fixed-node universe.')
        value={key:job['contract'][key] for key in INVARIANT_CONTRACT_KEYS}
        if invariant is None:invariant=value
        elif _canonical(value)!=_canonical(invariant):raise ValueError('Regional jobs do not share the same fixed inputs/graphs/targets/calendar.')
        declared_seeds=job['plan']['seeds']
        if seeds is None:seeds=declared_seeds
        elif set(seeds)!=set(declared_seeds):raise ValueError('All regional jobs must use the same predeclared seed set.')
        family=record['model_family'];protocol=_training_protocol(job['plan'])
        if family not in protocols:protocols[family]=protocol;family_models[family]=job['model']
        elif _canonical(protocol)!=_canonical(protocols[family]):
            raise ValueError('Model/training settings changed between regional conditions within '+family)
        protected.update(job['protected']);jobs.append(job)
    if len(set(family_models.values()))!=len(family_models):raise ValueError('Model families require distinct model names.')
    declarations,pairs=_resolved_comparisons(jobs,comparisons,families,seeds)
    invocation=_unique(Path(output_root)/split)
    additional=invocation/'Additional_Analysis';additional.mkdir()
    declaration={'schema_version':SCHEMA_VERSION,'config_sha256':digest,'conditions':list(conditions),
        'families':families,'seeds':seeds,'split':split,'final':bool(final),'invariant_contract':invariant,
        'within_family_training_protocols':protocols,'declared_region_pairs':comparisons,
        'resolved_run_id_pairs':[list(p) for p in pairs],
        'allowed_differences':['condition_id','observation_mask','allowed/excluded_station_keys','per-node PM normalizers','data_signature'],
        'holm_scope':'All finite p-values in this single invocation, including existing baseline comparisons, both families, all horizons, seeds and ensembles.',
        'target_context_preprocessing_same':True,'rows_must_not_be_intersected_or_dropped':True,
        'bootstrap_reps':int(bootstrap_reps),'bootstrap_seed':int(bootstrap_seed),'hac_lag':hac_lag}
    _json(additional/'regional_comparison_plan.json',declaration)
    predictions=pd.concat([job['frame'] for job in jobs],ignore_index=True)
    result=scorer.score_frames(grid_module.read_frame(grid_file),predictions,split=split,pairs=pairs,
        final=final,bootstrap_reps=bootstrap_reps,bootstrap_seed=bootstrap_seed,hac_lag=hac_lag,make_ensembles=True)
    for name in ('metrics','seed_metric_summary','tests','conditions'):
        result[name].to_csv(invocation/(name+'.csv'),index=False,encoding='utf-8-sig')
    result['score_rows'].to_parquet(invocation/'score_rows.parquet',index=False)
    audit=dict(result['audit']);audit.update(written_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        grid_path=str(grid_file.resolve()),grid_sha256=_sha(grid_file),prediction_sources=[job['provenance'] for job in jobs])
    _json(invocation/'scoring_audit.json',audit)
    gains=_regional_gains(result,declarations)
    gains.to_csv(additional/'regional_gains.csv',index=False,encoding='utf-8-sig')
    gains.loc[gains.prediction_kind.eq('seed')].to_csv(additional/'regional_gains_by_seed.csv',index=False,encoding='utf-8-sig')
    gains.loc[gains.prediction_kind.eq('ensemble')].to_csv(additional/'regional_gains_ensemble.csv',index=False,encoding='utf-8-sig')
    group=['model_family','model','comparison_id','kind','region_id','candidate','reference','horizon']
    individual=gains.loc[gains.prediction_kind.eq('seed')]
    summary=individual.groupby(group,dropna=False).agg(seed_count=('seed','size'),
        RMSE_gain_mean=('RMSE_gain','mean'),RMSE_gain_std=('RMSE_gain','std'),MAE_gain_mean=('MAE_gain','mean')).reset_index()
    summary['summary_kind']='Mean and sample SD of individual seed gains; not an ensemble forecast score or CI.'
    summary.to_csv(additional/'regional_seed_gain_summary.csv',index=False,encoding='utf-8-sig')
    for kind in ('seed','ensemble'):
        result['metrics'].loc[result['metrics'].prediction_kind.eq(kind)].to_csv(additional/('metrics_'+('by_seed' if kind=='seed' else 'ensemble')+'.csv'),index=False,encoding='utf-8-sig')
    selected=result['tests'].loc[[tuple(pair) in set(pairs) for pair in result['tests'][['run_id','versus_run_id']].itertuples(index=False,name=None)]]
    selected.to_csv(additional/'regional_tests.csv',index=False,encoding='utf-8-sig')
    if make_plots:_plots(gains,additional)
    if any(_sha(path)!=value for path,value in protected.items()):
        raise ValueError('A predeclared config, grid, source model, contract or prediction changed during regional scoring.')
    _json(additional/'regional_scoring_audit.json',{'status':'passed','schema_version':SCHEMA_VERSION,
        'protected_source_sha256':protected,'job_count':len(jobs),'region_comparison_count':len(comparisons),
        'resolved_run_pair_count':len(pairs),'regional_test_rows':len(selected),
        'global_holm_before_filtering':True,'standard_six_file_schema_unchanged':True,
        'core_audit_sha256':_sha(invocation/'scoring_audit.json'),'2024_scored':split=='test',
        'validation_is_early_stopping_diagnostic_not_independent_final_test':split=='validation'})
    return {'status':'passed','output_dir':str(invocation),'additional_analysis_dir':str(additional),
            'job_count':len(jobs),'condition_count':len(conditions),'model_families':families,
            'comparison_count':len(comparisons),'regional_test_rows':len(selected),
            'fixed_calendar_rules_changed':False,'2024_scored':split=='test'}
