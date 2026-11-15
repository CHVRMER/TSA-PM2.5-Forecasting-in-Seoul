"""Independent raw-mask and invariant checks; no training or forecast scoring."""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from regional_inputs import (load_regional_configuration,prepare_regional_condition,
    regional_universe_fingerprints,array_sha256,canonical_sha256,target_context_preprocessing_sha256)


def preserved_api(project):
    path=project/'Model/DCRNN/Training_Update_20261109/Seoul_PM25_DCRNN_AutoScore_Colab.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    ns={'__name__':'_regional29_preserved_inputs_verification'}
    for c in nb['cells']:
        purpose=c['metadata'].get('purpose','')
        if purpose.startswith('preserved_'):exec(compile(''.join(c['source']),purpose,'exec'),ns,ns)
    return ns


def synthetic(config,api):
    rng=np.random.default_rng(615)
    keys=['KR:111121']+config['conditions']['F']['station_keys']
    train=pd.date_range('2021-01-01',periods=72,freq='h',tz='Asia/Seoul')
    later=pd.date_range('2023-01-01',periods=24,freq='h',tz='Asia/Seoul')
    hours=train.append(later).as_unit('ns');positions=np.array([12,24,36,72,75])
    times=hours[positions]
    raw=np.exp(rng.normal(2,.3,(len(hours),len(keys)))).astype(np.float32)
    raw[::7,17]=np.nan
    y=rng.normal(25,3,(len(times),4));eligible=np.ones(y.shape,bool)
    eligible[1,2]=False;y[1,2]=np.nan
    bundle={'nodes':pd.DataFrame({'station_key':keys}), 'pm_values':raw,
        'hourly_index':hours,'origins':pd.DataFrame({'origin_time_kst':times,'origin_split':['train']*3+['validation']*2}),
        'origin_positions':positions,'context_names':['asos108_temperature_now','cal_hour_sin'],
        'context_values':rng.normal(size=(len(times),2)).astype(np.float32),
        'y_raw':y,'eligible':eligible,'horizons':(3,6,12,24),'lookback_hours':12}
    graphs={'distance':np.eye(635,dtype=np.float32),'trajectory_directed':np.tril(np.ones((635,635),dtype=np.float32),-1)}
    frozen=regional_universe_fingerprints(bundle,graphs)
    original=array_sha256(raw)
    _,full_prep,full_arrays,full_contract=prepare_regional_condition(bundle,'F',config,api,graphs,frozen)
    checks=[]
    for condition in config['condition_order']:
        masked,prep,arrays,contract=prepare_regional_condition(bundle,condition,config,api,graphs,frozen)
        allowed=set(config['conditions'][condition]['station_keys'])|{'KR:111121'}
        keep=np.array([k in allowed for k in keys],dtype=bool)
        assert array_sha256(bundle['pm_values'])==original
        assert not np.shares_memory(masked['pm_values'],bundle['pm_values'])
        assert np.array_equal(masked['pm_values'][:,keep],bundle['pm_values'][:,keep],equal_nan=True)
        assert np.isnan(masked['pm_values'][:,~keep]).all()
        assert np.array_equal(arrays['hourly_x'][:,keep],full_arrays['hourly_x'][:,keep])
        assert np.array_equal(arrays['context_x'],full_arrays['context_x']) and np.array_equal(arrays['y_z'],full_arrays['y_z'])
        assert contract['target_context_preprocessing_sha256']==full_contract['target_context_preprocessing_sha256']
        assert contract['allowed_external_series']==len(allowed)-1
        for k in ('y_raw','eligible','origins','origin_positions','context_values','nodes'):
            assert masked[k] is bundle[k]
        checks.append(condition)
    poisoned=dict(bundle,pm_values=bundle['pm_values'].copy())
    for i,k in enumerate(keys):
        if k not in set(config['conditions']['DN']['station_keys'])|{'KR:111121'}:
            poisoned['pm_values'][:,i]=np.inf
    _,a,aa,_=prepare_regional_condition(bundle,'DN',config,api,graphs)
    _,b,bb,_=prepare_regional_condition(poisoned,'DN',config,api,graphs)
    assert canonical_sha256(a)==canonical_sha256(b)
    assert np.array_equal(aa['hourly_x'],bb['hourly_x'])
    try:prepare_regional_condition(poisoned,'DN',config,api,graphs,frozen)
    except ValueError as error:assert 'changed' in str(error)
    else:raise AssertionError('A stale frozen source contract was accepted.')
    bad=copy.deepcopy(config);bad['conditions']['DN']['station_keys']=bad['conditions']['F']['station_keys']
    # Node coverage and graph checks fail before preparing a partial universe.
    smaller=dict(bundle,nodes=bundle['nodes'].iloc[:-1])
    try:prepare_regional_condition(smaller,'DN',config,api,graphs)
    except ValueError:pass
    else:raise AssertionError('A reduced-node universe was accepted.')
    return {'all_29_masks_checked':len(checks),'raw_source_preserved':True,'target_context_labels_preserved':True,
        'retained_node_scalers_and_inputs_identical_to_full':True,'excluded_raw_values_poison_invariant':True,
        'stale_fingerprint_rejected':True,'wrong_node_universe_rejected':True}


def real(config,api,project):
    started=time.perf_counter()
    bundle=api['load_neighbor_bundle'](project/'Dataset',condition='F634_BOTH_W24',
        config_dir=project/'Model/Common/Neighbor_Experiments_QC_20261108',lookback_hours=48)
    graphs,reports=api['build_neighbor_graphs'](bundle,include_trajectory=False)
    assert len(bundle['nodes'])==635 and tuple(bundle['horizons'])==(3,6,12,24)
    frozen=regional_universe_fingerprints(bundle,graphs)
    _,full_prep,full_arrays,full_contract=prepare_regional_condition(bundle,'F',config,api,graphs,frozen)
    target_scaler=target_context_preprocessing_sha256(full_prep)
    raw_hash=array_sha256(bundle['pm_values']);results=[]
    keys=bundle['nodes'].station_key.astype(str).tolist()
    for condition in config['condition_order']:
        masked,prep,arrays,contract=prepare_regional_condition(bundle,condition,config,api,graphs,frozen)
        keep=np.array([key=='KR:111121' or key in config['conditions'][condition]['station_keys'] for key in keys],dtype=bool)
        assert contract['target_context_preprocessing_sha256']==target_scaler
        assert np.array_equal(arrays['hourly_x'][:,keep],full_arrays['hourly_x'][:,keep])
        assert np.array_equal(arrays['context_x'],full_arrays['context_x'])
        assert np.array_equal(arrays['y_z'],full_arrays['y_z'])
        assert all(contract[name]==frozen[name] for name in frozen)
        results.append({'condition':condition,'allowed_external_series':int(keep.sum()-1),
            'excluded_train_observation_counts_zero':True,'retained_node_scalers_exact':True,
            'target_context_preprocessing_sha256':target_scaler,'fixed635_nodes':True})
        del masked,prep,arrays
    assert array_sha256(bundle['pm_values'])==raw_hash
    return {'status':'passed','nodes':635,'origins':len(bundle['origins']),'hourly_rows':len(bundle['hourly_index']),
        'conditions':results,'raw_source_preserved':True,'common_target_context_labels_and_graphs_preserved':True,
        'full_4year_data_loaded_once':True,'graph_test':'preserved distance graph; parent verifies transport-graph training separately',
        'elapsed_seconds':time.perf_counter()-started,'model_training_or_forecast_scoring_performed':False}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--config-dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--skip-real',action='store_true')
    args=parser.parse_args();config=load_regional_configuration(args.config_dir);api=preserved_api(args.project_root)
    result={'status':'passed','schema_version':config['schema_version'],'configuration_sha256':config['config_sha256'],
            'synthetic':synthetic(config,api)}
    if not args.skip_real:result['real']=real(config,api,args.project_root)
    result['module_source_sha256']=__import__('hashlib').sha256((Path(__file__).parent/'regional_inputs.py').read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
