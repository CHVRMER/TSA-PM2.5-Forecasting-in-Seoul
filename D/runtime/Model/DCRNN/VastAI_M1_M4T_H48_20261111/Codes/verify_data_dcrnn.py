"""Test six-arm masks with preserved preprocessors and train-only graph reads.

No original data is changed. Torch-free extraction of the existing numerical
preprocessor functions permits these data-contract checks on CPU-only hosts.
"""
from pathlib import Path
import argparse
import ast
import copy
import json
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import data_experiments as data
from trajectory_features import array_hash, pm_feature_layout
from horizon_data import HORIZONS, build_exact_labels


def preserved_api(project):
    path=Path(project)/'Model/DCRNN/V3_Data_20261107/Codes/original_data.py'
    source=path.read_text(encoding='utf-8')
    tree=ast.parse(source)
    wanted={'_dcrnn_moments','fit_dcrnn_preprocessors','transform_dcrnn_bundle'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if {n.name for n in functions}!=wanted:raise ValueError('Preserved numerical preprocessor API changed.')
    namespace={'np':np,'pd':pd,'DCRNN_TIMEZONE':'Asia/Seoul'}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace,namespace)
    return namespace


def fixture(api):
    keys=[data.TARGET]+[f'KR:{i}' for i in range(514)]+[f'CN:{i}' for i in range(77)]+[f'JP:{i}' for i in range(42)]+['MN:0']
    nodes=pd.DataFrame({'station_key':keys,'country':[k.split(':')[0] for k in keys]})
    nodes['latitude']=37.5;nodes['longitude']=126.98
    cn=nodes.index[nodes.country.eq('CN')]
    nodes.loc[cn[21:],'longitude']=105.
    nodes.loc[nodes.country.eq('JP'),'longitude']=140.
    nodes.loc[nodes.country.eq('MN'),'longitude']=106.
    hours=pd.date_range('2022-12-30','2023-01-02',freq='h',tz='Asia/Seoul')
    origins=pd.DatetimeIndex([pd.Timestamp('2022-12-31',tz='Asia/Seoul'),
                             pd.Timestamp('2023-01-01',tz='Asia/Seoul'),
                             pd.Timestamp('2023-01-01 03:00',tz='Asia/Seoul')])
    raw=(np.arange(len(hours))[:,None]%13 + np.arange(635)[None,:]%17+1.).astype(np.float32)
    raw[4,3]=np.nan
    context=np.arange(3*53,dtype=np.float32).reshape(3,53)
    y_raw,eligible,target_times=build_exact_labels(hours,raw[:,0],origins,
        ['train','validation','validation'])
    origin_table=pd.DataFrame({'origin_time_kst':origins,
        'origin_split':['train','validation','validation']})
    for j,h in enumerate(HORIZONS):
        origin_table[f'y_pm25_h{h}']=y_raw[:,j]
        origin_table[f'eligible_h{h}']=eligible[:,j]
        origin_table[f'target_time_h{h}']=target_times[h]
    bundle={'nodes':nodes,'hourly_index':hours,'pm_values':raw,
            'origins':origin_table,
            'context_values':context,'context_names':[f'asos_fixture_{i}' for i in range(53)],
            'y_raw':y_raw,'eligible':eligible,'origin_positions':hours.get_indexer(origins),
            'horizons':HORIZONS,'lookback_hours':48}
    names=[f'neutral_{i}' for i in range(114)]+[r[0] for r in pm_feature_layout()]
    countries=['NEUTRAL']*114+[r[1] for r in pm_feature_layout()]
    values=np.arange(3*322,dtype=np.float32).reshape(3,322)
    a=np.zeros((635,635),np.float32);a[1,0]=1.
    return {'bundle':bundle,'adjacency':a,'api':api,
            'source_contract':{'settings':data._settings({}),'graph_kind':'trajectory_directed',
                'graph_sha256':array_hash(a),'raw_pm_sha256':array_hash(raw),
                'base_context_sha256':array_hash(context),'build_trajectory':True,
                'source_signature':'synthetic-data-contract'},
            'conditions':data.define_conditions(bundle),
            'trajectory':{'values':values,'names':names,'countries':countries},
            'base_context_count':53,'baseline_preprocessing':api['fit_dcrnn_preprocessors'](bundle)}


def refresh(universe):
    result=copy.deepcopy({k:v for k,v in universe.items() if k!='api'})
    result['api']=universe['api']
    b=result['bundle'];c=result['source_contract']
    c['raw_pm_sha256']=array_hash(b['pm_values']);c['base_context_sha256']=array_hash(b['context_values'])
    result['baseline_preprocessing']=result['api']['fit_dcrnn_preprocessors'](b)
    return result


def check_masks_and_future(api):
    universe=fixture(api)
    raw=universe['bundle']['pm_values'].copy();context=universe['bundle']['context_values'].copy()
    expected={'M1':(514,0),'M2':(535,0),'M3':(591,0),'M4':(634,0),'M3T':(591,218),'M4T':(634,322)}
    for name,(count,tcount) in expected.items():
        p=data.prepare_condition(universe,name,{})
        assert p['contract']['allowed_external_series']==count
        assert len(p['contract']['trajectory_allowed_names'])==tcount
        assert p['arrays']['hourly_x'].shape==(len(raw),635,2)
        assert p['arrays']['context_x'].shape==(3,750)
        assert p['arrays']['y_z'].shape==(3,6)
        assert p['contract']['horizons']==list(HORIZONS)
        assert p['contract']['graph_kind']=='trajectory_directed'
        assert np.array_equal(p['bundle']['y_raw'],universe['bundle']['y_raw'],equal_nan=True)
        assert np.array_equal(p['bundle']['eligible'],universe['bundle']['eligible'])
    assert np.array_equal(raw,universe['bundle']['pm_values'],equal_nan=True)
    assert np.array_equal(context,universe['bundle']['context_values'],equal_nan=True)
    first=data.prepare_condition(universe,'M3T',{})
    altered=refresh(universe)
    foreign=altered['bundle']['nodes'].country.isin(['JP','MN']).to_numpy()
    altered['bundle']['pm_values'][:,foreign]=987654.
    tc=np.array(altered['trajectory']['countries'])
    altered['trajectory']['values'][:,np.isin(tc,['JP','MN'])]=999999.
    altered=refresh(altered)
    second=data.prepare_condition(altered,'M3T',{})
    assert np.array_equal(first['arrays']['hourly_x'],second['arrays']['hourly_x'])
    assert np.array_equal(first['arrays']['context_x'],second['arrays']['context_x'])
    assert first['preprocessing']==second['preprocessing']
    # Change actual validation/test observations after an origin. Fitted moments
    # and that origin's exact input window/context must remain unchanged.
    changed=refresh(universe);chosen=1
    end=int(changed['bundle']['origin_positions'][chosen])
    changed['bundle']['pm_values'][end+1:]=55555.
    changed['bundle']['context_values'][chosen+1:]=33333.
    changed['trajectory']['values'][chosen+1:]=77777.
    changed=refresh(changed)
    third=data.prepare_condition(changed,'M3T',{})
    assert first['preprocessing']==third['preprocessing']
    assert np.array_equal(first['arrays']['hourly_x'][:end+1],third['arrays']['hourly_x'][:end+1])
    assert np.array_equal(first['arrays']['context_x'][chosen],third['arrays']['context_x'][chosen])
    # Changing the graph after it was frozen must be rejected.
    bad=refresh(universe);bad['adjacency'][1,0]=2.
    try:data.prepare_condition(bad,'M1',{})
    except ValueError:pass
    else:raise AssertionError('Mutated fixed graph was accepted.')
    try:data.prepare_condition(universe,'M1',{'model':{'graph':'distance'}})
    except ValueError:pass
    else:raise AssertionError('Undeclared graph experiment was accepted.')


def check_train_only_graph(project,temp_parent):
    graph_path=Path(project)/'Model/DCRNN/V3_Data_20261107/Codes/original_graphs.py'
    namespace={'__name__':'_verified_original_graphs'}
    exec(compile(graph_path.read_text(encoding='utf-8'),str(graph_path),'exec'),namespace,namespace)
    adapter_path=Path(project)/'Model/DCRNN/V3_Data_20261107/Codes/v3_adapter.py'
    tree=ast.parse(adapter_path.read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='expanded_trajectory_adjacency')
    namespace['pq']=pq
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(adapter_path),'exec'),namespace,namespace)
    nodes=pd.DataFrame({'station_key':[data.TARGET,'KR:1','CN:1','JP:1'],
        'latitude':[37.5,37.7,40.,34.],'longitude':[126.98,126.5,120.,135.]})
    arrivals=pd.DatetimeIndex([pd.Timestamp(f'{year}-01-05',tz='Asia/Seoul') for year in (2021,2022,2023,2024)])
    points=[];edges=[]
    for number,arrival in enumerate(arrivals):
        for age in range(1,49):
            row={'point_id':f'{number}:{age}','arrival_time_kst':arrival,'lookback_hours':age}
            points.append(row)
            edges.append(dict(row,station_key='CN:1' if age>24 else 'KR:1',distance_km=10.))
    points=pd.DataFrame(points);edges=pd.DataFrame(edges)
    with tempfile.TemporaryDirectory(dir=temp_parent) as name:
        root=Path(name)
        (root/'matching_manifest.json').write_text(json.dumps({'parameters':{'radius_km':75}}),encoding='utf-8')
        points.to_parquet(root/'trajectory_point_coverage.parquet',index=False)
        edges.to_parquet(root/'trajectory_station_neighbors.parquet',index=False)
        b={'dataset_root':root,'node_set':'neighbor_F634_BOTH_W24','nodes':nodes}
        a,report=namespace['expanded_trajectory_adjacency'](b,matching_folder=root)
        edges.loc[edges.arrival_time_kst.dt.year>=2023,'station_key']='JP:1'
        edges.loc[edges.arrival_time_kst.dt.year>=2023,'distance_km']=0.
        edges.to_parquet(root/'trajectory_station_neighbors.parquet',index=False)
        altered,r2=namespace['expanded_trajectory_adjacency'](b,matching_folder=root)
        assert np.array_equal(a,altered)
        assert report['n_train_arrivals']==2 and report['arrival_max'].startswith('2022-')
        assert report['training_arrivals_only'] and not report['pollution_or_target_values_used']
        assert np.count_nonzero(a[3])==0 and a[2,1]>0 and a[1,0]>0
        # The graph-only M1 path must still fingerprint geometry and graph code,
        # even when T context is intentionally not generated.
        required=['Model/DCRNN/V3_Data_20261107/Codes/original_data.py',
            'Model/DCRNN/V3_Data_20261107/Codes/original_graphs.py',
            'Model/DCRNN/V3_Data_20261107/Codes/v3_adapter.py',
            'Model/Common/Neighbor_Experiments_QC_20261108/experiments.json',
            'Model/Common/Neighbor_Experiments_QC_20261108/station_registry.csv',
            'Dataset/Trajectory_Matching_National_v4/station_registry.csv',
            'Dataset/Origin_Features_2021_2024_v3/feature_dictionary.csv',
            'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet',
            'adapter.py']
        for relative in required:
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
        paths=data._source_paths(root,{'source_fingerprints':[],'neighbor_matching_folder':str(root)},
                                  root/'adapter.py',False,build_transport_graph=True)
        assert (root/'trajectory_station_neighbors.parquet').resolve() in paths
        assert (root/'trajectory_point_coverage.parquet').resolve() in paths
        assert (root/'matching_manifest.json').resolve() in paths
        assert (root/'Model/DCRNN/V3_Data_20261107/Codes/original_graphs.py').resolve() in paths


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[4])
    ap.add_argument('--temp-dir',type=Path)
    args=ap.parse_args()
    if args.temp_dir:args.temp_dir.mkdir(parents=True,exist_ok=True)
    api=preserved_api(args.project_root)
    check_masks_and_future(api)
    check_train_only_graph(args.project_root,args.temp_dir)
    print(json.dumps({'tests':'passed','full_635_node_condition_masks':True,
        'six_horizons':list(HORIZONS),
        'preserved_train_only_preprocessors':True,'M3T_JP_MN_mutation_invariance':True,
        'future_observation_and_context_mutation_invariance':True,'source_arrays_unchanged':True,
        'fixed_graph_mutation_rejected':True,'distance_graph_rejected':True,
        'validation_test_trajectory_mutation_does_not_change_graph':True,
        'graph_only_M1_fingerprints_matching_geometry_and_graph_code':True}))


if __name__=='__main__':main()
