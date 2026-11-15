"""Independent input/target/baseline-preservation tests for the new factor loader."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal,assert_series_equal
from independent_expectations import expected_scopes

FIXED={'train':{3:5762,6:5761,12:5759,24:5755},
       'validation':{3:2531,6:2530,12:2528,24:2524},
       'test':{3:2874,6:2873,12:2871,24:2867}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a,b):
    a=a.copy(deep=False);b=b.copy(deep=False);a.attrs={};b.attrs={}
    assert_frame_equal(a,b,check_exact=True)


def rejected(action,label):
    try:action()
    except (ValueError,KeyError,FileNotFoundError,TypeError):return label
    raise AssertionError('Required guard missing: '+label)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--common-dir',type=Path,required=True)
    parser.add_argument('--config-dir',type=Path,required=True)
    parser.add_argument('--new-history',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    sys.path.insert(0,str(args.common_dir.resolve()))
    import neighbor_data_qc
    from regional_data import RegionalDataLoader
    wanted=expected_scopes(args.project_root,args.archive)
    loader=neighbor_data_qc.NeighborDataLoader(project_root=args.project_root,config_dir=args.config_dir,
                                             source_overrides={'new_qc':args.new_history},strict_hashes=True)
    old=RegionalDataLoader(project_root=args.project_root,
        config_dir=args.project_root/'Model/Common/Regional_Experiments_20261107',strict_hashes=True)
    config=loader.conditions
    for ident,spec in config.items():
        f=spec['factors'];combo=(f['china_east'],f['japan_east'],f['west_selection']=='new25')
        assert set(loader.station_keys(ident))==wanted['conditions'][combo]['stations']
        assert set(loader.feature_names(ident))==wanted['conditions'][combo]['features']
    grid=pd.read_parquet(args.project_root/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet')
    for split,counts in FIXED.items():
        for h,n in counts.items():
            g=grid.loc[grid.split.eq(split)&grid.horizon.eq(h)&grid.eligible]
            assert len(g)==n and np.isfinite(g.y_true).all()
            assert (g.target_time_kst==g.origin_time_kst+pd.Timedelta(hours=h)).all()
    start=pd.Timestamp('2021-01-01',tz='Asia/Seoul');end=pd.Timestamp('2021-01-08',tz='Asia/Seoul')
    frames=loader.prepare(start,end);pilot=[]
    for ident in config:
        for h in (3,6,12,24):
            X,y,meta=loader.load_xy(ident,h,'train',start=start,end=end)
            g=grid.loc[grid.split.eq('train')&grid.horizon.eq(h)&grid.eligible&grid.origin_time_kst.ge(start)&grid.origin_time_kst.lt(end)].sort_values('origin_time_kst')
            assert len(X)==len(y)==len(meta)==len(g)
            assert list(X.columns)==loader.feature_names(ident)
            assert np.array_equal(y.to_numpy(),g.y_true.to_numpy())
            assert pd.DatetimeIndex(meta.origin_time_kst).equals(pd.DatetimeIndex(g.origin_time_kst))
            assert not np.isinf(X.to_numpy(float)).any()
            if ident=='F600_W24':
                ox,oy,om=old.load_xy('F',h,'train',start=start,end=end)
                equal(X,ox);assert_series_equal(y,oy,check_exact=True)
            pilot.append({'condition':ident,'horizon':h,'rows':len(X),'features':X.shape[1]})
    print('Pilot32 and originalF600 baseline values/column-order: passed.',flush=True)
    origins=pd.date_range(start,end,freq='3h',inclusive='left')
    cutoff=origins[28]
    poisoned=future=0
    for ident in config:
        columns=set(loader.feature_names(ident));reference=loader.assemble_inputs(ident,frames,origins)
        mutated={key:frame.copy(deep=True) for key,frame in frames.items()}
        for key,frame in mutated.items():
            for col in set(frame)-columns-{'origin_time_kst'}:frame[col]='EXCLUDED_SOURCE'
            frame['y_pm25_h6']='FUTURE_TARGET'
            frame['met_all_countries_fake']='OUT_OF_SCOPE_AGGREGATE'
        equal(reference,loader.assemble_inputs(ident,mutated,origins));poisoned+=1
        mutated={key:frame.copy(deep=True) for key,frame in frames.items()}
        for key,frame in mutated.items():
            mask=frame.origin_time_kst.gt(cutoff)
            for col in columns&set(frame):
                frame.loc[mask,col]=False if pd.api.types.is_bool_dtype(frame[col].dtype) else 0
        equal(reference.loc[origins<=cutoff].reset_index(drop=True),
              loader.assemble_inputs(ident,mutated,origins[origins<=cutoff]));future+=1
    union=sorted(set(loader.stations.station_key));node_keys=['KR:111121']+union
    masks=[]
    for ident in config:
        mask=loader.node_observation_mask(ident,node_keys)
        assert len(mask)==651 and mask.loc['KR:111121'] and int(mask.sum())==len(loader.station_keys(ident))+1
        assert set(mask.index[mask])==set(loader.station_keys(ident))|{'KR:111121'}
        masks.append({'condition':ident,'enabled_with_Junggu':int(mask.sum())})
    guards=[]
    guards.append(rejected(lambda:loader.load_xy('missing'),'unknown_condition'))
    guards.append(rejected(lambda:loader.load_xy('F600_W24',2),'unknown_horizon'))
    guards.append(rejected(lambda:loader.load_xy('F600_W24',6,'wrong'),'unknown_split'))
    guards.append(rejected(lambda:loader.load_xy('F600_W24',6,'test'),'test_labels_locked'))
    guards.append(rejected(lambda:loader.assemble_inputs('F600_W24',frames,origins.append(origins[:1])),'duplicate_origins'))
    broken={key:frame.copy() for key,frame in frames.items()};broken['v3']=broken['v3'].drop(columns=['seoul_pm25_now'])
    guards.append(rejected(lambda:loader.assemble_inputs('F600_W24',broken,origins),'missing_selected_column'))
    broken={key:frame.copy() for key,frame in frames.items()};broken['v3'].loc[0,'seoul_pm25_now']=np.inf
    guards.append(rejected(lambda:loader.assemble_inputs('F600_W24',broken,origins),'infinite_feature'))
    allnan={key:frame.copy() for key,frame in frames.items()}
    for key,frame in allnan.items():
        cols=[col for col in frame if col!='origin_time_kst'];frame[cols]=frame[cols].astype(float)
        frame.loc[0,cols]=np.nan
    X=loader.assemble_inputs('F635_BOTH_W25',allnan,origins)
    assert len(X)==56 and X.iloc[0].isna().all()
    loader.clear_cache()
    single=origins[3]
    X,y,m=loader.load_xy('F635_BOTH_W25',6,'train',start=single,end=single+pd.Timedelta(seconds=1))
    assert len(X)==len(y)==len(m)==1
    print('Poison/future8each, masks8, guards, allNaN preservation: passed.',flush=True)
    loader.prepare();full=[]
    for split in ('train','validation','test'):
        for ident in config:
            X,y,meta=loader.load_xy(ident,6,split,allow_test=split=='test')
            assert len(X)==len(y)==len(meta)==FIXED[split][6]
            if ident=='F600_W24':
                ox,oy,om=old.load_xy('F',6,split,allow_test=split=='test')
                equal(X,ox);assert_series_equal(y,oy,check_exact=True)
                assert pd.DatetimeIndex(meta.origin_time_kst).equals(pd.DatetimeIndex(om.origin_time_kst))
                del ox,oy,om
            full.append({'split':split,'condition':ident,'horizon':6,'rows':len(X),'features':X.shape[1]})
            del X,y,meta
        print('Full h6 '+split+': 8/8 passed.',flush=True)
    loader.clear_cache()
    report={'status':'passed','conditions':8,'pilot_loads':pilot,'full_h6_loads':full,
            'all_horizon_fixed_rows':FIXED,'baseline_F600_all_values_and_column_order_exact':True,
            'baseline_F600_three_splits_all_h6_rows_exact':True,
            'excluded_and_target_poison_invariance_conditions':poisoned,
            'future_row_perturbation_invariance_conditions':future,
            'graph_masks':masks,'all_input_NaN_row_preserved':True,'single_origin_uncached_read':True,
            'guards':guards,'runtime_input_hashes_checked':True,'no_model_training':True,
            'module_sha256':sha(args.common_dir/'neighbor_data_qc.py'),
            'elapsed_seconds':time.monotonic()-started}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('status','conditions','baseline_F600_all_values_and_column_order_exact','elapsed_seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
