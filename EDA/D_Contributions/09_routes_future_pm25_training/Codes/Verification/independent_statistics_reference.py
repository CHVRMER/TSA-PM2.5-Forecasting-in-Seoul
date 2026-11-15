"""Independent pandas recomputation of geographic-passage outcome statistics.

Passage input is LONG: origin_time_kst, region_id, window, passed (boolean).
The authoritative future targets and current concentration are read separately
from existing project files. This script never imputes a target or current PM.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

HORIZONS=(3,6,12,24)
BINS=['0_15','15_35','35_75','75_inf']

def kst(values):
    x=pd.DatetimeIndex(pd.to_datetime(values))
    return (x.tz_localize('Asia/Seoul') if x.tz is None else x.tz_convert('Asia/Seoul')).as_unit('ns')

def source_reference(dataset):
    f=pd.read_parquet(dataset/'Evaluation_Contract_20261107/evaluation_grid.parquet',
                      filters=[('split','==','train')],
                      columns=['origin_time_kst','horizon','target_time_kst','y_true','eligible','season'])
    f['origin_time_kst']=kst(f.origin_time_kst)
    f['target_time_kst']=kst(f.target_time_kst)
    expected=pd.date_range('2021-01-01','2023-01-01',freq='3h',inclusive='left',tz='Asia/Seoul')
    assert len(expected)==5840 and len(f)==5840*4
    assert not f.duplicated(['origin_time_kst','horizon']).any()
    assert set(f.horizon)==set(HORIZONS)
    for h,n in zip(HORIZONS,[5762,5761,5759,5755]):
        g=f[f.horizon.eq(h)]
        assert kst(g.origin_time_kst.sort_values()).equals(expected.as_unit('ns'))
        assert int(g.eligible.sum())==n
        assert (g.loc[g.eligible,'target_time_kst']<pd.Timestamp('2023-01-01',tz='Asia/Seoul')).all()
        assert np.array_equal(kst(g.target_time_kst).asi8,
            (kst(g.origin_time_kst)+pd.Timedelta(hours=h)).asi8)
    current=pd.read_parquet(dataset/'Origin_Features_2021_2024/origin_features_2021_2024_kst.parquet',
                filters=[('origin_split','==','train')],columns=['origin_time_kst','seoul_pm25_now'])
    current['origin_time_kst']=kst(current.origin_time_kst)
    assert len(current)==5840 and current.origin_time_kst.is_unique
    assert int(current.seoul_pm25_now.isna().sum())==77
    current['current_bin']=pd.cut(current.seoul_pm25_now,[0,15,35,75,np.inf],right=False,labels=BINS)
    f=f.merge(current,on='origin_time_kst',validate='many_to_one')
    assert len(f)==23360
    f['stratum']=f.season.astype(str)+'|'+f.current_bin.astype('string').fillna('UNKNOWN')
    return f

def calculate(source,passage,minimum=10):
    passage=passage.copy();passage['origin_time_kst']=kst(passage.origin_time_kst)
    assert not passage.duplicated(['origin_time_kst','region_id','window']).any()
    if not pd.api.types.is_bool_dtype(passage.passed):
        mapped=passage.passed.map({True:True,False:False,0:False,1:True,'true':True,'false':False,'True':True,'False':False})
        assert mapped.notna().all();passage['passed']=mapped.astype(bool)
    original=pd.Index(source.origin_time_kst.unique())
    raw_rows=[];adjusted_rows=[];strata_rows=[];route_rows=[]
    for (region,window),route in passage.groupby(['region_id','window'],sort=False):
        assert len(route)==5840 and pd.Index(route.origin_time_kst).sort_values().equals(original.sort_values())
        route_rows.append({'region_id':region,'window':window,'origins':5840,
                           'pass_origins':int(route.passed.sum()),'pass_pct':100*route.passed.mean()})
        merged=source.merge(route,on='origin_time_kst',validate='many_to_one')
        for h in HORIZONS:
            eligible=merged[merged.horizon.eq(h)&merged.eligible].copy()
            assert np.isfinite(eligible.y_true).all()
            for side in [True,False]:
                g=eligible[eligible.passed.eq(side)]
                raw_rows.append({'region_id':region,'window':window,'horizon':h,'passed':side,
                   'n':len(g),'future_mean':g.y_true.mean(),'future_high_pct':100*g.y_true.ge(35).mean() if len(g) else np.nan,
                   'current_pm_valid_n':int(g.seoul_pm25_now.notna().sum()),'current_mean':g.seoul_pm25_now.mean()})
            valid=eligible[eligible.current_bin.notna()].copy()
            records=[]
            for stratum,g in valid.groupby('stratum',sort=True):
                yes=g[g.passed];no=g[~g.passed]
                records.append({'region_id':region,'window':window,'horizon':h,'stratum':stratum,
                    'pass_n':len(yes),'nonpass_n':len(no),'pooled_n':len(g),
                    'pass_mean':yes.y_true.mean(),'nonpass_mean':no.y_true.mean(),
                    'pass_high_pct':100*yes.y_true.ge(35).mean() if len(yes) else np.nan,
                    'nonpass_high_pct':100*no.y_true.ge(35).mean() if len(no) else np.nan,
                    'pass_current_mean':yes.seoul_pm25_now.mean(),'nonpass_current_mean':no.seoul_pm25_now.mean(),
                    'common_support':len(yes)>=minimum and len(no)>=minimum})
            frame=pd.DataFrame(records);supported=frame[frame.common_support].copy()
            common_n=int(supported.pooled_n.sum())
            record={'region_id':region,'window':window,'horizon':h,'current_valid_n':len(valid),
                    'common_pooled_n':common_n,'common_strata_n':len(supported),
                    'target_coverage_pct':100*common_n/len(valid) if len(valid) else np.nan}
            for field in ['mean','high_pct','current_mean']:
                for side in ['pass','nonpass']:
                    col=f'{side}_{field}'
                    record['adjusted_'+col]=float(np.average(supported[col],weights=supported.pooled_n)) if common_n else np.nan
                record['adjusted_difference_'+field]=record['adjusted_pass_'+field]-record['adjusted_nonpass_'+field]
            frame['reference_weight']=np.where(frame.common_support,frame.pooled_n/common_n if common_n else 0,0)
            strata_rows.extend(frame.to_dict('records'));adjusted_rows.append(record)
    return {'passage_reference':pd.DataFrame(route_rows),'raw_reference':pd.DataFrame(raw_rows),
            'adjusted_reference':pd.DataFrame(adjusted_rows),'strata_reference':pd.DataFrame(strata_rows)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-root',type=Path,required=True)
    ap.add_argument('--passages',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--minimum',type=int,default=10)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    source=source_reference(args.dataset_root)
    path=args.passages
    passage=pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
    results=calculate(source,passage,args.minimum)
    for name,f in results.items():f.to_csv(args.out/(name+'.csv'),index=False,encoding='utf-8-sig')
    summary={'status':'passed','original_training_origins':5840,'eligible_counts':{str(h):int(source[source.horizon.eq(h)].eligible.sum()) for h in HORIZONS},
        'current_missing_origins':77,'current_bins_left_closed':True,'high_threshold_ge':35,
        'minimum_per_side_per_stratum':args.minimum,'uses_persistence_filled_values_as_current':False,
        'future_targets_crossing_training_boundary_used':False,'no_outcome_model_fitted':True,
        'rows':{k:len(v) for k,v in results.items()}}
    (args.out/'independent_reference_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
