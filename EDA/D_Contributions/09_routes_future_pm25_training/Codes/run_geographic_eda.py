"""Training-only geographic route / future Junggu PM2.5 descriptive EDA.

Region passage is geographic, never proximity to an observation station.
Adjusted contrasts compare passage vs no passage within each region's common
season/current-concentration support; different regions may target different
subpopulations. These are associations, not causal/source contributions.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HORIZONS=(3,6,12,24)
WINDOWS=('01-48','07-48','13-48')
SEASONS=('DJF','MAM','JJA','SON')
CURRENT_LABELS=('0-<15','15-<35','35-<75','75+','unknown')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def bootstrap_plan(n,reps,seed,block=56):
    if n<block:raise ValueError('Calendar too short for a 7-day block.')
    rng=np.random.default_rng(seed)
    full=n//block;tail=n%block
    starts=rng.integers(0,n-block+1,size=(reps,full))
    tails=rng.integers(0,n-tail+1,size=reps) if tail else None
    return starts,tails,block,tail

def bootstrap_sums(series,plan):
    """Moving calendar blocks, without first removing unknown/invalid rows."""
    starts,tails,block,tail=plan
    cumulative=np.vstack([np.zeros((1,series.shape[1])),np.cumsum(series,axis=0,dtype=np.float64)])
    sums=cumulative[block:]-cumulative[:-block]
    out=np.zeros((len(starts),series.shape[1]),float)
    for i in range(0,len(starts),25):
        j=min(i+25,len(starts));out[i:j]=sums[starts[i:j]].sum(axis=1)
        if tail:out[i:j]+=cumulative[tails[i:j]+tail]-cumulative[tails[i:j]]
    return out

def ratio(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=np.asarray(b)>0)

def finite_ci(values,reps,minimum_fraction=.8):
    values=np.asarray(values);finite=values[np.isfinite(values)]
    valid=len(finite)
    if valid<minimum_fraction*reps:return np.nan,np.nan,valid
    lo,hi=np.quantile(finite,[.025,.975]);return float(lo),float(hi),valid

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--eda-root',type=Path,required=True)
    ap.add_argument('--bootstrap-reps',type=int,default=500)
    ap.add_argument('--minimum-stratum',type=int,default=10)
    ap.add_argument('--high-threshold',type=float,default=35.)
    ap.add_argument('--seed',type=int,default=20261107)
    args=ap.parse_args();started=time.monotonic()
    out=args.eda_root/'Results';out.mkdir(parents=True,exist_ok=True)
    cfg={'period':'2021-2022 training only','windows':WINDOWS,'main_association_window':'07-48',
         'current_bins':'[0,15),[15,35),[35,75),[75,infinity); unknown separate',
         'high_threshold':args.high_threshold,'high_definition':'hourly research category, not official alert',
         'minimum_pass_and_nonpass_per_common_stratum':args.minimum_stratum,'bootstrap_reps':args.bootstrap_reps,
         'bootstrap_calendar_block_hours':168,'bootstrap_calendar_block_slots':56,'bootstrap_seed':args.seed,
         'bootstrap_fixed_original_support':True,'minimum_valid_bootstrap_fraction_for_CI':.8,
         'common_support_target':'pooled eligible current-observed training stratum frequencies restricted and renormalized per region',
         'different_regions_may_have_different_standardization_population':True,'no_formal_region_ranking_or_causal_test':True}
    (out/'analysis_configuration.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    origins=pd.date_range('2021-01-01','2023-01-01',freq='3h',inclusive='left',tz='Asia/Seoul').as_unit('ns')
    n=len(origins);assert n==5840
    visits=pd.read_parquet(out/'origin_visits.parquet')
    assert visits.origin_time_kst.dt.tz is not None
    assert not visits.duplicated(['origin_time_kst','region_id','window']).any()
    assert set(visits.window)==set(WINDOWS)
    definitions=visits[['region_id','label_en','label_ko']].drop_duplicates().set_index('region_id')
    regions=definitions.index.tolist();nr=len(regions)
    assert len(visits)==n*nr*len(WINDOWS)
    grid_path=args.project_root/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
    grid=pq.read_table(grid_path,filters=[('split','=','train')]).to_pandas()
    grid['origin_time_kst']=pd.DatetimeIndex(grid.origin_time_kst).as_unit('ns')
    assert len(grid)==n*4 and not grid.duplicated(['origin_time_kst','horizon']).any()
    current_path=args.project_root/'Dataset/Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet'
    current=pq.read_table(current_path,columns=['time_kst','station_id','pm25','pm25_flag'],
                          filters=[('station_id','=','111121'),('time_kst','<',pd.Timestamp('2023-01-01',tz='Asia/Seoul').to_pydatetime())]).to_pandas()
    assert not current.time_kst.duplicated().any()
    current=current.set_index(pd.DatetimeIndex(current.time_kst).as_unit('ns')).reindex(origins)
    now=current.pm25.where(current.pm25_flag.eq(0)).to_numpy(float)
    current_bin=pd.cut(now,[0,15,35,75,np.inf],labels=False,right=False)
    cb=np.where(np.isfinite(current_bin),current_bin,4).astype(int)
    months=origins.month
    season=np.select([np.isin(months,[12,1,2]),np.isin(months,[3,4,5]),np.isin(months,[6,7,8])],['DJF','MAM','JJA'],'SON')
    si=pd.Index(SEASONS).get_indexer(season)
    stratum=si*4+np.minimum(cb,3)
    stratum[cb==4]=16
    analysis=pd.DataFrame({'origin_time_kst':origins,'current_pm25':now,'season':season,
                           'current_bin':[CURRENT_LABELS[x] for x in cb],'stratum_code':stratum})
    plan=bootstrap_plan(n,args.bootstrap_reps,args.seed)
    assert np.allclose(bootstrap_sums(np.ones((n,1)),plan),n)
    raw_rows=[];stratum_rows=[];adjusted_rows=[];passage_rows=[];global_rows=[];season_rows=[]
    primary_support=[]
    for window in WINDOWS:
        v=visits.loc[visits.window.eq(window)]
        def matrix(column):
            t=v.pivot(index='origin_time_kst',columns='region_id',values=column).reindex(index=origins,columns=regions)
            assert t.notna().all().all()
            return t.to_numpy()
        passed=matrix('visited_segments').astype(bool);point=matrix('visited_points').astype(bool)
        for j,reg in enumerate(regions):
            analysis[f'visit_{window}_{reg}']=passed[:,j]
            p=passed[:,j];episodes=int(np.count_nonzero(p&~np.r_[False,p[:-1]]))
            for sea in ['ALL',*SEASONS]:
                mask=np.ones(n,bool) if sea=='ALL' else season==sea
                passage_rows.append({'window':window,'region_id':reg,**definitions.loc[reg].to_dict(),'season':sea,
                                     'all_arrivals':int(mask.sum()),'segment_pass_arrivals':int((mask&p).sum()),
                                     'segment_pass_pct':100*(mask&p).sum()/mask.sum(),
                                     'point_visit_arrivals':int((mask&point[:,j]).sum()),'point_visit_pct':100*(mask&point[:,j]).sum()/mask.sum(),
                                     'continuous_pass_episodes_all_training':episodes})
        for horizon in HORIZONS:
            g=grid.loc[grid.horizon.eq(horizon)].set_index('origin_time_kst').reindex(origins)
            assert g.eligible.notna().all()
            valid=g.eligible.to_numpy(bool)
            y=np.where(valid,g.y_true.to_numpy(float),np.nan)
            assert np.isfinite(y[valid]).all() and g.loc[valid,'target_time_kst'].dt.year.le(2022).all()
            analysis[f'y_h{horizon}']=y;analysis[f'eligible_h{horizon}']=valid
            high=np.isfinite(y)&(y>=args.high_threshold)
            ns=17
            # Axis order is region,stratum,pass(1)/nonpass(0). Missing current is its own stratum16.
            code=stratum[None,:]*2+passed.T.astype(int)
            indicators=(code[:,:,None]==np.arange(ns*2)[None,None,:])&valid[None,:,None]
            series=indicators.transpose(1,0,2).reshape(n,nr*ns*2).astype(float)
            counts=series.sum(axis=0).reshape(nr,ns,2)
            sums=(series*np.where(valid,y,0)[:,None]).sum(axis=0).reshape(nr,ns,2)
            highs=(series*high[:,None]).sum(axis=0).reshape(nr,ns,2)
            bc=bootstrap_sums(series,plan).reshape(args.bootstrap_reps,nr,ns,2)
            bs=bootstrap_sums(series*np.where(valid,y,0)[:,None],plan).reshape(args.bootstrap_reps,nr,ns,2)
            bh=bootstrap_sums(series*high[:,None],plan).reshape(args.bootstrap_reps,nr,ns,2)
            population=np.bincount(stratum[valid&(stratum<16)],minlength=16)
            population_weights=population/population.sum()
            supports=[]
            for j,reg in enumerate(regions):
                p=passed[:,j]
                raw={**definitions.loc[reg].to_dict(),'window':window,'region_id':reg,'horizon':horizon,
                     'eligible_arrivals':int(valid.sum()),'passed_target_n':int(counts[j,:,1].sum()),
                     'nonpassed_target_n':int(counts[j,:,0].sum()),
                     'passed_future_mean':float(ratio(sums[j,:,1].sum(),counts[j,:,1].sum())),
                     'nonpassed_future_mean':float(ratio(sums[j,:,0].sum(),counts[j,:,0].sum())),
                     'passed_high_pct':float(100*ratio(highs[j,:,1].sum(),counts[j,:,1].sum())),
                     'nonpassed_high_pct':float(100*ratio(highs[j,:,0].sum(),counts[j,:,0].sum())),
                     'passed_current_observed_n':int((valid&p&np.isfinite(now)).sum()),
                     'passed_current_mean':float(np.nanmean(now[valid&p])) if (valid&p&np.isfinite(now)).any() else np.nan}
                raw['raw_mean_difference']=raw['passed_future_mean']-raw['nonpassed_future_mean']
                raw['raw_high_difference_pp']=raw['passed_high_pct']-raw['nonpassed_high_pct']
                raw['nonpassed_current_mean']=float(np.nanmean(now[valid&~p])) if (valid&~p&np.isfinite(now)).any() else np.nan
                rm=ratio(bs[:,j,:,1].sum(axis=1),bc[:,j,:,1].sum(axis=1))-ratio(bs[:,j,:,0].sum(axis=1),bc[:,j,:,0].sum(axis=1))
                rh=100*(ratio(bh[:,j,:,1].sum(axis=1),bc[:,j,:,1].sum(axis=1))-ratio(bh[:,j,:,0].sum(axis=1),bc[:,j,:,0].sum(axis=1)))
                for name,array in [('mean',rm),('high',rh)]:
                    lo,hi,nb=finite_ci(array,args.bootstrap_reps)
                    raw[f'raw_{name}_diff_ci95_low']=lo;raw[f'raw_{name}_diff_ci95_high']=hi;raw[f'raw_{name}_bootstrap_valid']=nb
                raw_rows.append(raw)
                for sea in SEASONS:
                    sm=valid&(season==sea);sp=sm&p;sn=sm&~p
                    season_rows.append({'window':window,'region_id':reg,**definitions.loc[reg].to_dict(),'horizon':horizon,
                                        'season':sea,'eligible_arrivals':int(sm.sum()),'passed_target_n':int(sp.sum()),
                                        'nonpassed_target_n':int(sn.sum()),
                                        'passed_future_mean':float(y[sp].mean()) if sp.any() else np.nan,
                                        'nonpassed_future_mean':float(y[sn].mean()) if sn.any() else np.nan,
                                        'passed_high_pct':float(100*high[sp].mean()) if sp.any() else np.nan,
                                        'nonpassed_high_pct':float(100*high[sn].mean()) if sn.any() else np.nan,
                                        'passed_current_mean':float(np.nanmean(now[sp])) if (sp&np.isfinite(now)).any() else np.nan})
                for s in range(ns):
                    sm=valid&(stratum==s);sp=sm&p;sn=sm&~p
                    stratum_rows.append({'window':window,'region_id':reg,'horizon':horizon,'stratum_code':s,
                                         'season':SEASONS[s//4] if s<16 else 'all_current_unknown',
                                         'current_bin':CURRENT_LABELS[s%4] if s<16 else 'unknown',
                                         'pass_n':int(counts[j,s,1]),'nonpass_n':int(counts[j,s,0]),
                                         'pass_future_mean':float(ratio(sums[j,s,1],counts[j,s,1])),
                                         'nonpass_future_mean':float(ratio(sums[j,s,0],counts[j,s,0])),
                                         'current_mean_pass':float(now[sp].mean()) if sp.any() and s<16 else np.nan,
                                         'current_mean_nonpass':float(now[sn].mean()) if sn.any() and s<16 else np.nan,
                                         'pass_high_pct':float(100*ratio(highs[j,s,1],counts[j,s,1])),
                                         'nonpass_high_pct':float(100*ratio(highs[j,s,0],counts[j,s,0]))})
                support=(counts[j,:16,0]>=args.minimum_stratum)&(counts[j,:16,1]>=args.minimum_stratum)
                supports.append(support)
                coverage=float(population_weights[support].sum())
                record={'window':window,'region_id':reg,**definitions.loc[reg].to_dict(),'horizon':horizon,
                        'common_strata':','.join(map(str,np.flatnonzero(support))),'common_strata_count':int(support.sum()),
                        'target_population_coverage_pct':100*coverage,'current_missing_eligible_n':int((valid&~np.isfinite(now)).sum()),
                        'pass_n_on_support':int(counts[j,:16,1][support].sum()),'nonpass_n_on_support':int(counts[j,:16,0][support].sum()),
                        'status':'estimable' if support.any() else 'not_estimable_no_common_support'}
                if support.any():
                    w=population_weights[support]/coverage
                    means=ratio(sums[j,:16,:][support],counts[j,:16,:][support]);rates=100*ratio(highs[j,:16,:][support],counts[j,:16,:][support])
                    record.update(adjusted_pass_future_mean=float(w@means[:,1]),adjusted_nonpass_future_mean=float(w@means[:,0]),
                                  adjusted_mean_difference=float(w@(means[:,1]-means[:,0])),
                                  adjusted_pass_high_pct=float(w@rates[:,1]),adjusted_nonpass_high_pct=float(w@rates[:,0]),
                                  adjusted_high_difference_pp=float(w@(rates[:,1]-rates[:,0])))
                    cm=bc[:,j,:16,:][:,support,:];sm=bs[:,j,:16,:][:,support,:];hm=bh[:,j,:16,:][:,support,:]
                    available=(cm>0).all(axis=(1,2))
                    dm=(ratio(sm,cm)[:,:,1]-ratio(sm,cm)[:,:,0])@w
                    dh=100*(ratio(hm,cm)[:,:,1]-ratio(hm,cm)[:,:,0])@w
                    dm[~available]=np.nan;dh[~available]=np.nan
                    for name,array in [('mean',dm),('high',dh)]:
                        lo,hi,nb=finite_ci(array,args.bootstrap_reps)
                        record[f'adjusted_{name}_diff_ci95_low']=lo;record[f'adjusted_{name}_diff_ci95_high']=hi
                        record[f'adjusted_{name}_bootstrap_valid']=nb
                    record['CI_status']='stable_enough' if available.sum()>=.8*args.bootstrap_reps else 'insufficient_valid_bootstrap_replicates'
                    record['support_warning']='limited_population_under50pct' if coverage<.5 else ''
                adjusted_rows.append(record)
            # A shared stratum set across all actual land regions is explicitly audited, not assumed.
            land=[i for i,r in enumerate(regions) if not r.startswith('SEA')]
            common=np.logical_and.reduce(np.asarray(supports)[land])
            global_rows.append({'window':window,'horizon':horizon,'regions_in_shared_support':len(land),
                                'shared_strata_count':int(common.sum()),'shared_target_coverage_pct':100*float(population_weights[common].sum()),
                                'cross_region_adjusted_rank_valid':bool(common.any()),
                                'note':'Pairwise adjusted tables are not directly comparable across different supports.'})
            print(f'{window} h{horizon}: {valid.sum()} eligible; {time.monotonic()-started:.1f}s',flush=True)
    for name,rows in [('passage_rates',passage_rows),('raw_region_future_outcomes',raw_rows),
                      ('season_current_stratified_outcomes',stratum_rows),('adjusted_region_contrasts',adjusted_rows),
                      ('seasonal_region_future_outcomes',season_rows),('global_common_support',global_rows)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False,encoding='utf-8-sig')
    analysis.to_parquet(out/'origin_analysis_table.parquet',index=False,compression='zstd')
    cfg.update({'status':'passed','origins':n,'regions':nr,'current_missing_origins':int((~np.isfinite(now)).sum()),
                'training_target_eligible_counts':{str(h):int(analysis[f'eligible_h{h}'].sum()) for h in HORIZONS},
                'input_hashes':{str(p):sha(p) for p in [grid_path,current_path,out/'origin_visits.parquet']},
                'no_2023_2024_targets_used':True,'no_target_imputation':True,'no_model_training':True,
                'elapsed_seconds':time.monotonic()-started})
    (out/'analysis_manifest.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:cfg[k] for k in ['status','origins','regions','current_missing_origins','training_target_eligible_counts','elapsed_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
