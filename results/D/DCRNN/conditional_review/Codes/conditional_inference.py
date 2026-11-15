"""Exploratory paired subgroup audit; no fitting, source changes, or new forecasts.

Protocol is saved before numerical inference. Keep the complete annual calendar:
excluded conditions and missing targets remain missing, never time-compressed.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
PREVIOUS = WORK.parent/'dcrnn_h48_results_review_20261112'
BATCH = PREVIOUS/'extracted/20261112_003040_429483_85af4013'
PROJECT = Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
SCORER = PROJECT/'Model/Common/DCRNN_H48_20261111'
sys.path.insert(0, str(SCORER))
import score

H = [12, 24, 36, 48]
LEVELS = {'season':['DJF','MAM','JJA','SON'], 'route_group':['west','north','east','local']}
SEED = 8211
REPS = 5000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    return {'analysis_status':'post_hoc_exploratory_not_confirmatory',
      'candidate':'M3', 'reference':'M1', 'seed':11, 'horizons':H,
      'subgroup_levels':LEVELS, 'subgroups_defined_by':'forecast origin, not target time',
      'separate_axes_not_cross_product':True, 'splits':['validation','test'],
      'family_size':64, 'holm_family':'both years x four horizons x (four seasons + four existing route groups)',
      'bootstrap':'paired moving calendar blocks with subgroup/target missingness retained',
      'block_days_primary':7, 'block_days_sensitivity':[3,14], 'bootstrap_reps_each':REPS,
      'bootstrap_seed':SEED, 'bootstrap_interval':'pointwise percentile 95%, not simultaneous family intervals',
      'metrics':['M3-RMSE minus M1-RMSE', '100*(1-M3-RMSE/M1-RMSE)'],
      'dm_primary':'squared-error loss; Bartlett exact-calendar HAC lag max(h/3-1,0)',
      'dm_sensitivity':'same loss; Bartlett HAC 56 slots (=7 days) at every horizon',
      'improvement_flag':'7day delta CI high<0 AND Holm64 primary p<.05; exploratory only',
      'robust_improvement_flag':'primary improvement AND 3day/14day CI high<0 AND Holm64 HAC56 p<.05',
      'small_sample_annotation':'n<100 OR active nonoverlapping 7day bins<10 OR bootstrap valid<80%; not a power calculation',
      'independent_blocks_not_claimed':True, 'new_training_or_model_selection':False,
      'original_common_scorer_unchanged':True}


def block_draws(a, b, block_slots, reps=REPS, seed=SEED):
    """Exact full-calendar block resampling using prefix sums, including last tail."""
    a, b = np.asarray(a,float), np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=1 or not len(a):
        raise ValueError('Equal nonempty 1D errors required.')
    good = np.isfinite(a)&np.isfinite(b)
    if not np.array_equal(np.isfinite(a),np.isfinite(b)):
        raise ValueError('Paired errors have different masks.')
    n=len(a);length=min(int(block_slots),n);blocks=math.ceil(n/length)
    tail=n-(blocks-1)*length
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,n-length+1,size=(reps,blocks))
    def sum_values(values):
        prefix=np.r_[0., np.cumsum(values,dtype=float)]
        if blocks>1:
            s=starts[:,:-1]
            total=(prefix[s+length]-prefix[s]).sum(axis=1)
        else:
            total=np.zeros(reps)
        last=starts[:,-1]
        return total+prefix[last+tail]-prefix[last]
    counts=sum_values(good.astype(float))
    sa=sum_values(np.where(good,a*a,0.))
    sb=sum_values(np.where(good,b*b,0.))
    valid=counts>0
    rmse_a=np.sqrt(sa[valid]/counts[valid]);rmse_b=np.sqrt(sb[valid]/counts[valid])
    delta=rmse_a-rmse_b
    pct=np.divide(100*(rmse_b-rmse_a),rmse_b,out=np.full(len(rmse_b),np.nan),where=rmse_b>0)
    return delta,pct,counts[valid],starts,tail


def verify_fast_bootstrap():
    # Meaningful verification against literal block concatenation with gaps.
    a=np.array([1.,np.nan,3.,2.,np.nan,-1.,-2.,4.,5.,np.nan,3.])
    b=np.array([2.,np.nan,4.,1.,np.nan,-2.,-1.,3.,6.,np.nan,2.])
    checks=[]
    for length in [1,3,5,11,20]:
        dd,pp,nn,starts,tail=block_draws(a,b,length,37,32)
        literal=[];pcts=[];sizes=[]
        for row in starts:
            pos=np.concatenate([np.arange(int(s),int(s)+min(length,len(a))) for s in row])[:len(a)]
            aa,bb=a[pos],b[pos];mask=np.isfinite(aa)&np.isfinite(bb)
            if not mask.any():continue
            ra=np.sqrt(np.mean(aa[mask]**2));rb=np.sqrt(np.mean(bb[mask]**2))
            literal.append(ra-rb);pcts.append(100*(1-ra/rb));sizes.append(mask.sum())
        np.testing.assert_allclose(dd,literal,atol=1e-12)
        np.testing.assert_allclose(pp,pcts,atol=1e-12)
        np.testing.assert_array_equal(nn,sizes)
        checks.append({'block_slots':length,'passed':True,'repetitions':37})
    return checks


def summarize_bootstrap(a,b,days):
    d,p,n,_,_=block_draws(a,b,8*days)
    enough=len(d)>=max(10,int(.8*REPS))
    ci=np.percentile(d,[2.5,97.5]) if enough else [np.nan,np.nan]
    pc=np.percentile(p[np.isfinite(p)],[2.5,97.5]) if enough and np.isfinite(p).sum()>=.8*REPS else [np.nan,np.nan]
    return {f'delta_ci_low_{days}d':float(ci[0]),f'delta_ci_high_{days}d':float(ci[1]),
        f'improvement_ci_low_{days}d':float(pc[0]),f'improvement_ci_high_{days}d':float(pc[1]),
        f'bootstrap_valid_reps_{days}d':len(d),f'bootstrap_n_min_{days}d':int(n.min()) if len(n) else 0,
        f'bootstrap_n_median_{days}d':float(np.median(n)) if len(n) else 0,
        f'bootstrap_n_max_{days}d':int(n.max()) if len(n) else 0}


def main():
    start=time.monotonic();WORK.mkdir(parents=True,exist_ok=True)
    tables=WORK/'Tables';tables.mkdir(exist_ok=True)
    spec=protocol()
    p=WORK/'analysis_protocol.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=spec:
        raise ValueError('Existing protocol differs; write a separate analysis version.')
    p.write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    audit={'protocol_sha256':sha(p),'bootstrap_implementation_tests':verify_fast_bootstrap(),
           'source_hashes':{},'scorer_sha256':sha(SCORER/'score.py'), 'splits':{}}
    grid_path=PREVIOUS/'Verification/rebuilt_evaluation_grid.parquet'
    grid=pd.read_parquet(grid_path);audit['source_hashes'][str(grid_path)]=sha(grid_path)
    records=[]
    for split in ['validation','test']:
        source=next((BATCH/'Common_Scoring/Combined'/split).glob('*/score_rows.parquet'))
        audit['source_hashes'][str(source)]=sha(source)
        raw=pd.read_parquet(source)
        m1=raw.loc[raw.input_condition.eq('M1')&raw.prediction_kind.eq('seed')&raw.seed.astype(str).eq('11')&raw.horizon.isin(H)].copy()
        m3=raw.loc[raw.input_condition.eq('M3')&raw.prediction_kind.eq('seed')&raw.seed.astype(str).eq('11')&raw.horizon.isin(H)].copy()
        keys=['origin_time_kst','horizon']
        m1=m1.set_index(keys).sort_index();m3=m3.set_index(keys).sort_index()
        expected=grid.loc[grid.split.eq(split)&grid.horizon.isin(H)&grid.eligible].set_index(keys).sort_index()
        if not m1.index.is_unique or not m3.index.is_unique or not m1.index.equals(expected.index) or not m3.index.equals(expected.index):
            raise ValueError('Fixed eligible rows differ.')
        for col in ['y_true','target_time_kst','season','route_group']:
            pd.testing.assert_series_equal(m1[col],m3[col],check_names=True,check_exact=True)
            pd.testing.assert_series_equal(m1[col],expected[col],check_names=True,check_exact=True)
        if not np.isfinite(m1.y_hat).all() or not np.isfinite(m3.y_hat).all():raise ValueError('Nonfinite forecasts.')
        audit['splits'][split]={'fixed_eligible_rows_exact':True,'truth_and_target_time_exact':True,'conditions_exact':True,
           'model_count':2,'seed_count':1,'source_rows':len(m1)}
        for h in H:
            ref=m1.xs(h,level='horizon');cand=m3.xs(h,level='horizon')
            full=pd.DatetimeIndex(grid.loc[grid.split.eq(split)&grid.horizon.eq(h),'origin_time_kst']).sort_values()
            for axis,levels in LEVELS.items():
                if set(ref[axis].dropna().unique())-set(levels):raise ValueError('Unexpected condition levels.')
                for level in levels:
                    chosen=ref[axis].eq(level)
                    rr=ref.loc[chosen];cc=cand.loc[chosen]
                    e1=cc.y_hat.to_numpy()-cc.y_true.to_numpy();e2=rr.y_hat.to_numpy()-rr.y_true.to_numpy()
                    a,b=score.calendar_errors(cc.index,e1,e2,full)
                    mask=np.isfinite(a)&np.isfinite(b)
                    positions=np.flatnonzero(mask)
                    n=int(mask.sum())
                    rmse1=float(np.sqrt(np.mean(e1*e1)));rmse2=float(np.sqrt(np.mean(e2*e2)))
                    days=pd.DatetimeIndex(rr.index).normalize()
                    bins=np.arange(len(full))//56
                    active_bins=len(np.unique(bins[mask]))
                    spells=int((mask&~np.r_[False,mask[:-1]]).sum())
                    df={'split':split,'year':2023 if split=='validation' else 2024,'candidate':'M3','reference':'M1',
                        'condition_type':axis,'condition_value':level,'horizon':h,'n':n,
                        'unique_origin_days':days.nunique(),'active_nonoverlapping_7day_bins':active_bins,
                        'observed_contiguous_runs':spells,'calendar_slots':len(full),
                        'M1_RMSE':rmse2,'M3_RMSE':rmse1,'rmse_diff':rmse1-rmse2,
                        'improvement_pct':100*(1-rmse1/rmse2),
                        'M1_MAE':float(np.mean(np.abs(e2))),'M3_MAE':float(np.mean(np.abs(e1))),
                        'M1_MBE':float(np.mean(e2)),'M3_MBE':float(np.mean(e1)),
                        'target_mean':float(rr.y_true.mean()),'target_ge35_n':int(rr.y_true.ge(35).sum()),
                        'small_sample':bool(n<100 or active_bins<10)}
                    dm=score.dm_calendar(a,b,h);sensitive=score.dm_calendar(a,b,h,lag=56)
                    df.update(dm_p=dm['p'],dm_stat=dm['dm_stat'],hac_lag_slots=dm['hac_lag_slots'],
                              dm_p_hac56=sensitive['p'],dm_stat_hac56=sensitive['dm_stat'])
                    for length in [7,3,14]:df.update(summarize_bootstrap(a,b,length))
                    records.append(df)
            print(json.dumps({'split':split,'horizon':h,'completed_cells':len(records),'elapsed_seconds':round(time.monotonic()-start,2)}),flush=True)
    out=pd.DataFrame(records)
    assert len(out)==64
    out['p_holm64']=score.holm(out.dm_p.to_numpy())
    out['p_holm64_hac56']=score.holm(out.dm_p_hac56.to_numpy())
    out['exploratory_improved']=(out.delta_ci_high_7d<0)&(out.p_holm64<.05)
    out['exploratory_worsened']=(out.delta_ci_low_7d>0)&(out.p_holm64<.05)
    out['robust_improved']=out.exploratory_improved&(out.delta_ci_high_3d<0)&(out.delta_ci_high_14d<0)&(out.p_holm64_hac56<.05)
    out['robust_worsened']=out.exploratory_worsened&(out.delta_ci_low_3d>0)&(out.delta_ci_low_14d>0)&(out.p_holm64_hac56<.05)
    out['ci_excludes_zero_benefit_7d']=out.delta_ci_high_7d<0
    out['ci_excludes_zero_harm_7d']=out.delta_ci_low_7d>0
    out.to_csv(tables/'conditional_primary.csv',index=False,encoding='utf-8-sig')
    for split in ['validation','test']:
        for axis in LEVELS:
            out.loc[out.split.eq(split)&out.condition_type.eq(axis)].to_csv(tables/f'{split}_{axis}_uncertainty.csv',index=False,encoding='utf-8-sig')
    keys=['condition_type','condition_value','horizon']
    a=out.loc[out.split.eq('validation')].set_index(keys)
    b=out.loc[out.split.eq('test')].set_index(keys)
    keep=['n','unique_origin_days','active_nonoverlapping_7day_bins','small_sample','improvement_pct','improvement_ci_low_7d','improvement_ci_high_7d','rmse_diff','delta_ci_low_7d','delta_ci_high_7d','p_holm64','robust_improved','robust_worsened']
    stability=a[keep].add_suffix('_2023').join(b[keep].add_suffix('_2024')).reset_index()
    stability['improves_both_years']=(stability.improvement_pct_2023>0)&(stability.improvement_pct_2024>0)
    stability['worsens_both_years']=(stability.improvement_pct_2023<0)&(stability.improvement_pct_2024<0)
    stability.to_csv(tables/'cross_year_stability.csv',index=False,encoding='utf-8-sig')
    audit['family_size']=len(out)
    audit['results_summary']={'exploratory_improved':int(out.exploratory_improved.sum()),'exploratory_worsened':int(out.exploratory_worsened.sum()),
        'robust_improved':int(out.robust_improved.sum()),'robust_worsened':int(out.robust_worsened.sum()),
        'small_sample_cells':int(out.small_sample.sum()),'both_year_point_improvement_cells':int(stability.improves_both_years.sum()),
        'ci7d_benefit_cells':int(out.ci_excludes_zero_benefit_7d.sum()),'ci7d_harm_cells':int(out.ci_excludes_zero_harm_7d.sum())}
    audit['all_bootstrap_repetitions_valid']=bool(out[['bootstrap_valid_reps_'+str(d)+'d' for d in [3,7,14]]].eq(REPS).all().all())
    audit['models_or_sources_modified']=False;audit['elapsed_seconds']=time.monotonic()-start
    (WORK/'conditional_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(audit['results_summary'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
