"""Post-hoc descriptive seasonal/route stability; no model fitting or p-values.

Only original seed11 score rows are read. Authoritative eligibility, targets and
conditions are checked against the independently rebuilt frozen H48 grid. An
exact-origin persistence observation identifies current Junggu PM2.5; any
fallback is left missing for current-concentration summaries.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent
OLD=WORK.parent/'dcrnn_h48_results_review_20261112'
SOURCE=OLD/'extracted/20261112_003040_429483_85af4013/Common_Scoring/Combined'
GRID=OLD/'Verification/rebuilt_evaluation_grid.parquet'
TABLES=WORK/'Tables'
TABLES.mkdir(parents=True,exist_ok=True)
HORIZONS=(12,24,36,48)
PAIRS=(('M3','M1','primary'),('M2','M1','supporting_only'),('M3','M2','supporting_only'))
KEY=['origin_time_kst','horizon']
audit={'purpose':'Post-hoc descriptive stability, not confirmatory hypothesis testing or source attribution. No p-values computed here.',
 'main_join_keys':['split','condition_type','condition_value','horizon','candidate','reference'],
 'source_grid':str(GRID),'source_grid_sha256':hashlib.sha256(GRID.read_bytes()).hexdigest(),
 'sources':{},'checks':{},'current_pm_rule':'persistence_y_hat only when persistence_age_hours==0 and persistence_fallback_kind==exact_observation; all fallback observations remain NaN here.',
 'n_definition':'Fixed eligible 3-hour arrival rows, not independent events. n_days and contiguous_runs show temporal concentration; runs break at absent/nonselected 3-hour slots.',
 'season_definition':'Original season at origin_time_kst (DJF/MAM/JJA/SON), including Dec/Jan/Feb within each calendar year; not target-time season.',
 'route_definition':'Original route_group derived from current arrival trajectory history, not future target-time trajectory.',
 'comparisons':PAIRS,'uncertainty':'Parent computes time-block CI and multiplicity-adjusted inference separately; these tables contain descriptive point estimates only.'}

grid=pd.read_parquet(GRID)
grid=grid.loc[grid.horizon.isin(HORIZONS)].copy()
main=[];joint=[]

def exact_rows(frame,split,condition):
    seed=pd.to_numeric(frame.seed,errors='coerce')
    out=frame.loc[frame.input_condition.eq(condition)&frame.prediction_kind.eq('seed')&seed.eq(11)&frame.horizon.isin(HORIZONS)].copy()
    if len(out.model.unique())!=1 or out.model.iloc[0]!='DCRNN_trajectory_directed_H48':raise ValueError('Unexpected model')
    if not out.eligible.all() or not np.isfinite(out.y_hat).all():raise ValueError('Ineligible or nonfinite score row')
    out=out.set_index(KEY).sort_index()
    expected=grid.loc[grid.split.eq(split)&grid.eligible].set_index(KEY).sort_index()
    if not out.index.is_unique or not out.index.equals(expected.index):raise ValueError((split,condition,'fixed grid mismatch'))
    for c in ['y_true','target_time_kst','season','route_group','persistence_y_hat','persistence_fallback_kind','persistence_age_hours']:
        pd.testing.assert_series_equal(out[c],expected[c],check_names=False,check_dtype=False)
    audit['checks'][split+'_'+condition]={'fixed_rows_exact':True,'targets_exact':True,'conditions_exact':True,'n':len(out)}
    return out

def stats(sub,condition_type,condition_value,split,candidate,reference,role):
    e1=sub.yhat_candidate.to_numpy(float)-sub.y_true.to_numpy(float)
    e0=sub.yhat_reference.to_numpy(float)-sub.y_true.to_numpy(float)
    rmse1=float(np.sqrt(np.mean(e1**2)));rmse0=float(np.sqrt(np.mean(e0**2)))
    current=sub.current_pm25.dropna()
    times=pd.DatetimeIndex(sub.origin_time_kst).sort_values()
    gaps=times.to_series().diff().ne(pd.Timedelta(hours=3))
    return {'split':split,'year':2023 if split=='validation' else 2024,'condition_type':condition_type,'condition_value':condition_value,
      'horizon':int(sub.horizon.iloc[0]),'candidate':candidate,'reference':reference,'comparison_role':role,
      'n':len(sub),'n_days':int(times.normalize().nunique()),'contiguous_runs':int(gaps.sum()),
      'candidate_RMSE':rmse1,'reference_RMSE':rmse0,'RMSE_diff':rmse1-rmse0,'RMSE_improvement_pct':100*(1-rmse1/rmse0),
      'candidate_MAE':float(np.mean(np.abs(e1))),'reference_MAE':float(np.mean(np.abs(e0))),'MAE_diff':float(np.mean(np.abs(e1))-np.mean(np.abs(e0))),
      'candidate_MBE':float(e1.mean()),'reference_MBE':float(e0.mean()),
      'future_pm25_mean':float(sub.y_true.mean()),'future_pm25_std':float(sub.y_true.std(ddof=1)),
      'current_pm25_n':len(current),'current_pm25_missing_n':int(sub.current_pm25.isna().sum()),
      'current_pm25_mean':float(current.mean()),'current_pm25_median':float(current.median()),
      'current_pm25_q25':float(current.quantile(.25)),'current_pm25_q75':float(current.quantile(.75)),
      **{f'fraction_{s}':float(sub.season.eq(s).mean()) for s in ['DJF','MAM','JJA','SON']}}

for split in ['validation','test']:
    paths=list((SOURCE/split).rglob('score_rows.parquet'))
    assert len(paths)==1
    path=paths[0]
    audit['sources'][split]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    raw=pd.read_parquet(path)
    models={condition:exact_rows(raw,split,condition) for condition in ['M1','M2','M3']}
    for candidate,reference,role in PAIRS:
        c=models[candidate];r=models[reference]
        if not c.index.equals(r.index):raise ValueError('Different candidate/reference rows')
        for col in ['y_true','season','route_group','target_time_kst']:
            pd.testing.assert_series_equal(c[col],r[col])
        frame=c[['y_true','season','route_group','persistence_y_hat','persistence_fallback_kind','persistence_age_hours']].copy()
        frame['yhat_candidate']=c.y_hat;frame['yhat_reference']=r.y_hat
        exact=frame.persistence_fallback_kind.eq('exact_observation')&frame.persistence_age_hours.eq(0)
        frame['current_pm25']=frame.persistence_y_hat.where(exact)
        frame=frame.reset_index()
        for h in HORIZONS:
            hrows=frame.loc[frame.horizon.eq(h)]
            for kind in ['season','route_group']:
                for value,sub in hrows.groupby(kind,dropna=False,observed=True):
                    main.append(stats(sub,kind,str(value),split,candidate,reference,role))
            for (route,season),sub in hrows.groupby(['route_group','season'],dropna=False,observed=True):
                record=stats(sub,'route_group|season',str(route)+'|'+str(season),split,candidate,reference,role)
                record.update(route_group=str(route),season=str(season))
                joint.append(record)

main=pd.DataFrame(main);joint=pd.DataFrame(joint)
main.to_csv(TABLES/'yearly_descriptive.csv',index=False,encoding='utf-8-sig')
joint.to_csv(TABLES/'yearly_route_season_descriptive.csv',index=False,encoding='utf-8-sig')

idcols=['condition_type','condition_value','horizon','candidate','reference','comparison_role']
fields=['RMSE_diff','RMSE_improvement_pct','MAE_diff','n','current_pm25_mean','future_pm25_mean','contiguous_runs']
wide=main.pivot(index=idcols,columns='year',values=fields)
wide.columns=[f'{metric}_{year}' for metric,year in wide.columns]
wide=wide.reset_index()
wide['both_years_RMSE_lower']=(wide.RMSE_diff_2023<0)&(wide.RMSE_diff_2024<0)
wide['both_years_RMSE_higher']=(wide.RMSE_diff_2023>0)&(wide.RMSE_diff_2024>0)
wide['RMSE_direction_same']=(np.sign(wide.RMSE_diff_2023)==np.sign(wide.RMSE_diff_2024))
wide['both_years_MAE_lower']=(wide.MAE_diff_2023<0)&(wide.MAE_diff_2024<0)
wide.to_csv(TABLES/'yearly_direction_stability.csv',index=False,encoding='utf-8-sig')

# Descriptive stratification diagnostic: among the four seasonal cells, how
# many improve? This does not standardize causal effects or prove significance.
support=[]
for keys,part in joint.groupby(['split','route_group','horizon','candidate','reference']):
    split,route,h,candidate,reference=keys
    base=main.loc[(main.split==split)&(main.condition_type=='route_group')&(main.condition_value==route)&(main.horizon==h)&(main.candidate==candidate)&(main.reference==reference)].iloc[0]
    support.append({'split':split,'route_group':route,'horizon':h,'candidate':candidate,'reference':reference,
      'pooled_RMSE_diff':float(base.RMSE_diff),'seasons_observed':len(part),'seasons_RMSE_lower':int(part.RMSE_diff.lt(0).sum()),
      'min_season_n':int(part.n.min()),'max_season_n':int(part.n.max()),
      'season_RMSE_diffs':json.dumps(dict(zip(part.season,part.RMSE_diff)),sort_keys=True),
      'season_counts':json.dumps(dict(zip(part.season,part.n)),sort_keys=True),
      'season_current_pm25_means':json.dumps(dict(zip(part.season,part.current_pm25_mean)),sort_keys=True)})
pd.DataFrame(support).to_csv(TABLES/'route_season_mixing_diagnostic.csv',index=False,encoding='utf-8-sig')
audit['row_counts']={'main':len(main),'route_season':len(joint),'stability':len(wide)}
audit['main_M3_M1_both_years_lower']=wide.loc[wide.candidate.eq('M3')&wide.reference.eq('M1')&wide.both_years_RMSE_lower,idcols+['RMSE_improvement_pct_2023','RMSE_improvement_pct_2024','n_2023','n_2024']].to_dict('records')
audit['confounding_warning']='Season and route are correlated; route-group aggregate differences mix season, current PM concentration, future PM variability and weather. See per-row season fractions, current PM summaries and route×season cells. Current PM is summary only, not an adjusted causal estimate. Serially adjacent rows are not independent events.'
(WORK/'descriptive_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
print('M3 versus M1, both years RMSE lower (descriptive only):')
print(wide.loc[wide.candidate.eq('M3')&wide.reference.eq('M1')&wide.both_years_RMSE_lower,idcols+['RMSE_improvement_pct_2023','RMSE_improvement_pct_2024','n_2023','n_2024']].round(5).to_string(index=False))
print('\nM3 versus M1 route-group details:')
print(main.loc[main.candidate.eq('M3')&main.reference.eq('M1')&main.condition_type.eq('route_group'),['year','condition_value','horizon','n','contiguous_runs','RMSE_improvement_pct','MAE_diff','current_pm25_mean','future_pm25_mean','fraction_DJF','fraction_MAM','fraction_JJA','fraction_SON']].round(5).to_string(index=False))
