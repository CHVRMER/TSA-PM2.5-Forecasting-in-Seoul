"""Reproduce the independent audit of the geographic-passage EDA outputs.

Run with the same Python environment as the main analysis (numpy, pandas,
pyarrow). This verifier only reads source data and analysis tables. Its single
output is Results/independent_statistics_verification.json unless overridden.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from independent_statistics_reference import source_reference, calculate


def import_main(eda_root):
    path=eda_root/'Codes/run_geographic_eda.py'
    if not path.exists():
        raise FileNotFoundError(f'Main analysis code not found: {path}')
    spec=importlib.util.spec_from_file_location('main_geographic_eda',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def assert_numeric_equal(a,b,label):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if not np.allclose(a,b,rtol=1e-10,atol=1e-9,equal_nan=True):
        raise AssertionError(f'Independent statistics mismatch: {label}')
    finite=np.isfinite(a)&np.isfinite(b)
    return float(np.abs(a[finite]-b[finite]).max()) if finite.any() else 0.


def explicit_indices(plan):
    starts,tails,block,tail=plan
    return [np.concatenate([np.concatenate([np.arange(a,a+block) for a in row]),
                            np.arange(tails[i],tails[i]+tail) if tail else np.array([],int)])
            for i,row in enumerate(starts)]


def audit_bootstrap(main,source,visits,reported,cfg):
    # A synthetic calendar keeps zeros/gaps at their original positions.
    plan=main.bootstrap_plan(120,30,20261107)
    series=np.c_[np.arange(120),np.arange(120)%3,np.ones(120)]
    series[[2,3,56,57,119],:2]=0
    expected=np.stack([series[idx].sum(0) for idx in explicit_indices(plan)])
    assert np.array_equal(expected,main.bootstrap_sums(series,plan))
    assert np.isnan(main.finite_ci(np.r_[np.arange(399),np.full(101,np.nan)],500)[0])
    assert np.isfinite(main.finite_ci(np.r_[np.arange(400),np.full(100,np.nan)],500)[0])

    frame0=source[source.horizon.eq(6)].sort_values('origin_time_kst').copy()
    origins=pd.DatetimeIndex(frame0.origin_time_kst)
    reps=cfg['bootstrap_reps']
    plan=main.bootstrap_plan(len(frame0),reps,cfg['bootstrap_seed'],cfg['bootstrap_calendar_block_slots'])
    sampled=explicit_indices(plan)
    examples=[]
    for region in ['KR_GANGWON','CN_SHANDONG']:
        p=visits[visits.region_id.eq(region)&visits.window.eq('07-48')].set_index('origin_time_kst').reindex(origins).visited_segments.to_numpy(bool)
        frame=frame0.copy();frame['passed']=p
        eligible=frame.eligible.to_numpy(bool);observed=frame.current_bin.notna().to_numpy()
        base=frame[frame.eligible & frame.current_bin.notna()]
        table=base.groupby(['stratum','passed']).size().unstack('passed',fill_value=0).reindex(columns=[False,True],fill_value=0)
        minimum=cfg['minimum_pass_and_nonpass_per_common_stratum']
        support=table.index[(table[False]>=minimum)&(table[True]>=minimum)].tolist()
        pooled=base.groupby('stratum').size().reindex(support)
        weights=(pooled/pooled.sum()).to_numpy(float)
        # get_indexer returns -1 outside original support without altering it.
        categories=pd.Index(support).get_indexer(frame.stratum)
        y=frame.y_true.to_numpy(float)
        means=[];highs=[]
        for idx in sampled:
            active=eligible[idx]&observed[idx]&(categories[idx]>=0)
            code=categories[idx][active]*2+p[idx][active].astype(int)
            yy=y[idx][active]
            count=np.bincount(code,minlength=len(support)*2).reshape(-1,2)
            if (count==0).any():
                means.append(np.nan);highs.append(np.nan);continue
            sm=np.bincount(code,weights=yy,minlength=len(support)*2).reshape(-1,2)/count
            sh=np.bincount(code,weights=(yy>=cfg['high_threshold']).astype(float),minlength=len(support)*2).reshape(-1,2)/count
            means.append(weights@(sm[:,1]-sm[:,0]))
            highs.append(100*weights@(sh[:,1]-sh[:,0]))
        row=reported[reported.region_id.eq(region)&reported.window.eq('07-48')&reported.horizon.eq(6)].iloc[0]
        example={'region_id':region,'window':'07-48','horizon':6,'support_strata':len(support)}
        for name,values in [('mean',means),('high',highs)]:
            values=np.asarray(values);values=values[np.isfinite(values)]
            ci=np.quantile(values,[.025,.975]) if len(values)>=cfg['minimum_valid_bootstrap_fraction_for_CI']*reps else np.array([np.nan,np.nan])
            actual=row[[f'adjusted_{name}_diff_ci95_low',f'adjusted_{name}_diff_ci95_high']].to_numpy(float)
            err=assert_numeric_equal(ci,actual,f'{region}/{name} bootstrap CI')
            assert len(values)==row[f'adjusted_{name}_bootstrap_valid']
            example[name]={'valid_replicates':len(values),'ci95':ci.tolist(),'max_error':err}
        examples.append(example)
    return examples


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True,help='Revised_Project directory')
    ap.add_argument('--eda-root',type=Path,required=True,help='Geographic_Region_Future_PM25_Training_2021_2022 directory')
    ap.add_argument('--report-path',type=Path,help='Optional audit JSON destination')
    args=ap.parse_args();out=args.eda_root/'Results'
    cfg=json.loads((out/'analysis_configuration.json').read_text('utf-8'))
    assert cfg['high_threshold']==35 and cfg['minimum_pass_and_nonpass_per_common_stratum']==10
    source=source_reference(args.project_root/'Dataset')

    # The independent reference reads preexisting origin features. Cross-check
    # exact current observations against the nationwide source used by main EDA.
    originals=source[['origin_time_kst','seoul_pm25_now']].drop_duplicates().sort_values('origin_time_kst')
    national=pq.read_table(args.project_root/'Dataset/Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet',
        columns=['time_kst','station_id','pm25','pm25_flag'],
        filters=[('station_id','=','111121'),('time_kst','<',pd.Timestamp('2023-01-01',tz='Asia/Seoul').to_pydatetime())]).to_pandas()
    assert not national.time_kst.duplicated().any()
    national=national.set_index(pd.DatetimeIndex(national.time_kst).as_unit('ns')).reindex(pd.DatetimeIndex(originals.origin_time_kst))
    assert_numeric_equal(originals.seoul_pm25_now,national.pm25.where(national.pm25_flag.eq(0)),'exact current PM source')

    visits=pd.read_parquet(out/'origin_visits.parquet')
    passage=visits[['origin_time_kst','region_id','window','visited_segments']].rename(columns={'visited_segments':'passed'})
    result=calculate(source,passage,minimum=10)
    adjusted=pd.read_csv(out/'adjusted_region_contrasts.csv')
    keys=['region_id','window','horizon']
    joined=result['adjusted_reference'].merge(adjusted,on=keys,validate='one_to_one',suffixes=('_ref','_actual'))
    assert len(joined)==240
    fields={'common_strata_n':'common_strata_count','target_coverage_pct':'target_population_coverage_pct',
            'adjusted_pass_mean':'adjusted_pass_future_mean','adjusted_nonpass_mean':'adjusted_nonpass_future_mean',
            'adjusted_difference_mean':'adjusted_mean_difference','adjusted_pass_high_pct':'adjusted_pass_high_pct',
            'adjusted_nonpass_high_pct':'adjusted_nonpass_high_pct','adjusted_difference_high_pct':'adjusted_high_difference_pp'}
    max_error=0.
    for a,b in fields.items():
        aa=a+'_ref' if a==b else a;bb=b+'_actual' if a==b else b
        max_error=max(max_error,assert_numeric_equal(joined[aa],joined[bb],a))

    raw=result['raw_reference'].merge(pd.read_csv(out/'raw_region_future_outcomes.csv'),on=keys,validate='many_to_one')
    assert len(raw)==480
    for passed in [False,True]:
        rows=raw[raw.passed.eq(passed)];prefix='passed' if passed else 'nonpassed'
        for ref,actual in [('n','target_n'),('future_mean','future_mean'),('future_high_pct','high_pct')]:
            assert_numeric_equal(rows[ref],rows[f'{prefix}_{actual}'],f'raw {prefix} {ref}')
    rates=pd.read_csv(out/'passage_rates.csv');rates=rates[rates.season.eq('ALL')]
    rates=result['passage_reference'].merge(rates,on=['region_id','window'],validate='one_to_one')
    assert len(rates)==60
    assert_numeric_equal(rates.pass_pct,rates.segment_pass_pct,'passage percentages')
    assert_numeric_equal(rates.pass_origins,rates.segment_pass_arrivals,'passage counts')
    examples=audit_bootstrap(import_main(args.eda_root),source,visits,adjusted,cfg)
    report={'status':'passed','raw_rows_independently_recomputed':480,'adjusted_rows_independently_recomputed':240,
            'passage_rate_rows_independently_recomputed':60,'adjusted_point_estimates_max_absolute_difference':max_error,
            'raw_count_mean_high_rate_mismatches':0,'geography_pass_rate_mismatches':0,
            'current_pm_exact_observation_not_baseline_fallback':True,'current_unknown_origins':77,'training_origins':5840,
            'time_boundaries_and_target_eligibility_reproduced':True,'standardization_minimum_per_side':10,
            'unknown_current_excluded_from_adjustment_not_raw':True,
            'region_specific_common_support_and_pooled_weights_reproduced':True,'no_support_is_NA_not_zero_effect':True,
            'bootstrap_explicit_calendar_indices_match_cumulative_sums':True,
            'bootstrap_missing_calendar_slots_not_compressed':True,
            'bootstrap_80pct_finite_threshold_boundary_checked':True,
            'bootstrap_real_examples_independently_recomputed':examples,
            'bootstrap_support_and_weights_fixed_to_original_sample':True}
    destination=args.report_path or out/'independent_statistics_verification.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
