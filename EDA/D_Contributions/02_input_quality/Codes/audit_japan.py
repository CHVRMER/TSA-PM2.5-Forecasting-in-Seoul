"""Read-only Japan PM2.5 audit; no source rewriting or model fitting."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys
import numpy as np
import pandas as pd

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--project-root',type=Path,default=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project'))
_parser.add_argument('--output-root',type=Path,default=None)
_args=_parser.parse_args()
PROJECT=_args.project_root.resolve()
_scope='japan' if 'japan' in Path(__file__).name else 'korea_weather'
_default_output=Path(__file__).resolve().parents[1]/'Results'/_scope/('recheck_'+datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8])

BASE=PROJECT/'Dataset'
OUT=_args.output_root.resolve() if _args.output_root is not None else _default_output
OUT.mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def native(value):
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,)):return None if not np.isfinite(value) else float(value)
    if isinstance(value,(np.bool_,)):return bool(value)
    if isinstance(value,pd.Timestamp):return value.isoformat()
    raise TypeError(type(value).__name__)

def audit_subset(df,name):
    neg=df.pm25_negative_raw.notna()
    actual=df.pm25.notna()
    signed=df.pm25.copy()
    signed.loc[neg]=df.loc[neg,'pm25_negative_raw']
    signed0=signed.clip(lower=0)
    years=[]
    for year,sub in df.groupby(df.time_kst.dt.year):
        years.append({'scope':name,'year':int(year),'rows':len(sub),'stations':sub.station_id.nunique(),
            'missing_rows':int(sub.pm25.isna().sum()),'missing_pct':100*float(sub.pm25.isna().mean()),
            'negative_raw_count':int(sub.pm25_negative_raw.notna().sum()),
            'negative_raw_pct':100*float(sub.pm25_negative_raw.notna().mean()),
            'pm25_max':float(sub.pm25.max()),'flag_counts':json.dumps({str(k):int(v) for k,v in sub.pm25_flag.value_counts().items()},sort_keys=True)})
    stations=[]
    for station,sub in df.groupby('station_id',sort=False):
        times=sub.time_kst.sort_values()
        stations.append({'scope':name,'station_id':station,'rows':len(sub),'unique_times':sub.time_kst.nunique(),
            'nonhourly_steps':int(times.diff().dropna().ne(pd.Timedelta(hours=1)).sum()),
            'missing_pct':100*float(sub.pm25.isna().mean()),'negative_count':int(sub.pm25_negative_raw.notna().sum()),
            'negative_min':sub.pm25_negative_raw.min(),'pm25_max':sub.pm25.max()})
    return {'scope':name,'rows':len(df),'stations':df.station_id.nunique(),
            'first_kst':str(df.time_kst.min()),'last_kst':str(df.time_kst.max()),
            'duplicate_station_time':int(df.duplicated(['station_id','time_kst']).sum()),
            'missing_count':int((~actual).sum()),'missing_pct':100*float((~actual).mean()),
            'flag_counts':{str(k):int(v) for k,v in df.pm25_flag.value_counts().items()},
            'negative_preserved':int(neg.sum()),'negative_min':df.pm25_negative_raw.min(),
            'negative_ge_minus5':int((neg & df.pm25_negative_raw.ge(-5)).sum()),
            'negative_lt_minus10':int(df.pm25_negative_raw.lt(-10).sum()),
            'negative_lt_minus20':int(df.pm25_negative_raw.lt(-20).sum()),
            'max_current_analysis_value':df.pm25.max(),
            'negative_values_remaining_in_analysis':int(df.pm25.lt(0).sum()),
            'infinite_values':int(np.isinf(df.pm25.to_numpy()).sum()),
            'flag0_but_pm25_null':int((df.pm25_flag.eq(0)&df.pm25.isna()).sum()),
            'nonzero_flag_but_pm25_nonnull':int((df.pm25_flag.ne(0)&df.pm25.notna()).sum()),
            'flag5_negative_raw_missing':int((df.pm25_flag.eq(5)&~neg).sum()),
            'negative_raw_with_non5_flag':int((neg & df.pm25_flag.ne(5)).sum()),
            'mean_original_signed_measured_values':float(signed.mean()),
            'mean_if_negative_to_zero':float(signed0.mean()),
            'mean_current_negative_to_null':float(df.pm25.mean()),
            'negative_to_zero_minus_signed_mean':float(signed0.mean()-signed.mean()),
            'negative_to_null_minus_signed_mean':float(df.pm25.mean()-signed.mean()),
            'note_means':'Descriptive available-record means; not a claim about model forecast bias or certified instrument QC.'},years,stations

def main():
    paths=[BASE/'Japan/japan_west_hourly_2021_2024_kst.parquet',
           BASE/'Neighbors_Expansion_QC_20261108/japan_east18_hourly_2021_2024_kst.parquet',
           BASE/'Origin_Features_2021_2024_v3/japan_balanced_stations.csv',
           BASE/'Trajectory_Matching_2021_2024_v3/station_registry.csv']
    fingerprints={str(p):sha(p) for p in paths}
    west=pd.read_parquet(paths[0]);west.station_id=west.station_id.astype(str).str.zfill(8)
    east=pd.read_parquet(paths[1]);east.station_id=east.station_id.astype(str).str.zfill(8)
    selected=pd.read_csv(paths[2],dtype=str).station_id.str.zfill(8)
    registry=pd.read_csv(paths[3],dtype=str)
    eligible=set(registry.loc[registry.country.eq('JP'),'station_id'].str.zfill(8))
    summaries=[];all_year=[];all_station=[];hist=[];candidates=[];contexts=[]
    for name,df in [('west_all228',west),('west_matching188',west[west.station_id.isin(eligible)]),
                    ('west_model24',west[west.station_id.isin(selected)]),('east_model18',east)]:
        summary,year,station=audit_subset(df,name)
        summaries.append(summary);all_year.extend(year);all_station.extend(station)
        v=df.pm25_negative_raw.dropna()
        for value,count in v.value_counts().sort_index().items():hist.append({'scope':name,'negative_raw':float(value),'count':int(count)})
        print(name,summary['rows'],summary['missing_pct'],summary['negative_preserved'],summary['negative_min'],summary['max_current_analysis_value'],flush=True)
    for source,df in [('west',west),('east',east)]:
        cand=df[df.pm25.ge(300)|df.pm25_negative_raw.lt(-20)].copy()
        cand['source']=source;cand['selected_current_model']=cand.station_id.isin(selected) if source=='west' else True
        candidates.append(cand)
        # Show actual records around each extreme positive event without editing.
        for _,row in df[df.pm25.ge(300)].iterrows():
            ctx=df[df.station_id.eq(row.station_id)&df.time_kst.between(row.time_kst-pd.Timedelta(hours=3),row.time_kst+pd.Timedelta(hours=3))].copy()
            ctx['event_time_kst']=row.time_kst;ctx['source']=source
            contexts.append(ctx)
    pd.DataFrame(all_year).to_csv(OUT/'japan_missing_by_scope_year.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_station).to_csv(OUT/'japan_station_quality.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(hist).to_csv(OUT/'japan_negative_raw_histogram.csv',index=False,encoding='utf-8-sig')
    pd.concat(candidates,ignore_index=True).to_csv(OUT/'japan_extreme_review_candidates.csv',index=False,encoding='utf-8-sig')
    if contexts:pd.concat(contexts,ignore_index=True).to_csv(OUT/'japan_positive_extreme_context.csv',index=False,encoding='utf-8-sig')
    official={
        'negative_measurement_guidance':{'url':'https://www.env.go.jp/content/900403710.pdf','pages':'1 and 3 (PDF page indices 0 and 2)',
            'interpretation':'Japanese MOE guidance generally retains negative hourly measurements. Both forcing negatives to zero and discarding genuine negatives may bias averages upward. Abnormally negative measurements require instrument/distribution review; the blank-test SD example is not a universal -10 rule.'},
        'nies_hourly_format':{'url':'https://tenbou.nies.go.jp/download/TJ_manu.pdf','pages':'4 and 6 (PDF page indices 3 and 5)',
            'interpretation':'9999=missing;9998=unmeasured;9997=error. PM25/PMFL/PMBH are PM2.5 in ug/m3, different instrument method codes. 01h..24h layout is present; exact end-vs-start time semantics should remain documented rather than silently shifted.'},
        'no_source_edits':True,'zero_replacement_not_applied':True,
        'model_note':'Restoring signed inputs is incompatible with plain log1p for <=-1; needs a documented input-transform experiment, not a blind replacement.'}
    result={'status':'completed_read_only','summaries':summaries,'official_sources':official,'input_sha256':fingerprints,
            'input_hashes_unchanged':all(sha(Path(p))==h for p,h in fingerprints.items()),
            'notes':['181286 is the all228 scope; current model uses only western24 plus eastern18.',
                     'Extreme positive or negative values are review candidates, not automatically certified sensor errors.',
                     'No source or analysis datasets changed and no model retrained.']}
    (OUT/'japan_audit_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=native,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,default=native,allow_nan=False),flush=True)

if __name__=='__main__':main()
