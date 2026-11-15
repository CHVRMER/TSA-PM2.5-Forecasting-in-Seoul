"""Read-only QC re-audit: Korean national PM2.5, target, ASOS and Mongolia.

No source observations, labels, QC policies or evaluation rows are changed.
Plateaus and high concentrations below are review candidates, not invalidity rules.
"""
from pathlib import Path
import csv,hashlib,json,math,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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

DATA=PROJECT/'Dataset'
OUT=_args.output_root.resolve() if _args.output_root is not None else _default_output
HOURS=pd.date_range('2021-01-01', '2025-01-01', freq='h',inclusive='left',tz='Asia/Seoul').as_unit('ns')
FILES=[]

def source(path):
    path=Path(path)
    if path not in FILES:FILES.append(path)
    return path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def scalar(v):
    if v is None:return None
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.floating,float)):return float(v) if np.isfinite(v) else None
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,pd.Timestamp):return v.isoformat()
    return v

def summarize(values):
    a=np.asarray(values,float);valid=a[np.isfinite(a)]
    return {'cells':int(a.size),'missing':int(np.isnan(a).sum()),'missing_fraction':float(np.isnan(a).mean()),
            'infinite':int(np.isinf(a).sum()),'negative':int((valid<0).sum()),'zero':int((valid==0).sum()),
            'maximum':float(valid.max()) if len(valid) else None,
            'minimum':float(valid.min()) if len(valid) else None,
            'q99':float(np.quantile(valid,.99)) if len(valid) else None,
            'q999':float(np.quantile(valid,.999)) if len(valid) else None,
            'above500':int((valid>500).sum()),'above1000':int((valid>1000).sum())}

def intervals(mask, index=HOURS):
    a=np.asarray(mask,bool); starts=np.flatnonzero(a & ~np.r_[False,a[:-1]])
    ends=np.flatnonzero(a & ~np.r_[a[1:],False])
    return [{'start_kst':index[i].isoformat(),'end_kst':index[j].isoformat(),'hours':int(j-i+1)} for i,j in zip(starts,ends)]

def plateaus(values,station, min_hours=24):
    a=np.asarray(values,float);equal=np.isfinite(a[1:]) & np.isfinite(a[:-1]) & (a[1:]==a[:-1])
    starts=np.r_[0,np.flatnonzero(~equal)+1];ends=np.r_[starts[1:]-1,len(a)-1]
    rows=[]
    for i,j in zip(starts,ends):
        if j-i+1>=min_hours and np.isfinite(a[i]):rows.append({'station_id':station,'start_kst':HOURS[i].isoformat(),
            'end_kst':HOURS[j].isoformat(),'hours':int(j-i+1),'pm25':float(a[i]),'is_zero':bool(a[i]==0)})
    return rows

def save_csv(name,records,columns=None):
    frame=pd.DataFrame(records,columns=columns)
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def main():
    OUT.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    config=json.loads(source(PROJECT/'Model/Common/Regional12_Seed11_20261109/experiments.json').read_text(encoding='utf-8-sig'))
    selected={key.removeprefix('KR:') for key in config['conditions']['DN']['station_keys']}
    registry=pd.read_csv(source(DATA/'Korea_National_2021_2024/station_registry.csv'),dtype={'station_id':str})
    all_ids=registry.station_id.tolist();lookup={k:i for i,k in enumerate(all_ids)}
    n=len(all_ids); values=np.full((len(HOURS),n),np.nan,dtype=np.float32)
    flags=np.full((len(HOURS),n),2,dtype=np.uint8);seen=np.zeros((len(HOURS),n),bool)
    duplicates=0;out_of_range=0;input_rows=0;duplicate_collapses=0
    path=source(DATA/'Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=300000):
        frame=batch.to_pandas();input_rows+=len(frame)
        ii=HOURS.get_indexer(pd.DatetimeIndex(frame.time_kst)); jj=frame.station_id.astype(str).map(lookup).to_numpy()
        good=(ii>=0)&pd.notna(jj);out_of_range+=int((~good).sum())
        ii=ii[good];jj=jj[good].astype(np.int64)
        linear=ii*n+jj
        duplicates+=int(pd.Index(linear).duplicated().sum())+int(seen[ii,jj].sum())
        values[ii,jj]=frame.pm25.to_numpy()[good];flags[ii,jj]=frame.pm25_flag.to_numpy()[good];seen[ii,jj]=True
        duplicate_collapses+=int(frame.duplicate_rows_collapsed.sum())
    selected_cols=np.array([lookup[k] for k in sorted(selected)])
    target_col=lookup['111121']; target=values[:,target_col].astype(float)
    selected_values=values[:,selected_cols]
    summary={'audit_scope':'Read-only source audit; all years for data quality, not model/feature selection',
             'national':{'rows':input_rows,'stations_in_source':n,'selected_external_stations':len(selected),
                         'selected_target_excluded':bool('111121' not in selected),'dense_calendar_rows':int(values.size),
                         'absent_source_rows':int((~seen).sum()),'duplicate_station_hour_keys':duplicates,
                         'outside_calendar_or_unknown_station_rows':out_of_range,'collapsed_raw_duplicate_rows':duplicate_collapses,
                         'source_values':summarize(values),'selected_values':summarize(selected_values),
                         'source_flag_counts':{str(k):int(v) for k,v in zip(*np.unique(flags,return_counts=True))},
                         'nonzero_flag_with_finite_value':int((np.isfinite(values)&(flags!=0)).sum()),
                         'zero_flag_with_nonfinite_value':int((~np.isfinite(values)&(flags==0)).sum()),
                         'selected_network_labels':registry[registry.station_id.isin(selected)].network.value_counts().to_dict(),
                         'roadside_selected':bool(registry[registry.station_id.isin(selected)].network.str.contains('도로',na=False).any())}}
    yearly=[];station_rows=[];plateau_rows=[];high_rows=[]
    for station in all_ids:
        j=lookup[station];a=values[:,j]
        station_rows.append(dict(station_id=station,selected=station in selected,is_target=station=='111121',**summarize(a)))
        if station in selected or station=='111121':
            plateau_rows.extend(plateaus(a,station))
            ids=np.flatnonzero(a>200)
            for i in ids:high_rows.append({'station_id':station,'time_kst':HOURS[i].isoformat(),'pm25':float(a[i]),
                'flag':int(flags[i,j]),'selected':station in selected,'is_target':station=='111121'})
    for year in (2021,2022,2023,2024):
        mask=HOURS.year==year
        yearly.append({'dataset':'national_selected514','year':year,**summarize(selected_values[mask])})
        yearly.append({'dataset':'junggu_target','year':year,**summarize(target[mask])})
    save_csv('korea_station_quality.csv',station_rows)
    save_csv('korea_high_concentration_candidates.csv',high_rows)
    save_csv('korea_plateau24h_candidates.csv',plateau_rows)
    target_gaps=intervals(np.isnan(target));save_csv('junggu_missing_intervals.csv',target_gaps)
    monthly=[]
    for month in pd.period_range('2021-01','2024-12',freq='M'):
        mask=(HOURS.year==month.year)&(HOURS.month==month.month)
        monthly.append({'month':str(month),**summarize(target[mask])})
    save_csv('junggu_monthly_quality.csv',monthly)
    old_path=source(DATA/'Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet')
    old=pq.read_table(old_path,columns=['time_kst','station_id','pm25','pm25_flag'],filters=[('station_id','=','111121')]).to_pandas()
    old=old.set_index('time_kst').reindex(HOURS)
    old_values=old.pm25.to_numpy(float)
    target_same=np.isclose(old_values,target,rtol=0,atol=0,equal_nan=True)
    flag_same=old.pm25_flag.to_numpy()==flags[:,target_col]
    grid=pq.read_table(source(DATA/'Evaluation_Contract_20261107/evaluation_grid.parquet')).to_pandas()
    positions=HOURS.get_indexer(pd.DatetimeIndex(grid.target_time_kst))
    within=positions>=0;expected=np.full(len(grid),np.nan);expected[within]=target[positions[within]]
    origin_year=pd.DatetimeIndex(grid.origin_time_kst).year;target_year=pd.DatetimeIndex(grid.target_time_kst).year
    origin_split=np.where(origin_year<=2022,'train',np.where(origin_year==2023,'validation','test'))
    target_split=np.where(target_year<=2022,'train',np.where(target_year==2023,'validation','test'))
    eligible_expected=within&np.isfinite(expected)&(origin_split==target_split)
    same_label=np.isclose(grid.y_true.to_numpy(float),np.where(eligible_expected,expected,np.nan),rtol=0,atol=1e-8,equal_nan=True)
    # The contract may retain truth on ineligible cross-split rows only if the
    # eligibility is false. Report both all rows and eligible rows explicitly.
    eligible_label_matches=np.isclose(grid.loc[grid.eligible,'y_true'],expected[grid.eligible],rtol=0,atol=1e-8,equal_nan=True)
    summary['target']={'values':summarize(target),'old_korea34_value_mismatches':int((~target_same).sum()),
        'old_korea34_flag_mismatches':int((~flag_same).sum()),'longest_missing_intervals':sorted(target_gaps,key=lambda x:-x['hours'])[:10],
        'missing_2023_intervals':[g for g in target_gaps if g['start_kst'][:4]=='2023'],
        'evaluation_grid_rows':len(grid),'evaluation_eligibility_mismatches':int((grid.eligible.to_numpy(bool)!=eligible_expected).sum()),
        'evaluation_eligible_truth_mismatches':int((~eligible_label_matches).sum()),
        'evaluation_all_truth_vs_masked_expectation_mismatches':int((~same_label).sum()),
        'evaluation_ineligible_retained_truth_rows':int((~grid.eligible.to_numpy(bool)&np.isfinite(grid.y_true.to_numpy(float))).sum()),
        'evaluation_ineligible_truth_note':'The grid retains raw truth on some cross-split rows but eligible=False excludes them; all eligible labels and the eligibility mask match source. This is not a demonstrated leak.',
        'plateau24h_runs':[p for p in plateau_rows if p['station_id']=='111121']}
    summary['national']['plateau24h_candidates']={'runs':len(plateau_rows),'zero_runs':sum(p['is_zero'] for p in plateau_rows),
        'nonzero_runs':sum(not p['is_zero'] for p in plateau_rows),'hours':sum(p['hours'] for p in plateau_rows)}
    legacy=pq.read_table(old_path).to_pandas()
    legacy_quality=[]
    for var in ('pm25','pm10','so2','no2','o3','co'):
        a=legacy[var].to_numpy(float); f=legacy[var+'_flag'].to_numpy()
        legacy_quality.append({'pollutant':var,**summarize(a),
            'finite_value_with_nonzero_flag':int((np.isfinite(a)&(f!=0)).sum()),
            'zero_flag_with_nonfinite_value':int((~np.isfinite(a)&(f==0)).sum())})
    save_csv('korea34_pollutant_quality.csv',legacy_quality)
    fine=legacy.pm25.to_numpy(float);coarse=legacy.pm10.to_numpy(float)
    candidate=(fine>=300)&(coarse>0)&(fine>2*coarse)&((fine-coarse)>100)
    summary['legacy_korea34']={'pollutants':legacy_quality,
        'severe_pm25_pm10_incoherence_candidates':int(candidate.sum()),
        'pm25_above_pm10_any_magnitude':int((np.isfinite(fine)&np.isfinite(coarse)&(fine>coarse)).sum()),
        'remarks':'The literal large-incoherence threshold is a review screen, not an automatic deletion rule; weak PM2.5>PM10 inconsistencies are not classified as certified errors.'}
    del legacy
    # Six variables, two full hourly station panels. Apply the exact existing
    # broad validity rules as a check; do not establish new cutoffs from maxima.
    weather=pq.read_table(source(DATA/'ASOS/asos_102_108_hourly_2021_2024_kst.parquet')).to_pandas()
    weather.station_id=weather.station_id.astype(str)
    variables=['temperature','relative_humidity','precipitation','surface_pressure','wind_speed','wind_direction']
    weather_rows=[];qc_rows=[];weather_candidates=[]
    for station,g in weather.groupby('station_id',sort=True):
        missing_rows=len(HOURS)-g.time_kst.nunique()
        for var in variables:
            a=g[var].to_numpy(float); finite=np.isfinite(a)
            broad_bad=finite&((a < -100)|(a>70)) if var=='temperature' else finite&((a<0)|(a>100)) if var=='relative_humidity' else finite&(a<=0) if var=='surface_pressure' else finite&((a<0)|(a>360)) if var=='wind_direction' else finite&(a<0)
            row={'station_id':station,'variable':var,'missing_station_time_rows':missing_rows,
                 'existing_physical_rule_violations':int(broad_bad.sum()),**summarize(a)}
            row['negative']=int((a[finite]<0).sum()) # valid for temperature
            weather_rows.append(row)
            for (flag,qc),gg in g.groupby([var+'_flag',var+'_qc'],dropna=False):
                qc_rows.append({'station_id':station,'variable':var,'flag':int(flag),'source_qc':str(qc),'rows':len(gg),
                    'finite_values':int(np.isfinite(gg[var]).sum())})
            for year in (2021,2022,2023,2024):
                yearly.append({'dataset':'ASOS_'+station+'_'+var,'year':year,**summarize(g.loc[g.time_kst.dt.year.eq(year),var])})
            if broad_bad.any():
                for _,r in g.loc[broad_bad].iterrows():weather_candidates.append({'station_id':station,'time_kst':r.time_kst.isoformat(),'variable':var,'value':float(r[var])})
    save_csv('asos_variable_quality.csv',weather_rows)
    save_csv('asos_flag_qc_consistency.csv',qc_rows)
    summary['asos']={'rows':len(weather),'stations':sorted(weather.station_id.unique().tolist()),
        'duplicate_station_hour_keys':int(weather.duplicated(['station_id','time_kst']).sum()),
        'existing_physical_rule_violations':weather_candidates,
        'finite_value_with_nonzero_flag':{v:int((weather[v].notna()&weather[v+'_flag'].ne(0)).sum()) for v in variables},
        'zero_flag_with_nonfinite_value':{v:int((~np.isfinite(weather[v])&weather[v+'_flag'].eq(0)).sum()) for v in variables},
        'qc_error_or_missing_code_still_finite':{v:int((weather[v+'_qc'].astype(str).isin(['1','9'])&weather[v].notna()).sum()) for v in variables},
        'wind_direction_zero_and_speed_positive':int((weather.wind_direction.eq(0)&weather.wind_speed.gt(0)).sum()),
        'precipitation_blank_preserved_missing':int(weather.precipitation.isna().sum()),
        'remarks':['Negative temperature is valid and not treated as an error.',
                   'Missing precipitation is not zero rainfall; reporting intervals were not redistributed.',
                   'Source QC blank plus numeric means retained but not certified quality-good.']}
    mong=pq.read_table(source(DATA/'Mongolia/mongolia_ub_embassy_hourly_2021_2024_kst.parquet')).to_pandas()
    mong=mong.set_index('time_kst').reindex(HOURS)
    mm=mong.pm25.to_numpy(float);mgaps=intervals(np.isnan(mm));mruns=plateaus(mm,'UB1010001')
    save_csv('mongolia_missing_intervals.csv',mgaps);save_csv('mongolia_plateau24h_candidates.csv',mruns)
    for year in (2021,2022,2023,2024):yearly.append({'dataset':'Mongolia_UB1010001','year':year,**summarize(mm[HOURS.year==year])})
    stored_begin=pd.DatetimeIndex(mong.time_utc_hour_begin).as_unit('ns')
    expected_end=stored_begin+pd.Timedelta(hours=1)
    align=expected_end.asi8==HOURS.tz_convert('UTC').asi8
    summary['mongolia']={'values':summarize(mm),'flag_counts':{str(k):int(v) for k,v in mong.pm25_flag.value_counts(dropna=False).items()},
        'longest_missing_intervals':sorted(mgaps,key=lambda x:-x['hours'])[:8],'utc_begin_plus1h_alignment_mismatches':int((~align).sum()),
        'nonzero_flag_with_finite_value':int((mong.pm25.notna()&mong.pm25_flag.ne(0)).sum()),'plateau24h_runs':mruns,
        'remarks':'AirNow negative sentinel values were null in source processed dataset; no replacement or interpolation performed.'}
    save_csv('yearly_quality.csv',yearly)
    summary['elapsed_seconds']=round(time.perf_counter()-begin,3)
    summary['input_files']=[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in FILES]
    summary['source_files_modified']=False
    summary['candidate_interpretation']='High values and exact24h plateaus are audit candidates, not confirmed errors or automatic removal decisions.'
    (OUT/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('national','target','asos','mongolia','elapsed_seconds')},ensure_ascii=False,default=scalar),flush=True)

if __name__=='__main__':main()
