"""Read-only China PM data audit; no cleaning, imputation or model fitting.

Long constant runs and isolated peaks below are candidates for review, not
certified instrument errors. Future neighbours are used ONLY for descriptive
spike diagnosis; they must not be copied into an operational QC filter.
"""
from pathlib import Path
from io import BytesIO
import argparse,hashlib,json,zipfile
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root',type=Path,default=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project'))
parser.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]/'Results/china'/('recheck_'+__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+__import__('uuid').uuid4().hex[:8]))
args=parser.parse_args()
PROJECT=args.project_root.resolve()
DATA=PROJECT/'Dataset'
OUT=args.output_root.resolve()
DATASETS={
 'china_48':DATA/'China/china_48stations_hourly_2021_2024_kst.parquet',
 'china_border_13':DATA/'China_Border_Proxy/china_border_cities_hourly_2021_2024_kst.parquet',
 'china_east_30_qc':DATA/'Neighbors_Expansion_QC_20261108/china_east_hourly_qc_2021_2024_kst.parquet'}
BUFFERS={
 'china_48':DATA/'China/china_48stations_context_buffer.parquet',
 'china_border_13':DATA/'China_Border_Proxy/china_border_cities_context_buffer.parquet',
 'china_east_30_qc':DATA/'Neighbors_Expansion_QC_20261108/china_east_context_buffer_qc.parquet'}
SOURCE_REGISTRIES={'china_48':DATA/'China/source_registry.csv',
 'china_border_13':DATA/'China_Border_Proxy/source_registry.csv'}
POLICY_PATH=DATA/'Neighbors_Expansion_QC_20261108/qc_policy.json'
REGISTRY=PROJECT/'Model/Common/Regional12_Seed11_20261109/station_registry.csv'
CNY_DATES={'2021-02-12','2022-02-01','2023-01-22','2024-02-10'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()

def json_ready(value):
 if isinstance(value,dict):return {str(k):json_ready(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [json_ready(v) for v in value]
 if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
 if isinstance(value,np.generic):return value.item()
 return value

def dump(name,value):
 (OUT/name).write_text(json.dumps(json_ready(value),ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')

def table(name,rows,columns=None):
 frame=pd.DataFrame(rows,columns=columns)
 frame.to_csv(OUT/name,index=False,encoding='utf-8-sig')

def finite(x):return np.isfinite(np.asarray(x,dtype=float))

def evaluate_policy(d):
 """Independently apply the already documented new-city PM25 policy to all CN inputs.

 This does not change any value; it only locates additional review candidates.
 Past medians are calculated from original source-flag0 observations, once.
 """
 x=d.pm25_raw if 'pm25_raw' in d else d.pm25
 flags=d.pm25_raw_flag if 'pm25_raw_flag' in d else d.pm25_flag
 coarse=d.pm10_raw if 'pm10_raw' in d else d.pm10
 cflags=d.pm10_raw_flag if 'pm10_raw_flag' in d else d.pm10_flag
 valid=x.where(flags.eq(0)&np.isfinite(x))
 past=valid.shift(1).rolling(6,min_periods=4).median()
 jump=flags.eq(0)&x.ge(1000)&x.gt(10*past.clip(lower=10))&(x-past).gt(500)
 incoherent=flags.eq(0)&x.ge(300)&cflags.eq(0)&coarse.gt(0)&x.gt(2*coarse)&(x-coarse).gt(100)
 return past,jump.fillna(False),incoherent.fillna(False)

def numeric_summary(values,flags):
 a=np.asarray(values,dtype=float);good=np.isfinite(a);v=a[good]
 f=np.asarray(flags)
 return {'rows':len(a),'null_count':int(pd.isna(a).sum()),'missing_fraction':float(pd.isna(a).mean()),
  'negative_remaining_count':int((a<0).sum()),'nonfinite_nonnull_count':int((~good & ~pd.isna(a)).sum()),
  'zero_count':int((a==0).sum()),'max':float(v.max()) if len(v) else None,
  'q50':float(np.quantile(v,.5)) if len(v) else None,'q99':float(np.quantile(v,.99)) if len(v) else None,
  'q999':float(np.quantile(v,.999)) if len(v) else None,'greater_500':int((a>500).sum()),
  'greater_equal_1000':int((a>=1000).sum()),
  'flag_counts':{str(k):int(v) for k,v in pd.Series(f).value_counts(dropna=False).items()},
  'flag0_null_count':int(((f==0)&pd.isna(a)).sum()),
  'nonzero_flag_numeric_count':int(((f!=0)&good).sum())}

def rows_for_events(frame,mask,kind,group,selected):
 columns=['station_id','time_kst','time_cst','pm25','pm25_flag','pm10','pm10_flag']
 columns += [name for name in ['pm25_source_id','pm10_source_id','source_id','pm25_raw','pm25_raw_flag','pm25_qc_reason_bits'] if name in frame]
 out=frame.loc[mask,columns].copy()
 out.insert(0,'dataset',group);out.insert(1,'candidate_kind',kind)
 out['station_key']='CN:'+out.station_id.astype(str)
 out['used_in_current_F634']=out.station_key.isin(selected)
 out['cst_date']=out.time_cst.dt.strftime('%Y-%m-%d')
 out['is_cny_cst_date']=out.cst_date.isin(CNY_DATES)
 return out.to_dict('records')

def audit():
 OUT.mkdir(parents=True,exist_ok=True)
 registry=pd.read_csv(REGISTRY,dtype={'station_id':str})
 selected=set(registry.station_key.astype(str))
 sources={};station_year=[];long_runs=[];all_extremes=[];spikes=[];policy_events=[];daily=[];summaries={};freeze_summary=[]
 source_snapshot={str(p):sha(p) for p in [*DATASETS.values(),*BUFFERS.values(),REGISTRY,POLICY_PATH]}
 for label,path in DATASETS.items():
  frame=pd.read_parquet(path);frame['station_id']=frame.station_id.astype(str)
  frame=frame.sort_values(['station_id','time_kst'],kind='stable').reset_index(drop=True)
  buffer=pd.read_parquet(BUFFERS[label]);buffer['station_id']=buffer.station_id.astype(str)
  expected_times=pd.date_range('2021-01-01','2025-01-01',inclusive='left',freq='h',tz='Asia/Seoul')
  by_station=frame.groupby('station_id',sort=True)
  station_counts=by_station.time_kst.agg(['size','nunique','min','max'])
  summary={'file':str(path),'rows':len(frame),'station_count':int(frame.station_id.nunique()),
   'duplicates_station_time':int(frame.duplicated(['station_id','time_kst']).sum()),
   'each_station_exact_expected_hours':bool(station_counts['size'].eq(len(expected_times)).all() and station_counts['nunique'].eq(len(expected_times)).all()
       and station_counts['min'].eq(expected_times[0]).all() and station_counts['max'].eq(expected_times[-1]).all()
       and frame.time_kst.dt.minute.eq(0).all() and frame.time_kst.dt.second.eq(0).all()),
   'time_kst_dtype':str(frame.time_kst.dtype),'time_cst_dtype':str(frame.time_cst.dtype),
   'cst_and_kst_same_instant':bool((frame.time_kst.astype('int64')==frame.time_cst.astype('int64')).all()),
   'time_min_kst':str(frame.time_kst.min()),'time_max_kst':str(frame.time_kst.max()),
   'pm25':numeric_summary(frame.pm25,frame.pm25_flag),
   'pm10':numeric_summary(frame.pm10,frame.pm10_flag),
   'station_pm25_flag0_counts':{str(k):int(v) for k,v in frame.loc[frame.pm25_flag.eq(0)].groupby('station_id').size().items()},
   'used_in_current_F634_stations':int(sum('CN:'+s in selected for s in frame.station_id.unique()))}
  summary['other_pollutants']={name:numeric_summary(frame[name],frame[name+'_flag'])
    for name in ('so2','no2','o3','co') if name in frame and name+'_flag' in frame}
  paired=frame.pm25.notna()&frame.pm10.notna()
  summary['pm25_greater_pm10_count']=int((paired&frame.pm25.gt(frame.pm10)).sum())
  summary['pm25_ge300_gt2pm10_difference100_count']=int((paired&frame.pm25.ge(300)&frame.pm10.gt(0)&frame.pm25.gt(2*frame.pm10)&(frame.pm25-frame.pm10).gt(100)).sum())
  all_extremes += rows_for_events(frame,frame.pm25.gt(500),'pm25_gt500_review_only',label,selected)
  counts=frame.loc[frame.pm25.gt(500)].groupby(frame.loc[frame.pm25.gt(500)].time_cst.dt.strftime('%Y-%m-%d')).agg(n=('pm25','size'),max=('pm25','max'),stations=('station_id','nunique')).reset_index()
  for row in counts.to_dict('records'):
   row.update(dataset=label,is_cny_cst_date=row['time_cst'] in CNY_DATES);daily.append(row)
  summary['pm25_gt500_on_four_cny_cst_dates']=int((frame.pm25.gt(500)&frame.time_cst.dt.strftime('%Y-%m-%d').isin(CNY_DATES)).sum())
  summary['pm25_gt500_on_four_cny_kst_dates']=int((frame.pm25.gt(500)&frame.time_kst.dt.strftime('%Y-%m-%d').isin(CNY_DATES)).sum())
  summary['pm25_gt500_on_2024_cny_cst_date']=int((frame.pm25.gt(500)&frame.time_cst.dt.strftime('%Y-%m-%d').eq('2024-02-10')).sum())
  summary['pm25_gt500_on_2024_cny_kst_date']=int((frame.pm25.gt(500)&frame.time_kst.dt.strftime('%Y-%m-%d').eq('2024-02-10')).sum())
  for sid,g in by_station:
   g=g.reset_index(drop=True);x=g.pm25.to_numpy(dtype=float);valid=np.isfinite(x)&g.pm25_flag.eq(0).to_numpy()
   # Pandas may retain Arrow microsecond datetime units; normalize explicitly.
   t=pd.DatetimeIndex(g.time_kst).as_unit('ns').asi8;adjacent=np.r_[False,np.diff(t)==3600*10**9]
   same=np.r_[False,(x[1:]==x[:-1])&valid[1:]&valid[:-1]]&adjacent
   starts=np.flatnonzero(~same);ends=np.r_[starts[1:],len(g)]
   for lo,hi in zip(starts,ends):
    if not valid[lo] or hi-lo<24:continue
    h=g.iloc[lo:hi];coarse=h.pm10.dropna()
    long_runs.append({'dataset':label,'station_id':sid,'station_key':'CN:'+sid,
      'used_in_current_F634':'CN:'+sid in selected,'start_kst':str(h.time_kst.iloc[0]),'end_kst':str(h.time_kst.iloc[-1]),
      'start_cst':str(h.time_cst.iloc[0]),'end_cst':str(h.time_cst.iloc[-1]),
      'observations':len(h),'elapsed_hours':int((h.time_kst.iloc[-1]-h.time_kst.iloc[0]).total_seconds()/3600),
      'constant_pm25':float(x[lo]),'zero_value':bool(x[lo]==0),'pm10_valid_hours':len(coarse),
      'pm10_range':float(coarse.max()-coarse.min()) if len(coarse) else None,
      'pm10_std_ddof0':float(coarse.std(ddof=0)) if len(coarse) else None,
      'diagnostic_only':True,'not_certified_sensor_failure':True})
   for year,gy in g.groupby(g.time_kst.dt.year):
    vv=gy.pm25.dropna();station_year.append({'dataset':label,'station_id':sid,'year':int(year),'used_in_current_F634':'CN:'+sid in selected,
      'rows':len(gy),'valid_hours':len(vv),'missing_fraction':float(gy.pm25.isna().mean()),'max_pm25':float(vv.max()) if len(vv) else None,
      'gt500_hours':int(gy.pm25.gt(500).sum()),'zero_hours':int(gy.pm25.eq(0).sum())})
   context=buffer[buffer.station_id.eq(sid)&buffer.time_kst.lt(g.time_kst.iloc[0])].sort_values('time_kst').tail(6)
   with_context=pd.concat([context,g],ignore_index=True)
   past,jump,incoherent=[v.iloc[len(context):].reset_index(drop=True) for v in evaluate_policy(with_context)]
   review=jump|incoherent
   ev=rows_for_events(g,review,'existing_newcity_policy_applied_for_audit_only',label,selected)
   for event,idx in zip(ev,np.flatnonzero(review.to_numpy())):
    event.update(past6_median=float(past.iloc[idx]) if pd.notna(past.iloc[idx]) else None,
      extreme_own_history_jump=bool(jump.iloc[idx]),coarse_incoherence=bool(incoherent.iloc[idx]),uses_future_context=False)
   policy_events+=ev
   # Symmetric isolated-peak screen: descriptive; never a production feature or filter.
   history=pd.concat([g.pm25.shift(k) for k in (1,2,3,-1,-2,-3)],axis=1)
   surrounding=history.median(axis=1);left=g.pm25.shift(1);right=g.pm25.shift(-1)
   # Require actual one-hour adjacent neighbours; all datasets are separately checked on calendar.
   one_before=g.time_kst.diff().eq(pd.Timedelta(hours=1));one_after=g.time_kst.shift(-1).sub(g.time_kst).eq(pd.Timedelta(hours=1))
   candidate=g.pm25_flag.eq(0)&g.pm25.ge(100)&history.count(axis=1).ge(4)&g.pm25.gt(5*surrounding.clip(lower=10))&left.notna()&right.notna()&g.pm25.gt(3*left.clip(lower=10))&g.pm25.gt(3*right.clip(lower=10))&one_before&one_after
   ev=rows_for_events(g,candidate,'symmetric_isolated_peak_review_only',label,selected)
   for event,idx in zip(ev,np.flatnonzero(candidate.to_numpy())):
    event.update(prev_hour_pm25=float(left.iloc[idx]),next_hour_pm25=float(right.iloc[idx]),surrounding6_median=float(surrounding.iloc[idx]),uses_future_context=True,diagnostic_only=True)
   spikes+=ev
  runs=[r for r in long_runs if r['dataset']==label]
  summary['constant_runs']={'definition':'At least24 equal, finite sourceflag0 values at contiguous hourly timestamps; no invalid/missing bridge',
   'runs_ge24_observations':len(runs),'hours_ge24_observations':sum(r['observations'] for r in runs),
   'nonzero_runs_ge24_observations':sum(not r['zero_value'] for r in runs),
   'nonzero_hours_ge24_observations':sum(r['observations'] for r in runs if not r['zero_value']),
   'zero_hours_ge24_observations':sum(r['observations'] for r in runs if r['zero_value']),
   'hours_constant_value_3':sum(r['observations'] for r in runs if r['constant_pm25']==3),
   'runs_constant_value_3':sum(r['constant_pm25']==3 for r in runs),
   'pm10_also_constant_runs':sum(r['pm10_range']==0 and r['pm10_valid_hours']>0 for r in runs),
   'hours_elapsed_at_least24':sum(r['observations'] for r in runs if r['elapsed_hours']>=24)}
  summary['symmetric_isolated_peak_candidates']=sum(e['dataset']==label for e in spikes)
  summary['existing_policy_review_candidates']=sum(e['dataset']==label for e in policy_events)
  summary['existing_policy_candidate_breakdown']={
    'jump':sum(e['dataset']==label and e['extreme_own_history_jump'] for e in policy_events),
    'coarse_incoherence':sum(e['dataset']==label and e['coarse_incoherence'] for e in policy_events),
    'union_on_four_cny_cst_dates':sum(e['dataset']==label and e['is_cny_cst_date'] for e in policy_events)}
  if 'pm25_raw' in frame:
   summary['raw_pm25']=numeric_summary(frame.pm25_raw,frame.pm25_raw_flag)
   summary['analysis_pm25_cells_different_from_raw']=int((~np.isclose(frame.pm25,frame.pm25_raw,equal_nan=True)).sum())
   summary['analysis_pm10_cells_different_from_raw']=int((~np.isclose(frame.pm10,frame.pm10_raw,equal_nan=True)).sum())
   summary['flag20_pm25_count']=int(frame.pm25_flag.eq(20).sum());summary['flag20_pm10_count']=int(frame.pm10_flag.eq(20).sum())
  summary['buffer']={'rows':len(buffer),'time_min_kst':str(buffer.time_kst.min()),'time_max_kst':str(buffer.time_kst.max()),
    'overlap_main_station_time':int(buffer.merge(frame[['station_id','time_kst']],on=['station_id','time_kst'],how='inner').shape[0]),
    'pm25_missing_fraction':float(buffer.pm25.isna().mean())}
  summaries[label]=summary
  print(label,json.dumps({'missing':summary['pm25']['missing_fraction'],'max':summary['pm25']['max'],'gt500':summary['pm25']['greater_500'],'constant':summary['constant_runs'],'policy_review':summary['existing_policy_review_candidates']},ensure_ascii=False),flush=True)
  if label=='china_48':
   pollutant_top=[]
   for species in ('so2','no2','o3','co'):
    top=frame.nlargest(5,species).copy();top['review_species']=species;pollutant_top.append(top)
   pd.concat(pollutant_top).to_csv(OUT/'other_pollutants_top5_review.csv',index=False,encoding='utf-8-sig')
   moment=pd.Timestamp('2021-03-15 16:00',tz='Asia/Seoul');window=frame.station_id.eq('3578A')&frame.time_kst.between(moment-pd.Timedelta(hours=12),moment+pd.Timedelta(hours=12))
   frame.loc[window].to_csv(OUT/'3578A_9985_context.csv',index=False,encoding='utf-8-sig')
   frame.loc[frame.time_kst.eq(moment)].to_csv(OUT/'3578A_9985_simultaneous_cn48.csv',index=False,encoding='utf-8-sig')
   stale=frame.station_id.eq('2230A')&frame.time_kst.between(pd.Timestamp('2021-01-26 02:00',tz='Asia/Seoul'),pd.Timestamp('2021-01-28 21:00',tz='Asia/Seoul'))
   frame.loc[stale].to_csv(OUT/'2230A_constant48_context_all_pollutants.csv',index=False,encoding='utf-8-sig')
   source=pd.read_csv(SOURCE_REGISTRIES[label]);source=source.loc[source.source_id.eq(79)].iloc[0].to_dict()
   raw_result={'source_registry_record':source,'upstream_content_checked':False}
   raw_path=Path(source['source_path'])
   try:
    if raw_path.suffix.lower()=='.zip':
     with zipfile.ZipFile(raw_path) as z:raw=z.read(source['zip_member'])
    else:raw=raw_path.read_bytes()
    raw_result.update(upstream_content_checked=True,actual_sha256=hashlib.sha256(raw).hexdigest(),matches_registered_sha256=hashlib.sha256(raw).hexdigest()==source['sha256_raw'])
    native=pd.read_csv(BytesIO(raw));native=native[(native.hour==15)&native.type.isin(['PM2.5','PM10'])]
    raw_result['raw_rows']=native[[col for col in ['date','hour','type','3578A'] if col in native]].to_dict('records')
   except Exception as err:raw_result.update(check_failure_type=type(err).__name__,check_failure_message=str(err))
   dump('3578A_9985_source_evidence.json',raw_result)
 table('station_year_quality.csv',station_year)
 table('constant_run_intervals.csv',long_runs)
 table('pm25_gt500_events.csv',all_extremes)
 table('isolated_spike_review_candidates.csv',spikes)
 table('existing_causal_qc_policy_review_candidates.csv',policy_events)
 table('daily_extreme_counts_cst.csv',daily)
 source_unchanged={str(p):sha(p) for p in [*DATASETS.values(),*BUFFERS.values(),REGISTRY,POLICY_PATH]}
 assert source_snapshot==source_unchanged,'An input changed during audit.'
 dump('source_manifest.json',{'status':'passed','inputs_unchanged':True,'sha256':source_snapshot})
 dump('china_audit_summary.json',{'status':'passed','datasets':summaries,
   'notes':['Read-only audit; no data replaced or model trained.','Greater500 is a review threshold, not physical cap or automatic removal.','Constant runs are not proof of sensor failure.','Symmetric isolated peaks use future context for audit only.','Chinese New Year timing coincidence is not certified fireworks source attribution.','Existing new-city QC was preserved and independently evaluated; base48 was not automatically cleaned.']})

if __name__=='__main__':audit()
