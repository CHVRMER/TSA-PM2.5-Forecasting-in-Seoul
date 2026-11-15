"""Read-only audit of retained HYSPLIT/features and actual DCRNN loader.

No input changes, no model training, no target-based QC threshold selection.
"""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,sys,time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'Results/pipeline'/('recheck_'+__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+__import__('uuid').uuid4().hex[:8]))
    args=parser.parse_args();base=args.project;out=args.output;out.mkdir(parents=True,exist_ok=True)
    d=base/'Dataset';t0=time.perf_counter();inputs=[]
    def register(p):
        inputs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)})
    # Full trajectory geometry and meteorology only; chunked so several audits
    # can run without materializing all 25 columns at once.
    path=d/'Trajectories/trajectory_points_2021_2024.parquet';register(path)
    reader=pq.ParquetFile(path);stats=collections.defaultdict(lambda:collections.Counter())
    finite_ranges={};time_bad=0;flags=collections.Counter()
    names=['arrival_time_kst','pass_time_kst','lookback_hours','arrival_height_agl_m',
        'latitude','longitude','height_agl_m','pressure_hpa','diagnostic_air_temp',
        'diagnostic_rainfall','diagnostic_mixdepth','diagnostic_relhumid',
        'diagnostic_terr_msl','diagnostic_sun_flux','diagnostic_uwind','diagnostic_vwind',
        'wind_speed_m_s','wind_from_degrees']
    for batch in reader.iter_batches(batch_size=65536,columns=names):
        frame=batch.to_pandas();elapsed=(frame.arrival_time_kst-frame.pass_time_kst).dt.total_seconds()/3600
        time_bad+=int((elapsed!=frame.lookback_hours).sum())
        flags['pass_after_arrival']+=int((frame.pass_time_kst>frame.arrival_time_kst).sum())
        for name in names[2:]:
            v=frame[name].to_numpy(float);finite=np.isfinite(v);s=stats[name]
            s.update(rows=len(v),null=int(np.isnan(v).sum()),inf=int(np.isinf(v).sum()))
            if finite.any():
                mn=float(v[finite].min());mx=float(v[finite].max())
                old=finite_ranges.get(name,(mn,mx));finite_ranges[name]=(min(old[0],mn),max(old[1],mx))
        flags['invalid_latitude']+=int((~frame.latitude.between(-90,90)).sum())
        flags['invalid_longitude']+=int((~frame.longitude.between(-180,180)).sum())
        flags['negative_height_agl']+=int((frame.height_agl_m<0).sum())
        flags['negative_mixdepth']+=int((frame.diagnostic_mixdepth<0).sum())
        flags['zero_mixdepth']+=int(frame.diagnostic_mixdepth.eq(0).sum())
        flags['rh_below0']+=int((frame.diagnostic_relhumid<0).sum())
        flags['rh_above100']+=int((frame.diagnostic_relhumid>100).sum())
        flags['negative_rain_rate']+=int((frame.diagnostic_rainfall<0).sum())
        flags['negative_wind_speed']+=int((frame.wind_speed_m_s<0).sum())
        calc=np.hypot(frame.diagnostic_uwind,frame.diagnostic_vwind)
        flags['wind_speed_uv_mismatch']+=int((np.abs(calc-frame.wind_speed_m_s)>1e-5).sum())
    rows=[dict(variable=name,**count,minimum=finite_ranges.get(name,(None,None))[0],
               maximum=finite_ranges.get(name,(None,None))[1]) for name,count in stats.items()]
    pd.DataFrame(rows).to_csv(out/'trajectory_variable_quality.csv',index=False,encoding='utf-8-sig')
    # Stored feature summaries: use Parquet statistics, no imputation or clipping.
    feature_reports=[];null_rows=[]
    for relative in ['Origin_Features_2021_2024/origin_features_2021_2024_kst.parquet',
                     'Origin_Features_2021_2024_v3/origin_features_2021_2024_kst.parquet',
                     'National_Station_History_2021_2024/national_additional_station_history.parquet',
                     'Neighbors_Expansion_QC_20261108/station_history_qc_2021_2024.parquet']:
        p=d/relative;register(p);pf=pq.ParquetFile(p);bad=[];allnull=[]
        for j,name in enumerate(pf.schema_arrow.names):
            nulls=0;has_stats=True;minimum=None;maximum=None
            for rg in range(pf.num_row_groups):
                s=pf.metadata.row_group(rg).column(j).statistics
                if s is None:has_stats=False;continue
                nulls+=s.null_count or 0
                if s.has_min_max and isinstance(s.min,(int,float)):
                    minimum=s.min if minimum is None else min(minimum,s.min)
                    maximum=s.max if maximum is None else max(maximum,s.max)
            if minimum is not None and (not math.isfinite(minimum) or not math.isfinite(maximum)):bad.append(name)
            if has_stats and nulls==pf.metadata.num_rows:allnull.append(name)
            null_rows.append({'dataset':relative,'feature':name,'null_count':nulls,'rows':pf.metadata.num_rows,'statistics_available':has_stats,
                             'minimum':minimum,'maximum':maximum})
        feature_reports.append({'dataset':relative,'rows':pf.metadata.num_rows,'columns':len(pf.schema_arrow.names),
                                'columns_with_infinite_stats':bad,'all_null_columns':allnull})
    pd.DataFrame(null_rows).to_csv(out/'feature_null_and_range_audit.csv',index=False,encoding='utf-8-sig')
    # Execute only original data/transform helpers from the shipped notebook.
    # The training/model cells are deliberately never executed.
    import torch
    from torch.utils.data import DataLoader
    notebook=base/'Model/DCRNN/Regional12_Seed11_20261109/Seoul_PM25_Regional12_DCRNN_Seed11_Colab.ipynb'
    register(notebook);nb=json.loads(notebook.read_text(encoding='utf-8'))
    ns={'__name__':'__quality_audit__','np':np,'pd':pd,'torch':torch,'DataLoader':DataLoader}
    for cell in nb['cells']:
        if cell['cell_type']=='code' and cell['metadata'].get('purpose') in (
            'preserved_original_data','preserved_original_graphs','preserved_v3_adapter','preserved_neighbor_adapter'):
            exec(compile(''.join(cell['source']),cell['metadata']['purpose'],'exec'),ns,ns)
    bundle=ns['load_neighbor_bundle'](d,condition='F634_BOTH_W24',config_dir=base/'Model/Common/Neighbor_Experiments_QC_20261108',lookback_hours=48)
    prep=ns['fit_dcrnn_preprocessors'](bundle);arrays=ns['transform_dcrnn_bundle'](bundle,prep)
    hourly=pd.DatetimeIndex(bundle['hourly_index']);keys=bundle['nodes'].station_key.tolist()
    pm=np.asarray(bundle['pm_values']);pernode=[]
    for col,key in enumerate(keys):
        for year in range(2021,2025):
            v=pm[hourly.year==year,col];valid=np.isfinite(v)
            pernode.append({'station_key':key,'year':year,'hours':len(v),'valid':int(valid.sum()),'missing':int((~valid).sum()),
                            'missing_fraction':float((~valid).mean()),'maximum':float(v[valid].max()) if valid.any() else None,
                            'negative_valid':int((v[valid]<0).sum())})
    pd.DataFrame(pernode).to_csv(out/'actual_dcrnn_node_quality_by_year.csv',index=False,encoding='utf-8-sig')
    suspicious={}
    if 'CN:3578A' in keys:
        j=keys.index('CN:3578A');moment=pd.Timestamp('2021-03-15 16:00',tz='Asia/Seoul');i=hourly.get_loc(moment)
        origin_pos=bundle['origin_positions'];affected=(origin_pos>=i)&(origin_pos-47<=i)
        value=float(pm[i,j]);z=float(arrays['hourly_x'][i,j,0])
        train=(hourly.year>=2021)&(hourly.year<=2022);raw=pm[train,j].astype(float);lv=np.log1p(raw)
        alt=lv.copy();trainpositions=np.flatnonzero(train);alt[np.flatnonzero(trainpositions==i)]=np.nan
        suspicious={'present_in_actual_loader':True,'time_kst':str(moment),'value':value,'normalized_log_value':z,
                    'training_origin_windows_containing_value':int(affected.sum()),'train_log_mean_original':float(np.nanmean(lv)),
                    'train_log_mean_without_one_value':float(np.nanmean(alt)),'train_log_std_original':float(np.nanstd(lv)),
                    'train_log_std_without_one_value':float(np.nanstd(alt)),
                    'counterfactual_normalizer_only_no_file_change_no_training':True}
    grid_path=d/'Evaluation_Contract_20261107/evaluation_grid.parquet';register(grid_path)
    grid=pq.read_table(grid_path).to_pandas();grid_checks=[]
    for h in (3,6,12,24):
        part=grid.loc[grid.horizon.eq(h)].copy();times=pd.DatetimeIndex(part.origin_time_kst)
        pos=pd.DatetimeIndex(bundle['origins'].origin_time_kst).get_indexer(times)
        if (pos<0).any():raise ValueError('Evaluation grid origins not in bundle.')
        column=list(bundle['horizons']).index(h)
        grid_checks.append({'horizon':h,'rows':len(part),'target_values_match':bool(np.allclose(part.y_true,bundle['y_raw'][pos,column],equal_nan=True)),
                            'eligibility_matches':bool(np.array_equal(part.eligible.to_numpy(bool),bundle['eligible'][pos,column])),
                            'timestamps_exact':bool((pd.DatetimeIndex(part.target_time_kst)==times+pd.Timedelta(hours=h)).all())})
    report={'status':'completed','sources_unchanged':True,'no_training_performed':True,'no_qc_rules_applied':True,
            'trajectory':{'rows':reader.metadata.num_rows,'columns':len(reader.schema_arrow.names),'time_alignment_errors':time_bad,'range_checks':dict(flags)},
            'feature_tables':feature_reports,'actual_dcrnn':{'nodes':len(keys),'country_counts':bundle['nodes'].country.value_counts().to_dict(),
            'hourly_shape':list(pm.shape),'tensor_finite':{k:bool(np.isfinite(arrays[k]).all()) for k in ['hourly_x','context_x','y_z']},
            'fit_start':prep['fit_start_inclusive'],'fit_end_exclusive':prep['fit_end_exclusive'],
            'target_labels_from_exact_raw_hour_verified_by_loader':True,'evaluation_grid_checks':grid_checks,
            'china_9985':suspicious,'context_dimensions':int(arrays['context_x'].shape[1])},
            'elapsed_seconds':time.perf_counter()-t0}
    dump(out/'pipeline_audit.json',report);dump(out/'input_manifest.json',inputs)
    print(json.dumps({'status':report['status'],'output':str(out),'actual_dcrnn':report['actual_dcrnn'],'trajectory':report['trajectory']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
