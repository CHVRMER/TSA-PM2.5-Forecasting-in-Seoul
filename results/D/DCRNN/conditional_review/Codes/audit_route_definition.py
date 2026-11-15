"""Audit existing route labels and sample context; no conditional tests or fitting."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORK=Path(__file__).resolve().parent
OLD=WORK.parent/'dcrnn_h48_results_review_20261112'
PROJECT=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
OUT=WORK/'Tables'
SEASONS=['DJF','MAM','JJA','SON']
ROUTE_MAP={'NW':'west','W':'west','SW':'west','N':'north','NE':'north',
           'E':'east','SE':'east','S':'east','LOCAL':'local'}
SEASON_MAP={12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',
            6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def fingerprints(paths):
    return [{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]

def plain(value):
    return json.loads(value.to_json(orient='records'))

def route_from_xy(lon,lat):
    # Identical great-circle convention to the original feature builder.
    x1,y1,x2,y2=[np.deg2rad(np.asarray(v,dtype=float)) for v in (126.98,37.57,lon,lat)]
    dx=x2-x1;dy=y2-y1
    a=np.sin(dy/2)**2+np.cos(y1)*np.cos(y2)*np.sin(dx/2)**2
    distance=6371*2*np.arctan2(np.sqrt(np.clip(a,0,1)),np.sqrt(np.clip(1-a,0,1)))
    bearing=np.arctan2(np.sin(dx)*np.cos(y2),np.cos(y1)*np.sin(y2)-np.sin(y1)*np.cos(y2)*np.cos(dx))
    bearing=np.where(distance>1e-9,bearing,np.nan)
    compass=np.array(['N','NE','E','SE','S','SW','W','NW'])
    sector=compass[np.floor(((np.degrees(bearing)%360)+22.5)%360/45).astype(int)]
    return np.where(distance<25,'LOCAL',sector),distance,np.degrees(bearing)%360

def summarize(part,scope,horizon):
    rows=[]
    for (split,route),g in part.groupby(['split','route_group'],dropna=False):
        x=g.origin_pm25_exact.dropna()
        row={'split':split,'year':int(g.origin_time_kst.dt.year.iloc[0]),'scope':scope,
             'horizon':horizon,'route_group':route,'n_origins':len(g),
             'unique_origin_days':g.origin_time_kst.dt.normalize().nunique(),
             'origin_first_kst':g.origin_time_kst.min().isoformat(),
             'origin_last_kst':g.origin_time_kst.max().isoformat(),
             'origin_pm25_observed_n':len(x),'origin_pm25_missing_n':int(g.origin_pm25_exact.isna().sum()),
             'origin_pm25_missing_pct':100*g.origin_pm25_exact.isna().mean(),
             'origin_pm25_mean':x.mean(),'origin_pm25_std':x.std(ddof=0),
             'origin_pm25_min':x.min(),'origin_pm25_p10':x.quantile(.10),
             'origin_pm25_p25':x.quantile(.25),'origin_pm25_median':x.median(),
             'origin_pm25_p75':x.quantile(.75),'origin_pm25_p90':x.quantile(.90),
             'origin_pm25_max':x.max(),'origin_pm25_ge35_pct_among_observed':100*x.ge(35).mean()}
        for s in SEASONS:
            row[f'season_{s}_n']=int(g.season.eq(s).sum())
            row[f'season_{s}_pct']=100*g.season.eq(s).mean()
        rows.append(row)
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    grid_path=OLD/'Verification/rebuilt_evaluation_grid.parquet'
    feature_path=PROJECT/'Dataset/Origin_Features_2021_2024_v3/origin_features_2021_2024_kst.parquet'
    target_path=WORK.parent/'dcrnn_vast48_20261111/verification_data/target_hourly_audit.parquet'
    target_source=PROJECT/'Dataset/Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet'
    source_code=[PROJECT/'Dataset/Codes/Origin_Features/build_origin_features.py',
                 PROJECT/'Dataset/Codes/Origin_Features/feature_spec.py',
                 PROJECT/'Dataset/Codes/Evaluation_Contract/build_evaluation_grid.py',
                 PROJECT/'Model/Common/DCRNN_H48_20261111/grid.py',
                 PROJECT/'Model/Common/DCRNN_H48_20261111/horizon_contract.py']
    grid=pd.read_parquet(grid_path)
    calendar=grid[grid.horizon.eq(3)&grid.split.isin(['validation','test'])].copy()
    assert not calendar.origin_time_kst.duplicated().any()
    assert (calendar.feature_cutoff_time_kst==calendar.origin_time_kst).all()
    assert np.array_equal(calendar.season,calendar.origin_time_kst.dt.month.map(SEASON_MAP))
    assert np.array_equal(calendar.cold_season,np.where(calendar.origin_time_kst.dt.month.isin([11,12,1,2,3]),'cold','warm'))
    assert np.array_equal(calendar.route_group,calendar.route_history12h_sector.map(ROUTE_MAP).fillna('unknown'))
    # Recompute original eight-sector labels from the actual 12h coordinates.
    f=pq.read_table(feature_path,columns=['origin_time_kst','route_latitude_at12h',
                    'route_longitude_at12h','route_history12h_sector']).to_pandas()
    sectors,distance,bearing=route_from_xy(f.route_longitude_at12h,f.route_latitude_at12h)
    assert np.array_equal(sectors,f.route_history12h_sector)
    route_audit=f[['origin_time_kst','route_history12h_sector']].copy()
    route_audit['distance_at12h_km']=distance
    route_audit['bearing_at12h_deg']=bearing
    merged=calendar.merge(route_audit,on=['origin_time_kst','route_history12h_sector'],validate='one_to_one')
    assert len(merged)==len(calendar)
    # Exact QC-clean current observation; never use a filled persistence forecast.
    target=pd.read_parquet(target_path)
    assert not target.time_kst.duplicated().any()
    current=pq.read_table(target_source,columns=['time_kst','pm25','pm25_flag'],
                filters=[('station_id','==','111121')]).to_pandas()
    valid=np.isfinite(current.pm25)&current.pm25.ge(0)&current.pm25_flag.eq(0)
    current['pm25_qc']=current.pm25.where(valid)
    check=current[['time_kst','pm25_qc']].merge(target,on='time_kst',validate='one_to_one')
    assert len(check)==len(current)
    assert np.allclose(check.pm25_qc,check.pm25,rtol=0,atol=0,equal_nan=True)
    target=target.rename(columns={'time_kst':'origin_time_kst','pm25':'origin_pm25_exact'})
    calendar=merged.merge(target,on='origin_time_kst',how='left',validate='one_to_one')
    all_rows=summarize(calendar,'all_calendar_origins',0)
    active=grid[grid.split.isin(['validation','test'])&grid.eligible].copy()
    active=active.merge(calendar[['origin_time_kst','origin_pm25_exact']],on='origin_time_kst',how='left',validate='many_to_one')
    for h,g in active.groupby('horizon'):
        assert np.array_equal(g.season,g.origin_time_kst.dt.month.map(SEASON_MAP))
        all_rows.extend(summarize(g,'eligible_forecast_rows',int(h)))
    summary=pd.DataFrame(all_rows)
    summary.to_csv(OUT/'route_sample_context.csv',index=False,encoding='utf-8-sig')
    # Keep a compact, nonduplicated route × season composition table for reading.
    cross=calendar.groupby(['split','route_group','season'],observed=True).size().rename('n_origins').reset_index()
    cross['within_route_pct']=100*cross.n_origins/cross.groupby(['split','route_group']).n_origins.transform('sum')
    cross.to_csv(OUT/'route_season_composition.csv',index=False,encoding='utf-8-sig')
    exact=calendar.origin_pm25_exact.notna()
    fallback_checks=[]
    for split,g in calendar.groupby('split'):
        good=g.origin_pm25_exact.notna()
        diff=g.loc[good,'persistence_y_hat']-g.loc[good,'origin_pm25_exact']
        assert np.max(abs(diff))<1e-8
        fallback_checks.append({'split':split,'calendar_origins':len(g),
            'current_observed_n':int(good.sum()),'current_missing_n':int((~good).sum()),
            'persistence_fallback_n':int(g.persistence_fallback.sum()),
            'max_abs_exact_current_vs_persistence_when_current_observed':float(np.max(abs(diff)))})
    full_summary=summary[summary.scope.eq('all_calendar_origins')]
    audit={'analysis':'Existing label definition and route sample context audit; no inferential tests or model fitting',
       'route_definition':{'receptor_latitude':37.57,'receptor_longitude':126.98,
           'arrival_height_agl_m':500,'trajectory_age_h':12,
           'meaning':'Bearing from the Seoul receptor to the position 12 hours before arrival on the backward trajectory ending at the current origin t.',
           'distance':'Great-circle distance on a sphere of radius 6371 km',
           'local_if_distance_km_strictly_less_than':25,
           'compass':'Eight 45-degree bins, clockwise from north; north spans [337.5,360) and [0,22.5).',
           'route_group_map':ROUTE_MAP,
           'caveats':['east also includes the S sector; north includes NE; groups have unequal angular widths.',
                      'This is not instantaneous wind direction, a country-of-origin label, an emission source, or pollution mass contribution.',
                      'The endpoint at t-12h belongs to the trajectory arriving at t; no trajectory arriving at t+h is used.',
                      'Timestamp causality does not establish historical GDAS publication/processing availability for real-time operation.']},
       'season_definition':{'timestamp':'origin_time_kst (Asia/Seoul), not target_time_kst',
           'month_map':SEASON_MAP,'cold_months':[11,12,1,2,3],
           'note':'At month/season boundaries, all horizons keep the season of forecast issuance t.'},
       'implementation_locations':[
          {'path':str(source_code[0]),'lines':[55,61,190,200,233,234]},
          {'path':str(source_code[1]),'lines':[11,12]},
          {'path':str(source_code[2]),'lines':[63,75]},
          {'path':str(source_code[3]),'lines':[65,86]},
          {'path':str(source_code[4]),'lines':[108,131]}],
       'checks':{'route_8sector_recomputed_from_actual_12h_coordinates':True,
           'all_feature_origins_recomputed_n':len(f),'group_map_matches_grid':True,
           'season_matches_origin_month':True,'cold_season_matches_origin_month':True,
           'feature_cutoff_equals_origin':True,'exact_current_matches_original_QC_target':True,
           'current_concentration_imputed':False,'persistence_used_as_current_concentration':False,
           'seasons_or_routes_changed':False,'season_route_inferential_tests_performed':False},
       'calendar_route_summary':plain(full_summary),
       'current_observation_vs_persistence_audit':fallback_checks,
       'table_scope_notes':{'horizon_0':'all unique calendar origins, including origins without an eligible future target',
           'horizon_3_to_48':'fixed eligible rows at the specified future horizon; counts differ due to missing future targets and split boundaries',
           'unique_origin_days':'KST dates on which a route appears; routes may share the same day and observations are serially dependent',
           'concentration_units':'µg/m³; observed exact origin concentration only; missing current values excluded only from concentration summaries',
           'percentile_definition':'pandas linear empirical quantile','std_ddof':0},
       'provenance':fingerprints([grid_path,feature_path,target_path,target_source]+source_code)}
    (WORK/'route_definition_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(full_summary[['split','route_group','n_origins','unique_origin_days','origin_pm25_observed_n',
                       'origin_pm25_mean','origin_pm25_median','origin_pm25_p90',
                       'season_DJF_pct','season_MAM_pct','season_JJA_pct','season_SON_pct']].to_string(index=False))
    print(json.dumps(fallback_checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
