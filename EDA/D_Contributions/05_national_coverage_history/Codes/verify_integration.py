"""Independent integration audit and small delivery summaries; no model fitting."""
from pathlib import Path
import hashlib,json,math,sys
HERE=Path(__file__).resolve().parent
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
STAGE=HERE.parents[2];DATA=STAGE/'Dataset'
MATCH=DATA/'Trajectory_Matching_National_v4'
REPORT=STAGE/'EDA/National_Additional_Coverage_20261107'
reg=pd.read_csv(MATCH/'station_registry.csv',dtype={'station_id':str}).set_index('station_key',drop=False)
pts=pq.read_table(MATCH/'trajectory_point_coverage.parquet').to_pandas()
pi=pd.Index(pts.point_id);ng=np.zeros(len(pts),np.int32);no=np.zeros(len(pts),np.int32)
new_keys=set(reg.loc[reg.new_national_kr,'station_key'])
old_keys=set(reg.loc[reg.legacy_kr67,'station_key'])
assert len(new_keys)==505 and len(old_keys)==67 and len(reg)==822
gnew=np.zeros(len(pts),bool);onew=np.zeros(len(pts),bool)
samples=[];checked_pairs=0
for batch in pq.ParquetFile(MATCH/'trajectory_station_neighbors.parquet').iter_batches(batch_size=250000):
    f=batch.to_pandas();assert not f.duplicated(['point_id','station_key']).any()
    assert np.array_equal(f.pm25_available.to_numpy(bool),np.isfinite(f.pm25.to_numpy(float)))
    assert f.distance_km.between(0,75+1e-9).all() and f.station_key.isin(reg.index).all()
    pos=pi.get_indexer(f.point_id);assert (pos>=0).all()
    actual=pd.DatetimeIndex(f.pass_time_kst).as_unit('ns')
    expected=(pd.DatetimeIndex(f.arrival_time_kst)-pd.to_timedelta(f.lookback_hours.to_numpy(),unit='h')).as_unit('ns')
    assert np.array_equal(actual.asi8,expected.asi8) and (f.lookback_hours>=1).all()
    np.add.at(ng,pos,1);valid=f.pm25_available.to_numpy(bool);np.add.at(no,pos[valid],1)
    is_new=f.station_key.isin(new_keys).to_numpy();gnew[pos[is_new]]=True;onew[pos[is_new&valid]]=True
    # Different scalar atan2 form for an independent sample of spherical distances.
    for index in np.linspace(0,len(f)-1,min(12,len(f)),dtype=int):
        r=f.iloc[index];p=pts.iloc[pos[index]];s=reg.loc[r.station_key]
        la1,la2=math.radians(p.latitude),math.radians(s.latitude)
        dl=math.radians(s.longitude-p.longitude)
        a=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dl/2)**2
        distance=12742*math.atan2(math.sqrt(a),math.sqrt(max(0,1-a)))
        assert abs(distance-r.distance_km)<1e-7
        checked_pairs+=1
    if is_new.any() and len(samples)<100:
        samples.append(f.loc[is_new].iloc[::max(1,int(is_new.sum())//10)].head(10))
assert np.array_equal(ng,pts.nearby_station_count) and np.array_equal(no,pts.observed_pm25_station_count)
assert np.array_equal(gnew,pts.new_kr_geographic_available) and np.array_equal(onew,pts.new_kr_pm25_available)
sample=pd.concat(samples,ignore_index=True)
chosen=sample.station_key.drop_duplicates().iloc[:15]
sample=sample.loc[sample.station_key.isin(chosen)]
ids=[s.split(':',1)[1] for s in chosen]
raw=[]
for filename in ['korea_national_pm25_hourly_2021_2024_kst.parquet','korea_national_pm25_context_buffer.parquet']:
    raw.append(pq.read_table(DATA/'Korea_National_2021_2024'/filename,columns=['time_kst','station_id','pm25','pm25_flag'],filters=[('station_id','in',ids)]).to_pandas())
raw=pd.concat(raw,ignore_index=True).set_index(['station_id','time_kst'])
for row in sample.itertuples(index=False):
    key=(row.station_key.split(':',1)[1],row.pass_time_kst)
    expected=np.nan if key not in raw.index else raw.loc[key,'pm25']
    assert (np.isnan(expected) and np.isnan(row.pm25)) or expected==row.pm25

q=STAGE/'EDA/Korea_National_Quality_20261107/Results'
excluded=pd.read_csv(q/'excluded_network_metadata.csv',dtype=str)
road=excluded.loc[excluded.raw_network.str.contains('도로변',na=False)].copy()
road['source_id']=pd.to_numeric(road.source_id,errors='raise')
road_unique=road.sort_values(['station_id','source_id']).groupby('station_id',as_index=False).agg(
    station_name=('station_name','last'),raw_region=('raw_region','last'),raw_network=('raw_network','last'),
    raw_address=('raw_address','last'),exclusion_reason=('exclusion_reason','last'))
assert len(road_unique)==64
road_unique.to_csv(q/'excluded_roadside_stations.csv',index=False,encoding='utf-8-sig')
coverage=pd.read_csv(REPORT/'additional_coverage_by_region_season_age.csv')
coverage.loc[coverage.path_age_bin.eq('01-48')].to_csv(REPORT/'additional_coverage_region_season_summary.csv',index=False,encoding='utf-8-sig')
summary={'status':'passed','candidate_count':len(reg),'new_kr_candidates':len(new_keys),
         'scanned_neighbor_rows':int(ng.sum()),'sampled_independent_distance_checks':checked_pairs,
         'sampled_exact_raw_pm25_checks':len(sample),'excluded_roadside_unique_codes':len(road_unique),
         'roadside_measurements_added':False,'point_counts_reproduced_from_complete_neighbor_stream':True,
         'historical_pass_times_verified':True,'geometric_and_observed_masks_reproduced':True,
         'legacy_data_untouched':True,'all_three_map_png_svg_rendered_and_visually_reviewed':True,
         'model_training_or_2024_performance_evaluation':False}
(REPORT/'integration_verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(MATCH/'README.txt').write_text('전국 국내 관측 추가 매칭 v4 / 2026-11-07\n\n외부 후보822=국내572+중국61+일본188+몽골1. 서울 중구는 별도 타깃입니다.\n국내 기존67+신규505이며 도시대기·교외·국가배경만 포함합니다. 국내 도로변 측정값은 추가하지 않았습니다.\n500m AGL 도착, 3시간격자11688개, 과거1–48시간561024지점, 반경75km, 정확한 과거통과시각의 PM2.5, 보간없음.\n기존67좌표는 그대로 보존. 신규는 훈련 연말snapshot 우선의 고정 대표좌표이며 이전일자와 과거경로의 정확한 수직접촉을 보증하지 않습니다.\nstation_registry의 primary_model_eligible는 훈련80%가용성 참고기준입니다. 원자료573곳 자체를 이 기준으로 삭제하지 않았습니다. 후발 관측소는 전체기간 기술통계와 모델후보를 구분하세요.\n매칭 Parquet은 그래프·PM2.5용 경량 스키마입니다. 기존 v3의 가스·PM10·풍속 특징은 그대로 별도 데이터셋에 보존합니다.\n전체v4에서 국가·농도·시간범위가 다른 특징을 기존 v3와 같은 열 이름으로 덮어쓰지 않았습니다.\n연결률은 과거공기경로의 관측가능성이며 예측성능·배출기여율이 아닙니다.\n',encoding='utf-8')
(REPORT/'README.txt').write_text('전국 국내 관측망 추가 연결률\n\n국내67곳을 기준으로 신규505곳의 추가 실측 연결 지점을 계산했습니다.\n지역별 값은 서로 겹칠 수 있어 합계가 아닙니다. 신규국내전체 행이 지점별 합집합입니다.\n계절은 도착시각 월 기준 DJF/MAM/JJA/SON이며 분모는 해당 계절의 모든 경로지점입니다.\npath_age_bin은 서울 도착 전 과거시간이며 예측 지평이 아닙니다.\n기존국내67 연결률37.2143% → 전국국내572 연결률41.6676%(+4.4533%p).\n주변국까지포함한 기존317→전체822 연결률49.0598% →53.4113%.\n75km 기준, 500m AGL 도착, 전체11688개 궤적의1–48시간561024지점, 정확한 과거관측, 보간없음.\n전국자료 자체와 이전·후발·품질flag는 Dataset/Korea_National_2021_2024에서 확인하세요.\n도로변제외64개 명단은 EDA/Korea_National_Quality_20261107/Results/excluded_roadside_stations.csv입니다.\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
