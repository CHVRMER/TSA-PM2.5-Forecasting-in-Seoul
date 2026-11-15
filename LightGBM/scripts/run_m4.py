"""Stage 4: 일본 균형판·울란바토르 검증 후 M4만 학습. 공통 함수 재사용."""
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.train import train_lightgbm
from src.data_loader import load_split
from src.feature_groups import feature_names
from src.common_scoring import sha,write_json,export_predictions,evaluate,holm
from scripts.run_m2 import protect
from scripts.run_m3 import HORIZONS
OUT=ROOT/'results/validation_2023/M4'
MODELS=ROOT/'Models/M4'
PREVIOUS=['M1','M2','M3']
PAIRS=['M2:M1','M3:M2','M3:M1','M4:M3','M4:M2','M4:M1']

def preflight():
    if (OUT/'preflight.json').exists() or MODELS.exists():raise FileExistsError('기존 M4 실행 기록 발견: 중단')
    files=[p for p in (ROOT/'results').rglob('*') if p.is_file() and 'validation_2023/M4' not in str(p) and '.matplotlib_cache' not in str(p)]
    for p in (ROOT/'Models').iterdir():
        if p.is_dir() and (p.name.startswith('M1') or p.name in ['M2','M3']):files.extend(f for f in p.rglob('*') if f.is_file())
    files += [ROOT/'Models/predictions/M1.parquet',ROOT/'Models/predictions/M1.meta.json']
    files += [p for p in (ROOT/'configs').glob('*.json') if not p.name.startswith('m4_')]
    snapshot={str(f.relative_to(ROOT)):sha(f) for f in files};protect(snapshot)
    registry=pd.read_csv(ROOT/'Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv',dtype=str)
    balanced=pd.read_csv(DATA/'japan_balanced_stations.csv',dtype=str)
    if len(balanced)!=24 or not balanced.station_key.is_unique or len(balanced.pref_code.unique())!=12 or not balanced.groupby('pref_code').size().eq(2).all():raise ValueError('일본 균형판 구성 오류')
    jp=registry[registry.station_key.isin(balanced.station_key)].copy();mn=registry[registry.country.eq('MN')].copy()
    if len(jp)!=24 or not jp.country.eq('JP').all() or len(mn)!=1 or mn.station_id.iloc[0]!='UB1010001':raise ValueError('관측소 ID·국가 오류')
    m3=feature_names(DATA,'local_plus_station_history_kr_cn',2118);m4=feature_names(DATA,'local_plus_station_history_all_balanced',2518)
    if not set(m3)<=set(m4) or len(m4)!=len(set(m4)):raise ValueError('M3 전체 포함·중복 오류')
    additions=[c for c in m4 if c not in set(m3)]
    import csv
    with (DATA/'feature_dictionary.csv').open(encoding='utf-8-sig') as f:dictionary={r['name']:r for r in csv.DictReader(f)}
    selected=pd.concat([jp,mn],ignore_index=True)
    keys={json.loads(dictionary[c]['parameters_json'])['station_key'] for c in additions}
    if len(additions)!=400 or keys!=set(selected.station_key) or sum(c.startswith('obs_jp_') for c in additions)!=384 or sum(c.startswith('obs_mn_') for c in additions)!=16:raise ValueError('일본·몽골 Feature 구성 오류')
    if any(c.startswith(('pm25_','met_','route_')) for c in m4):raise ValueError('M4에 경로 입력 혼입')
    if not set(m4)<=set(pq.read_schema(DATA/'origin_features_2021_2024_kst.parquet').names):raise ValueError('실제 필수 열 누락')
    # 기존 검증의 독립 재계산 함수만 읽기 전용으로 사용한다.
    path=ROOT/'EDA/Model_Data_Validation_20261108/audit_model_data.py'
    spec=importlib.util.spec_from_file_location('prior_audit',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    frame=pq.read_table(DATA/'origin_features_2021_2024_kst.parquet',columns=['origin_time_kst','origin_split',*additions],filters=[('origin_split','in',['train','validation'])],use_threads=False).to_pandas()
    hours=audit.hour_ids(frame.origin_time_kst);cutoff=pd.Timestamp('2024-01-01',tz='Asia/Seoul');start=pd.Timestamp('2021-01-01',tz='Asia/Seoul')
    source_files={'Japan':['japan_west_hourly_2021_2024_kst.parquet','japan_west_context_buffer.parquet'],'Mongolia':['mongolia_ub_embassy_hourly_2021_2024_kst.parquet','mongolia_ub_embassy_context_buffer.parquet']}
    quality=[]
    for dataset,names in source_files.items():
        group=selected[selected.source_dataset.eq(dataset)];cols=['time_kst','station_id','pm25','pm25_flag']+(['time_utc_hour_begin'] if dataset=='Mongolia' else [])
        parts=[]
        for filename in names:
            file=ROOT/'Dataset'/dataset/filename
            if pq.read_schema(file).field('time_kst').type!=pa.timestamp('us',tz='Asia/Seoul'):raise ValueError('시간대 타입 오류')
            parts.append(pq.read_table(file,columns=cols,filters=[('station_id','in',group.station_id.tolist()),('time_kst','<',cutoff)],use_threads=False).to_pandas())
        source=pd.concat(parts,ignore_index=True)
        if source.duplicated(['station_id','time_kst']).any() or not (source.pm25.notna()==source.pm25_flag.eq(0)).all():raise ValueError('시간 중복·값/flag 오류')
        if (pd.DatetimeIndex(source.time_kst).as_unit('us').asi8%audit.HOUR_US).any() or not (source.time_kst<cutoff).all():raise ValueError('시각 정렬·기간 오류')
        if dataset=='Mongolia' and not (source.time_kst.dt.tz_convert('UTC')-source.time_utc_hour_begin).eq(pd.Timedelta(hours=1)).all():raise ValueError('울란바토르 시작→끝 시각 정렬 오류')
        for sid,g in source.groupby('station_id'):
            country='jp' if dataset=='Japan' else 'mn'
            expected=audit.history_expected(g,hours,f'obs_{country}_{sid}_pm25_')
            for col,values in expected.items():audit.equal_numeric(frame[col],values,col)
        for period,mask in [('train',source.time_kst.ge(start)&source.time_kst.lt(pd.Timestamp('2023-01-01',tz='Asia/Seoul'))),('validation',source.time_kst.ge(pd.Timestamp('2023-01-01',tz='Asia/Seoul')))]:
            sub=source[mask];quality.append({'dataset':dataset,'period':period,'rows':len(sub),'pm25_missing':int(sub.pm25.isna().sum()),'pm25_missing_pct':100*sub.pm25.isna().mean()})
        logging.info('독립 검증: %s %s곳 x 16개 이력, %s개 시점 일치',dataset,len(group),len(frame))
    # 선정 결과를 바꾸지 않고, 2021-22 품질만 쓴 기존 균형판 선정 규칙을 대조한다.
    train_quality=pd.read_csv(ROOT/'EDA/Quality_reports/Neighbors/japan_west_quality.csv',dtype={'station_id':str})
    missing=train_quality[train_quality.period.eq('train')].set_index('station_id').pm25_missing_pct
    candidates=registry[registry.country.eq('JP')].copy();candidates['lon']=candidates.longitude.astype(float);candidates['lat']=candidates.latitude.astype(float)
    candidates['distance']=audit.distance_bearing(126.98,37.57,candidates.lon.to_numpy(),candidates.lat.to_numpy())[0]
    candidates['train_missing']=candidates.station_id.map(missing)
    chosen=[]
    for pref in sorted(balanced.pref_code.unique()):
        group=candidates[candidates.station_id.str.startswith(pref)].sort_values(['distance','station_id']);first=group.iloc[0]
        distance=audit.distance_bearing(first.lon,first.lat,group.lon.to_numpy(),group.lat.to_numpy())[0]
        rest=group[distance>=25].sort_values(['train_missing','distance','station_id']);chosen += [first.station_key,rest.iloc[0].station_key]
    if set(chosen)!=set(balanced.station_key):raise ValueError('일본 균형판의 기존 선정 규칙과 실제 목록 불일치')
    feature_quality=[]
    for period,g in frame.groupby('origin_split'):
        for country in ['jp','mn']:
            cols=[c for c in additions if c.startswith(f'obs_{country}_')]
            feature_quality.append({'country':country,'period':period,'features':len(cols),'rows':len(g),'missing_cells':int(g[cols].isna().sum().sum()),'missing_cell_pct':100*g[cols].isna().to_numpy().mean(),'all_missing_columns':[c for c in cols if g[c].isna().all()]})
    counts=[]
    for h in HORIZONS:
        config=json.loads((ROOT/f'configs/m4_{h}h.json').read_text());base=json.loads((ROOT/f'configs/m3_{h}h.json').read_text());ignore={'model','feature_set','expected_features'}
        if {k:v for k,v in config.items() if k not in ignore}!={k:v for k,v in base.items() if k not in ignore}:raise ValueError('공통 설정 불일치')
        sizes={}
        for split in ['train','validation']:
            d=load_split(DATA,config['feature_set'],h,split,2518)
            if int(d.eligible.sum())!=config['expected_train_rows' if split=='train' else 'expected_validation_rows']:raise ValueError('적격 행 수 오류')
            if not d.meta.origin_time_kst.dt.year.isin([2021,2022] if split=='train' else [2023]).all():raise ValueError('학습/검증 기간 오류')
            if d.meta.loc[d.meta[f'target_time_h{h}'].dt.year.eq(2024),f'y_pm25_h{h}'].notna().any():raise ValueError('시험 정답 노출')
            if split=='validation':
                old=pd.read_parquet(ROOT/f'results/validation_2023/M3/h{h}/predictions_M3_h{h}_validation.parquet')
                if not np.array_equal(d.eligible,old.eligible) or not np.allclose(d.meta.loc[d.eligible,f'y_pm25_h{h}'],old.loc[old.eligible,'y']):raise ValueError('기존 평가 행·정답 불일치')
            sizes[split]=int(d.eligible.sum())
        counts.append({'horizon':h,**sizes})
    selected.to_csv(OUT/'added_station_manifest.csv',index=False);pd.DataFrame(quality).to_csv(OUT/'source_missingness_train_validation.csv',index=False)
    result={'status':'passed','n_features':2518,'M3_features_preserved':2118,'added_features':400,'japan_stations':24,'mongolia_stations':1,'balanced_selection_reproduced_from_2021_2022_only':True,'history_columns_recomputed':400,'origin_rows_recomputed':len(frame),'history_mismatches':0,'japan_time_rule':'기존 승인된 README 끝시각 해석·24h→다음날00시·JST=KST 유지','mongolia_time_alignment_verified':True,'imputation_performed':False,'source_missingness':quality,'feature_missingness':feature_quality,'rows':counts,'protected_sha256':snapshot,'scorer_sha256':sha(ROOT/'src/common_scoring.py'),'test_2024_read':False}
    write_json(OUT/'preflight.json',result);return result

DATA=ROOT/'Dataset/Origin_Features_2021_2024_v3'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8',mode='x'),logging.StreamHandler()])
    try:
        check=preflight();logging.info('M4 검증 통과: M3 포함, 일본24·몽골1, 2518열. 결측 대체 없음.')
        comparisons=[(m,'persistence') for m in ['M1','M2','M3','M4']]+[tuple(x.split(':')) for x in PAIRS]
        family=[f'{h}:{a}:{b}' for h in HORIZONS for a,b in comparisons]
        write_json(OUT/'scope_before_training.json',{'train_model':'M4','reuse_models':PREVIOUS,'horizons':HORIZONS,'train_years':[2021,2022],'validation_year':2023,'test_enabled':False,'holm_family':family,'holm_family_size':40,'improvement_rule':'RMSE 차이 95% CI 상한<0 AND Holm p<0.05'})
        selected=[]
        for h in HORIZONS:
            config=json.loads((ROOT/f'configs/m4_{h}h.json').read_text());out=OUT/f'h{h}';models=MODELS/f'h{h}';out.mkdir(exist_ok=False);models.mkdir(parents=True,exist_ok=False)
            handler=logging.FileHandler(out/'run.log',encoding='utf-8');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logging.getLogger().addHandler(handler)
            try:
                logging.info('M4 %s시간 시작. 2024 Test 비활성.',h);info=train_lightgbm(ROOT,config,out,models)
            finally:logging.getLogger().removeHandler(handler);handler.close()
            selected.append({'horizon':h,'train_rows':info['eligible_train_rows'],'validation_rows':info['eligible_validation_rows'],'selected':info['selected'],'ensemble_validation':info['metrics']})
            protect(check['protected_sha256'])
        pred=OUT/'predictions';pred.mkdir(exist_ok=False)
        for model in PREVIOUS:
            source=ROOT/f'results/validation_2023/M3/predictions/{model}.parquet'
            shutil.copyfile(source,pred/source.name);shutil.copyfile(source.with_suffix('.meta.json'),pred/source.with_suffix('.meta.json').name)
        frames=[];metadata={}
        for h in HORIZONS:
            source=OUT/f'h{h}/M4.parquet';frames.append(pd.read_parquet(source));metadata[str(h)]=json.loads(source.with_suffix('.meta.json').read_text())
        export_predictions(pd.concat(frames,ignore_index=True),pred/'M4.parquet',{'seeds':[0,1,2],'horizons':HORIZONS,'n_features':2518,'training_years':[2021,2022],'selection_year':2023,'no_refit_on_2023':True,'validation_only':True,'by_horizon':metadata})
        score_dir,scores,tests=evaluate('validation',['M1','M2','M3','M4'],HORIZONS,pairs=PAIRS,prediction_dir=pred,output_root=OUT/'common_scores')
        rows=pd.read_parquet(score_dir/'score_rows_validation.parquet')
        if sha(ROOT/'src/common_scoring.py')!=check['scorer_sha256'] or len(tests)!=40 or not np.allclose(tests.p_holm,holm(tests.p)):raise ValueError('채점기·Holm 집합 오류')
        if not np.array_equal(tests.improved,(tests.ci95_high<0)&(tests.p_holm<.05)):raise ValueError('공식 판정 오류')
        if not rows.origin_time_kst.dt.year.eq(2023).all() or not rows.target_time_kst.dt.year.eq(2023).all():raise ValueError('2024 채점 발생')
        old_manifest=json.loads((ROOT/'results/validation_2023/M3/manifest.json').read_text());old_rows=pd.read_parquet(ROOT/old_manifest['common_scores']/'score_rows_validation.parquet')
        for h in HORIZONS:
            details=pd.read_parquet(OUT/f'h{h}/predictions_M4_h{h}_validation.parquet');info=json.loads((OUT/f'h{h}/metrics_and_selection.json').read_text())
            if not np.allclose(details.y_hat_raw,details[[f'y_hat_seed{s}' for s in [0,1,2]]].mean(axis=1)) or any(r['max_absolute_prediction_difference']!=0 for r in info['model_reload_checks']):raise ValueError('앙상블·재로드 오류')
            g=rows[rows.horizon.eq(h)].sort_values('origin_time_kst');old=old_rows[old_rows.horizon.eq(h)].sort_values('origin_time_kst')
            if g.origin_time_kst.tolist()!=old.origin_time_kst.tolist() or not np.array_equal(g[['y','persistence',*PREVIOUS]].to_numpy(),old[['y','persistence',*PREVIOUS]].to_numpy()):raise ValueError('기존 공통 평가 행·예측 변경')
            for model in ['persistence',*PREVIOUS,'M4']:
                r=scores[(scores.horizon==h)&(scores.model==model)].iloc[0];e=g[model]-g.y
                if not np.allclose([r.RMSE,r.MAE,r.MBE],[np.sqrt(np.mean(e**2)),np.mean(abs(e)),e.mean()]):raise ValueError('지표 재계산 오류')
        scores.to_csv(OUT/'M1_M2_M3_M4_Persistence_2023_validation_metrics.csv',index=False);tests.to_csv(OUT/'M1_M2_M3_M4_2023_validation_tests.csv',index=False)
        direct=tests[(tests.model=='M4')&tests.versus.isin(PREVIOUS)].rename(columns={'rmse_diff':'delta_rmse','ci95_low':'delta_rmse_ci_lower','ci95_high':'delta_rmse_ci_upper','p_holm':'holm_adjusted_p'})
        direct[['horizon','model','versus','n','delta_rmse','delta_rmse_ci_lower','delta_rmse_ci_upper','dm_stat','p','holm_adjusted_p','improved']].to_csv(OUT/'M4_vs_M1_M2_M3_2023_validation_statistics.csv',index=False)
        shutil.copyfile(score_dir/'rmse_improvement_validation.png',OUT/'M1_M2_M3_M4_2023_validation_intermediate_rmse_improvement.png')
        write_json(OUT/'horizon_settings.json',selected);protect(check['protected_sha256'])
        code=[ROOT/'src/train.py',ROOT/'src/feature_groups.py',ROOT/'src/data_loader.py',ROOT/'src/common_scoring.py',Path(__file__)]+[ROOT/f'configs/m4_{h}h.json' for h in HORIZONS]
        write_json(OUT/'manifest.json',{'status':'completed','model':'M4','horizons':HORIZONS,'official_result_path':str(OUT.relative_to(ROOT)),'model_root':str(MODELS.relative_to(ROOT)),'common_scores':str(score_dir.relative_to(ROOT)),'n_features':2518,'M3_subset':True,'japan_balanced_stations':24,'mongolia_stations':1,'imputation_performed':False,'train_years':[2021,2022],'validation_year':2023,'test_2024_executed':False,'M1_M2_M3_retrained_or_repredicted':False,'M3T_M4T_SHAP_Adaptive_started':False,'new_models':12,'reload_prediction_max_difference':0,'previous_artifacts_preserved':len(check['protected_sha256']),'scorer_unchanged':True,'common_rows':rows.groupby('horizon').size().to_dict(),'holm_family':family,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code}})
        logging.info('Stage 4 M4 네 Horizon 완료. 후속 모델·Test 미실행.')
        print(scores.to_string(index=False));print(tests[tests.model.eq('M4')].to_string(index=False));print('공식 결과:',OUT)
    except Exception:
        logging.exception('오류 발생: 후속 작업 없이 중단합니다.');raise
if __name__=='__main__':main()
