"""M2 전용 검증·실행 조율. 공통 로더·학습 함수·채점기를 재사용한다."""
import csv
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.train import train_lightgbm
from src.data_loader import load_split
from src.feature_groups import feature_names
from src.common_scoring import sha,write_json,export_predictions,evaluate,holm
OUT=ROOT/'results/validation_2023/M2'
MODEL_ROOT=ROOT/'Models/M2'
DATA=ROOT/'Dataset/Origin_Features_2021_2024_v3'

def protect(snapshot):
    changed=[p for p,h in snapshot.items() if sha(ROOT/p)!=h]
    if changed:raise ValueError('기존 M1 결과 변경: '+str(changed))
    if (ROOT/'results/final_test_2024.lock').exists():raise ValueError('2024 시험 잠금이 소비되었습니다.')

def preflight():
    if (OUT/'preflight.json').exists() or MODEL_ROOT.exists():raise FileExistsError('기존 M2 실행 기록: 덮어쓰지 않습니다.')
    protected=[]
    for p in (ROOT/'results').rglob('*'):
        if p.is_file() and 'validation_2023/M2' not in str(p) and '.matplotlib_cache' not in str(p):protected.append(p)
    for p in (ROOT/'Models').glob('M1*'):
        if p.is_dir():protected.extend(f for f in p.rglob('*') if f.is_file())
    protected.extend(ROOT/p for p in ['Models/predictions/M1.parquet','Models/predictions/M1.meta.json'])
    protected.extend((ROOT/'configs').glob('m1_*.json'))
    snapshot={str(p.relative_to(ROOT)):sha(p) for p in protected}
    protect(snapshot)
    registry=pd.read_csv(ROOT/'Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv',dtype=str)
    lo,la=np.deg2rad(registry.longitude.astype(float)),np.deg2rad(registry.latitude.astype(float));lo0,la0=np.deg2rad([126.98,37.57])
    a=np.sin((la-la0)/2)**2+np.cos(la)*np.cos(la0)*np.sin((lo-lo0)/2)**2
    registry['distance_km']=6371*2*np.arcsin(np.sqrt(a))
    near=registry[(registry.country=='CN')&(registry.source_dataset=='China')&(registry.distance_km<=600)].copy()
    if len(near)!=21 or not near.station_key.is_unique:raise ValueError('중국 21곳 선정 오류')
    definition=json.loads((ROOT/'configs/feature_groups.json').read_text())['M2']
    if set(near.station_key)!=set(definition['station_keys']):raise ValueError('중국 거리 계산과 입력 정의 불일치')
    base=feature_names(DATA,'local_plus_station_history_kr',1142);names=feature_names(DATA,'M2',1478)
    additions=[n for n in names if n not in set(base)]
    if not set(base)<=set(names) or len(additions)!=336 or len(names)!=len(set(names)):raise ValueError('M2 Feature 구성 오류')
    schema=pq.read_schema(DATA/'origin_features_2021_2024_kst.parquet')
    if not set(names)<=set(schema.names):raise ValueError('실제 데이터에 입력 열 누락')
    frame=pq.read_table(DATA/'origin_features_2021_2024_kst.parquet',columns=['origin_time_kst',*additions],filters=[('origin_split','in',['train','validation'])],use_threads=False).to_pandas()
    cutoff=pd.Timestamp('2024-01-01',tz='Asia/Seoul')
    pieces=[]
    for filename in ['china_48stations_hourly_2021_2024_kst.parquet','china_48stations_context_buffer.parquet']:
        pieces.append(pq.read_table(ROOT/'Dataset/China'/filename,columns=['time_kst','time_cst','station_id','pm25','pm25_flag'],filters=[('station_id','in',near.station_id.tolist()),('time_kst','<',cutoff)],use_threads=False).to_pandas())
    source=pd.concat(pieces,ignore_index=True)
    if source.duplicated(['station_id','time_kst']).any() or not (source.time_kst==source.time_cst).all():raise ValueError('중국 시간 정렬/중복 오류')
    if not (source.pm25.notna()==source.pm25_flag.eq(0)).all():raise ValueError('중국 값/flag 오류')
    if not (source.time_kst<cutoff).all():raise ValueError('2024 원자료가 포함되었습니다.')
    # 이전 독립 검증 함수 재사용: 모든 21곳 x 16개 이력을 t 이하 원자료로 재계산.
    path=ROOT/'EDA/Model_Data_Validation_20261108/audit_model_data.py'
    spec=importlib.util.spec_from_file_location('prior_audit',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    hours=audit.hour_ids(frame.origin_time_kst)
    for sid,g in source.groupby('station_id'):
        expected=audit.history_expected(g,hours,f'obs_cn_{sid}_pm25_')
        for col,values in expected.items():audit.equal_numeric(frame[col],values,col)
    count_rows=[]
    for h in [3,6,12,24]:
        c=json.loads((ROOT/f'configs/m2_{h}h.json').read_text());m1=json.loads((ROOT/f'configs/m1_{h}h.json').read_text())
        ignore={'model','feature_set','expected_features'}
        if {k:v for k,v in c.items() if k not in ignore}!={k:v for k,v in m1.items() if k not in ignore}:raise ValueError('M1과 학습 조건이 다릅니다.')
        sizes={}
        for split in ['train','validation']:
            d=load_split(DATA,'M2',h,split,1478)
            expected=c['expected_train_rows' if split=='train' else 'expected_validation_rows']
            if d.eligible.sum()!=expected:raise ValueError('적격 행 수 불일치')
            if not d.meta.origin_time_kst.dt.year.isin([2021,2022] if split=='train' else [2023]).all():raise ValueError('분할 기간 오류')
            if d.meta.loc[d.meta[f'target_time_h{h}'].dt.year.eq(2024),f'y_pm25_h{h}'].notna().any():raise ValueError('2024 정답 노출')
            sizes[split]=int(d.eligible.sum())
        count_rows.append({'horizon':h,**sizes})
    near.to_csv(OUT/'china_21_station_manifest.csv',index=False)
    result={'status':'passed','features':1478,'M1_features_included':1142,'additional_features':336,'chinese_stations':21,'distance_range_km':[float(near.distance_km.min()),float(near.distance_km.max())],'distance_rule':'Haversine R=6371, Seoul(126.98,37.57), <=600; no boundary station, same 21 as existing <600 rule','cn_station_history_rows_checked':len(frame),'cn_history_columns_recomputed':336,'history_mismatches':0,'timezones_same_instants':True,'rows':count_rows,'protected_sha256':snapshot,'scorer_sha256':sha(ROOT/'src/common_scoring.py'),'test_2024_read':False}
    write_json(OUT/'preflight.json',result);return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8',mode='x'),logging.StreamHandler()])
    try:
        check=preflight();logging.info('M2 사전 검증 통과: 1478열, 중국 21곳, 2024 미사용.')
        family=[f'{h}:{a}:{b}' for h in [3,6,12,24] for a,b in [('M1','persistence'),('M2','persistence'),('M2','M1')]]
        write_json(OUT/'scope_before_training.json',{'model_to_train':'M2','horizons':[3,6,12,24],'reuse_model':'M1','holm_family':family,'holm_scope':'현재 요청된 비교쌍 x 4개 Horizon, 기존 공통 채점 규칙 그대로. 전체 프로젝트 최종 검정 family를 새로 정의하지 않음.','test_enabled':False})
        selected=[]
        for h in [3,6,12,24]:
            c=json.loads((ROOT/f'configs/m2_{h}h.json').read_text());result=OUT/f'h{h}';models=MODEL_ROOT/f'h{h}'
            result.mkdir(exist_ok=False);models.mkdir(parents=True,exist_ok=False)
            handler=logging.FileHandler(result/'run.log',encoding='utf-8');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logging.getLogger().addHandler(handler)
            try:
                logging.info('M2 %s시간 학습 시작',h);info=train_lightgbm(ROOT,c,result,models)
            finally:logging.getLogger().removeHandler(handler);handler.close()
            selected.append({'horizon':h,'train_rows':info['eligible_train_rows'],'validation_rows':info['eligible_validation_rows'],'selected':info['selected'],'ensemble_validation':info['metrics']})
            protect(check['protected_sha256'])
        pred=OUT/'predictions';pred.mkdir(exist_ok=False)
        m1=ROOT/'results/M1_allhorizon_validation_20261108_182422/common_predictions/M1.parquet'
        shutil.copyfile(m1,pred/'M1.parquet')
        m1_meta=json.loads(m1.with_suffix('.meta.json').read_text());m1_meta['n_features']=1142
        write_json(pred/'M1.meta.json',m1_meta)
        frames=[];by_horizon={}
        for h in [3,6,12,24]:
            path=OUT/f'h{h}/M2.parquet';frames.append(pd.read_parquet(path));by_horizon[str(h)]=json.loads(path.with_suffix('.meta.json').read_text())
        export_predictions(pd.concat(frames,ignore_index=True),pred/'M2.parquet',{'seeds':[0,1,2],'horizons':[3,6,12,24],'n_features':1478,'training_years':[2021,2022],'selection_year':2023,'no_refit_on_2023':True,'validation_only':True,'by_horizon':by_horizon})
        score_dir,scores,tests=evaluate('validation',['M1','M2'],[3,6,12,24],pairs=['M2:M1'],prediction_dir=pred,output_root=OUT/'common_scores')
        if sha(ROOT/'src/common_scoring.py')!=check['scorer_sha256']:raise ValueError('공통 채점기 변경')
        if len(tests)!=12:raise ValueError('Holm 검정 집합 오류')
        if not np.allclose(tests.p_holm,holm(tests.p)):raise ValueError('Holm 보정 오류')
        if not np.array_equal(tests.improved,(tests.ci95_high<0)&(tests.p_holm<.05)):raise ValueError('개선 판정 오류')
        rows=pd.read_parquet(score_dir/'score_rows_validation.parquet')
        if not rows.origin_time_kst.dt.year.eq(2023).all() or not rows.target_time_kst.dt.year.eq(2023).all():raise ValueError('2024 채점 발생')
        for h in [3,6,12,24]:
            original=pd.read_parquet(OUT/f'h{h}/predictions_M2_h{h}_validation.parquet');info=json.loads((OUT/f'h{h}/metrics_and_selection.json').read_text())
            if not np.allclose(original.y_hat_raw,original[[f'y_hat_seed{s}' for s in [0,1,2]]].mean(axis=1)):raise ValueError('앙상블 오류')
            if any(x['max_absolute_prediction_difference']!=0 for x in info['model_reload_checks']):raise ValueError('재로드 오류')
            if original.loc[original.target_time_kst.dt.year.eq(2024),'y'].notna().any():raise ValueError('시험 정답 노출')
            g=rows[rows.horizon.eq(h)]
            for model in ['persistence','M1','M2']:
                m=scores[(scores.horizon==h)&(scores.model==model)].iloc[0];error=g[model]-g.y
                if not np.isclose(m.RMSE,np.sqrt(np.mean(error**2))) or not np.isclose(m.MBE,error.mean()):raise ValueError('공통 지표 재계산 불일치')
        scores.to_csv(OUT/'M1_M2_Persistence_2023_validation_metrics.csv',index=False);tests.to_csv(OUT/'M1_M2_2023_validation_tests.csv',index=False)
        shutil.copyfile(score_dir/'rmse_improvement_validation.png',OUT/'M1_M2_2023_validation_intermediate_rmse_improvement.png')
        write_json(OUT/'horizon_settings.json',selected)
        protect(check['protected_sha256'])
        paths=[ROOT/'src/train.py',ROOT/'src/feature_groups.py',ROOT/'src/evaluate.py',ROOT/'src/common_scoring.py',Path(__file__),ROOT/'configs/feature_groups.json',ROOT/'configs/experiment_rules.json']+[ROOT/f'configs/m2_{h}h.json' for h in [3,6,12,24]]
        write_json(OUT/'manifest.json',{'status':'completed','model':'M2','horizons':[3,6,12,24],'official_result_path':str(OUT.relative_to(ROOT)),'model_root':str(MODEL_ROOT.relative_to(ROOT)),'common_scores':str(score_dir.relative_to(ROOT)),'n_features':1478,'chinese_station_count':21,'train_years':[2021,2022],'validation_year':2023,'test_2024_executed':False,'M1_retrained_or_repredicted':False,'new_models':12,'new_model_reload_max_difference':0,'original_M1_artifacts_preserved':len(check['protected_sha256']),'scorer_unchanged':True,'common_rows':rows.groupby('horizon').size().to_dict(),'holm_family':family,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
        logging.info('Stage 2 완료: M1 보존, M2 4개 Horizon 완료. Test 미실행.')
        print(scores.to_string(index=False));print(tests.to_string(index=False));print('공식 결과:',OUT)
    except Exception:
        logging.exception('검증 또는 실행 오류. 후속 작업 없이 중단합니다.');raise
if __name__=='__main__':main()
