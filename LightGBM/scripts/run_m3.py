"""Stage 3 조율: 검증 후 M3 네 Horizon 학습. M1/M2 및 공통 채점기 재사용."""
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
import pyarrow.parquet as pq
from src.train import train_lightgbm
from src.data_loader import load_split
from src.feature_groups import feature_names
from src.common_scoring import sha,write_json,export_predictions,evaluate,holm
from scripts.run_m2 import protect
OUT=ROOT/'results/validation_2023/M3'
MODELS=ROOT/'Models/M3'
DATA=ROOT/'Dataset/Origin_Features_2021_2024_v3'
HORIZONS=[3,6,12,24]
PAIRS=['M2:M1','M3:M2','M3:M1']

def preflight():
    if (OUT/'preflight.json').exists() or MODELS.exists():raise FileExistsError('기존 M3 실행 기록: 덮어쓰기 없이 중단')
    files=[p for p in (ROOT/'results').rglob('*') if p.is_file() and 'validation_2023/M3' not in str(p) and '.matplotlib_cache' not in str(p)]
    for p in (ROOT/'Models').iterdir():
        if p.is_dir() and (p.name.startswith('M1') or p.name=='M2'):files.extend(f for f in p.rglob('*') if f.is_file())
    files += [ROOT/'Models/predictions/M1.parquet',ROOT/'Models/predictions/M1.meta.json']
    files += list((ROOT/'configs').glob('m1_*.json'))+list((ROOT/'configs').glob('m2_*.json'))+[ROOT/'configs/feature_groups.json',ROOT/'configs/experiment_rules.json']
    snapshot={str(f.relative_to(ROOT)):sha(f) for f in files};protect(snapshot)
    registry=pd.read_csv(ROOT/'Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv',dtype=str)
    cn=registry[registry.country.eq('CN')].copy()
    if len(cn)!=61 or not cn.station_key.is_unique or cn.source_dataset.value_counts().to_dict()!={'China':48,'China_Border_Proxy':13}:raise ValueError('중국 61곳 구성 오류')
    near=set(json.loads((ROOT/'configs/feature_groups.json').read_text())['M2']['station_keys'])
    if len(near)!=21 or not near<=set(cn.station_key):raise ValueError('M2의 중국 21곳이 모두 포함되지 않음')
    m1=feature_names(DATA,'local_plus_station_history_kr',1142);m2=feature_names(DATA,'M2',1478);m3=feature_names(DATA,'local_plus_station_history_kr_cn',2118)
    if not set(m1)<=set(m2)<=set(m3) or len(m3)!=len(set(m3)):raise ValueError('M1/M2/M3 Feature 포함·중복 오류')
    additions=[c for c in m3 if c not in set(m1)]
    if len(additions)!=976 or any(c.startswith(('pm25_','met_','route_')) for c in m3):raise ValueError('M3 관측소 이력 범위 오류')
    with (DATA/'feature_dictionary.csv').open(encoding='utf-8-sig') as f:
        import csv
        dictionary={r['name']:r for r in csv.DictReader(f)}
    keys={json.loads(dictionary[c]['parameters_json'])['station_key'] for c in additions}
    if keys!=set(cn.station_key):raise ValueError('중국 Feature와 registry 불일치')
    if not set(m3)<=set(pq.read_schema(DATA/'origin_features_2021_2024_kst.parquet').names):raise ValueError('필수 열 누락')
    frame=pq.read_table(DATA/'origin_features_2021_2024_kst.parquet',columns=['origin_time_kst',*additions],filters=[('origin_split','in',['train','validation'])],use_threads=False).to_pandas()
    path=ROOT/'EDA/Model_Data_Validation_20261108/audit_model_data.py'
    spec=importlib.util.spec_from_file_location('prior_audit',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    hours=audit.hour_ids(frame.origin_time_kst);cutoff=pd.Timestamp('2024-01-01',tz='Asia/Seoul')
    sources={'China':['china_48stations_hourly_2021_2024_kst.parquet','china_48stations_context_buffer.parquet'],'China_Border_Proxy':['china_border_cities_hourly_2021_2024_kst.parquet','china_border_cities_context_buffer.parquet']}
    for dataset,filenames in sources.items():
        ids=cn[cn.source_dataset.eq(dataset)].station_id.tolist()
        source=pd.concat([pq.read_table(ROOT/'Dataset'/dataset/f,columns=['time_kst','time_cst','station_id','pm25','pm25_flag'],filters=[('station_id','in',ids),('time_kst','<',cutoff)],use_threads=False).to_pandas() for f in filenames],ignore_index=True)
        if source.duplicated(['station_id','time_kst']).any() or not (source.time_kst==source.time_cst).all():raise ValueError('중국 시간대·중복 오류')
        if not (source.pm25.notna()==source.pm25_flag.eq(0)).all() or not (source.time_kst<cutoff).all():raise ValueError('원자료 플래그·기간 오류')
        for sid,g in source.groupby('station_id'):
            expected=audit.history_expected(g,hours,f'obs_cn_{sid}_pm25_')
            for col,values in expected.items():audit.equal_numeric(frame[col],values,col)
        logging.info('독립 이력 검증: %s %s곳 x 16개 x %s시점 일치',dataset,len(ids),len(frame))
    counts=[]
    for h in HORIZONS:
        config=json.loads((ROOT/f'configs/m3_{h}h.json').read_text());base=json.loads((ROOT/f'configs/m2_{h}h.json').read_text());ignore={'model','feature_set','expected_features'}
        if {k:v for k,v in config.items() if k not in ignore}!={k:v for k,v in base.items() if k not in ignore}:raise ValueError('M2와 공통 학습 조건 불일치')
        sizes={}
        for split in ['train','validation']:
            d=load_split(DATA,config['feature_set'],h,split,2118)
            expected=config['expected_train_rows' if split=='train' else 'expected_validation_rows']
            if int(d.eligible.sum())!=expected:raise ValueError('적격 표본 수 불일치')
            if not d.meta.origin_time_kst.dt.year.isin([2021,2022] if split=='train' else [2023]).all():raise ValueError('분할 기간 오류')
            if d.meta.loc[d.meta[f'target_time_h{h}'].dt.year.eq(2024),f'y_pm25_h{h}'].notna().any():raise ValueError('2024 정답 노출')
            if split=='validation':
                old=pd.read_parquet(ROOT/f'results/validation_2023/M2/h{h}/predictions_M2_h{h}_validation.parquet')
                if not np.array_equal(d.eligible,old.eligible) or not np.allclose(d.meta.loc[d.eligible,f'y_pm25_h{h}'],old.loc[old.eligible,'y']):raise ValueError('M2와 평가 시점·정답 불일치')
            sizes[split]=int(d.eligible.sum())
        counts.append({'horizon':h,**sizes})
    cn['included_in_M2']=cn.station_key.isin(near);cn.to_csv(OUT/'china_61_station_manifest.csv',index=False)
    result={'status':'passed','n_features':2118,'M1_features':1142,'M2_features':1478,'M2_subset':True,'chinese_information_points':61,'real_stations':48,'border_proxy_cities':13,'china_history_columns_recomputed':976,'origin_rows_recomputed':len(frame),'history_mismatches':0,'timestamps_aligned':True,'no_trajectory_inputs':True,'rows':counts,'protected_sha256':snapshot,'scorer_sha256':sha(ROOT/'src/common_scoring.py'),'test_2024_read':False}
    write_json(OUT/'preflight.json',result);return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8',mode='x'),logging.StreamHandler()])
    try:
        check=preflight();logging.info('M3 사전 검증 통과: 중국61, M2 포함, 2118열')
        comparisons=[('M1','persistence'),('M2','persistence'),('M3','persistence'),('M2','M1'),('M3','M2'),('M3','M1')]
        family=[f'{h}:{a}:{b}' for h in HORIZONS for a,b in comparisons]
        write_json(OUT/'scope_before_training.json',{'train_model':'M3','reuse_models':['M1','M2'],'horizons':HORIZONS,'train_years':[2021,2022],'validation_year':2023,'test_enabled':False,'holm_family':family,'holm_family_size':24,'improvement_rule':'RMSE 차이 95% CI 상한<0 AND Holm p<0.05'})
        selected=[]
        for h in HORIZONS:
            config=json.loads((ROOT/f'configs/m3_{h}h.json').read_text());out=OUT/f'h{h}';models=MODELS/f'h{h}';out.mkdir(exist_ok=False);models.mkdir(parents=True,exist_ok=False)
            handler=logging.FileHandler(out/'run.log',encoding='utf-8');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logging.getLogger().addHandler(handler)
            try:
                logging.info('M3 %s시간 시작. 2024 Test 비활성.',h);info=train_lightgbm(ROOT,config,out,models)
            finally:logging.getLogger().removeHandler(handler);handler.close()
            selected.append({'horizon':h,'train_rows':info['eligible_train_rows'],'validation_rows':info['eligible_validation_rows'],'selected':info['selected'],'ensemble_validation':info['metrics']})
            protect(check['protected_sha256'])
        pred=OUT/'predictions';pred.mkdir(exist_ok=False)
        for model in ['M1','M2']:
            source=ROOT/f'results/validation_2023/M2/predictions/{model}.parquet'
            shutil.copyfile(source,pred/source.name);shutil.copyfile(source.with_suffix('.meta.json'),pred/source.with_suffix('.meta.json').name)
        frames=[];metadata={}
        for h in HORIZONS:
            source=OUT/f'h{h}/M3.parquet';frames.append(pd.read_parquet(source));metadata[str(h)]=json.loads(source.with_suffix('.meta.json').read_text())
        export_predictions(pd.concat(frames,ignore_index=True),pred/'M3.parquet',{'seeds':[0,1,2],'horizons':HORIZONS,'n_features':2118,'training_years':[2021,2022],'selection_year':2023,'no_refit_on_2023':True,'validation_only':True,'by_horizon':metadata})
        score_dir,scores,tests=evaluate('validation',['M1','M2','M3'],HORIZONS,pairs=PAIRS,prediction_dir=pred,output_root=OUT/'common_scores')
        rows=pd.read_parquet(score_dir/'score_rows_validation.parquet')
        if sha(ROOT/'src/common_scoring.py')!=check['scorer_sha256'] or len(tests)!=24 or not np.allclose(tests.p_holm,holm(tests.p)):raise ValueError('채점기·Holm 집합 불일치')
        if not np.array_equal(tests.improved,(tests.ci95_high<0)&(tests.p_holm<.05)):raise ValueError('공식 판정 오류')
        if not rows.origin_time_kst.dt.year.eq(2023).all() or not rows.target_time_kst.dt.year.eq(2023).all():raise ValueError('2024 채점 발생')
        old_metrics=pd.read_csv(ROOT/'results/validation_2023/M2/M1_M2_Persistence_2023_validation_metrics.csv')
        for h in HORIZONS:
            details=pd.read_parquet(OUT/f'h{h}/predictions_M3_h{h}_validation.parquet');info=json.loads((OUT/f'h{h}/metrics_and_selection.json').read_text())
            if not np.allclose(details.y_hat_raw,details[[f'y_hat_seed{s}' for s in [0,1,2]]].mean(axis=1)) or any(r['max_absolute_prediction_difference']!=0 for r in info['model_reload_checks']):raise ValueError('앙상블·재로드 오류')
            g=rows[rows.horizon==h]
            for model in ['persistence','M1','M2','M3']:
                value=scores[(scores.horizon==h)&(scores.model==model)].iloc[0];error=g[model]-g.y
                if not np.allclose([value.RMSE,value.MAE,value.MBE],[np.sqrt(np.mean(error**2)),np.mean(abs(error)),error.mean()]):raise ValueError('공통 지표 불일치')
                if model!='M3':
                    old=old_metrics[(old_metrics.horizon==h)&(old_metrics.model==model)].iloc[0]
                    if not np.allclose([value.RMSE,value.MAE,value.MBE],[old.RMSE,old.MAE,old.MBE]) or value.n!=old.n:raise ValueError('기존 M1/M2 평가 행·점수 변경')
        scores.to_csv(OUT/'M1_M2_M3_Persistence_2023_validation_metrics.csv',index=False);tests.to_csv(OUT/'M1_M2_M3_2023_validation_tests.csv',index=False)
        direct=tests[(tests.model=='M3')&tests.versus.isin(['M1','M2'])].rename(columns={'rmse_diff':'delta_rmse','ci95_low':'delta_rmse_ci_lower','ci95_high':'delta_rmse_ci_upper','p_holm':'holm_adjusted_p'})
        direct[['horizon','model','versus','n','delta_rmse','delta_rmse_ci_lower','delta_rmse_ci_upper','dm_stat','p','holm_adjusted_p','improved']].to_csv(OUT/'M3_vs_M1_M2_2023_validation_statistics.csv',index=False)
        shutil.copyfile(score_dir/'rmse_improvement_validation.png',OUT/'M1_M2_M3_2023_validation_intermediate_rmse_improvement.png')
        write_json(OUT/'horizon_settings.json',selected);protect(check['protected_sha256'])
        code=[ROOT/'src/train.py',ROOT/'src/feature_groups.py',ROOT/'src/data_loader.py',ROOT/'src/common_scoring.py',Path(__file__)]+[ROOT/f'configs/m3_{h}h.json' for h in HORIZONS]
        write_json(OUT/'manifest.json',{'status':'completed','model':'M3','horizons':HORIZONS,'official_result_path':str(OUT.relative_to(ROOT)),'model_root':str(MODELS.relative_to(ROOT)),'common_scores':str(score_dir.relative_to(ROOT)),'n_features':2118,'chinese_information_points':61,'real_stations':48,'border_proxy_cities':13,'M2_subset':True,'train_years':[2021,2022],'validation_year':2023,'test_2024_executed':False,'M1_M2_retrained_or_repredicted':False,'M4_started':False,'new_models':12,'reload_prediction_max_difference':0,'previous_artifacts_preserved':len(check['protected_sha256']),'scorer_unchanged':True,'common_rows':rows.groupby('horizon').size().to_dict(),'holm_family':family,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code}})
        logging.info('Stage 3 M3 네 Horizon 완료. M4·Test 미실행.')
        print(scores.to_string(index=False));print(tests[tests.model.eq('M3')].to_string(index=False));print('공식 결과:',OUT)
    except Exception:
        logging.exception('오류 발생: 후속 작업 없이 중단합니다.');raise
if __name__=='__main__':main()
