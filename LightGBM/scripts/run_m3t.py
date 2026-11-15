"""승인된 경로 정보 묶음 검증 후 M3T만 학습. 공통 함수 재사용."""
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
OUT=ROOT/'results/validation_2023/M3T'
MODELS=ROOT/'Models/M3T'
PREVIOUS=['M1','M2','M3','M4']
PAIRS=['M2:M1','M3:M2','M3:M1','M4:M3','M4:M2','M4:M1','M3T:M4','M3T:M3','M3T:M2','M3T:M1']

def preflight():
    if (OUT/'preflight.json').exists() or MODELS.exists():raise FileExistsError('기존 M3T 실행 기록 발견: 중단')
    files=[p for p in (ROOT/'results').rglob('*') if p.is_file() and 'validation_2023/M3T' not in str(p) and '.matplotlib_cache' not in str(p)]
    for p in (ROOT/'Models').iterdir():
        if p.is_dir() and (p.name.startswith('M1') or p.name in ['M2','M3','M4']):files.extend(f for f in p.rglob('*') if f.is_file())
    files += [ROOT/'Models/predictions/M1.parquet',ROOT/'Models/predictions/M1.meta.json']
    files += [p for p in (ROOT/'configs').glob('*.json') if not p.name.startswith('m3t_')]
    snapshot={str(f.relative_to(ROOT)):sha(f) for f in files};protect(snapshot)
    names=feature_names(DATA,'M3T',2424);base=feature_names(DATA,'local_plus_station_history_kr_cn',2118)
    if not set(base)<=set(names):raise ValueError('M3 전체 포함 실패')
    additions=[c for c in names if c not in set(base)]
    groups={scope:[c for c in additions if f'_{scope}_' in c] for scope in ['all','kr','cn']}
    weather=[c for c in additions if c.startswith('met_')];geometry=[c for c in additions if c.startswith('route_')]
    if len(additions)!=306 or [len(groups[k]) for k in ['all','kr','cn']]!=[64,64,64] or len(weather)!=84 or len(geometry)!=30:raise ValueError('승인된 경로 Feature 구성 불일치')
    registry=pd.read_csv(ROOT/'Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv')
    countries=registry.groupby('country').size().to_dict()
    if countries!={'KR':67,'CN':61,'JP':188,'MN':1}:raise ValueError('경로 all 관측소 범위 불일치')
    # 2021–23, 도착 높이 500m, 과거 1–48h만 읽는다. 2024 관측·정답은 열지 않는다.
    counts={};cutoff=pd.Timestamp('2024-01-01',tz='Asia/Seoul');start=pd.Timestamp('2021-01-01',tz='Asia/Seoul')
    import pyarrow.dataset as ds
    filters=(ds.field('arrival_time_kst')>=start)&(ds.field('arrival_time_kst')<cutoff)&(ds.field('arrival_height_agl_m')==500)&(ds.field('lookback_hours')>=1)&(ds.field('lookback_hours')<=48)
    for filename in ['trajectory_point_coverage','trajectory_station_neighbors']:
        cols=['arrival_time_kst','pass_time_kst','lookback_hours']+(['observation_time_kst'] if filename.endswith('neighbors') else [])
        count=0
        dataset=ds.dataset(ROOT/f'Dataset/Trajectory_Matching_2021_2024_v3/{filename}.parquet',format='parquet')
        for batch in dataset.scanner(columns=cols,filter=filters,batch_size=65536,use_threads=False).to_batches():
            frame=batch.to_pandas();delta=frame.arrival_time_kst-frame.pass_time_kst
            if frame[cols].isna().any().any() or not delta.eq(pd.to_timedelta(frame.lookback_hours,unit='h')).all() or not frame.pass_time_kst.lt(frame.arrival_time_kst).all():raise ValueError('경로 미래 시점·시간 정렬 오류')
            if 'observation_time_kst' in cols and not frame.observation_time_kst.eq(frame.pass_time_kst).all():raise ValueError('경로 농도 시각 오류')
            count+=len(frame)
        counts[filename]=count
    if counts!={'trajectory_point_coverage':420480,'trajectory_station_neighbors':2377159}:raise ValueError('기존 시간 가용성 검증 건수 불일치')
    quality=[];sizes=[]
    for h in HORIZONS:
        cfg=json.loads((ROOT/f'configs/m3t_{h}h.json').read_text());oldcfg=json.loads((ROOT/f'configs/m3_{h}h.json').read_text())
        ignore={'model','feature_set','expected_features','feature_names','excluded_features','trajectory_policy'}
        if {k:v for k,v in cfg.items() if k not in ignore}!={k:v for k,v in oldcfg.items() if k not in ignore} or cfg['feature_names']!=names or cfg['excluded_features']!=[]:raise ValueError('학습 설정·입력 목록 불일치')
        size={}
        for split in ['train','validation']:
            d=load_split(DATA,'M3T',h,split,2424)
            if int(d.eligible.sum())!=cfg['expected_train_rows' if split=='train' else 'expected_validation_rows']:raise ValueError('적격 행 수 오류')
            if not d.meta.origin_time_kst.dt.year.isin([2021,2022] if split=='train' else [2023]).all():raise ValueError('학습/검증 기간 오류')
            if d.meta.loc[d.meta[f'target_time_h{h}'].dt.year.eq(2024),f'y_pm25_h{h}'].notna().any():raise ValueError('2024 정답 노출')
            missing=d.X.loc[d.eligible,additions].isna()
            all_missing=missing.all();quality.append({'horizon':h,'split':split,'added_features':306,'missing_cell_pct':float(100*missing.to_numpy().mean()),'all_missing_columns':all_missing[all_missing].index.tolist()})
            if all_missing.any():raise ValueError('승인 범위 밖 전체 결측 경로 입력 발견')
            if split=='validation':
                old=pd.read_parquet(ROOT/f'results/validation_2023/M3/h{h}/predictions_M3_h{h}_validation.parquet')
                if not np.array_equal(d.eligible,old.eligible) or not np.allclose(d.meta.loc[d.eligible,f'y_pm25_h{h}'],old.loc[old.eligible,'y']):raise ValueError('M3와 평가 행·정답 불일치')
            size[split]=int(d.eligible.sum())
        sizes.append({'horizon':h,**size})
    result={'status':'passed','n_features':2424,'M3_features_preserved':2118,'added_features':306,'trajectory_components':{'all':64,'kr':64,'cn':64,'weather':84,'geometry':30},'all_station_pool':countries,'feature_names':names,'excluded_features':[],'trajectory_time_checks':counts,'feature_missingness':quality,'imputation_performed':False,'rows':sizes,'protected_sha256':snapshot,'scorer_sha256':sha(ROOT/'src/common_scoring.py'),'test_2024_read':False,'interpretation':cfg['trajectory_policy']['interpretation'],'availability_limit':cfg['trajectory_policy']['publication_latency']}
    write_json(OUT/'preflight.json',result);return result

DATA=ROOT/'Dataset/Origin_Features_2021_2024_v3'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',encoding='utf-8',mode='x'),logging.StreamHandler()])
    try:
        check=preflight();logging.info('M3T 검증 통과: M3 전체 2118 + 경로 306 = 2424열. all 64개 유지, 결측 대체 없음.')
        comparisons=[(m,'persistence') for m in ['M1','M2','M3','M4','M3T']]+[tuple(x.split(':')) for x in PAIRS]
        family=[f'{h}:{a}:{b}' for h in HORIZONS for a,b in comparisons]
        write_json(OUT/'scope_before_training.json',{'train_model':'M3T','reuse_models':PREVIOUS,'horizons':HORIZONS,'train_years':[2021,2022],'validation_year':2023,'test_enabled':False,'holm_family':family,'holm_family_size':60,'improvement_rule':'RMSE 차이 95% CI 상한<0 AND Holm p<0.05'})
        selected=[]
        for h in HORIZONS:
            config=json.loads((ROOT/f'configs/m3t_{h}h.json').read_text());out=OUT/f'h{h}';models=MODELS/f'h{h}';out.mkdir(exist_ok=False);models.mkdir(parents=True,exist_ok=False)
            handler=logging.FileHandler(out/'run.log',encoding='utf-8');handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'));logging.getLogger().addHandler(handler)
            try:
                logging.info('M3T %s시간 시작. 2024 Test 비활성.',h);info=train_lightgbm(ROOT,config,out,models)
            finally:logging.getLogger().removeHandler(handler);handler.close()
            selected.append({'horizon':h,'train_rows':info['eligible_train_rows'],'validation_rows':info['eligible_validation_rows'],'selected':info['selected'],'ensemble_validation':info['metrics']})
            protect(check['protected_sha256'])
        pred=OUT/'predictions';pred.mkdir(exist_ok=False)
        for model in PREVIOUS:
            source=ROOT/f'results/validation_2023/M4/predictions/{model}.parquet'
            shutil.copyfile(source,pred/source.name);shutil.copyfile(source.with_suffix('.meta.json'),pred/source.with_suffix('.meta.json').name)
        frames=[];metadata={}
        for h in HORIZONS:
            source=OUT/f'h{h}/M3T.parquet';frames.append(pd.read_parquet(source));metadata[str(h)]=json.loads(source.with_suffix('.meta.json').read_text())
        export_predictions(pd.concat(frames,ignore_index=True),pred/'M3T.parquet',{'seeds':[0,1,2],'horizons':HORIZONS,'n_features':2424,'training_years':[2021,2022],'selection_year':2023,'no_refit_on_2023':True,'validation_only':True,'by_horizon':metadata})
        score_dir,scores,tests=evaluate('validation',['M1','M2','M3','M4','M3T'],HORIZONS,pairs=PAIRS,prediction_dir=pred,output_root=OUT/'common_scores')
        rows=pd.read_parquet(score_dir/'score_rows_validation.parquet')
        if sha(ROOT/'src/common_scoring.py')!=check['scorer_sha256'] or len(tests)!=60 or not np.allclose(tests.p_holm,holm(tests.p)):raise ValueError('채점기·Holm 집합 오류')
        if not np.array_equal(tests.improved,(tests.ci95_high<0)&(tests.p_holm<.05)):raise ValueError('공식 판정 오류')
        if not rows.origin_time_kst.dt.year.eq(2023).all() or not rows.target_time_kst.dt.year.eq(2023).all():raise ValueError('2024 채점 발생')
        old_manifest=json.loads((ROOT/'results/validation_2023/M4/manifest.json').read_text());old_rows=pd.read_parquet(ROOT/old_manifest['common_scores']/'score_rows_validation.parquet')
        for h in HORIZONS:
            details=pd.read_parquet(OUT/f'h{h}/predictions_M3T_h{h}_validation.parquet');info=json.loads((OUT/f'h{h}/metrics_and_selection.json').read_text())
            if not np.allclose(details.y_hat_raw,details[[f'y_hat_seed{s}' for s in [0,1,2]]].mean(axis=1)) or any(r['max_absolute_prediction_difference']!=0 for r in info['model_reload_checks']):raise ValueError('앙상블·재로드 오류')
            g=rows[rows.horizon.eq(h)].sort_values('origin_time_kst');old=old_rows[old_rows.horizon.eq(h)].sort_values('origin_time_kst')
            if g.origin_time_kst.tolist()!=old.origin_time_kst.tolist() or not np.array_equal(g[['y','persistence',*PREVIOUS]].to_numpy(),old[['y','persistence',*PREVIOUS]].to_numpy()):raise ValueError('기존 공통 평가 행·예측 변경')
            for model in ['persistence',*PREVIOUS,'M3T']:
                r=scores[(scores.horizon==h)&(scores.model==model)].iloc[0];e=g[model]-g.y
                if not np.allclose([r.RMSE,r.MAE,r.MBE],[np.sqrt(np.mean(e**2)),np.mean(abs(e)),e.mean()]):raise ValueError('지표 재계산 오류')
        scores.to_csv(OUT/'M1_M2_M3_M4_M3T_Persistence_2023_validation_metrics.csv',index=False);tests.to_csv(OUT/'M1_M2_M3_M4_M3T_2023_validation_tests.csv',index=False)
        direct=tests[(tests.model=='M3T')&tests.versus.isin(PREVIOUS)].rename(columns={'rmse_diff':'delta_rmse','ci95_low':'delta_rmse_ci_lower','ci95_high':'delta_rmse_ci_upper','p_holm':'holm_adjusted_p'})
        direct[['horizon','model','versus','n','delta_rmse','delta_rmse_ci_lower','delta_rmse_ci_upper','dm_stat','p','holm_adjusted_p','improved']].to_csv(OUT/'M3T_vs_M1_M2_M3_M4_2023_validation_statistics.csv',index=False)
        shutil.copyfile(score_dir/'rmse_improvement_validation.png',OUT/'M1_M2_M3_M4_M3T_2023_validation_intermediate_rmse_improvement.png')
        write_json(OUT/'horizon_settings.json',selected);protect(check['protected_sha256'])
        code=[ROOT/'configs/feature_groups.json',ROOT/'src/train.py',ROOT/'src/feature_groups.py',ROOT/'src/data_loader.py',ROOT/'src/common_scoring.py',Path(__file__)]+[ROOT/f'configs/m3t_{h}h.json' for h in HORIZONS]
        write_json(OUT/'manifest.json',{'status':'completed','model':'M3T','horizons':HORIZONS,'official_result_path':str(OUT.relative_to(ROOT)),'model_root':str(MODELS.relative_to(ROOT)),'common_scores':str(score_dir.relative_to(ROOT)),'n_features':2424,'M3_subset':True,'trajectory_features':306,'all_features_retained':64,'excluded_features':[], 'interpretation':check['interpretation'],'imputation_performed':False,'train_years':[2021,2022],'validation_year':2023,'test_2024_executed':False,'M1_M2_M3_M4_retrained_or_repredicted':False,'M4T_SHAP_Adaptive_started':False,'new_models':12,'reload_prediction_max_difference':0,'previous_artifacts_preserved':len(check['protected_sha256']),'scorer_unchanged':True,'common_rows':rows.groupby('horizon').size().to_dict(),'holm_family':family,'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code}})
        logging.info('M3T 네 Horizon 완료. 후속 모델·Test 미실행.')
        print(scores.to_string(index=False));print(tests[tests.model.eq('M3T')].to_string(index=False));print('공식 결과:',OUT)
    except Exception:
        logging.exception('오류 발생: 후속 작업 없이 중단합니다.');raise
if __name__=='__main__':main()
