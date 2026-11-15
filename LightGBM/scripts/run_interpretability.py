"""승인된 2023 1차 TreeSHAP/권역 묶음 permutation. 저장 모델만 사용, test/학습 금지."""
from pathlib import Path
import os,sys,json,csv,logging,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from src.common_scoring import sha,write_json
from src.data_loader import load_split
from scripts.run_m2 import protect
OUT=ROOT/'results/shap/validation_2023';DATA=ROOT/'Dataset/Origin_Features_2021_2024_v3'
MODELS=['M1','M2','M3','M4','M3T','M4T'];HORIZONS=[6,3,12,24]
def mapping(names,dictionary,registry):
 result=[]
 for i,name in enumerate(names):
  d=dictionary[name];params=json.loads(d['parameters_json']);key=params.get('station_key','');scope=params.get('scope','')
  row={'feature_index':i,**d,'station_key':key,'scope':scope,'station_country':'','station_region':'','station_latitude':None,'station_longitude':None,'distance_to_target_km':None}
  if key:
   station=registry.loc[key];country=station['country'];row.update(station_country=country,station_region=station.region_group,station_latitude=float(station.latitude),station_longitude=float(station.longitude))
   lon,lat,lo,la=np.radians([float(station.longitude),float(station.latitude),126.98,37.57]);a=np.sin((lat-la)/2)**2+np.cos(lat)*np.cos(la)*np.sin((lon-lo)/2)**2;row['distance_to_target_km']=float(6371*2*np.arcsin(np.sqrt(a)))
   primary=country;detail=f'{country}:{station.region_group}'
  elif name.startswith('seoul_'):primary='local';detail='local:Seoul'
  elif name.startswith('asos'):primary='local';detail='local:ASOS'
  elif name.startswith('cal_'):primary='local';detail='local:calendar'
  elif name.startswith(('pm25_','pm10_','geo_')):
   primary='trajectory';detail='trajectory:'+scope
  elif name.startswith('met_'):primary='trajectory';detail='trajectory:weather'
  elif name.startswith('route_'):primary='trajectory';detail='trajectory:geometry'
  else:raise ValueError('분류되지 않은 입력 '+name)
  if d['role']!='feature':raise ValueError('비입력 열 혼입')
  row.update(primary_group=primary,detail_group=detail,mixed_country_pool='KR67/CN61/JP188/MN1' if scope=='all' else '',shap_unit='log1p_PM2.5')
  result.append(row)
 return pd.DataFrame(result)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'manifest.json').exists():raise FileExistsError('완료된 분석 보존')
 logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',mode='a',encoding='utf-8'),logging.StreamHandler()])
 try:
  source_manifest=ROOT/'results/validation_2023/M4T/manifest.json';source=json.loads(source_manifest.read_text());score=ROOT/source['common_scores'];rows=pd.read_parquet(score/'score_rows_validation.parquet');metrics=pd.read_csv(score/'score_metrics_validation.csv')
  assert rows.origin_time_kst.dt.year.eq(2023).all() and rows.target_time_kst.dt.year.eq(2023).all()
  assert rows.groupby('horizon').size().to_dict()=={3:2509,6:2504,12:2497,24:2486}
  if (ROOT/'results/final_test_2024.lock').exists():raise ValueError('시험 잠금 상태 변경')
  files=[p for folder in ['Models','results','configs','src'] for p in (ROOT/folder).rglob('*') if p.is_file() and '/results/shap/' not in str(p) and '.matplotlib_cache' not in str(p) and '__pycache__' not in str(p)]
  snapshot={str(p.relative_to(ROOT)):sha(p) for p in files}
  audit=json.loads((ROOT/'EDA/Model_Data_Validation_20261108/validation_results_final.json').read_text());source_hashes={}
  for rel,digest in audit['input_sha256'].items():
   if rel=='Models/score.py':assert sha(ROOT/'Models/score_legacy_20261108.py')==digest
   else:assert sha(ROOT/rel)==digest,rel
   source_hashes[rel]=sha(ROOT/rel)
  ensembles={}
  for p in (ROOT/'Models').rglob('ensemble_manifest.json'):
   d=json.loads(p.read_text());key=(d['model'],d['horizon']);assert key not in ensembles;ensembles[key]=(p.parent,d)
  assert len(ensembles)==24
  with (DATA/'feature_dictionary.csv').open(encoding='utf-8-sig') as f:dictionary={r['name']:r for r in csv.DictReader(f)}
  registry=pd.read_csv(ROOT/'Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv',dtype=str).set_index('station_key');assert registry.index.is_unique
  policy={'models':MODELS,'horizons':[3,6,12,24],'primary_horizon':6,'seeds':[0,1,2],'split':'validation','training':False,'test_enabled':False,'adaptive_enabled':False,'tree_shap':{'model_output':'raw','feature_perturbation':'tree_path_dependent','approximate':False,'aggregation':'mean signed feature SHAP across seeds; explains mean log output, not final original-unit ensemble','feature_importance':'mean(abs(mean_seed_SHAP))','group_importance':'mean(abs(sum_group(mean_seed_SHAP)))','additivity_atol':1e-6},'permutation':{'repeats':30,'seed':0,'same_permutation_for_all_group_columns_and_all_seeds':True,'shared_permutations_across_models_and_groups_per_horizon':True,'metric':'RMSE(max(0,mean(expm1(seed raw))), y); permuted minus baseline','uncertainty':'repetition SD, not statistical CI','levels':['primary_group','detail_group']},'source_stage_manifest':str(source_manifest.relative_to(ROOT)),'source_score_rows':str((score/'score_rows_validation.parquet').relative_to(ROOT)),'source_score_sha256':sha(score/'score_rows_validation.parquet'),'protected_sha256':snapshot,'original_inputs_sha256':source_hashes,'versions':{k:importlib.metadata.version(k) for k in ['shap','lightgbm','numpy','pandas','pyarrow']}}
  if not (OUT/'analysis_config.json').exists():write_json(OUT/'analysis_config.json',policy)
  else:
   previous=json.loads((OUT/'analysis_config.json').read_text());assert previous==policy,'재시도 범위 불일치'
  (OUT/'evaluation').mkdir(exist_ok=True);(OUT/'mapping').mkdir(exist_ok=True);(OUT/'permutations').mkdir(exist_ok=True)
  allmaps=[]
  for model in MODELS:
   names=ensembles[(model,6)][1]['feature_names'];m=mapping(names,dictionary,registry);m.insert(0,'model',model);m.to_csv(OUT/f'mapping/{model}_feature_station_region.csv',index=False);allmaps.append(m)
   for h in HORIZONS:assert ensembles[(model,h)][1]['feature_names']==names
  pd.concat(allmaps,ignore_index=True).to_parquet(OUT/'mapping/feature_station_region.parquet',index=False)
  shap_features=[];shap_groups=[];perms=[];checks=[]
  for h in HORIZONS:
   ev=rows[rows.horizon.eq(h)].sort_values('origin_time_kst').reset_index(drop=True);ev.insert(0,'row_index',np.arange(len(ev)));ev.to_parquet(OUT/f'evaluation/h{h}.parquet',index=False)
   rng=np.random.default_rng(0);orders=np.stack([rng.permutation(len(ev)) for _ in range(30)]);np.save(OUT/f'permutations/h{h}_row_indices.npy',orders,allow_pickle=False)
   for model in MODELS:
    folder,manifest=ensembles[(model,h)];names=manifest['feature_names'];cfg=json.loads((ROOT/f'configs/{model.lower()}_{h}h.json').read_text());loaded=load_split(DATA,cfg['feature_set'],h,'validation',len(names));ix=pd.Index(loaded.meta.origin_time_kst).get_indexer(ev.origin_time_kst);assert (ix>=0).all() and loaded.eligible[ix].all();assert np.array_equal(loaded.meta.iloc[ix][f'y_pm25_h{h}'].to_numpy(),ev.y.to_numpy())
    X=loaded.X.iloc[ix].reset_index(drop=True);assert list(X)==names
    m=allmaps[MODELS.index(model)];dest=OUT/model/f'h{h}';dest.mkdir(parents=True,exist_ok=False);X.to_parquet(dest/'feature_values.parquet',index=False);write_json(dest/'feature_order.json',names)
    boosters=[];seed_phi=[];seed_raw=[];base_values=[];maxdiff=0
    logging.info('%s h%s: TreeSHAP 시작 (%s행 x %s열)',model,h,len(X),len(names))
    for seed in [0,1,2]:
     model_file=folder/f'{model}_h{h}_seed{seed}.txt';booster=lgb.Booster(params={'num_threads':2},model_file=str(model_file));assert booster.feature_name()==names
     raw=booster.predict(X,raw_score=True,num_threads=2);explainer=shap.TreeExplainer(booster,model_output='raw',feature_perturbation='tree_path_dependent');phi=explainer.shap_values(X,approximate=False,check_additivity=True);base=float(np.asarray(explainer.expected_value).reshape(-1)[0]);assert phi.shape==X.shape and np.isfinite(phi).all();err=float(np.max(abs(phi.sum(axis=1)+base-raw)));assert err<1e-6;maxdiff=max(maxdiff,err)
     np.savez_compressed(dest/f'shap_seed{seed}.npz',values=phi,base_value=base,raw_prediction=raw,row_index=np.arange(len(X)),feature_names=np.array(names))
     seed_phi.append(phi);seed_raw.append(raw);base_values.append(base);boosters.append(booster);logging.info('%s h%s seed%s SHAP 완료; 가산성 오차 %.3g',model,h,seed,err)
    mean_phi=np.mean(seed_phi,axis=0);mean_raw=np.mean(seed_raw,axis=0);mean_base=float(np.mean(base_values));assert np.max(abs(mean_phi.sum(axis=1)+mean_base-mean_raw))<1e-6
    original=np.maximum(0,np.expm1(np.stack(seed_raw)).mean(axis=0));difference=float(np.max(abs(original-ev[model].to_numpy())));assert difference<1e-8
    y=ev.y.to_numpy();baseline=float(np.sqrt(np.mean((original-y)**2)));official=float(metrics[(metrics.model==model)&(metrics.horizon==h)].iloc[0].RMSE);assert abs(baseline-official)<1e-8
    np.savez_compressed(dest/'shap_mean_log_output.npz',values=mean_phi,base_value=mean_base,mean_log_prediction=mean_raw,ensemble_y_hat=original,row_index=np.arange(len(X)),feature_names=np.array(names))
    individual=np.stack([np.abs(p).mean(axis=0) for p in seed_phi]);importance=np.abs(mean_phi).mean(axis=0)
    for j,name in enumerate(names):shap_features.append({'model':model,'horizon':h,'feature':name,'mean_abs_signed_seed_average_shap':float(importance[j]),'seed_mean_abs_shap_mean':float(individual[:,j].mean()),'seed_mean_abs_shap_sd':float(individual[:,j].std(ddof=1))})
    array=X.to_numpy();group_rows=[]
    for level in ['primary_group','detail_group']:
     for group in sorted(m[level].unique()):
      cols=np.flatnonzero(m[level].to_numpy()==group);assert len(cols)>0
      seed_group=[float(abs(p[:,cols].sum(axis=1)).mean()) for p in seed_phi]
      shap_groups.append({'model':model,'horizon':h,'level':level,'group':group,'n_features':len(cols),'n':len(X),'mean_abs_group_shap':float(abs(mean_phi[:,cols].sum(axis=1)).mean()),'sum_feature_mean_abs_shap':float(importance[cols].sum()),'seed_group_importance_mean':float(np.mean(seed_group)),'seed_group_importance_sd':float(np.std(seed_group,ddof=1))})
      for repeat,order in enumerate(orders):
       xp=array.copy();xp[:,cols]=array[order][:,cols];pred=np.maximum(0,np.mean([np.expm1(b.predict(xp,num_threads=2)) for b in boosters],axis=0));rmse=float(np.sqrt(np.mean((pred-y)**2)));rec={'model':model,'horizon':h,'level':level,'group':group,'n_features':len(cols),'n':len(X),'repeat':repeat,'baseline_rmse':baseline,'permuted_rmse':rmse,'delta_rmse':rmse-baseline};group_rows.append(rec)
      logging.info('%s h%s %s %s: permutation 30회 완료',model,h,level,group)
    perms.extend(group_rows);pd.DataFrame(group_rows).to_parquet(dest/'permutation_repeats.parquet',index=False)
    check={'model':model,'horizon':h,'n':len(X),'n_features':len(names),'shap_additivity_max_error':maxdiff,'saved_ensemble_prediction_max_difference':difference,'baseline_rmse':baseline,'negative_original_predictions':int((np.expm1(np.stack(seed_raw)).mean(axis=0)<0).sum()),'model_sha256':{str((folder/f'{model}_h{h}_seed{s}.txt').relative_to(ROOT)):sha(folder/f'{model}_h{h}_seed{s}.txt') for s in [0,1,2]}};checks.append(check);write_json(dest/'meta.json',{**check,'evaluation_rows':f'evaluation/h{h}.parquet','mapping':f'mapping/{model}_feature_station_region.csv','raw_shap':'shap_seed{0,1,2}.npz; float64, row_index aligns evaluation, columns align feature_order','aggregate_shap_unit':'log1p PM2.5; mean of log outputs, not final original-unit ensemble','excluded_features':cfg.get('excluded_features',[])})
    pd.DataFrame(shap_features).to_csv(OUT/'feature_shap_importance.csv',index=False);pd.DataFrame(shap_groups).to_csv(OUT/'group_shap_importance.csv',index=False);pd.DataFrame(perms).to_parquet(OUT/'permutation_repeats.parquet',index=False)
    protect(snapshot);logging.info('%s h%s 완료·기존 파일 보존 통과',model,h)
  df=pd.DataFrame(perms);summary=df.groupby(['model','horizon','level','group','n_features','n'],as_index=False).agg(baseline_rmse=('baseline_rmse','first'),delta_rmse_mean=('delta_rmse','mean'),delta_rmse_sd=('delta_rmse','std'),delta_rmse_min=('delta_rmse','min'),delta_rmse_max=('delta_rmse','max'),repeats=('repeat','size'));summary.to_csv(OUT/'group_permutation_importance.csv',index=False)
  protect(snapshot);protect(source_hashes)
  assert not (ROOT/'results/final_test_2024.lock').exists()
  write_json(OUT/'verification.json',{'status':'passed','models':24,'seed_models':72,'checks':checks,'previous_files_unchanged':len(snapshot),'original_inputs_unchanged':len(source_hashes),'test_executed':False,'adaptive_executed':False,'refit_performed':False})
  write_json(OUT/'manifest.json',{'status':'analysis_completed','models':MODELS,'horizons':[3,6,12,24],'primary_horizon':6,'source_stage_manifest':str(source_manifest.relative_to(ROOT)),'source_stage_sha256':sha(source_manifest),'analysis_config':'analysis_config.json','code_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},'feature_mapping':'mapping/feature_station_region.parquet','evaluation_rows':{str(h):f'evaluation/h{h}.parquet' for h in HORIZONS},'raw_shap':'{model}/h{h}/shap_seed{seed}.npz','feature_values':'{model}/h{h}/feature_values.parquet','test_2024_executed':False,'adaptive_executed':False,'previous_files_preserved':len(snapshot)})
  logging.info('全24개 앙상블 분석 완료. 후속 분석·Adaptive·Test 없이 중단.')
 except Exception:
  logging.exception('분석 중 오류: 후속 작업 중단');raise
if __name__=='__main__':main()
