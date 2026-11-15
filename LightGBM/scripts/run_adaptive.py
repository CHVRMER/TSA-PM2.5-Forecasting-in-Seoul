"""Stage8: approved frozen M1/M3 blends; 2023 in-sample reference only."""
from pathlib import Path
import os,sys,json,logging,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from src.adaptive import condition_frame,fit_horizon,apply_horizon,CONDITIONS
from src.common_scoring import sha,write_json,export_predictions,metrics
from scripts.run_m2 import protect
OUT=ROOT/'results/validation_2023/Adaptive';DATA=ROOT/'Dataset/Origin_Features_2021_2024_v3';HS=[3,6,12,24];NAMES={'wind':'Adaptive_wind','route':'Adaptive_route'}
def records(d):return d.astype(object).where(pd.notna(d),None).to_dict('records')
def preflight():
    if any((ROOT/f'Models/{name}').exists() for name in NAMES.values()):raise FileExistsError('기존 Adaptive 가중치 보존')
    source_manifest=ROOT/'results/validation_2023/M4T/manifest.json';manifest=json.loads(source_manifest.read_text());source=ROOT/manifest['common_scores'];rows=pd.read_parquet(source/'score_rows_validation.parquet').sort_values(['horizon','origin_time_kst']).reset_index(drop=True)
    assert rows.groupby('horizon').size().to_dict()=={3:2509,6:2504,12:2497,24:2486};assert rows.origin_time_kst.dt.year.eq(2023).all() and rows.target_time_kst.dt.year.eq(2023).all()
    assert not (ROOT/'results/final_test_2024.lock').exists()
    files=[p for folder in ['Models','results','src','configs'] for p in (ROOT/folder).rglob('*') if p.is_file() and '/validation_2023/Adaptive/' not in str(p) and '.matplotlib_cache' not in str(p) and '__pycache__' not in str(p)]
    snapshot={str(p.relative_to(ROOT)):sha(p) for p in files}
    original=json.loads((ROOT/'results/shap/validation_2023/analysis_config.json').read_text())['original_inputs_sha256'];protect(original)
    predictors=[];metadata={}
    for model in ['M1','M3']:
        p=ROOT/f'results/validation_2023/M4T/predictions/{model}.parquet';meta=json.loads(p.with_suffix('.meta.json').read_text());assert meta['prediction_sha256']==sha(p) and meta['no_refit_on_2023'] is True and meta['seeds']==[0,1,2]
        d=pd.read_parquet(p);assert len(d)==11680 and d.split.eq('validation').all() and d.origin_time_kst.dt.year.eq(2023).all() and not d.duplicated(['origin_time_kst','horizon']).any();assert np.isfinite(d.y_hat).all();predictors.append(d[['origin_time_kst','horizon','y_hat']].rename(columns={'y_hat':model+'_raw'}));metadata[model]={'prediction':str(p.relative_to(ROOT)),'sha256':sha(p),'meta_sha256':sha(p.with_suffix('.meta.json'))}
    full=predictors[0].merge(predictors[1],on=['origin_time_kst','horizon'],validate='one_to_one')
    columns=['origin_time_kst','route_history12h_sector','asos108_wind_u_now','asos108_wind_v_now','asos108_wind_calm_now']
    context=pq.read_table(DATA/'origin_features_2021_2024_kst.parquet',columns=columns,filters=[('origin_split','=','validation')],use_threads=False).to_pandas();assert len(context)==2920 and context.origin_time_kst.is_unique and context.origin_time_kst.dt.year.eq(2023).all()
    full=condition_frame(full.merge(context,on='origin_time_kst',validate='many_to_one')).sort_values(['horizon','origin_time_kst']).reset_index(drop=True)
    compare=rows[['origin_time_kst','horizon','M1_raw','M3_raw','season','wind_sector','route_group']].merge(full[['origin_time_kst','horizon','M1_raw','M3_raw','season','wind_sector','route_group']],on=['origin_time_kst','horizon'],suffixes=('_old','_new'),validate='one_to_one')
    for col in ['M1_raw','M3_raw','season','wind_sector','route_group']:assert compare[col+'_old'].fillna('unknown').tolist()==compare[col+'_new'].fillna('unknown').tolist(),col
    # Confirm all fit rows still satisfy the original horizon eligibility, without reading any new target column.
    eligible=pq.read_table(DATA/'origin_features_2021_2024_kst.parquet',columns=['origin_time_kst',*[f'eligible_h{h}' for h in HS]],filters=[('origin_split','=','validation')],use_threads=False).to_pandas().set_index('origin_time_kst')
    for h,g in rows.groupby('horizon'):assert eligible.loc[g.origin_time_kst,f'eligible_h{h}'].all()
    config={'stage':8,'models':list(NAMES.values()),'base_models':['M1','M3'],'base_model_retraining':False,'horizons':HS,'primary_horizon':6,'fit_year':2023,'evaluation_type':'in_sample_reference','formal_improvement_assessed':False,'test_enabled':False,'formula':'(1-w)*M1+w*M3; original units, raw base predictions, clip mixture only for scoring','seasons':{'cold':[11,12,1,2,3],'warm':[4,5,6,7,8,9,10]},'conditions':CONDITIONS,'minimum_cell_rows':100,'small_cell_rule':'n_cell<100: season weight; denominator exception is applied to the chosen estimation pool','unknown_condition_rule':'season weight, no extra learned condition','threshold_approval':{'status':'approved','user_reply':'제안 기준 승인','definition':'epsilon_h=max(1e-6, median_2023_common(abs(M3-M1))*1e-4); denominator<=n_estimation_pool*epsilon_h**2 -> w=0.5','calibration_uses_y':False,'epsilon_unit':'ug/m3','denominator_unit':'(ug/m3)^2','zero_denominator_weight':.5,'historical_pending_record':'EDA/Model_Data_Validation_20261108/modeling_decisions_20261108.json; superseded only by this Stage8 approved config'},'fit_rows_source':str((source/'score_rows_validation.parquet').relative_to(ROOT)),'fit_rows_sha256':sha(source/'score_rows_validation.parquet'),'source_stage_manifest_sha256':sha(source_manifest),'predictions':metadata,'test_lock_unconsumed':True}
    config['epsilon_by_horizon']={str(h):max(1e-6,float(np.median(abs(g.M3_raw-g.M1_raw)))*1e-4) for h,g in rows.groupby('horizon')}
    proposal=[]
    for h,g in rows.groupby('horizon'):
        delta=(g.M3_raw-g.M1_raw).to_numpy();record={'horizon':h,'n':len(g),'epsilon':config['epsilon_by_horizon'][str(h)],'mean_squared_difference':float(np.mean(delta**2))}
        for q in [0,.01,.05,.25,.5,.75,.95,.99,1]:record[f'abs_delta_quantile_{q:g}']=float(np.quantile(abs(delta),q));record[f'signed_delta_quantile_{q:g}']=float(np.quantile(delta,q))
        proposal.append(record)
    pd.DataFrame(proposal).to_csv(OUT/'prediction_difference_distribution.csv',index=False);write_json(OUT/'config.json',config)
    config_path=ROOT/'configs/adaptive_validation_2023.json';write_json(config_path,config)
    write_json(OUT/'preflight.json',{'status':'passed','protected_sha256':snapshot,'original_inputs_sha256':original,'common_rows':rows.groupby('horizon').size().to_dict(),'all_validation_forecasts':len(full),'context_alignment_matches_common_scorer':True,'eligibility_verified':True,'test_read':False})
    return rows,full,config,snapshot,original

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.FileHandler(OUT/'run.log',mode='x',encoding='utf-8'),logging.StreamHandler()])
    try:
        rows,full,config,snapshot,original=preflight();logging.info('검증 통과. 저장된 M1/M3만 사용; 2023 공통행에서 가중치 추정; Test 없음.')
        seasons_all=[];cells_all=[];details_all=[];pool_records=[];artifacts={}
        pred=OUT/'predictions';pred.mkdir()
        for h in HS:
            fit=rows[rows.horizon.eq(h)].copy();eps=config['epsilon_by_horizon'][str(h)];seasons,cells=fit_horizon(fit,h,eps);seasons_all.append(seasons);cells_all.append(cells)
            season_pools=seasons.copy();season_pools['kind']='season';season_pools['condition']='season';pool_records.append(season_pools)
            pool_records.append(cells[~cells.season_fallback_used].drop(columns=['cell_numerator','cell_denominator']).copy())
            for kind,name in NAMES.items():
                base=full[full.horizon.eq(h)].copy();applied=apply_horizon(base,seasons,cells,kind);d=pd.concat([base,applied],axis=1);d['model']=name;d['split']='validation';d['horizon']=h
                assert d.weight.between(0,1).all() and np.isfinite(d.y_hat_raw).all();assert np.allclose(d.y_hat_raw,(1-d.weight)*d.M1_raw+d.weight*d.M3_raw,rtol=0,atol=1e-12)
                if d.near_zero_exception_used.any():assert d.loc[d.near_zero_exception_used,'weight'].eq(.5).all()
                # A stored estimation-pool result is the only source of a fallback weight.
                for _,g in d[d.season_fallback_used].groupby('season'):assert np.allclose(g.weight,seasons.set_index('season').loc[g.season.iloc[0],'final_weight'])
                details_all.append(d)
                path=ROOT/f'Models/{name}/h{h}';path.mkdir(parents=True,exist_ok=False)
                write_json(path/'weights.json',{'model':name,'horizon':h,'fitted_year':2023,'fit_type':'least_squares_in_original_units','season_rules':records(seasons),'cell_rules':records(cells[cells.kind.eq(kind)]),'unknown_condition':'season fallback','source_predictions':config['predictions'],'epsilon':eps,'near_zero_rule':config['threshold_approval']})
                write_json(path/'config.json',config);artifacts[str(path.relative_to(ROOT))+'/weights.json']=sha(path/'weights.json')
            logging.info('h%s 가중치 추정·2023 전체 Validation 예측 완료: epsilon=%.10g',h,eps)
        seasonal=pd.concat(seasons_all,ignore_index=True);celltable=pd.concat(cells_all,ignore_index=True);details=pd.concat(details_all,ignore_index=True);pools=pd.concat(pool_records,ignore_index=True).drop_duplicates('estimation_pool')
        seasonal.to_csv(OUT/'season_weights.csv',index=False);celltable.to_csv(OUT/'condition_weights.csv',index=False);pools.to_csv(OUT/'estimation_pools.csv',index=False);details.to_parquet(OUT/'prediction_details_validation.parquet',index=False)
        for kind,name in NAMES.items():
            d=details[details.model.eq(name)].copy();d['y_hat']=d.y_hat_raw
            export_predictions(d,pred/f'{name}.parquet',{'seeds':[0,1,2],'horizons':HS,'weights_fit_year':2023,'base_models_retrained':False,'evaluation_type':'in_sample_reference','formal_improvement_assessed':False,'settings':config,'weight_artifacts_sha256':{k:v for k,v in artifacts.items() if f'Models/{name}/' in k},'validation_only':True})
        # Keep the original common rows and join new predictions without changing old forecasts.
        scored=rows.copy();assignment=[]
        for name in NAMES.values():
            sub=details[details.model.eq(name)].copy();x=sub[['origin_time_kst','horizon','y_hat_raw']].rename(columns={'y_hat_raw':name+'_raw'});scored=scored.merge(x,on=['origin_time_kst','horizon'],validate='one_to_one');scored[name]=np.maximum(0,scored[name+'_raw'])
            selected=sub.merge(rows[['origin_time_kst','horizon']],on=['origin_time_kst','horizon'],validate='one_to_one');selected['evaluation_scope']='common_rows';assignment.append(selected)
        common_details=pd.concat(assignment,ignore_index=True)
        metric_rows=[];condition_metrics=[]
        allmodels=['M1','M2','M3','M4','M3T','M4T',*NAMES.values()]
        for h,g in scored.groupby('horizon'):
            for model in ['persistence',*allmodels]:
                raw=g[model+'_raw'];metric_rows.append({'horizon':int(h),'model':model,**metrics(g.y,g[model],g.persistence),'negative_predictions_clipped':int((raw<0).sum()),'evaluation_type':'in_sample_reference' if model.startswith('Adaptive_') else 'existing_validation_reference','formal_improvement_assessed':False})
                for condition in ['season','route_group','wind_sector','high_pm']:
                    for level,gg in g.groupby(condition,dropna=False):condition_metrics.append({'horizon':int(h),'model':model,'condition':condition,'level':str(level),**metrics(gg.y,gg[model],gg.persistence),'negative_predictions_clipped':int((gg[model+'_raw']<0).sum()),'evaluation_type':'in_sample_reference' if model.startswith('Adaptive_') else 'existing_validation_reference'})
        metricframe=pd.DataFrame(metric_rows);metricframe.to_csv(OUT/'score_metrics_validation_reference.csv',index=False);pd.DataFrame(condition_metrics).to_csv(OUT/'score_conditions_validation_reference.csv',index=False)
        fallback=[]
        for scope,d in [('all_validation_origins',details),('common_evaluation_rows',common_details)]:
            for (name,h),g in d.groupby(['model','horizon']):fallback.append({'scope':scope,'model':name,'horizon':int(h),'n_predictions':len(g),'small_cell_fallback_rows':int(g.fallback_reason.eq('cell_n_less_than_100').sum()),'unknown_condition_fallback_rows':int(g.fallback_reason.eq('unknown_condition').sum()),'total_season_fallback_rows':int(g.season_fallback_used.sum()),'near_zero_exception_prediction_rows':int(g.near_zero_exception_used.sum()),'clipped_weight_prediction_rows':int(g.weight_clipped.sum()),'negative_predictions_clipped':int((g.y_hat_raw<0).sum())})
        pd.DataFrame(fallback).to_csv(OUT/'fallback_exception_counts.csv',index=False)
        pools.groupby('horizon').agg(unique_estimation_pools=('estimation_pool','nunique'),near_zero_exception_pools=('near_zero_exception_used','sum')).reset_index().to_csv(OUT/'exception_pool_counts.csv',index=False)
        scored.to_parquet(OUT/'score_rows_validation_reference.parquet',index=False);write_json(OUT/'score_rows_validation_reference.meta.json',{'features_version':json.loads((DATA/'feature_manifest.json').read_text())['version'],'seeds':[0,1,2],'settings':config,'adaptive_weights_fitted_on_these_rows':True,'evaluation_type':'in_sample_reference','formal_improvement_assessed':False,'DM_Holm_bootstrap_improvement_not_computed':'2023 labels reused for fitting; no final inferential improvement claim','source_common_rows_sha256':config['fit_rows_sha256']})
        # Check old columns/metrics and closed-form optimality on each actual selected pool.
        assert len(scored)==len(rows)
        aligned=scored.set_index(['horizon','origin_time_kst']).sort_index();old=rows.set_index(['horizon','origin_time_kst']).sort_index();pd.testing.assert_frame_equal(aligned[old.columns],old)
        for pool in pools.to_dict('records'):
            frame=rows[(rows.horizon==pool['horizon'])&rows.season.eq(pool['season'])]
            if pool['kind']!='season':frame=frame[frame['wind_sector' if pool['kind']=='wind' else 'route_group'].eq(pool['condition'])]
            assert len(frame)==pool['n_estimation_pool']
            if not pool['near_zero_exception_used']:
                delta=frame.M3_raw-frame.M1_raw;y=frame.y;a=frame.M1_raw;w=pool['final_weight'];loss=float(np.sum((y-a-w*delta)**2));assert loss<=min(float(np.sum((y-a)**2)),float(np.sum((y-frame.M3_raw)**2)))+1e-8
        assert (OUT/'score_metrics_validation_reference.csv').exists() and not (ROOT/'results/final_test_2024.lock').exists();protect(snapshot);protect(original)
        write_json(OUT/'verification.json',{'status':'passed','no_base_model_refit':True,'source_predictions_and_conditions_match':True,'same_common_rows_and_old_predictions':True,'actual_pool_SSE_optimality_checked':True,'weight_bounds_and_affine_formula_checked':True,'season_fallback_checked':True,'approved_epsilon_rule_checked':True,'unique_estimation_pools':len(pools),'near_zero_exception_pools':int(pools.near_zero_exception_used.sum()),'near_zero_exception_prediction_rows':int(details.near_zero_exception_used.sum()),'existing_files_preserved':len(snapshot),'original_inputs_preserved':len(original),'test_executed':False,'in_sample_reference_only':True})
        write_json(OUT/'manifest.json',{'status':'completed','stage':8,'models':list(NAMES.values()),'horizons':HS,'common_scores':str(OUT.relative_to(ROOT)),'evaluation_type':'in_sample_reference','formal_improvement_assessed':False,'test_2024_executed':False,'base_model_refit':False,'weight_artifacts_sha256':artifacts,'source_stage_manifest':'results/validation_2023/M4T/manifest.json','source_shap_manifest':'results/shap/validation_2023/manifest.json','config':'configs/adaptive_validation_2023.json','config_sha256':sha(ROOT/'configs/adaptive_validation_2023.json'),'source_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/adaptive.py',Path(__file__),ROOT/'src/common_scoring.py',ROOT/'scripts/check_adaptive.py']},'common_rows':rows.groupby('horizon').size().to_dict(),'existing_files_preserved':len(snapshot),'weights_frozen_for_later_separately_authorized_test':True})
        logging.info('Stage8 완료. 표본 내 참고 평가이며 최종 개선 판정·Test 없음.');print(metricframe[metricframe.model.str.startswith('Adaptive_')].to_string(index=False))
    except Exception:logging.exception('오류 발생: 후속 작업 중단');raise
if __name__=='__main__':main()
