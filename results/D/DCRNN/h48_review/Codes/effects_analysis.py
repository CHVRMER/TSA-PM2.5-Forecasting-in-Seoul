from pathlib import Path
import json
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent
BATCH=WORK/'extracted/20261112_003040_429483_85af4013'
OUT=WORK/'Tables';OUT.mkdir(exist_ok=True)
ORDER=['M1','M2','M3','M4','M3T','M4T']
H=[3,6,12,24,36,48]

def condition_from_run(value):
    bits=str(value).split('|')
    return bits[1] if len(bits)>1 else value

def safe_records(df):
    return json.loads(df.to_json(orient='records'))

def main():
    report={'analysis':'original combined six-horizon outputs, no Holm family changes','splits':{}}
    for split in ('validation','test'):
        folder=sorted((BATCH/'Common_Scoring/Combined'/split).glob('*'))[-1]
        m=pd.read_csv(folder/'metrics.csv');t=pd.read_csv(folder/'tests.csv');c=pd.read_csv(folder/'conditions.csv')
        audit=json.loads((folder/'scoring_audit.json').read_text(encoding='utf-8'))
        own=m.loc[m.prediction_kind.eq('seed')].copy()
        own['input_condition']=pd.Categorical(own.input_condition,ORDER,ordered=True)
        own=own.sort_values(['input_condition','horizon'])
        own.to_csv(OUT/f'{split}_condition_metrics.csv',index=False,encoding='utf-8-sig')
        own.pivot(index='input_condition',columns='horizon',values='RMSE').reindex(ORDER).to_csv(OUT/f'{split}_rmse_matrix.csv',encoding='utf-8-sig')
        own.pivot(index='input_condition',columns='horizon',values='MAE').reindex(ORDER).to_csv(OUT/f'{split}_mae_matrix.csv',encoding='utf-8-sig')
        t['candidate_condition']=t.run_id.map(condition_from_run)
        t['reference_condition']=t.versus_run_id.map(condition_from_run)
        ct=t.loc[t.reference_condition.isin(ORDER)].copy()
        assert len(ct)==42 and len(t)==84
        assert np.all(ct.improved.eq((ct.ci95_high<0)&(ct.p_holm<.05)))
        ct['significant_worsening']=(ct.ci95_low>0)&(ct.p_holm<.05)
        refs=own.set_index(['input_condition','horizon']).RMSE
        ct['reference_RMSE']=[refs.loc[(r,h)] for r,h in zip(ct.reference_condition,ct.horizon)]
        ct['improvement_pct']=-100*ct.rmse_diff/ct.reference_RMSE
        ct['statistical_outcome']=np.select([ct.improved,ct.significant_worsening],['improved','worsened'],default='uncertain')
        ct.to_csv(OUT/f'{split}_predeclared_condition_comparisons.csv',index=False,encoding='utf-8-sig')
        summary=ct.groupby(['candidate_condition','reference_condition']).agg(
            improved=('improved','sum'),worsened=('significant_worsening','sum'),comparisons=('horizon','size')).reset_index()
        summary.to_csv(OUT/f'{split}_comparison_summary.csv',index=False,encoding='utf-8-sig')
        base=m.loc[m.prediction_kind.eq('single')].copy();base.to_csv(OUT/f'{split}_baseline_metrics.csv',index=False,encoding='utf-8-sig')
        bp=t.loc[t.versus_run_id.eq('persistence|common_baseline|single')].copy()
        model_vs_p=bp.loc[bp.candidate_condition.isin(ORDER)]
        count_check=own.groupby('horizon',observed=True).agg(n_unique=('n','nunique'),n=('n','first'))
        assert count_check.n_unique.eq(1).all()
        descriptive=[]
        for candidate,reference in [('M3','M1'),('M4','M1'),('M3T','M3'),('M4T','M4')]:
            aa=c.loc[c.input_condition.eq(candidate)&c.prediction_kind.eq('seed')]
            bb=c.loc[c.input_condition.eq(reference)&c.prediction_kind.eq('seed')]
            keys=['horizon','condition','level']
            joined=aa[keys+['n','RMSE','MAE','MBE']].merge(bb[keys+['n','RMSE','MAE','MBE']],on=keys,suffixes=('_candidate','_reference'),validate='one_to_one')
            assert joined.n_candidate.eq(joined.n_reference).all()
            joined['candidate']=candidate;joined['reference']=reference
            joined['rmse_diff']=joined.RMSE_candidate-joined.RMSE_reference
            joined['improvement_pct']=100*(1-joined.RMSE_candidate/joined.RMSE_reference)
            joined['adequate_descriptive_n']=joined.n_candidate>=100
            descriptive.append(joined)
        d=pd.concat(descriptive,ignore_index=True)
        d.to_csv(OUT/f'{split}_conditional_descriptive_differences.csv',index=False,encoding='utf-8-sig')
        filtered=d.loc[d.adequate_descriptive_n & d.horizon.isin([6,24,36,48]) & d.condition.isin(['season','route_group','wind_sector','traj_wind_sector','cold_season'])]
        winners=own.loc[own.groupby('horizon',observed=True).RMSE.idxmin(),['horizon','input_condition','RMSE','MAE']]
        mae_winners=own.loc[own.groupby('horizon',observed=True).MAE.idxmin(),['horizon','input_condition','RMSE','MAE']]
        scored=pd.read_parquet(folder/'score_rows.parquet',columns=['run_id','origin_time_kst','horizon','y_true','y_hat'])
        assert not scored.duplicated(['run_id','origin_time_kst','horizon']).any()
        assert np.isfinite(scored[['y_true','y_hat']].to_numpy(float)).all()
        metric_audit=[]
        for (run,h),part in scored.groupby(['run_id','horizon']):
            err=part.y_hat.to_numpy(float)-part.y_true.to_numpy(float)
            metric_audit.append({'run_id':run,'horizon':h,'RMSE_recomputed':float(np.sqrt(np.mean(err**2))),
                                 'MAE_recomputed':float(np.mean(abs(err))),'n_recomputed':len(err)})
        checked=m.merge(pd.DataFrame(metric_audit),on=['run_id','horizon'],validate='one_to_one')
        assert len(checked)==len(m) and checked.n.eq(checked.n_recomputed).all()
        max_metric_error=float(np.max(abs(checked[['RMSE','MAE']].to_numpy()-checked[['RMSE_recomputed','MAE_recomputed']].to_numpy())))
        assert max_metric_error<1e-8
        report['splits'][split]={
            'source_folder':str(folder),'all_tests':len(t),'finite_p':int(t.p.notna().sum()),'condition_tests':len(ct),
            'condition_significant_improvements':int(ct.improved.sum()),'condition_significant_worsenings':int(ct.significant_worsening.sum()),
            'significant_condition_rows':safe_records(ct.loc[ct.improved|ct.significant_worsening]),
            'model_vs_persistence_improvements':int(model_vs_p.improved.sum()),'model_vs_persistence_count':len(model_vs_p),
            'condition_summary':safe_records(summary),'rmse_winners_descriptive':safe_records(winners),
            'mae_winners_descriptive':safe_records(mae_winners),
            'score_rows_recomputed_metric_max_abs_error':max_metric_error,
            'score_rows_unique_run_origin_horizon':True,
            'score_rows_finite_truth_predictions':True,
            'fixed_n':{str(h):int(v) for h,v in count_check.n.items()},
            'negative_predictions_clipped_total':int(own.negative_predictions_clipped.sum()),
            'r2_range':[float(own.R2.min()),float(own.R2.max())],
            'mbe_range':[float(own.MBE.min()),float(own.MBE.max())],
            'largest_conditional_improvements_descriptive':safe_records(filtered.sort_values('improvement_pct',ascending=False).head(20)),
            'largest_conditional_worsenings_descriptive':safe_records(filtered.sort_values('improvement_pct').head(20)),
            'conditional_count_threshold':100,'conditional_tests_performed':False,
            'row_selection_using_prediction':audit['rows_selected_using_predictions'],
            'model_prediction_intersection_used':audit['model_prediction_intersection_used'],
        }
    validation=pd.read_csv(OUT/'validation_condition_metrics.csv')
    test=pd.read_csv(OUT/'test_condition_metrics.csv')
    relation=validation[['input_condition','horizon','RMSE']].merge(test[['input_condition','horizon','RMSE']],on=['input_condition','horizon'],suffixes=('_validation','_test'))
    relation.to_csv(OUT/'validation_test_metrics.csv',index=False,encoding='utf-8-sig')
    (WORK/'effects_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({split:{k:v for k,v in value.items() if k not in ('largest_conditional_improvements_descriptive','largest_conditional_worsenings_descriptive','significant_condition_rows','condition_summary')} for split,value in report['splits'].items()},indent=2,ensure_ascii=False))
    print('TEST RMSE')
    print(test.pivot(index='input_condition',columns='horizon',values='RMSE').reindex(ORDER).to_string(float_format=lambda v:f'{v:.5f}'))
    print('TEST contrasts')
    print(pd.read_csv(OUT/'test_predeclared_condition_comparisons.csv')[['candidate_condition','reference_condition','horizon','rmse_diff','improvement_pct','ci95_low','ci95_high','p_holm','statistical_outcome']].to_string(index=False,float_format=lambda v:f'{v:.6f}'))

if __name__=='__main__':main()
