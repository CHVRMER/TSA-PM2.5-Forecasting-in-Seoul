from pathlib import Path
import io
import json
import hashlib
import sys
import zipfile
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
PROJECT=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
sys.path.insert(0,str(PROJECT/'Model/Common'))
import score

GZIP=Path('C:/Users/kimjy/Downloads/VastAI-20261111T003043Z-1-001.zip')
DZIP=Path('C:/Users/kimjy/Downloads/Regional12_Seed11_20261109-20261110T090404Z-1-001.zip')
BATCH='VastAI/20261111_014538_406520_eaa72758/'
GRID=PROJECT/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
LGB=PROJECT/'Model/LightGBM/Neighbor_Expansion_QC_20261108/Results/Validation_Reference_20261108/Common_Score/score_rows.parquet'
grid=pd.read_parquet(GRID)
audit={'GWN_archive':str(GZIP),'DCRNN_archive':str(DZIP),'GWN_batch':BATCH,
       'grid_path':str(GRID),'grid_sha256':hashlib.sha256(GRID.read_bytes()).hexdigest(),
       'method':'Post-hoc exploratory paired review; same complete eligible grid; no intersection dropping; squared-loss calendar HAC h/3-1; 7-day calendar bootstrap, 1000 draws seed0. Holm4 per M4-vs-F, Holm8 sensitivity including domestic comparison; not original planned inferential family.',
       'scorer_path':str(Path(score.__file__).resolve()),'sources':{},'comparisons':{}}

def paired(candidate,reference,split,cid):
    rows=[]
    keys=['origin_time_kst','horizon']
    c=candidate.set_index(keys).sort_index()
    r=reference.set_index(keys).sort_index()
    g=grid.loc[grid.split.eq(split)&grid.eligible].set_index(keys).sort_index()
    assert c.index.is_unique and r.index.is_unique
    assert c.index.equals(g.index), (cid,'candidate grid mismatch',len(c),len(g))
    assert r.index.equals(g.index), (cid,'reference grid mismatch',len(r),len(g))
    assert np.array_equal(c.y_true.to_numpy(),g.y_true.to_numpy())
    assert np.array_equal(r.y_true.to_numpy(),g.y_true.to_numpy())
    assert c.target_time_kst.equals(g.target_time_kst)
    assert r.target_time_kst.equals(g.target_time_kst)
    for h in (3,6,12,24):
        ch=c.xs(h,level='horizon');rh=r.xs(h,level='horizon')
        e1=ch.y_hat.to_numpy()-ch.y_true.to_numpy()
        e2=rh.y_hat.to_numpy()-rh.y_true.to_numpy()
        full=grid.loc[grid.split.eq(split)&grid.horizon.eq(h),'origin_time_kst']
        a,b=score.calendar_errors(ch.index,e1,e2,full)
        rmsec=float(np.sqrt(np.mean(e1**2)));rmser=float(np.sqrt(np.mean(e2**2)))
        row={'split':split,'comparison':cid,'horizon':h,'n':len(ch),'candidate_RMSE':rmsec,'reference_RMSE':rmser,
             'rmse_diff':rmsec-rmser,'improvement_pct':100*(1-rmsec/rmser),
             'candidate_MAE':float(np.mean(np.abs(e1))),'reference_MAE':float(np.mean(np.abs(e2))),
             **score.dm_calendar(a,b,h),**score.bootstrap_calendar(a,b,reps=1000,seed=0)}
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame['p_holm4']=score.holm(frame.p)
    frame['candidate_better_holm4']=(frame.ci95_high<0)&(frame.p_holm4<.05)
    frame['candidate_worse_holm4']=(frame.ci95_low>0)&(frame.p_holm4<.05)
    audit['comparisons'][cid+'_'+split]={'rows_match':True,'targets_match':True,'n_by_horizon':frame.set_index('horizon').n.to_dict(),'primary_review_family':4}
    return frame

all_tables=[]
gmetrics={}
with zipfile.ZipFile(GZIP) as gz,zipfile.ZipFile(DZIP) as dz:
    for split in ('validation','test'):
        prefix=BATCH+'Common_Scoring/Combined/'+split+'/'
        n=next(n for n in gz.namelist() if n.startswith(prefix) and n.endswith('/score_rows.parquet'))
        raw=gz.read(n);gf=pd.read_parquet(io.BytesIO(raw))
        audit['sources']['GWN_'+split]={'member':n,'sha256':hashlib.sha256(raw).hexdigest()}
        mn=next(n for n in gz.namelist() if n.startswith(prefix) and n.endswith('/metrics.csv'))
        gm=pd.read_csv(gz.open(mn));gmetrics[split]=gm
        gm.to_csv(OUT/(split+'_GWN_metrics.csv'),index=False)
        ts=[]
        for gc,dc in [('M4','F'),('M1','DN')]:
            gn=gf.loc[gf.input_condition.eq(gc)&gf.prediction_kind.eq('seed')&gf.seed.astype(str).eq('11')].copy()
            dn=next(n for n in dz.namelist() if '/Jobs/'+dc+'/' in n and n.endswith('/Exports/'+split+'_predictions_by_seed.parquet'))
            draw=dz.read(dn);df=pd.read_parquet(io.BytesIO(draw));df=df.loc[df.eligible].copy()
            df['y_hat']=df.y_hat.clip(lower=0)
            audit['sources']['DCRNN_'+dc+'_'+split]={'member':dn,'sha256':hashlib.sha256(draw).hexdigest()}
            t=paired(gn,df,split,'GWN_'+gc+'_vs_DCRNN_'+dc)
            ts.append(t)
        table=pd.concat(ts,ignore_index=True)
        table['p_holm8_sensitivity']=score.holm(table.p)
        table['candidate_better_holm8']=(table.ci95_high<0)&(table.p_holm8_sensitivity<.05)
        table['candidate_worse_holm8']=(table.ci95_low>0)&(table.p_holm8_sensitivity<.05)
        table.to_csv(OUT/(split+'_GWN_vs_DCRNN_paired_review.csv'),index=False)
        all_tables.append(table)
        if split=='validation':
            lf=pd.read_parquet(LGB)
            lf=lf.loc[lf.input_condition.eq('F634_BOTH_W24')&lf.prediction_kind.eq('seed')&lf.seed.astype(str).eq('11')].copy()
            gn=gf.loc[gf.input_condition.eq('M4')&gf.prediction_kind.eq('seed')&gf.seed.astype(str).eq('11')].copy()
            lt=paired(gn,lf,split,'GWN_M4_vs_local_LightGBM_F634_seed11')
            lt.to_csv(OUT/'validation_GWN_M4_vs_local_LightGBM_F634_seed11_paired_review.csv',index=False)
            all_tables.append(lt)

team=json.loads((OUT.parents[1]/'dcrnn_regional12_review_20261111/team_sources/team_results_audit.json').read_text(encoding='utf-8'))
table=[]
def add_row(model,condition,values,split,scope,aggregation,caveat):
    for h,v in zip((3,6,12,24),values):
        table.append(dict(model=model,input_condition=condition,split=split,horizon=h,RMSE=v,scope=scope,aggregation=aggregation,comparison_caveat=caveat))
for name,values in team['A_validation_RMSE'].items():
    add_row('A_'+name,'WNF' if name=='SARIMAX_WNF' else '',values,'validation','KR67/W24, CN61 regional means','single deterministic statistical fit','Published rounded table; raw predictions unavailable; cannot paired-test exact same rows')
for row in team['ABC_docx_validation_table'][1:]:
    if row[0] in ['LightGBM','C_GRU','C_GCNGRU'] and row[1] in ['DN','F']:
        add_row(row[0],row[1],[float(x) for x in row[2:6]],'validation','KR514 DN; F600 for full','seed11/22/33 ensemble','Published rounded table; F600 differs from current F634; raw predictions unavailable')
for split,key in [('validation','B_pasted_validation_RMSE'),('test','B_pasted_test_RMSE')]:
    for condition,values in team[key].items():
        add_row('B_LightGBM_pasted',condition,values,split,'KR67, CN61, JP24, MN1 maximum; older source','seed11/22/33 ensemble','Different eligible rows; KR67 vs KR514; M3T includes all-country trajectory aggregates. Descriptive numbers only, not same-contract paired comparison')
for split,gm in gmetrics.items():
    for c in ['M1','M2','M3','M4','M3T','M4T']:
        m=gm.loc[gm.input_condition.eq(c)].set_index('horizon')
        add_row('GraphWaveNet',c,[float(m.loc[h,'RMSE']) for h in (3,6,12,24)],split,'fixed635 nodes; external PM arm '+c,'seed11','current archive fixed evaluation grid')
    dm=pd.read_csv(OUT.parents[1]/'dcrnn_regional12_review_20261111/independent_stats'/f'{split}_metrics.csv')
    for c in ['DN','F']:
        m=dm.loc[dm.input_condition.eq(c)].set_index('horizon')
        add_row('DCRNN',c,[float(m.loc[h,'RMSE']) for h in (3,6,12,24)],split,'KR514 DN; F634 full','seed11','same fixed evaluation grid; model architecture, graph, training settings differ')
pd.DataFrame(table).to_csv(OUT/'team_comparison_long.csv',index=False,encoding='utf-8-sig')
all_review=pd.concat(all_tables,ignore_index=True)
assert len(all_review)==20
all_review['p_holm20_all_review']=score.holm(all_review.p)
all_review['candidate_better_holm20']=(all_review.ci95_high<0)&(all_review.p_holm20_all_review<.05)
all_review.to_csv(OUT/'all_cross_model_tests_holm20_sensitivity.csv',index=False)
audit['all_review_sensitivity_family']=20
audit['GWN_M4_vs_DCRNN_F_test12_holm20']=float(all_review.loc[all_review.comparison.eq('GWN_M4_vs_DCRNN_F')&all_review.split.eq('test')&all_review.horizon.eq(12),'p_holm20_all_review'].iloc[0])
(OUT/'comparison_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
for t in all_tables:
    print(t[['split','comparison','horizon','candidate_RMSE','reference_RMSE','improvement_pct','ci95_low','ci95_high','p_holm4','candidate_better_holm4','candidate_worse_holm4']].round(6).to_string(index=False))
