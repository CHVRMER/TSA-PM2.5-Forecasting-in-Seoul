"""Read-only, post-hoc cross-run audit. Never deserialize model weights.

The 12 contrasts per split are declared here before p-values are computed:
new DCRNN M1 vs previous GWN M1; new DCRNN M4 vs previous GWN M4;
new DCRNN M4 vs previous regional DCRNN F, at h=3,6,12,24 each.
They are exploratory and separate from the run's original 84 contrasts.
"""
from pathlib import Path
import hashlib
import io
import json
import sys
import zipfile
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
PROJECT = Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
SCORER = PROJECT/'Model/Common/DCRNN_H48_20261111'
sys.path.insert(0,str(SCORER))
import score

NEW = WORK/'extracted/20261112_003040_429483_85af4013'
OLD_GWN = WORK.parent/'gwn_results_review_20261111/selected_results'
OLD_D_ZIP = Path('C:/Users/kimjy/Downloads/Regional12_Seed11_20261109-20261110T090404Z-1-001.zip')
GRID = PROJECT/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
TEAM_AUDIT = WORK.parent/'dcrnn_regional12_review_20261111/team_sources/team_results_audit.json'
TABLES = WORK/'Tables'
TABLES.mkdir(exist_ok=True)
H = (3,6,12,24)
audit = {'method': 'Exploratory post-hoc cross-run comparisons, not a confirmatory 2024 test. Exact fixed eligible rows; no intersection dropping. Squared-loss DM with original max(h/3-1,0) calendar HAC; 1000 moving 7-day calendar block draws, seed0. Holm12 within each split; Holm24 sensitivity across both splits.',
 'declared_pairs':[['new_DCRNN_M1','old_GWN_M1'],['new_DCRNN_M4','old_GWN_M4'],['new_DCRNN_M4','old_DCRNN_F']],
 'scorer':str(Path(score.__file__).resolve()),'scorer_sha256':hashlib.sha256(Path(score.__file__).read_bytes()).hexdigest(),
 'grid':str(GRID),'grid_sha256':hashlib.sha256(GRID.read_bytes()).hexdigest(),'sources':{},'checks':{},
 'interpretation_limit': 'Cross-run performance comparisons change architecture/graph, context, training recipe and number of output horizons; they cannot isolate the cause of differences. Previous 2024 results have already been inspected.'}

grid = pd.read_parquet(GRID)
grid = grid.loc[grid.horizon.isin(H)].copy()

def read_unique(folder,name):
    paths=list(folder.rglob(name))
    if len(paths)!=1:raise ValueError((str(folder),name,len(paths)))
    path=paths[0]
    audit['sources'][str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)

def filter_model(frame,condition):
    seed=pd.to_numeric(frame.seed,errors='coerce')
    return frame.loc[frame.input_condition.eq(condition)&frame.prediction_kind.eq('seed')&seed.eq(11)&frame.horizon.isin(H)].copy()

def check_rows(frame,split,label):
    keys=['origin_time_kst','horizon']
    frame=frame.loc[frame.eligible].set_index(keys).sort_index()
    expected=grid.loc[grid.split.eq(split)&grid.eligible].set_index(keys).sort_index()
    if not frame.index.is_unique or not frame.index.equals(expected.index):raise ValueError((label,split,'fixed row mismatch',len(frame),len(expected)))
    if not np.array_equal(frame.y_true.to_numpy(),expected.y_true.to_numpy()):raise ValueError((label,'truth mismatch'))
    if not frame.target_time_kst.equals(expected.target_time_kst):raise ValueError((label,'target time mismatch'))
    if not np.isfinite(frame.y_hat.to_numpy()).all():raise ValueError((label,'nonfinite prediction'))
    audit['checks'][label+'_'+split]={'fixed_rows_exact':True,'truth_exact':True,'target_times_exact':True,'n_by_horizon':frame.groupby(level='horizon').size().to_dict()}
    return frame

def compare(candidate,reference,split,label):
    c=check_rows(candidate,split,label+'_candidate');r=check_rows(reference,split,label+'_reference')
    records=[]
    for h in H:
        ch=c.xs(h,level='horizon');rh=r.xs(h,level='horizon')
        e1=ch.y_hat.to_numpy()-ch.y_true.to_numpy();e2=rh.y_hat.to_numpy()-rh.y_true.to_numpy()
        full=grid.loc[grid.split.eq(split)&grid.horizon.eq(h),'origin_time_kst']
        a,b=score.calendar_errors(ch.index,e1,e2,full)
        rmse1=float(np.sqrt(np.mean(e1**2)));rmse2=float(np.sqrt(np.mean(e2**2)))
        records.append({'split':split,'comparison':label,'horizon':h,'n':len(ch),'candidate_RMSE':rmse1,'reference_RMSE':rmse2,'RMSE_diff':rmse1-rmse2,'RMSE_improvement_pct':100*(1-rmse1/rmse2),
          'candidate_MAE':float(np.mean(np.abs(e1))),'reference_MAE':float(np.mean(np.abs(e2))),
          **score.dm_calendar(a,b,h),**score.bootstrap_calendar(a,b,reps=1000,seed=0),
          'comparison_status':'posthoc_exploratory_configuration_confounded'})
    return pd.DataFrame(records)

all_results=[];metrics=[]
with zipfile.ZipFile(OLD_D_ZIP) as dz:
    for split in ('validation','test'):
        nf=read_unique(NEW/'Common_Scoring/Combined'/split,'score_rows.parquet')
        gf=read_unique(OLD_GWN/'Common_Scoring/Combined'/split,'score_rows.parquet')
        nm=read_unique(NEW/'Common_Scoring/Combined'/split,'metrics.csv')
        gm=read_unique(OLD_GWN/'Common_Scoring/Combined'/split,'metrics.csv')
        for generation,model,df in [('new_H48','DCRNN_H48',nm),('previous','GraphWaveNet',gm)]:
            df=df.loc[df.input_condition.isin(['M1','M2','M3','M4','M3T','M4T'])].copy()
            df['generation']=generation;df['model_family']=model;metrics.append(df)
        dn=next(n for n in dz.namelist() if '/Jobs/F/' in n and n.endswith('/Exports/'+split+'_predictions_by_seed.parquet'))
        raw=dz.read(dn);df=pd.read_parquet(io.BytesIO(raw));df['y_hat']=df.y_hat.clip(lower=0)
        audit['sources']['old_DCRNN_F_'+split]={'archive':str(OLD_D_ZIP),'member':dn,'sha256':hashlib.sha256(raw).hexdigest()}
        tables=[compare(filter_model(nf,'M1'),filter_model(gf,'M1'),split,'new_DCRNN_M1_vs_old_GWN_M1'),
                compare(filter_model(nf,'M4'),filter_model(gf,'M4'),split,'new_DCRNN_M4_vs_old_GWN_M4'),
                compare(filter_model(nf,'M4'),df.loc[df.horizon.isin(H)],split,'new_DCRNN_M4_vs_old_DCRNN_F')]
        joined=pd.concat(tables,ignore_index=True)
        assert len(joined)==12
        joined['p_holm12']=score.holm(joined.p)
        joined['candidate_better_holm12']=(joined.ci95_high<0)&(joined.p_holm12<.05)
        joined['candidate_worse_holm12']=(joined.ci95_low>0)&(joined.p_holm12<.05)
        joined.to_csv(TABLES/f'team_comparison_{split}_paired_posthoc_holm12.csv',index=False,encoding='utf-8-sig')
        all_results.append(joined)

paired=pd.concat(all_results,ignore_index=True)
paired['p_holm24_sensitivity']=score.holm(paired.p)
paired['candidate_better_holm24']=(paired.ci95_high<0)&(paired.p_holm24_sensitivity<.05)
paired['candidate_worse_holm24']=(paired.ci95_low>0)&(paired.p_holm24_sensitivity<.05)
paired.to_csv(TABLES/'team_comparison_all_paired_posthoc_holm24.csv',index=False,encoding='utf-8-sig')
pd.concat(metrics,ignore_index=True).to_csv(TABLES/'team_comparison_new_and_previous_GWN_all_metrics.csv',index=False,encoding='utf-8-sig')

team=json.loads(TEAM_AUDIT.read_text(encoding='utf-8'))
descriptive=[]
def add(model,condition,values,scope,aggregation,caveat):
    for h,value in zip(H,values):
        descriptive.append({'model':model,'input_condition':condition,'split':'validation','horizon':h,'RMSE':value,'scope':scope,'aggregation':aggregation,'comparison_caveat':caveat})
for model,values in team['A_validation_RMSE'].items():
    add('TeamA_'+model,'WNF' if model=='SARIMAX_WNF' else '',values,'KR67/W24; CN61 regional means','deterministic statistical fit','Rounded published table; raw rows unavailable; different input scope; no paired inference')
for row in team['ABC_docx_validation_table'][1:]:
    if row[0] in ['LightGBM','C_GRU','C_GCNGRU'] and row[1] in ['DN','F']:
        add('Team_'+row[0],row[1],[float(x) for x in row[2:6]],'KR514 DN; F600 full','seed11/22/33 ensemble','Rounded published table; F600 differs from F634; single seed vs ensemble; raw rows unavailable')
for model,subfolder in [('new_DCRNN_H48',NEW),('previous_GWN',OLD_GWN)]:
    frame=read_unique(subfolder/'Common_Scoring/Combined/validation','metrics.csv')
    for condition in ('M1','M4'):
        selected=frame.loc[frame.input_condition.eq(condition)].set_index('horizon')
        add(model,condition,[float(selected.loc[h,'RMSE']) for h in H],'KR514 M1; F634 M4','seed11','Same frozen eligible grid for paired GWN comparison; context/training/model configurations differ')
pd.DataFrame(descriptive).to_csv(TABLES/'team_comparison_validation_descriptive.csv',index=False,encoding='utf-8-sig')
(WORK/'comparison_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
print(paired[['split','comparison','horizon','candidate_RMSE','reference_RMSE','RMSE_improvement_pct','ci95_low','ci95_high','p_holm12','candidate_better_holm12','candidate_worse_holm12']].round(6).to_string(index=False))
print('\nVALIDATION DESCRIPTIVE\n'+pd.DataFrame(descriptive).pivot(index=['model','input_condition'],columns='horizon',values='RMSE').round(4).to_string())
