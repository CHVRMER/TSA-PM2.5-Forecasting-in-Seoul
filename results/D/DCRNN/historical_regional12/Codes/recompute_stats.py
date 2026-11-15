"""Read-only review re-scoring; incomplete original family remains explicit."""
from pathlib import Path
import sys
import io
import zipfile
import json
import hashlib
import numpy as np
import pandas as pd

PROJECT = Path(r'C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
ARCHIVE = Path(r'C:/Users/kimjy/Downloads/Regional12_Seed11_20261109-20261110T090404Z-1-001.zip')
OUT = Path(__file__).resolve().parents[1]/'Results/Tables'
OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(PROJECT/'Model/Common'))
import score

def safe_json(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value))

grid_path=PROJECT/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
grid=pd.read_parquet(grid_path)
audit={'source_archive':str(ARCHIVE),'grid':str(grid_path),
       'scorer':str(Path(score.__file__).resolve()),
       'scorer_sha256':hashlib.sha256(Path(score.__file__).read_bytes()).hexdigest(),
       'review_only_not_original_completed_12_condition_family':True,
       'sources':{},'comparisons':{}}
with zipfile.ZipFile(ARCHIVE) as z:
    names=z.namelist()
    for split in ['validation','test']:
        sources={}
        for n in names:
            if n.endswith('/Exports/'+split+'_predictions_by_seed.parquet'):
                cond=n.split('/Jobs/')[1].split('/')[0]
                sources.setdefault(cond,[]).append(n)
        frames=[]
        selected={}
        for cond,options in sorted(sources.items()):
            assert len(options)==1, options
            n=options[0]
            raw=z.read(n)
            frame=pd.read_parquet(io.BytesIO(raw))
            assert not frame.duplicated(['origin_time_kst','horizon']).any()
            assert np.isfinite(frame.y_hat).all()
            assert frame.input_condition.eq(cond).all()
            assert frame.seed.astype(str).eq('11').all()
            dt=(frame.target_time_kst-frame.origin_time_kst)/pd.Timedelta(hours=1)
            assert (dt==frame.horizon).all()
            selected[cond]={'path':n,'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(frame)}
            frames.append(frame)
        combined=pd.concat(frames,ignore_index=True)
        ids={c:f'DCRNN_trajectory_directed|{c}|seed|seed=11' for c in selected}
        pairs=[(ids[c],ids['DN']) for c in selected if c!='DN']
        result=score.score_frames(grid,combined,split=split,pairs=pairs,
                                  final=split=='test',bootstrap_reps=1000,
                                  bootstrap_seed=0,make_ensembles=False)
        metrics=result['metrics'];tests=result['tests']
        # score_frames' observed 88-test Holm is NOT a completed planned family.
        tests=tests.rename(columns={'p_holm':'p_holm_observed88','improved':'improved_observed88'})
        regional=tests.loc[tests.versus_run_id.eq(ids['DN'])].copy()
        assert len(regional)==40
        regional['p_holm_planned44']=score.holm(np.r_[regional.p.to_numpy(float),np.ones(4)])[:len(regional)]
        regional['improved_planned44']=(regional.ci95_high<0)&(regional.p_holm_planned44<0.05)
        # Full scorer family: 13 vs-persistence runs (12 conditions+seasonal),
        # plus11 regional contrasts, all4h =>96. MN's8 missing tests get p=1.
        assert len(tests)==88
        tests['p_holm_planned96']=score.holm(np.r_[tests.p.to_numpy(float),np.ones(8)])[:len(tests)]
        tests['improved_planned96']=(tests.ci95_high<0)&(tests.p_holm_planned96<0.05)
        metrics.to_csv(OUT/f'{split}_metrics.csv',index=False)
        tests.to_csv(OUT/f'{split}_tests_review.csv',index=False)
        regional.to_csv(OUT/f'{split}_regional_tests_planned44.csv',index=False)
        result['conditions'].to_csv(OUT/f'{split}_conditional_metrics.csv',index=False)
        dn=metrics.loc[metrics.input_condition.eq('DN')].set_index('horizon').RMSE
        pivot=metrics.loc[metrics.prediction_kind.eq('seed')].pivot(index='input_condition',columns='horizon',values='RMSE')
        gain=(1-pivot/dn)*100
        gain.to_csv(OUT/f'{split}_rmse_gain_pct_vs_DN.csv')
        pivot.to_csv(OUT/f'{split}_rmse_pivot.csv')
        audit['sources'][split]=selected
        audit['comparisons'][split]={**result['audit'],
            'observed_score_family_count':len(tests),'regional_declared_family':44,
            'regional_missing_MN_tests':4,'joint_common_declared_family':96,
            'joint_missing_MN_tests':8,'missing_p_placeholder':1.0,
            'regional_improvements_planned44':int(regional.improved_planned44.sum()),
            'all_improvements_planned96':int(tests.improved_planned96.sum()),
            'negative_clipped_total':int(metrics.negative_predictions_clipped.sum())}
        print(split,'RMSE')
        print(pivot.round(5).to_string())
        print('gain_pct_vs_DN')
        print(gain.round(3).to_string())
        cols=['horizon','run_id','rmse_diff','p','p_holm_planned44','ci95_low','ci95_high','improved_planned44']
        print('REGIONAL_SIGNIFICANT')
        print(regional.loc[regional.improved_planned44,cols].to_string(index=False))
        print('BASELINES',metrics.loc[metrics.model.isin(['persistence','seasonal_naive']),['model','horizon','n','RMSE','MAE']].to_dict('records'))
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,default=safe_json)+'\n',encoding='utf-8')
