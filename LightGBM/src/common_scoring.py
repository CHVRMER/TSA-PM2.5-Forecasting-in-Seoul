"""공통 평가. 분할 필터, 실제 3시간 격자 HAC, 달력 7일 블록, 시험 1회 잠금."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]
FEATURES=ROOT/'Dataset/Origin_Features_2021_2024_v3'
FILE=FEATURES/'origin_features_2021_2024_kst.parquet'
PRED=ROOT/'Models/predictions'
COLUMNS=['origin_time_kst','horizon','model','split','y_hat']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()

def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False,default=str);f.write('\n')

def export_predictions(frame,path,metadata):
    path=Path(path);sidecar=path.with_suffix('.meta.json')
    if path.exists() or sidecar.exists():raise FileExistsError('기존 공통 예측 보존')
    d=frame[COLUMNS].copy()
    if d.duplicated(['origin_time_kst','horizon']).any() or d['model'].nunique()!=1 or not np.isfinite(d.y_hat).all():raise ValueError('예측 키/값 오류')
    d['horizon']=d.horizon.astype('int8');d.to_parquet(path,index=False)
    v=json.loads((FEATURES/'feature_manifest.json').read_text())
    write_json(sidecar,{'model':d['model'].iloc[0],'features_version':v['version'],'features_created_kst':v['created_kst'],'features_rows':v['rows'],'features_columns':v['columns'],**metadata,'prediction_sha256':sha(path),'y_hat_policy':'원 단위 복원 후 음수 절단 전; 채점 때만 0 절단'})

def timeline(split):
    ranges={'validation':('2023-01-01','2024-01-01'),'test':('2024-01-01','2025-01-01')}
    a,b=ranges[split];return pd.date_range(a,b,freq='3h',tz='Asia/Seoul',inclusive='left')

def load_grid(split,h):
    cols=['origin_time_kst',f'y_pm25_h{h}',f'target_time_h{h}','seoul_pm25_now','route_history12h_sector','asos108_wind_u_now','asos108_wind_v_now','asos108_wind_calm_now']
    d=pq.read_table(FILE,columns=cols,filters=[('origin_split','=',split),(f'eligible_h{h}','=',True)],use_threads=False).to_pandas()
    d=d.rename(columns={f'y_pm25_h{h}':'y',f'target_time_h{h}':'target_time_kst'});d['horizon']=h
    d['season']=np.where(d.origin_time_kst.dt.month.isin([11,12,1,2,3]),'cold','warm')
    d['route_group']=d.route_history12h_sector.map({'NW':'west','W':'west','SW':'west','N':'north','NE':'north','E':'east','SE':'east','S':'east','LOCAL':'local'})
    u,v=d.asos108_wind_u_now,d.asos108_wind_v_now;a=(np.degrees(np.arctan2(-u,-v))+360)%360
    direction=np.select([(a>=225)&(a<315),(a>=315)|(a<45),(a>=45)&(a<135)],['west','north','east'],default='south')
    d['wind_sector']=pd.Series(direction,index=d.index).where(u.notna()&v.notna(),'unknown').where(d.asos108_wind_calm_now.ne(1),'calm');d['high_pm']=d.y.ge(35)
    if not d.target_time_kst.dt.year.eq(2023 if split=='validation' else 2024).all():raise ValueError('정답 분할 경계 오류')
    return d.set_index('origin_time_kst').sort_index()

def dm_test(e1,e2,h):
    valid=np.isfinite(e1)&np.isfinite(e2);d=e1[valid]**2-e2[valid]**2;n=len(d)
    if n<2:raise ValueError('DM 표본 부족')
    z=np.zeros(len(e1));z[valid]=d-d.mean();lag=h//3
    variance=(z@z+2*sum((1-k/(lag+1))*(z[k:]@z[:-k]) for k in range(1,lag+1)))/(n*n)
    if variance<=0:return (0.,1.) if np.allclose(d,0) else (None,None)
    stat=float(d.mean()/np.sqrt(variance));return stat,float(2*norm.sf(abs(stat)))

def bootstrap(e1,e2,reference,reps=1000,seed=0):
    rng=np.random.default_rng(seed);n=len(e1);block=56;diffs=[];skills=[];counts=[]
    if n<block:raise ValueError('7일 격자 부족')
    for _ in range(reps):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        idx=(starts[:,None]+np.arange(block)).ravel()[:n]
        a,b,r=e1[idx],e2[idx],reference[idx];valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(r)
        if not valid.any():continue
        ra,rb,rr=[float(np.sqrt(np.mean(x[valid]**2))) for x in (a,b,r)]
        if rr==0:continue
        diffs.append(ra-rb);skills.append(100*(1-ra/rr));counts.append(int(valid.sum()))
    if len(diffs)<.9*reps:raise ValueError('유효 bootstrap 반복 부족')
    return {'ci95_low':float(np.percentile(diffs,2.5)),'ci95_high':float(np.percentile(diffs,97.5)),'skill_ci95_low_pct':float(np.percentile(skills,2.5)),'skill_ci95_high_pct':float(np.percentile(skills,97.5)),'bootstrap_valid_reps':len(diffs),'bootstrap_sample_n_min':min(counts),'bootstrap_sample_n_max':max(counts)}

def holm(p):
    p=np.asarray(p,float);out=np.full(len(p),np.nan);finite=np.flatnonzero(np.isfinite(p));order=finite[np.argsort(p[finite])];running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(len(p)-rank)*p[i]));out[i]=running
    return out

def metrics(y,pred,ref):
    e=pred-y;rmse=float(np.sqrt(np.mean(e**2)));rr=float(np.sqrt(np.mean((ref-y)**2)))
    return {'n':len(y),'RMSE':rmse,'MAE':float(np.mean(np.abs(e))),'MBE':float(np.mean(e)),'skill_vs_persistence':1-rmse/rr if rr>0 else None,'rmse_improvement_pct':100*(1-rmse/rr) if rr>0 else None}

def authorize_test(split,final,frozen,models,horizons,lock,prediction_dir=None):
    if split!='test':return None
    if not final:raise PermissionError('2024 시험 잠금: --final 필요')
    if frozen is None:raise PermissionError('시험 전 동결 manifest 필요')
    spec=json.loads(Path(frozen).read_text())
    if sorted(spec['models'])!=sorted(models) or sorted(spec['horizons'])!=sorted(horizons):raise ValueError('동결된 비교 범위와 다름')
    if spec.get('no_refit_on_2023') is not True or not spec['artifacts_sha256']:raise ValueError('재학습 금지/동결 산출물 확인 필요')
    pred_root=Path(prediction_dir) if prediction_dir else PRED
    required={str((pred_root/f'{m}.parquet').relative_to(ROOT)) for m in models}
    if not required<=set(spec['artifacts_sha256']):raise ValueError('시험 예측 파일을 모두 동결해야 함')
    for rel,digest in spec['artifacts_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('동결한 산출물이 변경됨')
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as f:json.dump({'status':'started','frozen_sha256':sha(frozen),'models':models,'horizons':horizons},f)
    return sha(frozen)

def evaluate(split,models,horizons=(6,),pairs=(),final=False,frozen=None,output_root=None,reps=1000,seed=0,prediction_dir=None):
    models=list(dict.fromkeys(m for m in models if m!='persistence'))
    if not models or split not in ('validation','test') or any(h not in (3,6,12,24) for h in horizons):raise ValueError('비교 범위 오류')
    comparisons=list(dict.fromkeys([(m,'persistence') for m in models]+[tuple(x.split(':')) for x in pairs]))
    if any(a==b or a not in ['persistence',*models] or b not in ['persistence',*models] for a,b in comparisons):raise ValueError('비교 쌍 오류')
    pred_root=Path(prediction_dir) if prediction_dir else PRED
    frozen_digest=authorize_test(split,final,frozen,models,list(horizons),ROOT/'results/final_test_2024.lock',pred_root)
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f');out=(Path(output_root) if output_root else ROOT/'results/common_scores')/f'{split}_{stamp}';out.mkdir(parents=True,exist_ok=False)
    policy={'split':split,'models':models,'horizons':list(horizons),'holm_family':[f'{h}:{a}:{b}' for h in horizons for a,b in comparisons],'improvement_rule':'RMSE 차이 95% 구간 상한<0 AND Holm 보정 p<0.05','HAC':'Bartlett lag=h/3; 전체 3시간 달력 격자, 결측 영향량=0, 분모=공통 관측 수','bootstrap':'56개 달력 슬롯(168시간) 이동 블록; 모델별 동일 재표집, 결측 슬롯 유지','bootstrap_reps':reps,'seed':seed,'frozen_sha256':frozen_digest,'prediction_metadata':{},'source_feature_sha256':sha(FILE),'scorer_sha256':sha(Path(__file__))}
    write_json(out/'policy.meta.json',policy)
    preds={}
    for model in models:
        path=pred_root/f'{model}.parquet';meta=json.loads(path.with_suffix('.meta.json').read_text())
        current_version=json.loads((FEATURES/'feature_manifest.json').read_text())
        if meta.get('features_version')!=current_version['version'] or meta.get('features_created_kst')!=current_version['created_kst']:raise ValueError('다른 데이터 버전의 예측은 함께 채점할 수 없습니다.')
        if meta.get('prediction_sha256') and sha(path)!=meta['prediction_sha256']:raise ValueError('예측 체크섬 오류')
        d=pq.read_table(path,columns=COLUMNS,filters=[('split','=',split)],use_threads=False).to_pandas()
        if d.duplicated(['origin_time_kst','horizon']).any() or not d['model'].eq(model).all():raise ValueError('예측 키/모델 오류')
        preds[model]=d;policy['prediction_metadata'][model]=meta
    scores=[];tests=[];conditions=[];row_frames=[]
    for h in horizons:
        g=load_grid(split,h);g['persistence_raw']=g.seoul_pm25_now
        for model in models:
            p=preds[model];p=p[p.horizon.eq(h)].set_index('origin_time_kst');g[model+'_raw']=p.y_hat.reindex(g.index)
        raw_cols=[m+'_raw' for m in ['persistence',*models]];g=g[np.isfinite(g[raw_cols]).all(axis=1)].copy()
        if len(g)<56:raise ValueError('공통 평가 표본 부족')
        for model in ['persistence',*models]:g[model]=g[model+'_raw'].clip(lower=0)
        for model in ['persistence',*models]:scores.append({'horizon':h,'model':model,**metrics(g.y.to_numpy(),g[model].to_numpy(),g.persistence.to_numpy()),'negative_predictions_clipped':int(g[model+'_raw'].lt(0).sum())})
        full=timeline(split);errors={m:(g[m]-g.y).reindex(full).to_numpy(float) for m in ['persistence',*models]}
        for a,b in comparisons:
            stat,p=dm_test(errors[a],errors[b],h);ci=bootstrap(errors[a],errors[b],errors['persistence'],reps,seed)
            tests.append({'horizon':h,'model':a,'versus':b,'n':len(g),'rmse_diff':float(np.sqrt(np.nanmean(errors[a]**2))-np.sqrt(np.nanmean(errors[b]**2))),'dm_stat':stat,'p':p,**ci})
        for col in ['season','route_group','wind_sector','high_pm']:
            for value,part in g.groupby(col,dropna=False):
                for model in ['persistence',*models]:conditions.append({'horizon':h,'condition':col,'level':str(value),'model':model,**metrics(part.y.to_numpy(),part[model].to_numpy(),part.persistence.to_numpy()),'negative_predictions_clipped':int(part[model+'_raw'].lt(0).sum())})
        row_frames.append(g.reset_index())
    table=pd.DataFrame(tests);table['p_holm']=holm(table.p);table['improved']=(table.ci95_high<0)&(table.p_holm<.05);score=pd.DataFrame(scores)
    score.to_csv(out/f'score_metrics_{split}.csv',index=False);table.to_csv(out/f'score_tests_{split}.csv',index=False);pd.DataFrame(conditions).to_csv(out/f'score_conditions_{split}.csv',index=False)
    row_path=out/f'score_rows_{split}.parquet';pd.concat(row_frames,ignore_index=True).to_parquet(row_path,index=False)
    write_json(row_path.with_suffix('.meta.json'),{**policy,'score_rows_sha256':sha(row_path),'status':'completed','interpretation':'2023년은 설정 선택 이후 탐색적 검증 통계이며 최종 시험 성능이 아님' if split=='validation' else '동결된 2024 최종 평가'})
    plot_skill(score,table,out/f'rmse_improvement_{split}.png',split=split);return out,score,table

def plot_skill(scores,tests,path,split='validation'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for model,g in scores[scores.model.ne('persistence')].groupby('model'):
        g=g.sort_values('horizon');ci=tests[(tests.model==model)&(tests.versus=='persistence')].set_index('horizon').reindex(g.horizon);x=g.horizon.to_numpy();y=g.rmse_improvement_pct.to_numpy()
        ax.plot(x,y,marker='o',label=model);ax.vlines(x,ci.skill_ci95_low_pct,ci.skill_ci95_high_pct);ax.scatter(x,ci.skill_ci95_low_pct,marker='_',color='gray');ax.scatter(x,ci.skill_ci95_high_pct,marker='_',color='gray')
    title='2023년 Validation 중간 결과' if split=='validation' else '2024년 Test 최종 결과'
    ax.axhline(0,color='gray',linestyle='--');ax.set_xticks([3,6,12,24]);ax.set(xlabel='예측 시간 (시간)',ylabel='Persistence 대비 RMSE 개선률 (%)',title=title+' · 7일 블록 bootstrap 95% 구간');ax.legend();fig.savefig(path,dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description='공통 채점: 시험은 --final·동결 manifest·1회 잠금 필요')
    p.add_argument('--split',choices=['validation','test'],default='validation');p.add_argument('--models',nargs='+',required=True);p.add_argument('--horizons',nargs='+',type=int,default=[6]);p.add_argument('--pairs',nargs='*',default=[]);p.add_argument('--final',action='store_true');p.add_argument('--frozen',type=Path);p.add_argument('--output-root',type=Path)
    p.add_argument('--prediction-dir',type=Path)
    a=p.parse_args();out,s,t=evaluate(a.split,a.models,a.horizons,a.pairs,a.final,a.frozen,a.output_root,prediction_dir=a.prediction_dir);print(s.to_string(index=False));print(t.to_string(index=False));print('저장 위치:',out)
