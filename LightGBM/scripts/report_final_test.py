"""Read completed one-shot results only. No inference, fitting, rescoring or new tests."""
from pathlib import Path
import os,sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.final_test import sha,write_json,MODELS,HS,CORE,PRIMARY,SECONDARY
OUT=ROOT/'results/test_2024';EXEC=OUT/'execution'
def table(d):
    return '| '+' | '.join(map(str,d.columns))+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None))
def test_table(d):
    return pd.DataFrame([{'h':int(x.horizon),'candidate':x.model,'reference':x.versus,'ΔRMSE':f'{x.rmse_diff:.4f}','95% CI':f'[{x.ci95_low:.4f}, {x.ci95_high:.4f}]','DM':f'{x.dm_stat:.4f}','DM p':f'{x.p:.6g}','Holm144 p':f'{x.p_holm:.6g}','후보 공식 개선':'충족' if x.improved else '미충족','역방향 유의':bool(x.ci95_low>0 and x.p_holm<.05)} for x in d.itertuples()])
def main():
    manifest=json.loads((EXEC/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['mode']=='production';plan=json.loads((OUT/'frozen_plan.json').read_text());score=ROOT/manifest['common_scores']
    for rel,digest in manifest['artifacts_sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert sha(EXEC/'events.jsonl')==manifest['event_log_sha256'];assert sha(OUT/'frozen_plan.json')==manifest['plan_sha256']
    metrics=pd.read_csv(score/'score_metrics_test.csv');tests=pd.read_csv(score/'score_tests_test.csv');conditions=pd.read_csv(score/'score_conditions_test.csv');details=pd.read_parquet(EXEC/'adaptive_application_details.parquet');scored=pd.read_parquet(score/'score_rows_test.parquet')
    assert len(tests)==144 and len(metrics)==36 and len(scored)==11319
    report=OUT/'report';report.mkdir(exist_ok=False);figures=report/'figures';figures.mkdir()
    metrics.to_csv(report/'all_horizons_metrics.csv',index=False);tests.to_csv(report/'all_144_comparisons.csv',index=False)
    core=tests[[tuple(x) in CORE for x in tests[['model','versus']].itertuples(index=False,name=None)]];assert len(core)==80;core.to_csv(report/'core_80_comparisons_same_Holm144.csv',index=False)
    five=['M1','M3','M3T','Adaptive_wind','Adaptive_route'];primary=tests[tests.horizon.eq(6)&tests.model.isin(five)&tests.versus.isin(five)];primary.to_csv(report/'h6_primary_five_model_comparisons.csv',index=False)
    reverse=tests[(tests.ci95_low>0)&(tests.p_holm<.05)];reverse.to_csv(report/'significant_reverse_direction_same_tests.csv',index=False)
    # Figures use only the saved skill/CI values. No Bootstrap/DM is recalculated.
    plt.rcParams['font.family']=['Apple SD Gothic Neo','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    for filename,models in [('final_rmse_improvement_all_models.png',MODELS),('final_rmse_improvement_primary_models.png',five)]:
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained');colors=plt.cm.tab10(np.arange(len(models)))
        for i,(model,color) in enumerate(zip(models,colors)):
            g=metrics[metrics.model.eq(model)].sort_values('horizon');ci=tests[(tests.model==model)&tests.versus.eq('persistence')].set_index('horizon').reindex(g.horizon);x=g.horizon.to_numpy(float)+(i-(len(models)-1)/2)*.065;y=g.rmse_improvement_pct.to_numpy()
            ax.plot(x,y,marker='o',label=model,color=color);ax.vlines(x,ci.skill_ci95_low_pct,ci.skill_ci95_high_pct,color=color,alpha=.65);ax.scatter(x,ci.skill_ci95_low_pct,marker='_',color=color);ax.scatter(x,ci.skill_ci95_high_pct,marker='_',color=color)
        ax.axhline(0,color='grey',linestyle='--');ax.set_xticks(HS);ax.set_xlabel('예측 시간 (h)');ax.set_ylabel('Persistence 대비 RMSE 개선률 (%)');ax.set_title('2024 최종 Test · 동결 모델\n7일 블록 Bootstrap 95% CI (가독성을 위해 모델별 x위치 소폭 이동)');ax.legend(ncol=2,fontsize=9);fig.savefig(figures/filename,dpi=180);plt.close(fig)
    # Compare adaptive assignments on all origins and on the actual common scoring rows.
    counts=[]
    for scope,d in [('all_test_origins',details),('common_scoring_rows',details.merge(scored[['origin_time_kst','horizon']],on=['origin_time_kst','horizon'],validate='many_to_one'))]:
        for (model,h),g in d.groupby(['model','horizon']):counts.append({'scope':scope,'model':model,'horizon':int(h),'n':len(g),'season_fallback':int(g.season_fallback_used.sum()),'small_cell_fallback':int(g.fallback_reason.eq('cell_n_less_than_100').sum()),'unknown_fallback':int(g.fallback_reason.eq('unknown_condition').sum()),'frozen_near_zero_exception_applied':int(g.near_zero_exception_used.sum())})
    adaptive_counts=pd.DataFrame(counts);adaptive_counts.to_csv(report/'adaptive_fallback_counts.csv',index=False)
    common=[]
    for h in HS:
        a=manifest['row_audit'][str(h)];common.append({'h':h,'eligible':a['eligible_rows'],'Persistence 결측 제외':a['persistence_missing_rows'],'공통 평가':a['common_rows']})
    pd.DataFrame(common).to_csv(report/'common_row_counts.csv',index=False)
    full=metrics[['horizon','model','n','RMSE','MAE','MBE','rmse_improvement_pct','negative_predictions_clipped']].copy()
    for col in ['RMSE','MAE','MBE','rmse_improvement_pct']:full[col]=full[col].map(lambda v:f'{v:.4f}')
    rmse=metrics.pivot(index='horizon',columns='model',values='RMSE').reindex(columns=['persistence',*MODELS]).reset_index().round(4);rmse.to_csv(report/'all_horizons_RMSE.csv',index=False)
    main_pairs=[('M2','M1'),('M3','M2'),('M3','M1'),('M4','M3'),('M3T','M3'),('M4T','M4'),('M4T','M3T'),('Adaptive_wind','M1'),('Adaptive_wind','M3'),('Adaptive_route','M1'),('Adaptive_route','M3'),('Adaptive_route','Adaptive_wind')]
    major=tests[[tuple(x) in main_pairs for x in tests[['model','versus']].itertuples(index=False,name=None)]];major.to_csv(report/'major_pairs_all_horizons.csv',index=False)
    brief=major.pivot(index=['model','versus'],columns='horizon',values='improved').reset_index()
    for h in HS:brief[h]=brief[h].map({True:'충족',False:'미충족'})
    cond6=conditions[conditions.horizon.eq(6)&conditions.model.isin(five)].pivot(index=['condition','level','n'],columns='model',values='RMSE').reset_index().round(4)
    events=[json.loads(x) for x in (EXEC/'events.jsonl').read_text().splitlines()];names=[x['event'] for x in events];assert names.index('pipeline_claimed')<names.index('feature_columns_read')<names.index('predictions_frozen')<names.index('label_columns_read');assert names.count('scoring_requested')==1 and names.count('completed')==1
    timing=pd.DataFrame([{'event':x['event'],'time_kst':x['time_kst']} for x in events if x['event'] in ['pipeline_claimed','feature_columns_read','predictions_saved','predictions_frozen','scoring_requested','label_columns_read','completed']])
    text=f'''# Stage 9 — 2024 최종 Test 완료 보고

**동결 계획으로 최종 Test를 한 번 실행했습니다.** Run ID `{manifest['run_id']}`. 기존 모델·가중치·설정을 변경하거나 재학습하지 않았습니다. 결과를 본 뒤 추가 튜닝·가중치 변경·재학습·재채점은 하지 않았습니다. 이 보고서는 이미 저장된 결과만 읽어 작성했습니다.

## 1. 주요 결론 — 6시간 주 Horizon

- M1 RMSE7.2113, M3 7.3558, M3T 7.1228, Adaptive-풍향7.3243, Adaptive-경로7.3489 µg/m³입니다. 전체 후보 중 RMSE 점 추정 최저는 M4T7.1115이며 M3T와의 차이는 공식 개선 조건 미충족입니다.
- **M3T vs M3는 공식 개선 충족**: ΔRMSE−0.2330,95%CI[−0.3486,−0.1437], Holm144 p≈0.000376. M4T vs M4도 Δ−0.2504,CI[−0.3584,−0.1505],p≈0.0000371로 충족합니다.
- M3 vs M1은 Δ+0.1445,CI[0.0154,0.2678],Holm p≈0.4714입니다. 중국 전체 관측망 추가만으로 M1 대비 개선이 확인되지 않았습니다. 중국 근거리21곳의 M2 vs M1도 Δ−0.0066,CI가0을 포함하고p=1로 미충족입니다.
- M3 vs M2는 Δ+0.1511,CI[0.0504,0.2392],Holm p≈0.03997입니다. 원래 후보 M3의 improved는False이며, 동일한 사전 등록 양측 검정에서 **반대 방향인 M2 우세**가 확인됩니다. 새로운 역방향 검정을 추가하거나 재보정하지 않았습니다.
- M3T vs M1은 Δ−0.0885,CI[−0.2652,0.0589],p=1로 공식 개선 미충족입니다. 따라서 경로 정보 묶음이 M3를 개선했다는 결과와, 국내 M1보다 유의하게 우수하다는 주장은 구분합니다.
- Adaptive-풍향·경로 모두 M1 대비 공식 개선을 충족하지 못했습니다. M3 대비로도6시간에서는 둘 다 미충족입니다. M3T를 기준으로 한 동일한 양측 검정에서는 Adaptive-풍향/경로의 RMSE가 각각+0.2015/+0.2261 높고CI가양수·Holm p≈0.00791/0.00115로 **M3T 우세**입니다.
- 8개 후보 모두 Persistence 대비로는 네 Horizon 모두 공식 개선을 충족했습니다. 전체144개 후보 방향 판정 중 improved=True는40개이며 그중32개는 Persistence 비교입니다.

## 2. 수행·동결·검증 기록

계획 SHA256 `{manifest['plan_sha256']}`

예측 동결 Manifest SHA256 `{manifest['prediction_manifest_sha256']}`

144개 검정 집합과 공통 채점 코드를 그대로 사용했습니다. 전체36개 쌍×4Horizon에 Holm을 한 번 적용했습니다. 핵심80개는 표시용이며 별도 작은 집합으로 재보정하지 않았습니다. 제곱오차 DM·Bartlett HAC lag=h/3, 달력3시간 격자 유지, 56슬롯=7일 paired 이동 블록 Bootstrap1000회·seed0, 원 단위 채점·음수0절단을 유지했습니다. 공식 기준은 **후보−기준 ΔRMSE95%CI상한<0 AND Holm p<.05**입니다.

시드별 expm1 복원 후3개 예측 평균,2023 고정 Adaptive 가중치를 적용했습니다. M3T2424·M4T2924입력과 제외28개를 그대로 유지했습니다. 실제자료 해시·입력시간격자·feature cutoff·타깃 정확한t+h·전체 예측 저장 재열람·공통행·지표 재계산 검증이 모두 통과했습니다.

잠금과 열 접근 순서:

{table(timing)}

최초 입력 체크섬 접근도 첫 잠금 후에 수행됐고, 정답 열은 모든 예측을 동결한 뒤 두 번째 잠금을 통과해서 읽었습니다. 정상 완료는1회, 채점 호출도1회입니다. 잠금2개를 유지합니다. 실행기·모델 설정·가중치는 수정하지 않았습니다.

## 3. 공통 평가 행

{table(pd.DataFrame(common))}

각 모델은 동일한 Horizon별 행에서 평가했습니다. origin2928개×4Horizon=11712개 예측을 모델별로 저장했고, 실제 공통 평가 행은 합계11319개입니다. 정답 결측은 대체하지 않았고 Persistence가 없는 행만 해당 공통평가에서 추가 제외했습니다. 모델 예측 누락이나 비유한값으로 인한 임의 표본 축소는 없었습니다. origin과 채점 타깃 시점 모두2024입니다. 전체9모델·4Horizon의 음수 절단 건수는0입니다.

## 4. 모든 Horizon 성능

오차 단위는 µg/m³, rmse_improvement_pct는 Persistence 대비%입니다. MBE=예측−실측입니다.

{table(full)}

RMSE 비교 요약:

{table(rmse)}

점 추정 최저는3h M3T,6/12h M4T,24h M3T입니다. 이 순위만으로 모든 모델 간 유의한 우월성을 주장하지 않습니다.

## 5. 주요 비교의 공식 개선 판정

다음 표의 ΔRMSE는 후보−기준입니다. 원래 improved 열은 그대로이며, 역방향 유의는CI하한>0 AND 같은 Holm144 p<.05일 때 기준 모델의 우세를 설명하는 표시입니다. 별도 역방향 검정·새로운 Holm 집합은 만들지 않았습니다.

{table(brief)}

주요 비교 전체 수치:

{table(test_table(major))}

6시간 주 분석5개 모델 사이의 사전 포함된10개 비교:

{table(test_table(primary))}

원래 방향과 반대 방향의 유의 차이가 확인된 비교5개:

{table(test_table(reverse))}

6시간에서 M4T−M3T의 차이는−0.0113,CI[−0.0744,0.0558],p=1로 미충족입니다. M3T/M4T 비교는 기본 관측망과 경로 Feature가 함께 다르므로 일본·몽골의 개별 효과로 분해하지 않습니다.

## 6. 중국 관측·경로·Adaptive 효과 해석

중국 근거리 추가 M2−M1, 중국 전체 M3−M1 및 M3−M2의 개선 조건은 네 Horizon에서 모두 충족되지 않았습니다. 특히6시간에는 M3보다 M2가 우세한 반대 방향 차이가 있었습니다. 이 결과는 이번 고정 모델과2024 공통 평가에서 중국 관측을 늘리는 것만으로 일관된 추가 개선을 입증하지 못했다는 뜻이며 중국 자료의 보편적 무용성이나 인과적 악영향을 의미하지 않습니다.

경로 정보 추가는 M3T−M3에서6시간, M4T−M4에서6·12시간 공식 개선을 충족했습니다. 나머지Horizon은 점 추정 감소가 있더라도 같은144개 보정의 조건을 충족하지 못했습니다. M3T의24시간 ΔRMSE−0.1903은CI가음수지만Holm p≈0.0687로 미충족입니다. 유리한Horizon이나 검정만 골라 해석하지 않았습니다.

경로 효과는 **농도·기상·형태·가용성을 포함한 확장 정보 묶음**의 효과이며 순수 경로 형상이나 중국 수송의 인과 효과가 아닙니다. all64개에 국내·중국·일본188곳·몽골 정보가 함께 들어가는 승인된 설계를 유지했습니다.

Adaptive 두 버전은 M3 대비3시간에서만 공식 개선을 충족했습니다. M1 대비는 네Horizon 모두 미충족입니다. 6시간에서는 둘 다 M3T보다 RMSE가 유의하게 높았습니다. 2023표본 내 참고 성능이2024에서 자동으로 재현되는 것은 아니며, 이 결과를 본 뒤 가중치를 다시 맞추지 않았습니다.

## 7. 조건별 성능

계절·12시간 전 경로 방향·ASOS풍향·고농도≥35별 MAE/RMSE/MBE/개선률·n·절단 건수는 원 공통 score_conditions_test.csv에 모두 있습니다. 다음은6시간 주요5모델 RMSE입니다.

{table(cond6)}

M3T는 M3 대비 고농도에서 RMSE15.4860→15.0974, 비고농도에서5.7115→5.4984로 낮았습니다. 다만 고농도에서는 M1의15.0128이 M3T보다 소폭 낮고, 비고농도에서는 M3T가 낮았습니다. 한랭기에는 M1이 M3T보다 낮고 온난기에는 M3T가 낮았습니다. 이런 조건별 차이는 기술 통계이며 별도 조건별 유의성 검정이나새로운 분할을 추가하지 않았습니다. 분류불가풍향 n=10처럼 작은 범주는 강하게 일반화하지 않습니다.

Adaptive 적용 건수(전체origin과 공통평가행 구분):

{table(adaptive_counts)}

2024 표본 수로 가중치를 재추정하지 않았습니다.2023표본 부족·unknown일 때 고정 계절 가중치를 사용했고, 고정 분모 예외 적용은0행입니다.

## 8. 산출물·보존·종료 상태

- 실행 원본: `{str(EXEC.relative_to(ROOT))}/manifest.json`, events.jsonl, frozen_predictions.json, predictions/8모델 Parquet+.meta.json, adaptive_application_details.parquet.
- 공통 채점 원본: `{manifest['common_scores']}`의 score_metrics_test.csv / score_tests_test.csv / score_conditions_test.csv / score_rows_test.parquet+.meta.json / policy.meta.json / rmse_improvement_test.png.
- 보고용 표: report/all_horizons_metrics.csv, all_horizons_RMSE.csv, all_144_comparisons.csv, core_80_comparisons_same_Holm144.csv, major_pairs_all_horizons.csv, h6_primary_five_model_comparisons.csv, significant_reverse_direction_same_tests.csv, adaptive_fallback_counts.csv, common_row_counts.csv.
- 최종 그림: report/figures/final_rmse_improvement_all_models.png 및 final_rmse_improvement_primary_models.png. x=3/6/12/24h, y=Persistence 대비RMSE개선률%, 선=모델, 오차막대=기존 계산된7일Bootstrap95%CI. 가독성을 위해 표시 x위치만소폭 이동했으며 결과를 재계산하지 않았습니다. 공통 채점기의 원 그림도 그대로 보존했습니다.
- 원 Manifest는수정하지 않고 보고용 report_manifest.json과 file_inventory.csv를별도 저장했습니다.

실행기에서 기존 보호산출물 **{manifest['originals_preserved']}항목**과 모델·설정·코드 해시가 보존됨을 검증했고, 보고 작성 전후에도 원 실행결과 해시를 확인했습니다. 기존 검증·SHAP·Adaptive 산출물은 보존했습니다. 잠금2개는 소비된 상태로 남아 있으며 자동 삭제·재실행하지 않습니다. 동결 계획의 대기 상태 문자열은 실행 전 불변 기록이고 현재 완료 상태는 execution/manifest.json이 기준입니다.

**최종 Test 완료 후 추가 튜닝·가중치 변경·재학습·새 검정·재채점은 하지 않았습니다. 여기서 중단합니다.**
'''
    (OUT/'REPORT.md').write_text(text)
    # Preserve the preparation note, then update only the non-frozen landing page.
    if (OUT/'README.md').exists():
        with (OUT/'README.preparation.md').open('xb') as f:f.write((OUT/'README.md').read_bytes())
    (OUT/'README.md').write_text('Stage9 실제2024 최종 Test가 완료되었습니다. 잠금2개를 유지하며 재실행하지 않습니다.\n\n현재 결과: [최종 보고서]('+str(OUT/'REPORT.md')+')\n실행 근거: [실행 Manifest]('+str(EXEC/'manifest.json')+')\n동결 계획과 사전 검증 기록은 변경하지 않았습니다.\n')
    for ledger in [plan['preserve_sha256'],plan['static_artifacts_sha256'],manifest['artifacts_sha256']]:
        for rel,digest in ledger.items():assert sha(ROOT/rel)==digest,rel
    assert sha(EXEC/'events.jsonl')==manifest['event_log_sha256'] and (ROOT/PRIMARY).exists() and (ROOT/SECONDARY).exists()
    write_json(OUT/'report_manifest.json',{'status':'completed','stage':9,'reporting_only':True,'new_inference_or_fitting':False,'new_statistical_tests_or_rescoring':False,'run_id':manifest['run_id'],'execution_manifest_sha256':sha(EXEC/'manifest.json'),'plan_sha256':manifest['plan_sha256'],'report_code_sha256':sha(Path(__file__)),'input_score_metrics_sha256':sha(score/'score_metrics_test.csv'),'input_score_tests_sha256':sha(score/'score_tests_test.csv'),'holm_family_size':144,'core_subset_uses_same_holm':True,'prior_artifacts_preserved':manifest['originals_preserved'],'final_locks_retained':True,'figures':[str(p.relative_to(ROOT)) for p in figures.glob('*.png')]})
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='file_inventory.csv']+[Path(__file__)]
    with (OUT/'file_inventory.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
        for p in sorted(files):w.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
    print('Final report:',OUT/'REPORT.md');print('Original test outputs unchanged; no rescoring/refit; locks retained.');print(rmse.to_string(index=False))
if __name__=='__main__':main()
