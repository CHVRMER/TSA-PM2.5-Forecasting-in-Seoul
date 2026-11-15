"""완료된 M4T 결과 보고서·무결성 검사·파일 목록. 학습·채점 재실행 없음."""
from pathlib import Path
import sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
from src.common_scoring import sha
from scripts.run_m2 import protect
OUT=ROOT/'results/validation_2023/M4T'
def table(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
def comparisons(t):
 return pd.DataFrame([{'Horizon':int(r.horizon),'ΔRMSE':f'{r.rmse_diff:.4f}','7일 Bootstrap 95% CI':f'[{r.ci95_low:.4f}, {r.ci95_high:.4f}]','DM':f'{r.dm_stat:.4f}','DM p':f'{r.p:.6g}','Holm p':f'{r.p_holm:.6g}','공식 개선':'충족' if r.improved else '미충족'} for r in t.itertuples()])
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());pre=json.loads((OUT/'preflight.json').read_text());score=ROOT/manifest['common_scores'];protect(pre['protected_sha256'])
 assert sha(ROOT/'src/common_scoring.py')==pre['scorer_sha256']
 assert not (ROOT/'results/final_test_2024.lock').exists() and manifest['test_2024_executed'] is False
 metrics=pd.read_csv(score/'score_metrics_validation.csv');tests=pd.read_csv(score/'score_tests_validation.csv');conditions=pd.read_csv(score/'score_conditions_validation.csv')
 assert len(tests)==84
 features=json.loads((OUT/'h3/feature_names.json').read_text());excluded=pre['excluded_features'];assert len(features)==2924 and len(excluded)==28 and not set(features)&set(excluded)
 settings=[]
 for h in [3,6,12,24]:
  cfg=json.loads((ROOT/f'configs/m4t_{h}h.json').read_text());info=json.loads((OUT/f'h{h}/metrics_and_selection.json').read_text());meta=json.loads((OUT/f'h{h}/M4T.meta.json').read_text());modeldir=ROOT/f'Models/M4T/h{h}'
  assert cfg['feature_names']==features==meta['settings']['feature_names'] and cfg['excluded_features']==meta['settings']['excluded_features']==excluded
  assert all(x['max_absolute_prediction_difference']==0 for x in info['model_reload_checks'])
  assert len(info['trials'])==4 and len(info['seed_metrics'])==3
  for model in modeldir.glob('M4T*.txt'):
   # Stored LightGBM headers verify the actual feature order without inference.
   with model.open() as f:
    header=next(line for line in f if line.startswith('feature_names='))
   assert header.strip().split('=',1)[1].split()==features
  settings.append({'Horizon':h,'Train':info['eligible_train_rows'],'Validation':info['eligible_validation_rows'],'공통 평가':manifest['common_rows'][str(h)],'leaves':info['selected']['params']['num_leaves'],'fraction':info['selected']['params']['feature_fraction'],'iterations':info['selected']['best_iteration']})
 verification={'status':'passed','previous_artifact_sha256_unchanged':len(pre['protected_sha256']),'original_data_sha256_verified_by_training':True,'scorer_sha256_unchanged':True,'new_models':12,'model_reload_max_difference':0,'feature_names_match_configs_metadata_and_model_headers':True,'feature_count':len(features),'excluded_feature_count':len(excluded),'excluded_features':excluded,'same_common_rows_and_previous_predictions':True,'test_lock_unconsumed':True,'SHAP_Permutation_Adaptive_executed':False,'holm_family_size':84,'validation_all_missing_retained':['pm25_mn_b25_48_inside_mix_mean'],'initial_preflight_guard_correction':'학습 기간 기준으로 검증 조건 수정; 입력 변경 없음'}
 verification['initial_preflight_guard_correction']='초기 검증기의 train+validation 전체 결측 중단 조건을 승인된 train 기준으로 수정. 입력 변경 없음.'
 (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
 t=tests[tests.model.eq('M4T')];direct=t[t.versus.eq('M4')];tcompare=t[t.versus.eq('M3T')];baseline=t[t.versus.eq('persistence')]
 perf=metrics[['horizon','model','n','RMSE','MAE','MBE','rmse_improvement_pct']].copy()
 for c in ['RMSE','MAE','MBE','rmse_improvement_pct']:perf[c]=perf[c].map(lambda x:f'{x:.4f}')
 cond=conditions[(conditions.horizon==6)&conditions.model.isin(['M4','M3T','M4T'])]
 pivot=cond.pivot(index=['condition','level','n'],columns='model',values='RMSE').reset_index().round(4)
 negative=int(metrics.negative_predictions_clipped.sum())
 improved=direct.loc[direct.improved,'horizon'].tolist();desc='없음' if not improved else ', '.join(str(h)+'시간' for h in improved)
 quality=pd.DataFrame(pre['feature_missingness']);quality['all_missing_columns']=quality.all_missing_columns.map(lambda x:', '.join(x) or '없음');quality['missing_cell_pct']=quality.missing_cell_pct.map(lambda x:f'{x:.4f}')
 text=f'''# M4T 네 Horizon — 2023 Validation 완료 보고

**M4 대비 공식 개선을 충족한 Horizon: {desc}.** 2021–22년만 학습하고 2023년으로 조기 종료·설정을 선택했습니다. 2023년을 포함한 재학습은 없습니다. 2024년 Test, TreeSHAP, Permutation Importance, Adaptive는 실행하지 않았으며 이 단계에서 중단합니다.

## 1. 입력·원본 보존·검증

**M4T 2,924개 = M4 전체 2,518개 + 승인된 경로 406개.** 원본 `feature_sets.json`의 `local_plus_trajectory_and_station_history_balanced_without_optional_gases` 2,952개에서 승인 목록 28개만 모델 입력 설정에서 제외했습니다. 원본 데이터·전처리 파일·원본 Feature 목록은 변경하지 않았습니다. 숫자 열 전체 선택은 하지 않았습니다.

경로 구성은 all 64 / kr 64 / cn 64 / jp 55 / mn 45 / 기상 84 / 형태 30입니다. all 범위는 국내 67·중국 61·일본 188·몽골 1곳이며 기본 관측 이력은 일본 균형판 24곳을 유지합니다. M4 전체 포함·중복 없음·role=feature·실제 스키마·명시적 목록 일치를 확인했습니다.

제외된 28개는 일본 PM2.5 b01_06 농도 통계 5개, 몽골 PM2.5 b01_06/b07_12/b13_24 농도 통계 15개, 일본·몽골 PM10 평균 8개입니다. 앞의 20개는 해당 지리적 구간의 유효 농도 매칭 부재, 뒤의 8개는 PM10 관측의 구조적 부재입니다. 기존 승인 목록과 정확히 동일하며 2021–22 train에서 모두 결측임을 다시 확인했습니다. 검증에서도 이 28개는 전체 결측입니다. 실제 이름은 설정과 메타데이터, preflight.json에 기록했습니다.

추가 406개에는 각 Horizon 적격 **학습** 행에서 전체 결측인 열이 없습니다. 검증에서는 `pm25_mn_b25_48_inside_mix_mean` 1개가 전체 결측입니다. 이 열은 학습 기간에 유효값이 있어 승인된 제외 대상이 아니므로 유지했습니다. 울란바토르 결측을 임의로 채우지 않았습니다. 아래 결측률은 가용성 지표를 포함한 경로 입력 전체 셀 기준으로, 원농도 결측률과 다릅니다.

{table(quality)}

초기 검증 코드가 검증 기간의 전체 결측도 중단 대상으로 잘못 분류하여 **학습 전** 한 차례 중단했습니다. 읽기 전용 확인으로 학습 데이터에는 추가 전체 결측이 없음을 확인한 뒤, 조건을 승인된 학습 기간 기준으로 수정했습니다. 데이터·입력 목록 변경이나 모델 재학습은 없었습니다. 초기 로그와 확인 결과는 `preflight_stopped.log`, `preflight_failure_details.json`에 보존했습니다.

시간 검증: 2021–23년 도착 높이 500m·과거 1–48시간의 경로 시점 420,480건과 관측 연결 2,377,159건에서 pass=t−lookback, observation=pass, pass<t를 확인했습니다. 일본은 승인된 README 시간 해석을 유지합니다. 각 Horizon eligible_h·학습/검증 기간·정답 경계와 기존 M4 정답·행 일치를 확인했습니다. 미래 정답·가용성 label은 입력 목록에 포함하지 않았습니다.

승인된 연구 한계도 유지합니다. 오프라인 과거 자료를 이용한 예측력 비교이며 실제 공개 지연·기상자료 빈티지를 복원한 실시간 운영 검증은 아닙니다. 전체 데이터 파일의 무결성 해시는 검사하지만 2024 관측·정답의 학습·채점은 수행하지 않았습니다.

## 2. 공통 학습·설정

{table(pd.DataFrame(settings))}

기존 네 설정(leaves 15/31 × feature_fraction .3/.5), 최대 5,000회·log1p L2 조기 종료 인내 200회, seed0 원 단위 RMSE 설정 선택, 시드 0/1/2 앙상블, expm1 복원·채점 때만 음수 0 절단을 유지했습니다. 공통 학습 함수·로더·채점기를 재사용했습니다. 모델별 12개 TXT와 설정·메타데이터를 저장했습니다. 12개 모두 재로드 예측 최대 차이 0이며 모델 헤더의 실제 Feature 순서까지 설정 목록과 일치합니다.

## 3. Horizon별 성능

Persistence/M1/M2/M3/M4/M3T/M4T가 동일한 평가 행을 사용합니다. 모든 오차 단위는 µg/m³, rmse_improvement_pct는 Persistence 대비 %입니다.

{table(perf)}

### M4T vs M4 — 주 비교

ΔRMSE=RMSE(M4T)−RMSE(M4). 음수가 개선입니다.

{table(comparisons(direct))}

### M4T vs M3T — 보조 비교

ΔRMSE=RMSE(M4T)−RMSE(M3T).

{table(comparisons(tcompare))}

M3T와 M4T는 기본 관측망과 경로 Feature가 함께 다릅니다. 두 모델 간 차이를 일본·몽골 개별 효과나 순수 경로 형태 효과로 분리할 수 없습니다. M4→M4T 역시 **경로 농도·기상·형태·가용성을 포함한 확장 정보 묶음의 추가 효과**로 해석합니다.

### M4T vs Persistence

{table(comparisons(baseline))}

공식 판정은 **ΔRMSE의 95% CI 상한<0 AND Holm 보정 p<0.05**입니다. 채점 전 범위를 고정하여 전체 21개 쌍×4 Horizon=84개 검정을 Holm family로 사용했습니다. 기존 단계의 원본 p-value·판정 파일은 보존하고 이번 통합 결과에만 84개 보정을 적용했습니다. 과거 단계와 Holm 값이 달라질 수 있는 이유는 검정 집합 확대이며 예측값 변경이 아닙니다.

DM은 제곱오차·Bartlett HAC lag=h/3이며 원래 3시간 시간축을 유지합니다. Bootstrap은 달력 7일=56슬롯 paired 이동 블록 1,000회·seed0입니다. RMSE 차이 CI와 Persistence 대비 % 개선률 CI는 구분해 저장했습니다. 전체 모델·Horizon 음수 절단 합계는 {negative}건입니다. 2023년은 설정 선택에도 쓴 기간이므로 결과는 탐색적 Validation이며 최종 Test 성능 증거와 구분합니다.

## 4. 조건별 분석

계절(cold=11~3월, warm=4~10월), 궤적 방향, ASOS 풍향, 고농도(정답 PM2.5≥35)별 n/MAE/RMSE/MBE/개선률을 공통 채점기로 계산했습니다. 작은 범주도 n과 함께 보존했고 조건별 별도 유의성·인과 해석은 하지 않았습니다. 6시간 RMSE 비교는 아래와 같으며 모든 Horizon의 상세 값은 공통 채점 폴더의 `score_conditions_validation.csv`에 있습니다.

{table(pivot)}

## 5. 저장 경로·그래프·보존 검증

- 공식 결과: `results/validation_2023/M4T/`; 모델: `Models/M4T/h3`, `h6`, `h12`, `h24`.
- 설정: `configs/m4t_3h.json`, `m4t_6h.json`, `m4t_12h.json`, `m4t_24h.json`. 실제 2,924개 목록·제외 28개를 모델 설정과 예측 .meta.json의 settings에 기록.
- 각 Horizon: 진단 예측 CSV/Parquet, 공통 5열 예측+메타데이터, Feature 목록·선택 결과·로그·진단 그림. 모델 폴더에는 시드3개 TXT와 ensemble_manifest/config/Feature 목록.
- `predictions/`: 기존 M1~M4·M3T 예측 파일의 동일 사본, 신규 M4T 11,680행과 메타데이터.
- `M1_M2_M3_M4_M3T_M4T_Persistence_2023_validation_metrics.csv`: 전체 성능.
- `M4T_vs_M1_M2_M3_M4_M3T_2023_validation_statistics.csv`: 직접 비교 ΔRMSE·CI·DM·Holm·판정.
- `M1_M2_M3_M4_M3T_M4T_2023_validation_tests.csv`: 전체 84개 검정.
- `M1_M2_M3_M4_M3T_M4T_2023_validation_intermediate_rmse_improvement.png`: x=3/6/12/24h, y=Persistence 대비 RMSE 개선률%, 모델별 선·7일 Bootstrap 95% 오차막대. **2023 Validation 중간 그림**으로 명시.
- 공통 채점: `{manifest['common_scores']}`. score_rows_validation.parquet와 .meta.json에 시점·정답 시점·Horizon·정답·모델별 예측·조건·데이터 버전·시드·설정 기록.
- preflight.json / scope_before_training.json / verification.json / manifest.json / horizon_settings.json / run.log / file_inventory.csv: 사전 검증·범위 고정·검증·파일 체크섬.

기존 관련 모델·결과·설정 **{len(pre['protected_sha256'])}개 파일 SHA256 불변**을 확인했습니다. M1~M4·M3T 재학습·재추론은 없습니다. 공통 평가 행·이전 모델 예측은 직전 단계와 행 단위로 동일합니다. 원본 데이터 해시는 공통 학습 코드에서 기존 검증 해시와 대조했고 공통 채점기 SHA256도 불변입니다. Feature 조합 설정은 M4T 항목만 추가했고 이전 정의는 동일하며 변경 전 원문을 source_before에 보존했습니다.

## 6. 중단 상태

2024 Test 잠금은 미소비 상태입니다. TreeSHAP, Permutation Importance, Adaptive 및 2024 Test는 실행하지 않았습니다. 완료된 고정 경로를 재실행하면 덮어쓰지 않고 중단합니다.

현재 단계의 작업을 완료했습니다. 결과를 검토한 뒤 승인해 주시면 다음 단계로 진행하겠습니다.
'''
 (OUT/'REPORT.md').write_text(text)
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='file_inventory.csv']+list((ROOT/'Models/M4T').rglob('*'))+[ROOT/f'configs/m4t_{h}h.json' for h in [3,6,12,24]]+[ROOT/p for p in ['configs/feature_groups.json','src/train.py','src/feature_groups.py','scripts/run_m4t.py','scripts/report_m4t.py']]
 with (OUT/'file_inventory.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader()
  for p in sorted(files):
   if p.is_file():writer.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 print(perf[perf.model.eq('M4T')].to_string(index=False));print(comparisons(direct).to_string(index=False));print('M4T vs M3T');print(comparisons(tcompare).to_string(index=False));print('REPORT',OUT/'REPORT.md')
if __name__=='__main__':main()
