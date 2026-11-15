"""완료된 M3T 검증 산출물의 보고서·목록 작성. 학습·추론·채점 재실행 없음."""
from pathlib import Path
import json,csv,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/validation_2023/M3T'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def table(df):
 return '| '+' | '.join(df.columns)+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());pre=json.loads((OUT/'preflight.json').read_text());score=ROOT/manifest['common_scores']
 metrics=pd.read_csv(score/'score_metrics_validation.csv');tests=pd.read_csv(score/'score_tests_validation.csv');conditions=pd.read_csv(score/'score_conditions_validation.csv')
 for rel,digest in pre['protected_sha256'].items():
  assert sha(ROOT/rel)==digest,rel
 assert sha(ROOT/'src/common_scoring.py')==pre['scorer_sha256']
 assert not (ROOT/'Models/M4T').exists() and not (ROOT/'results/final_test_2024.lock').exists()
 assert manifest['status']=='completed' and manifest['new_models']==12
 features=json.loads((OUT/'h3/feature_names.json').read_text())
 models=list((ROOT/'Models/M3T').glob('h*/M3T_h*_seed*.txt'));assert len(models)==12
 settings=[]
 for h in [3,6,12,24]:
  cfg=json.loads((ROOT/f'configs/m3t_{h}h.json').read_text());info=json.loads((OUT/f'h{h}/metrics_and_selection.json').read_text());meta=json.loads((OUT/f'h{h}/M3T.meta.json').read_text())
  assert cfg['feature_names']==features==meta['settings']['feature_names'] and cfg['excluded_features']==meta['settings']['excluded_features']==[]
  assert all(x['max_absolute_prediction_difference']==0 for x in info['model_reload_checks'])
  settings.append({'Horizon':h,'Train':info['eligible_train_rows'],'Validation':info['eligible_validation_rows'],'공통 평가':manifest['common_rows'][str(h)],'leaves':info['selected']['params']['num_leaves'],'fraction':info['selected']['params']['feature_fraction'],'iterations':info['selected']['best_iteration']})
 assert not (ROOT/'Models/M4T').exists()
 verification={'status':'passed','previous_artifact_sha256_unchanged':len(pre['protected_sha256']),'original_data_sha256_verified_by_training':True,'scorer_sha256_unchanged':True,'new_models':12,'model_reload_max_difference':0,'feature_names_match_configs_and_prediction_metadata':True,'feature_count':len(features),'excluded_features':[],'same_common_rows_and_previous_predictions':True,'test_lock_unconsumed':True,'M4T_executed':False,'SHAP_Permutation_Adaptive_executed':False,'holm_family_size':60}
 (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
 direct=tests[(tests.model=='M3T')&(tests.versus=='M3')].copy();base=tests[(tests.model=='M3T')&(tests.versus=='persistence')].copy()
 def comparisons(t):
  return pd.DataFrame([{'Horizon':int(r.horizon),'ΔRMSE':f'{r.rmse_diff:.4f}','7일 Bootstrap 95% CI':f'[{r.ci95_low:.4f}, {r.ci95_high:.4f}]','DM':f'{r.dm_stat:.4f}','DM p':f'{r.p:.6g}','Holm p':f'{r.p_holm:.6g}','공식 개선':'충족' if r.improved else '미충족'} for r in t.itertuples()])
 perf=metrics[['horizon','model','n','RMSE','MAE','MBE','rmse_improvement_pct']].copy()
 for c in ['RMSE','MAE','MBE','rmse_improvement_pct']:perf[c]=perf[c].map(lambda x:f'{x:.4f}')
 mainperf=perf[perf.model.isin(['M3','M3T'])]
 # 조건별 세부 표는 공통 채점 CSV를 그대로 사용한다.
 cond6=conditions[(conditions.horizon==6)&conditions.model.isin(['M3','M3T'])]
 text=f'''# M3T 3/6/12/24시간 — 2023 Validation 보고

M3T 네 Horizon의 학습·검증을 완료했습니다. 2021–22년만 학습하고 2023년으로 조기 종료·설정을 선택했습니다. 2023년을 포함한 재학습과 2024년 Test는 수행하지 않았습니다. **M4T는 실행하지 않았으며 별도 승인 대기입니다.**

## Feature·시간·가용성 검증

M3T **2,424개 = M3 전체 2,118개 + 경로 Feature 306개**입니다. 추가 구성은 all 64 / kr 64 / cn 64 / 기상 84 / 형태 30입니다. `feature_sets.json`의 세 명시적 목록을 순서 보존 합집합으로 사용했고, 숫자 열 전체 선택은 하지 않았습니다. 제외 목록은 빈 목록입니다. M4T용 28개 제외 규칙을 M3T에 적용하지 않았습니다.

실제 Feature 2,424개와 제외 목록은 Horizon별 설정, `feature_names.json`, 모델 `ensemble_manifest.json`, 예측 `.meta.json`의 settings에 기록했습니다. 원본 데이터·전처리 파일은 수정하지 않았습니다. 기존 공통 로더·학습·채점 구현을 재사용했습니다. 변경은 명시적 Feature 목록 합집합 지원, M3T 학습 허용 및 실행 설정·조율 코드입니다. 변경 전 소스와 Feature 조합 설정은 `source_before/`에 보존했습니다.

all 집계는 국내 67·중국 61·일본 188·몽골 1곳을 포함합니다. M3→M3T는 **경로 농도·기상·형태·가용성을 포함한 확장 정보 묶음의 추가 효과**이며 순수한 경로 형상 효과가 아닙니다.

2021–23년 도착 높이 500m, 과거 1–48시간의 경로 시점 420,480건·관측 연결 2,377,159건을 확인했습니다. 통과 시각=t−lookback이며 관측 연결 시각은 통과 시각과 일치했습니다. 일본 시간은 기존 승인된 README 해석을 유지하고, 울란바토르 결측을 포함한 입력 결측은 대체하지 않았습니다. 각 Horizon 적격 학습·검증 행에서 추가 306개 중 전체 결측 열은 없었습니다. 결측률은 `preflight.json`에 기록했습니다.

경로 생성은 과거 시점 자료를 오프라인으로 결합한 연구 데이터입니다. 자료 공개 지연과 실시간 기상자료 버전을 재현하지 않았다는 승인된 한계를 유지합니다. 미래 정답은 입력에 포함되지 않으며 2024 경계 정답도 로딩하지 않았습니다. 전체 파일 SHA256 무결성 확인은 수치적 Test 열람과 별개로 수행했습니다.

## 학습·표본·재현성

{table(pd.DataFrame(settings))}

기존 네 설정(leaves 15/31 × feature_fraction .3/.5), seed0 탐색 후 원 단위 RMSE 선택, 시드 0/1/2 앙상블, log1p 학습 후 expm1 복원, 최대 5,000회·조기 종료 200회, 기존 LightGBM 설정을 유지했습니다. 모델 12개를 저장했고 재로드 예측 최대 차이는 0입니다. 각 `eligible_h`와 정답은 기존 M3와 일치합니다. 공통 평가 행 및 기존 모델의 예측값도 행 단위로 동일합니다.

기존 결과·모델·설정 {len(pre['protected_sha256'])}개 파일의 SHA256 불변을 확인했습니다. 공통 채점기 SHA256도 불변입니다. M1~M4 재학습·재추론은 없었습니다.

## 공통 성능

모든 지표는 Persistence/M1/M2/M3/M4/M3T 동일 평가 행 기준이며 오차 단위는 µg/m³입니다. RMSE 개선률은 Persistence 대비 %입니다.

{table(mainperf)}

### 주 비교: M3T vs M3

ΔRMSE=RMSE(M3T)−RMSE(M3). 음수는 M3T의 RMSE가 낮다는 뜻입니다.

{table(comparisons(direct))}

### M3T vs Persistence

{table(comparisons(base))}

공식 판정은 **ΔRMSE의 95% CI 상한 < 0 AND Holm 보정 p < 0.05**입니다. 통합 비교의 15개 쌍×4 Horizon=60개 검정 집합을 학습·채점 전에 고정했습니다. 기존 단계의 통계 파일은 보존했고 이번 통합 결과에만 60개 집합 보정을 적용했습니다. DM은 제곱오차·Bartlett HAC lag=h/3, Bootstrap은 원래 3시간 달력 격자의 7일(56슬롯) paired 이동 블록 1,000회·seed0입니다. 선택에 사용한 2023년의 탐색적 검증 결과이며 최종 Test 성능을 뜻하지 않습니다.

전체 24개 모델×Horizon 성능과 60개 검정은 별도 CSV에 있습니다. M4→M4T 비교는 M4T 미실행으로 아직 없습니다.

## 조건별 분석

계절(cold=11~3월, warm=4~10월), 궤적 방향, ASOS 풍향, 정답 고농도≥35의 n/MAE/RMSE/MBE/Persistence 대비 개선률을 동일 공통 채점기로 계산했습니다. 작은 표본 범주도 삭제하지 않았습니다. 전체 값은 `{manifest['common_scores']}/score_conditions_validation.csv`에 있습니다. 조건별 별도 유의성 또는 인과 효과를 주장하지 않습니다.

주 Horizon인 6시간의 조건별 상세 값:

{table(cond6.round(5))}

## 생성 파일·보존·중단

- 공식 결과: `results/validation_2023/M3T/`; 모델: `Models/M3T/h3`, `h6`, `h12`, `h24`.
- 설정: `configs/m3t_3h.json`, `m3t_6h.json`, `m3t_12h.json`, `m3t_24h.json`; 조합 정의: `configs/feature_groups.json`의 M3T.
- 각 Horizon: 모델 시드3개, 설정·Feature 목록·선택 결과·예측 CSV/Parquet·공통 5열 예측과 메타데이터·진단 그림.
- `predictions/`: 기존 M1~M4의 동일 사본과 신규 M3T 11,680행 예측 및 메타데이터.
- `M1_M2_M3_M4_M3T_Persistence_2023_validation_metrics.csv`: 전체 지표.
- `M3T_vs_M1_M2_M3_M4_2023_validation_statistics.csv`: 직접 비교 차이·CI·DM·Holm·판정.
- `M1_M2_M3_M4_M3T_2023_validation_tests.csv`: 전체 60개 검정.
- `M1_M2_M3_M4_M3T_2023_validation_intermediate_rmse_improvement.png`: x=3/6/12/24h, y=Persistence 대비 RMSE 개선률, 모델별 선, 7일 Bootstrap 95% CI. **2023 Validation 중간 그림**으로 명시.
- 공통 채점 폴더: `{manifest['common_scores']}`. `score_rows_validation.parquet`와 `.meta.json`에 예측·정답 시점, Horizon, 정답, 모델별 예측, 조건, 데이터 버전·시드·설정을 저장.
- `preflight.json`, `scope_before_training.json`, `verification.json`, `manifest.json`, `horizon_settings.json`, `run.log`, `file_inventory.csv`: 사전 검증·범위 고정·보존·재현성 기록.

M4T, TreeSHAP, Permutation Importance, Adaptive 및 2024 최종 Test를 수행하지 않았습니다. 시험 잠금은 미소비 상태입니다. M3T 실행은 기존 고정 경로가 있으므로 재실행 시 덮어쓰지 않고 중단합니다.

현재 단계의 작업을 완료했습니다. 결과를 검토한 뒤 승인해 주시면 다음 단계로 진행하겠습니다.
'''
 (OUT/'REPORT.md').write_text(text)
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='file_inventory.csv']+list((ROOT/'Models/M3T').rglob('*'))+[ROOT/f'configs/m3t_{h}h.json' for h in [3,6,12,24]]+[ROOT/p for p in ['configs/feature_groups.json','src/train.py','src/feature_groups.py','scripts/run_m3t.py','scripts/report_m3t.py']]
 with (OUT/'file_inventory.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);writer.writeheader()
  for p in sorted(files):
   if p.is_file():writer.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 print(mainperf.to_string(index=False));print(comparisons(direct).to_string(index=False));print('REPORT',OUT/'REPORT.md')
if __name__=='__main__':main()
