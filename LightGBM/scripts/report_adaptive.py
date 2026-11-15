"""Stage8 descriptive report/figures. No new weights, labels, tests or inference."""
from pathlib import Path
import os,sys,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common_scoring import sha,write_json
from scripts.run_m2 import protect
OUT=ROOT/'results/validation_2023/Adaptive';HS=[3,6,12,24]
def table(d):
 return '| '+' | '.join(map(str,d.columns))+' |\n|'+'|'.join(['---']*len(d.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in d.itertuples(index=False,name=None))
def main():
 m=json.loads((OUT/'manifest.json').read_text());assert m['status']=='completed';pre=json.loads((OUT/'preflight.json').read_text());protect(pre['protected_sha256']);protect(pre['original_inputs_sha256'])
 config=json.loads((OUT/'config.json').read_text());metrics=pd.read_csv(OUT/'score_metrics_validation_reference.csv');cells=pd.read_csv(OUT/'condition_weights.csv');season=pd.read_csv(OUT/'season_weights.csv');counts=pd.read_csv(OUT/'fallback_exception_counts.csv');pools=pd.read_csv(OUT/'estimation_pools.csv');verify=json.loads((OUT/'verification.json').read_text());roundtrip=json.loads((OUT/'artifact_roundtrip_verification.json').read_text())
 assert len(cells)==72 and len(season)==8 and len(pools)==56 and roundtrip['status']=='passed' and not (ROOT/'results/final_test_2024.lock').exists()
 plt.rcParams['font.family']=['Apple SD Gothic Neo','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 (OUT/'figures').mkdir(exist_ok=False)
 for kind,conditions in [('wind',['west','north','east','south','calm']),('route',['west','north','east','local'])]:
  fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
  for ax,h in zip(axes.ravel(),HS):
   sub=cells[(cells.kind==kind)&(cells.horizon==h)].set_index(['season','condition']);a=np.array([[sub.loc[(s,c),'final_weight'] for c in conditions] for s in ['cold','warm']]);image=ax.imshow(a,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(len(conditions)),conditions);ax.set_yticks([0,1],['cold (11~3월)','warm (4~10월)']);ax.set_title(f'{h}시간')
   for i,s in enumerate(['cold','warm']):
    for j,c in enumerate(conditions):
     row=sub.loc[(s,c)];star=' *' if row.season_fallback_used else '';ax.text(j,i,f'{row.final_weight:.3f}{star}\nn={row.n_cell}',ha='center',va='center',color='white' if row.final_weight<.5 else 'black',fontsize=10)
  fig.colorbar(image,ax=list(axes.ravel()),label='w: 0=M1, 1=M3');fig.suptitle(f'2023 Validation 추정 가중치 · Adaptive-{kind}\n* 표본100개 미만 → 계절 가중치; 풍향 unknown은 별도 계절 fallback');fig.savefig(OUT/f'figures/weights_{kind}_validation.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,5))
 for model in ['M1','M3','Adaptive_wind','Adaptive_route']:
  d=metrics[metrics.model.eq(model)].sort_values('horizon');ax.plot(d.horizon,d.RMSE,marker='o',label=model)
 ax.set_xticks(HS);ax.set_xlabel('예측 시간 (h)');ax.set_ylabel('RMSE (µg/m³)');ax.legend();ax.set_title('2023 Validation · Adaptive 표본 내 참고 성능\n가중치 추정에 같은 정답 사용 · 최종 개선 판정 아님');fig.tight_layout();fig.savefig(OUT/'figures/rmse_validation_in_sample_reference.png',dpi=180);plt.close(fig)
 performance=metrics[metrics.model.isin(['M1','M3','Adaptive_wind','Adaptive_route'])][['horizon','model','n','RMSE','MAE','MBE','rmse_improvement_pct','negative_predictions_clipped']].copy()
 for col in ['RMSE','MAE','MBE','rmse_improvement_pct']:performance[col]=performance[col].round(5)
 h6=cells[cells.horizon.eq(6)][['kind','season','condition','n_cell','final_weight','season_fallback_used','estimation_pool']].copy();h6.final_weight=h6.final_weight.round(6)
 cs=counts[counts.scope.eq('common_evaluation_rows')][['model','horizon','n_predictions','small_cell_fallback_rows','unknown_condition_fallback_rows','total_season_fallback_rows','near_zero_exception_prediction_rows']]
 eps=pd.DataFrame([{'horizon':h,'epsilon_ug_m3':config['epsilon_by_horizon'][str(h)]} for h in HS]);ep=eps.copy();ep.epsilon_ug_m3=ep.epsilon_ug_m3.map(lambda x:f'{x:.12g}')
 s=season[['horizon','season','n_estimation_pool','final_weight','weight_clipped']].copy();s.final_weight=s.final_weight.round(6)
 reference_comparison=metrics.pivot(index='horizon',columns='model',values='RMSE')
 delta=[]
 for h in HS:
  for name in ['Adaptive_wind','Adaptive_route']:delta.append({'horizon':h,'model':name,'RMSE_minus_M1':float(reference_comparison.loc[h,name]-reference_comparison.loc[h,'M1']),'RMSE_minus_M3':float(reference_comparison.loc[h,name]-reference_comparison.loc[h,'M3']),'evaluation_type':'in_sample_reference','formal_improvement_assessed':False})
 pd.DataFrame(delta).to_csv(OUT/'adaptive_vs_M1_M3_reference_differences.csv',index=False)
 text=f'''# Stage 8 — Adaptive-풍향·Adaptive-경로 구축 완료

**2023년 가중치 추정 표본 내 참고 결과**입니다. M1/M3의 기존 시드3개 앙상블 저장 예측만 사용했습니다. 기본 LightGBM 재학습·재추론은 없고 2024 Test도 수행하지 않았습니다. 최종 개선 판정이나 독립 표본 성능으로 해석하지 않습니다.

## 1. 고정 설계·정답·행

`y_adaptive=(1-w)*y_M1+w*y_M3`, 원 단위 µg/m³에서 `w=clip(Σ(y−M1)(M3−M1)/Σ(M3−M1)²,0,1)`입니다. 시드별 log 출력이나 SHAP 값을 혼합하지 않았습니다. 기존 M1/M3 원 단위 앙상블 예측을 결합하고 최종 채점 때만 음수0 절단합니다.

각 Horizon에서 Stage6의 동일 공통 평가 행을 사용해 가중치를 추정하고 참고 평가했습니다. 3/6/12/24h 각각 2509/2504/2497/2486행이며 정답·정답 시점 모두2023입니다. eligible_h를 재확인했고 기존 예측과 조건 분류도 일치합니다. 평가 행은 변경하지 않았습니다. 가중치 추정 뒤에는 각 모델에 대해 2023 전체2920개 origin×4 Horizon=11680개 참고 예측을 저장했습니다. 검증 경계에서 t+h가2024가 되는 origin의 **예측만** 포함하며 그2024 정답은 열거나 채점하지 않았습니다.

계절 cold=11~3월/warm=4~10월입니다. 풍향은 t의 서울 ASOS108 서/북/동/남/정온, 경로는 기존12시간 전 위치 그룹 west/north/east/local입니다. 기존 경로 매핑(NW/W/SW→west, N/NE→north, E/SE/S→east, LOCAL→local)을 유지했습니다. 경로 local은 정온과 다른 정의입니다.

## 2. 승인된 분모 기준·fallback

사용자가 이번 단계에서 승인한 정의:

`epsilon_h=max(1e-6, median_2023_common(|M3−M1|)*1e-4)`

선택된 추정 풀의 `D=Σ(M3−M1)² <= n_estimation_pool*epsilon_h²`이면 **w=0.5**입니다. 분모0·빈 풀도 같은 예외로 처리합니다. epsilon의 단위는 µg/m³, D·임계값 단위는(µg/m³)²입니다. 중앙값의0.01%에 해당하는 차이만 퇴화로 보는 보호 규칙이며, 통계적으로 최적인 임계값을 찾았다는 주장은 하지 않습니다. 이 기준의 보정은 예측값 차이만 사용하고 정답·2024를 사용하지 않았습니다.

{table(ep)}

상황별 n<100이면 해당 Horizon·계절 풀의 가중치를 사용합니다. **분모 예외는 실제 선택된 풀에 적용**합니다. 작은 상황 자체의 분모가 작더라도 계절 fallback을 선택했다면 계절 분모·임계값으로 판단합니다. 분류 불가 풍향은 추가 상황으로 학습하지 않고 계절 가중치로 처리합니다. 일반 상황 가중치는 총72개(풍향40·경로32), 공통 계절 가중치는8개이며, 중복 제거한 실제 추정 풀은56개입니다.

실제 분모 예외는 **56개 풀 중0개·예측 적용0행**입니다. 실제 데이터에서 발생하지 않은 예외도 합성 자료로 분모0/작은분모/임계값 경계/빈 풀/계절 fallback을 검증했습니다. n99/n100 경계, w0~1 제한, 정답 없이 가중치 적용도 검증했습니다.

각 표에는 상황 n, 상황 분자·분모, 실제 풀 ID/n, 실제 분자·분모, epsilon·임계값, 계절 fallback 여부·사유, 분모 예외 여부·사유, 제한 전후 w를 기록했습니다. 원래의 과거 결정 파일·공통 규칙은 보존하고 새 승인값은 `configs/adaptive_validation_2023.json`에 고정했습니다.

### 계절 가중치

{table(s)}

### 6시간 상황별 가중치

{table(h6)}

6시간 한랭기 계절 w=1이며 온난기 약0.866입니다. 온난기 풍향 north/east와 경로 north/east에서는 더 작은 M3 가중치가 추정됩니다. 이는2023 표본에 맞춘 가중치이며 실제 수송의 인과적 기여율로 해석하지 않습니다. 모든 Horizon의 상황별 값은 condition_weights.csv에 보존했습니다.

### 공통 평가 행의 fallback·예외 적용 건수

{table(cs)}

풍향은 각 Horizon에서 표본 부족154행+분류 불가12행=166행이 계절 가중치를 사용합니다. 경로는98/97/97/95행입니다. 전체11680 origin의 적용 건수도 fallback_exception_counts.csv에서 공통 평가 행과 구분했습니다. 동일한 계절 풀을 여러 상황이 참조하더라도 estimation_pools.csv에서는 한 번만 세어 중복 예외 건수를 방지했습니다. 0~1 제한은 분모 예외와 구분해 기록했습니다.

## 3. 2023 표본 내 참고 성능

MAE/RMSE/MBE 단위는 µg/m³, rmse_improvement_pct는 Persistence 대비 %입니다. 같은 공통 평가 행의 공통 metrics 함수를 사용했습니다.

{table(performance)}

전체8모델+Persistence의 지표는 score_metrics_validation_reference.csv, 계절/경로 방향/ASOS풍향/고농도≥35 조건별 지표는 score_conditions_validation_reference.csv에 있습니다. 기존 모델의 예측·조건·정답을 그대로 사용했고 M1/M3와의 참고 RMSE 차이는 adaptive_vs_M1_M3_reference_differences.csv에 저장했습니다.

**가중치 추정과 평가에 같은2023 정답을 사용했습니다.** 새 Adaptive 결과에 DM/Holm p-value, Bootstrap 개선 신뢰구간, 공식 improved 판정을 부여하지 않았습니다. formal_improvement_assessed=false로 기록했습니다. 모델별 RMSE 감소를 독립 검증의 유의한 개선으로 주장하지 않으며 최종 비교는 별도로 승인될2024 Test에서만 가능합니다. 기존 공식 모델 성능·검정 파일은 변경하지 않았습니다.

## 4. 저장·재현·기존 산출물 보존

- 공식 결과: `results/validation_2023/Adaptive/`.
- 적용 코드: `src/adaptive.py`; 실행·검증: scripts/run_adaptive.py / check_adaptive.py / check_adaptive_artifacts.py.
- `Models/Adaptive_wind/h{{3,6,12,24}}/weights.json`, `Models/Adaptive_route/h{{3,6,12,24}}/weights.json`: 고정된 계절·상황 가중치 및 원 예측 출처. 각 폴더 config.json도 저장. LightGBM 모델을 새로 만들지 않았습니다.
- `season_weights.csv`, `condition_weights.csv`, `estimation_pools.csv`, `prediction_difference_distribution.csv`: 계절·상황·고유 풀·임계값 보정 분포.
- `prediction_details_validation.parquet`: 각 origin/Horizon/모델의 M1/M3·상황·실제 w·풀·fallback/예외·원 단위 예측. 정답 없이도 재적용 가능.
- `predictions/Adaptive_wind.parquet`, `Adaptive_route.parquet`와.meta.json: 공통5열 출력. 각11680행, 데이터 버전·시드·설정·가중치 체크섬·표본 내 평가 표시 포함.
- `score_rows_validation_reference.parquet`와.meta.json: 기존9996개 공통 평가 행·예측 시점·정답 시점·정답·조건·기존 및 Adaptive 예측. 가중치 세부 적용 기록은 prediction_details_validation.parquet로 연결.
- `fallback_exception_counts.csv`, `exception_pool_counts.csv`: 전체origin/공통평가 행을 구분한 예외·fallback·가중치제한·음수절단 건수.
- `figures/weights_wind_validation.png`, `weights_route_validation.png`, `rmse_validation_in_sample_reference.png`: 가중치·표본 내 참고 성능. 별표는 계절 fallback이고 n은 실제 가중치 추정용 상황 표본 수입니다.
- preflight.json / verification.json / artifact_roundtrip_verification.json / manifest.json / report_manifest.json / file_inventory.csv / run.log: 무결성·가중치 재로드·코드/데이터 버전·산출물 추적.

기존 관련 파일 **{verify['existing_files_preserved']}개** 및 원본 입력 **{verify['original_inputs_preserved']}개**의 SHA256 불변을 확인했습니다. M1~M4·M3T/M4T 모델·기존 예측·공통 채점 결과와 Stage7의 SHAP 원시값·보고서·그림을 모두 보존했습니다. 고정된 가중치JSON8개를 재로드해 정답 없는 적용 함수로 전체23360개 예측·fallback을 재현했고 최대 차이는0입니다. 가중치 각 실제 추정 풀에서 제한된 최소제곱 해가 M1/M3 끝점보다 SSE가 크지 않음을 검증했습니다. 상황별 fallback 성능이 해당 소표본의 최적값임을 주장하는 검사는 아닙니다.

## 5. 중단 상태

2024 Test 잠금은 미소비 상태입니다. 원래 LightGBM 재학습과 Stage7의 관측소별·거리별·기상 조건별 추가 SHAP/Permutation 분석은 하지 않았습니다. 최종 Test는 이번에 실행하지 않았으며, 별도 승인을 받아야 합니다. 저장된 가중치를 2024 정답으로 다시 맞추지 않는 적용 구조를 준비했습니다. 현재 완료된 고정 경로는 재실행 시 덮어쓰지 않고 중단합니다.

현재 단계의 작업을 완료했습니다. 결과를 검토한 뒤 승인해 주시면 다음 단계로 진행하겠습니다.
'''
 (OUT/'REPORT.md').write_text(text)
 write_json(OUT/'report_manifest.json',{'status':'completed','source_manifest_sha256':sha(OUT/'manifest.json'),'report_code_sha256':sha(Path(__file__)),'artifact_check_code_sha256':sha(ROOT/'scripts/check_adaptive_artifacts.py'),'evaluation_type':'in_sample_reference','formal_improvement_assessed':False,'test_executed':False})
 protect(pre['protected_sha256']);protect(pre['original_inputs_sha256'])
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='file_inventory.csv']+[p for name in ['Adaptive_wind','Adaptive_route'] for p in (ROOT/f'Models/{name}').rglob('*') if p.is_file()]+[ROOT/'configs/adaptive_validation_2023.json',ROOT/'src/adaptive.py']+[ROOT/'scripts'/n for n in ['run_adaptive.py','report_adaptive.py','check_adaptive.py','check_adaptive_artifacts.py']]
 with (OUT/'file_inventory.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
  for p in sorted(files):w.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 print('Stage8 보고 완료:',OUT/'REPORT.md');print(performance.to_string(index=False))
if __name__=='__main__':main()
