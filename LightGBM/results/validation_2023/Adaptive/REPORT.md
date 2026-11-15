# Stage 8 — Adaptive-풍향·Adaptive-경로 구축 완료

**2023년 가중치 추정 표본 내 참고 결과**입니다. M1/M3의 기존 시드3개 앙상블 저장 예측만 사용했습니다. 기본 LightGBM 재학습·재추론은 없고 2024 Test도 수행하지 않았습니다. 최종 개선 판정이나 독립 표본 성능으로 해석하지 않습니다.

## 1. 고정 설계·정답·행

`y_adaptive=(1-w)*y_M1+w*y_M3`, 원 단위 µg/m³에서 `w=clip(Σ(y−M1)(M3−M1)/Σ(M3−M1)²,0,1)`입니다. 시드별 log 출력이나 SHAP 값을 혼합하지 않았습니다. 기존 M1/M3 원 단위 앙상블 예측을 결합하고 최종 채점 때만 음수0 절단합니다.

각 Horizon에서 Stage6의 동일 공통 평가 행을 사용해 가중치를 추정하고 참고 평가했습니다. 3/6/12/24h 각각 2509/2504/2497/2486행이며 정답·정답 시점 모두2023입니다. eligible_h를 재확인했고 기존 예측과 조건 분류도 일치합니다. 평가 행은 변경하지 않았습니다. 가중치 추정 뒤에는 각 모델에 대해 2023 전체2920개 origin×4 Horizon=11680개 참고 예측을 저장했습니다. 검증 경계에서 t+h가2024가 되는 origin의 **예측만** 포함하며 그2024 정답은 열거나 채점하지 않았습니다.

계절 cold=11~3월/warm=4~10월입니다. 풍향은 t의 서울 ASOS108 서/북/동/남/정온, 경로는 기존12시간 전 위치 그룹 west/north/east/local입니다. 기존 경로 매핑(NW/W/SW→west, N/NE→north, E/SE/S→east, LOCAL→local)을 유지했습니다. 경로 local은 정온과 다른 정의입니다.

## 2. 승인된 분모 기준·fallback

사용자가 이번 단계에서 승인한 정의:

`epsilon_h=max(1e-6, median_2023_common(|M3−M1|)*1e-4)`

선택된 추정 풀의 `D=Σ(M3−M1)² <= n_estimation_pool*epsilon_h²`이면 **w=0.5**입니다. 분모0·빈 풀도 같은 예외로 처리합니다. epsilon의 단위는 µg/m³, D·임계값 단위는(µg/m³)²입니다. 중앙값의0.01%에 해당하는 차이만 퇴화로 보는 보호 규칙이며, 통계적으로 최적인 임계값을 찾았다는 주장은 하지 않습니다. 이 기준의 보정은 예측값 차이만 사용하고 정답·2024를 사용하지 않았습니다.

| horizon | epsilon_ug_m3 |
|---|---|
| 3 | 4.43218736064e-05 |
| 6 | 7.86562766978e-05 |
| 12 | 0.000136737334629 |
| 24 | 0.000188885033058 |

상황별 n<100이면 해당 Horizon·계절 풀의 가중치를 사용합니다. **분모 예외는 실제 선택된 풀에 적용**합니다. 작은 상황 자체의 분모가 작더라도 계절 fallback을 선택했다면 계절 분모·임계값으로 판단합니다. 분류 불가 풍향은 추가 상황으로 학습하지 않고 계절 가중치로 처리합니다. 일반 상황 가중치는 총72개(풍향40·경로32), 공통 계절 가중치는8개이며, 중복 제거한 실제 추정 풀은56개입니다.

실제 분모 예외는 **56개 풀 중0개·예측 적용0행**입니다. 실제 데이터에서 발생하지 않은 예외도 합성 자료로 분모0/작은분모/임계값 경계/빈 풀/계절 fallback을 검증했습니다. n99/n100 경계, w0~1 제한, 정답 없이 가중치 적용도 검증했습니다.

각 표에는 상황 n, 상황 분자·분모, 실제 풀 ID/n, 실제 분자·분모, epsilon·임계값, 계절 fallback 여부·사유, 분모 예외 여부·사유, 제한 전후 w를 기록했습니다. 원래의 과거 결정 파일·공통 규칙은 보존하고 새 승인값은 `configs/adaptive_validation_2023.json`에 고정했습니다.

### 계절 가중치

| horizon | season | n_estimation_pool | final_weight | weight_clipped |
|---|---|---|---|---|
| 3 | cold | 840 | 0.621973 | False |
| 3 | warm | 1669 | 0.512609 | False |
| 6 | cold | 837 | 1.0 | True |
| 6 | warm | 1667 | 0.866284 | False |
| 12 | cold | 833 | 0.949544 | False |
| 12 | warm | 1664 | 0.486938 | False |
| 24 | cold | 826 | 1.0 | True |
| 24 | warm | 1660 | 0.586435 | False |

### 6시간 상황별 가중치

| kind | season | condition | n_cell | final_weight | season_fallback_used | estimation_pool |
|---|---|---|---|---|---|---|
| wind | cold | west | 393 | 1.0 | False | h6:wind:cold:west |
| wind | cold | north | 134 | 0.574419 | False | h6:wind:cold:north |
| wind | cold | east | 192 | 0.751974 | False | h6:wind:cold:east |
| wind | cold | south | 85 | 1.0 | True | h6:season:cold |
| wind | cold | calm | 21 | 1.0 | True | h6:season:cold |
| route | cold | west | 636 | 1.0 | False | h6:route:cold:west |
| route | cold | north | 67 | 1.0 | True | h6:season:cold |
| route | cold | east | 129 | 1.0 | False | h6:route:cold:east |
| route | cold | local | 5 | 1.0 | True | h6:season:cold |
| wind | warm | west | 641 | 1.0 | False | h6:wind:warm:west |
| wind | warm | north | 209 | 0.255395 | False | h6:wind:warm:north |
| wind | warm | east | 513 | 0.219342 | False | h6:wind:warm:east |
| wind | warm | south | 256 | 1.0 | False | h6:wind:warm:south |
| wind | warm | calm | 48 | 0.866284 | True | h6:season:warm |
| route | warm | west | 889 | 1.0 | False | h6:route:warm:west |
| route | warm | north | 241 | 0.0 | False | h6:route:warm:north |
| route | warm | east | 512 | 0.24221 | False | h6:route:warm:east |
| route | warm | local | 25 | 0.866284 | True | h6:season:warm |

6시간 한랭기 계절 w=1이며 온난기 약0.866입니다. 온난기 풍향 north/east와 경로 north/east에서는 더 작은 M3 가중치가 추정됩니다. 이는2023 표본에 맞춘 가중치이며 실제 수송의 인과적 기여율로 해석하지 않습니다. 모든 Horizon의 상황별 값은 condition_weights.csv에 보존했습니다.

### 공통 평가 행의 fallback·예외 적용 건수

| model | horizon | n_predictions | small_cell_fallback_rows | unknown_condition_fallback_rows | total_season_fallback_rows | near_zero_exception_prediction_rows |
|---|---|---|---|---|---|---|
| Adaptive_route | 3 | 2509 | 98 | 0 | 98 | 0 |
| Adaptive_route | 6 | 2504 | 97 | 0 | 97 | 0 |
| Adaptive_route | 12 | 2497 | 97 | 0 | 97 | 0 |
| Adaptive_route | 24 | 2486 | 95 | 0 | 95 | 0 |
| Adaptive_wind | 3 | 2509 | 154 | 12 | 166 | 0 |
| Adaptive_wind | 6 | 2504 | 154 | 12 | 166 | 0 |
| Adaptive_wind | 12 | 2497 | 154 | 12 | 166 | 0 |
| Adaptive_wind | 24 | 2486 | 154 | 12 | 166 | 0 |

풍향은 각 Horizon에서 표본 부족154행+분류 불가12행=166행이 계절 가중치를 사용합니다. 경로는98/97/97/95행입니다. 전체11680 origin의 적용 건수도 fallback_exception_counts.csv에서 공통 평가 행과 구분했습니다. 동일한 계절 풀을 여러 상황이 참조하더라도 estimation_pools.csv에서는 한 번만 세어 중복 예외 건수를 방지했습니다. 0~1 제한은 분모 예외와 구분해 기록했습니다.

## 3. 2023 표본 내 참고 성능

MAE/RMSE/MBE 단위는 µg/m³, rmse_improvement_pct는 Persistence 대비 %입니다. 같은 공통 평가 행의 공통 metrics 함수를 사용했습니다.

| horizon | model | n | RMSE | MAE | MBE | rmse_improvement_pct | negative_predictions_clipped |
|---|---|---|---|---|---|---|---|
| 3 | M1 | 2509 | 5.97262 | 3.82482 | -0.47093 | 16.7654 | 0 |
| 3 | M3 | 2509 | 5.95903 | 3.86895 | -0.45547 | 16.95472 | 0 |
| 3 | Adaptive_wind | 2509 | 5.92346 | 3.82624 | -0.48162 | 17.45043 | 0 |
| 3 | Adaptive_route | 2509 | 5.93123 | 3.8272 | -0.49139 | 17.34216 | 0 |
| 6 | M1 | 2504 | 8.33591 | 5.19522 | -1.09391 | 16.57255 | 0 |
| 6 | M3 | 2504 | 8.12427 | 5.21335 | -0.95307 | 18.6907 | 0 |
| 6 | Adaptive_wind | 2504 | 8.10466 | 5.18187 | -0.97888 | 18.88696 | 0 |
| 6 | Adaptive_route | 2504 | 8.10819 | 5.19721 | -0.97258 | 18.85159 | 0 |
| 12 | M1 | 2497 | 10.67499 | 6.7116 | -1.93478 | 18.73106 | 0 |
| 12 | M3 | 2497 | 10.29677 | 6.66598 | -1.29804 | 21.61041 | 0 |
| 12 | Adaptive_wind | 2497 | 10.2331 | 6.59224 | -1.32198 | 22.09513 | 0 |
| 12 | Adaptive_route | 2497 | 10.21578 | 6.57196 | -1.45511 | 22.22698 | 0 |
| 24 | M1 | 2486 | 12.8561 | 8.13256 | -2.78946 | 15.61839 | 0 |
| 24 | M3 | 2486 | 11.95863 | 7.78613 | -1.50051 | 21.50896 | 0 |
| 24 | Adaptive_wind | 2486 | 11.89458 | 7.71372 | -1.66129 | 21.92939 | 0 |
| 24 | Adaptive_route | 2486 | 11.89886 | 7.71074 | -1.66757 | 21.90128 | 0 |

전체8모델+Persistence의 지표는 score_metrics_validation_reference.csv, 계절/경로 방향/ASOS풍향/고농도≥35 조건별 지표는 score_conditions_validation_reference.csv에 있습니다. 기존 모델의 예측·조건·정답을 그대로 사용했고 M1/M3와의 참고 RMSE 차이는 adaptive_vs_M1_M3_reference_differences.csv에 저장했습니다.

**가중치 추정과 평가에 같은2023 정답을 사용했습니다.** 새 Adaptive 결과에 DM/Holm p-value, Bootstrap 개선 신뢰구간, 공식 improved 판정을 부여하지 않았습니다. formal_improvement_assessed=false로 기록했습니다. 모델별 RMSE 감소를 독립 검증의 유의한 개선으로 주장하지 않으며 최종 비교는 별도로 승인될2024 Test에서만 가능합니다. 기존 공식 모델 성능·검정 파일은 변경하지 않았습니다.

## 4. 저장·재현·기존 산출물 보존

- 공식 결과: `results/validation_2023/Adaptive/`.
- 적용 코드: `src/adaptive.py`; 실행·검증: scripts/run_adaptive.py / check_adaptive.py / check_adaptive_artifacts.py.
- `Models/Adaptive_wind/h{3,6,12,24}/weights.json`, `Models/Adaptive_route/h{3,6,12,24}/weights.json`: 고정된 계절·상황 가중치 및 원 예측 출처. 각 폴더 config.json도 저장. LightGBM 모델을 새로 만들지 않았습니다.
- `season_weights.csv`, `condition_weights.csv`, `estimation_pools.csv`, `prediction_difference_distribution.csv`: 계절·상황·고유 풀·임계값 보정 분포.
- `prediction_details_validation.parquet`: 각 origin/Horizon/모델의 M1/M3·상황·실제 w·풀·fallback/예외·원 단위 예측. 정답 없이도 재적용 가능.
- `predictions/Adaptive_wind.parquet`, `Adaptive_route.parquet`와.meta.json: 공통5열 출력. 각11680행, 데이터 버전·시드·설정·가중치 체크섬·표본 내 평가 표시 포함.
- `score_rows_validation_reference.parquet`와.meta.json: 기존9996개 공통 평가 행·예측 시점·정답 시점·정답·조건·기존 및 Adaptive 예측. 가중치 세부 적용 기록은 prediction_details_validation.parquet로 연결.
- `fallback_exception_counts.csv`, `exception_pool_counts.csv`: 전체origin/공통평가 행을 구분한 예외·fallback·가중치제한·음수절단 건수.
- `figures/weights_wind_validation.png`, `weights_route_validation.png`, `rmse_validation_in_sample_reference.png`: 가중치·표본 내 참고 성능. 별표는 계절 fallback이고 n은 실제 가중치 추정용 상황 표본 수입니다.
- preflight.json / verification.json / artifact_roundtrip_verification.json / manifest.json / report_manifest.json / file_inventory.csv / run.log: 무결성·가중치 재로드·코드/데이터 버전·산출물 추적.

기존 관련 파일 **786개** 및 원본 입력 **39개**의 SHA256 불변을 확인했습니다. M1~M4·M3T/M4T 모델·기존 예측·공통 채점 결과와 Stage7의 SHAP 원시값·보고서·그림을 모두 보존했습니다. 고정된 가중치JSON8개를 재로드해 정답 없는 적용 함수로 전체23360개 예측·fallback을 재현했고 최대 차이는0입니다. 가중치 각 실제 추정 풀에서 제한된 최소제곱 해가 M1/M3 끝점보다 SSE가 크지 않음을 검증했습니다. 상황별 fallback 성능이 해당 소표본의 최적값임을 주장하는 검사는 아닙니다.

## 5. 중단 상태

2024 Test 잠금은 미소비 상태입니다. 원래 LightGBM 재학습과 Stage7의 관측소별·거리별·기상 조건별 추가 SHAP/Permutation 분석은 하지 않았습니다. 최종 Test는 이번에 실행하지 않았으며, 별도 승인을 받아야 합니다. 저장된 가중치를 2024 정답으로 다시 맞추지 않는 적용 구조를 준비했습니다. 현재 완료된 고정 경로는 재실행 시 덮어쓰지 않고 중단합니다.

현재 단계의 작업을 완료했습니다. 결과를 검토한 뒤 승인해 주시면 다음 단계로 진행하겠습니다.
