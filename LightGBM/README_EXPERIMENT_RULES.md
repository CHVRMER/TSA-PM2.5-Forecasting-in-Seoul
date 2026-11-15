# 공통 실험 규칙 및 공식 결과 경로

확정일: 2026-11-08. 기계 판독 설정은 `configs/experiment_rules.json`, 관측소 조합은 `configs/feature_groups.json`에 있습니다. 원본 Dataset과 기존 M1 공식 결과는 이동·변경하지 않습니다.

## 데이터·학습

- 중구 111121 PM2.5의 정확한 t+h 실측값. h=3/6/12/24, 6시간이 주 분석.
- v3의 KST 3시간 간격 11,688개 예측 시점.
- 2021–22 학습 / 2023 조기 종료·설정·지연 선택·Adaptive 가중치 / 2024 프로젝트 최종 시험 1회.
- **2023년을 합쳐 재학습하지 않습니다.** 현재 M2 단계에서 Test를 읽거나 예측·채점하지 않습니다.
- log1p 학습 → 시드별 expm1 복원 → 원 단위 예측 평균. 채점 시 음수 0 절단 및 건수 기록.
- 정답 결측 미대체, Horizon별 eligible_h 적용. LightGBM 입력 NaN 유지.
- 신경망(향후): 0+결측 채널, 정규화는 2021–22. SARIMAX(다른 담당 모델): 과거로만 최대 24시간 채운 뒤 2021–22 중앙값.
- t 이하 관측과 t까지 도착한 궤적, 달력만 사용. 최종확정 관측·GDAS 공개 지연 미재현은 승인된 과거 데이터 비교 연구의 한계로 명시합니다.

## Feature 정의

모든 입력은 원본 feature_sets.json의 명시적 목록 또는 해당 목록의 명시적 부분집합·합집합을 사용하며 role=feature 검사와 중복 검사를 거칩니다. 원본 feature_sets.json은 수정하지 않습니다.

| 모델 | 승인된 실제 입력 수 | 범위 |
|---|---:|---|
| M1 | 1,142 | 서울·ASOS·달력 70 + 국내 67곳 × 16 |
| M2 | 1,478 | M1 + 600km 이내 중국 21곳 × 16 |
| M3 | 2,118 | M1 + 중국 61곳(실측 48 + 국경 대리 도시 평균 13) |
| M4 | 2,518 | M3 + 일본 균형판 24곳 + 울란바토르 1곳 |
| M3T | 2,424 | 기존 승인된 경로 합집합. all 경로 집계에 일본·몽골 포함 허용 |
| M4T | 2,924 | 기존 2,952에서 학습 기간 전체 결측 경로 특징 28개 제외 |

M3T의 과거 예상치 2,594나 M4T의 원래 2,952에 억지로 맞추지 않습니다. 위 수는 앞서 승인된 결정과 실제 데이터 정의를 따릅니다. M3 이후는 개념 정의만 기록했으며 이 단계에서 구현·학습하지 않습니다.

## LightGBM 공통 절차

`src/train.py`의 `train_lightgbm` 하나를 재사용합니다. 기존 M1 호출 이름 train_m1은 같은 함수의 호환 별칭입니다.

- num_leaves={15,31}, feature_fraction={0.3,0.5}의 네 조합.
- learning_rate=0.03, min_data_in_leaf=50, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.
- 최대 5,000회, log1p 타깃 L2 기준 조기 종료 인내 200회.
- 시드 0의 2023 원 단위 RMSE로 설정·반복 수 선택.
- 선택된 시드 0 모델을 유지하고 같은 반복 수로 시드 1·2를 2021–22만 사용해 학습한 뒤 평균. M1과 동일합니다.
- CPU 스레드 2, deterministic/force_col_wise 유지.
- Horizon별 모델·예측·설정·meta 분리. 모델 저장·재로드 예측 일치를 확인.

## 공통 채점

`src/common_scoring.py`와 `Models/score.py`를 그대로 재사용합니다. Stage 2에서는 채점기를 변경하지 않았습니다.

- 모델·persistence가 모두 유한한 해당 eligible_h 공통 행에서 RMSE/MAE/MBE·persistence 대비 RMSE 개선률.
- 제곱오차 DM, Bartlett HAC lag=h/3. 실제 3시간 달력 슬롯을 유지하고 누락 시각을 압축하지 않음.
- 달력 7일=56슬롯 이동 블록 paired bootstrap 1,000회, seed 0. 개선률(%) CI와 candidate−reference RMSE 차이(µg/m³) CI를 별도로 저장.
- Holm family는 채점 실행 전 고정한 비교쌍×Horizon. **이번 Stage 2는 M1−Persistence, M2−Persistence, M2−M1 × 4개 Horizon = 12개**입니다.
- 공식 improved = RMSE 차이 95% CI 상한<0 AND Holm p<0.05.
- 계절, 12시간 전 궤적 방향, ASOS 현재 풍향, 고농도≥35 조건별 평가 및 표본 수.
- 2023년 선택 이후 검증 통계이며 최종 시험 성능 입증으로 해석하지 않음.

전체 프로젝트 최종 평가도 기존의 고정된 비교쌍×Horizon 원칙을 따릅니다. 아직 실행하지 않은 모델을 포함한 최종 비교 목록은 최종 설정 동결 시 명시해야 하며, Stage 2가 전체 프로젝트의 새로운 최종 family를 임의로 확정한 것은 아닙니다.

## 공식 저장 구조

| 논리적 역할 | 실제 경로 |
|---|---|
| 기존 M1 전체 Horizon 공식 manifest | `results/M1_allhorizon_validation_20261108_182422/completion.json` |
| 기존 M1 6시간 원본 | `results/M1_h6_20261108_174529_127972/` 및 대응 Models 하위 폴더 |
| 새 M2 설정 | `configs/m2_3h.json`, `m2_6h.json`, `m2_12h.json`, `m2_24h.json` |
| 새 M2 모델 | `Models/M2/h3/`, `h6/`, `h12/`, `h24/` |
| 새 M2 검증 공식 루트 | `results/validation_2023/M2/` |
| M2 개별 결과 | 위 루트의 `h3/`, `h6/`, `h12/`, `h24/` |
| 비교용 공통 예측 | 위 루트의 `predictions/M1.parquet`, `M2.parquet` 및 .meta.json |
| 공통 채점 | 위 루트의 `common_scores/` 아래 기존 채점기가 생성한 한 실행 폴더; manifest에서 정확한 경로 관리 |
| M2 공식 결과 manifest | `results/validation_2023/M2/manifest.json` |
| 최종 시험·SHAP | 향후 승인 후 `results/test_2024/`, `results/shap/` 사용 예정; 이번에는 생성·사용하지 않음 |

Models/models 대소문자 충돌을 피하기 위해 기존 `Models/`를 유지합니다. 기존 결과를 이동하거나 별도 timestamp 루트를 계속 늘리지 않습니다. M2 실행기는 고정 경로의 기존 로그·preflight·모델이 있으면 덮어쓰지 않고 중단합니다. 공통 채점기의 기존 실행 이력 하위 폴더는 그대로 재사용합니다. 전체 정리는 모델 개발 이후 별도 승인 사항입니다.

예측 필수 열은 `origin_time_kst, horizon, model, split, y_hat`입니다. .meta.json에 데이터 버전·시드·설정·Feature 수·학습 기간을 기록하고, score_rows 및 sidecar에 정답 시각·모델별 예측·조건·버전을 남깁니다. 새 비교용 M1 사본은 값 그대로 재사용하며 원래 공식 M1 파일은 보존합니다.

## 시험 잠금 및 후속 범위

2024 시험은 --final, 설정 동결 manifest, 프로젝트 단위 배타적 1회 잠금을 모두 요구합니다. 아직 시험을 실행하지 않았고 실제 잠금도 사용하지 않았습니다.

M2 완료 뒤 승인 전에는 M3·M4·Trajectory 모델·TreeSHAP·Permutation Importance·Adaptive·2024 시험을 실행하지 않습니다. Adaptive는 M1/M3, 100개 미만 계절 fallback, 분모 퇴화 시 w=0.5 규칙을 유지합니다. 작은 분모의 구체적 임계값은 2023 M1/M3 차이 분포를 보고 해당 단계에서 제안·승인받아야 합니다.
