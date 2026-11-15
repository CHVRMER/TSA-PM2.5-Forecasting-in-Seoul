# D — DCRNN·Graph WaveNet 코드와 결과

서울 중구 PM2.5 예측에서 **국외 관측값과 궤적 요약 변수를 추가했을 때 얻는 이득**을 비교한 자료입니다. 학습 코드, 공통 채점기, 결과 분석과 발표용 그림을 함께 정리했습니다.

## 먼저 볼 자료

| 목적 | 위치 |
|---|---|
| DCRNN 핵심 결과와 해석 | [DCRNN 결과 안내](../results/D/DCRNN/README.md) |
| Graph WaveNet 실행 결과·구현 범위 | [Graph WaveNet 결과 안내](../results/D/GraphWaveNet/README.md) |
| 중간발표용 DCRNN 2023 validation 그림 | [presentation2023](../results/D/DCRNN/presentation2023/Results/) |
| 관측망·품질·권역·수송 경로 EDA | [EDA/D_Contributions](../EDA/D_Contributions/) |
| 데이터 연결부터 pilot·full까지 | [RUN_GUIDE.md](RUN_GUIDE.md) |

현재 중간발표의 모델 비교는 **2023 validation**을 중심으로 합니다. 2024 결과도 보관했지만, 개발 중 이미 확인한 **preliminary evaluation**으로 구분합니다. 2023와 2024의 RMSE를 같은 표에서 직접 순위 비교하지 않습니다.

## 공유 코드 구조

```text
D/
  README.md
  RUN_GUIDE.md
  runtime/
    Model/
      Common/
        DCRNN_H48_20261111/
        GraphWaveNet_H48_20261112/
        Neighbor_Experiments_QC_20261108/
        grid.py, neural_evaluation.py, score.py
      DCRNN/
        V3_Data_20261107/Codes/
        Neighbor_Expansion_QC_20261108/Codes/
        VastAI_M1_M4T_H48_20261111/
        Graph_WaveNet/VastAI_M1_M4T_H48_20261112/
    Dataset/          # 별도로 연결하는 데이터; Git 공유 대상 아님

results/D/
  DCRNN/
    README.md
    presentation2023/
    h48_review/
    conditional_review/
    historical_regional12/
  GraphWaveNet/
```

`runtime/Model`은 기존 실행기의 상대 경로와 파일 무결성 검사를 보존하기 위한 구조입니다. 저장소의 다른 팀원 모델 A·B·C와 별도로 사용합니다. 원자료·학습 가중치·개인 Google Drive 인증은 이 코드 공유에 포함하지 않습니다.

## 공통 입력 비교

서울 중구 타깃은 아래 외부 계열 수에서 제외합니다.

| 조건 | 사용할 수 있는 외부 PM2.5 관측 | 추가 궤적 요약 변수 |
|---|---|---|
| M1 | 국내 514 | 없음 |
| M2 | 국내 514 + 서울 600 km 이내 중국 21 | 없음 |
| M3 | 국내 514 + 중국 77 | 없음 |
| M4 | M3 + 일본 42 + 몽골 1 | 없음 |
| M3T | M3와 같은 관측 범위 | 국내·중국 218개 |
| M4T | M4와 같은 관측 범위 | 전체 허용 국가 322개 |

모든 조건은 **같은 635노드와 입력 차원**을 유지합니다. 제외한 관측값은 정규화를 적합하기 전에 가리고, 노드는 잠재 전달 노드로 남깁니다. DCRNN에는 학습구간에서 만든 수송 방향 그래프를, Graph WaveNet에는 같은 노드의 거리 그래프와 학습형 그래프를 사용합니다.

따라서 M1은 **국내 PM2.5 관측 입력 조건**이라는 뜻입니다. DCRNN의 공통 수송 그래프까지 제거한 모델은 아닙니다. M3T·M4T는 그래프 도입 효과가 아니라 **추가 요약 변수의 효과**를 비교합니다.

## 학습·평가 범위

- 학습: 2021–2022년. 정규화도 이 구간에서 적합합니다.
- 설정 선택·early stopping: 2023년의 6h RMSE.
- 예측시간: 3·6·12·24·36·48h. 발표용 주 그림은 3–24h입니다.
- 공유된 최신 DCRNN 결과는 **seed 11 단일 실행**입니다. 3개 시드 평균이나 앙상블 결과가 아닙니다.
- DCRNN은 hidden 64, encoder/decoder 각 2층, diffusion 2를 유지하며 최대 50 epoch·patience 10입니다.
- Graph WaveNet H48 구현은 최대 75 epoch·patience 10이며, 결과의 실행 버전과 실제 실행 여부는 별도 [결과 안내](../results/D/GraphWaveNet/README.md)를 확인합니다.
- 같은 평가 행·기준선을 사용하고, 시드별 성능과 앙상블 성능을 구분합니다. 공통 채점기의 파일명·열 형식은 기존 규칙을 유지합니다.

## 관측소 수를 읽는 기준

최신 H48 실행의 외부 입력 전체는 **국내 514 + 중국 77 + 일본 42 + 몽골 1 = 634계열**이며, 타깃을 포함하면 635노드입니다. 중국 도시평균도 포함되므로 모든 계열을 개별 측정소라고 부르지는 않습니다.

중국 동부 16·일본 동부 18을 추가하기 전 기준은 **F600 = 국내 514 + 중국 61 + 일본 24 + 몽골 1**입니다. EDA의 전체 후보 합집합 650계열, 또는 동부 후보를 제외한 합집합 616계열과 실제 모델의 입력 수를 혼동하지 않아야 합니다. 발표용 관측소 수 그림의 기준과 성능곡선의 실제 실행 기준을 함께 확인하세요.

## 이 결과로 알 수 있는 것과 한계

2023 validation의 3–24h에서 국내 입력만 사용하는 M1은 경쟁력 있는 기준이며, 12·24h의 수치상 최저 RMSE를 보였습니다. 국외 관측과 궤적 요약 변수를 추가한 이득은 일관되지 않았습니다. 이를 곧바로 “국외 정보가 쓸모없다” 또는 “궤적이 잘못됐다”로 해석하지 않습니다.

후속 검증 대상은 입력 수의 불균형, 중복·잡음, 변수 결합 방식, 경로·계절별 선택, 학습 설정입니다. 현재 분석은 예측에 도움이 되는 정보의 출처를 다루며 국가별 배출원 기여율이나 인과적 오염 기여를 추정하지 않습니다.

원래 검증 기록, 통합 채점표, 조건별 표본 수와 불확실성은 결과 폴더에 보관했습니다. 관측소·데이터 범위와 실험 설정이 달라진 과거 결과는 `historical_regional12`로 분리합니다.


## 공유본의 경로 및 재현 검사

저장소 루트에서 코드 누락·버전 검사와 대표 그림 재생성을 실행할 수 있습니다.

```bash
python D/verify_share.py --code-only
python -m pip install -r D/requirements_figures.txt
python D/reproduce_figure.py validation_curve
python D/reproduce_figure.py pre_east_counts
```

`reproduce_figure.py`는 입력 표·자산을 새 실행 폴더로 복사해 그리므로 기존 그림을 덮어쓰지 않습니다.
`actual_network`, `expansion_stages`, `outlier_ranges`, `current_counts`, `seasonal_concept`, `all`도 선택할 수 있습니다.
결과는 `results/D/reproduced/Run_...`에 저장합니다. 이 경량 재그리기는 모델 학습이나 원자료 재처리가 아닙니다.
모든 원자료부터 재계산하는 분석은 각 README의 Dataset 의존성을 따릅니다.


## 공유 복사본의 체크포인트 검증 수정

2026-11-15 공유 재현 검사에서 선택 옵션 `selection_horizon=24`의 점수를 저장할 때 6h 점수와 대조해 거부하는 오류를 발견했습니다.
공유본의 `checkpoint_store.py`만 선택 지평의 `validation_selection_score`를 검증하도록 수정했습니다.
기본6h 기록과 계약·학습 기록의 지평 일치 검사도 유지합니다. Graph WaveNet H48에서 사용하던 동일 검증 방식입니다.
모델 구조·기본6h 선택·기존 학습 결과는 바꾸지 않았고 Revised_Project 원본은 보존했습니다.
변경 diff와 원본/공유 해시는 `D/patches/`에 있습니다. DCRNN 업로드 매니페스트는 공유본 해시로 갱신했습니다.
이 변경 이후의 새 학습은 새 코드 버전으로 기록되며, 과거 체크포인트 자동 재개는 코드 계약 검증을 따릅니다.
