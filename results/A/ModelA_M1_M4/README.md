# Model A 수정: SARIMAX M1–M4 (팀 공통 입력 조건)

**팀 M1–M4 조건으로 수정한 Model A 결과는 이 폴더입니다.** 같은 `results/A/`에 있는 `ModelA_3region_results.zip`, `ModelA_regional12_results.zip`, PDF 2개, PNG 2개는 수정 전 실험(3권역, Regional12) 결과입니다.

| 구분 | 입력 조건 | 위치 |
|---|---|---|
| **수정** | 팀 M1–M4 (`feature_sets.json`의 `model_B_M1`–`M4`와 같은 관측소) | `results/A/ModelA_M1_M4/` (이 폴더) |
| 수정 전 1차 | 3권역(서해안·가까운 중국·먼 중국) + 나라별, Granger | `results/A/ModelA_3region_results.zip` |
| 수정 전 2차 | Regional12 (DN, F, ADD 10개), 3–48시간 | `results/A/ModelA_regional12_results.zip` |

## 수정한 결과(M1–M4)에 쓰인 코드 (`A/` 폴더)

`A/`에는 수정 전 실험 코드도 함께 있습니다. 수정한 결과에 쓰인 파일은 아래 코드 11개와 실행 노트북 1개입니다.

| 역할 | 파일 |
|---|---|
| 공통 | `config.py`, `data_access.py`, `grid.py`, `predictions_io.py`, `a_sarimax.py` |
| 권역 평균 | `region_hourly.py` |
| SARIMA 차수 선택 | `a_sarima.py` |
| SARIMAX M1–M4 | `a_sarimax_ablation.py` (M1–M4 조건 추가된 버전) |
| 채점 | `run_score_m.py` |
| 그림 | `make_fig_M.py`, `make_fig_M_heat.py` |
| 실행 노트북 | `ModelA_SARIMAX_M1_M4_colab.ipynb` (위 코드를 순서대로 실행, 결과 표·그림 출력 포함) |

나머지(`a_granger.py`, `run_score.py`, `r12_data.py`, `grid_ext.py`, `a_r12.py`, `score_r12.py`, `check_*.py`, `ModelA_SARIMAX_colab.ipynb`)는 수정 전 실험(3권역, Regional12)용입니다.

## 입력 조건

관측소 묶음마다 시간별 평균 1개를 log1p로 넣음. 지연 L ∈ {h, h+3, h+6}은 2023 RMSE로 선택(모든 조건에서 L = h).

| 조건 | 관측소 | SARIMAX 외생변수 |
|---|---|---|
| M1 | 국내 67곳 | 국내 평균 |
| M2 | M1 + 서울에서 600 km 이내 중국 21곳 | 국내 평균, 가까운 중국 평균 |
| M3 | 국내 67곳 + 중국 전체 61곳(국경 대리 13곳 포함) | 국내 평균, 중국 전체 평균 |
| M4 | M3 + 일본 균형판 24곳 + 울란바토르 | 국내, 중국 전체, 일본, 몽골 평균 |

M3T·M4T(궤적 특징)는 팀 결정에 따라 하지 않음.

## 결과 (2023 검증 RMSE, µg/m³)

| | 3h | 6h | 12h | 24h |
|---|---|---|---|---|
| Persistence | 7.18 | 9.97 | 13.11 | 15.20 |
| M1 | 6.56 | 9.39 | 11.80 | 13.28 |
| M2 | 6.53 | 9.29 | 11.57 | 12.98 |
| M3 | 6.55 | 9.28 | 11.53 | 12.75 |
| M4 | 6.54 | 9.27 | 11.49 | 12.74 |

팀 개선 판정(95% 구간 상한 < 0, Holm p < 0.05):
- M2 vs M1: 6, 12, 24h 유의
- M3 vs M2: 24h만 유의
- M4 vs M3: 전부 유의하지 않음

## 폴더

| 폴더 | 내용 |
|---|---|
| `predictions/` | M1–M4 예측(팀 공통 형식) + `.meta.json` |
| `results/fit_records/` | 조건별 선택 지연 L, 수렴 여부, 계수와 95% 신뢰구간 |
| `results/score/` | 팀 공통 채점기(`Model/Common/score.py`) 결과 |
| `results/planA_M_table.csv` | 예측 시간별 RMSE·MAE 요약 |
| `figures/` | RMSE 그림, 입력 추가 시 RMSE 변화(%) 표, 각 그림의 값 |

## 다시 실행

**노트북:** Google Drive `MyDrive/TSA_planA/`에 `ModelA_SARIMAX_M1_M4_colab.ipynb`와 데이터 묶음 `PlanA_data.zip`을 넣고 Colab에서 모두 실행.

**스크립트:** `A/`의 코드에서:
1. `python region_hourly.py` (권역·나라별 시간 평균)
2. `python a_sarima.py` (SARIMA 차수 선택)
3. `python a_sarimax_ablation.py --only M1 M2 M3 M4`
4. `python run_score_m.py`
5. `python make_fig_M.py`, `python make_fig_M_heat.py`
