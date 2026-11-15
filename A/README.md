# 모델 A: SARIMAX + 권역 시차 변수, 그랜저 검정

코드는 업로드 예정입니다. 결과는 `results/A/`에 둡니다.

## 팀 M1–M4 조건으로 수정한 결과에 쓰인 코드

이 폴더에는 수정 전 실험 코드도 함께 있습니다. 수정한 결과(`results/A/ModelA_M1_M4/`)에 쓰인 파일은 아래와 같습니다.

| 역할 | 파일 |
|---|---|
| 공통 | `config.py`, `data_access.py`, `grid.py`, `predictions_io.py`, `a_sarimax.py` |
| 권역 평균 | `region_hourly.py` |
| SARIMA 차수 선택 | `a_sarima.py` |
| SARIMAX M1–M4 | `a_sarimax_ablation.py` |
| 채점 | `run_score_m.py` |
| 그림 | `make_fig_M.py`, `make_fig_M_heat.py` |
| 실행 노트북 | `ModelA_SARIMAX_M1_M4_colab.ipynb` |

수정 전 실험(3권역, Regional12)용: `a_granger.py`, `run_score.py`, `r12_data.py`, `grid_ext.py`, `a_r12.py`, `score_r12.py`, `check_order_convergence.py`, `check_start_params.py`, `ModelA_SARIMAX_colab.ipynb`

결과와 자세한 설명: `results/A/ModelA_M1_M4/README.md`
