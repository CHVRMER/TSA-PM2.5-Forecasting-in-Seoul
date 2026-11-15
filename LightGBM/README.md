# LightGBM PM2.5 예측

서울 중구 PM2.5의 3·6·12·24시간 후 농도를 예측합니다. M1~M4는 관측소 범위를 비교하고, M3T·M4T는 역궤적 변수를 추가합니다. Adaptive_wind와 Adaptive_route는 2023년 검증에서 추정한 조건별 가중치를 사용합니다.

## 결과

- [2024년 테스트 보고서](results/test_2024/REPORT.md) · [예측 시간별 RMSE](results/test_2024/report/all_horizons_RMSE.csv)
- [M1~M4·M3T·M4T RMSE 그래프](results/test_2024/report/figures/final_rmse_m1_m4_m3t_m4t_wide_large_legend.png) · [Adaptive RMSE 그래프](results/test_2024/report/figures/final_rmse_adaptive_models_wide_large_legend.png)
- [2023년 검증 결과](results/validation_2023/) · [Adaptive 가중치와 검증 결과](results/validation_2023/Adaptive/REPORT.md)
- [2023년 TreeSHAP 및 권역 묶음 순열 중요도](results/shap/validation_2023/REPORT.md)

## 코드

- [scripts/](scripts/)에는 학습, 평가, 해석 및 보고서 생성 진입점이 있습니다.
- [src/](src/)에는 데이터 로딩, 학습, 공통 채점, Adaptive 구현이 있습니다.
- [configs/](configs/)에는 모델별 입력 범위와 학습·평가 설정이 있습니다.
- [실험 규칙](README_EXPERIMENT_RULES.md) · [M1 실행 기록](README_STAGE1.md) · [공통 채점 안내](README_COMMON_SCORING.md)

스크립트의 프로젝트 루트는 이 폴더입니다. 의존성은 [requirements.txt](requirements.txt)에 기록했습니다. 학습·채점을 재현하려면 로컬에서 `Dataset/` 입력을 지정된 위치에 준비해야 합니다. 기존 실행을 검증하거나 보고서를 다시 생성하려면 해당 실행의 모델과 중간 산출물도 필요합니다. 이 공유본은 보고서·집계표·그래프를 포함합니다.
