DCRNN / Graph WaveNet 자동 공통 채점
===================================

역할
----
neural_evaluation.py는 참조 ZIP의 score.py 계산 함수와 grid.py 규칙을 호출하는 실행 모듈입니다.
관측소·국가·노드 수와 무관하게 같은 고정 평가 행으로 시드별 점수, 원단위 예측 앙상블,
기준선, 그래프 간 비교를 정리합니다. 새로운 학습·정답 보간·평가 행 축소는 하지 않습니다.

Colab 필수 위치
---------------
TSA/Model/Common/neural_evaluation.py
TSA/Model/Common/score.py
TSA/Model/Common/grid.py
TSA/Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet

모델 노트북과 시간별 관측·버퍼·그래프 매칭 파일은 각 모델 안내에 따릅니다.
LightGBM이나 추가 시계열 모델 패키지는 필요하지 않습니다.
이 모듈에는 numpy, pandas, pyarrow, matplotlib이 필요합니다.

노트북 호출 순서
----------------
1. 학습 전에 write_neural_scoring_plan()으로 scoring_plan.json을 동결합니다.
2. full 학습 후 최적 가중치로 원단위 미절단 시드 예측을 저장합니다.
3. score_neural_run(..., output_root=SCORE_ROOT)를 호출하면
   SCORE_ROOT/validation/<KST시각_마이크로초_uuid8>/에 결과가 생성됩니다.
   권장 SCORE_ROOT는 TSA/Result/Common_Scoring/<모델계열>/<입력조건>/<프로필>/<실행이름>입니다.
   모델 checkpoint·학습 로그·예측은 기존 모델별 run_root에 두고 공통 채점 결과를 분리합니다.
   output_root를 생략하면 run_root/Scoring/validation/<고유실행ID>/를 사용합니다.
   같은 인자로 다시 호출해도 항상 새 하위 폴더를 만들며, 이전 출력은 덮어쓰지 않습니다.
   pilot 건너뛰기 기록도 매번 새 실행 폴더에 저장합니다.
4. 2024 평가를 사용자가 명시적으로 켠 경우에만 split='test', final=True를 사용합니다.
   이미 확인한 2024년을 최초 블라인드 시험으로 바꾸는 기능은 아닙니다.

기본 예측 파일
--------------
run_root/Exports/validation_predictions_by_seed.parquet
run_root/Exports/test_predictions_by_seed.parquet

기존 노트북의 Summary/Common_Exports/<split>_predictions_by_seed.parquet도 읽습니다.
파일에는 origin_time_kst, horizon, model, y_hat, seed, prediction_kind='seed',
input_condition, data_version가 필요합니다. split, target_time_kst, y_true,
experiment_mode='full', common_export_unclipped=True도 함께 저장하는 것이 권장됩니다.
y_hat은 µg/m³로 역변환한 뒤 0으로 자르기 전 값입니다.
제공된 앙상블 파일은 함께 전달하지 않습니다. 시드 예측으로 채점기가 직접 만듭니다.
원래 예측 옆의 .meta.json은 읽기만 합니다. 없다면 원래 실행 폴더에 새로 쓰지 않고
이번 채점 실행의 Additional_Analysis/prediction_source_metadata.json에 보충 정보를 저장합니다.

API 예시
--------
from neural_evaluation import write_neural_scoring_plan, score_neural_run

specs = [
    {'model': 'DCRNN_distance', 'graph': 'distance'},
    {'model': 'DCRNN_trajectory_directed', 'graph': 'trajectory_directed'},
    {'model': 'DCRNN_trajectory_symmetric', 'graph': 'trajectory_symmetric'},
]
metadata = {
    'target_station_key': 'KR:111121', 'horizons': [3, 6, 12, 24],
    'lookback_hours': 48,
    'target_transform': 'log1p + train-only standardization',
    'data_signature': DATA_SIGNATURE,
    'preprocessing_signature': PREPROCESSING_SIGNATURE,
    'train_split': '2021-2022', 'validation_split': '2023',
    'prediction_unit': 'ug/m3',
    # 실제 아키텍처·optimizer·scheduler·loss·학습 설정도 추가 기록합니다.
}
write_neural_scoring_plan(RUN_ROOT, 'DCRNN', CONDITION, specs,
                          [11, 22, 33], metadata)
# 여기에서 학습 및 시드 예측 저장
score_neural_run(TSA_ROOT, RUN_ROOT, 'DCRNN', CONDITION,
                 split='validation', experiment_mode=MODE,
                 output_root=SCORE_ROOT)

그래프 비교의 의미
------------------
기본 비교는 궤적 방향 vs 거리, 궤적 대칭 vs 거리, 궤적 방향 vs 궤적 대칭입니다.
같은 입력 조건·시드 집합에서 동일 시드끼리 비교하고, 앙상블도 별도로 비교합니다.
자동 비교는 model_specs에서 graph/model/label/description 이외 설정이 동일한 그룹끼리만
만듭니다. Graph WaveNet의 adaptive_mode가 다르면 자동으로 같은 비교에 섞지 않습니다.
직접 비교를 지정하려면 pair_specs=[(candidate_model, reference_model), ...]를 학습 전에
write_neural_scoring_plan에 전달합니다. 그 후 비교 목록을 바꾸면 새 실행 폴더가 필요합니다.

graph_comparison_metrics.csv의 RMSE 차이는 candidate-reference입니다. 음수면 candidate가
더 낮은 RMSE입니다. 별도로 저장하는 gain=reference-candidate는 양수면 개선 방향입니다.
score.py가 기본 생성하는 Persistence 대비 비교는 그대로 유지합니다.
검정의 Holm family에는 그 실행에서 선언한 모든 모델·시드·지평·비교쌍이 포함됩니다.
결과를 본 뒤 유의한 비교만 남겨 다시 채점해 보정 범위를 줄이면 안 됩니다.

표준 출력: output_root/<validation 또는 test>/<고유실행ID>/
---------------------------------------------------------
다음 6개는 Dataset_11_07_regions_v5.zip의 Model/Common/score.py CLI 양식을 그대로 유지합니다.
파일명, 열 이름·순서·자료형, Parquet 기본 압축, scoring_audit.json의 키 구조를 바꾸지 않습니다.
새 모델·시드·그래프·비교가 추가되면 해당 행만 늘어납니다.

metrics.csv: 시드·앙상블·기준선 전체 지표.
seed_metric_summary.csv: 개별 시드 점수 평균과 표본표준편차(ddof=1).
tests.csv: 기본 기준선 비교와 선언한 추가 비교의 DM·7일 달력 블록 bootstrap·Holm 보정.
conditions.csv: 계절·경로·바람·고농도 구간의 기술통계.
score_rows.parquet: 고정 평가 행·정답·원예측·최종예측·기준선·시드·입력 버전 기록.
scoring_audit.json: 기존 CLI의 audit + written_kst/grid_path/grid_sha256/prediction_sources.
  prediction_sources의 항목은 기존과 같은 path/sha256/sidecar 3개 키입니다.

추가 분석: 위 폴더의 Additional_Analysis/ 아래에만 저장
-----------------------------------------------------
evaluation_invocation.json: 동결 계획·예측·격자·scorer 해시, 비교 목록과 추론 설정.
neural_scoring_audit.json: 자동화 상태·데이터/코드 보존 검사·추가 비교·선택 자료의 해석 한계.
metrics_by_seed.csv: 개별 시드 점수만 추출.
metrics_ensemble.csv: 원단위 시드 예측 평균의 별도 성능만 추출.
metrics_baselines.csv: 동일 행 Persistence와 Seasonal naive만 추출.
graph_comparison_metrics.csv / graph_comparison_tests.csv: 선언한 그래프/모델 비교만 추출.
cross_model_comparison.json: 두 모델 계열 비교를 실행한 경우에만 비교 계획 사본.
Figures/rmse_seed_scores_and_ensemble.png: 시드 점수 평균±표준편차와 앙상블을 구분.
Figures/graph_or_model_comparison_gains.png: 선언 비교의 지평별 RMSE 개선량.
Figures/learning_curves_and_learning_rate.png: 학습 로그가 있으면 train loss·검증6hRMSE·LR.

출력 양식 검증
--------------
tests/test_neural_evaluation.py의 참조 ZIP CLI 검사는 동일 입력·동일 비교·동일 bootstrap 시드로
원본 ZIP의 score.py/grid.py를 별도 실행합니다. 4 CSV의 열/순서/자료형/값, Parquet의 열/순서/
자료형/값/압축, core audit의 키 순서·구조·타입과 provenance 항목 구조를 대조합니다.
원래 score.py 계산 함수(main 제외)의 원문은 동일해야 하며 grid.py는 바이트까지 대조합니다.
현재 score.py CLI의 실행 폴더 관리만 달라질 수 있으므로 전체 파일이 ZIP과 동일하다고 주장하지 않습니다.
TSA_REFERENCE_ZIP 환경변수로 원본 ZIP 위치를 지정할 수 있습니다.
참조 ZIP이 없는 환경에서는 해당 비교 테스트만 skip하고 이유를 표시합니다.

학습 로그 자동 인식
------------------
run_root 아래 learning_curve.csv, training_history.csv, history.csv를 찾습니다.
권장 열은 epoch, train_loss, validation_rmse_h6, learning_rate_used입니다.
learning_rate_next도 학습 로그에 보존하되 그림은 해당 epoch에 실제 사용한 LR을 표시합니다.
learning_histories=[Path(...), ...]를 직접 전달할 수도 있습니다.

두 모델 계열 비교
-----------------
write_cross_model_comparison_plan(left_run_root, right_run_root, output_root,
                                  model_pairs=[('DCRNN_distance', 'GWNet_distance')])
compare_neural_runs(TSA_ROOT, left_run_root, right_run_root, output_root,
                     scoring_output_root=SCORE_ROOT)

위 output_root는 고정 비교 선언 cross_model_plan.json을 보관하는 상위 폴더입니다.
compare_neural_runs를 호출할 때마다 output_root/<고유비교ID>/에 새 scoring_plan·결합 예측을 만듭니다.
scoring_output_root를 지정하면 공통 채점 결과는 SCORE_ROOT/<split>/<고유실행ID>/에 저장합니다.
생략하면 새 비교 실행 폴더의 Scoring/<split>/<고유실행ID>/를 사용합니다.
반환값 comparison_run_root/output_dir/additional_analysis_dir가 실제 생성 경로입니다.
이전에 완료된 두 모델의 실행 폴더와 고정 비교 선언은 읽기만 합니다.

두 scoring_plan.json의 입력 조건·정답·지평·lookback·변환·원자료 서명·전처리 서명·
학습/검증 분할·단위·시드 집합이 같아야 합니다. 누락되거나 다르면 오류로 중단합니다.
아키텍처·손실·optimizer·학습 일정이 다른 부분은 비교 계획에 함께 기록됩니다.
따라서 이는 전체 모델 설정의 비교이며, 방향성이나 아키텍처 하나만의 효과로 해석하지 않습니다.
비교 쌍은 직접 지정해야 합니다. 결과를 보고 정했다면 analysis_status='exploratory'를 사용합니다.

보존한 평가 규칙과 한계
-----------------------
pilot은 전체 고정 행이 없으므로 점수 파일을 만들지 않고 명시적인 skipped_pilot 기록을 남깁니다.
eligible 행의 예측 누락·중복·NaN/inf를 발견하면 중단합니다. 모델별 교집합으로 행을 줄이지 않습니다.
정답이 결측인 시간은 달력에서 유지하고 bootstrap/HAC에서 이웃 시간으로 압축하지 않습니다.
정답·기준선 결측 규칙, 음수 처리, 7일 블록, HAC 기본 지연은 기존 score.py/grid.py 그대로입니다.
2023년은 조기종료·설정 선택에도 사용합니다. 해당 구간 p값·신뢰구간·improved 표시를 독립
최종 우위의 증거로 사용하지 않습니다. 예측 기여와 실제 오염 배출원 기여도도 구분해야 합니다.
