권역 29조건의 공통 채점 연결
==========================

regional_evaluation.py는 기존 Model/Common/score.py와 grid.py를 읽어서 호출합니다.
기존 계산 함수나 단일조건 neural_evaluation.py의 입력 일치 검사를 바꾸지 않습니다.
같은 권역 계획에 선언된 모든 조건·모델 계열이 끝나야 권역별 최종 비교를 수행합니다.

호출
----
score_regional_runs(
    project_root=TSA_ROOT,
    completed_runs=records,
    experiment_config=REGIONAL_CONFIG_DIR,
    output_root=TSA_ROOT/'Result/Common_Scoring/Regional'/BATCH_ID,
    split='validation', common_dir=TSA_ROOT/'Model/Common',
    bootstrap_reps=1000,
)

records의 각 행:
  condition: DN / F / ADD_... / DROP_...
  model_family: DCRNN / GraphWaveNet
  run_root: 원래 학습기가 반환한 실제 timestamp_UUID 실행 폴더
  regional_contract_path: 모델 학습 전 저장한 regional_contract.json 전체 경로

regional_contract_path를 생략하면 run_root/regional_contract.json을 읽습니다.
계약 파일을 원래 학습 실행 폴더에 새로 쓰거나 기존 예측 sidecar를 수정하지 않습니다.
config_dir에는 experiments.json, regions.json, comparisons.json이 있어야 합니다.
flattened load_regional_configuration() 반환값 대신 config_dir 경로를 전달하세요.

계약과 공정성
-------------
- 원래 노드 순서·그래프·타깃·문맥·정답·격자·원본 관측 우주는 모든 조건에서 같습니다.
- allowed/excluded station 목록과 관측 마스크는 조건 정의와 정확히 일치해야 합니다.
- 제외 노드의 pm_valid_count는 0이어야 하며, 중구/문맥 정규화는 동일해야 합니다.
- 각 계열에서 모델 구조·그래프 종류·하이퍼파라미터·시드 집합은 고정합니다.
- 관측 마스크 때문에 data_signature와 노드별 PM 정규화가 달라지는 것은 허용합니다.
- 단일 조건의 예측에는 그 조건의 실제 input_condition과 data_version을 유지합니다.
  여러 조건을 같은 이름/서명으로 위장해 단일조건 검사를 우회하지 않습니다.
- 고정 평가 행의 누락·비유한 예측·오염된 정답·실패/파일럿 실행은 오류입니다.
  성공한 조건만 뽑거나 모델 공통 예측 행의 교집합으로 표본을 줄이지 않습니다.

채점과 다중 비교
---------------
DN과 F의 전체 국외 비교 1개, 국외 권역별 추가 10개, 전체에서 권역 제외 17개를 선언합니다.
같은 시드끼리 비교하고, 원단위 시드 예측 평균을 별도 앙상블로 채점합니다.
모든 조건·두 계열·예측시간·시드/앙상블 비교를 하나의 원래 score_frames 호출로 처리합니다.
Holm 보정은 지역 결과를 필터링하기 전에 기존 기준선 비교까지 포함하여 한 번 적용합니다.
시드 점수 평균/표준편차는 예측 앙상블의 점수 또는 시간 표본 신뢰구간과 다릅니다.
7일 달력 블록 bootstrap 구간은 고정된 학습 모델들의 평가 오차 차이에 대한 구간입니다.
모델 재학습 불확실성 전체를 포괄하는 구간이라고 해석하지 않습니다.

표준 출력
---------
output_root/<validation 또는 test>/<한국시간_마이크로초_UUID>/
  metrics.csv
  seed_metric_summary.csv
  tests.csv
  conditions.csv
  score_rows.parquet
  scoring_audit.json

위 6개는 기존 공통 채점기 양식과 열 순서·자료형·Parquet 기본 압축을 유지합니다.
채점만 재실행해도 새 하위 폴더를 만들고 기존 결과와 소스는 그대로 보존합니다.

Additional_Analysis/
  regional_comparison_plan.json: 사전 조건·비교쌍·불변 계약·Holm 범위
  regional_gains.csv: 전체 시드/앙상블 권역 개선량과 신뢰구간
  regional_gains_by_seed.csv / regional_gains_ensemble.csv
  regional_seed_gain_summary.csv: 시드별 개선량의 평균과 표본표준편차
  regional_tests.csv: 기존 tests에서 사전 지역 비교만 필터링한 표
  metrics_by_seed.csv / metrics_ensemble.csv
  regional_scoring_audit.json
  Figures/<계열>_<add 또는 drop>_rmse_gain.png

부호
----
원래 tests.csv의 rmse_diff는 candidate-reference이며 음수가 개선입니다.
regional_gains의 RMSE_gain은 reference-candidate이며 양수가 개선입니다.
신뢰구간은 [-원래상한, -원래하한]으로 부호와 순서를 함께 뒤집습니다.
ADD/전체 국외 개선율 분모는 기준선 RMSE, DROP 손실률 분모는 전체 입력 F의 RMSE입니다.
추가 이득과 제외 손실은 다른 조건부 정보 가치를 측정하며 합계 100%인 비율이 아닙니다.
관측 정보의 예측 가치를 보여줄 뿐 국가별 오염 질량·발원지 기여율을 추정하지 않습니다.

실행 범위
---------
2023 검증은 checkpoint 선택에 사용했으므로 독립 최종 우월성의 근거가 아닙니다.
2024 시험은 split='test', final=True로 명시할 때만 읽습니다.
실제 29조건·두 계열·시드3개 결과에서는 예측과 score_rows가 수백만 행일 수 있으므로
최종 채점 전에 모델/학습 텐서 등 더 이상 필요 없는 메모리를 해제하는 것을 권장합니다.
조건을 쪼개서 서로 별도로 유의성 보정한 결과를 전체 보정 결과처럼 합치지는 않습니다.
