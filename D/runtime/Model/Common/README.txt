공통 채점기 — 고정 평가 행·인과적 기준선·실제 시간 블록
버전: fixed-calendar-seed-v1.0.0 / 2026-11-07

이 채점기는 학습하거나 모델 구조를 바꾸지 않습니다. 원래 지평별 eligible 행 전체를 사용하고, 팀원 ZIP 채점기의 모델별 예측 교집합/dropna는 사용하지 않습니다.

설치: python -m pip install numpy pandas pyarrow

입력
1. 평가 격자: 원래 Origin_Features의 wide 표 또는 long evaluation_grid.parquet.
   long 열: origin_time_kst,target_time_kst,horizon,split,y_true,eligible.
   3시간 간격의 모든 origin을 유지하며, 결측 정답도 eligible=False 행으로 남겨야 합니다.
2. 모델 예측 CSV/Parquet: origin_time_kst,horizon,model,y_hat; split,seed,input_condition,data_version 권장.
   원래 DCRNN의 horizon_h,y_pred,input_set,dataset_version 열 이름도 인식합니다.
3. 기준선이 격자에 없다면 --target-hourly로 원본 가공 시계열과 buffer를 전달합니다.
   station_id 열이 있으면 111121(서울 중구)을 선택합니다. --target-station-id로 변경 가능하지만 프로젝트 타깃은 중구로 고정입니다.

Git Bash 예시(프로젝트 루트에서)
python Model/Common/score.py \
  --grid Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet \
  --predictions path/to/seed11_predictions.parquet path/to/seed22_predictions.parquet path/to/seed33_predictions.parquet \
  --split validation --output-dir Model/Common/Results/validation_run_name

고정 기준선이 포함된 새 evaluation_grid를 기본 경로로 사용하므로 --grid도 생략할 수 있습니다.
원래 wide 표를 직접 쓰려면 --grid Dataset/Origin_Features_2021_2024/origin_features_2021_2024_kst.parquet 와
--target-hourly Dataset/Korea_Air/korea_34stations_hourly_2021_2024_kst.parquet Dataset/Korea_Air/korea_context_buffer.parquet 를 함께 지정합니다.
시험은 --split test --final을 명시해야 합니다. 이는 설정 동결을 요구하는 실수 방지 장치이며, 이미 본 2024년을 처음 보는 시험으로 바꾸지는 않습니다.
baseline은 채점기가 생성하므로 외부 persistence/seasonal_naive 예측 파일을 함께 전달하지 마세요.

확정된 평가 규칙
- train=2021–22, validation=2023, test=2024. 지평=3/6/12/24시간, 6시간 주평가.
- eligible=유효 실측 정답 AND 예측시점/정답시점 동일 분할 AND 연구기간 안.
- 모델에 예측이 없는 eligible 행, NaN/inf 예측, 중복 행, 잘못된 타깃은 즉시 오류입니다.
- Persistence: 시점 t 실측; 없으면 t 이전 마지막 유효 관측. 경과시간과 fallback 종류를 기록합니다.
- Seasonal naive: t+h−24 실측; 없으면 그 시점 이전 마지막 유효 관측. h=24일 때 Persistence와 같습니다.
- 과거 관측이 전혀 없을 때만 2021–22 중구 log1p 평균의 역변환을 최후 기준선 fallback으로 씁니다(원래 DCRNN과 동일). 정답을 채우는 처리가 아닙니다.
- 미래 관측으로 backward-fill하지 않고, 기준선 결측을 이유로 평가 행을 제거하지 않습니다.
- 기존 규칙상 2023년 장기공백 종료 부근 일부 eligible 행에서 기준선의 마지막 관측이 최대1063시간 전입니다. 경과시간을 숨기거나 그 행을 사후 제거하지 않습니다.
- 모든 예측은 µg/m³에서 평가하며 음수는 0으로 자르고 횟수를 run·지평별 기록합니다.
- 시드별 점수는 각각 계산합니다. seed_metric_summary는 점수들의 평균과 표본표준편차(ddof=1)입니다.
- 2개 이상 시드가 있으면 µg/m³ 예측값 산술평균의 별도 ensemble을 만듭니다. log 공간 평균·시드 점수 평균과 구분합니다.
- supplied ensemble과 자동 ensemble을 동시에 넣으면 오류입니다. supplied ensemble만 유지하려면 --no-ensemble.
- DM: 제곱오차, Bartlett HAC 기본 지연 max(h/3−1,0), 실제 3시간 달력 지연을 사용합니다.
- 7일 블록 bootstrap은 56개 실제 달력 슬롯을 재표집합니다. 44일 결측을 제거해 서로 떨어진 날짜를 인접하게 만들지 않습니다.
- HAC 겹침 최소 지연만으로 모든 추가 자기상관이 해결되지는 않습니다. 필요 시 사전 지정한 --hac-lag 및 별도 블록 길이 민감도를 보고하며 유의확률을 보고 설정을 고르지 않습니다.
- Holm family는 한 실행에서 선언한 모든 지평·run·비교쌍의 유효 p값입니다. 비교 목록을 결과에 따라 바꾸지 않습니다.
- 조건별 표는 기술통계이며, 별도 사건 탐지/경보 서비스 성능을 주장하지 않습니다.

출력
metrics.csv: 개별 시드·단일모델·예측 앙상블을 구분한 지표.
seed_metric_summary.csv: 개별 시드 지표 평균/표준편차. 앙상블 성능이 아님.
tests.csv: 같은 고정 행에서의 DM, 시간 블록 bootstrap 구간, Holm 보정.
conditions.csv: 계절·경로·바람·고농도 구간의 기술통계.
score_rows.parquet: 고정 eligible 행의 타깃·예측·기준선·fallback·시드·입력조건·데이터버전.
scoring_audit.json: 파일 SHA256, 격자 수, 시간 블록·추론 설정, sidecar provenance.

검증
python -m unittest discover -s Model/Common/tests -p test_score.py -v
Results/Verification의 32-origin 수치는 기존 파일럿 export와의 호환성 검사이며 연구 성능 결과가 아닙니다.
기존 전체 Colab ZIP은 현재 로컬 경로에 없어 실제 전체 모델 예측의 재채점은 하지 않았습니다.
원래 평가 격자 전체의 baseline 유효성·행 수는 기존 가공 시계열에서 검증했습니다.

연구 질문·가설·실험 분리 규칙: RESEARCH_PROTOCOL.txt 및 experiment_contract.json 참고.


권역별26개공통입력실험 추가 (2026-11-07)
------------------------------------
설정·실제관측소/feature목록: Regional_Experiments_20261107/README.txt 및 experiments.json.
새 공통 데이터 로더: regional_data.py의 RegionalDataLoader 또는 load_regional_xy.
D67/DN/F, 국외권역개별추가8개, 전체입력에서권역제외15개를 정의합니다.
이전 Dataset의 M1/M3/M4 feature_sets와 새26조건을 혼용하지 마세요. 기존 score.py·평가행·
DCRNN 모델/그래프는 그대로 유지합니다. 상세 Colab/Git Bash 사용법과 검증은 새 폴더에 있습니다.


2026-11-08 QC 이웃자료 확장
-------------------------
기존26조건/600계열은 Regional_Experiments_20261107 +regional_data.py로 유지합니다.
신규중국16·일본동부18·서부재선정의8조건은 Neighbor_Experiments_QC_20261108 +neighbor_data_qc.py로 구분합니다.
두버전의이름/feature_sets/loader를혼합하지마세요. 상세한QC·실험순서·Colab실행법은새폴더README.txt에있습니다.
기존score.py/grid.py의고정평가행·시드/앙상블·시간블록규칙은그대로입니다.
