권역 통과와 이후 중구 PM2.5 — 학습구간 EDA
========================================

대상: 2021–2022년, 3시간 간격 5,840도착, 500m AGL, 과거 48시간 경로.
정답: 기존 고정 평가 행의 3·6·12·24시간 후 중구 실측 농도.
주 관계 분석: 과거 7–48시간. 1–48 및 13–48시간도 함께 저장.

먼저 Results/report.html을 브라우저에서 열면 한글 해석·전체 요약표·그림 5종을 볼 수 있다.
report.txt는 같은 해석의 텍스트 사본이다. 팀 공유 시 이 EDA 폴더 전체를 전달하면 HTML의
그림 링크가 유지된다. CSV는 UTF-8 BOM으로 저장했다.

폴더
- Codes: 지리 분류, 통계 EDA, 그림, 보고서 생성 코드.
- Codes/assets: 실제 사용한 행정경계·출처 메타데이터. 인터넷 재다운로드 없이 재현 가능.
- Results: 표·그림 PNG/SVG·학습구간 EDA 표·출처와 독립 검증 기록.

구분해서 해석할 것
- 지리적 통과율은 관측소 50/75/100km 반경 연결률이 아니다.
- 통과 권역은 서로 겹친다. 비율을 합해 100% 기여율로 만들 수 없다.
- 고농도는 시간값 ≥35 µg/m³라는 연구용 구분이며 공식 경보 기준이 아니다.
- 표준화는 계절×현재 농도 구간을 고려하는 기술통계다. 지역별 오염 배출·인과 기여나
  모델 예측 성능을 추정하지 않는다. 권역별 공통층이 달라 조정값 순위를 매기지 않는다.
- origin_analysis_table.parquet에는 y_h*, eligible_h* 등 정답이 있다. EDA용 중간표이며
  모델 입력으로 모든 열을 사용하면 안 된다. 기존 모델 데이터·DCRNN 구조는 수정하지 않았다.
- 이 EDA는 2023·2024년 정답 관계를 계산하지 않았다. 모형 선택에 사용할 탐색은 학습자료에
  한정하고, 2023에서 선택·2024에서 고정 평가한다는 기존 원칙을 따른다.

사용 원자료(읽기 전용)
1. Dataset/Trajectories/trajectory_points_2021_2024.parquet
2. Dataset/Korea_National_2021_2024/korea_national_pm25_hourly_2021_2024_kst.parquet
3. Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet

재실행 — Git Bash
아래 python은 pandas/numpy/pyarrow/matplotlib/shapely를 설치할 분석용 환경의 Python이다.

PROJECT="C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project"
EDA_DIR="$PROJECT/EDA/Geographic_Region_Future_PM25_Training_2021_2022"
python -m pip install -r "$EDA_DIR/Codes/requirements.txt"
python "$EDA_DIR/Codes/classify_geographic_routes.py" --trajectories "$PROJECT/Dataset/Trajectories/trajectory_points_2021_2024.parquet" --out "$EDA_DIR"
python "$EDA_DIR/Codes/run_geographic_eda.py" --project-root "$PROJECT" --eda-root "$EDA_DIR"
python "$EDA_DIR/Codes/plot_results.py" --eda-root "$EDA_DIR"
python "$EDA_DIR/Codes/make_report.py" --eda-root "$EDA_DIR"

같은 EDA 폴더의 이번 실행 결과만 다시 생성한다. Dataset·Model·기존 EDA 결과를 이동하거나
삭제하지 않는다. 보고서 해석 문장은 이번 고정 데이터와 기본 설정(35, 최소 층 표본10,
500회 블록 재표집)에 맞춰 작성됐다. 다른 자료·설정으로 분석하면 해석 문장도 새로 검토해야 한다.

경계 출처
GeoBoundaries/Natural Earth(한국), geoBoundaries/Wikimedia Commons(중국),
MLIT National Land Numerical Information via geoBoundaries(일본, CC BY4.0),
Natural Earth(그 외 국가). 정확한 출처·라이선스·해시와 가공 내역은
Codes/GEOGRAPHIC_CLASSIFICATION.md 및 Results/geographic_classification_manifest.json에 있다.
