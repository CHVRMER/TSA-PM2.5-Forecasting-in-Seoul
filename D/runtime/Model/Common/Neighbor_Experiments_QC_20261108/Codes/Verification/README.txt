QC 신규 자료와8개 입력 조건의 독립 검증
====================================

이 코드는 생산 코드의 QC·관측소 선택·특징 생성 함수를 이용하지 않고 기존 원자료와 사전에서
예상 결과를 재계산한다. neighbor_data_qc.py 자체의 입출력은 실제로 호출하여 비교한다.
ZIP에 포함된 구현은 실행하지 않는다. 변조 검사는 복사한 메모리 표에만 적용한다.

주요 통과 결과
- 인과적 QC: 중국30도시+버퍼1,054,800행을 비교해PM2.5 3개·PM10 1개 격리,
  다른 관측·원래값·원래플래그·시각·출처ID·분할·QC이유 비트가 모두 예상과 일치했다.
- 신규544특징×11,688시점=6,358,272값을 QC 원자료에서 독립 재계산해 모두 일치했다.
  기존 sidecar와 달라진 것은 QC3도시의21특징·24원점에 걸친93개 셀뿐이며 일본과 나머지는 그대로다.
-8조건은 중국 동부 추가·일본 동부 추가·일본 서부24/25 선정의2×2×2 완전 구분이다.
  공통70열·원래F600의 열 순서·값을 유지하고600/601/616/617/618/619/634/635계열을 정확히 선택한다.
- 신규 서부 선정표는188행 중 selected=True25행만 사용한다. 기존24와 교집합은9다.
- 사전 비교13개(단일요인 변화12+중국/일본 동부 동시추가1), 주 비교4개가 정확하다.
- 실제32개 주간 로드와24개 전체기간 h6 로드를 확인했다. 모든 지평의 고정 적격 행 수도 확인했다.
- F600_W24는 원래26조건의F와 학습·검증·시험h6 모든 입력값·열순서·정답·평가시점이 일치했다.
- 제외 자료·정답·범위 밖 집계열을 변경해도X불변, 미래 행을 변경해도 과거X불변이다.
  입력 전체NaN 행도 보존되며, 잘못된조건·중복/누락·무한대·시험잠금은 오류로 처리한다.
-650계열 통합 카탈로그+중구651노드의 관측사용mask를 조건별로 대조했다.
- 매칭은 기존16,769,834행의8열이 모두 그대로이며 신규6,916행은QC원자료·시각·거리·실측표시와
  일치한다.856매칭후보는 모델의650통합카탈로그나 조건별600~635선택 수와 다르다.
- 같은601노드 순서로 비교한1주일 기준선의 최근접 노드와 궤적 인접행렬도 기존v4와 같다.

재실행 환경: numpy,pandas,pyarrow. 입력 캐시는 수백MB이므로 여러X를 동시에 보관하지 않는다.
원본 ZIP은 독립적인 선정·값 참조에 필요하며, 모델 실행 로더의 필수 입력은 아니다.
아래PROJECT·ARCHIVE를 각 환경의 경로로 바꾸고 Git Bash에서 실행한다.

PROJECT="C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project"
ARCHIVE="C:/Users/kimjy/Downloads/Dataset_11_07_regions_v5.zip"
CONFIG="$PROJECT/Model/Common/Neighbor_Experiments_QC_20261108"
VERIFY="$CONFIG/Codes/Verification"
DATA="$PROJECT/Dataset/Neighbors_Expansion_QC_20261108"
MATCH="$PROJECT/Dataset/Trajectory_Matching_Expansion_QC_20261108"

python -B "$VERIFY/verify_config_scopes.py" --project-root "$PROJECT" --archive "$ARCHIVE" --config-dir "$CONFIG" --report "$CONFIG/Results/Verification/input_scope_verification.json"
python -B "$VERIFY/verify_neighbor_loader.py" --project-root "$PROJECT" --archive "$ARCHIVE" --common-dir "$PROJECT/Model/Common" --config-dir "$CONFIG" --new-history "$DATA/station_history_qc_2021_2024.parquet" --report "$CONFIG/Results/Verification/neighbor_loader_verification.json"
python -B "$VERIFY/verify_matching_union.py" --project-root "$PROJECT" --matching-dir "$MATCH" --qc-data-dir "$DATA" --report "$CONFIG/Results/Verification/matching_union_verification.json"

QC·544특징 검사인 verify_data_qc_and_history.py에는 --archive, --qc-hourly, --qc-buffer,
--history, --dictionary, --report를 전달한다. dictionary에는 신규544열의 stat/station_key가
명시된 첨부 사전 또는 이를 정규화한 사전을 사용한다. CLI는 같은 이름의 검증JSON을
덮어쓰므로 비교 이력을 보존하려면 새로운 --report 파일명을 사용한다.

시간 누출 검사의 범위
로더의 미래 행 변조는 이미 계산된 특징표에서 올바른 시점을 선택하는지 검사한다.
QC 규칙은 현재와 과거만 사용하고,544개 특징도 과거만으로 재계산한 참조와 모두 일치한다.
이 검증은 미래 측정값이나 정답을 보고 규칙을 선택하는 것을 허용한다는 뜻이 아니다.
입력·변환·보존 검증이며, 모델의 예측 성능이 개선됐다는 증거와는 구분한다.
