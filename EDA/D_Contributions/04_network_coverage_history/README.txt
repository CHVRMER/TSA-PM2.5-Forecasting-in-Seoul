관측망 확장에 따른 궤적–PM2.5 관측 연결률 비교 (2021–2024)

범위: 서울 도착 500 m AGL, 3시간 간격 11,688개 궤적, 각 궤적의 과거 1–48시간 위치 561,024개.
반경: 75 km. 표의 관측 연결률은 해당 위치 주변 후보 중 당시 PM2.5 실측이 하나 이상 유효한 지점의 비율.
서울 중구(예측 대상)는 외부 관측망 개수·연결률에서 제외. 원래 외부 3곳은 백령도, 웨이하이(1662A), 순의(1008A).
모든 구성은 같은 지점·같은 시각·같은 반경으로 비교. 공백을 채운 값은 사용하지 않음.

핵심 결과:
외부 3곳: 2.8020%. 기존 115곳: 47.5179%. 확장 317곳: 49.0598%.
3→115: +44.7159%p, 실측 연결 지점 +250,867개.
115→317: +1.5418%p, 실측 연결 지점 +8,650개; 1,686개 도착 궤적에서 추가 과거 관측 확보.
일본 188→균형판 24를 사용하면 전체 153개 후보의 연결률은 48.9034%.

해석 제한:
1. 이 값은 과거 경로의 관측 가능성으로, 예측 오차 감소·오염 질량·국가별 배출 기여율이 아님.
2. 25–48시간 전 경로가 잘 연결된다는 것은 24/48시간 후 예측이 좋아진다는 뜻이 아님.
3. 수평 근접으로 연결했으므로 궤적 고도와 지상 관측의 혼합 가능성은 별도 점검 필요.
4. 국경 13곳은 중국 도시 평균과 근사 좌표이며, 러시아 관측은 없음.
5. 후보별·권역별 연결 수는 중복될 수 있음. 네트워크 비율은 지점별 합집합으로 계산.
6. 전체 기간 지도·연결률은 기술통계. 2024 결과를 보고 입력·그래프·임계값을 고르면 독립 시험으로 해석할 수 없음.
7. 궤적의 시간 간격은 모형 위치의 표본 간격이며, 48개 지점이 48개의 독립 관측은 아님.

파일:
Results/network_expansion_coverage.png(.svg): 요약 그림.
Results/network_coverage.csv: 네트워크별 연결률.
Results/new_candidate_coverage.csv: 신규 권역별 증가분.
Results/coverage_by_path_age.csv: 경로 과거 시간 구간별 비교.
Results/coverage_by_12h_route_sector.csv: 12시간 전 위치 방위별 비교(지상 풍향이 아님).
Results/coverage_by_year.csv: 연도별 비교.
Results/new_station_point_connections.csv: 신규 후보별 연결 수(예측 중요도 아님).
Results/coverage_audit.json: 정의, 원래 3곳 코드, 중복·재현 검사 결과.

재계산(Git Bash, pip로 numpy pandas pyarrow matplotlib 설치 후):
python Codes/analyze_network_coverage.py --archive "/c/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project/Data_with_neighbors_v3_20261105.zip"
python Codes/plot_network_coverage.py

원본 ZIP 및 기존 지도·모델 파일은 수정하지 않음.
