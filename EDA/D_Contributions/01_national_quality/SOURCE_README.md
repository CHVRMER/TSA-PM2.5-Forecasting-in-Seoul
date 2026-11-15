# 전국 국내 PM2.5 품질 점검

분석 대상은 2021–2024년 도시대기·교외대기·국가배경 측정망입니다. 본 결과는 예측 성능 평가가 아닌 데이터 품질 감사입니다.

- `Results/pilot_verification.json`: 2020년 말 경계를 포함한 2021년 1월 파일럿. 기존 34곳의 시각·PM2.5 값과 정확히 비교합니다.
- `Results/station_year_quality.csv`: 측정소·연도별 실제 행 수, 유효값, 공란·이상값, 관측 기간 내 행 부재, 전체 연간 대비 관측 비율, 큰 양수 검토 횟수.
- `Results/quality_flags.csv`: 측정소·연도·원본 품질 분류별 시간 수.
- `Results/region_year_quality.csv`: 수도권·강원·충청·호남·영남·제주 권역별 요약.
- `Results/missing_runs_over24h.csv`: 최초–최종 원본 행 사이에 나타나는 24시간 초과 연속 결측. 원본 행 부재와 원본 이상·공란을 함께 반영합니다.
- `Results/excluded_network_metadata.csv`: 사용자 범위에 따라 제외한 도로변·항만 등 측정망의 **메타데이터만**. 해당 농도는 분석용 Dataset에 추가하지 않습니다.
- `Results/legacy_preservation.csv`: 기존 34곳 및 동쪽 36곳의 전 기간 PM2.5·품질 분류 보존 비교. 기존 표에만 존재하는 원본 행 부재(flag2)는 sparse 새 표의 행 부재와 대응합니다.
- `Results/quality_summary.json`: 전체 실행과 보존 검증 요약.

‘최초–최종 원본 행 사이’는 공식 가동 기간과 동일하다고 단정할 수 없습니다. 연말 측정소 스냅샷은 정확한 개소·폐쇄·이전 날짜를 제공하지 않으므로 해석에 이 제한을 유지합니다. 원자료의 양수 피크는 분포만으로 오류라고 보고 삭제하지 않았습니다.

데이터와 가공 코드의 위치·시간 변환·단위·후보 선정 참고 표시는 `Dataset/Korea_National_2021_2024/README.md`를 확인하세요.
