# DCRNN H48 전용 공통 채점기

기존 공통 출력 형식을 보존하면서3·6·12·24·36·48h를 평가합니다.
원래 Model/Common/score.py, grid.py, neural_evaluation.py는 바꾸지 않습니다.
이 폴더 전체를 새 모델 ZIP과 함께 Model/Common 아래에 설치합니다.
평가 격자는 ensure_h48_grid로 Model 캐시에 생성하고 경로·해시를 scoring_plan에 동결합니다.
기존4지평은 그대로 복사하고36/48h는 정확한 중구t+h 정답 및 미래참조 없는 seasonal baseline을 생성합니다.
출력6종 파일명·열 순서·자료형, 달력블록·기준선 결측·seed/ensemble 분리를 유지합니다.
전체6조건 seed11의84개 검정이 Holm 보정 family입니다.
검증: verify_h48_scoring.py --original-common <TSA/Model/Common> --report <새보고서경로>.
