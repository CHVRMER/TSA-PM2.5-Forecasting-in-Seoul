권역별 관측 정보 비교 — 2026-11-09
================================

범위
----
기존 국내67 비교(D67)를 제외한 29조건입니다.
국내514·중국77·일본42·몽골1 = 외부634계열, 서울 중구 타깃을 포함해635노드입니다.
기존 Neighbor_Experiments_QC_20261108의 F634_BOTH_W24 구성만 사용합니다.
일본 서부는 기존24를 유지하며, 서부 재선정25와 후보 전체650을 혼합하지 않습니다.
중국 도시평균 계열도 포함하므로 모든 계열을 개별 측정소라고 부르지 않습니다.

권역
----
국내7: 수도권162, 강원30, 충청85, 호남86, 대경60, 동남82, 제주9.
국외10: 산둥11, 베이징·톈진·허베이9, 랴오닝18, 지린9, 헤이룽장6,
        내몽골12, 기타 중국동부12, 일본서부24, 일본동부18, 몽골1.
새 중국동부16은 산둥4와 기타 중국동부12에 배정합니다. 별도 권역으로 중복 배정하지 않습니다.
중국 국경도시 자료는 중국 권역입니다. 러시아 실측이나 러시아 기여율 자료가 아닙니다.

입력 조건과 비교
----------------
DN: 국내514의 PM2.5 관측만 허용.
F: 외부634의 PM2.5 관측을 모두 허용.
ADD_<국외권역>: DN에 해당 국외 권역 하나만 추가, 10조건.
DROP_<국내·국외권역>: F에서 해당 권역 하나만 제외, 17조건.
합계29조건이며 D67은 없습니다.
사전 비교28쌍: F vs DN 1개, ADD vs DN 10개, F vs DROP 17개.
추가 이득은 RMSE(DN)-RMSE(DN+R), 제외 손실은 RMSE(F-R)-RMSE(F)입니다.
모두 양수일 때 해당 권역 관측이 도움이 된 방향입니다.

모델 입력의 통제
----------------
모든 조건에 같은635노드 순서·그래프를 사용합니다. 제외된 노드도 잠재 경유점으로 남습니다.
원시 PM2.5 행렬을 복사한 뒤 제외 노드의 모든 시간 값을 NaN으로 만듭니다.
그 후 기존 2021–22 전용 정규화 함수를 적합하고 기존 두 채널 변환을 적용합니다.
제외된 노드의 정규화 농도는 항상0, 결측 채널은 항상1이며 원래 결측 양상도 노출하지 않습니다.
조건마다 모델을 다시 학습합니다. 전체 모델을 학습한 뒤 추론 시점에만 가리는 실험이 아닙니다.
중구 관측·정답·eligible·예측시점·서울·백령도 ASOS·달력은 그대로 보존합니다.
DROP_KR_CAPITAL에서도 중구 자체 이력과 공통 기상을 유지합니다.
모든 지역의 모든 정보를 제거했다는 뜻이 아니라 해당 권역 외부 PM2.5 관측을 제외했다는 뜻입니다.
F600의601노드 모델 결과와 이번635노드 관측 마스크 실험을 같은 실행으로 재사용하지 않습니다.

파일
----
experiments.json:29조건의 실제 관측 계열 목록과 고정 정책.
regions.json:중복 없는17권역.
comparisons.json:사전 선언된28비교쌍.
station_registry.csv:이번634계열만 추린 좌표·자료원·권역 목록(모델 노드 순서는 별도 node_order.csv).
condition_summary.csv:조건별 허용 관측 수.
Codes/regional_inputs.py:원시 관측 마스크·원형 전처리 연결·불변성 계약.
Codes/verify_regional_inputs.py:합성 및 실제635노드4년 자료의29조건 독립 검증.
Results/Verification/:실행 검증 결과. 새 예측 성능 결과가 아닙니다.

Python API
----------
config = load_regional_configuration(config_dir)
frozen = regional_universe_fingerprints(universe_bundle, graphs)
masked_bundle, preprocessing, arrays, contract = prepare_regional_condition(
    universe_bundle, condition_id, config, preserved_api, graphs, frozen)

load_regional_configuration은 experiments.json의 키와 함께 regions·comparisons·config_sha256를 반환합니다.
universe_fingerprints는 조건 실행 사이 원본·그래프가 바뀌지 않았는지 재검증하는 고정 계약입니다.
contract를 학습 전에 저장하고, 실제 checkpoint의 preprocessing.json과 교차 검증합니다.
원본 bundle의 관측 행렬과 기존 Dataset 파일은 수정하지 않습니다.

해석
----
이 비교는 같은 모델·평가 행에서 어떤 지역의 관측 정보가 예측에 유용한지를 평가합니다.
실제 배출원 기여율이나 합계100%인 국가별 오염 기여 비율을 추정하지 않습니다.
몽골은 울란바토르 한 계열이며 장기 결측이 있으므로 국가 전체의 정보를 대표하지 않습니다.
