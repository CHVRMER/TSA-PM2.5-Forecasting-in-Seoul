기존600계열 보존 + QC 신규 자료 비교 / 2026-11-08
================================================

기존 Regional_Experiments_20261107의26조건과 regional_data.py는 그대로 보존했다.
이 버전은 기존F를 기준으로 신규 중국16, 일본 동부18, 일본 서부24→25 선정 교체를 분리한다.
새 공통 로더는 상위 Common/neighbor_data_qc.py의 NeighborDataLoader다.

조건ID               외부계열  중구포함노드  특징열  역할
F600_W24                 600         601     9670  기존기준선 정확재현
F616_CNE_W24             616         617     9926  중국동부16만 추가
F618_JPE_W24             618         619     9958  일본동부18만 추가
F634_BOTH_W24            634         635    10214  신규34함께 추가
F601_W25                 601         602     9686  일본서부 재선정만
F617_CNE_W25             617         618     9942  재선정+중국동부(보조)
F619_JPE_W25             619         620     9974  재선정+일본동부(보조)
F635_BOTH_W25            635         636    10230  세요인 모두(보조)

F뒤 숫자는 중구를 제외한 외부 계열 수다.650은8조건의 합집합 카탈로그이며 단일 모델 노드 수가 아니다.
600기준선의 관측소·feature 열 순서·값·정답·평가행은 기존F와 정확히 같다.
기존600에는 국내514+중국61+일본24+몽골1이 들어 있다. 국내58후보를 다시 포함하지 않았다.
서부25는 selected=True인25개만 읽고, 원래24와 공통9·기존15제외·다른16추가임을 확인했다.
자료유형은 중국도시평균과 개별관측소를 구분한다. 일본·몽골을 음성 대조로 고정하지 않는다.

파일
experiments.json: 조건별 실제station_key,3개요인,주/보조구분,예상열수,런타임자료원.
feature_sets.json: 조건별 실제X열과순서. feature_registry.csv:합집합10470열의 소유권·출처.
station_registry.csv:650계열의 메타데이터·추가블록·서부선정소속. regions.json:중복없는지리권역.
comparisons.json:요인별 단순효과12쌍+신규34동시추가1쌍=13개 사전비교.
source_manifest.json:고정된원자료·설정입력 해시. condition_summary.csv:팀공유용8조건표.
Codes/Verification와Results/Verification:독립 입력·QC·시간·혼입·기준선보존 검증.

China_East16은 신규자료블록이다. 산둥4곳은CN_SHANDONG에 합치고 나머지12곳은CN_OTHER에 배정했다.
자료블록과 지리권역을 혼동하거나 기존 산둥과 중복되는 권역을 새로 만들지 않았다.
새서부16곳의 좌표·자료원은 기존 Japan 메타데이터를 사용해 임의좌표변경을 피했다.
이는 고정된사후관측망의 비교이며 실시간메타데이터 배포지연까지 재현한 실험이라고 주장하지 않는다.

Python/Colab
from pathlib import Path
import sys
TSA_ROOT=Path('/content/drive/MyDrive/Colab_Notebooks/TSA')
sys.path.insert(0,str(TSA_ROOT/'Model/Common'))
from neighbor_data_qc import NeighborDataLoader
loader=NeighborDataLoader(project_root=TSA_ROOT,strict_hashes=True)
X,y,rows=loader.load_xy('F616_CNE_W24',horizon=6,split='train')

런타임 데이터는 기존v3표·전국국내sidecar·평가격자3개와 새 QCsidecar1개다.
기존 regional_data.py도 부모 로더로 필요하다. Common 폴더 전체를 함께 공유하는 편이 간단하다.
값 변환·보간·정규화는 로더가 하지 않는다. 정답은 별도로 반환하며 입력결측으로 평가행을 줄이지 않는다.
시험라벨은 allow_test=True를 명시해야 한다. 전체650캐시 prepare() 후에도 각조건의목록만 X로선택한다.

이 버전의학습 비교는2021–22학습/2023검증 참고 비교다. 최종독립시험의증거로 제시하지 않는다.
시드점수평균·표준편차와µg/m³시드예측앙상블을 구분한다. 기존공통채점기의평가행·음수·시간블록 규칙을 유지한다.
DCRNN 구조는 보존하고 별도데이터adapter/notebook을 Model/DCRNN/Neighbor_Expansion_QC_20261108에 제공한다.
