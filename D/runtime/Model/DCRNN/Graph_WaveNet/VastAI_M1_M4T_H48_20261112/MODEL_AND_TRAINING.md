# Graph WaveNet H48 모델·학습 설명

## 반영한 변경

1. **여섯 예측시간:** 3·6·12·24·36·48h를 공동 학습합니다. 마지막 합성곱의 48개 직접 출력 채널에서 해당 여섯 개를 선택합니다. 순차적으로 미래를 생성하는 DCRNN decoder와 다릅니다.
2. **시간 수용영역 RF13 → RF61:** 블록 4개, 블록당 층 4개, kernel 2입니다. 각 블록의 dilation은 1·2·4·8이며 RF=1+4×(1+2+4+8)=61입니다. 시간·그래프 층은 8개에서 16개로 늘었습니다.
3. **결측 표시 패딩:** 실제 입력은 48h를 유지합니다. 앞의 13개 패딩은 정규화 PM 0, 결측 표시 1로 채웁니다. 관측된 평균 농도와 구분하기 위한 처리입니다.
4. **문맥 dropout 0.2:** `Linear750→32→ReLU→Dropout0.2→Linear6`을 모든 조건에 동일하게 적용합니다. 기존 그래프 합성곱 dropout 0.3은 따로 유지합니다.
5. **학습률 scheduler:** Adam LR0.001, eps1e-8, weight decay1e-4를 유지하면서 ReduceLROnPlateau(factor0.5, patience3, minLR1e-5)를 추가했습니다.
6. **학습 한도:** 최대 75 epoch, early stopping patience 10입니다. 최적 epoch와 scheduler의 기준은 2023년 6h RMSE입니다.

기본 채널 수 32/32/256/512와 학습형 그래프 embedding 10을 유지합니다. 기본 모델의 파라미터는 511,186개입니다.
이는 서울 PM2.5 작업에 맞춘 확장이므로 원 논문 설정을 그대로 재현한 실험으로 설명하지 않습니다.

## 그래프·입력·조건 비교

중구 포함 635노드와 기존 GWN 거리 그래프를 보존합니다. 거리 그래프와 학습형 adjacency를 함께 사용하며, 여섯 조건을 독립적으로 재학습합니다.
학습형 adjacency는 softmax(ReLU(E1E2))입니다. 예측 의존성을 나타내며, 매시각 바람으로 바뀌는 그래프나 오염 수송량·국가별 배출 기여율이 아닙니다.

이번 기본 실행은 M1–M4T의 정보 범위 비교를 유지합니다. 지역 gate·수송 그래프 교체·top-k 희소화는 별도 연구 실험에 해당합니다.
원시 PM 관측에 권역 mask를 적용한 뒤 2021–22 자료만으로 정규화합니다. 문맥은 ASOS·달력 53개와 경로 자리 322개로, 값·결측 채널을 합쳐 750입니다.
미사용 경로 변수는 상수 결측으로 가립니다. M3T의 관측 집계는 국내·중국, M4T는 전체 허용 국가로 제한합니다.

## 학습과 평가

- 학습 2021–22 / 선택 2023 / 시험 2024. 예측 시점은 3시간 간격입니다.
- 정답 결측은 채우지 않습니다. 여섯 지평의 원단위 MAE를 유효 정답에 대해 공동 학습합니다.
- log1p 및 train-only 정규화를 적용하고, 원단위로 복원해 채점합니다.
- effective batch64는 micro8의 gradient를 누적한 것입니다. BatchNorm이 보는 통계는 micro8과 노드·시간 축이므로 한 번에 64개를 넣은 것과 동일하지 않습니다.
- gradient clip5, AMPfalse, seed11입니다. 한 시드로 초기화 변동의 강건성을 입증하지는 못합니다.
- 6h RMSE로 최적 epoch를 고르므로 24·36·48h만 따로 최적화한 모델은 아닙니다.

기존 네 예측시간의 정답·eligible·기준선·metadata를 보존합니다. 36/48h 정답은 정확한 t+h 관측이며 연도 경계와 결측을 검사합니다.
36h seasonal naive는 t−12h, 48h는 t입니다. HAC 지연은 36h=11slot, 48h=15slot이며, 7일 달력 블록 bootstrap과 고정 평가 행 규칙을 유지합니다.
전용 채점기는 `Model/Common/GraphWaveNet_H48_20261112`에 둡니다. 원래 공통 채점기는 변경하지 않으며 표준 6종 출력의 열 순서·자료형을 유지합니다.
전체 여섯 조건·seed11은 baseline42+입력 비교42=84개 검정을 함께 보정합니다. 시드별 점수와 원단위 평균 예측 앙상블은 구분합니다.

## 별도 비교 설정

기본 명령은 `configs/default.json` 하나만 여섯 조건에 적용합니다. 아래 파일들은 자동으로 모두 실행하지 않습니다.

| 파일 | 기본 대비 변경 |
|---|---|
| configs/lr0003_comparison.json | LR0.0003 |
| configs/selection24_separate.json | 최적 epoch·scheduler 선택 지평 24h |

예:

```bash
TSA_GWN48_OUTPUT_ROOT=/workspace/TSA/Result/GraphWaveNet/VastAI_H48_Selection24 bash start_full.sh --fresh --final-test --config configs/selection24_separate.json
```

설정을 바꾸면 기존 체크포인트를 이어 쓰지 않습니다. 모든 입력 조건에 같은 설정을 적용하고, 설정 선택은 2023년 검증 자료로 수행합니다.

## 해석 범위

이전 GWN은 네 예측시간·RF13·문맥 dropout 없음·scheduler 없음·최대50 설정이었습니다.
새 버전은 여러 요소를 함께 바꾸므로 성능 차이가 나더라도 한 요소의 효과로 분리해 주장할 수 없습니다. 원인 분리는 별도 대조 실험이 필요합니다.
36·48h 추가는 정답 시점을 확장하는 것이며 미래 기상 실측이나 미래 도착 공기의 사후 역궤적을 입력하는 것은 아닙니다.
GDAS 재분석의 실제 배포 지연을 재현한 실시간 운영 예보 평가라고 주장하지 않습니다. 2024년은 이미 여러 차례 확인한 시험 연도입니다.

## 검증과 참고

References에 실제 데이터·거리 그래프·조건별 입력 보존 검사, CPU 학습·추론·중단 재개·공통 채점 결과를 보관합니다.
실제 635노드 크기의 CPU 순전파도 확인했습니다. GPU 전체 학습 성공이나 성능 개선을 보고하는 문서는 아닙니다.

- [원 논문](https://arxiv.org/abs/1906.00121)
- [저자 모델 구현](https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py)
- [저자 학습 구현](https://github.com/nnzhan/Graph-WaveNet/blob/master/engine.py)
- [ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
