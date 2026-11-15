# DCRNN H48 모델·학습 설명

## 유지한 구조와 확장한 출력

기존 DCRNN의 diffusion support 방향, 자기 연결, Chebyshev 확산, DCGRU cell, encoder2/decoder2, hidden64, diffusion2를 유지합니다. `vendor/dcrnn_model.py`는 기존 버전과 바이트까지 같습니다.
새 runner가 `horizons=(3,6,12,24,36,48)`을 명시적으로 전달하여 디코더를48단계로 돌리고 문맥 출력 head를6개로 만듭니다. 입력 이력은48시간, 노드635·학습기간 그래프5,942개 연결은 동일합니다.
원본 모델 파일의 기본 인자·일부 설명은4개 지평을 나타내지만 새 runner가 덮어써서 전달하는 실제 실행 설정이 기준입니다. `architecture_config`에 실제6개 지평/decoder_steps48이 기록됩니다.

수송 경로는 실측 오염원이나 국가별 질량 기여율을 나타내지 않습니다. 모든 조건에 같은 학습기간 정적 수송 그래프를 사용하므로, 여기서 평가하는 것은 **해당 그래프 위의 관측 가용성·궤적 요약 특징의 예측 가치**입니다.

## 권장 설정 변경

1. **Curriculum decay3000 →200.** 기존 짧은 학습에서는 teacher forcing 확률이 거의1로 유지될 수 있었습니다. train에서는 정답을 자주 보는데 실제 추론에서는 볼 수 없는 차이를 줄이도록 더 빠르게 자유 실행으로 이행합니다. optimizer update 수를 기준으로 `p=τ/(τ+exp(step/τ))`를 계산하며 확률을 매 epoch 기록합니다. 모든 decoder 단계의 정답을 주는 것이 아니라, 관측이 있는 중구의 선택된 지평에서만 **다음 입력**에 사용할 수 있습니다. 다른 노드의 미래 관측은 주지 않습니다.
2. **고정 milestone → ReduceLROnPlateau.** 선택 지평의 검증 RMSE 개선이 멈추면 factor0.5, patience3으로 LR을 줄입니다. 최소LR1e-5. early stopping patience10 안에서 작은 학습률로 회복할 기회를 줍니다. 정확히 몇 epoch에서 감소하는지는 검증 결과에 따라 달라집니다.
3. **LR0.001, max50, early stopping10 유지.** 과거 발산·일반화 문제를 고려해 원 논문의 더 큰 학습률로 되돌리거나 모델 크기를 확대하지 않습니다.
4. **Adam·weight decay0·dropout0 유지.** AdamW나 정규화가 이 데이터에서 더 낫다는 검증이 없으므로 여러 변경을 동시에 넣지 않았습니다. 구현에는 AdamW·weight decay·dropout 옵션이 있지만 별도 실험으로만 사용합니다.
5. **6h 주평가 유지.** 기본 체크포인트 선택과 scheduler는2023년6h RMSE를 사용합니다. 여섯 지평 손실은 모두 학습합니다. 장기 예측 중심의 다른 질문을 확인하려면 제공한24h 선택 설정을 별도 실험으로 사용합니다.

이 설정은 서울 작업에 맞춘 권장안이며, 원 논문 기본값을 그대로 재현한 결과라고 주장하지 않습니다.

## 기본 프로토콜

- 입력:48시간 ×635노드 ×(정규화log1p PM2.5, 결측 표시), ASOS·달력·허용 궤적 문맥.
- 정규화:2021–22만 적합. 제외 권역의 원시 PM 관측을 먼저 가립니다.
- 정답:정확한 t+3/6/12/24/36/48, 결측은 손실·평가에서 마스킹.
- 손실:원단위MAE; 지평별 유효 정답 수를 존중하는 기존 masked loss 유지.
- optimizer:Adam(lr=.001, eps=.001, weight_decay=0), clip norm5, effective batch64/micro8.
- AMPfalse, seed11. seed 하나로 초기화 변동의 강건성을 확인했다고 말할 수 없습니다.
- epoch 종료 때 free-running validation; `forward_train`은 검증·시험에서 금지됩니다.
- 36/48h 추가로 autoregressive decoder 연산·메모리가 증가합니다. 정확한 시간은 pilot/실제 로그로 확인합니다.
- 6개 지평을 공동 학습하므로 기존 네 지평 성능도 달라질 수 있습니다. 과거4지평 체크포인트는 호환되지 않습니다.

## 별도 비교용 설정

기본 full 명령은 자동으로 아래 조합을 모두 돌리지 않습니다. 비용을 늘리는 탐색을 막기 위해6입력조건에 `configs/default.json` 하나를 사용합니다.

| 파일 | 기본 대비 바꾸는 것 |
|---|---|
| configs/lr0003_comparison.json | LR0.0003 |
| configs/teacher3000_comparison.json | curriculum decay3000 |
| configs/selection24_separate.json | 검증·체크포인트 선택 지평24h |

예:기존 실행이 끝난 뒤 `bash start_full.sh --final-test --fresh --config configs/selection24_separate.json`.
실행 전 설정을 정하고 같은 설정을 모든 입력 조건에 적용합니다. 이전 설정 체크포인트를 이어 쓰지 않습니다.

## 채점 보존과 확장

표준6종 파일명·열 순서·자료형을 유지하는 private scorer를 `Model/Common/DCRNN_H48_20261111`에 둡니다. 원래 `Model/Common/score.py` 등은 변경하지 않습니다.
기존4지평33개 평가열은 정확히 보존합니다. 확장6지평 격자는70,128행(11,688 origins ×6)이며 지평별 정답·eligible을 검사합니다.
36/48h Seasonal naive는 예측시점 이전의 같은 일주기 관측(t−12/t)을 사용합니다. 정답·입력·기준선은 split 경계와 미래참조를 검사합니다.
7일 달력 블록 bootstrap, 지평별HAC(36h=11slot,48h=15slot), seed/ensemble 분리, 고정 평가 행 규칙을 유지합니다. 전체6조건 seed11 통합Holm family84개가 최종 조건 비교 기준입니다.

## 검증 기록

`References/`에 실제 입력·정답·그래프 감사, 합성 채점·재개·자동실행 검증 결과를 보관합니다.
Torch가 없는 로컬에서 실행할 수 있는 검사를 수행했고, 실제 텐서·모델 수치 검사는 서버 `run_checks.sh`에서 실행합니다. 이 문서는 GPU full 학습 성공이나 성능 개선을 보고하는 문서가 아닙니다.

참고: [DCRNN 원 논문](https://arxiv.org/abs/1707.01926), [저자 구현](https://github.com/liyaguang/DCRNN), [PyTorch ReduceLROnPlateau](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html).
