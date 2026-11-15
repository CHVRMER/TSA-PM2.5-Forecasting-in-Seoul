# DCRNN M1–M4T: 3·6·12·24·36·48시간 예측 / Vast.ai

기존 Graph WaveNet을 실행했던 Vast.ai 인스턴스의 `/workspace/TSA`를 사용합니다.
새 ZIP은 **코드 업데이트**입니다. Dataset을 다시 업로드하지 않으며 기존 rclone `gdrive` 인증을 사용합니다.
새 전용 경로만 설치하므로 이전 DCRNN·Graph WaveNet 코드·결과·데이터를 덮어쓰지 않습니다.

## 업로드와 실행

1. JupyterLab에서 `/workspace`를 열고 `TSA_DCRNN_VastAI_M1_M4T_H48_20261111.zip`을 업로드합니다.
2. JupyterLab의 **Terminal**에서 아래 명령을 차례로 실행합니다. 로컬 PC의 Git Bash가 아니라 Vast.ai 서버 터미널입니다.

```bash
python3 -m zipfile -e /workspace/TSA_DCRNN_VastAI_M1_M4T_H48_20261111.zip /workspace/TSA
cd /workspace/TSA/Model/DCRNN/VastAI_M1_M4T_H48_20261111
bash setup_vast.sh
bash run_checks.sh
bash run_pilot.sh --conditions M1 M3T
```

`run_checks.sh`가 성공하고 pilot의 두 조건이 `passed`이면 전체 여섯 조건을 실행합니다.

```bash
bash start_full.sh --final-test
```

- 순서: **M1 → M2 → M3 → M4 → M3T → M4T**, seed11, 각 조건에서 6개 예측시간 동시 학습.
- 각 조건 학습 종료 즉시 최적 체크포인트로 2023·2024 추론 → 공통 채점 → Drive 백업.
- 모든 조건 완료 후 권역 추가·궤적 추가의 **통합 비교**를 별도 채점합니다.
- `--final-test`를 생략하면 2023 검증만 추론·채점합니다. 시험까지 원하면 처음부터 붙이세요.
- tmux를 자동 생성하며 창·PC를 닫아도 서버가 살아 있는 동안 실행됩니다. Vast.ai 인스턴스 중지·삭제는 별개입니다.
- 분리: `Ctrl+B`를 누른 후 `D`. 다시 접속한 뒤 **같은 `cd`와 `bash start_full.sh --final-test`**를 실행하세요.
- 실행 중이면 기존 작업에 접속하고, 중단되었다면 가장 최근 **동일 설정의 H48 full** 체크포인트를 자동 검색합니다.
- 설정을 바꾸면 기존 체크포인트 재개를 거부합니다. 별도 실험을 시작하려면 설정을 정한 뒤 `--fresh`를 추가합니다.
- 4개 예측시간으로 훈련한 이전 DCRNN·GWN 체크포인트는 사용할 수 없습니다. 첫 H48 full은 새로 학습합니다.

## 학습 설정

| 항목 | 기본값 |
|---|---|
| 예측시간 | 3·6·12·24·36·48h |
| 이력 / 디코더 길이 | 48h / 48단계 |
| DCGRU | encoder2 / decoder2, hidden64, diffusion2 |
| 노드 / 그래프 | 중구 포함635, 기존 학습구간 수송 방향 그래프 고정 |
| Optimizer | Adam, LR0.001, eps0.001, weight decay0 |
| 손실 | 원단위 MAE, 정답 결측 마스킹 |
| LR scheduler | ReduceLROnPlateau, factor0.5, patience3, minLR0.00001 |
| 최적 epoch 선택 | 2023 **6h RMSE** |
| Early stopping / 최대 epoch | patience10 / 50 |
| Curriculum / scheduled sampling | decay200; train에만 정답 일부 허용, 추론은 항상 자유 실행 |
| Batch | effective64, micro8, eval8 |
| Gradient clipping / dropout / AMP | 5 / 0 / false |
| Seed | 11 하나 |

자세한 근거와 별도 비교 설정은 `MODEL_AND_TRAINING.md`를 확인하세요.
새 설정은 기존 성능을 보장하지 않으며 GPU 학습 결과를 확인해야 합니다.

## 입력 조건

모든 조건이 같은635개 노드와 그래프를 유지합니다. 제외 권역의 **원시 농도 입력을 결측으로 가린 후** train-only 정규화를 적합합니다. 제외 노드는 잠재 전달 노드로 남습니다.

| 조건 | 허용 외부 PM2.5 계열 | 궤적 요약 입력 |
|---|---:|---|
| M1 | 국내514 | 없음 |
| M2 | 국내514 + 서울600km 이내 중국21 =535 | 없음 |
| M3 | 국내514 + 중국77 =591 | 없음 |
| M4 | 국내514 + 중국77 + 일본42 + 몽골1 =634 | 없음 |
| M3T | M3와 같음 | 국내·중국만 |
| M4T | M4와 같음 | 모든 허용 국가 |

서울 중구 타깃은 위 외부 계열 수에서 제외합니다. 공통 서울·백령도 ASOS와 달력은 유지합니다.
문맥 입력은 모든 조건에서 같은750채널이며 사용하지 않는 궤적 칸은 고정 결측입니다.
일본·몽골이 도움이 되는지는 실험으로 확인하며, '도움이 없어야 정상'이라는 가정은 하지 않습니다.

## 결과 확인

서버:

```text
/workspace/TSA/Result/DCRNN/VastAI_H48/full/<실행ID>/
  batch_manifest.json
  console.log
  Conditions/<M1…M4T>/<조건실행ID>/
    Models/  History/  Exports/  run_summaries.json
  Common_Scoring/<조건>/validation/…
  Common_Scoring/<조건>/test/…
  Common_Scoring/Combined/validation/<채점ID>/
  Common_Scoring/Combined/test/<채점ID>/
```

Google Drive:

```text
내 드라이브/Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48/<실행ID>/
```

최종 조건 비교는 `Common_Scoring/Combined/test/<채점ID>/metrics.csv`와 `tests.csv`입니다.
`candidate − reference` RMSE가 음수면 개선입니다. `input_condition_comparison_plan.json`에 비교 방향을 저장합니다.
표준 출력은 기존과 같은 `metrics.csv`, `seed_metric_summary.csv`, `tests.csv`, `conditions.csv`, `score_rows.parquet`, `scoring_audit.json`입니다.
seed11 전체6조건·6예측시간 통합 비교는 기준선42 + 사전 지정 입력 비교42 =84개 검정을 함께 보정합니다.
조건별 채점보다 **통합 채점 결과**로 입력 조건 간 개선을 판단하세요.

## 체크포인트와 재개

로컬은 매 epoch 커밋, 별도 snapshot은5 epoch 간격입니다. Drive는1·5·10… epoch, 종료·조건 완료에 동기 복사·검증합니다.
학습·optimizer·scheduler·RNG·early stopping 상태를 함께 저장합니다. epoch 중단 시 마지막 **완전히 저장된 epoch**부터 복구하므로 일부 계산은 다시 합니다.
출력은 매 실행 별도 폴더이며, 기존 체크포인트를 새 실행으로 복사해 이어 갑니다.
자동 재개는 `/workspace`의 최근 호환 full을 찾습니다. 디스크가 사라졌다면 먼저 Drive의 **검증된 백업**을 복구해야 합니다(`GOOGLE_DRIVE_BACKUP.md`).
Drive 인증정보는 ZIP과 결과에 포함하지 않습니다.

## 평가와 검증 범위

- 기존3·6·12·24h 정답·평가행·기준선은 그대로 유지합니다.
- 36·48h 정답은 원시 중구 관측의 정확한 `t+h`이며 보간하지 않습니다. 연도 경계를 넘거나 결측이면 해당 지평 평가 제외입니다.
- Seasonal naive는36h에서 `t−12h`,48h에서 `t`입니다. `t+h−24`를 그대로 쓰면 미래를 보게 되므로 금지합니다.
- 확장 평가 격자는 Model의 `Cache/evaluation/`에 생성합니다. 기존 Dataset과 공통 채점기는 수정하지 않습니다.
- 학습2021–22 / 선택2023 / 시험2024. 2024 결과를 이미 여러 번 확인한 프로젝트이므로 완전히 새 블라인드 시험이라고 해석하지 않습니다.
- 기존 데이터 실제 검사와 CPU 가능한 로더·채점·재개 검증은 수행했습니다. 로컬에는 Torch가 없어 실제 모델의 텐서 계산·GPU 학습 검증은 수행하지 않았습니다. 서버 `run_checks.sh`와 pilot에서 실행합니다.
