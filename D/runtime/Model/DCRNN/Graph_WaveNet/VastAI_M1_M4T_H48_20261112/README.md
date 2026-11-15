# Graph WaveNet H48 — Vast.ai 실행 안내

서울 중구 PM2.5를 3·6·12·24·36·48시간 후에 예측합니다. 최대 75 epoch, early stopping patience 10, seed 11을 사용합니다.
기존 데이터·DCRNN·Graph WaveNet·공통 채점기·실험 결과를 보존하는 별도 버전입니다.

## 1. 업로드

`TSA_GraphWaveNet_VastAI_M1_M4T_H48_20261112.zip`을 Vast.ai JupyterLab의 `/workspace`에 올립니다.
현재 DCRNN H48을 실행하는 같은 인스턴스라면 **Dataset과 Google Drive 인증을 재사용**합니다.

## 2. 설치·검사·pilot

아래 명령은 Vast.ai의 Terminal에 한 줄씩 입력합니다. 각 단계가 성공하면 다음으로 진행합니다.

```bash
python3 -m zipfile -e /workspace/TSA_GraphWaveNet_VastAI_M1_M4T_H48_20261112.zip /workspace/TSA
cd /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112
bash setup_vast.sh
bash run_checks.sh
bash run_pilot.sh --conditions M1 M3T
```

`setup_vast.sh`는 템플릿의 CUDA PyTorch를 교체하지 않고 별도 환경에 필요한 패키지를 준비합니다.
`run_checks.sh`에는 실제 Torch 계산·미래 정답 불사용·체크포인트 재개·공통 채점 검사가 포함됩니다.
Pilot은 작은 표본으로 2 epoch를 실행하며 2024 시험과 정식 고정 행 채점은 수행하지 않습니다.

### 새 인스턴스에 공유 입력이 없는 경우

이전 보충 파일 `TSA_DCRNN_H48_Required_Data_20261111.zip`을 `/workspace`에 올려 재사용합니다.
새 GWN 설치기가 이 파일의 기존 데이터·공통 코드 40개를 검증합니다. DCRNN H48 모델 ZIP을 별도로 설치할 필요는 없습니다.

```bash
python3 Codes/install_shared_dependencies.py --archive /workspace/TSA_DCRNN_H48_Required_Data_20261111.zip --project-root /workspace/TSA
bash setup_vast.sh
```

동일한 기존 파일은 유지하고 누락 파일만 추가합니다. 내용이 다르면 덮어쓰지 않고 중단합니다.
원본 GDAS1·ERA5·HYSPLIT 실행파일은 필요하지 않습니다.

## 3. 전체 학습·추론·채점

기존 Drive 설정을 확인한 뒤 실행합니다. 설정 파일에는 인증정보가 있으므로 공유 ZIP에 넣지 않습니다.

```bash
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
rclone listremotes
rclone lsd "gdrive:Colab_Notebooks/TSA/Result"
bash start_full.sh --final-test
```

- 순서: **M1 → M2 → M3 → M4 → M3T → M4T**, seed 11.
- 각 조건은 여섯 예측시간을 한 모델에서 공동 학습합니다.
- 조건 하나가 끝나면 최적 가중치로 2023·2024 추론, 공통 채점, Drive 백업을 수행합니다.
- 전체 완료 후 여섯 조건의 비교를 통합 채점합니다.
- `--final-test`를 생략하면 2023 검증까지만 진행합니다.
- **75는 최대값**입니다. 검증 오차가 10 epoch 동안 개선되지 않으면 더 일찍 종료할 수 있습니다.

tmux를 자동으로 생성합니다. 화면만 분리하려면 **Ctrl+B → D**입니다. 서버가 실행되는 동안 브라우저나 내 PC를 닫아도 계속 학습합니다.
중단은 Ctrl+C를 누른 뒤 저장·백업이 끝날 때까지 기다립니다. 다시 연결하거나 재개할 때도 같은 명령을 사용합니다.

```bash
cd /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
bash start_full.sh --final-test
```

실행 중인 동일 작업이 있으면 연결하고, 중단됐으면 최근 호환 H48 full 체크포인트를 찾아 새 실행 폴더에서 재개합니다.
이전 네 예측시간 GWN 및 DCRNN 체크포인트는 사용할 수 없습니다. Drive 연결만으로 체크포인트가 자동 다운로드되지는 않습니다.
디스크 손실 후 복구 절차는 `GOOGLE_DRIVE_BACKUP.md`를 참고합니다.

## 4. 설정

| 항목 | 기본값 |
|---|---|
| 예측시간 | 3·6·12·24·36·48h |
| 실제 입력 이력 | 48h |
| 시간 수용영역 | RF61, 4 blocks × 4 layers, kernel 2 |
| 앞쪽 13개 패딩 | 정규화 PM=0, 결측 표시=1 |
| 채널 | residual32 / dilation32 / skip256 / end512 |
| 그래프 | 같은 635노드, 기존 거리 그래프 + 학습형 그래프 |
| graph / context dropout | 0.3 / 0.2 |
| Optimizer | Adam, LR0.001, eps1e-8, weight decay1e-4 |
| Scheduler | ReduceLROnPlateau, factor0.5, patience3, minLR1e-5 |
| 최적 epoch 선택 | 2023년 6h RMSE |
| 최대 epoch / early stopping | 75 / 10 |
| effective / micro / eval batch | 64 / 8 / 8 |
| Loss / gradient clip / AMP | 원단위 MAE / 5 / false |
| Seed / 파라미터 수 | 11 / 511,186개 |

직접 예측 방식이며 teacher forcing이나 autoregressive decoder를 사용하지 않습니다.
자세한 근거와 해석 한계는 `MODEL_AND_TRAINING.md`에 있습니다.

## 5. 입력 조건

| 조건 | 허용 외부 PM 계열 | 추가 경로 문맥 |
|---|---:|---|
| M1 | 국내514 | 없음 |
| M2 | 국내514 + 가까운 중국21 =535 | 없음 |
| M3 | 국내514 + 중국77 =591 | 없음 |
| M4 | M3 + 일본42 + 몽골1 =634 | 없음 |
| M3T | M3와 같음 | 국내·중국 218개 |
| M4T | M4와 같음 | 전체 허용 국가 322개 |

중구 타깃은 외부 계열 수에서 제외합니다. 모든 조건의 노드 순서·고정 거리 그래프·모델 차원·문맥 750채널을 유지합니다.
제외 관측은 정규화 전에 가리며, 제외 노드는 잠재 전달 노드로 남습니다. 학습형 그래프의 가중치는 조건별 학습 결과에 따라 달라질 수 있습니다.
M3T는 원시 관측 집계부터 국내·중국으로 제한합니다. 국가별 대기오염 기여율을 추정하는 실험은 아닙니다.

## 6. 결과 확인

서버:

```text
/workspace/TSA/Result/GraphWaveNet/VastAI_H48/full/<실행ID>/
```

Google Drive:

```text
내 드라이브/Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI_H48/<실행ID>/
```

조건별 가중치·예측은 `Conditions/<조건>/<조건실행ID>/Models`, `History`, `Exports`에 저장합니다.
조건별 채점은 `Common_Scoring/<조건>/`, 최종 조건 비교는 **`Common_Scoring/Combined/test/<채점ID>/`**의 `metrics.csv`, `tests.csv`에서 확인합니다.
RMSE 비교는 candidate−reference로 음수면 개선입니다.

표준 출력 형식은 유지합니다: `metrics.csv`, `seed_metric_summary.csv`, `tests.csv`, `conditions.csv`, `score_rows.parquet`, `scoring_audit.json`.
전체 여섯 조건·seed11의 84개 검정을 함께 Holm 보정하므로 입력 조건의 최종 판단에는 통합 채점을 사용합니다.

## 7. 보존·검증

로컬은 매 epoch 저장하며 5 epoch마다 별도 snapshot을 만듭니다. Drive는 1·5·10… epoch와 종료·조건 완료 때 복사·검증합니다.
optimizer·scheduler·RNG·최적 epoch·조기종료 상태까지 복구합니다. 매 실행은 별도 폴더에 저장합니다.

정답은 정확한 t+h 실측이며 채우지 않습니다. 기존 네 예측시간의 평가 행·기준선을 보존합니다.
36h seasonal naive는 t−12h, 48h는 t로 미래 참조를 방지합니다. 기존 공통 채점기와 DCRNN은 수정하지 않습니다.

실제 데이터 전체 점검, CPU 수치 계산·학습·재개·통합 채점 검사와 635노드 크기의 CPU 추론 검사를 수행했습니다.
실제 635노드 GPU 전체 학습과 성능 개선은 아직 검증하지 않았습니다. 서버 pilot 후 full을 실행하세요.
