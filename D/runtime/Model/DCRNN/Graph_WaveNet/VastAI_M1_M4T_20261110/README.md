> **2026-11-11 종료 조건 변경:** 기본 full은 최대 50 epoch, patience 10입니다. [기존 체크포인트 재개와 적용 안내](EPOCH50_PATIENCE10_20261111.md)

> **2026-11-11 중단 처리 수정:** Ctrl+C 이후 저장·백업 종료를 기다리고 Bash 입력창으로 돌아옵니다. 휠 스크롤을 켭니다. [복구·업데이트 안내](INTERRUPT_AND_SCROLL_FIX_20261111.md)

> **2026-11-10 자동 재개 업데이트:** 일반 full 실행은 `bash start_full.sh`를 사용하세요. 최신 full 체크포인트 탐색·중복 실행 방지·tmux 연결을 한 번에 처리합니다. 자세한 내용은 [AUTO_RESUME_TMUX.md](AUTO_RESUME_TMUX.md)를 참고하세요.

# Vast.ai용 Graph WaveNet 저장·복구 보완

기존 M1/M2/M3/M4/M3T/M4T 실험과 모델 구조·손실·optimizer·평가 행·공통 채점기 출력은 유지합니다. 이 폴더는 **저장·복구·Google Drive 백업을 보완한 별도 버전**이며, 기존 Runpod 코드와 결과는 그대로 남습니다. Dataset을 바꾸거나 다시 전처리하지 않습니다.

## 무엇이 달라졌나

기존 코드도 매 epoch 종료 후 `last.pt`와 개선 시 `best.pt`를 저장했습니다. 이번에는 이 두 파일을 독립적으로 믿는 대신 **현재 학습 상태와 최적 모델을 한 저장 단위에 함께 기록**합니다.

| 시점 | 서버 디스크 | Google Drive |
|---|---|---|
| 매 epoch의 학습·검증 완료 | 현재 상태 저장, 최근 2개 epoch 보존 | 아래 주기 외에는 전송하지 않음 |
| 5·10·15·… epoch | 덮어쓰지 않는 스냅숏 보존 | 결과·체크포인트 복사 후 검증 |
| 첫 epoch | 현재 상태 저장 | 초기 복구 지점 확보 |
| 학습 종료·early stopping | 최종 저장 상태 고정 | 최종 체크포인트 백업 |
| 조건별 추론·채점 종료 | 예측·기존 공통 채점 결과 저장 | 해당 결과까지 백업 |
| 전체 완료 | 전체 조건 비교 저장 | 전체 결과 백업 |

full 시작 전에도 작은 초기 결과를 복사·검사하여 실제 Drive 쓰기 권한을 확인합니다. pilot은 기본적으로 로컬 동작 검사이며 Drive에 전송하지 않습니다.

## 체크포인트 내용과 무결성

`Models/<model>/seed11/Checkpoints/epoch_000005_.../`처럼 epoch마다 고유 폴더를 만듭니다. 각 폴더에는 다음이 들어 있습니다.

- `state.pt`: 현재 모델, optimizer, scheduler, AMP scaler, epoch·optimizer step·early-stopping 상태, 학습 이력, Python/NumPy/PyTorch/CPU·CUDA RNG 상태, 최적 모델의 별도 CPU 복사본·최적 epoch·검증 점수, 데이터·모델·학습 계약.
- `manifest.json`: 크기·SHA256·epoch·최적 epoch·계약 서명·스냅숏 여부.
- `COMMITTED.json`: 완전히 저장된 단위임을 식별하는 표식.

임시 폴더에 기록하고 파일 디스크 동기화·체크섬 확인 후 최종 폴더로 전환합니다. 포인터가 오래되었거나 최신 저장본이 손상되면 검증 가능한 이전 저장본을 찾고 복구 내역을 기록합니다. 모든 저장본이 손상되거나 데이터·설정이 다르면 조용히 처음부터 학습하지 않고 중단합니다.

재개 및 최적 모델 추론은 **같은 검증된 저장본 안의 현재 상태·최적 가중치**를 사용합니다. `last.pt`·`best.pt`도 편의상 내보내지만, 이 파일만 독립적으로 복사해서 엄격한 재개를 할 수는 없습니다. `CURRENT.json`도 힌트일 뿐이며 실제 저장 단위의 체크섬을 확인합니다.

최근 2개 epoch 이외의 일반 저장본은 같은 실행의 새 저장이 완료된 뒤 정리하지만, 5 epoch 간격·최종 고정 상태는 보존합니다. 정리는 해당 실행의 체크포인트 폴더 안에서만 수행합니다. Drive의 이전 파일·실행은 삭제하거나 `sync`하지 않으므로 장기적으로 저장 용량이 늘어납니다.

## 중단 시 어디서 이어지나

- 한 epoch 도중 중단: 마지막으로 **학습·검증·저장이 모두 완료된 epoch 다음**부터 시작합니다. 저장되지 않은 epoch의 일부 작업은 다시 수행합니다.
- Stop/Restart 후 디스크가 남아 있는 경우: 마지막 정상 서버 저장본을 재사용합니다.
- 인스턴스 디스크 소실: Google Drive에서 복사·검증까지 완료된 백업으로 복구합니다. 보통 마지막 백업 이후 최대 5개 epoch의 작업이 필요할 수 있습니다. 백업 실패·네트워크 장애가 있었다면 해당 보장은 적용되지 않습니다.
- SIGINT/정상 SIGTERM 요청은 가능한 경우 최신 완료 상태의 백업을 시도합니다. 강제 종료·전원 차단 때 새 백업 실행을 보장하지 않습니다.

가중치뿐 아니라 optimizer·scheduler·RNG를 복원합니다. 같은 데이터·설정·라이브러리·환경에서 이어가는 것이 원칙이며 GPU 종류나 수치 연산 환경이 바뀌어도 항상 비트 단위로 동일한 결과가 나온다고 보장하지 않습니다.

새 버전은 검증된 generation 체크포인트로 재개합니다. 기존 Runpod의 별도 `last.pt`/`best.pt`는 기존 실행기로 계속 사용할 수 있으며 이 버전으로 자동 변환하지 않습니다.

## Drive 자동 백업

기본 경로:

```text
내 드라이브/Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI/<실행 ID>/
```

인스턴스의 기존 rclone `gdrive` 연결을 사용합니다. 사용자 인증 파일은 코드나 ZIP에 넣지 않습니다. 설정 방법은 `GOOGLE_DRIVE_BACKUP.md`를 따르세요. 아직 실제 계정 인증이나 Drive 전송을 실행한 상태는 아닙니다.

백업은 `rclone copy` 후 `rclone check --one-way`로 검사합니다. 검증이 끝나야 성공으로 기록합니다. 검증된 파일 목록·SHA256를 담은 불변 `Backup_receipts/Published/<ID>/verified.json`도 별도로 전송·검증합니다. 백업이 실패하면 full은 중단하여 로컬 저장만 된 상태를 클라우드 저장 완료로 오인하지 않게 합니다.

원격 실행 폴더는 다음 백업 때 새 완료 상태로 갱신될 수 있습니다. 마지막 전송이 도중 실패했다면 예전 성공 표식만 보고 폴더 전체가 완전하다고 가정하지 마세요. 복구 도구는 선택한 표식에 포함되어 있고 실제 해시가 일치하는 체크포인트만 새 폴더로 복사합니다. 원본 백업·이전 결과는 보존합니다.

## 설치·실행

`VAST_START_HERE.txt`를 먼저 읽으세요. 기존 큰 Runpod ZIP의 Dataset·공통 코드가 필요하며 이번 작은 ZIP을 같은 프로젝트 루트에 추가로 풉니다.

```bash
cd /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110
bash setup_vast.sh
bash run_pilot.sh
# GOOGLE_DRIVE_BACKUP.md에 따라 인증·쓰기 검사를 마친 후
bash run_full.sh
```

2024 최종 추론·채점은 `bash run_full.sh --final-test`입니다. 기본 full은 2023 검증까지입니다.

이전 서버 결과가 있다면:

```bash
bash run_full.sh --resume-source /workspace/TSA/Result/GraphWaveNet/VastAI/full/이전_실행ID
```

재개도 새 실행 폴더·새 Drive 폴더를 만듭니다. 이전 폴더를 덮어쓰지 않습니다. 기본 6조건·seed11·50 epoch·patience10과 원래 학습 설정을 유지합니다.

## 결과·백업 상태 확인

서버 결과: `/workspace/TSA/Result/GraphWaveNet/VastAI/`.

- `batch_manifest.json`: 조건별 진행·실제 학습/채점 경로.
- `Models/.../seed11/Checkpoints/`: 재개·스냅숏 원본.
- `checkpoint_recovery.json`: 실제로 복구한 epoch와 손상 우회 여부.
- `Backup_receipts/last_successful.json`: 마지막 백업 성공 시각·이유·원격 경로.
- `Backup_receipts/Published/`: 검증된 원격 복사본의 파일 목록·SHA256.
- `[Drive backup] verified:`: 복사·검증 성공 로그. 로컬 `checkpoint saved`와 구분합니다.

공통 채점기의 기존 `metrics.csv`, `seed_metric_summary.csv`, `tests.csv`, `conditions.csv`, `score_rows.parquet`, `scoring_audit.json` 양식은 그대로입니다.

## 검증 범위

강제 프로세스 종료·최신 저장 손상·설정 불일치·스냅숏 보존·원본 보존 복사를 점검했습니다. 실제 학습기를 작은 CPU 시계열에 연결하여 8 epoch 연속 학습과 3 epoch 후 중단·재개 학습의 상태·예측을 비교했습니다. Drive 코드는 가짜 전송 프로세스로 실패·성공·검증 순서를 시험했습니다. 실제 Google OAuth·Drive 전송·Vast GPU 학습은 수행하지 않았습니다. 자세한 결과는 `References/verification_summary.json`에 기록합니다.

공식 저장소 설명: https://docs.vast.ai/guides/instances/storage/types
Google Drive 연결: https://rclone.org/drive/
