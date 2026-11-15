> 2026-11-11: 기본 종료 조건은 50/10입니다. [정책 변경·재개 안내](EPOCH50_PATIENCE10_20261111.md)를 참고하세요.

> 2026-11-11: [Ctrl+C·스크롤 수정 안내](INTERRUPT_AND_SCROLL_FIX_20261111.md)를 적용하세요. 실행 명령과 자동 재개 기능은 같습니다.

# 자동 재개와 tmux 실행

업데이트: 2026-11-10. 앞으로 일반 full 실행은 `run_full.sh` 대신 `start_full.sh`를 사용한다. 데이터·모델 구조·그래프·학습 설정·공통 채점기·출력 양식은 바뀌지 않는다.

## 업데이트 설치

`TSA_GraphWaveNet_AutoResume_Tmux_20261110.zip`을 Vast.ai JupyterLab의 `/workspace`에 업로드한 뒤 서버 Terminal에서 실행한다.

```bash
python3 -m zipfile -e /workspace/TSA_GraphWaveNet_AutoResume_Tmux_20261110.zip /workspace/TSA
```

코드·설명서·무결성 목록만 갱신한다. `Dataset`, `Result`, 체크포인트, Google 인증 파일은 ZIP에 포함되지 않으며 변경하지 않는다. 이미 진행 중인 이전 학습을 종료할 필요는 없다. 새 실행 명령이 기존 프로세스를 발견하면 중복 학습을 시작하지 않는다.

## 매번 사용할 한 명령

Vast.ai JupyterLab의 Terminal에서 다음 명령을 실행한다. 내 PC의 Git Bash를 쓴다면 먼저 SSH로 Vast.ai 서버에 접속한다.

```bash
bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/start_full.sh
```

동작:

1. 이미 같은 프로젝트의 pilot/full 학습이 실행 중이면 새 학습을 만들지 않는다. 기존 학습이 tmux 안에 있으면 그 화면에 연결한다. tmux 밖에서 시작한 기존 학습은 그대로 유지하고 PID를 알려준다.
2. 아직 실제 학습 프로세스가 뜨지 않은 tmux 준비 단계도 중복 실행으로 판단한다.
3. 이전 full 체크포인트가 없으면 첫 full을 시작한다. pilot 저장본은 사용하지 않는다.
4. 이전 full이 중단됐다면 최신 full 실행에서 조건별 체크포인트를 찾는다. 재개한 배치가 M1에서 끊겼다면 이전 배치 기록을 따라 M2 이후 저장본도 찾는다.
5. 마지막 완료 epoch 다음부터 이어간다. optimizer·RNG·early stopping·최적 모델 복원은 기존 trainer가 수행한다. epoch 중간 batch부터 이어지는 방식은 아니다.
6. 학습·공통 채점까지 이미 끝났다면 `[completed]`를 출력하고 재학습하지 않는다.
7. 매번 결과는 새 실행 폴더에 저장하며 이전 결과는 덮어쓰지 않는다.

기본 조건 순서는 M1 → M2 → M3 → M4 → M3T → M4T, seed 11, 최대 50 epoch, patience 10이다. 2024는 기본 실행에서 열지 않는다.

tmux가 없으면 root/apt-get 환경에서 최초 한 번 자동 설치한다. 일반 사용자 환경이면 설치가 필요하다는 오류를 출력한다. 터미널이 비대화형이면 detached로 시작하고 연결 명령을 출력한다.

## 연결 끊기와 재접속

Ctrl+B를 누르고 손을 뗀 다음 D를 누르면 tmux에서 빠져나오고 서버 학습은 계속된다. 브라우저·내 PC를 꺼도 서버가 켜져 있으면 계속된다. 다음 접속 때 같은 `start_full.sh` 명령으로 다시 연결한다.

Ctrl+C는 학습 중단이다. 인스턴스 정지·삭제·컨테이너 재시작·호스트 장애는 tmux도 견디지 못한다. GPU 비용은 PC/브라우저 종료와 무관하게 서버가 켜진 동안 계속 발생하며 자동 정지 기능을 추가하지 않았다.

## 선택 옵션

화면에 붙지 않고 백그라운드로 시작:

```bash
bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/start_full.sh --detached
```

2023 검증 후 설정을 동결한 다음 2024 최종 추론·채점까지 진행:

```bash
bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/start_full.sh --final-test
```

이미 학습이 진행 중이면 이 옵션을 붙여도 중복 실행하거나 현재 작업의 설정을 바꾸지 않는다. 현재 작업이 끝난 뒤 같은 옵션으로 다시 실행한다.

체크포인트를 사용하지 않는 별도의 새 실험을 의도할 때만:

```bash
bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/start_full.sh --fresh
```

`--config`, `--conditions`, `--seeds`도 지원한다. 데이터·모델·학습 설정이나 seed가 기존 full과 다르면 자동 재개를 거부한다. 설정이 다른 실험은 `--fresh`로 분리한다. 원래 trainer의 데이터·그래프·전처리·라이브러리 계약도 복원 전에 다시 검증한다.

## 저장 위치와 백업

- 학습 결과: `/workspace/TSA/Result/GraphWaveNet/VastAI/full/<새 실행ID>/`
- 실행 제어 기록: `/workspace/TSA/Result/GraphWaveNet/VastAI/Control/Invocations/<실행ID>/`
- 현재 tmux 상태: 위 `Control/tmux_status.json` (현재 상태를 가리키는 포인터)
- 자동 재개 출처: 각 제어 기록의 `resume_plan.json` 및 결과의 `batch_manifest.json`
- 학습 콘솔 로그: 기존 `Launch_full_<ID>/console.log`와 `full/<ID>/console.log`
- Drive 결과: `내 드라이브/Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI/<실행ID>/`

기본 Google 인증 경로 `$HOME/.config/rclone/tsa-vast.conf`가 있으면 자동으로 사용한다. 사용자 지정 `RCLONE_CONFIG`, Python 환경·PATH 등 실행 경로 설정도 tmux에 전달한다. 인증 토큰 자체를 실행 기록에 복사하지 않는다.

로컬 매 완료 epoch 저장, epoch 1·5·10…·학습/조건/배치 종료 시 Drive 백업 정책은 그대로다. `/workspace` 볼륨까지 잃으면 Drive의 마지막 검증 백업을 먼저 복원해야 한다. 이 도우미는 Drive의 체크포인트를 자동 다운로드하지 않는다.

최신 저장본 하나만 손상됐으면 해당 실행의 이전 정상 generation으로 복구한다. 모든 저장본이 손상됐거나 설정·계약이 다르면 오류를 내고 중단한다. 이전 진행 상황을 무시하고 조용히 새 학습을 시작하지 않는다.

## 검증 범위

조건별 계보·손상·설정·완료 판정, 실제 두 프로세스의 잠금 경합과 강제 종료 후 잠금 해제, tmux 생성·재연결 제어와 환경변수 전달을 검증했다. 로컬 Windows에서 tmux 호출은 모의 테스트했으며 실제 Vast.ai 서버에서 새 도우미를 실행하거나 GPU 학습을 시작하지는 않았다.
