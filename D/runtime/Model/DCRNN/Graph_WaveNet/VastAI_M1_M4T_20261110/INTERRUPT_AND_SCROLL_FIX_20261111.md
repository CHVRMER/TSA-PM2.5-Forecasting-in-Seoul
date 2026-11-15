# Ctrl+C 이후 입력창과 스크롤 복구

2026-11-11 수정. 기존 실행 도우미의 supervisor가 Ctrl+C로 종료되고 `remain-on-exit` 설정에 의해 dead pane이 남던 문제를 수정했다. 데이터·모델·그래프·학습 설정·공통 채점기·체크포인트 형식은 변경하지 않는다.

## 현재 멈춰 보이는 화면에서 빠져나오기

`Pane is dead`는 그 tmux 패널의 프로그램이 종료됐다는 뜻이다. 일반 명령은 받지 않지만 tmux 단축키는 사용할 수 있다.

- Ctrl+B를 누르고 손을 뗀 다음 D: tmux에서 나와 이전 Terminal로 돌아간다.
- 반응이 없으면 JupyterLab Launcher에서 새 Terminal을 연다.
- 이전 기록: Ctrl+B를 누르고 손을 뗀 다음 [: copy mode에 들어가 PageUp/방향키로 조회한다. q 또는 Esc로 나온다.
- 기존 dead pane은 기록 보존을 위해 삭제하지 않아도 된다. 인스턴스나 Jupyter 서버를 재시작할 필요 없이 새 Terminal로 복구할 수 있다.

## 업데이트

`TSA_GraphWaveNet_InterruptFix_20261111.zip`을 `/workspace`에 업로드하고 새 Terminal에서 한 번 실행한다.

```bash
python3 -m zipfile -e /workspace/TSA_GraphWaveNet_InterruptFix_20261111.zip /workspace/TSA
```

그다음 최신 full 체크포인트를 자동으로 찾아 학습·최종 추론·공통 채점까지 진행:

```bash
bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/start_full.sh --final-test
```

2024 최종시험을 원하지 않으면 `--final-test`를 생략한다. 이미 학습 프로세스가 있으면 중복 실행을 차단한다. 아직 이전 작업이 종료·백업 처리 중이라면 기다린다.

## 수정된 동작

1. 학습 파이프라인을 터미널의 Ctrl+C에서 분리한다. 로그를 저장하는 tee가 함께 종료되는 일을 막는다.
2. Ctrl+C를 받으면 자신이 시작한 학습 Python 프로세스에만 종료 요청을 전달한다. 다른 작업이나 tmux 서버를 종료하지 않는다.
3. `[stopping]` 메시지 이후 학습 프로세스가 상태 기록·중단 백업을 처리하고 실제로 종료할 때까지 기다린다. 백업 실패가 있다면 위 로그의 실패 메시지를 확인해야 한다.
4. 정상 종료·오류·중단 이후 사용 가능한 Bash 프롬프트로 돌아간다. 이미 종료된 이전 버전 패널은 소급 변경하지 않는다.
5. 새 실행과 재연결 시 세션의 mouse 옵션을 켜서 휠 스크롤을 지원한다. copy mode에서 q 또는 Esc로 빠져나오면 다시 입력할 수 있다.

학습을 유지하고 화면만 닫을 때는 Ctrl+C 대신 Ctrl+B → D를 사용한다. `[stopping]` 중에는 종료 처리가 완료될 때까지 기다린다.

로컬 볼륨은 마지막 완료 epoch 저장본으로 이어간다. Drive는 마지막으로 검증 완료된 백업까지만 복구 가능하다. 사용자가 제공한 로그에는 epoch 30의 Drive 검증 백업이 있으므로, 해당 저장본이 정상이라면 epoch 31부터 재개할 수 있다. 최적 모델 가중치와 현재 학습 상태는 기존 저장 방식대로 각각 복원된다.

## 전체 기록을 파일로 보기

사용자가 제공한 배치의 콘솔 로그를 새 Terminal에서 조회:

```bash
less /workspace/TSA/Result/GraphWaveNet/VastAI/full/20261110_192802_468013_928db0a5/console.log
```

PageUp/PageDown/방향키로 이동하고 q로 나온다. tmux 화면에 남아 있는 기록보다 로그 파일을 우선 확인한다. history-limit 설정은 이미 버려진 과거 스크롤 기록을 복원하지 않는다.

## 검증 범위

기존 자동 재개·잠금·tmux 제어 검증에 더해 중단 신호 전달 대상, 반복 중단, 시작 전 중단 요청, 종료 대기, 상태 기록, Bash 복귀를 회귀 테스트했다. 신호·tmux 경계는 모의 테스트했으며 실제 Vast.ai 인스턴스에서 패치를 실행하거나 GPU 작업을 시작하지는 않았다.

참고: https://man.openbsd.org/tmux
