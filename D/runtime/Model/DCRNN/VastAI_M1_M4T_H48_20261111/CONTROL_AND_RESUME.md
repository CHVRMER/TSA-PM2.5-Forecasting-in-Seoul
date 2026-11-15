# DCRNN H48 실행·재개·백업

명령은 Vast.ai의 Linux Terminal에서 실행한다. 로컬 Windows Git Bash에 입력하는 명령이 아니다. 이미 연결한 Google Drive rclone 설정은 그대로 재사용한다.

이 패키지는 3·6·12·24·36·48시간을 함께 예측한다. 기존 4개 예측시간 실험과 폴더·환경·체크포인트를 분리한다. 과거 4개 예측시간 DCRNN, Regional12, Graph WaveNet 체크포인트를 여기로 복사해 이어 학습하지 않는다. 출력층과 48시간 디코더가 달라 새 학습이 필요하다.

## 기본 실행

```bash
cd /workspace/TSA/Model/DCRNN/VastAI_M1_M4T_H48_20261111
bash setup_vast.sh
bash run_pilot.sh
bash start_full.sh --final-test
```

`start_full.sh --final-test`는 M1→M2→M3→M4→M3T→M4T 순서의 학습·검증·2024 추론·채점을 수행한다. `--final-test`를 생략하면 검증 결과까지 생성하고2024를 열지 않는다. 파일럿 결과는 본실험의 성능으로 사용하지 않는다.

`--detached`를 붙이면 tmux에 연결하지 않고 즉시 터미널로 돌아온다. 학습은 서버에서 계속된다. 터미널·브라우저·로컬 컴퓨터를 닫아도 서버 인스턴스가 계속 실행되는 동안 tmux 작업은 유지된다. 이 코드는 유료 인스턴스를 정지하거나 삭제하지 않는다.

## 중단 후 재개

같은 명령을 다시 실행한다.

```bash
bash /workspace/TSA/Model/DCRNN/VastAI_M1_M4T_H48_20261111/start_full.sh --final-test
```

- 같은 DCRNN 작업이 실행 중이면 중복 학습을 시작하지 않고 기존 tmux에 연결한다.
- 중단됐다면 가장 최근 본실험의 검증된 epoch 체크포인트와 조건별 재개 이력을 찾는다.
- epoch 중간 계산은 다시 수행할 수 있다. 저장된 epoch의 가중치뿐 아니라 optimizer·scheduler·RNG·최적 epoch·조기종료 카운터도 복원한다.
- 재개 실행은 새 실행ID에 저장하고, 이전 실행 폴더는 보존한다.
- 이미 필요한 추론·통합 채점까지 완료됐다면 학습을 반복하지 않는다.
- 체크포인트·데이터·그래프·시드·학습 설정이 맞지 않으면 조용히 다른 실험을 이어 붙이지 않고 중단한다.
- 이 H48 패키지의 체크포인트만 재개한다. 과거 4개 예측시간 DCRNN·Regional12·Graph WaveNet의 체크포인트를 가져오지 않는다.
- `max_epochs=50`, `patience=10`을 포함한 학습 설정을 바꾸면 기존 체크포인트를 자동으로 이어 쓰지 않는다. 변경 실험은 `--fresh`와 별도 출력 경로로 처음부터 시작한다.

완전히 다른 설정을 의도적으로 처음부터 시작할 때는 `--fresh`를 사용한다. 설정이 다른 실험은 별도 출력 경로로 분리하는 편이 명확하다.

```bash
TSA_DCRNN48_OUTPUT_ROOT=/workspace/TSA/Result/DCRNN/VastAI_H48_SeparateJob bash start_full.sh --fresh --final-test
```

## 화면과 중단

- `Ctrl+B`를 누른 다음 `D`: 화면만 분리. 학습은 계속된다.
- `Ctrl+B`를 누른 다음 `[`: 이전 로그 탐색. `q`로 탐색 종료.
- 마우스 휠: tmux의 이전 로그 탐색.
- `Ctrl+C`: 학습 중단 요청. 저장·Drive 백업 처리가 끝나고 프롬프트가 돌아올 때까지 기다린다.

같은 명령에 새 설정을 붙여 실행 중인 작업을 변경할 수는 없다. 다른 설정의 작업이 이미 실행 중이면 변경하지 않고 안내한다. 실행 중인 작업과 입력 조건·시드·`--final-test` 선택이 다를 때도 중단한다. 기존 작업이 끝난 뒤 같은 학습 설정에 `--final-test`를 추가하면 저장된 모델로 최종 추론·채점을 진행할 수 있다.

## 결과 경로

서버:

```text
/workspace/TSA/Result/DCRNN/VastAI_H48/full/<실행ID>/
```

Google Drive:

```text
내 드라이브/Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48/<실행ID>/
```

Drive 경로에는 `full` 단계가 없다. 각 조건의 추론·채점 직후,5epoch 간격,최종 완료 시 결과를 복사하고 검사한다. 로컬 epoch 체크포인트는 더 자주 기록된다. Drive의 최근 검증 백업 이후 계산은 디스크 손실 시 다시 수행할 수 있다.

기존 공통 채점 출력 파일 이름·열은 유지한다. 통합 비교는 `Common_Scoring/Combined/validation`과 `test`에서 확인한다. 필수 조건이 실패하면 남은 조건만으로 전체 검정 가족을 줄이지 않는다. 개별 완료 결과는 보존하지만 통합 완료로 표시하지 않는다.

## Graph WaveNet과 동시에 실행할 경우

이 H48 도우미는 Graph WaveNet 및 기존 4개 예측시간 DCRNN의 프로세스·잠금·tmux·결과 폴더를 자신의 것으로 취급하지 않는다. 단,GPU를 각각 지정해야 한다. 두GPU가 같은 인스턴스에 보이는 경우의 예:

```bash
CUDA_VISIBLE_DEVICES=1 bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/start_full.sh --final-test --detached
CUDA_VISIBLE_DEVICES=0 bash /workspace/TSA/Model/DCRNN/VastAI_M1_M4T_H48_20261111/start_full.sh --final-test --detached
```

이미 완료한 Graph WaveNet 작업은 기존 도우미가 완료로 판단해 다시 학습하지 않는다. 두 명령을 같은 GPU로 지정하면 메모리·연산 자원을 공유하게 된다. GPU 한 개짜리 인스턴스 두 대라면 각 인스턴스에서 모델 하나씩 실행하면 된다.

기존 Graph WaveNet 도우미는 다른 모델의 실행을 넓게 감지하는 구버전일 수 있다. 그 경우 Graph WaveNet을 먼저 실행한 뒤 이 새 DCRNN 도우미를 실행한다. 이 패키지는 기존 Graph WaveNet 파일을 수정하지 않는다.

이 DCRNN 패키지의 환경 변수는 `TSA_DCRNN48_PYTHON`, `TSA_DCRNN48_ENV_ROOT`, `TSA_DCRNN48_OUTPUT_ROOT`로 분리했다. `TSA_PROJECT_ROOT`, `RCLONE_CONFIG`, GPU 지정은 공유할 수 있다. 환경 변수를 설정하거나 인증 정보를 자동 생성하지 않는다.
