# Graph WaveNet H48 실행·자동 재개

명령은 Vast.ai JupyterLab의 Linux Terminal에 입력합니다. 기존 DCRNN이나 예전 Graph WaveNet 코드·프로세스·체크포인트·폴더를 변경하지 않습니다.

## 실행

```bash
cd /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
bash setup_vast.sh
bash run_checks.sh
bash run_pilot.sh --conditions M1 M3T
bash start_full.sh --final-test
```

각 단계가 성공했을 때 다음 명령을 실행합니다. pilot은 부분 자료로 두 epoch를 실행하여 입출력·학습·추론의 동작을 확인하며 최종 성능이나 전체 공통 채점으로 사용하지 않습니다. 기본 pilot은 Drive 백업을 생략합니다.

full은 M1→M2→M3→M4→M3T→M4T, seed11로 실행합니다. 예측시간은3·6·12·24·36·48h, 최대75epoch·patience10입니다. 각 조건 학습 완료 후 추론·채점·Drive 검증 백업을 즉시 수행하고, 모든 조건 완료 후 통합 비교를 저장합니다. `--final-test`를 생략하면2023검증까지만 수행합니다. full은 실제 초기 백업이 성공해야 학습을 시작합니다.

## 재접속·재개

```bash
cd /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
bash start_full.sh --final-test
```

- 이미 같은 작업이 실행 중이면 중복 실행하지 않고 기존 tmux 세션에 연결합니다.
- 중단된 작업은 가장 최근 **호환되는 GWN H48 full**의 검증된 체크포인트부터 이어갑니다.
- 가중치·optimizer·scheduler·RNG·best epoch·early stopping 카운터를 함께 복원합니다.
- 새 호출은 새 실행ID 폴더를 사용하며 이전 실행을 덮어쓰지 않습니다. 이전 체크포인트를 새 실행으로 복사한 뒤 이어갑니다.
- 학습 중인 입력조건·시드·설정·최종시험 여부가 현재 요청과 다르면 작업을 변경하지 않고 중단합니다.
- 설정을 바꾸면 기존 체크포인트와 자동으로 연결하지 않습니다. 기본75/10도 계약에 포함됩니다.
- 이 버전은 RF61,6개지평,문맥dropout0.2,mask-aware padding을 사용하는 새 실험입니다. 기존4개지평 GWN·DCRNN·Regional 체크포인트는 사용할 수 없습니다.
- 자동 검색 대상은 이 서버의 결과 폴더입니다. Drive 재연결만으로 체크포인트가 자동 다운로드되지는 않습니다.

별도 설정을 처음부터 실행할 경우에는 설정을 먼저 확정하고 별도 출력 경로를 사용합니다.

```bash
TSA_GWN48_OUTPUT_ROOT=/workspace/TSA/Result/GraphWaveNet/VastAI_H48_SeparateJob bash start_full.sh --fresh --final-test
```

## 화면 조작

- `Ctrl+B` 다음 `D`: 화면만 분리하고 학습은 계속합니다.
- `Ctrl+B` 다음 `[`: 이전 로그를 탐색합니다. `q`로 종료합니다.
- `Ctrl+C`: 학습 중단 요청입니다. 체크포인트·백업 정리가 끝나고 프롬프트가 돌아올 때까지 기다립니다.
- `--detached`: 시작 후 tmux에 붙지 않고 바로 터미널로 돌아갑니다.

브라우저·내 PC가 꺼져도 Vast.ai 인스턴스가 계속 실행되면 tmux 작업은 계속됩니다. 이 코드가 인스턴스를 정지·삭제하지는 않습니다.

## 결과 경로

```text
서버: /workspace/TSA/Result/GraphWaveNet/VastAI_H48/full/<실행ID>/
Drive: 내 드라이브/Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI_H48/<실행ID>/
```

Drive에는 `full` 중간 폴더가 없습니다. `Common_Scoring/Combined/test/<채점ID>/metrics.csv`와 `tests.csv`에서 최종 입력조건 비교를 확인합니다. 기본전체6조건·seed11의84개 검정을 함께 보정합니다. 실패한 조건이 있으면 남은 조건으로 가족을 축소하지 않고 통합 채점을 미완료로 표시합니다.

## 다른 모델과 구분

별도 환경은 `Environment/GraphWaveNet_VastAI_H48`, 별도 변수는 `TSA_GWN48_PYTHON`, `TSA_GWN48_ENV_ROOT`, `TSA_GWN48_OUTPUT_ROOT`입니다. DCRNN의 환경·잠금·tmux와 구분합니다.

인스턴스에GPU가2개 실제로 보일 때만 아래처럼 각각 지정할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=0 bash /workspace/TSA/Model/DCRNN/VastAI_M1_M4T_H48_20261111/start_full.sh --final-test --detached
CUDA_VISIBLE_DEVICES=1 bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112/start_full.sh --final-test --detached
```

GPU한개에서 두 모델을 함께 실행하면 같은 메모리·연산을 공유합니다. 위 명령은 GPU를 대여하거나 늘리는 명령이 아닙니다.
