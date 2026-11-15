# D 모델 실행 안내 — Vast.ai / Linux GPU

저장소의 `D/runtime`를 프로젝트 루트로 사용해 DCRNN 또는 Graph WaveNet을 실행하는 방법입니다. 이 안내의 명령은 **Vast.ai JupyterLab의 Terminal 또는 Linux 서버 SSH 터미널**에서 실행합니다. 저장소 복사와 문서 확인에는 로컬 Git Bash를 사용할 수 있습니다.

새 GPU를 임대하거나 기존 모델을 자동 실행하는 문서는 아닙니다. 먼저 데이터·인증·GPU 환경을 준비한 다음 pilot을 확인하고 full을 시작합니다.

## 1. 필요한 파일과 데이터

공유한 `D/runtime/Model`에는 모델 코드·설정·기존 데이터 어댑터·공통 채점기가 있습니다. 대형 Dataset, 학습 가중치, 개인 rclone 인증파일은 Git에 포함하지 않습니다.

아래처럼 프로젝트 루트 바로 아래에서 Dataset을 찾을 수 있어야 합니다.

```text
<저장소>/D/runtime/
  Model/
  Dataset/
    China/
    China_Border_Proxy/
    Japan/
    Korea_Air/
    Korea_East_Candidates/
    Korea_National_2021_2024/
    Mongolia/
    Evaluation_Contract_20261107/
    Neighbors_Expansion_QC_20261108/
    Origin_Features_2021_2024/
    Origin_Features_2021_2024_v3/
    Trajectory_Matching_2021_2024/
    Trajectory_Matching_2021_2024_v3/
    Trajectory_Matching_Expansion_QC_20261108/
    Trajectory_Matching_National_v4/
```

필요한 개별 파일과 SHA-256은 각 모델의 `References/upload_manifest.json`에 기록되어 있습니다. DCRNN H48 기준 Dataset 의존성은 31개, 약 601 MB입니다. 원본 GDAS1이나 HYSPLIT 실행 프로그램은 이 학습 입력을 읽는 데 필요하지 않습니다.

기존 Vast.ai 인스턴스의 `/workspace/TSA/Dataset`을 재사용할 경우, 다음 예처럼 연결할 수 있습니다. 아래는 저장소가 `/workspace/TSA-PM2.5-Forecasting-in-Seoul`에 있다는 예시입니다. 실제 clone 위치가 다르면 첫 줄만 바꾸세요.

```bash
cd /workspace/TSA-PM2.5-Forecasting-in-Seoul
export TSA_PROJECT_ROOT="$PWD/D/runtime"

test -d /workspace/TSA/Dataset
test ! -e "$TSA_PROJECT_ROOT/Dataset"
test ! -L "$TSA_PROJECT_ROOT/Dataset"
ln -sT -- /workspace/TSA/Dataset "$TSA_PROJECT_ROOT/Dataset"
```

각 `test`가 성공한 것을 확인하고 다음 줄로 진행합니다. 이미 올바른 Dataset 폴더나 연결이 있다면 `ln -s`를 다시 실행하지 않습니다. 원본 Dataset을 이동하거나 덮어쓰지 않습니다. 실행기는 별도의 `--dataset-root` 옵션 대신 항상 `TSA_PROJECT_ROOT/Dataset`을 읽습니다.

## 2. Google Drive 백업 인증

pilot은 기본 설정에서 Drive 없이 실행할 수 있지만, full은 검증 백업을 요구합니다. 각 실행자가 자신의 rclone 인증을 서버의 개인 파일에 준비해야 합니다. 인증 토큰·비밀번호·`rclone.conf`는 저장소나 결과 폴더에 넣지 않습니다.

이미 다음 경로에 인증된 설정이 있는 경우:

```bash
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
test -f "$RCLONE_CONFIG"
rclone listremotes
rclone lsd "gdrive:Colab_Notebooks/TSA/Result"
```

`gdrive:`가 표시되고 결과 경로를 읽을 수 있는지 확인합니다. 설정 파일을 다른 위치에 저장했다면 `RCLONE_CONFIG`를 그 경로로 지정합니다. 인증이 아직 없다면 각 모델 폴더의 `GOOGLE_DRIVE_BACKUP.md`에 있는 원격 인증 절차를 사용합니다. 저장소에 인증이 들어 있지 않은 것은 정상입니다.

## 3. DCRNN — setup → checks → pilot

새 터미널을 열었으면 프로젝트 루트 환경변수를 다시 설정합니다.

```bash
cd /workspace/TSA-PM2.5-Forecasting-in-Seoul
export TSA_PROJECT_ROOT="$PWD/D/runtime"
export TSA_DCRNN48_OUTPUT_ROOT=/workspace/TSA/Result/DCRNN/VastAI_H48

cd "$TSA_PROJECT_ROOT/Model/DCRNN/VastAI_M1_M4T_H48_20261111"
bash setup_vast.sh
bash run_checks.sh
bash run_pilot.sh --conditions M1 M3T
```

각 단계가 성공한 뒤 다음을 실행합니다. `setup_vast.sh`는 파일 무결성과 GPU 커널을 확인하고 템플릿의 CUDA PyTorch를 유지한 별도 Python 환경을 준비합니다. 파일 검사가 실패하면 누락된 파일이나 버전을 먼저 확인합니다.

### DCRNN full

2023 validation까지 학습·추론·공통 채점을 수행합니다.

```bash
bash start_full.sh
```

2024 평가까지 수행하기로 정한 별도 실행에서는 다음을 사용합니다.

```bash
bash start_full.sh --final-test
```

## 4. Graph WaveNet — setup → checks → pilot

같은 데이터와 개인 Drive 인증을 재사용할 수 있습니다. 모델별 Python 환경과 출력 루트는 분리됩니다.

```bash
cd /workspace/TSA-PM2.5-Forecasting-in-Seoul
export TSA_PROJECT_ROOT="$PWD/D/runtime"
export TSA_GWN48_OUTPUT_ROOT=/workspace/TSA/Result/GraphWaveNet/VastAI_H48

cd "$TSA_PROJECT_ROOT/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112"
bash setup_vast.sh
bash run_checks.sh
bash run_pilot.sh --conditions M1 M3T
```

### Graph WaveNet full

```bash
bash start_full.sh
```

2024까지 평가할 별도 실행:

```bash
bash start_full.sh --final-test
```

한 GPU에서 두 모델을 동시에 실행하는 방식은 기본 실행 순서가 아닙니다. 하나의 full을 마친 뒤 다른 모델을 시작하면 메모리·계산 자원 경쟁을 피할 수 있습니다.

## 5. 진행 순서와 결과 위치

각 모델은 **M1 → M2 → M3 → M4 → M3T → M4T**, seed 11 순서로 진행합니다. 한 조건에서 3·6·12·24·36·48h를 함께 학습합니다. DCRNN 최대 50 epoch, Graph WaveNet H48 최대 75 epoch이며 둘 다 patience 10이므로 더 일찍 종료될 수 있습니다.

조건 하나의 학습이 끝나면 최적 가중치로 추론·공통 채점·Drive 백업을 진행합니다. 전체 조건 완료 후 통합 채점에서 사전 지정 조건 비교를 수행합니다.

```text
<모델별 OUTPUT_ROOT>/full/<실행ID>/
  Conditions/<조건>/<조건실행ID>/
    Models/       # 체크포인트
    History/      # epoch별 학습 기록
    Exports/      # 예측과 실행 계약
  Common_Scoring/<조건>/validation/...
  Common_Scoring/Combined/validation/<채점ID>/
  Common_Scoring/Combined/test/<채점ID>/     # --final-test 사용 시
```

`metrics.csv`에서 RMSE·MAE 등을 확인하고 `tests.csv`에서 통계 비교를 확인합니다. `candidate − reference`의 RMSE 차이가 음수면 수치상 개선입니다. 최종 유의성은 조건별 단독 채점보다 **Combined 통합 채점**을 기준으로 봅니다. 출력 파일명과 열 형식은 기존 공통 채점 규칙을 유지합니다.

## 6. 화면을 닫거나 중단한 뒤 이어서 실행

`start_full.sh`는 tmux 세션을 생성하거나 기존 작업에 연결합니다. 화면만 분리하려면 **Ctrl+B를 누른 뒤 D**를 누릅니다. 서버가 살아 있으면 브라우저나 개인 PC를 꺼도 학습은 계속됩니다. Vast.ai 인스턴스 자체를 중지·삭제하는 것과는 다릅니다.

새 터미널에서는 모델별 환경변수와 모델 폴더를 다시 설정한 뒤 같은 `bash start_full.sh` 명령을 사용합니다. 실행 중인 동일 작업이 있으면 연결하며, 중단된 경우 가장 최근 호환 full 체크포인트를 찾아 **새 실행 폴더**에서 재개합니다. `--final-test`를 사용했던 작업은 같은 옵션을 유지합니다.

로컬은 매 epoch를 완전히 저장하고, 5 epoch 간격 별도 snapshot을 만듭니다. Drive는 1·5·10… epoch와 종료·조건 완료 때 백업합니다. epoch 도중 중단됐다면 마지막으로 완전히 저장된 epoch부터 일부 계산을 다시 합니다. 설정이 달라진 checkpoint를 자동으로 섞어 재개하지 않습니다.

Drive 연결만으로 이전 가중치가 서버에 자동 복원되지는 않습니다. 디스크 손실 후에는 모델 폴더의 `GOOGLE_DRIVE_BACKUP.md`와 `CONTROL_AND_RESUME.md`에 따라 검증된 백업을 복구합니다.

## 7. 결과를 발표·공유할 때

중간발표는 2023 validation을 중심으로 비교합니다. 2024 결과를 보고 설정을 수정한 이력이 있으므로, 해당 결과를 완전히 독립적인 최종 블라인드 시험으로 부르지 않습니다. 새 설정 선택은 validation 또는 시간순 검증을 기준으로 하고, 2024는 preliminary 결과로 구분합니다.

실행 코드와 데이터 버전, 조건, 실제 학습 epoch, 최적 epoch, 시드, 평가 기간을 함께 기록합니다. Dataset·checkpoint·`Environment`·개인 인증·새 실행 결과는 Git에 자동 추가하지 않습니다. 저장소의 기존 결과는 보존하고 신규 결과는 별도 실행 폴더로 관리합니다.


## 공유본의 경로 및 재현 검사

저장소 루트에서 코드 누락·버전 검사와 대표 그림 재생성을 실행할 수 있습니다.

```bash
python D/verify_share.py --code-only
python -m pip install -r D/requirements_figures.txt
python D/reproduce_figure.py validation_curve
python D/reproduce_figure.py pre_east_counts
```

`reproduce_figure.py`는 입력 표·자산을 새 실행 폴더로 복사해 그리므로 기존 그림을 덮어쓰지 않습니다.
`actual_network`, `expansion_stages`, `outlier_ranges`, `current_counts`, `seasonal_concept`, `all`도 선택할 수 있습니다.
결과는 `results/D/reproduced/Run_...`에 저장합니다. 이 경량 재그리기는 모델 학습이나 원자료 재처리가 아닙니다.
모든 원자료부터 재계산하는 분석은 각 README의 Dataset 의존성을 따릅니다.


## 공유 복사본의 체크포인트 검증 수정

2026-11-15 공유 재현 검사에서 선택 옵션 `selection_horizon=24`의 점수를 저장할 때 6h 점수와 대조해 거부하는 오류를 발견했습니다.
공유본의 `checkpoint_store.py`만 선택 지평의 `validation_selection_score`를 검증하도록 수정했습니다.
기본6h 기록과 계약·학습 기록의 지평 일치 검사도 유지합니다. Graph WaveNet H48에서 사용하던 동일 검증 방식입니다.
모델 구조·기본6h 선택·기존 학습 결과는 바꾸지 않았고 Revised_Project 원본은 보존했습니다.
변경 diff와 원본/공유 해시는 `D/patches/`에 있습니다. DCRNN 업로드 매니페스트는 공유본 해시로 갱신했습니다.
이 변경 이후의 새 학습은 새 코드 버전으로 기록되며, 과거 체크포인트 자동 재개는 코드 계약 검증을 따릅니다.
