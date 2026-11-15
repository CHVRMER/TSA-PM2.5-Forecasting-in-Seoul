# DCRNN H48: Google Drive 자동 백업 설정

같은 Vast.ai 인스턴스에서 Graph WaveNet 또는 기존 DCRNN용 `gdrive` 인증을 이미 완료했다면 그대로 재사용한다. 인증 설정을 처음부터 반복할 필요는 없다. 아래 백업 경로만 H48 전용으로 분리된다.

백업 대상은 `내 드라이브/Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48`이다. rclone 표기는 다음과 같다.

```text
gdrive:Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48
```

여기서 `gdrive`는 사용자가 rclone에 등록하는 연결 이름이다. 이 문서 작성 과정에서 Google 계정에 연결하거나 OAuth 인증을 완료하지 않았다. **full 실행 전 사용자가 아래 연결을 준비해야 한다.**

## 1. 내 PC에서 연결 준비

1. Windows용 rclone을 [공식 설치 안내](https://rclone.org/install/)에서 받는다. 압축을 푼 폴더를 Git Bash의 PATH에 추가하거나 `rclone.exe`의 전체 경로로 실행한다. 서버에도 rclone이 필요하다.
2. [공식 Drive client_id 안내](https://rclone.org/drive/#making-your-own-client-id)에 따라 본인 Google Cloud 프로젝트에서 Drive API를 켜고, OAuth 클라이언트 유형을 **Desktop app**으로 만든다. 2026년 공식 안내는 공용 client_id 중단 때문에 **자기 client_id 사용을 요구**한다. client_id/client_secret은 개인 터미널의 설정 과정에만 입력한다.
3. 아래처럼 TSA 전용 설정 파일을 만든다. 다른 서비스의 인증 정보까지 서버로 복사하지 않도록 별도 파일을 사용한다.

```bash
mkdir -p "$HOME/.config/rclone"
rclone --config "$HOME/.config/rclone/tsa-vast.conf" config
```

대화형 선택:

```text
New remote: n
name: gdrive
Storage: drive
client_id / client_secret: 본인이 만든 값
scope: drive
service_account_file: 비움
Use web browser to automatically authenticate: y
Shared Drive: n  (개인 내 드라이브인 경우)
```

브라우저에서 백업할 Google 계정으로 직접 로그인한다. `drive` 권한은 기존 `Colab_Notebooks/TSA` 경로를 탐색하고 쓸 수 있지만 계정 전체 파일 접근 범위가 넓다. `drive.file`은 rclone이 만든 파일·폴더만 보므로 기존 폴더를 못 볼 수 있다. 경로 제한만으로 OAuth 권한 자체가 줄어드는 것은 아니다. 전용 계정 사용도 고려한다. OAuth 앱이 Testing 상태면 약 7일 후 재인증이 필요할 수 있다. 자세한 정책은 위 공식 안내를 따른다.

연결 이름과 기존 TSA 경로 확인:

```bash
rclone --config "$HOME/.config/rclone/tsa-vast.conf" listremotes
rclone --config "$HOME/.config/rclone/tsa-vast.conf" lsd "gdrive:Colab_Notebooks/TSA"
```

`gdrive:`가 보여야 한다. 읽기 목록 확인만으로 실제 업로드 성공이 검증되는 것은 아니다. 아래 서버 쓰기 검증까지 진행한다.

## 2. 서버로 전용 설정 파일 전달

Vast Connect 화면에서 제시한 SSH 사용자·주소·포트를 사용한다. 아래 `root`, `SERVER_IP`, `SSH_PORT`는 실제 연결 정보로 바꾼다.

내 PC Git Bash:

```bash
ssh -p SSH_PORT -i ~/.ssh/id_ed25519 root@SERVER_IP 'umask 077; mkdir -p ~/.config/rclone; chmod 700 ~/.config/rclone'
scp -P SSH_PORT -i ~/.ssh/id_ed25519 "$HOME/.config/rclone/tsa-vast.conf" root@SERVER_IP:~/.config/rclone/tsa-vast.conf
```

서버 SSH 터미널:

```bash
chmod 600 "$HOME/.config/rclone/tsa-vast.conf"
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
rclone version
rclone listremotes
```

서버·PC는 같은 rclone 버전을 사용하는 편이 좋다. 위 설정 파일에는 refresh token이 들어갈 수 있다. 파일 내용, OAuth 결과, client_secret을 채팅·Notion·학습 로그에 붙이지 말고 공유 ZIP에도 넣지 않는다. 프로젝트 밖에 둔 이 파일은 학습 결과 백업 대상이 아니다. 서버의 계정 소유자는 이 파일을 읽을 수 있으므로 신뢰하는 호스트를 사용한다.

이 방법은 rclone 공식 문서의 [설정 파일 복사 방식](https://rclone.org/remote_setup/#configuring-by-copying-the-config-file)에 해당한다.

## 3. 대안: 서버에서 headless 인증

설정 파일을 복사하는 대신 서버에서 `rclone config`로 `gdrive`를 만들 수 있다. 브라우저 사용 질문에 `n`을 고르면 로컬 PC에서 실행할 `rclone authorize ...` 명령이 표시된다. **서버에 표시된 명령을 그대로 로컬 개인 터미널에서 실행**한다. Google Drive 설정이면 authorize 대상은 `drive`이며 own client_id와 scope 설정에 따라 추가 인수가 붙을 수 있다.

로컬 브라우저에서 로그인한 뒤 나온 인증 결과를 서버의 `config_token>` 입력란에만 붙인다. 채팅이나 코드에 붙이지 않는다. 인증 결과를 `tee`로 로그에 남기지 않는다. 이 방식과 앞의 파일 복사 방식 중 하나만 완료하면 된다. [공식 headless 안내](https://rclone.org/remote_setup/#configuring-using-rclone-authorize).

## 4. full 전에 업로드와 검증 확인

서버 터미널에서 실제 쓰기와 읽기 검증을 수행한다. 아래는 새 확인 폴더만 만들며 기존 백업을 수정하지 않는다.

```bash
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
task_probe_dir=$(mktemp -d /tmp/tsa_gdrive_probe_XXXXXXXX)
task_probe_name=$(basename "$task_probe_dir")
printf 'TSA DCRNN backup connection check\n' > "$task_probe_dir/connection_ok.txt"
task_probe_remote="gdrive:Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48/ConnectionChecks/$task_probe_name"
rclone copy "$task_probe_dir" "$task_probe_remote" --checksum
rclone check "$task_probe_dir" "$task_probe_remote" --one-way
```

두 명령 모두 정상 종료하고 Drive 웹에서도 새 확인 파일이 보여야 한다. 인증·쓰기 권한·용량·네트워크 문제가 있으면 먼저 해결한다. full은 Google Drive 백업을 필수로 사용하므로 연결이 준비되지 않으면 학습을 시작하지 않는다.

`copy`는 목적지의 기존 파일을 삭제하지 않는다. `check --one-way`는 이번에 저장한 원본 파일이 목적지에도 일치하는지 검사하며 파일을 바꾸지 않는다. 이 패키지는 다른 실행 백업을 지우는 `sync`를 사용하지 않는다. [copy 문서](https://rclone.org/commands/rclone_copy/), [check 문서](https://rclone.org/commands/rclone_check/).

## 5. 자동 저장과 백업 주기

| 시점 | 서버 저장 | Drive 복사·검증 |
| --- | --- | --- |
| 매 epoch 완료 | 검증 가능한 immutable generation에 현재 상태와 최적 모델을 함께 저장 | 주기 백업 사이의 최신 상태는 서버에만 있을 수 있음 |
| epoch 1 | 첫 정상 체크포인트 | 첫 외부 백업의 성공 확인 |
| epoch 5, 10, 15, … | epoch 번호별 스냅샷 | 저장 완료본을 실행별 Drive 폴더에 copy 후 check |
| 조기 종료·정상 종료 | 마지막 완료 상태, 예측·채점 결과 | 마지막 복사와 검증 |

학습이 강제 종료되거나 호스트가 사라지면 종료 처리가 실행되지 않을 수 있다. 이때 다른 서버에서 복구 가능한 범위는 **마지막으로 외부 복사와 검증까지 성공한 체크포인트**다. 매 epoch의 서버 저장을 매 epoch의 외부 백업으로 해석하면 안 된다. 백업 오류가 표시됐는데도 성공했다고 간주하지 않는다.

새 실행은 새 서버 결과 폴더와 새 Drive 실행 폴더에 저장한다. 같은 실행 안의 최신 저장 위치를 가리키는 포인터 갱신은 재개를 위한 의도된 갱신이다. 조건별 스냅샷과 이전 실행 결과는 별개로 보존한다. 실제 로그의 백업 목적지·완료 상태를 확인한다.

## 6. 다른 인스턴스에서 복구

새 서버에 같은 코드·데이터·환경을 준비하고 `gdrive` 인증을 다시 연결한다. Drive에서 이전 실행 폴더 이름을 확인한다.

```bash
rclone lsd "gdrive:Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48"
```

이전 백업의 **학습 batch root 내용**을 새 로컬 폴더로 복사한다. 아래 백업 경로는 실제 로그에 기록된 값으로 바꾼다.

```bash
mkdir -p /workspace/TSA/Restored/DCRNN_H48
rclone copy "gdrive:Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48/실제_이전백업경로" /workspace/TSA/Restored/DCRNN_H48/raw_batch --checksum
rclone check "gdrive:Colab_Notebooks/TSA/Result/DCRNN/VastAI_H48/실제_이전백업경로" /workspace/TSA/Restored/DCRNN_H48/raw_batch --one-way
cd /workspace/TSA/Model/DCRNN/VastAI_M1_M4T_H48_20261111
python Codes/restore_verified_backup.py \
  --downloaded-root /workspace/TSA/Restored/DCRNN_H48/raw_batch \
  --marker /workspace/TSA/Restored/DCRNN_H48/raw_batch/Backup_receipts/Published/선택한_백업ID/verified.json \
  --output-root /workspace/TSA/Restored/DCRNN_H48/verified_batch
bash run_full.sh --resume-source /workspace/TSA/Restored/DCRNN_H48/verified_batch
```

`선택한_백업ID`는 내려받은 `Backup_receipts/Published`에 있는 날짜·ID 폴더로 바꾼다. 백업 검증 표식이 있고 체크포인트가 포함된 복사본을 선택한다. full 시작 전의 작은 초기 쓰기 검사에는 모델이 없으므로 그 표식만으로는 복구할 수 없다.

복구 도구는 표식에 포함되어 있고 실제 크기·SHA256가 맞는 체크포인트 세 파일만 선택한다. 더 최신이지만 전송이 덜 끝났거나 표식에 없는 저장본은 제외한다. 다음 백업 실패로 상태 포인터·진행 JSON이 달라졌다면 이를 보고하고 새 폴더에서 다시 만든다. `verified_batch/restore_report.json`의 실제 복구 epoch와 제외 파일을 확인한다. 원본 `raw_batch`는 수정하지 않으며 `verified_batch`는 새 경로여야 한다.

`RCLONE_CONFIG` export를 새 셸에서도 유지한다. 데이터·설정·입력조건·seed·전처리 서명이 같아야 재개할 수 있다. `Launch_*` 폴더나 체크포인트 파일 하나 대신 코드가 요구하는 batch root를 지정한다. 새 실행은 이전 폴더를 보존한다.

Vast 기본 디스크는 Stop 때 남지만 Destroy 때 사라진다. Local Volume도 같은 물리 호스트에 묶이므로 Drive 복사본과 같은 역할은 아니다. [Vast 저장소 문서](https://docs.vast.ai/guides/instances/storage/types).
