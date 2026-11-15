# Google Drive → Vast.ai 다운로드와 현재 볼륨 사용

이 추가 코드는 큰 ZIP을 Google Drive에서 Vast.ai로 받는 단계에만 gdown을 사용합니다. 학습·추론·공통 채점·5 epoch 스냅숏·Google Drive 업로드 백업은 기존 구현 그대로입니다. gdown은 업로드 백업 도구가 아니며, 모든 환경에서 rclone보다 빠르다고 보장하지 않습니다.

## 현재 확인된 저장소

사용자가 보내준 `findmnt` 출력에서 다음을 확인했습니다.

```text
인스턴스: 60029417
볼륨: V.60029416
연결 경로: /workspace
Source: /dev/loop0[/volumes/V.60029416/_data]
파일시스템: xfs
관찰된 크기: 20.8G
관찰된 여유 공간: 20.5G
```

따라서 **`/workspace/TSA`에 프로젝트를 두는 기존 경로가 맞습니다.** Storage 화면에 직접 업로드하는 대신, 연결된 인스턴스의 JupyterLab/SSH에서 `/workspace`에 파일을 쓰면 이 볼륨에 저장됩니다.

볼륨은 임대한 21GB 기준으로 과금됩니다. 화면의 `$6.94/mo`는 월 환산 요율이며 파일을 올릴 때부터 과금한다는 뜻이 아닙니다. 실제 청구 금액은 사용 시간·호스트 상태 등에 따릅니다. GPU 사용률 0%도 켜진 인스턴스의 대여료가 무료라는 뜻은 아닙니다. Local Volume은 같은 물리 호스트에 묶이며, Google Drive 외부 백업도 유지합니다.

현재 GPU 출력은 RTX 5090·약32GB VRAM·driver580.178.04를 보여주지만, `nvidia-smi`의 CUDA 표시는 드라이버의 지원 정보입니다. 실제 PyTorch CUDA 연산은 기존 `setup_vast.sh`와 pilot으로 별도 확인합니다.

## 가장 간단한 방법: 작은 Python 도우미 하나 업로드

1. 본인 PC에서 기존 두 ZIP을 Google Drive에 보관하고 각 **파일**의 다운로드 가능한 공유 링크를 준비합니다.

```text
데이터: TSA_GraphWaveNet_Runpod_M1_M4T_20261110.zip  (477,518,306 bytes)
코드:   TSA_GraphWaveNet_VastAI_M1_M4T_20261110.zip  (105,355 bytes)
```

폴더 링크나 이번 gdown 추가 ZIP 링크를 위 두 파일 대신 넣지 않습니다. 파일을 새로 압축하면 SHA256가 달라지므로 기존 배포 ZIP 그대로 사용합니다.

2. 이 폴더의 `Codes/drive_download_gdown.py`만 본인 PC에서 Vast.ai JupyterLab의 `/workspace`에 업로드합니다. 큰 ZIP을 Jupyter로 다시 올릴 필요는 없습니다.

3. Vast.ai 터미널에서 별도 전송 환경을 만들고 실행합니다.

```bash
cd /workspace
python3 -m venv /workspace/.tsa_gdown_env
/workspace/.tsa_gdown_env/bin/python -m pip install gdown

/workspace/.tsa_gdown_env/bin/python /workspace/drive_download_gdown.py \
  --project-root /workspace/TSA \
  --data-url '데이터_ZIP의_Google_Drive_공유링크' \
  --code-url '기존_VastAI_코드_ZIP의_Google_Drive_공유링크'
```

두 링크 부분만 실제 값으로 바꿉니다. URL에 `&`가 있을 수 있으므로 작은따옴표를 유지합니다. gdown은 파일의 공유 권한·Google 다운로드 제한을 우회하는 수단이 아닙니다. 공유 링크를 제공할 수 없는 비공개 파일은 기존 인증된 rclone 연결로 받으세요. 브라우저의 로그인 쿠키를 Vast.ai에 복사할 필요는 없습니다.

다운로드 ZIP은 `/workspace/.tsa_downloads`의 시도별 캐시에, 압축 해제된 파일은 `/workspace/TSA/{Dataset,Model}`에 저장합니다. 중단 후 동일 명령을 다시 실행하면 정상 캐시는 재사용하고 중단 전송은 이어받습니다. 이미 완성된 ZIP이 SHA256 검사에 실패하면 자동으로 원본을 지우지 않습니다. 명시적으로 새 시도를 하려면 같은 명령에 `--redownload`를 추가합니다.

## 추가 ZIP 전체를 업로드한 경우

이번 추가 ZIP은 기존 파일을 바꾸지 않는 새 도우미 파일만 포함합니다. `/workspace/TSA`에 압축 해제한 뒤 다음처럼 실행할 수 있습니다.

```bash
export TSA_VOLUME_MOUNT=/workspace
export TSA_PROJECT_ROOT=/workspace/TSA

bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/download_with_gdown.sh \
  --data-url '데이터_ZIP의_Google_Drive_공유링크' \
  --code-url '기존_VastAI_코드_ZIP의_Google_Drive_공유링크'
```

이 실행기는 실제 별도 볼륨 연결을 읽기 전용으로 검사한 뒤 별도 gdown 환경을 준비합니다. 컨테이너 루트 또는 RAM 파일시스템을 볼륨으로 잘못 선택하면 멈춥니다. 다른 서버에서 사용한다면 실제 mount를 확인해 환경변수를 바꾸세요.

현재 볼륨 ID까지 확인하려면:

```bash
python3 /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/Codes/inspect_vast_storage.py \
  --mount-path /workspace --project-root /workspace/TSA --volume-id 60029416
```

## 다운로드 이후

```bash
export TSA_PROJECT_ROOT=/workspace/TSA
cd /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110
bash setup_vast.sh
bash run_pilot.sh --conditions M1
```

기본 pilot의 Drive 백업은 꺼져 있습니다. pilot에서도 백업을 확인하려면 앞서 안내한 `configs/pilot_drive.json`을 만든 뒤 `--config configs/pilot_drive.json`을 추가하세요. Full의 결과 업로드에는 **기존 rclone Google Drive 인증이 여전히 필요**하며, 설정 방법은 `GOOGLE_DRIVE_BACKUP.md`를 따릅니다.

## 검증·보존

- 두 원본 ZIP의 SHA256·크기·CRC·필수 구조 검사.
- 정상 캐시 재사용, 중단 파일 재개, 손상본 보존 후 명시적 새 다운로드.
- ZIP 경로 탈출·심볼릭 링크·특수 파일·중복 이름·결과 폴더 침입 차단.
- 기존 파일과 내용이 같으면 건너뛰고, 다르면 원본을 보존한 채 설치 중단.
- 모델·optimizer·데이터 전처리·채점기·Drive 백업 코드는 수정하지 않음.
- Google Drive 실전송이나 Vast.ai 원격 파일 쓰기는 이 로컬 검증에서 실행하지 않음.

출처:
https://github.com/wkentaro/gdown
https://docs.vast.ai/guides/instances/storage/volumes
https://docs.vast.ai/guides/reference/faq/billing
