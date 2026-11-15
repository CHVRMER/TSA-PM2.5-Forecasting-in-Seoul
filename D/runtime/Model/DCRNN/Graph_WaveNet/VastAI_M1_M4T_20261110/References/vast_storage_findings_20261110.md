# Vast.ai 저장·체크포인트 확인

확인일: 2026-11-10. 공식 Vast.ai 문서와 PyTorch 문서만 근거로 작성했다. 계정 연결, 플러그인 설치, 인스턴스 생성, 외부 저장소 전송은 실행하지 않았다.

## 저장소 수명

| 상황 | 공식 문서상 동작 | 프로젝트에 주는 의미 |
| --- | --- | --- |
| 인스턴스 Stop/Restart | 기본 container 데이터 유지, 저장소 과금 계속 | 정지 자체가 곧 파일 삭제라는 설명은 틀리다. GPU가 다른 사용자에게 배정되면 재시작은 기다릴 수 있다. |
| 인스턴스 Destroy | 해당 container 데이터 영구 삭제 | 결과·체크포인트를 외부로 복사한 뒤 삭제해야 한다. |
| 별도 Local Volume | 인스턴스 삭제 후 보존, 같은 물리 호스트에만 재연결 | 다른 호스트로 이동하는 공유 네트워크 볼륨과 다르다. 볼륨도 별도 과금된다. |
| 호스트 Offline | 전원·네트워크·유지보수 문제 등이 원인일 수 있음 | 일시 접근 불가를 곧 데이터 삭제라고 단정할 수 없다. 같은 호스트 볼륨은 독립적인 외부 백업을 대신하지 못한다. |
| 계약 lifetime 만료 | 재시작 불가, 만료 48시간 뒤 삭제될 수 있음 | 만료 전에 결과를 외부로 회수한다. 48시간을 안전한 유예기간으로 의존하지 않는다. |

출처: [Storage Types](https://docs.vast.ai/guides/instances/storage/types), [Volumes](https://docs.vast.ai/guides/instances/storage/volumes), [Managing Instances](https://docs.vast.ai/guides/instances/manage-instances), [Instances FAQ](https://docs.vast.ai/guides/reference/faq/instances).

## 5 epoch 체크포인트 설계에 대한 판단

아래는 공식 서비스 보장 사항이 아니라 코드 설계 권고다.

- 매 epoch 재개용 `last` 상태를 원자적으로 저장하고, 검증 최적 `best`는 따로 유지한다.
- 5 epoch마다 `epoch_0005`, `epoch_0010`처럼 변경하지 않는 스냅샷을 추가한다. 조기 종료·정상 종료 시점도 저장한다.
- 가중치뿐 아니라 optimizer, scheduler, epoch, best metric, early-stop 상태, RNG, 데이터/설정 서명을 함께 저장한다.
- 같은 디스크의 체크포인트는 프로세스 중단 복구용이다. 호스트·디스크 상실에 대비하려면 저장 완료된 스냅샷과 로그·전처리·설정을 별도 저장소로 복사하고 성공 여부를 남겨야 한다.
- 외부 전송에 실패하면 로컬 학습 상태를 없애지 않고 오류 로그를 남긴다. '클라우드 백업 완료'라고 표시하지 않는다.
- 단순한 5 epoch 저장 주기와 외부 백업 주기는 별개다. 마지막 외부 전송 이후 진행분까지 보존된다는 보장은 없다.

## 공식 외부 백업 경로

Vast Cloud Sync는 Google Drive, S3, Backblaze, Dropbox, Hugging Face와 데이터를 복사한다. Docker 기반 인스턴스에서 지원하며 VM/KVM은 제외된다. Cloud Connections를 먼저 연결해야 하고, 연결 권한은 호스트에 임시로 전달되므로 공식 문서는 범위 제한 계정/키와 Secure Cloud 호스트를 안내한다.

공식 예약 백업은 hourly/daily/weekly이며, 정확히 '학습 5 epoch 완료 직후'는 학습 코드의 완료 훅이나 별도 외부 복사 로직이 필요하다. 예약 복사와 파일의 완전한 체크포인트 저장 시점을 혼동하지 않는다.

출처: [Cloud Sync](https://docs.vast.ai/guides/instances/storage/cloud-sync), [Scheduled Cloud Backups](https://docs.vast.ai/guides/instances/storage/cloud-backups).

## RTX 5090 환경

Blackwell 지원은 PyTorch 2.7의 CUDA 12.8 바이너리부터 제공됐다. 현재 2.12 공식 안내는 Blackwell에 CUDA 13.0+ 바이너리를 사용하라고 명시한다. 이 경우 Linux NVIDIA 드라이버 580.65.06 이상이 요구된다. 따라서 오래된 CUDA 12.1/12.6 PyTorch 이미지를 무조건 재사용하지 말고, 실제 템플릿의 torch/CUDA/드라이버 조합을 확인하고 GPU 연산 pilot을 통과시킨다. 단순히 `torch.cuda.is_available()`가 참인 것만으로 모델의 GPU kernel 호환성을 충분히 검증한 것은 아니다.

출처: [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/), [PyTorch 2.12](https://pytorch.org/blog/pytorch-2-12-release-blog/).
