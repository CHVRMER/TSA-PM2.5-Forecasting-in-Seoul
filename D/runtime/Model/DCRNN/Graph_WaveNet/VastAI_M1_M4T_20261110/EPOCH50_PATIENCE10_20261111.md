# Graph WaveNet 종료 조건: 최대 50 epoch / patience 10

2026-11-11 사용자 요청 반영. 기본 full 학습의 `training.max_epochs=50`, `training.patience=10`이다. epoch 50은 재개 이후 추가 50회가 아니라 **누적 학습 epoch의 상한**이다. 검증 6시간 RMSE가 연속 10 epoch 개선되지 않으면 종료한다. Pilot은 기존 2 epoch 설정을 유지한다.

## Vast.ai 적용

실행 중인 프로그램은 설정을 시작할 때 읽으므로 파일 수정만으로 값이 바뀌지 않는다.

1. 학습 화면에서 Ctrl+C를 누르고 `[stopping]`의 상태 기록·백업 처리가 끝나 Bash 프롬프트로 돌아올 때까지 기다린다. 이전 버전 dead pane이면 새 Terminal을 연다.
2. `TSA_GraphWaveNet_Epoch50_Patience10_20261111.zip`을 `/workspace`에 업로드한다. 이 ZIP은 최신 자동 재개·중단 처리 수정도 포함한 코드 전용 묶음이다.
3. 새 Terminal에서 적용한다.

```bash
python3 -m zipfile -e /workspace/TSA_GraphWaveNet_Epoch50_Patience10_20261111.zip /workspace/TSA
```

4. 기존 명령으로 자동 재개한다. 처음부터 다시 시작하는 `--fresh`는 필요하지 않다.

```bash
bash /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_20261110/start_full.sh --final-test
```

## 기존 저장본 처리

- 원본 체크포인트는 유지하고 새로운 실행 폴더에 가져온다.
- `max_epochs`, `patience`의 감소만 재개 계약 변경으로 허용한다. optimizer·학습률·모델·그래프·데이터·전처리·라이브러리 등 다른 계약 검증은 유지한다.
- 현재/최적 모델, optimizer, scheduler, RNG, 학습 이력, 연속 미개선 횟수 `bad_epochs`를 보존한다.
- `bad_epochs >= 10`이면 추가 학습 없이 현재 저장된 최적 모델로 추론·채점한다.
- 기존 저장본이 50 epoch를 넘었다면 50 이하에서 가장 최근의 정상 저장본을 선택한다. 보통 5 epoch 간격 snapshot 중 epoch 50을 사용한다. 50이 손상됐다면 이전 정상 snapshot을 찾는다. 적합한 저장본이 없으면 오류로 중단하고 원본을 보존한다.
- 100/20 설정으로 완료된 조건도 변경된 종료 정책에 맞춰 가져오고 추론·채점을 새 폴더에 기록한다.

이는 **기존 학습 도중 종료 정책을 변경한 실험**으로 기록된다. patience는 재개한 시점부터 적용하며 과거의 학습 순서를 소급해 재현하지 않는다.

## 변경 기록

각 조건·seed 폴더의 `stopping_policy_revision.json`, 실행의 `resume_copy_manifest.json`, `run_summary.json`에서 원래 제한·새 제한·선택 epoch·보존된 미개선 횟수를 확인할 수 있다. 공통 채점기의 표와 열 형식은 그대로다.

데이터·모델 구조·그래프·optimizer·loss·scheduler·배치 크기는 유지했다. 새 기본값은 config와 runner의 fallback 양쪽에 반영했다. 원본 결과·코드는 보존한다.

## 검증 범위

종료 조건 변경과 재개·완료 판정·중단·tmux 제어를 검증했다. 체크포인트 재등록은 모의 저장소로 가중치·optimizer·RNG·이력 보존과 epoch 제한을 확인했다. 실제 PyTorch 저장본 이관 및 Vast.ai GPU 학습은 이 환경에서 실행하지 않았다. 실제 복원 때 기존 CheckpointStore와 trainer가 전체 payload·계약을 다시 검사한다.
