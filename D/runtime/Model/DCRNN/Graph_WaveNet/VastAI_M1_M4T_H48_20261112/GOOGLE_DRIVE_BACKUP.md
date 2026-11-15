# Graph WaveNet H48 Google Drive 백업

## 기존 인증 재사용

기존 DCRNN에서 연결한 동일한Google계정·`gdrive` remote를 사용합니다. 데이터는 서버의`/workspace/TSA/Dataset`에서 읽고, 결과만 Drive에 복사합니다.

```bash
export RCLONE_CONFIG="$HOME/.config/rclone/tsa-vast.conf"
rclone listremotes
rclone lsd "gdrive:Colab_Notebooks/TSA/Result"
```

`gdrive:`와 기존 결과 폴더가 보이면 인증을 다시 만들 필요 없습니다. 다른 위치에 개인설정 파일을 보관했다면 `RCLONE_CONFIG`를 실제 위치로 지정합니다. 이 환경변수는 tmux의 학습 프로세스에도 전달됩니다.

`gdrive`는 내드라이브 최상위를 기준으로 연결해야 합니다. `root_folder_id`를TSA폴더로 제한하면 코드가`Colab_Notebooks/TSA`를 다시 찾게 되어 경로가 중첩됩니다.

새 인스턴스의 설정이 없다면 [rclone공식 원격설정](https://rclone.org/remote_setup/)을 따라 연결합니다. 서버에서브라우저인증질문에는`n`, 서버가제시한`rclone authorize`전체 명령을내PC에서실행하고, 결과는서버`config_token>`에만붙여넣습니다. 기존TSA OAuth클라이언트를 사용할수있습니다. **토큰·client_secret·설정파일내용을채팅·공유ZIP에넣지않습니다.**

## 백업 목적지

```text
gdrive:Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI_H48/<실행ID>
```

기존GraphWaveNet/VastAI와DCRNN/VastAI_H48결과를덮어쓰지않습니다. 각실행이별도ID를사용합니다. 학습·추론·채점결과가같은실행폴더에모입니다.

- 로컬은매epoch마다현재·최적가중치,optimizer,scheduler,RNG,조기종료상태를완전한generation으로저장합니다.
- 별도snapshot은5epoch간격,기본최근일반generation은2개보존합니다.
- Drive동기백업은epoch1,5,10…와최종종료·각조건추론채점완료에수행합니다.
- 임시파일·인증정보는포함하지않습니다. 로그일부는백업제외규칙에해당하며주요학습곡선·예측·지표·상태JSON은보존합니다.
- full은remote등록·읽기와실제초기백업을확인한뒤학습하며,필수백업실패시로컬만으로조용히계속하지않습니다.
- 네트워크실패전마지막검증백업이복구기준입니다. 서버디스크가사라지면그이후epoch는다시계산할수있습니다.

## Drive에서 복구해야 하는 경우

서버디스크가유지된다면`start_full.sh`가로컬에서자동재개합니다. 새인스턴스라면먼저본인소유H48백업을서버로다운로드하고,해당실행의검증된publication marker와함께복구합니다. Drive를연결한것만으로다운로드되지는않습니다.

`Codes/restore_verified_backup.py`의`--help`로인수를확인합니다. 복구기는선택한marker에기록된파일만받아들이고각SHA256과불변checkpoint triple을검증합니다. 변경·누락된최신generation보다이전검증generation이남아있으면이를사용할수있습니다. 출력은반드시새폴더여야하며원본백업을변경하지않습니다.

```bash
cd /workspace/TSA/Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112
bash setup_vast.sh
```

실제복구명령의다운로드경로와marker는해당백업의파일목록을확인한뒤지정합니다. DCRNN·예전4개지평GWN의백업은이모델과호환되지않습니다. 이문서·코드는실제Google인증을자동완료하거나새GPU를대여하지않습니다.
