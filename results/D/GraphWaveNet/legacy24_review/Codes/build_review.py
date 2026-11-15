"""Assemble a new result-review folder. Original data/models/scorer remain intact."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import pandas as pd

WORK = Path(__file__).resolve().parent
SOURCE = WORK/'selected_results'
STAMP = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d_%H%M%S_%f')
RELATIVE = Path('Model/DCRNN/Graph_WaveNet/Results') / ('VastAI_M1_M4T_Review_'+STAMP)
DEST = WORK/'staging'/RELATIVE
TABLES = DEST/'Results/Tables'
CHECKS = DEST/'Results/Verification'
CODES = DEST/'Codes'
ORDER = ['M1','M2','M3','M4','M3T','M4T']
HORIZONS = [3,6,12,24]

def markdown(frame):
    result=['| '+' | '.join(map(str,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False,name=None):
        result.append('| '+' | '.join(str(v).replace('|',' / ') for v in row)+' |')
    return '\n'.join(result)

def write_csv(frame,name):
    frame.to_csv(TABLES/name,index=False,encoding='utf-8-sig')

def main():
    for folder in (TABLES,CHECKS,CODES): folder.mkdir(parents=True,exist_ok=False)
    for dirname in ('independent_stats','team_comparison','protocol'):
        folder=WORK/dirname
        for source in folder.iterdir():
            if not source.is_file(): continue
            if source.suffix=='.csv':
                # Keep the original common-score CSVs in Original_Combined instead.
                if '_original_' not in source.stem:
                    shutil.copy2(source,TABLES/source.name)
            elif source.suffix=='.json': shutil.copy2(source,CHECKS/(dirname+'_'+source.name))
            elif source.suffix=='.py':
                target=CODES/dirname/source.name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
    for filename in ('archive_integrity.json','inventory.json'):
        shutil.copy2(WORK/filename,CHECKS/filename)
    original=DEST/'Results/Original_Combined'
    shutil.copytree(SOURCE/'Common_Scoring/Combined',original)
    manifests=CHECKS/'Final_Run_Metadata'
    manifests.mkdir()
    for path in SOURCE.glob('*.json'): shutil.copy2(path,manifests/path.name)
    shutil.copytree(SOURCE/'Condition_Contracts',manifests/'Condition_Contracts')
    shutil.copy2(Path(__file__),CODES/'build_review.py')

    metrics={s:pd.read_csv(next((SOURCE/'Common_Scoring/Combined'/s).glob('*/metrics.csv'))) for s in ('validation','test')}
    tables={}
    for split,frame in metrics.items():
        rows=[]
        for c in ['persistence']+ORDER:
            sub=frame[frame.model.eq('persistence')] if c=='persistence' else frame[frame.input_condition.eq(c)]
            assert len(sub)==4
            rows.append({'조건':c,**{f'{h}h RMSE':float(sub.loc[sub.horizon.eq(h),'RMSE'].iloc[0]) for h in HORIZONS}})
        output=pd.DataFrame(rows)
        write_csv(output,f'공유용_{2023 if split=="validation" else 2024}_전체조건_RMSE.csv')
        tables[split]=output.assign(**{f'{h}h RMSE':output[f'{h}h RMSE'].map(lambda v:f'{v:.3f}') for h in HORIZONS})

    planned=pd.read_csv(WORK/'independent_stats/test_planned_input_comparisons.csv')
    rows=[]
    for c,r in [('M2','M1'),('M3','M2'),('M3','M1'),('M4','M3'),('M3T','M3'),('M4T','M4')]:
        sub=planned[planned.candidate.eq(c)&planned.reference.eq(r)].set_index('horizon')
        rows.append({'후보 / 기준':f'{c} / {r}',**{f'{h}h 개선율(%)':round(float(sub.loc[h,'gain_pct']),2) for h in HORIZONS},
                     '유의 개선':int(sub.improved.sum()),'유의 악화':int(sub.significant_worsening.sum())})
    comparisons=pd.DataFrame(rows)
    write_csv(comparisons,'공유용_2024_입력추가_개선율과판정.csv')
    paired=pd.read_csv(WORK/'team_comparison/test_GWN_vs_DCRNN_paired_review.csv')
    selected=paired[paired.comparison.eq('GWN_M4_vs_DCRNN_F')].sort_values('horizon')
    dcrnn_table=pd.DataFrame({'예측시간':selected.horizon.map(lambda v:f'{int(v)}h'),
        'GWN M4 RMSE':selected.candidate_RMSE.map(lambda v:f'{v:.3f}'),
        'DCRNN F RMSE':selected.reference_RMSE.map(lambda v:f'{v:.3f}'),
        'GWN 개선율(%)':selected.improvement_pct.map(lambda v:f'{v:.2f}'),
        '사후 짝비교 판정':['개선' if v else '차이 불확실' for v in selected.candidate_better_holm8]})
    write_csv(dcrnn_table,'공유용_2024_GWN_DCRNN_비교.csv')
    oldteam=pd.read_csv(WORK.parent/'dcrnn_regional12_review_20261111/staging/Model/DCRNN/Results/Regional12_Seed11_Review_20261111_092119_362839/Results/Tables/공유용_2023_팀원_및_동일범위참고_비교.csv')
    g4=metrics['validation'][metrics['validation'].input_condition.eq('M4')].set_index('horizon')
    oldteam.loc[len(oldteam)]=['Graph WaveNet M4 F634 seed11']+[float(g4.loc[h,'RMSE']) for h in HORIZONS]
    write_csv(oldteam,'공유용_2023_팀원결과_참고비교.csv')
    team_md=oldteam.copy()
    for col in team_md.columns[1:]:team_md[col]=team_md[col].map(lambda v:f'{v:.3f}')
    protocol=pd.read_csv(WORK/'protocol/training_protocol_summary.csv')
    small=protocol[['condition','trained_epochs','best_epoch','val6_rmse','allowed_external_series','trajectory_allowed_names']].copy()
    small.columns=['조건','누적 학습 epoch','최적 epoch','검증 6h RMSE','허용 외부 계열','허용 경로 특징']
    small['검증 6h RMSE']=small['검증 6h RMSE'].map(lambda v:f'{v:.3f}')

    report=f'''# Graph WaveNet Vast.ai M1–M4T 결과 검토

검토일: 2026-11-11. 원본 ZIP·데이터·학습 코드·공통 채점기·기존 결과를 수정하지 않았다. 완료 실행의 원래 공통 채점 양식은 `Results/Original_Combined`에 그대로 보존했다.

## 판단

Graph WaveNet은 모든 조건·예측시간에서 Persistence를 유의하게 개선했다. 기존 DCRNN보다도 2024년 3·6·12시간 예측이 개선됐다. 그러나 **국외 관측이나 HYSPLIT 특징을 추가하면 국내 전용보다 개선된다는 핵심 주장에 대한 증거는 이번 실행에서도 확보되지 않았다.** 팀의 LightGBM·GRU보다 일관되게 우수하다는 근거도 없다.

국내 기준선을 안정적으로 학습한 뒤에는 DCRNN에서 크게 보였던 국외 추가 효과가 재현되지 않았다. 따라서 이전 DCRNN의 큰 추가 개선율에는 약한 DN 기준선과 최적화 불안정의 영향이 섞였을 가능성을 검토해야 한다. 이는 원인 확정이 아니라 두 실행을 함께 해석한 진단이다.

## 1. 실제 완료 실행과 파일 무결성

- 최종 완료 폴더: `20261111_014538_406520_eaa72758`.
- `batch_manifest.json`: `status=passed`, `mode=full`, `final_test=true`, 실패 조건 없음.
- M1·M2·M3·M4·M3T·M4T 여섯 조건 모두 완료. 2023 검증·2024 시험의 개별·통합 공통 채점 결과가 있다.
- 원본 ZIP 두 개의 CRC 검사 모두 통과. ZIP 간 같은 경로의 중복·충돌 없음.
- 두 번째 ZIP은 최종 배치의 체크포인트 `state.pt` 다섯 개만 담는다. 별도 실험이 아니라 Google Drive의 분할 다운로드 결과다. 두 ZIP을 합치면 최종 배치 648개 파일이 모인다.
- 저장된 여섯 최적 가중치의 SHA가 각 요약 기록과 일치한다. 제외된 관측·경로 특징의 정규화 유효 표본 수가0이며, 정답과 기본53개 문맥의 전처리도 조건 사이에 일치한다.
- 나머지 네 실행 폴더는 파일럿 두 개와 중단·재개 이전 본실험 두 개다. 중단 폴더의 `running` 상태는 프로세스가 현재도 실행 중이라는 증거가 아니다.
- 최종 배치 소요시간은 약 2시간 50분이다. 이전 실행에서 수행한 M1 학습 시간 등은 포함하지 않으므로 총 GPU 사용시간·비용으로 해석하지 않는다.

## 2. 평가 검증

각 조건의 원시 예측을 권위 평가 격자와 연결하여 공통 채점을 다시 계산했다. 점수표·52개 검정·행 단위 점수 자료가 원본과 수치 허용오차 1e-10 이내로 일치했다.

| 구간 | 3h 평가행 | 6h | 12h | 24h |
| --- | --- | --- | --- | --- |
| 2023 검증 | 2,531 | 2,530 | 2,528 | 2,524 |
| 2024 시험 | 2,874 | 2,873 | 2,871 | 2,867 |

평가 행은 예측 성공 여부의 교집합으로 줄이지 않았다. 정답 결측은 채우지 않았으며, 기준선의 현재 관측 결측에 대한 공통 과거값 대체 규칙을 유지했다. 음수 예측을 잘라 채점한 개수는 모두 0이었다.

주 결과의 유의 판정은 원래 통합 채점 호출의 **52검정 Holm 보정**과 **7일 달력 블록 bootstrap 95% 구간**을 함께 사용한다. 52검정은 기준선 대비 28개와 사전 선언한 입력 조건 6대비×4지평 24개다. 아래 별도 모델 비교와 추가 국내 대비는 이번 검토의 사후 탐색으로 구분했다.

2023은 최적 epoch 선택에 사용한 검증 구간이므로 독립적인 최종시험으로 해석하지 않는다. 2024도 이전 프로젝트에서 이미 검토한 연도이므로, 이후 결과를 보며 설계를 바꾸면 새 미사용 시험자료가 생긴 것으로 표현할 수 없다.

## 3. 입력 조건의 의미

모든 조건은 서울 중구를 포함한 635개 노드와 동일한 거리+학습형 그래프 구조를 유지한다. 제외된 관측은 정규화 전에 가리고, 제외 노드는 잠재 중계 노드로 남긴다. 국내 전용은 국외 PM2.5 관측값을 사용하지 않는다는 의미이며 국외 노드와 위치 정보까지 삭제한 설계는 아니다.

| 조건 | 사용 가능한 외부 PM2.5 계열 | 경로 요약 특징 |
| --- | --- | --- |
| M1 | 국내 514 | 없음 |
| M2 | 국내 514 + 서울 600 km 이내 중국 21 = 535 | 없음 |
| M3 | 국내 514 + 중국 전체 77 = 591 | 없음 |
| M4 | M3 + 일본 42 + 몽골 1 = 634 | 없음 |
| M3T | M3와 같음 | 국내·중국으로 제한한 218개 |
| M4T | M4와 같음 | 국내·중국·일본·몽골 322개 |

중국에는 개별 관측소와 도시 평균이 섞여 있어 모두를 동일한 실측 관측소 개수로 부르면 부정확하다. 러시아 국경 부근 중국 도시 자료는 중국 자료이며 러시아 관측자료가 아니다.

M3T의 경로 농도·가용성은 국내·중국 범위로 제한되어 있다. 이전 팀원 결과에서 우려했던 일본·몽골 포함 `all` 경로 집계 혼입은 이번 계약 검사에서 발견되지 않았다. 경로 위치·기상 등 물리적 경로 자체가 국경을 넘는 것과 제외 국가의 오염 관측값이 들어가는 것은 구분한다.

## 4. 실험 간 성능

2024년 RMSE, 단위 µg/m³. 낮을수록 좋다.

{markdown(tables['test'])}

- 3·6h 최저 RMSE는 M1, 12·24h는 M3T다. 후자는 시험자료에서 관측된 최소값이며 독립적으로 검증된 최종 모델 선정 결과가 아니다.
- M1의 Persistence 대비 RMSE 개선율은 16.81 / 20.55 / 22.24 / 22.42%다. 전 지평에서 유의하다. 여섯 조건 모두 Persistence 대비 전 지평에서 유의한 개선이다.
- **원래 선언한 입력 추가 비교 24개 중 유의한 개선은 0개다.**
- M2는 M1보다 RMSE가 3·6·12·24h 모두 크다. 6h 악화는 ΔRMSE=+0.288892(+3.98%), CI=[+0.150993,+0.465691], Holm p=0.013218로 유의하다.
- M3는 M2보다는 전 지평에서 수치가 좋아지지만, M1보다 전 지평에서 나쁘다. '먼 중국을 더하니 좋아졌다'만 보여 주면 국내 기준선보다 개선되지 않았다는 사실을 놓친다.
- M4의 M3 대비 변화도 일관되지 않고 유의하지 않다. 일본과 몽골을 함께 추가했으므로 둘의 효과를 각각 분리할 수도 없다. 이들이 도움 없어야 정상이라는 전제도 두지 않는다.
- M3T는 M3보다 12h 1.75%, 24h 2.11% 수치 개선이다. 24h ΔRMSE=-0.231360, CI=[-0.638135,+0.134319], Holm p=1.0으로 유의하지 않다. M1 대비 개선은 12h 약 1.00%, 24h 약 1.68%다.
- M4T는 M4보다 3·6h에서 나쁘고 12·24h에서 수치상 좋아진다. 전 지평에서 유의한 개선은 아니다. 2023 검증에서는 M4T의 3h가 M4보다 유의하게 나빴다.
- 유의하지 않음은 효과가 정확히 0이거나 두 모델이 동등하다는 입증이 아니다. 이를 주장하려면 의미 있는 차이의 허용범위와 동등성·비열등성 검증을 미리 정해야 한다.

개선율은 `100×(기준 RMSE−후보 RMSE)/기준 RMSE`다. 음수는 악화다.

{markdown(comparisons)}

2023 검증 RMSE:

{markdown(tables['validation'])}

2023에서 M3가 M1보다 6·12h 수치상 좋았던 관계가 2024에서 뒤집혔다. 따라서 검증 오차가 조금 낮다는 것만으로 권역의 안정적인 추가 가치를 판단할 수 없다.

## 5. 이전 DCRNN과의 비교

외부 PM 범위가 같은 GWN M4와 DCRNN F를 동일 평가 시점·동일 정답으로 짝지어 비교했다. 2024년 결과:

{markdown(dcrnn_table)}

3·6·12h는 이번 사후 4지평 Holm 보정 및 국내 비교까지 포함한 8검정 민감도에서도 개선 판정을 통과했다. 24h는 CI가 0을 포함한다. 이 검정은 원래 GWN 52검정 가족에 포함된 결과가 아니며, seed에 따른 훈련 변동도 나타내지 않는다.

국내 전용끼리 비교한 GWN M1은 DCRNN DN보다 전 지평에서 유의하게 좋았다. 이전 DCRNN은 DN의 단기 오차가 매우 크고 일부 epoch의 손실·검증 오차가 폭증했다. 안정적인 GWN 국내 기준선을 쓰자 국외 추가 이득이 사라진 점은, 이전 DCRNN의 국외 효과 크기를 그대로 일반화하기 어렵게 한다.

다만 두 모델은 구조뿐 아니라 그래프, 학습률, 정규화, 출력 방식, 직접 활용하는 과거 범위가 다르다. 따라서 'Graph WaveNet 구조가 DCRNN 구조보다 원래 우월하다' 또는 '거리 그래프가 수송 그래프보다 우월하다'는 결론은 이번 비교만으로 낼 수 없다.

## 6. 다른 팀원 결과와의 비교

공통으로 자료가 있는 2023 검증 RMSE를 참고로 나란히 놓았다.

{markdown(team_md)}

- A SARIMAX WNF보다 GWN M4는 전 지평 수치상 좋다. A의 원시 예측이 현재 비교 입력에 없어 이에 대한 짝지은 유의성은 판정하지 않았다.
- 팀 B LightGBM F600 앙상블보다 GWN M4가 전 지평에서 수치상 나쁘다.
- 팀 C GRU F600 앙상블보다 GWN M4는 3·6h가 좋고 12·24h가 나쁘다. 팀 C GCN+GRU F600보다 24h는 나쁘다.
- 팀 결과는 주로 F600·3개 seed 앙상블이고 GWN은 F634·seed11이므로, 위 표는 같은 정보·같은 학습 조건의 구조 비교가 아니다.
- 같은 F634·seed11의 로컬 LightGBM 참고 실행과 GWN M4는 원시 예측의 평가 시점·정답이 일치한다. GWN M4가 전 지평에서 수치상 나쁘지만, 사후 4지평 Holm·bootstrap 기준으로 유의한 차이는 없다. 이 로컬 실행은 팀원 B의 최신 본실험과 구분한다.
- 팀 B의 별도 2024 결과는 평가 행이 2840/2835/2828/2816으로 이번 2874/2873/2871/2867보다 적고, 일부 M3T에 일본·몽골 포함 경로 집계가 섞여 있다. 직접적인 최종시험 우열·유의성 비교에 쓰지 않는다.

## 7. 학습·모델 설정 점검

{markdown(small)}

- seed11 하나, Adam LR0.001, weight decay0.0001, dropout0.3, 학습률 scheduler 없음, 원단위 MAE, 유효 batch64(micro8), 최대50 epoch·patience10.
- 여섯 조건의 파라미터 수는 모두 339,832개. 노드 순서·고정 거리 그래프 해시가 같다. ASOS·달력 기본 53개와 경로 자리 322개를 모든 조건에 동일하게 두고, 값과 결측 채널을 합친 문맥 크기750을 유지한다. 미사용 경로 자리는 상수 결측으로 가린다.
- 모든 기록된 학습 곡선이 유한하고 검증 6h RMSE의 최대값도 9.66 미만이다. DCRNN처럼 수천 단위 폭증한 흔적은 없다. 안정성이 좋아졌다고 해서 이번 차이의 원인을 특정 설정 하나로 귀속할 수는 없다.
- M1은 이전 100/20 실행에서 누적40 epoch·최적29 epoch 상태를 가져왔다. 정책 변경 시 이미 bad_epochs=11이어서 새 학습을 더하지 않고 보존된 최적 가중치로 예측했다. 따라서 최종 배치만 보고 M1을 처음부터50/10으로 재학습한 것으로 설명하면 안 된다.
- M2는50 epoch 한도까지 도달했고 마지막 최적 epoch는42다. '모든 조건이 조기종료했다'는 설명은 틀리다.
- 모든 최적 epoch는 검증 **6h RMSE**로 선택했다. 12·24h마다 따로 최적화한 모델이 아니다. 24h를 주요 목표로 바꾸려면 검증 단계에서 선택 기준을 미리 바꾸고 별도 실험으로 관리해야 한다.
- 원논문 프로파일의 시간 수용영역은 **13h**다. 로더가48h를 제공해도 추론 때 마지막 출력의 PM 시계열 합성곱은 최근13개 시간 위치를 직접 이용한다. 학습 BatchNorm의 과거 위치 집계와 경로 요약 문맥의 오래된 정보는 별개다. 따라서48h 전체의 중국 시차 신호를 직접 학습했다고 설명할 수 없다.
- `adaptive`는 학습한 인접행렬이며 추론 시 매시각 바람에 따라 자동으로 바뀌는 그래프라는 뜻이 아니다. M3T/M4T도 그래프를 HYSPLIT 수송 그래프로 교체하는 실험이 아니라 경로 요약 문맥을 추가한 실험이다.
- 고정 거리 지지행렬의 비영 원소는284,370개이고 학습형 지지행렬은635²=403,225개가 모두 비영이다. 연결이 이처럼 넓게 열려 있는 설계에서 '바람에 맞는 몇 관측소만 자동 선정했다'고 설명하면 안 된다. 학습 가중치를 물리적 수송량이나 예측 기여율로 해석하지 않는다.
- 국외 미관측 노드도 잠재 중계 노드로 남는다. 이 실험은 관측정보 사용 효과를 비교하며 노드·그래프 자체 제거 효과와 구분한다.
- 개별 `run_summary.test_scored=false`는 갱신되지 않은 요약 필드다. 실제 `completion.json`의test_scored=true, 예측 파일, 통합test 채점이 모두 있어 시험이 빠진 실행으로 오해할 필요는 없다. 향후 상태 요약 필드 정리는 별도 코드 수정사항이다.

## 8. 조건별 추가 인사이트

조건별 표와 농도별 오차는 모두 사후 기술 분석이며 다중 비교를 통과한 하위집단 효과라는 뜻이 아니다.

- M4 대 M1의24h에서 겨울 RMSE는 약2.36% 악화했고, 여름은 약3.85% 개선했다. 경로 바람이 서쪽인 조건의 개선은 약0.30%에 불과하다. 이번 결과는 '겨울·서풍이면 국외 추가가 유리하다'는 단순 설명을 지지하지 않는다.
- 실제24h 후 농도가35 µg/m³ 이상인297행에서 M3T는 M3의 RMSE25.52를22.91로 약10.25% 줄였다. 그러나35 미만2570행에서는7.68에서8.25로 약7.39% 악화했다. 전체 개선은 약2.11%이며 유의하지 않다.
- 이297행에서 M3와M3T의 MBE는 각각 약−21.61,−17.16 µg/m³로, 고농도 과소예측이 여전히 크다. 고농도에서 좋아진다는 단정에 앞서 사건 단위·계절별 재현성을 확인해야 한다.
- 위35 기준은 **실제로 미래에 관측한 정답으로 오차를 진단한 것**이다. 실행 시점에 알 수 없는 미래 고농도 여부를 모델 선택 조건으로 쓰면 누출이다. 경보 적중률·오경보율·리드타임을 계산한 결과도 아니다.
- 2023 겨울24h 평가행357개, 2024 겨울712개로 표본 구성이 다르다. 2023의 겨울 정답 결측 때문에 연도별·계절별 수치 차이를 모두 모델 변화로 해석하면 안 된다.
- 해외 자료가 독립적인 예측정보를 추가하지 않았다는 결과는 물리적 국외 기여율이0이라는 뜻이 아니다. 국내 다수 관측에 이미 담긴 정보, 모형 용량·정규화, 관측 결측·오류, 수송 시차와 수용영역의 불일치 등이 후보 설명이다. 이번 결과만으로 국제협력 비용 절감액을 산정할 수 없다.

## 9. RQ·가설에 대한 판단과 다음 실험의 우선순위

| 질문·주장 | 이번 결과가 허용하는 결론 |
| --- | --- |
| 국외 관측이 국내 기준선에 추가 가치가 있는가 | 이 GWN 설정·seed11의 전체 평균에서는 입증하지 못함 |
| 예측 지평이 길수록 중국 추가 이득이 커지는가 | M3가 M1보다 전 지평에서 나빠 단조 가설 지지 못함 |
| HYSPLIT 요약 특징이 도움이 되는가 | 장기 수치 개선 후보는 있으나 전체 평균 유의 개선 미확보 |
| 일본·몽골은 항상 무익한가 | 국가별 분리 실험도 아니고 통계적 동등성도 입증하지 않아 결론 불가 |
| 풍향·계절에 따른 조건부 가치가 있는가 | 탐색 표는 있으나 일관된 겨울·서풍 가설과 동적 선택 성능은 미입증 |
| DCRNN보다 이 실행의 예측이 나은가 | 같은F634 평가행에서3·6·12h 개선.24h 불확실 |
| 학습형 그래프가 실제 수송을 찾았는가 | 고정 그래프/학습형 그래프 대조·안정성 검증 없이 결론 불가 |

1. M1과M3 또는M3T 소수 조건의 추가 seed로 비교 안정성을 먼저 확인한다. 같은seed 하나의 달력 bootstrap은 훈련 초기화에 따른 변동을 대신하지 않는다.
2. 목적이12·24h라면 검증 선택 지표를 사전 지정한 별도 장기 실험으로 관리한다. 기존6h 주평가 결과를 소급 변경하지 않는다.
3. 13h 대48h 이상 시간 수용영역을 한 번에 바꾸는 비교를 검증구간에서 진행할 수 있다. 이것은 알려진 시차 범위의 사용 여부를 검증하는 설계이지 개선 보장이 아니다.
4. 국외 전체·수백 특징을 계속 늘리기 전에 적은 경로 특징·국내 기준선 강도·가용성·시간 정렬을 나누어 본다. 현재의 통계 재검증에서는 정렬·평가행 오류를 발견하지 않았다.
5. 계절별 모듈/가중치 선택은 학습·검증 자료에서 정하고, 이미 본2024에서 탐색한 조건을 같은2024의 확증 결과로 부르지 않는다. 사용 시점에 아는 바람·경로·현재 농도만 게이트 입력으로 허용한다.
6. 이번 요청에서는 추가 학습·데이터 재QC·모델·채점기 수정은 수행하지 않았다. 사후 비교 CSV는 기존 출력과 구분해 저장했다.

## 10. 다음에 공유할 폴더

Google Drive:

`내 드라이브/Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI/20261111_014538_406520_eaa72758/`

Vast 서버:

`/workspace/TSA/Result/GraphWaveNet/VastAI/full/20261111_014538_406520_eaa72758/`

이번에는 위 완료 폴더 하나면 성능 검토에 충분하다. 앞으로도 단순 생성시각이 아니라 `batch_manifest.json`의 `mode=full`, `status=passed`, 조건 완료 여부, 필요한validation/test 통합 채점 존재 여부로 선택한다. Google Drive가 ZIP을 여러개로 나누면 같은 폴더의 나머지 ZIP도 원본 보관·체크포인트 복구에는 필요하다.

용량을 줄여 성능만 공유할 때는 `Common_Scoring/Combined`, 각조건의 `Exports`, `Models/.../run_summary.json`, 학습곡선과 설정·계약 JSON을 포함하면 된다. 체크포인트를 제외한 공유본은 학습 재개용 전체 백업과 구분한다. 인증설정 파일은 결과 검토에 필요 없다.

## 근거 파일

- 두 첨부 ZIP의 해시·CRC·구성: `Results/Verification/archive_integrity.json`, `inventory.json`.
- 완료·입력·학습 계약: `Results/Verification/Final_Run_Metadata`, `protocol_protocol_audit.json`.
- 원본 공통 채점 6개 파일 및 부가자료: `Results/Original_Combined`.
- 원시 예측 재채점 일치: `Results/Verification/independent_stats_statistics_audit.json`.
- 이전 DCRNN·팀 자료 비교 출처·동일 격자 검사: `Results/Verification/team_comparison_comparison_audit.json`.
- 공유용 한국어 요약 표 및 상세 비교 표: `Results/Tables`.
'''
    (DEST/'README.md').write_text(report,encoding='utf-8')
    files=[]
    for file in sorted(DEST.rglob('*')):
        if file.is_file():
            files.append({'path':str(file.relative_to(DEST)).replace('\\','/'),'bytes':file.stat().st_size,
                          'sha256':hashlib.file_digest(file.open('rb'),'sha256').hexdigest()})
    manifest={'destination_relative':str(RELATIVE),'staging':str(DEST),'files':files,
              'source_models_modified':False,'source_data_modified':False,'source_scorer_modified':False}
    (WORK/'delivery_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'staging':str(DEST),'destination_relative':str(RELATIVE),'files':len(files),'bytes':sum(v['bytes'] for v in files)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
