"""Package a conditional model-result audit without altering prior outputs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent
PREVIOUS=WORK.parent/'dcrnn_h48_results_review_20261112'
LABELS={'DJF':'겨울(12·1·2월)','MAM':'봄(3–5월)','JJA':'여름(6–8월)','SON':'가을(9–11월)',
        'west':'서쪽(NW·W·SW)','north':'북쪽(N·NE)','east':'동·남쪽(E·SE·S)','local':'국지(<25km)'}


def table(df):
    columns=list(df.columns)
    def fmt(x):
        if isinstance(x,float):return f'{x:.3f}'
        return str(x).replace('|','\\|')
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def detailed(df):
    rows=[]
    for r in df.itertuples():
        rows.append({'조건':LABELS[r.condition_value],'시간':f'{r.horizon}h','예측 표본 n':r.n,
            '발생 날짜':r.unique_origin_days,'활성 7일 구간':r.active_nonoverlapping_7day_bins,
            '국내 M1 RMSE':round(r.M1_RMSE,3),'중국 추가 M3 RMSE':round(r.M3_RMSE,3),
            '개선율(%)':f'{r.improvement_pct:+.2f}',
            '개선율 95% CI':f'[{r.improvement_ci_low_7d:+.2f}, {r.improvement_ci_high_7d:+.2f}]',
            'Holm p':f'{r.p_holm64:.3f}', '소표본 주의':'예' if r.small_sample else ''})
    return table(pd.DataFrame(rows))


def main():
    out=WORK/'Deliverable'
    if out.exists():raise FileExistsError('Refusing to overwrite a prepared review.')
    for sub in ['Codes','Results/Tables','Results/Figures','Audit']:(out/sub).mkdir(parents=True,exist_ok=True)
    for p in (WORK/'Tables').glob('*.csv'):shutil.copy2(p,out/'Results/Tables'/p.name)
    for p in (WORK/'Figures').glob('*'):
        if p.is_file():shutil.copy2(p,out/'Results/Figures'/p.name)
    for name in ['conditional_inference.py','descriptive_stability.py','audit_route_definition.py','plot_conditional.py','build_conditional_report.py']:
        shutil.copy2(WORK/name,out/'Codes'/name)
    for name in ['analysis_protocol.json','conditional_audit.json','descriptive_audit.json','route_definition_audit.json']:
        shutil.copy2(WORK/name,out/'Audit'/name)
    (out/'Codes/README.md').write_text('코드 기록입니다. 각 파일 상단의 WORK/PROJECT/PREVIOUS/BATCH가 이번 로컬 분석 위치를 가리킵니다. 기존 원시 예측·공통 평가 격자·시계열 원본을 보존해야 재현할 수 있습니다. 모델을 재학습하거나 예측을 새로 생성하지 않았고 공통 채점기를 바꾸지 않았습니다. conditional_inference.py는 analysis_protocol.json과 일치하는 설정만 허용하며, 전체 연간 3시간 달력의 조건 외·정답 결측을 유지한 paired moving-block bootstrap을 사용합니다. 빠른 prefix-sum 계산은 직접 블록 연결 방식과 대조했습니다.\n',encoding='utf-8')
    d=pd.read_csv(WORK/'Tables/conditional_primary.csv')
    desc=pd.read_csv(WORK/'Tables/yearly_descriptive.csv')
    keys=['split','condition_type','condition_value','horizon','candidate','reference']
    joined=desc.loc[desc.candidate.eq('M3')&desc.reference.eq('M1')].merge(d,on=keys,validate='one_to_one')
    assert len(joined)==64
    np.testing.assert_array_equal(joined.n_x,joined.n_y)
    np.testing.assert_allclose(joined.RMSE_diff,joined.rmse_diff,atol=1e-12)
    np.testing.assert_allclose(joined.RMSE_improvement_pct,joined.improvement_pct,atol=1e-12)
    np.testing.assert_array_equal(joined.n_days,joined.unique_origin_days)
    season_test=d.loc[d.split.eq('test')&d.condition_type.eq('season')].sort_values(['condition_value','horizon'])
    route_test=d.loc[d.split.eq('test')&d.condition_type.eq('route_group')].sort_values(['condition_value','horizon'])
    def ordered(frame,axis):
        order=['DJF','MAM','JJA','SON'] if axis=='season' else ['west','north','east','local']
        return pd.concat([frame.loc[frame.condition_value.eq(k)].sort_values('horizon') for k in order])
    summary=[]
    for axis,level,h in [('season','SON',24),('season','SON',36),('season','SON',48),('route_group','east',24),('route_group','local',24)]:
        vals=d.loc[d.condition_type.eq(axis)&d.condition_value.eq(level)&d.horizon.eq(h)].set_index('year')
        summary.append({'条件':LABELS[level], '时间':f'{h}h',
            '2023 n':int(vals.loc[2023,'n']), '2023 改善与CI':f"{vals.loc[2023,'improvement_pct']:+.2f}% [{vals.loc[2023,'improvement_ci_low_7d']:+.2f}, {vals.loc[2023,'improvement_ci_high_7d']:+.2f}]",
            '2024 n':int(vals.loc[2024,'n']), '2024 改善与CI':f"{vals.loc[2024,'improvement_pct']:+.2f}% [{vals.loc[2024,'improvement_ci_low_7d']:+.2f}, {vals.loc[2024,'improvement_ci_high_7d']:+.2f}]"})
    selected=pd.DataFrame(summary).rename(columns={'条件':'조건','时间':'시간','2023 改善与CI':'2023 개선율과 95% CI','2024 改善与CI':'2024 개선율과 95% CI'})
    selected.to_csv(out/'Results/Tables/key_conditions_korean.csv',index=False,encoding='utf-8-sig')
    for split,year in [('validation',2023),('test',2024)]:
        rows=[]
        for r in d.loc[d.split.eq(split)].itertuples():
            rows.append({'연도':year,'조건 종류':'계절' if r.condition_type=='season' else '경로','조건':LABELS[r.condition_value],
                '예측시간':r.horizon,'표본수':r.n,'발생날짜수':r.unique_origin_days,'활성비중첩7일구간수':r.active_nonoverlapping_7day_bins,
                'M1_RMSE':r.M1_RMSE,'M3_RMSE':r.M3_RMSE,'RMSE개선율_pct':r.improvement_pct,
                '개선율95CI하한_pct':r.improvement_ci_low_7d,'개선율95CI상한_pct':r.improvement_ci_high_7d,
                'RMSE차이_M3minusM1':r.rmse_diff,'차이95CI하한':r.delta_ci_low_7d,'차이95CI상한':r.delta_ci_high_7d,
                'Holm64_p':r.p_holm64,'소표본주의':r.small_sample})
        pd.DataFrame(rows).to_csv(out/f'Results/Tables/{year}_조건별_표본수_개선율_신뢰구간.csv',index=False,encoding='utf-8-sig')
    context24=desc.loc[desc.candidate.eq('M3')&desc.reference.eq('M1')&desc.condition_type.eq('route_group')&desc.horizon.eq(24)]
    cr=[]
    for r in context24.itertuples():
        cr.append({'연도':r.year,'경로':LABELS[r.condition_value],'평가 n':r.n,'현재 중구 농도 평균':r.current_pm25_mean,
          '현재 농도 유효 n':r.current_pm25_n,'현재 농도 결측 n':r.current_pm25_missing_n,'가을 비중(%)':100*r.fraction_SON})
    audit=json.loads((WORK/'conditional_audit.json').read_text())
    assert audit['family_size']==64
    valid_cols=[f'bootstrap_valid_reps_{length}d' for length in [3,7,14]]
    assert d[valid_cols].ge(4000).all().all()
    bootstrap_min=int(d[valid_cols].min().min())
    report=f'''# DCRNN 중국 관측의 계절·수송경로별 가치 — 2026-11-12

**가을 24시간 예측은 후속 검증 후보입니다. 그러나 이번 조건부 분석에서도 중국 자료의 확실한 추가 이득을 확인한 조건은 없습니다.** 가을 24h는 2023·2024 모두 약 3.6–4.0%의 RMSE 개선을 보였지만, 2024의 신뢰구간은 블록 길이에 민감하고 다중비교 기준을 통과하지 못합니다. 서쪽 경로는 12·24·36·48h 모두 두 해에서 RMSE가 소폭 나빠졌습니다.

## 비교 범위

- 대상 실행: `20261112_003040_429483_85af4013`, seed11. 2023 validation·2024 test의 보존된 원시 예측만 분석했습니다.
- **M1=국내 514계열 PM2.5 관측**, **M3=M1+중국 77계열**입니다. 중구 노드, ASOS·달력, 635노드 및 수송 방향 그래프는 동일합니다. 국외 위치/잠재 중계 노드와 HYSPLIT 그래프까지 제거한 비교는 아닙니다.
- 주 비교는 M3−M1, 지평은 12·24·36·48h입니다. M2(가까운 중국21개)−M1과 M3−M2는 기술적 보조표만 제공했으며 새로운 유의성 주장을 하지 않았습니다.
- 계절과 경로는 **예측을 시작한 t시점 기준**입니다. t+h 시점의 실제 도착 공기 경로로 미래를 나눈 분석이 아닙니다.
- 이미 전체 결과와 일부 조건별 점수를 확인한 뒤 시행한 **사후 탐색 분석**입니다. 2023은 최적 epoch 선택에 사용된 해이고, 2024도 여러 차례 확인한 기간입니다. 두 해에 같은 방향이라고 독립적인 두 번의 최종 검증으로 보지는 않습니다.

## 조건 정의와 불확실성 계산

계절은 겨울12·1·2월, 봄3–5월, 여름6–8월, 가을9–11월입니다. 경로는 t에 서울(37.57N,126.98E)·500m AGL에 도착한 공기의 **12시간 전 위치를 서울에서 바라본 방위**로 분류합니다.

| 경로 이름 | 포함 방위·거리 | 해석상 주의 |
| --- | --- | --- |
| 서쪽 | NW·W·SW | 중국 통과 여부를 직접 뜻하지 않음 |
| 북쪽 | N·NE | 북서쪽은 서쪽 그룹에 포함됨 |
| 동·남쪽 | E·SE·S | 원래 코드 `east`에는 남쪽도 포함됨 |
| 국지 | 서울로부터 12h전 위치가25km 미만 | ‘국내 배출만’·‘정온’과 동의어가 아님 |

원래11,688개 원점의 실제 12h 좌표로 방위를 재계산해 라벨 일치를 확인했습니다. 기존 12h 방위는 국외 국가 분류나 전체72h 궤적 군집이 아니므로, 서쪽 그룹의 낮은 이득을 곧바로 ‘중국발 수송 효과 없음’으로 읽어서는 안 됩니다. 저장된 GDAS 사후 경로의 시간 정렬 검증과 실시간 자료 가용성 보장은 별개입니다.

개선율은 **100×(1−M3 RMSE/M1 RMSE)**입니다. 양수는 중국 추가가 좋고, 음수는 나쁩니다. 95% CI는 7일 시간 블록 paired bootstrap 5,000회(seed8211)로 계산했습니다. 같은 시점의 두 모델 오차를 함께 뽑고, 연간3시간 달력의 결측·조건 밖 시점은 그대로 유지했습니다. 조건에 해당하는 행만 모아 연속 시계열처럼 붙이지 않았습니다. prefix-sum 계산은 실제 블록을 연결하는 구현과 대조했습니다.

**n은 유효한 예측 시점 수**입니다. 하루8개 예측과 긴 지평의 중첩 때문에 n개의 독립 사건이 있다는 뜻은 아닙니다. 발생 날짜 수와 고정된 비중첩7일 구간 중 유효 행이 있는 구간 수도 별도 표시합니다. 이 구간 수 역시 독립 표본 수로 취급하지 않습니다. n<100 또는 활성7일 구간<10은 소표본 주의를 표시했습니다(검정력 계산으로 정한 기준이 아님).

신뢰구간은 **각 비교의 점별 구간**이며, 여러 조건 전체를 동시에 보장하는 구간이 아닙니다. 부가적으로 제곱오차 DM/HAC와 Holm 보정을 적용했습니다. 가족은 두 해×4지평×(4계절+4경로)=64개이고, 기존 주 비교84개와는 분리된 탐색 가족입니다. 기본 HAC 지연은 기존 max(h/3−1,0), 감도 분석은56슬롯(7일)입니다. 3일·14일 bootstrap도 계산했습니다. 유의성에 유리한 블록/지연을 고르지 않았습니다. 재표본에 해당 조건의 유효 관측이 전혀 없으면 그 반복은 제외하며, 실제 유효 반복 수는 셀·블록 길이별 {bootstrap_min:,}–5,000회입니다.

기본 동시 판정(7일 ΔRMSE CI 상한<0 및 Holm64 p<0.05)을 통과한 **개선0개·악화0개**입니다. 한 조건 안의 차이는 ‘그 조건이 다른 조건보다 더 유리하다’는 상호작용 검정과도 다릅니다. 계절/경로의 효과 차이 자체를 인과적으로 검정한 결과는 아닙니다.

## 2024 계절별 결과

RMSE 단위는 µg/m³, 개선율과 구간은 %입니다.

{detailed(ordered(season_test,'season'))}

## 2024 수송경로별 결과

{detailed(ordered(route_test,'route_group'))}

## 두 해에서 개선 방향이 겹치는 주요 후보

{table(selected)}

### 가을24h: 가장 먼저 재검증할 후보

2023은 RMSE8.894→8.541(3.97%),2024는7.793→7.512(3.60%)입니다. 표본은719/720개, 각각91개 날짜에 걸쳐 있습니다. 다만 활성7일 구간은 각14개이며, 719·720개의 독립 오염 사건으로 볼 수 없습니다.

2024의7일 개선율CI는 **[+0.03,+7.46]%**로 간신히0을 제외하지만,3일은 **[−0.27,+7.52]%**,14일은 **[−0.01,+7.00]%**로0을 포함합니다. 기본 Holm p는2023 .117,2024 1.000입니다. HAC56 감도에서는2023 가을24h가 보정 후 통과하더라도2024에는 유지되지 않습니다. 따라서 ‘가을 모듈을 바로 채택’이 아니라 **고정된 후보 규칙을 새 기간에서 검증할 근거**입니다.

보조표에서 가까운 중국만 넣은 M2도 가을24h에3.61%(2023)·4.92%(2024) 개선됐습니다. 2024 RMSE는M2 7.409로M3 7.512보다 낮습니다. 이 방향만으로 먼 중국의 정보까지 필수라고 설명할 수는 없으며, M2의 추가 유의성은 이번에 검정하지 않았습니다.

### 가을36·48h: 방향은 일치하나 구간이 넓음

2024 개선율은36h0.96%[−2.56,+5.26],48h3.05%[−0.89,+7.62]입니다. 2023도 개선 방향이지만 구간이0을 포함합니다. 최대지평을 늘린 것으로 국외 자료의 효용이 자동으로 드러난 것은 아닙니다.

### 동·남쪽24h: 작은 후보, 수송 인과로 설명하지 않기

2023 개선율1.00%[−0.59,+3.91],2024 2.13%[−0.05,+4.70]로 양해 방향은 같습니다. 하지만7일 구간이0을 포함하고 보정 p도 통과하지 못합니다. 이 그룹의 경로는직전12h의 E·SE·S일 뿐, 공기의 과거48·72h가 중국을 전혀 지나지 않았다는 뜻이 아닙니다. 중국 관측이 공통 기상·넓은 공간 패턴의 대리 정보일 가능성 등도 남습니다. 어느 경우든 이 결과로 중국 배출의 물리적 기여를 말할 수 없습니다.

### 국지24h: 큰 수치, 작은 표본

2024 개선율12.36%[+5.97,+18.06],2023 6.02%[+1.40,+12.32]입니다. 하지만 표본은2023 **30개/24일**,2024 **65개/38일**뿐입니다. 2024 기본 Holm p=.453, HAC56 p=.692로 통과하지 못합니다. 넓은 지역 전체에 적용할 모듈이나 ‘국지 정체에서도 중국 자료가 필수’라는 결론을 만들기에는 부족합니다. 이 조건의36·48h 구간도0을 포함하고 연도별 방향이 다릅니다.

### 겨울·서쪽 경로: 현재 가설을 뒷받침하지 못함

겨울24·36·48h는2024에서0.76–1.24% 개선됐지만 구간이0을 포함하고,2023에서는 모두 악화 방향입니다. 겨울24h 평가 표본도2023 357개/46일,2024 712개/90일로 크게 다릅니다. 이 차이는 정답 결측의 영향을 받습니다.

서쪽은 두 해의12·24·36·48h에서 모두 RMSE가 약간 나빠졌고,2024 개선율은−0.81/−0.08/−0.15/−0.06%입니다. 모든 구간이0을 포함합니다. ‘서풍이면 중국 자료가 도움된다’는 단순 규칙을 이번 결과가 뒷받침하지는 않습니다. 다만 MAE와 RMSE의 방향이 다른 지평이 있어, 모든 종류의 예측 이득이 없다고 일반화하지 않았습니다.

## 조건의 표본 구성은 같지 않음

{table(pd.DataFrame(cr))}

현재 농도 평균은정확한t시각 실측만 사용했고 결측은 과거값으로 채우지 않았습니다. 경로별 현재 농도·계절 구성과 예측 난이도가 다르므로, 경로별 개선율의 차이를 경로 자체의 원인 효과로 해석하면 안 됩니다. 예를 들어2024 24h에서 서쪽 현재 농도는평균20.94,동·남쪽은14.08µg/m³이고, 가을 비중은각15.0%·38.4%입니다. 계절×경로 보조표에서는동·남쪽24h 개선이여러계절에 분포하지만, 추가 통제나 상호작용 검정은 하지 않았습니다.

## 다음 단계와 발표 문장

이번에 선택한 조건으로 모델을 재학습하거나2024를 보고 가중치를 바꾸지는 않았습니다. 가을24h를 우선 후보로 남기되, M1/M2/M3와 비적응형 결합을 같은 조건에서 비교하는 규칙을 먼저 고정해야 합니다. 같은2024를 다시 평가해 독립적인 개선이라고 부르면 안 됩니다. 시간순 새 기간, 또는 훈련·선택을 다시 분리한 rolling-origin 검증이 필요합니다. 가능하면핵심조건의추가seed도확인해야하며, 지금CI는가중치를고정한기존예측의시간불확실성만반영합니다.

발표에는 다음처럼 정리할 수 있습니다.

> 중국 관측을 추가한 평균적 이득은 작았지만, 가을 24시간 예측에서는 두 해 모두 약 4%의 RMSE 감소가 나타났다. 다만 다중비교와 시간 블록 민감도를 고려하면 아직 확정적이지 않다. 서쪽 경로에서는 안정적인 이득을 확인하지 못했으며, 국지 경로의 큰 개선은 소표본 결과다. 따라서 조건부 관측 활용은 검증이 필요한 후속 가설로 남긴다.

## 파일과 검증 기록

- `Results/Tables/2024_조건별_표본수_개선율_신뢰구간.csv`: 시험기간32개 조건 전체의 한국어 공유표.
- `Results/Tables/2023_조건별_표본수_개선율_신뢰구간.csv`: 선택기간32개 조건 전체.
- `Results/Tables/conditional_primary.csv`: 모든3·7·14일구간,DM/HAC·Holm,표본수·유효bootstrap반복수.
- `Results/Tables/cross_year_stability.csv`: 두 해의방향·불확실성대조.
- `Results/Tables/yearly_descriptive.csv`: M2−M1/M3−M2 보조표를 포함하며통계적추가검정없음.
- `Results/Tables/route_sample_context.csv`, `route_season_composition.csv`: 경로표본·현재농도·계절구성.
- `Results/Figures`: 계절·경로별 개선율과95%CI 그림(PNG·PDF).
- `Audit/analysis_protocol.json`: 이번탐색계산규칙. `conditional_audit.json`·나머지감사파일에는원본해시와정합성검사.

두 독립 계산의 64개 셀 표본 수·RMSE 차이·개선율이 일치했고, 모든 셀·블록 길이의 유효 bootstrap 반복 수는 사전에 정한 80% 기준을 충족했습니다. 모델·데이터·공통 채점기·이전 결과는 보존했습니다. 결과 생성: {datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}.
'''
    (out/'README.md').write_text(report,encoding='utf-8')
    files=[]
    for p in sorted(out.rglob('*')):
        if p.is_file():files.append({'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest={'analysis':'conditional M3 versus M1, exploratory', 'source_batch':'20261112_003040_429483_85af4013',
        'model_fit_or_inference_executed':False,'source_files_modified':False,'source_hashes':audit['source_hashes'],
        'tables_independently_matched_cells':len(joined),'files':files}
    (out/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(out/'README.md'),'files':len(files)+1,'bytes':sum(x['bytes'] for x in files)},ensure_ascii=False))


if __name__=='__main__':main()
