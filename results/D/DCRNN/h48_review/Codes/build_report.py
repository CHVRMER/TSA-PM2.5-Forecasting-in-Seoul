"""Assemble a read-only result review; never train or modify source artifacts."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import pandas as pd

WORK = Path(__file__).resolve().parent
ORDER = ['M1', 'M2', 'M3', 'M4', 'M3T', 'M4T']
HORIZONS = [3, 6, 12, 24, 36, 48]
BATCH = WORK/'extracted/20261112_003040_429483_85af4013'


def md_table(frame, columns=None, decimals=3):
    if columns is not None:
        frame = frame[columns]
    def value(v):
        if isinstance(v, float):
            return f'{v:.{decimals}f}'
        return str(v).replace('|', '\\|')
    return '| ' + ' | '.join(map(str, frame.columns)) + ' |\n' + '| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(value(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))


def main():
    output = WORK/'Deliverable'
    if output.exists():
        raise FileExistsError('Never overwrite a prepared review: '+str(output))
    for name in ['Codes', 'Results/Tables', 'Results/Figures', 'Results/Original_Combined', 'Results/Verification', 'Audit']:
        (output/name).mkdir(parents=True, exist_ok=True)
    for source in sorted((WORK/'Tables').glob('*.csv')):
        shutil.copy2(source, output/'Results/Tables'/source.name)
    for source in sorted((WORK/'Figures').glob('*')):
        if source.is_file():
            shutil.copy2(source, output/'Results/Figures'/source.name)
    for split in ['validation', 'test']:
        src = next((BATCH/'Common_Scoring/Combined'/split).glob('*'))
        shutil.copytree(src, output/'Results/Original_Combined'/split/src.name)
    for source in (WORK/'Verification').rglob('*'):
        if source.is_file():
            target = output/'Results/Verification'/source.relative_to(WORK/'Verification')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ['archive_inventory.json', 'training_audit.json', 'effects_audit.json', 'comparison_audit.json', 'verification_audit.json']:
        shutil.copy2(WORK/name, output/'Audit'/name)
    for name in ['batch_manifest.json', 'resolved_configuration.json', 'predeclared_comparisons.json', 'evaluation_grid_contract.json', 'source_contract.json']:
        shutil.copy2(BATCH/name, output/'Audit'/name)
    # Preserve essential audit scripts as code records; no source data or weights.
    for name in ['prepare_review.py', 'training_audit.py', 'effects_analysis.py', 'comparison_analysis.py', 'verification_analysis.py', 'plot_review.py', 'build_report.py']:
        shutil.copy2(WORK/name, output/'Codes'/name)
    (output/'Codes/README.md').write_text('감사에 사용한 코드 기록입니다. 원본 ZIP, 기존 프로젝트의 공통 채점기·기준선 격자·이전 모델 결과가 필요합니다. 파일 상단의 WORK/PROJECT/SCORER 등 경로는 이번 로컬 감사 위치를 가리킵니다. 모델 가중치를 역직렬화하거나 학습하지 않았습니다. prepare_review → training_audit/effects_analysis/comparison_analysis/verification_analysis → plot_review → build_report 순서이며, 기존 출력과 겹치지 않는 새 작업 위치에서 재현하세요.\n', encoding='utf-8')

    metrics = pd.read_csv(WORK/'Tables/test_condition_metrics.csv')
    comparisons = pd.read_csv(WORK/'Tables/test_predeclared_condition_comparisons.csv')
    baseline = pd.read_csv(WORK/'Tables/test_baseline_metrics.csv')
    training = pd.read_csv(WORK/'Tables/training_protocol_summary.csv')
    previous = pd.read_csv(WORK/'Tables/team_comparison_test_paired_posthoc_holm12.csv')
    archive = json.loads((WORK/'archive_inventory.json').read_text(encoding='utf-8'))
    verified = json.loads((WORK/'verification_audit.json').read_text(encoding='utf-8'))
    pivot = metrics.pivot(index='input_condition', columns='horizon', values='RMSE').reindex(ORDER)[HORIZONS]
    pivot.index.name = '입력 조건'
    pivot.columns = [f'{h}h' for h in HORIZONS]
    rmse = pivot.reset_index()
    persistence = baseline.loc[baseline.model.eq('persistence')].set_index('horizon').reindex(HORIZONS)
    rmse.loc[len(rmse)] = ['Persistence'] + persistence.RMSE.tolist()
    epoch = training[['condition', 'best_epoch', 'trained_epochs', 'best_validation_rmse_h6', 'learning_rate_final']].copy()
    epoch.columns = ['조건', '최적 epoch', '종료 epoch', '검증 6h RMSE', '마지막 학습률']
    epoch['마지막 학습률'] = epoch['마지막 학습률'].map(lambda x: f'{x:.7f}')
    m1 = metrics.loc[metrics.input_condition.eq('M1')].set_index('horizon').reindex(HORIZONS)
    long = m1[['RMSE', 'MAE', 'R2', 'MBE', 'RMSE_improvement_pct']].reset_index()
    long.columns = ['예측 시간', 'RMSE', 'MAE', 'R²', '편향', 'Persistence 대비 RMSE 개선(%)']
    cross = previous.loc[previous.comparison.eq('new_DCRNN_M4_vs_old_DCRNN_F'), ['horizon', 'reference_RMSE', 'candidate_RMSE', 'RMSE_improvement_pct', 'candidate_better_holm12']].copy()
    cross.columns = ['예측 시간', '이전 DCRNN F', '새 DCRNN M4', '개선율(%)', '사후 비교 유의 개선']
    cross['사후 비교 유의 개선'] = cross['사후 비교 유의 개선'].map({True: '예', False: '아니오'})
    selected = comparisons.loc[(comparisons.candidate_condition.eq('M3')&comparisons.reference_condition.eq('M1')) | (comparisons.candidate_condition.eq('M3T')&comparisons.reference_condition.eq('M3')) | (comparisons.candidate_condition.eq('M4T')&comparisons.reference_condition.eq('M4'))].copy()
    selected = selected[['candidate_condition', 'reference_condition', 'horizon', 'rmse_diff', 'ci95_low', 'ci95_high', 'p_holm', 'improved']]
    selected.columns = ['추가 입력', '비교 기준', '예측 시간', 'ΔRMSE', 'CI 하한', 'CI 상한', 'Holm p', '유의 개선']
    selected['유의 개선'] = selected['유의 개선'].map({True: '예', False: '아니오'})
    selected.to_csv(output/'Results/Tables/key_comparisons_korean.csv', index=False, encoding='utf-8-sig')
    report = f'''# DCRNN H48 실행 결과 검토 — 2026-11-12

**학습과 추론은 정상 완료됐고, 이전 DCRNN의 단기 성능도 크게 개선됐습니다. 그러나 이번 실행에서 국외 관측이나 궤적 요약 특징이 국내 관측 기준선에 확실한 추가 이득을 준다는 증거는 없습니다.** 36·48시간까지 예측할 수 있게 됐지만, 그 자체가 장기 예측의 높은 정확도나 고농도 경보 효과를 입증하지는 않습니다.

## 1. 실행·자료·저장 검증

- 실행 ID: `{BATCH.name}`. 여섯 조건 M1/M2/M3/M4/M3T/M4T, seed11, 3·6·12·24·36·48h 모두 완료. 2023 검증과 2024 추론·통합 채점까지 실패 조건 없음. 총 실행 시간 약 6시간 46분.
- 첨부 ZIP CRC 정상. 70개 체크포인트의 commit/manifest/state 크기·해시, best 가중치 6개의 해시가 모두 일치합니다. 가중치를 역직렬화하지는 않았으며, 새 GPU 추론을 다시 실행한 검사는 아닙니다.
- 첨부된 Drive 백업 성공 기록 45개를 확인했고, 최종 백업 목록 497개 파일은 ZIP 내부와 일치합니다. 실시간 Drive에 접속해 확인한 것은 아닙니다.
- 모든 조건에서 같은 노드 순서·수송 그래프·전처리 규칙을 적용했으며 파라미터 수는 421,767개로 같습니다. 중구와 기본 기상·달력 문맥의 정규화 통계는 유지했고, 각 조건의 PM 정규화 통계는 허용된 관측만으로 따로 맞췄습니다. 제외된 관측값은 정규화 통계를 맞추기 전 제거됐고, 그 노드의 학습 관측 수는 0입니다.
- 고정 시험 평가 행은 지평 순서대로 2,874 / 2,873 / 2,871 / 2,867 / 2,863 / 2,859개입니다. 모델별 교집합으로 유리한 행만 고르지 않았습니다. 정답 결측을 채우지 않으며, split 경계 뒤의 정답도 배제합니다.
- 독립 재채점 결과는 `Audit/verification_audit.json`에 기록했습니다. 원본 채점 결과는 `Results/Original_Combined`, 재구성·재계산 결과는 `Results/Verification`에 별도로 보관했습니다. 기본 4개 지평의 원래 격자와 실제 중구 정답을 유지해 36·48h를 다시 구성했습니다.

## 2. 입력 조건은 무엇을 비교하나

| 조건 | 사용하는 외부 PM2.5 관측 계열 | 추가 궤적 요약 특징 |
| --- | --- | --- |
| M1 | 국내 514 | 없음 |
| M2 | 국내 514 + 가까운 중국 21(서울 600km 이내) | 없음 |
| M3 | 국내 514 + 중국 77 | 없음 |
| M4 | 국내 514 + 중국 77 + 일본 42 + 몽골 1 | 없음 |
| M3T | M3와 동일 | 국내·중국 범위 218개 |
| M4T | M4와 동일 | 전체 국가 범위 322개 |

서울 중구 정답 노드는 위 계열 수와 별개입니다. 모든 조건에서 **같은 635노드와 2021–22 궤적으로 만든 방향 그래프**를 유지하고, 제외된 외부 관측만 결측으로 가립니다. 제외 노드는 잠재 중계 노드로 남습니다. 따라서 M1은 ‘국내 PM 관측만 사용하는 조건’이며, ‘국외의 위치·그래프·HYSPLIT을 모두 제거한 시스템’은 아닙니다. M3T/M3 비교도 HYSPLIT 전체의 유무가 아닌 **추가 궤적 요약 변수의 효과**입니다.

## 3. 2024 시험 RMSE

단위는 µg/m³이며 낮을수록 좋습니다. 모두 seed11 결과입니다. 단일 seed의 복사본/평균을 다중 seed 앙상블의 증거로 취급하지 않습니다.

{md_table(rmse)}

수치상 최저는 3h M3, 6h M3T, 12h M1, 24h M3, 36·48h M1입니다. **최저 수치와 검증된 우월성은 구분해야 합니다.** M3가 M1보다 좋아진 비율은 3/6/12/24/36/48h에서 +0.33/+0.83/−0.57/+0.79/−0.24/−0.07%입니다. ‘예측 시간이 길수록 중국 자료의 이득이 커진다’는 패턴은 이번 실행에서 확인되지 않습니다.

### 입력 추가 효과에 대한 통계적 판단

원래 통합 채점의 **84개 검정 가족을 유지**했습니다. 36개 모델–Persistence 비교, 6개 Seasonal naive–Persistence 비교, 7개 입력 대비×6지평인 42개 비교를 한 가족으로 보정했습니다. 개선 판정은 ΔRMSE의 7일 시간 블록 bootstrap 95% 구간 상한<0 **동시에** 제곱오차 DM 검정의 Holm 보정 p<0.05입니다. HAC 지연은 기존 규칙 max(h/3−1,0), bootstrap 1,000회·seed0입니다.

- 모델–Persistence **36개는 모두 유의한 개선**입니다. 기본 예측 모델로서의 성과는 있습니다.
- 입력 조건 간 **42개에서는 유의한 개선 0개, 유의한 악화 0개**입니다. ‘효과가 정확히 0’, ‘서로 동등’하다는 검정은 아닙니다.
- M2는 모든 지평에서 M1보다 RMSE가 높습니다. M3/M2 개선은 성능이 낮아진 M2를 회복하는 것이며, M3/M1의 순이득과 다릅니다.
- M3T는 M3보다 6h RMSE만 낮습니다. M4T는 M4보다 6·24·48h에서 낮지만 일관되지는 않습니다. 궤적 요약 특징의 안정적 이득을 주장하기 어렵습니다.
- 일부 비교의 bootstrap 구간만 0을 제외해도 Holm 기준을 통과하지 못합니다. 유리한 검정만 골라 유의하다고 말하지 않았습니다.

아래는 핵심 비교입니다. ΔRMSE=추가 입력 조건−비교 기준이므로 음수가 개선입니다. 원래 42개 전체 표는 `Results/Tables/test_predeclared_condition_comparisons.csv`에 있습니다.

{md_table(selected)}

## 4. 36·48시간 예측은 어떻게 해석하나

{md_table(long)}

M1은 36·48h에서 Persistence보다 RMSE가 약 25.0%·27.7% 낮습니다. 다만 R²는 각각 0.193·0.146으로 농도 변동의 설명력이 제한적입니다. 긴 지평에서 Persistence가 약해지는 점도 개선율을 키웁니다. ‘48시간 전부터 고농도를 정확하게 경보한다’는 결론은 아직 낼 수 없습니다. 36h Seasonal naive는 t−12h 관측, 48h는 t 관측을 사용하므로 48h Seasonal naive와 Persistence는 같은 기준선입니다.

RMSE와 MAE의 순위도 다릅니다. 36·48h MAE 최저는 M2(8.436·8.721)이며 M1(8.665·9.059)보다 낮지만, M2의 편향은 −3.092·−3.279µg/m³로 과소 예측 경향이 있습니다. ‘MAE가 낮으니 가장 좋은 모델’ 또는 ‘국외 입력이 전혀 무익’으로 단정할 수 없습니다. 평균 오차·큰 오차·편향을 함께 봐야 합니다.

## 5. 학습 설정은 실제로 작동했나

{md_table(epoch)}

공통 설정은 hidden64, encoder/decoder 각 2층, diffusion2, 48시간 입력·48단계 decoder, Adam(lr0.001, eps0.001), 원단위 MAE 손실, gradient clipping5, plateau scheduler(계수0.5, 인내3, 최저1e−5), 최대50epoch·early stopping patience10입니다. 별도 weight decay/dropout은 0입니다. 최적 체크포인트는 **2023의 6h RMSE**로 고릅니다.

학습률은 실제로 여러 번 감소했고, 학습 곡선에 NaN/Inf는 없습니다. 모든 조건이 18–30epoch에서 조기 종료했으므로 이번 결과의 한계가 최대50epoch에 걸린 것은 아닙니다. 단순히 epoch 상한을 늘리는 것이 우선 해결책은 아닙니다.

T 조건은 훈련 손실은 더 낮아졌지만 검증 6h RMSE는 더 높았습니다. 추가 특징의 과적합·결측 및 특징 표현·최적화 문제를 점검할 근거는 되지만, 그 원인을 이번 결과만으로 특정하지는 못합니다. Teacher forcing을 쓰는 훈련 손실과 자유 추론 검증 손실은 같은 조건의 손실도 아닙니다.

## 6. 이전 모델·팀원 결과와 비교

### 이전 DCRNN: 단기 예측은 큰 개선

같은 전체 관측 범위의 이전 F와 새 M4를 고정된 평가 시점·정답·정답 시각이 완전히 같은 2024 행에서 비교했습니다.

{md_table(cross)}

3·6h 개선은 7일 bootstrap과 별도 사후 Holm12 기준을 모두 충족합니다. 검증·시험의 24개 비교를 한 가족으로 묶은 더 보수적 보정에서도 유지됩니다. 반면 12·24h는 유의한 개선이 아닙니다. 과거 대비 학습 설정·문맥 입력 차원·출력 지평 등이 함께 달라져, 개선을 특정 변경 하나의 효과라고 설명할 수는 없습니다.

### 이전 Graph WaveNet: 우월성이 확인되지는 않음

M1끼리·M4끼리 같은 행에서 비교했으며 3·6·12·24h 어느 지평도 이번 사후 보정·구간의 동시 개선 기준을 통과하지 않았습니다. 예를 들어 M1의 새 DCRNN은 5.595/7.502/9.562/10.925, 이전 Graph WaveNet은 5.670/7.258/9.416/10.922입니다. 3h는 조금 좋아지고 6·12h는 대체로 나빠졌습니다. 이전 Graph WaveNet에는 36·48h 결과가 없어 직접 비교하지 않았습니다. 새 H48 Graph WaveNet의 향후 결과와도 버전·평가 행을 맞춰야 합니다.

### 팀원 모델: 우열이 섞임

2023 새 M1 RMSE는 5.961/8.366/10.626/12.136입니다. 팀 B 국내 모델 5.70/8.08/10.79/13.10, 팀 C 전체 GRU 6.19/8.28/10.43/11.82 등과 비교하면 지평별 장단점이 섞입니다. 팀원 표는 반올림 수치만 있고 원시 예측이 없어 통계적 우열을 검정하지 않았습니다. 관측 범위 F600↔F634, 단일 seed↔3seed 앙상블도 다릅니다. 팀 A/SARIMAX의 제한된 외생변수 결과를 전국 국내 기준선 위의 동일한 추가 효과로 읽어서도 안 됩니다.

## 7. RQ와 다음 검토에 주는 의미

**RQ1:** 이번 범위·모델·seed에서 국내 관측 위에 국외 자료를 더한 안정적인 RMSE 이득을 확인하지 못했습니다. 중국 자료의 가치가 지평에 비례해 커진다는 가설도 지지하지 않습니다. 이것은 중국의 실제 배출 기여율이 낮거나, 어떤 조건에서도 중국 관측이 불필요하다는 결론이 아닙니다. 국내 관측과 정보가 겹치는지, 모델이 국외 정보를 활용하지 못하는지, 품질·가용성·통계적 검정력 문제인지도 이 결과만으로 분리할 수 없습니다.

**RQ2:** 평균 효과가 없으면 조건부 효과를 살펴볼 이유는 있지만, 유리한 계절을 결과를 보고 고른 뒤 확인된 사실처럼 발표하면 안 됩니다. 이번 기술적 탐색에서 M3/M1의 가을 24·36·48h 개선 방향은 2023·2024에 공통이었지만, 겨울 효과는 두 해에 일관되지 않았습니다. 서쪽 경로에서 T 특징이 이득을 준다는 근거도 부족합니다. 별도 계절 유의 검정은 수행하지 않았으며, 국지 경로·정온은 표본이 작아 큰 개선율을 해석하지 않는 편이 타당합니다.

추가 작업의 우선순위는 다음과 같습니다. 이번 검토에서는 새 학습을 실행하지 않았습니다.

1. 비교할 주 지표와 지평을 먼저 고정합니다. 36·48h를 중심 질문으로 삼으려면 현재의 ‘검증6h 선택’과 별개로 검증24h 또는 다중 지평 선택을 명시한 실험이 필요합니다. 2024 점수를 보고 선택 규칙을 정하면 사후 개발입니다.
2. 예산이 허용되면 모든 조건을 반복하기보다 M1·M3 등 핵심 조건만 추가 seed로 확인합니다. 지금의 시간 블록 검정은 초기화의 불확실성을 측정하지 않습니다.
3. 계절 모듈·수송 레짐은 2021–22 또는 사전에 정한 물리 규칙으로 구성하고, 2023에서 설정을 선택한 뒤 별도 새 기간/시간순 외부 검증으로 확인합니다. 이미 여러 번 본 2024를 다시 ‘최초의 블라인드 시험’이라고 부르지 않습니다.
4. T 특징은 모델을 더 깊게 만들기보다 훈련 구간의 결측·가용성·요약 방식, 문맥 채널 정규화와 규제를 먼저 확인합니다. 추가 개선의 원인은 같은 구조·같은 입력의 통제 비교로만 분리할 수 있습니다.

## 8. 공유할 내용과 한계

발표의 핵심 메시지는 **‘DCRNN의 단기 예측 자체는 개선했지만, 전국 국내 관측 기준선 위에 국외 관측·궤적 요약 특징을 더한 이득은 작고 일관되지 않았다’**입니다. 실패한 구현으로 보기보다는 **현재 시스템에서 추가 자료의 한계 가치를 검증한 결과**로 설명하는 것이 맞습니다. 다만 단일 seed·한 시험 연도이고 동등성 검정도 아니므로 강한 일반화는 보류해야 합니다.

`README.md`와 `Results/Figures`, 핵심 수치 표를 팀에 공유하면 됩니다. 모델 가중치까지 포함한 원본 ZIP은 이번 검토 폴더에 중복 복사하지 않았습니다. 학습 재개에는 원본 체크포인트가 필요하므로 이 보고서만 남기고 원본을 지우면 안 됩니다.

원본 ZIP: `{archive['archive']}`  
SHA256: `{archive['archive_sha256']}`  
원본 보존·Dataset/모델/채점기 수정 없음. 검토 완료 시각: {datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}
'''
    (output/'README.md').write_text(report, encoding='utf-8')
    files = []
    for p in sorted(output.rglob('*')):
        if p.is_file():
            files.append({'path':p.relative_to(output).as_posix(), 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    manifest = {'source_archive':archive['archive'], 'source_archive_sha256':archive['archive_sha256'], 'batch_id':BATCH.name, 'weights_deserialized':False, 'new_training_or_inference':False, 'verification_audit_present':bool(verified), 'files':files}
    (output/'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'report':str(output/'README.md'), 'file_count':len(files)+1, 'bytes':sum(x['bytes'] for x in files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
