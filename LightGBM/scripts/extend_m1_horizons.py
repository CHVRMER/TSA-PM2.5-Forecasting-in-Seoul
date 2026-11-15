"""M1 3/12/24시간만 추가 학습하고 기존 6시간과 함께 검증한다. Test는 실행하지 않는다."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'results/.matplotlib_cache'))

import numpy as np
import pandas as pd
from src.train import train_m1, sha, write_json
from src.common_scoring import export_predictions, evaluate, plot_skill


def check_protected(preflight):
    changed = [p for p, digest in preflight['protected_sha256'].items() if sha(ROOT/p) != digest]
    if changed:
        raise ValueError('기존 6시간 모델·결과 변경: '+str(changed))
    if (ROOT/'results/final_test_2024.lock').exists() != preflight['test_lock_existed']:
        raise ValueError('실제 시험 잠금 상태가 달라졌습니다.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    preflight = json.loads((batch/'preflight.json').read_text())
    if preflight['status'] != 'no_prior_M1_h3_h12_h24_training_found':
        raise ValueError('기존 실행 확인 필요')
    check_protected(preflight)
    for p in (ROOT/'results').rglob('metrics_and_selection.json'):
        record = json.loads(p.read_text())
        if record.get('model') == 'M1' and record.get('horizon') in (3, 12, 24):
            raise FileExistsError('이미 수행된 M1 추가 Horizon 발견. 재학습 없이 중단합니다: '+str(p))
    base = json.loads((ROOT/'configs/m1_6h.json').read_text())
    expected_counts = {3: (5762, 2531), 12: (5759, 2528), 24: (5755, 2524)}
    configs = {}
    for h, counts in expected_counts.items():
        c = json.loads((ROOT/f'configs/m1_{h}h.json').read_text())
        ignore = {'horizon', 'expected_train_rows', 'expected_validation_rows'}
        if {k:v for k,v in c.items() if k not in ignore} != {k:v for k,v in base.items() if k not in ignore}:
            raise ValueError('기존 6시간과 학습 절차·설정이 다릅니다.')
        if (c['expected_train_rows'],c['expected_validation_rows']) != counts:
            raise ValueError('Horizon별 적격 표본 설정 불일치')
        configs[h] = c
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(batch/'run.log', encoding='utf-8'), logging.StreamHandler()])
    write_json(batch/'scope_before_training.json', {
        'train_horizons':[3,12,24], 'reuse_only_horizons':[6], 'model':'M1',
        'train_years':[2021,2022], 'validation_year':2023, 'test_enabled':False,
        'holm_family':['3:M1:persistence','6:M1:persistence','12:M1:persistence','24:M1:persistence'],
        'improvement_rule':'RMSE 차이 95% CI 상한 < 0 AND Holm 보정 p < 0.05',
        'configs':configs})
    outputs = {}
    try:
        for h,c in configs.items():
            result_dir = batch/f'h{h}'
            model_dir = ROOT/'Models'/batch.name/f'h{h}'
            result_dir.mkdir(exist_ok=False)
            model_dir.mkdir(parents=True, exist_ok=False)
            handler = logging.FileHandler(result_dir/'run.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logging.getLogger().addHandler(handler)
            try:
                logging.info('M1 %s시간 시작. 2024 Test 비활성.', h)
                info = train_m1(ROOT,c,result_dir,model_dir)
            finally:
                logging.getLogger().removeHandler(handler)
                handler.close()
            outputs[h] = {'results':str(result_dir.relative_to(ROOT)), 'models':str(model_dir.relative_to(ROOT)),
                          'eligible_train':info['eligible_train_rows'],'eligible_validation':info['eligible_validation_rows']}
            check_protected(preflight)
        # 기존 6시간 공통 예측은 복사·결합만 한다. 모델 재학습·재추론은 하지 않는다.
        frames = []; metadata = {}
        for h in [3,6,12,24]:
            path = ROOT/'Models/predictions/M1.parquet' if h == 6 else batch/f'h{h}/M1.parquet'
            frame = pd.read_parquet(path)
            if not frame.horizon.eq(h).all() or not frame.split.eq('validation').all():
                raise ValueError('Horizon별 예측 파일 불일치')
            frames.append(frame)
            metadata[str(h)] = {'prediction_path':str(path.relative_to(ROOT)),
                               'prediction_sha256':sha(path), 'metadata':json.loads(path.with_suffix('.meta.json').read_text()),
                               'reused_without_training_or_prediction':h == 6}
        combined = pd.concat(frames, ignore_index=True).sort_values(['horizon','origin_time_kst'])
        combined_dir = batch/'common_predictions';combined_dir.mkdir(exist_ok=False)
        export_predictions(combined,combined_dir/'M1.parquet',{
            'seeds':[0,1,2], 'horizons':[3,6,12,24], 'training_years':[2021,2022],
            'selection_year':2023, 'no_refit_on_2023':True, 'validation_only':True,
            'by_horizon':metadata})
        score_dir, scores, tests = evaluate('validation',['M1'],[3,6,12,24],
                                           output_root=batch/'common_scores',prediction_dir=combined_dir)
        table = scores[scores.model.eq('M1')].merge(tests,on=['horizon','model','n'],validate='one_to_one')
        table.to_csv(batch/'M1_2023_validation_horizon_summary.csv',index=False)
        # 개별 결과의 설정·표본 수도 한 표에 기록한다.
        selected=[]
        for h in [3,6,12,24]:
            original=ROOT/'results/M1_h6_20261108_174529_127972' if h==6 else batch/f'h{h}'
            info=json.loads((original/'metrics_and_selection.json').read_text())
            selected.append({'horizon':h,'eligible_train_rows':info['eligible_train_rows'],
                             'eligible_validation_rows':info['eligible_validation_rows'],
                             'selected':info['selected'],'all_eligible_validation_metrics':info['metrics']})
        write_json(batch/'horizon_settings_and_counts.json',selected)
        plot_skill(scores,tests,batch/'M1_2023_validation_intermediate_rmse_improvement_3_6_12_24.png',split='validation')
        # 저장된 예측·모델·지표의 정합성을 실제 데이터로 확인한다.
        for h in [3,12,24]:
            detail=pd.read_parquet(batch/f'h{h}/predictions_M1_h{h}_validation.parquet')
            info=json.loads((batch/f'h{h}/metrics_and_selection.json').read_text())
            if int(detail.eligible.sum())!=expected_counts[h][1]:raise ValueError('적격 행 수 불일치')
            if not np.allclose(detail.y_hat_raw,detail[[f'y_hat_seed{s}' for s in [0,1,2]]].mean(axis=1)):
                raise ValueError('앙상블 평균 불일치')
            if detail.loc[detail.target_time_kst.dt.year.eq(2024),'y'].notna().any():
                raise ValueError('검증 경계의 시험 정답을 읽었습니다.')
            if not all(x['max_absolute_prediction_difference']==0 for x in info['model_reload_checks']):
                raise ValueError('저장 모델 재로드 예측 불일치')
            error=detail.loc[detail.eligible,'y_hat']-detail.loc[detail.eligible,'y']
            if not np.isclose(np.sqrt(np.mean(error**2)),info['metrics']['RMSE']):raise ValueError('RMSE 재계산 불일치')
        if len(tests)!=4 or list(tests.horizon)!=[3,6,12,24]:raise ValueError('공통 검정 family 누락')
        row_file=score_dir/'score_rows_validation.parquet';rows=pd.read_parquet(row_file)
        if not rows.origin_time_kst.dt.year.eq(2023).all() or not rows.target_time_kst.dt.year.eq(2023).all():
            raise ValueError('공통 평가에 2024 자료가 포함되었습니다.')
        from src.common_scoring import holm
        if not np.allclose(tests.p_holm,holm(tests.p)):raise ValueError('Holm 보정 불일치')
        if not np.array_equal(tests.improved,(tests.ci95_high<0)&(tests.p_holm<.05)):
            raise ValueError('공식 개선 판정 불일치')
        check_protected(preflight)
        paths = [ROOT/'src/train.py',ROOT/'src/evaluate.py',ROOT/'src/common_scoring.py',Path(__file__)]
        paths += [ROOT/f'configs/m1_{h}h.json' for h in [3,6,12,24]]
        write_json(batch/'completion.json',{'status':'completed','model':'M1','trained_horizons':[3,12,24],
            'reused_horizons':[6],'outputs':outputs,'common_scores':str(score_dir.relative_to(ROOT)),
            'seeds':[0,1,2],'existing_h6_artifacts_unchanged':len(preflight['protected_sha256']),
            'all_new_model_reload_checks_max_difference':0,'eligible_masks_verified':True,
            'common_scoring_holm_family_size':4,'test_2024_executed':False,'M2_or_later_executed':False,
            'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}})
        logging.info('M1 전 Horizon 검증 완료. 6시간 보존, Test 미실행.')
        print(table.to_string(index=False),flush=True)
        print('결과:',batch,flush=True)
    except Exception:
        logging.exception('오류 발생. 추가 작업 없이 중단합니다.')
        raise


if __name__=='__main__':main()
