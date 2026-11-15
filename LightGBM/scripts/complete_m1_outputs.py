"""기존 M1 검증 예측의 공통 출력 보완. 모델 학습·추론·2024 시험은 하지 않는다."""
import json
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import pandas as pd
from src.common_scoring import export_predictions,sha,write_json,evaluate

def main():
    source=ROOT/'results/M1_h6_20261108_174529_127972'
    models=ROOT/'Models/M1_h6_20261108_174529_127972'
    files=[*source.iterdir(),*models.iterdir()]
    snapshots={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()}
    info=json.loads((source/'metrics_and_selection.json').read_text())
    config=json.loads((source/'config.json').read_text())
    if info['eligible_train_rows']!=5761 or info['eligible_validation_rows']!=2530:raise ValueError('학습 이력 불일치')
    text=(ROOT/'src/train.py').read_text()
    if 'for split in ("train", "validation")' not in text or 'label=np.log1p(ytr)' not in text:raise ValueError('학습 코드 수동 검토 필요')
    d=pd.read_parquet(source/'predictions_M1_h6_validation.parquet')
    if not d.origin_time_kst.dt.year.eq(2023).all():raise ValueError('검증 예측 외 시점 발견')
    # 원본의 경계 정답 값은 읽었더라도 공통 예측에는 정답 열 자체를 내보내지 않는다.
    boundary=d.target_time_kst.dt.year.eq(2024)
    if d.loc[boundary,'eligible'].any():raise ValueError('2024 경계 정답 평가에 혼입')
    output=d.copy();output['y_hat']=d.y_hat_raw
    metadata={'seeds':config['seeds'],'settings':config,'selection':info['selected'],
        'training_years':[2021,2022],'selection_year':2023,'no_refit_on_2023':True,
        'training_evidence':{'code_sha256':json.loads((source/'code_sha256.json').read_text()),'log_sha256':sha(source/'run.log'),'eligible_train_rows':5761},
        'trained_models_sha256':{str(p.relative_to(ROOT)):sha(p) for p in models.glob('M1_h6_seed*.txt')},
        'source_prediction_sha256':sha(source/'predictions_M1_h6_validation.parquet'),
        'validation_only':True,'retraining_performed':False,'model_prediction_performed':False}
    export_predictions(output,ROOT/'Models/predictions/M1.parquet',metadata)
    out,s,t=evaluate('validation',['M1'],[6])
    changed=[rel for rel,h in snapshots.items() if sha(ROOT/rel)!=h]
    if changed:raise ValueError('기존 모델 또는 원본 결과 변경: '+str(changed))
    write_json(out/'completion_audit.json',{'status':'passed','existing_artifacts_unchanged':len(snapshots),'artifact_sha256':snapshots,
        'training_years':[2021,2022],'selection_year':2023,'refit_on_2023':False,'retraining_performed':False,
        'test_2024_performance_previously_found':False,'test_2024_performed_this_time':False,
        'test_decision':'재학습·시험 평가 이력 없음. 비적격 경계 정답 2개 노출은 시험 수행이 아니므로 1회 시험을 소비하지 않음.',
        'boundary_rows_in_original_prediction':int(boundary.sum()),'boundary_rows_eligible':0,
        'canonical_prediction':str(ROOT/'Models/predictions/M1.parquet')})
    print(s.to_string(index=False));print(t.to_string(index=False));print('보완 결과:',out)
if __name__=='__main__':main()
