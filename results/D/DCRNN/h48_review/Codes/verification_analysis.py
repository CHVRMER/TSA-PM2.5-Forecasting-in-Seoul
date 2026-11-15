"""Independently rebuild exact H48 labels, then reproduce the complete84-test family."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd

WORK=Path(__file__).resolve().parent
ROOT=WORK/'extracted/20261112_003040_429483_85af4013'
PROJECT=Path('C:/Users/kimjy/Desktop/Time Series Analysis/Project/Revised_Project')
COMMON=PROJECT/'Model/Common/DCRNN_H48_20261111'
TARGET=WORK.parent/'dcrnn_vast48_20261111/verification_data/target_hourly_audit.parquet'
OUT=WORK/'Verification'
CONDS=['M1','M2','M3','M4','M3T','M4T']

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def normalize_strings(frame):
    f=frame.copy()
    for col in f:
        if pd.api.types.is_object_dtype(f[col].dtype) or isinstance(f[col].dtype,pd.StringDtype):
            f[col]=f[col].fillna('').astype(str)
    return f

def compare_frames(expected,actual):
    result={'shape_expected':list(expected.shape),'shape_actual':list(actual.shape),'columns_equal':list(expected.columns)==list(actual.columns)}
    if not result['columns_equal'] or expected.shape!=actual.shape:
        result['passed']=False
        return result
    a,b=normalize_strings(expected),normalize_strings(actual)
    differences=[]
    exact=True
    max_delta=0.
    for col in a:
        x,y=a[col],b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xx=x.to_numpy(dtype=float);yy=y.to_numpy(dtype=float)
            same=np.array_equal(xx,yy,equal_nan=True)
            valid=np.isfinite(xx)&np.isfinite(yy)
            delta=float(np.max(np.abs(xx[valid]-yy[valid]))) if valid.any() else 0.
            max_delta=max(delta,max_delta)
            close=np.allclose(xx,yy,rtol=1e-12,atol=1e-12,equal_nan=True)
        else:
            same=bool(x.equals(y));close=same;delta=None
            if not same:
                # Arrow can change timestamp storage resolution without changing a timestamp.
                try:
                    same=bool(np.array_equal(x.to_numpy(),y.to_numpy(),equal_nan=True));close=same
                except TypeError:
                    pass
        exact=exact and same
        if not same:
            differences.append({'column':col,'max_abs_difference':delta,'within_tolerance':bool(close)})
    result.update(all_values_exact=exact,max_numeric_difference=max_delta,differences=differences,
                  passed=all(d['within_tolerance'] for d in differences))
    return result

def main():
    start=time.monotonic()
    OUT.mkdir(exist_ok=True)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(COMMON))
    hc=module('_independent_horizon_contract',COMMON/'horizon_contract.py')
    scorer=module('_independent_score',COMMON/'score.py')
    grid_api=module('_independent_grid',COMMON/'grid.py')
    base_path=PROJECT/'Dataset/Evaluation_Contract_20261107/evaluation_grid.parquet'
    original_sha=sha(base_path)
    base=pd.read_parquet(base_path)
    target=pd.read_parquet(TARGET)
    bundle={'hourly_index':pd.DatetimeIndex(target.time_kst),'pm_values':target.pm25.to_numpy(float)[:,None]}
    g=hc.build_h48_grid(base,bundle)
    independent_baselines=grid_api.add_baselines(g,target)
    baseline_cols=[c for c in g if c.startswith(('persistence_','seasonal_'))]
    baseline_audit=compare_frames(g[baseline_cols],independent_baselines[baseline_cols])
    hc.assert_original_rows_preserved(base,g)
    g.to_parquet(OUT/'rebuilt_evaluation_grid.parquet',index=False)
    timeline=pd.DatetimeIndex(g.loc[g.horizon.eq(3),'origin_time_kst'])
    grid_audit=dict(rows=len(g),origins=len(timeline),columns=len(g.columns),
                    three_hour_spacing=bool((np.diff(timeline.asi8)==pd.Timedelta(hours=3).value).all()),
                    original_four_horizons_preserved_exactly=True,baselines_independently_recomputed=baseline_audit,
                    source_grid_sha256_before=original_sha,source_grid_sha256_after=sha(base_path),
                    derived_grid_sha256=sha(OUT/'rebuilt_evaluation_grid.parquet'),
                    original_derived_grid_sha256=read(ROOT/'evaluation_grid_contract.json')['sha256'])
    requested={(b,c):int(g.loc[g.split.eq(b)&g.horizon.eq(c),'eligible'].sum())
               for b in ['train','validation','test'] for c in [3,6,12,24,36,48]}
    grid_audit['eligible_rows']={f'{b}/h{h}':n for (b,h),n in requested.items()}
    for field in ['persistence','seasonal']:
        request=g[field+'_request_time_kst'];observed=g[field+'_observation_time_kst']
        grid_audit[field+'_all_requests_at_or_before_origin']=bool(request.le(g.origin_time_kst).all())
        grid_audit[field+'_all_nonmissing_observations_at_or_before_requests']=bool(observed.isna().to_numpy().astype(bool).__or__(observed.le(request).to_numpy()).all())
        grid_audit[field+'_fallbacks_by_split_horizon']={f'{split}/h{h}':int(part[field+'_fallback'].sum())
             for (split,h),part in g.loc[g.eligible].groupby(['split','horizon'])}
        grid_audit[field+'_fallback_kind_counts_eligible']=g.loc[g.eligible,field+'_fallback_kind'].value_counts(dropna=False).to_dict()
    pairs=read(ROOT/'predeclared_comparisons.json')['pairs']
    audits={}
    for split in ['validation','test']:
        print('Re-scoring',split,'6conditions /6horizons /7extra pairs /1000bootstrap',flush=True)
        frames=[]
        input_audits=[]
        for condition in CONDS:
            path=next(ROOT.glob(f'Conditions/{condition}/*/Exports/{split}_predictions_by_seed.parquet'))
            f=pd.read_parquet(path)
            fields=['origin_time_kst','horizon','target_time_kst','y_true','eligible']
            expected=g.loc[g.split.eq(split),fields].reset_index(drop=True)
            actual=f[fields].sort_values(['origin_time_kst','horizon']).reset_index(drop=True)
            expected=expected.sort_values(['origin_time_kst','horizon']).reset_index(drop=True)
            audit=compare_frames(expected,actual)
            input_audits.append(dict(condition=condition,path=str(path),sha256=sha(path),label_grid_match=audit,
                eligible_prediction_nonfinite=int((f.eligible & ~np.isfinite(f.y_hat)).sum()),
                duplicate_keys=int(f.duplicated(['origin_time_kst','horizon']).sum()),rows=len(f)))
            frames.append(f)
        result=scorer.score_frames(g,pd.concat(frames,ignore_index=True),split=split,pairs=pairs,
                                   final=(split=='test'),bootstrap_reps=1000,bootstrap_seed=0,make_ensembles=True)
        assert len(result['tests'])==84
        original=next((ROOT/'Common_Scoring/Combined'/split).iterdir())
        out=OUT/split;out.mkdir(exist_ok=True)
        output_audits={}
        for name in ['metrics','tests','conditions','seed_metric_summary','score_rows']:
            frame=result[name]
            if name=='score_rows':
                archived=pd.read_parquet(original/(name+'.parquet'))
                frame.to_parquet(out/(name+'.parquet'),index=False)
            else:
                archived=pd.read_csv(original/(name+'.csv'),float_precision='round_trip',dtype={'seed':str,'seeds':str,'ensemble_rule':str})
                frame.to_csv(out/(name+'.csv'),index=False)
            output_audits[name]=compare_frames(archived,frame)
        (out/'scoring_audit.json').write_text(json.dumps(result['audit'],indent=2),encoding='utf-8')
        archive_rows=pd.read_parquet(original/'score_rows.parquet')
        distinct=archive_rows[list(g.columns)].drop_duplicates().sort_values(['origin_time_kst','horizon']).reset_index(drop=True)
        expected=g.loc[g.split.eq(split)&g.eligible].sort_values(['origin_time_kst','horizon']).reset_index(drop=True)
        logical_grid=compare_frames(expected,distinct)
        tests=result['tests']
        audit=dict(inputs=input_audits,outputs=output_audits,logical_scored_grid_match=logical_grid,
             complete_84_test_family=len(tests)==84,finite_p_count=int(tests.p.notna().sum()),
             hac_lags_by_horizon={str(h):sorted(p.hac_lag_slots.unique().tolist()) for h,p in tests.groupby('horizon')},
             tests_per_horizon=tests.groupby('horizon').size().to_dict(),extra_comparison_count=len(pairs),
             bootstrap_reps=1000,bootstrap_seed=0,audit=result['audit'])
        audits[split]=audit
        print(split,'checks=',{k:v['passed'] for k,v in output_audits.items()},'logical_grid=',logical_grid['passed'],flush=True)
    passed=(baseline_audit['passed'] and original_sha==sha(base_path)
       and all(all(x['label_grid_match']['passed'] and not x['eligible_prediction_nonfinite'] and not x['duplicate_keys'] for x in s['inputs'])
          and all(v['passed'] for v in s['outputs'].values()) and s['logical_scored_grid_match']['passed'] for s in audits.values()))
    report=dict(passed=passed,grid=grid_audit,splits=audits,code_sha256={p.name:sha(p) for p in [COMMON/'score.py',COMMON/'grid.py',COMMON/'horizon_contract.py']},
               elapsed_seconds=time.monotonic()-start,torch_weight_deserialization=False,source_files_modified=False,
               note='Parquet byte hash can differ with Arrow versions; all grid logical values and original output contents are compared independently.')
    (WORK/'verification_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
    print('PASSED',passed,'seconds',report['elapsed_seconds'],flush=True)

if __name__=='__main__':
    main()
