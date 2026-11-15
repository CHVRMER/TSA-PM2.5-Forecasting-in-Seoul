"""Read-only metadata/integrity audit; never deserialize archived Torch weights."""
from pathlib import Path
import collections
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

WORK = Path(__file__).resolve().parent
ZIP = Path('C:/Users/kimjy/Downloads/20261112_003040_429483_85af4013-20261112T020128Z-1-001.zip')
BATCH = '20261112_003040_429483_85af4013'
ROOT = WORK / 'extracted' / BATCH
CONDITIONS = ['M1', 'M2', 'M3', 'M4', 'M3T', 'M4T']

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def zip_hash(archive, name, cache):
    if name not in cache:
        with archive.open(name) as handle:
            cache[name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    return cache[name]

def canonical(value):
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())

def relative_zip_name(path):
    return BATCH + '/' + path.relative_to(ROOT).as_posix()

def target_and_base(prep):
    return dict(target=[prep[key][0] for key in ['pm_mean', 'pm_std', 'pm_valid_count']],
                base={key:prep[key][:53] for key in ['context_mean', 'context_std', 'context_valid_count', 'context_names']},
                node_order=prep['node_order'], fit_start=prep['fit_start_inclusive'], fit_end=prep['fit_end_exclusive'])

def main():
    manifest=read(ROOT/'batch_manifest.json')
    config=read(ROOT/'resolved_configuration.json')
    source=read(ROOT/'source_contract.json')
    rows=[]
    detail={}
    hashed={}
    checkpoint_checks=[]
    with zipfile.ZipFile(ZIP) as archive:
        member_names=set(archive.namelist())
        for condition in CONDITIONS:
            run_dirs=[p for p in (ROOT/'Conditions'/condition).iterdir() if p.is_dir()]
            assert len(run_dirs)==1
            run=run_dirs[0]
            model=next(run.glob('Models/**/run_summary.json')).parent
            summary=read(model/'run_summary.json')
            curve=pd.read_csv(model/'learning_curve.csv')
            contract=read(model/'contract.json')
            prep=read(run/'preprocessing.json')
            cc=read(ROOT/'Condition_Contracts'/f'{condition}.json')
            numeric=curve.select_dtypes(include='number')
            finite=bool(np.isfinite(numeric.to_numpy()).all())
            reductions=curve.loc[curve.learning_rate_next < curve.learning_rate_used, 'epoch'].astype(int).tolist()
            graph=np.load(run/'Graphs/trajectory_directed.npy', allow_pickle=False)
            allowed=set(cc['allowed_station_keys'])
            excluded=set(cc['excluded_station_keys'])
            nodes=prep['node_order']
            excluded_positions=[i for i,key in enumerate(nodes) if key in excluded]
            allowed_countries=collections.Counter(k.split(':')[0] for k in allowed)
            best_path=model/'best.pt'
            best_name=relative_zip_name(best_path)
            best_hash=zip_hash(archive,best_name,hashed)
            best_ok=best_hash==summary['checkpoint_sha256']
            check_items=[]
            for mp in sorted((model/'Checkpoints').glob('*/manifest.json')):
                checkpoint=read(mp)
                commit=read(mp.parent/'COMMITTED.json')
                candidates=[name for name in member_names if name.startswith(relative_zip_name(mp.parent)+'/') and name.endswith('.pt')]
                assert len(candidates)==1, candidates
                state=candidates[0]
                item=dict(condition=condition,generation=checkpoint['generation'],completed_epochs=checkpoint['completed_epochs'],
                          commit_matches_manifest=sha_bytes(mp.read_bytes())==commit['manifest_sha256'],
                          state_hash_matches=zip_hash(archive,state,hashed)==checkpoint['state_sha256'],
                          state_size_matches=archive.getinfo(state).file_size==checkpoint['state_bytes'])
                check_items.append(item)
                checkpoint_checks.append(item)
            row=dict(condition=condition,status=read(run/'completion.json').get('status'),seed=summary['seed'],
                     best_epoch=summary['best_epoch'],trained_epochs=summary['trained_epochs'],max_epochs=contract['training']['max_epochs'],
                     patience=contract['training']['patience'],early_stopped=len(curve)<contract['training']['max_epochs'],
                     best_validation_rmse_h6=summary['validation_rmse_h6'],last_validation_rmse_h6=float(curve.validation_rmse_h6.iloc[-1]),
                     first_train_mae=float(curve.train_loss.iloc[0]),last_train_mae=float(curve.train_loss.iloc[-1]),
                     minimum_train_mae=float(curve.train_loss.min()),maximum_validation_rmse_h6=float(curve.validation_rmse_h6.max()),
                     learning_rate_initial=float(curve.learning_rate_used.iloc[0]),learning_rate_final=float(curve.learning_rate_next.iloc[-1]),
                     learning_rate_reduction_epochs=','.join(map(str,reductions)),
                     teacher_forcing_initial=float(curve.teacher_forcing_probability.iloc[0]),
                     teacher_forcing_final=float(curve.teacher_forcing_probability.iloc[-1]),
                     global_steps=int(curve.global_step.iloc[-1]),mean_epoch_seconds=float(curve.seconds.mean()),
                     training_seconds=summary['session_training_seconds'],parameter_count=summary['parameter_count'],
                     external_series=cc['allowed_external_series'],trajectory_enabled=cc['trajectory_enabled'],
                     trajectory_features_allowed=len(cc['trajectory_allowed_names']),base_context_size=cc['base_context_size'],
                     raw_context_size=cc['raw_context_size'],transformed_context_size=cc['transformed_context_size'],
                     finite_learning_curve=finite,best_checkpoint_hash_matches=best_ok,
                     test_scored=summary['test_scored'],validation_scored=summary['validation_scored'])
            rows.append(row)
            detail[condition]=dict(run=str(run.relative_to(ROOT)),architecture=summary['architecture'],training=contract['training'],
                 data_signature=summary['data_signature'],graph_signature=summary['graph_signature'],
                 target_base_preprocessing_signature=canonical(target_and_base(prep)),
                 node_order_signature=cc['node_order_sha256'],fixed_graph_sha256=cc['fixed_graph_sha256'],
                 graph_shape=list(graph.shape),graph_nonzero=int(np.count_nonzero(graph)),graph_finite=bool(np.isfinite(graph).all()),
                 graph_file_sha256=sha_bytes((run/'Graphs/trajectory_directed.npy').read_bytes()),
                 allowed_external_countries=dict(allowed_countries),allowed_and_excluded_disjoint=not bool(allowed & excluded),
                 union_contains_all_external_nodes=(allowed|excluded)==(set(nodes)-{'KR:111121'}),
                 excluded_preprocessing_counts_zero=all(prep['pm_valid_count'][i]==0 for i in excluded_positions),
                 mask_applied_before_normalization=cc['mask_applied_before_normalization'],
                 targets_and_common_context_preserved=cc['targets_and_common_context_preserved'],
                 trajectory_countries=cc['trajectory_countries'],trajectory_allowed_names=cc['trajectory_allowed_names'],
                 preprocessing_train_start=prep['fit_start_inclusive'],preprocessing_train_end=prep['fit_end_exclusive'],
                 target_mean=prep['pm_mean'][0],target_std=prep['pm_std'][0],target_training_valid_count=prep['pm_valid_count'][0],
                 best_weights_sha256=best_hash,checkpoint_recovery=summary['checkpoint_recovery'],checkpoints=check_items,
                 best_epoch_matches_curve=int(curve.loc[curve.validation_selection_score.idxmin(),'epoch'])==summary['best_epoch'],
                 patience_accounting_matches=len(curve)-summary['best_epoch']==contract['training']['patience'])

        receipts=[]
        for p in sorted((ROOT/'Backup_receipts').rglob('*.json')):
            r=read(p)
            receipts.append(dict(path=str(p.relative_to(ROOT)),reason=r.get('reason'),main_copy_verified=r.get('main_copy_verified'),
                                 timestamp_utc=r.get('timestamp_utc'),file_count=len(r.get('files',[]))))
        final_paths=[p for p in (ROOT/'Backup_receipts').rglob('verified.json') if read(p).get('reason')=='batch_complete']
        assert len(final_paths)==1
        final=read(final_paths[0])
        publication_mismatches=[]
        for f in final['files']:
            name=BATCH+'/'+f['path']
            if name not in member_names:
                publication_mismatches.append(dict(path=f['path'],reason='missing'))
            elif archive.getinfo(name).file_size!=f['bytes'] or zip_hash(archive,name,hashed)!=f['sha256']:
                publication_mismatches.append(dict(path=f['path'],reason='size_or_hash_differs'))

    result=dict(batch=dict(status=manifest['status'],mode=manifest['mode'],selected_conditions=manifest['selected_conditions'],
               final_test=manifest['final_test'],resume_source=manifest.get('resume_source'),failed_conditions=manifest['failed_conditions'],
               started_kst=manifest['started_kst'],elapsed_seconds=manifest['elapsed_seconds']),
               config=config,training_protocol=rows,conditions=detail,
               consistency=dict(target_and_base_preprocessing_equal=len({d['target_base_preprocessing_signature'] for d in detail.values()})==1,
                   node_order_equal=len({d['node_order_signature'] for d in detail.values()})==1,
                   graph_files_identical=len({d['graph_file_sha256'] for d in detail.values()})==1,
                   graph_signatures_equal=len({d['graph_signature'] for d in detail.values()})==1,
                   parameter_counts_equal=len({r['parameter_count'] for r in rows})==1,
                   shared_horizons=source['horizons'],label_extension_contract=source['horizon_extension_contract']),
               checkpoint_integrity=dict(count=len(checkpoint_checks),all_commit_hashes_match=all(r['commit_matches_manifest'] for r in checkpoint_checks),
                   all_state_hashes_match=all(r['state_hash_matches'] for r in checkpoint_checks),all_sizes_match=all(r['state_size_matches'] for r in checkpoint_checks),
                   all_best_weight_hashes_match=all(r['best_checkpoint_hash_matches'] for r in rows),torch_deserialization_used=False),
               backups=dict(preflight=read(ROOT/'backup_preflight.json'),receipt_count=len(receipts),
                   all_receipts_main_copy_verified=all(r['main_copy_verified'] is True for r in receipts),receipts=receipts,
                   final_publication_manifest_files=len(final['files']),final_publication_mismatches=publication_mismatches,
                   final_verified_timestamp=final['timestamp_utc'],live_remote_not_queried=True),
               limitations=['Seed11 only: temporal tests do not assess initialization uncertainty.',
                 'All horizons selected by validation6h RMSE, not by long-horizon RMSE.',
                 'Previously inspected2024 test is reused: treat new model tuning comparisons as retrospective.',
                 'Non-T arms retain same trajectory-derived directed graph and latent relays: these are raw-PM-source ablations, not complete no-foreign/no-trajectory systems.',
                 'M3T/M4T add trajectory context to same architecture, but model may overfit; lower trainMAE is not validation gain.',
                 'Archived metadata and hashes verified; no checkpoint tensors deserialized or liveGoogleDrive queried.'])
    (WORK/'training_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(rows).to_csv(WORK/'Tables/training_protocol_summary.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(dict(training=rows,consistency=result['consistency'],checkpoint_integrity=result['checkpoint_integrity'],
                         backup_summary={k:v for k,v in result['backups'].items() if k not in ['receipts','preflight']}),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
