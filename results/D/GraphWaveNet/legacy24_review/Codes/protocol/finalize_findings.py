from pathlib import Path
import json,csv

HERE=Path(__file__).resolve().parent
r=json.loads((HERE/'protocol_audit.json').read_text(encoding='utf-8'))
rows=list(csv.DictReader((HERE/'training_protocol_summary.csv').open(encoding='utf-8-sig')))
result={
 'final_run':'20261111_014538_406520_eaa72758',
 'completion':{'all_six_conditions_passed':True,'final_test':True,'combined_validation_and_test_present':True,'batch_elapsed_seconds':10186.957554196008,'single_seed':11},
 'other_batches':{'20261110_161130_265396_df7c07df':'pilot, M1 only, passed','20261110_162311_230659_9ba7975a':'pilot, all six, passed','20261110_192802_468013_928db0a5':'earlier full interrupted; backed-up manifest remains running','20261111_010911_738330_5b7fa64b':'earlier full resumed/interrupted; backed-up manifest remains running'},
 'share_guidance':{'drive_folder':'MyDrive/Colab_Notebooks/TSA/Result/GraphWaveNet/VastAI/20261111_014538_406520_eaa72758',
                  'latest_complete_folder_sufficient_for_results_and_self_contained_checkpoint_copies':True,
                  'both_attached_zip_parts_required_for_entire_original_folder':'ZIP002 holds five state.pt files belonging to the same final folder. It is not a separate newer experiment. ZIP001 contains all metrics/predictions, but incomplete checkpoints without ZIP002.',
                  'date_name_alone_insufficient':'Confirm batch_manifest.status==passed, mode==full and final_test==true; a newly interrupted batch may have a newer name.'},
 'fixed_protocol':{'nodes':635,'external_universe':634,'KR':514,'CN':77,'JP':42,'MN':1,
    'horizons':[3,6,12,24],'primary_validation_selection':6,'seed':11,'model_parameters':339832,
    'same_distance_graph_all_arms':True,'same_node_order_all_arms':True,'observations_masked_before_normalization':True,
    'raw_base_context':53,'normalized_base_context_with_mask':106,
    'raw_trajectory_context_slots':322,'raw_context_slots':375,'transformed_context_dim':750,
    'excluded_nodes_retained_as_latent_relays':True,
    'all_arms_same_model_size_context_padding':True,
    'M3T_country_scope':['KR','CN'],'M3T_allowed_trajectory_features':218,'M4T_allowed_trajectory_features':322,
    'M3T_no_JP_MN_or_all_country_observation_aggregates':True},
 'architecture':{'profile':'paper topology with project context/output adaptation, not exact benchmark reproduction',
    'blocks':4,'layers_per_block':2,'residual_channels':32,'dilations':[1,2,1,2,1,2,1,2],
    'loaded_history_hours':48,'PM_convolution_eval_receptive_field_hours':13,
    'rf_caveat':'Last-time PM readout uses trailing13h in eval; training BatchNorm uses historical positions across the supplied48h. Trajectory summaries in context can describe older than13h. A24h forecast is still valid.',
    'adaptive_graph':'learned static dense support shared across dates; not meteorology-conditioned time-varying adjacency',
    'T_meaning':'trajectory summaries added to global context; not switching fixed distance graph to HYSPLIT transport graph'},
 'training':r['config']['training'],
 'stopping_policy_caveat':{'M1':'resumed old100/20 run at40 completed epochs, best29, bad11; prospective50/10 reduction retained history and immediately exported previous best. Not a fresh50/10 rerun. Model checkpoint still selected only by2023.',
                          'M2':'reached max50, best42, bad8; budget limit before patience10, so not evidence of full convergence'},
 'training_stability':{'no_nonfinite_training_or_validation_values':True,'largest_validation_6h_rmse':max(float(x['max_val6_rmse']) for x in rows),
    'all_best_checkpoints_within_recorded_history':True,
    'T_arms_best_early':'M3T best5/15; M4T best4/14, training loss continues down, validation worsens. Consistent with overfitting/optimization mismatch, not proof trajectory useless.'},
 'limitations':[
    'Single seed cannot quantify seed-to-seed stability; temporal block CIs only quantify sample error conditional on fitted model.',
    '2024 was already inspected in earlier experiments; do not relabel it untouched independent new test.',
    'DCRNN comparison also changes temporal architecture, graph support, optimization and head/context slots; not isolated adaptive-graph or DCRNN-vs-GWN architecture causality.',
    'Country PM observational information value is not emission-source attribution or material transport fraction.',
    'No fixed-only/adaptive-only/static-vs-dynamic graph ablation here, so adaptive graph benefit itself is not identified.',
    'Historical meteorological analyses/trajectory and final QC observations do not prove real-time data availability; operational latency not independently audited here.',
    'run_summary.test_scored remains false even when condition completion.test_scored and Combined test prove test scoring. Use completion + final scoring for completion judgement.',
    'Session seconds exclude earlier M1 source training. learning_curve.seconds is per-epoch, not cumulative elapsed.'
 ]
}
(HERE/'protocol_findings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
md='''# Graph WaveNet 실행·프로토콜 검토

최종 완료 실행은 `20261111_014538_406520_eaa72758`입니다. full, seed11, M1/M2/M3/M4/M3T/M4T가 모두 passed이고 2023 검증·2024 시험 및 통합 공통 채점이 들어 있습니다. 이 폴더 전체면 결과 검토에 충분합니다. 최신 날짜라는 이유만으로 고르면 안 되고 batch_manifest의 passed/full/final_test를 확인해야 합니다.

ZIP002는 새 실행 결과가 아니라 같은 최종 실행의 state.pt 다섯 개를 따로 담은 Google Drive 분할 다운로드입니다. ZIP001에는 성능표·예측값이 있지만 일부 체크포인트가 빠져 있습니다. 완전한 백업·재개용은 두 ZIP 모두 필요합니다. 앞의 2개 pilot과 중단된 full 2개는 최종 결과 비교에 필요하지 않습니다.

635노드와 고정 거리 그래프, 모델 크기339,832개, 공통 문맥과 평가 규칙을 유지했습니다. 국내514→가까운 중국21추가→중국77전체→일본42·몽골1추가이며 M3T는 한국·중국 경로 특징218개만 허용했습니다. 일본·몽골 관측 또는 all-country 농도 집계 혼입은 계약 목록에 없습니다. M4T는322개 경로 후보를 사용합니다. 제외 관측은 정규화 전 마스킹하고 노드는 잠재 중계 노드로 유지합니다.

모든 조건의 context_dim은750입니다. 원래 ASOS/달력53개+경로 슬롯322개, 그리고 각 결측 표시를 합친 것입니다. T가 없는 조건의322개 경로 슬롯은 상수 결측 상태입니다. 이전 DCRNN regional run은 context_dim106(기본53개+결측표시53개)이므로20개가 아닙니다.

현재 GWN은 거리+학습형 그래프이며, T실험은 그래프를 바꾸는 것이 아니라 출력 문맥에 경로 요약을 더하는 실험입니다. 학습형 인접행렬은 시점별 바람으로 달라지는 동적 그래프가 아니라 학습 후 모든 시점에 공통인 정적 행렬입니다. 학습된 연결 강도가 국가 배출 기여도나 물리적 수송량은 아닙니다.

48시간 자료를 로더에 넣지만 paper 구조의 dilation은1,2를4번 반복해 마지막 PM 출력의 추론시 수용범위는13시간입니다. 훈련 BatchNorm은48시간 창의 위치를 함께 보고, 경로 문맥에는 더 오래된 요약이 들어갈 수 있습니다. 따라서24시간 예측은 가능하지만48시간 PM이 모두 직접 반영되는 모델로 설명하면 부정확합니다.

최대50epoch/patience10이 적용됐습니다. M1은 이전100/20실행에서40epoch까지 진행한 상태(최적29, bad11)를 계승해 즉시 이전 최적 가중치로 결과를 냈습니다. 새50/10조건으로 처음부터 재학습한 것은 아닙니다. M2는50epoch 한도에서 종료(최적42,bad8)했고, M3=35/최적25, M4=31/최적21, M3T=15/최적5, M4T=14/최적4입니다.

모든 학습 곡선은 유한했고 최대 검증6hRMSE가9.656미만이어서 이전 DCRNN의 폭주와 같은 수치 불안정은 관찰되지 않았습니다. T조건은 일찍 최적점을 지나 훈련오차는 내려가지만 검증오차가 악화됩니다. 이는 이 결합 방식의 과적합·최적화·모델선택 문제와 부합하며, 역궤적 자체가 무가치하다는 결론은 아닙니다. 모든 지평은 공통적으로6시간 검증RMSE로 고른 checkpoint에서 평가했습니다.

한 시드의 블록bootstrap/DM검정은 시점 표본의 불확실성을 다루지만 초기화에 따른 학습 불확실성을 다루지는 않습니다. 2024는 이전에도 열람했으므로 새롭게 잠긴 미사용 시험구간으로 표현하지 않습니다. 운영 지연시간은 이번 결과만으로 검증되지 않습니다.

`run_summary.test_scored=false`는 갱신되지 않은 필드이며, completion과 실제 Combined/test가 시험 수행을 확인해 줍니다. 최신 배치2시간49분47초는 이전M1학습 시간을 포함하지 않습니다. learning_curve.seconds는 매epoch 시간입니다.
'''
(HERE/'protocol_findings_ko.md').write_text(md,encoding='utf-8')
print('saved',HERE/'protocol_findings.json')
