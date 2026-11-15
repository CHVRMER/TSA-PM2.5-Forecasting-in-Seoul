"""1차 분석 결과의 국가·권역 보고서와 그림. 추가 모델/관측소 분석 없음."""
from pathlib import Path
import sys,os,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common_scoring import sha,write_json
from scripts.run_m2 import protect
OUT=ROOT/'results/shap/validation_2023';MODELS=['M1','M2','M3','M4','M3T','M4T'];GROUPS=['local','KR','CN','JP','MN','trajectory']
LABEL={'local':'서울·ASOS·달력','KR':'국내 이력','CN':'중국 이력','JP':'일본 이력','MN':'몽골 이력','trajectory':'Trajectory 전체'}
def table(df):
 return '| '+' | '.join(map(str,df.columns))+' |\n|'+'|'.join(['---']*len(df.columns))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
def matrix(df,val,h):
 return df[(df.horizon==h)&(df.level=='primary_group')].pivot(index='model',columns='group',values=val).reindex(index=MODELS,columns=GROUPS)
def heatmap(data,title,label,path):
 fig,ax=plt.subplots(figsize=(10,5));a=data.to_numpy(float);limit=np.nanmax(abs(a));im=ax.imshow(np.ma.masked_invalid(a),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto');ax.set_xticks(range(len(data.columns)),[LABEL.get(x,x) for x in data.columns]);ax.set_yticks(range(len(data.index)),data.index)
 for i in range(a.shape[0]):
  for j in range(a.shape[1]):ax.text(j,i,'—' if not np.isfinite(a[i,j]) else f'{a[i,j]:.4f}',ha='center',va='center',fontsize=9,color='white' if np.isfinite(a[i,j]) and abs(a[i,j])>.6*limit else 'black')
 ax.set_title(title);fig.colorbar(im,ax=ax,label=label);fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
def main():
 m=json.loads((OUT/'manifest.json').read_text());assert m['status']=='analysis_completed';config=json.loads((OUT/'analysis_config.json').read_text());protect(config['protected_sha256']);protect(config['original_inputs_sha256'])
 s=pd.read_csv(OUT/'group_shap_importance.csv');p=pd.read_csv(OUT/'group_permutation_importance.csv');verify=json.loads((OUT/'verification.json').read_text());rep=pd.read_parquet(OUT/'permutation_repeats.parquet')
 assert p.repeats.eq(30).all() and len(rep)==len(p)*30 and p[['model','horizon','level','group']].duplicated().sum()==0
 for model in MODELS:
  for h in [3,6,12,24]:
   for level in ['primary_group','detail_group']:
    a=p[(p.model==model)&(p.horizon==h)&(p.level==level)];assert a.n_features.sum()==next(c['n_features'] for c in verify['checks'] if c['model']==model and c['horizon']==h)
 (OUT/'figures').mkdir(exist_ok=False)
 plt.rcParams['font.family']=['Apple SD Gothic Neo','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
 heatmap(matrix(p,'delta_rmse_mean',6),'2023 Validation · 6시간 권역 묶음 Permutation','교란 후 − 기존 RMSE (µg/m³)',OUT/'figures/h6_country_group_permutation.png')
 heatmap(matrix(s,'mean_abs_group_shap',6),'2023 Validation · 6시간 권역 TreeSHAP','mean |권역 합 SHAP| (평균 log 출력)',OUT/'figures/h6_country_group_shap.png')
 fig,axes=plt.subplots(2,3,figsize=(14,8));colors=plt.cm.tab10(np.arange(6))
 for ax,g in zip(axes.ravel(),GROUPS):
  for k,model in enumerate(MODELS):
   x=p[(p.level=='primary_group')&(p.group==g)&(p.model==model)].sort_values('horizon')
   if len(x):ax.errorbar(x.horizon,x.delta_rmse_mean,yerr=x.delta_rmse_sd,label=model,color=colors[k],marker='o',capsize=2)
  ax.axhline(0,color='grey',ls='--',lw=.8);ax.set_xticks([3,6,12,24]);ax.set_title(LABEL[g]);ax.set_xlabel('예측 시간 (h)');ax.set_ylabel('ΔRMSE (µg/m³)')
 axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('2023 Validation · Horizon별 그룹 Permutation (오차막대: 30회 반복 SD, 신뢰구간 아님)');fig.tight_layout();fig.savefig(OUT/'figures/all_horizons_country_permutation.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(14,8))
 for ax,g in zip(axes.ravel(),GROUPS):
  for k,model in enumerate(MODELS):
   x=s[(s.level=='primary_group')&(s.group==g)&(s.model==model)].sort_values('horizon')
   if len(x):ax.plot(x.horizon,x.mean_abs_group_shap,label=model,color=colors[k],marker='o')
  ax.set_xticks([3,6,12,24]);ax.set_title(LABEL[g]);ax.set_xlabel('예측 시간 (h)');ax.set_ylabel('mean |권역 합 SHAP| (log 출력)')
 axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('2023 Validation · Horizon별 그룹 TreeSHAP (평균 log 출력 설명)');fig.tight_layout();fig.savefig(OUT/'figures/all_horizons_country_shap.png',dpi=180);plt.close(fig)
 write_json(OUT/'mapping/mapping.meta.json',{'station_coordinates_source':'Dataset/Trajectory_Matching_2021_2024_v3/station_registry.csv','distance_to_target_km_reference':{'longitude':126.98,'latitude':37.57,'meaning':'기존 프로젝트의 서울 기준점; 111121의 정밀 측정소 좌표로 주장하지 않음','formula':'Haversine radius6371km'},'CN_border_proxies':'CN 입력망의 국경 대리 그룹으로 유지; 몽골 단일 관측소 그룹과 구분','mixed_all_pool':'KR67/CN61/JP188/MN1','trajectory_jp_pool':188,'trajectory_mn_pool':1,'history_jp_pool':24,'future_distance_analysis_executed':False})
 # Within-country regional ranks, not station ranks.
 d=p[(p.horizon==6)&(p.level=='detail_group')&p['group'].str.startswith(('KR:','CN:','JP:','MN:'))].copy();d=d.merge(s[(s.horizon==6)&(s.level=='detail_group')][['model','group','mean_abs_group_shap']],on=['model','group'],validate='one_to_one');d['country']=d['group'].str.split(':').str[0];d['rank_within_model_country']=d.groupby(['model','country']).delta_rmse_mean.rank(ascending=False,method='min');d['rank_within_model']=d.groupby('model').delta_rmse_mean.rank(ascending=False,method='min');d.sort_values(['model','rank_within_model']).to_csv(OUT/'h6_region_permutation.csv',index=False)
 td=p[(p.horizon==6)&(p.level=='detail_group')&p['group'].str.startswith('trajectory:')]
 td=td.merge(s[(s.horizon==6)&(s.level=='detail_group')][['model','group','mean_abs_group_shap']],on=['model','group'],validate='one_to_one')
 fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
 for ax,model in zip(axes,['M3T','M4T']):
  x=td[td.model==model].sort_values('delta_rmse_mean');ax.barh(x['group'],x.delta_rmse_mean,xerr=x.delta_rmse_sd,capsize=3);ax.axvline(0,color='grey',lw=.8);ax.set_title(model);ax.set_xlabel('ΔRMSE (µg/m³)')
 for missing_group in ['trajectory:jp','trajectory:mn']:
  axes[0].text(.015,axes[0].yaxis.convert_units(missing_group),'해당 없음',va='center',color='grey',fontsize=10)
 fig.suptitle('2023 Validation · 6시간 경로 묶음 (오차막대: 반복 SD)');fig.tight_layout();fig.savefig(OUT/'figures/h6_trajectory_permutation.png',dpi=180);plt.close(fig)
 primary=p[p.level.eq('primary_group')].merge(s[s.level.eq('primary_group')][['model','horizon','group','mean_abs_group_shap']],on=['model','horizon','group']);primary.to_csv(OUT/'country_shap_permutation_comparison.csv',index=False)
 h6=primary[primary.horizon.eq(6)][['model','group','n_features','mean_abs_group_shap','delta_rmse_mean','delta_rmse_sd']].round(5)
 change=p[(p.level=='primary_group')&p['group'].isin(['CN','JP','MN','trajectory'])].pivot(index=['model','group'],columns='horizon',values='delta_rmse_mean').reset_index().round(5)
 change_shap=s[(s.level=='primary_group')&s['group'].isin(['CN','JP','MN','trajectory'])].pivot(index=['model','group'],columns='horizon',values='mean_abs_group_shap').reset_index().round(5)
 m3_horizon=p[(p.level=='primary_group')&p.model.eq('M3')].pivot(index='horizon',columns='group',values='delta_rmse_mean').reset_index().round(5)
 regiontop=d.sort_values(['model','delta_rmse_mean'],ascending=[True,False]).groupby('model').head(5)[['model','group','n_features','mean_abs_group_shap','delta_rmse_mean','delta_rmse_sd']].round(5)
 cn_top=d[d.country.eq('CN')].sort_values(['model','delta_rmse_mean'],ascending=[True,False]).groupby('model').head(3)[['model','group','n_features','mean_abs_group_shap','delta_rmse_mean','delta_rmse_sd']].round(5)
 traj=td[['model','group','n_features','mean_abs_group_shap','delta_rmse_mean','delta_rmse_sd']].round(5)
 maxadd=max(c['shap_additivity_max_error'] for c in verify['checks']);maxpred=max(c['saved_ensemble_prediction_max_difference'] for c in verify['checks'])
 report=f'''# TreeSHAP·권역 묶음 Permutation 1차 분석 — 2023 Validation

M1/M2/M3/M4/M3T/M4T × 3/6/12/24시간, 총 24개 앙상블·72개 시드 모델을 분석했습니다. 주 Horizon은 6시간입니다. 모델·하이퍼파라미터·평가 행을 바꾸거나 재학습하지 않았습니다. Adaptive·2024 Test·관측소별 중요도·거리별 효과·기상 조건별 세부 분석은 실행하지 않았습니다. 향후 분석에 필요한 원시 SHAP·입력값·매핑·평가 시점은 보존했습니다.

주요 6시간 결과: 국내 관측 이력은 여섯 모델 모두 큰 의존도를 보입니다. 중국 전체 이력의 ΔRMSE는 M3 약0.897, M4 약0.954, M3T 약0.784, M4T 약0.843 µg/m³입니다. 경로 전체는 M3T 약1.570, M4T 약1.675로 양수입니다. 일본 이력은 M4/M4T에서 각각 약−0.039/−0.046, 몽골 이력은 약−0.004/0.0003으로 작습니다. 이는 교란 실험의 평균이며 일본·몽골의 보편적 무용성이나 인과적 악영향을 증명하지 않습니다.

T모델에서는 국내 이력 묶음 ΔRMSE가 기본 모델보다 낮고 경로 전체 ΔRMSE가 양수로 나타납니다. 중복된 정보를 포함한 입력 사이에서 모델 의존도가 달라지는 현상으로 볼 수 있으나, 이를 정보 대체의 인과 효과로 단정할 수 없습니다.

## 1. 방법과 검증

TreeSHAP: raw 출력·tree_path_dependent·근사 없음. 시드별 원시 SHAP과 기준값을 저장하고 부호 있는 SHAP을 시드 평균합니다. Feature 중요도는 mean(abs(mean_seed_SHAP)), 권역 중요도는 mean(abs(sum_group(mean_seed_SHAP)))입니다. 권역 내 상쇄가 있으므로 그룹 중요도와 개별 Feature 절댓값 합은 다릅니다. 두 값 모두 저장했습니다. 시드별 절댓값 중요도 평균·SD도 별도 저장했습니다.

**SHAP 단위는 log1p PM2.5이며 평균 log 출력을 설명합니다.** 공식 최종 예측 mean(expm1(seed raw))의 µg/m³ 분해나 log1p(최종 앙상블)의 분해가 아닙니다. 개별 SHAP에 expm1을 적용하지 않았습니다. 학습 트리에 기록된 표본 수를 배경으로 사용하며 2023·2024를 새 배경으로 넣지 않았습니다. 시드별 가산성 최대 오차 {maxadd:.3g}, 원 단위 앙상블 기존 예측과의 최대 차이 {maxpred:.3g}입니다.

Permutation: 각 그룹 전체 열에 같은 행 순열을 적용하며 시드3개에 동일 교란 입력을 전달합니다. Horizon별 평가 행·반복에 대해 모델·그룹 간 같은 순열 30개(seed0)를 사용했습니다. 농도/시차/가용성/결측 표시/NaN을 함께 이동하고 그 외 입력과 정답은 고정합니다. 지표는 음수 0 절단 후 원 단위 앙상블의 교란 RMSE−원래 RMSE입니다. 양수는 해당 입력을 섞으면 오차가 커진다는 뜻입니다. 음수도 삭제하지 않았습니다. 반복 SD는 순열의 Monte Carlo 변동이며 통계적 95% CI가 아닙니다. 중요도에 기존 모델 간 DM/Holm 공식 개선 판정을 적용하지 않았습니다.

평가 행은 Stage6 공통 채점 행을 재사용했습니다. 3h=2509 / 6h=2504 / 12h=2497 / 24h=2486이며 Horizon 간 추가 교집합은 없습니다. baseline RMSE와 예측 재현 검사 모두 통과했습니다. 계절·궤적 방향·ASOS 풍향·고농도 표시를 평가 행에 함께 보존했습니다.

## 2. 그룹 정의와 6시간 주요 결과

주 분석 그룹: local=서울17+ASOS46+달력7, KR=국내67곳×16, CN=M2 21곳×16 또는 전체61곳×16, JP=균형판24곳×16, MN=울란바토르1곳×16, trajectory=306 또는406개입니다. 없는 그룹은 해당 없음이며 0으로 채우지 않았습니다. 보조 그룹은 관측소 Registry의 국가+region_group, local 세부 묶음 및 trajectory scope/weather/geometry입니다. 각 그룹 체계 내에서 모든 입력이 정확히 한 번 배정되며 주·보조 그룹을 합산하지 않습니다.

{table(h6)}

## 3. Horizon별 차이

다음은 해외 관측 이력 및 경로 묶음의 ΔRMSE 평균입니다. 같은 그룹이라도 모델의 다른 입력·선택된 트리·공통 평가 행 수가 다르므로 단순 국가별 효과량으로 해석하지 않습니다.

{table(change)}

같은 그룹의 평균 log 출력 SHAP 중요도 변화:

{table(change_shap)}

M3의 국내·중국·공통 입력 의존도 비교(ΔRMSE):

{table(m3_horizon)}

M3의 중국 전체 ΔRMSE는 3/6/12/24h에서 약0.187/0.897/2.413/3.185로 커졌습니다. 반면 경로 전체의 의존도는 모델별로 변화 양상이 달라 단조 증가로 일반화하지 않습니다. M3의 국내 이력 ΔRMSE는 같은 순서로 약8.642/7.031/3.768/0.932였으며 24시간에서는 중국 이력의 값이 더 컸습니다. T모델의 경로 전체는 6~12시간에서 상대적으로 크고 24시간에는 M3T0.330/M4T0.297로 낮아졌습니다. 모두 원 단위 RMSE 기준의 고정 모델 의존도이며 시간별 관측망 인과 효과를 뜻하지 않습니다.

## 4. 국가 내부 권역 및 경로 구성

6시간에서 모델별 ΔRMSE가 큰 관측 권역 상위5개(국가:기존 권역, 관측소별 순위 아님):

{table(regiontop)}

중국 내부 권역 상위3개:

{table(cn_top)}

M3/M3T/M4T에서는 산둥 해안(Shandong_coast), 랴오닝 동해안(Liaoning_east_coast), 지린 중부(Jilin_central)가 반복적으로 상위권에 나타납니다. 동일 권역이라도 모델의 다른 입력과 경로와의 중복에 따라 중요도가 달라집니다. 국경 대리 권역은 기존 CN 입력망 분류를 유지하며 몽골 단일 관측소 이력과 혼동하지 않습니다.

6시간 경로 세부 그룹:

{table(traj)}

경로 all은 국내67/중국61/일본188/몽골1이 섞인 독립 그룹입니다. 이를 국가별로 배분하지 않았습니다. 국가별 경로 집계는 관측소 이력 그룹과 구분합니다. 경로 jp 범위도 등록 일본188곳이며 기본 관측 이력의 균형판24곳과 다릅니다. all/국가별 경로/기상/형태가 상관될 수 있어 각각의 Permutation 중요도를 합해 전체 경로 중요도로 간주하지 않습니다. M4T는 승인된 28개 열만 제외한 실제2924개를 사용했습니다. M3T는2424개입니다.

## 5. 연구 질문에 답할 수 있는 범위

이 분석은 **고정된 모델이 어느 입력 묶음·권역에 얼마나 의존하는지**, 그리고 그 의존도가 Horizon에 따라 어떻게 바뀌는지를 보여줍니다. 실제 국가별 오염 기여율이나 수송의 인과 효과를 추정하지 않습니다. 국가별 그룹은 Feature 수와 관측 밀도도 다르므로 중요도를 관측소당 기여나 국가별 배출량으로 환산하지 않습니다. 그룹을 섞으면 시간 연속성과 다른 그룹과의 상관이 깨지므로 교란 ΔRMSE는 새로운 관측망을 제거하고 재학습한 효과와 다릅니다. 정보가 중복되면 일부 그룹의 중요도가 작게 나올 수 있습니다. 서울 이력과 국내 관측, 중국 이력과 경로 농도 등의 상관도 고려해야 합니다.

SHAP은 평균 log 출력의 설명이며 Permutation은 최종 원 단위 RMSE 의존도입니다. 순위가 다른 것은 측정 대상과 상쇄·상관 구조가 다르기 때문일 수 있습니다. 2023은 설정 선택에도 사용한 기간이며 2024 일반화 성능 확인을 대신하지 않습니다. 공개 지연 미재현, 일본 README 시간 해석, 몽골의 높은 결측과 희소 경로 매칭이라는 기존 한계를 유지합니다. 결측인 입력을 모델이 활용하는 패턴도 중요도에 포함될 수 있습니다.

권역 단위 분석만으로 관측소별 대표성·거리 감쇠·기상 조건별 수송 차이는 결론 내릴 수 없습니다. 별도 승인 시 보존된 원시값으로 후속 분석할 수 있으며 이번에는 수행하지 않았습니다. 기존 M4T의 모델 간 공식 개선 판정도 변경하지 않았습니다.

## 6. 원시값 보존·재사용 방법

- `{{model}}/h{{h}}/shap_seed{{0,1,2}}.npz`: float64 Feature별 SHAP 행렬 values(n,p), 스칼라 base_value, raw_prediction, row_index, feature_names. 시드 평균·최종 원 단위 예측은 shap_mean_log_output.npz에 별도 저장.
- 같은 폴더 `feature_values.parquet`, `feature_order.json`: 당시 입력값/NaN·정확한 열 순서. 향후 색상·농도·거리/조건 분석을 위해 보존.
- `evaluation/h{{h}}.parquet`: row_index와 예측/정답 시각(KST), h, 정답, 기존 모든 모델 예측, 계절/궤적/ASOS풍향/고농도 정보. raw SHAP·feature_values의 행 순서는 이 row_index와 일치.
- `mapping/feature_station_region.parquet` 및 모델별CSV: Feature 순서·원정의/시간창·station_key·국가·권역·위경도·서울 기준점(126.98E,37.57N)의 Haversine 거리·Trajectory scope·주/보조 그룹. 순수 달력·경로기상/형태·다국가혼합 Feature에는 가상의 단일 관측소를 부여하지 않음. 좌표거리 보존은 거리별 중요도 분석을 실행했다는 뜻이 아님. 거리 기준점은 기존 프로젝트의 서울 기준점이며 정확한 111121 측정소 좌표로 주장하지 않음. mapping.meta.json에 기준 좌표·산식을 기록.
- `permutations/h{{h}}_row_indices.npy`: 실제 적용한30개 순열. 모든 그룹·시드에서 동일한 반복을 추적 가능.
- `permutation_repeats.parquet`: 모델/Horizon/체계/그룹/반복별 baseline·교란RMSE·차이.
- `feature_shap_importance.csv`, `group_shap_importance.csv`, `group_permutation_importance.csv`, `country_shap_permutation_comparison.csv`: 요약표.
- `figures/`: h6_country_group_shap.png, h6_country_group_permutation.png, h6_trajectory_permutation.png, all_horizons_country_permutation.png, all_horizons_country_shap.png. 모두2023 Validation 표기.
- `analysis_config.json`, `manifest.json`, `verification.json`, `file_inventory.csv`, `run.log`: 범위·버전·원본해시·검증·산출물 추적.

## 7. 보존·중단

기존 파일 {verify['previous_files_unchanged']}개와 원본 검증 입력 {verify['original_inputs_unchanged']}개의 해시를 확인했습니다. 기존 모델·예측·설정·공통 채점 결과는 불변입니다. 신규 분석 Manifest는 Stage6 Manifest와 채점 행 해시를 참조하고 기존 Manifest는 수정하지 않았습니다. 초기 실행에서 M1 6시간 모델 폴더에 config.json이 없음을 만나 분석 전에 중단한 뒤 기존 공식 configs 경로를 읽도록 수정했습니다. 데이터·모델 변경은 없었으며 해당 기록을 run.log에 보존했습니다.

현재 단계의 작업을 완료했습니다. 결과를 검토한 뒤 승인해 주시면 다음 단계로 진행하겠습니다.
'''
 (OUT/'REPORT.md').write_text(report)
 # Keep stage manifest immutable, add separate report manifest.
 write_json(OUT/'report_manifest.json',{'status':'completed','analysis_manifest_sha256':sha(OUT/'manifest.json'),'features_version':json.loads((ROOT/'Dataset/Origin_Features_2021_2024_v3/feature_manifest.json').read_text())['version'],'feature_manifest_sha256':sha(ROOT/'Dataset/Origin_Features_2021_2024_v3/feature_manifest.json'),'seeds':[0,1,2],'split':'validation','report_code_sha256':sha(Path(__file__)),'figures':[str(p.relative_to(OUT)) for p in (OUT/'figures').glob('*.png')],'test_executed':False,'adaptive_executed':False})
 files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='file_inventory.csv']+[ROOT/'scripts/run_interpretability.py',Path(__file__)]
 with (OUT/'file_inventory.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['path','bytes','sha256']);w.writeheader()
  for p in sorted(files):w.writerow({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)})
 print(h6.to_string(index=False));print(change.to_string(index=False));print(traj.to_string(index=False))
if __name__=='__main__':main()
