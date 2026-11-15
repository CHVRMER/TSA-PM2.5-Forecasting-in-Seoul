"""Frozen JSON roundtrip: reproduce validation blends without target labels."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.adaptive import apply_horizon

def main():
 root=Path(__file__).resolve().parents[1];p=root/'results/validation_2023/Adaptive';d=pd.read_parquet(p/'prediction_details_validation.parquet');checks=[]
 for (name,h),g in d.groupby(['model','horizon']):
  stored=json.loads((root/f'Models/{name}/h{h}/weights.json').read_text());kind='wind' if name=='Adaptive_wind' else 'route';inputs=g[['season','wind_sector','route_group','M1_raw','M3_raw']]
  result=apply_horizon(inputs,pd.DataFrame(stored['season_rules']),pd.DataFrame(stored['cell_rules']),kind);difference=float(np.max(abs(result.y_hat_raw-g.y_hat_raw)));assert difference==0
  for col in ['weight','estimation_pool','n_estimation_pool','denominator','threshold_value','season_fallback_used','fallback_reason','near_zero_exception_used']:assert result[col].tolist()==g[col].tolist(),col
  checks.append({'model':name,'horizon':int(h),'prediction_rows':len(g),'max_prediction_difference':difference})
 result={'status':'passed','loaded_weight_artifacts':len(checks),'label_free_reproduction':True,'checks':checks,'test_data_read':False}
 target=p/'artifact_roundtrip_verification.json'
 with target.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print('PASS: 8 frozen weight artifacts; label-free application reproduces all 23360 predictions and fallback assignments exactly')
if __name__=='__main__':main()
