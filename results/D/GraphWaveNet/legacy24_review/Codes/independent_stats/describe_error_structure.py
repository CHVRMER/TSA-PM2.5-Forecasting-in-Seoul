"""Additional descriptive diagnostics; no training or new significance tests."""
from pathlib import Path
import io
import zipfile
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ZIP = Path('C:/Users/kimjy/Downloads/VastAI-20261111T003043Z-1-001.zip')
BATCH = 'VastAI/20261111_014538_406520_eaa72758/'
rows = []
composition = []
with zipfile.ZipFile(ZIP) as z:
    for split in ['validation', 'test']:
        prefix = BATCH + f'Common_Scoring/Combined/{split}/'
        ns = [n for n in z.namelist() if n.startswith(prefix) and n.endswith('/score_rows.parquet')]
        assert len(ns) == 1
        f = pd.read_parquet(io.BytesIO(z.read(ns[0])))
        # Count only one copy of every fixed label for sample composition.
        target = f.loc[f.input_condition.eq('M1')]
        for (h, season), g in target.groupby(['horizon', 'season']):
            composition.append({'split': split, 'horizon': h, 'season': season, 'n': len(g),
                'y_mean': g.y_true.mean(), 'y_std': g.y_true.std(), 'y_ge35_n': int(g.y_true.ge(35).sum()),
                'y_ge35_fraction': g.y_true.ge(35).mean(), 'y_ge75_n': int(g.y_true.ge(75).sum()),
                'y_max': g.y_true.max()})
        for (model, condition, h), g in f.groupby(['model', 'input_condition', 'horizon']):
            for stratum, selected in [('all', g), ('future_ge35', g.loc[g.y_true.ge(35)]),
                                       ('future_lt35', g.loc[g.y_true.lt(35)])]:
                if selected.empty:
                    continue
                error = selected.y_hat-selected.y_true
                rows.append({'split': split, 'model': model, 'input_condition': condition, 'horizon': h,
                    'stratum': stratum, 'n': len(selected),
                    'RMSE': np.sqrt(np.mean(error**2)), 'MAE': np.abs(error).mean(), 'MBE': error.mean(),
                    'absolute_error_p50': np.abs(error).quantile(.5),
                    'absolute_error_p90': np.abs(error).quantile(.9),
                    'absolute_error_p95': np.abs(error).quantile(.95),
                    'absolute_error_p99': np.abs(error).quantile(.99),
                    'absolute_error_max': np.abs(error).max(),
                    'truth_mean': selected.y_true.mean(), 'prediction_mean': selected.y_hat.mean(),
                    'truth_max': selected.y_true.max(), 'prediction_max': selected.y_hat.max()})
pd.DataFrame(rows).to_csv(OUT/'all_model_error_distribution_descriptive.csv', index=False)
pd.DataFrame(composition).to_csv(OUT/'seasonal_evaluation_sample_composition.csv', index=False)
print(pd.DataFrame(composition).loc[lambda x: x.horizon.eq(24)].round(4).to_string(index=False))
print(pd.DataFrame(rows).loc[lambda x: x.horizon.eq(24) & x.stratum.eq('future_ge35') & x.split.eq('test')].round(4).to_string(index=False))
