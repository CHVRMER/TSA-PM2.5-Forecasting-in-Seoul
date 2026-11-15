"""A step 4: score Plan A on 2023 with the team scorer (TSA/Model/Common/score.py, v5, unchanged).

Baselines (persistence, seasonal_naive) are produced by the scorer itself.
Comparison pairs are fixed here before results are seen (team rule):
  every run vs persistence (automatic), every SARIMAX condition vs A_SARIMA,
  stacking X_W -> X_WN -> X_WNF, leave-one-out X_WNF vs X_NF / X_WF / X_WN.
Writes results/score_validation/* and results/planA_validation_table.csv.
Usage: python run_score.py
"""
import subprocess
import sys

import pandas as pd

from a_sarimax_ablation import CONDITIONS
from config import PRED, RESULTS, ROOT

runs = ["A_SARIMA"] + [f"A_SARIMAX__{c}" for c in CONDITIONS if (PRED / f"A_SARIMAX__{c}.parquet").exists()]
pairs = [f"{r}:A_SARIMA" for r in runs[1:]]
stack_loo = [("X_WN", "X_W"), ("X_WNF", "X_WN"), ("X_WNF", "X_NF"), ("X_WNF", "X_WF")]
pairs += [f"A_SARIMAX__{a}:A_SARIMAX__{b}" for a, b in stack_loo if f"A_SARIMAX__{a}" in runs and f"A_SARIMAX__{b}" in runs]
out = RESULTS / "score_validation"
cmd = [sys.executable, str(ROOT / "Model" / "Common" / "score.py"),
       "--grid", str(ROOT / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet"),
       "--predictions", *[str(PRED / f"{r}.parquet") for r in runs],
       "--split", "validation", "--pairs", *pairs, "--output-dir", str(out)]
subprocess.run(cmd, check=True, cwd=str(ROOT / "Model" / "Common"))

m = pd.read_csv(out / "metrics.csv")
keep = ["horizon", "model", "n", "RMSE", "MAE", "MBE", "R2", "RMSE_improvement_pct", "MAE_improvement_pct",
        "negative_predictions_clipped"]
table = m[keep].sort_values(["horizon", "RMSE"])
table.to_csv(RESULTS / "planA_validation_table.csv", index=False, encoding="utf-8-sig")
print(table.round(3).to_string(index=False))
