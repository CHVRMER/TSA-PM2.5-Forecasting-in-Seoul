"""Score SARIMAX M1-M4 (team station sets, feature_sets.json model_B_M1..M4) on 2023 with the team scorer.

Pairs fixed before results are seen: every run vs persistence (automatic), each M vs A_SARIMA,
stacking M2:M1, M3:M2, M4:M3. Writes results/score_M/* and results/planA_M_table.csv.
Usage: python run_score_m.py
"""
import subprocess
import sys

import pandas as pd

from config import PRED, RESULTS, ROOT

runs = ["A_SARIMA"] + [f"A_SARIMAX__M{k}" for k in (1, 2, 3, 4)]
pairs = [f"{r}:A_SARIMA" for r in runs[1:]]
pairs += ["A_SARIMAX__M2:A_SARIMAX__M1", "A_SARIMAX__M3:A_SARIMAX__M2", "A_SARIMAX__M4:A_SARIMAX__M3"]
out = RESULTS / "score_M"
cmd = [sys.executable, str(ROOT / "Model" / "Common" / "score.py"),
       "--grid", str(ROOT / "Dataset" / "Evaluation_Contract_20261107" / "evaluation_grid.parquet"),
       "--predictions", *[str(PRED / f"{r}.parquet") for r in runs],
       "--split", "validation", "--pairs", *pairs, "--output-dir", str(out)]
subprocess.run(cmd, check=True, cwd=str(ROOT / "Model" / "Common"))

m = pd.read_csv(out / "metrics.csv")
keep = ["horizon", "model", "n", "RMSE", "MAE", "MBE", "R2", "RMSE_improvement_pct"]
m[keep].sort_values(["horizon", "model"]).to_csv(RESULTS / "planA_M_table.csv", index=False, encoding="utf-8-sig")
print(m.pivot(index="model", columns="horizon", values="RMSE").round(3).to_string())
