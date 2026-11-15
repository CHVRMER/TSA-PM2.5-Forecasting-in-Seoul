"""Shared paths and fixed settings for models A-D. Every model reads from here."""
from pathlib import Path

import os
# Plan A: data from Dataset_11_07_regions_v5.zip. Set TSA_ROOT on Colab (e.g. /content/drive/MyDrive/TSA).
ROOT = Path(os.environ.get("TSA_ROOT", Path(__file__).resolve().parents[1] / "TSA"))
if os.name == "nt" and not str(ROOT).startswith("\\\\?\\"):
    ROOT = Path("\\\\?\\" + str(ROOT))                 # Windows long-path prefix (paths > 260 chars)
DATASET = ROOT / "Dataset"
FEATURES = DATASET / "Origin_Features_2021_2024_v3"
MATCHING = DATASET / "Trajectory_Matching_2021_2024_v3"
FEATURE_CODE = DATASET / "Codes" / "Origin_Features_v3"
MODELS = Path(__file__).resolve().parent
WORK = MODELS / "work"                 # derived inputs (region means, node matrix, graphs)
PRED = MODELS / "predictions"          # one parquet per model, common schema
RESULTS = MODELS / "results"           # scores, coefficient tables, importance, weights

HORIZONS = (3, 6, 12, 24)              # 6 h primary
SPLITS = ("train", "validation", "test")
TARGET_KEY = "KR:111121"               # Seoul Jung-gu
SEOUL = (126.98, 37.57)                # lon, lat (team receptor)
SEEDS = (0, 1, 2)
HIGH_PM = 35.0                         # ug/m3, high-concentration threshold


def split_of(year):
    return "train" if year <= 2022 else "validation" if year == 2023 else "test"


for d in (WORK, PRED, RESULTS):
    d.mkdir(exist_ok=True)
