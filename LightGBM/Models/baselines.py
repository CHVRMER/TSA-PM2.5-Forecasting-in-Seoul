"""Baselines on the common grid: persistence y(t) and seasonal naive y(t+h-24) (= persistence at h=24).

Usage: python -I baselines.py
"""
import pandas as pd

import predictions_io
from config import HORIZONS
from data_access import seoul_pm25
from grid import load_grid

g = load_grid()
y = seoul_pm25()
t = g.origin_time_kst
persist = {h: g.seoul_pm25_now.to_numpy() for h in HORIZONS}
seasonal = {h: y.reindex(t + pd.Timedelta(hours=h - 24)).to_numpy() for h in HORIZONS}
print(predictions_io.wide_to_long(t, g.origin_split, persist, "persistence"))
print(predictions_io.wide_to_long(t, g.origin_split, seasonal, "seasonal_naive"))
