"""원본을 수정하지 않고 지정 분할만 읽는다. 결측 대체는 하지 않는다."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .feature_groups import feature_names


@dataclass
class SplitData:
    X: pd.DataFrame
    meta: pd.DataFrame
    eligible: np.ndarray


def load_split(folder, set_name, horizon, split, expected_features=None):
    if horizon not in (3, 6, 12, 24) or split not in ("train", "validation", "test"):
        raise ValueError("지원하지 않는 Horizon 또는 분할입니다.")
    names = feature_names(folder, set_name, expected_features)
    label = f"y_pm25_h{horizon}"
    eligibility = f"eligible_h{horizon}"
    cols = ["origin_time_kst", "origin_split", "feature_cutoff_time_kst",
            f"target_time_h{horizon}", f"target_split_h{horizon}", eligibility]
    table = pq.read_table(folder / "origin_features_2021_2024_kst.parquet",
                          columns=names + cols, filters=[("origin_split", "=", split)],
                          use_threads=False).to_pandas()
    meta = table[cols].copy()
    # 정답은 적격 행만 따로 읽는다. 검증 경계의 2024 정답은 열지 않는다.
    labels = pq.read_table(folder / "origin_features_2021_2024_kst.parquet",
                           columns=["origin_time_kst", label],
                           filters=[("origin_split", "=", split), (eligibility, "=", True)],
                           use_threads=False).to_pandas().set_index("origin_time_kst")[label]
    meta[label] = meta.origin_time_kst.map(labels)
    X = table[names].copy()
    origin = meta.origin_time_kst
    eligible = meta[eligibility].to_numpy(dtype=bool)
    if not origin.is_unique or not origin.is_monotonic_increasing:
        raise ValueError("예측 시점 중복 또는 정렬 오류입니다.")
    if not (meta.feature_cutoff_time_kst == origin).all():
        raise ValueError("Feature cutoff가 예측 시각과 다릅니다.")
    if not (meta[f"target_time_h{horizon}"] == origin + pd.Timedelta(hours=horizon)).all():
        raise ValueError("정답 시각 오류입니다.")
    if not (meta.loc[eligible, f"target_split_h{horizon}"] == split).all():
        raise ValueError("정답이 분할 경계를 넘습니다.")
    if not np.isfinite(meta.loc[eligible, label].to_numpy(float)).all():
        raise ValueError("적격 정답에 결측 또는 무한대가 있습니다.")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("입력 Feature에 무한대가 있습니다.")
    return SplitData(X, meta, eligible)
