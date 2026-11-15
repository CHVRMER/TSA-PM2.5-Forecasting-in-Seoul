"""The one prediction format every model writes: origin_time_kst, horizon, model, split, y_hat (ug/m3).

Each <model>.parquet gets a sidecar <model>.meta.json: data version (feature manifest version/created time),
seeds and the settings the model passes in, so every prediction can be traced to its inputs.
"""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import FEATURES, HORIZONS, PRED, SPLITS

SCHEMA = pa.schema([("origin_time_kst", pa.timestamp("us", tz="Asia/Seoul")), ("horizon", pa.int8()),
                    ("model", pa.string()), ("split", pa.string()), ("y_hat", pa.float64())])


def data_version():
    m = json.loads((FEATURES / "feature_manifest.json").read_text(encoding="utf-8"))
    return {"features_version": m["version"], "features_created_kst": m["created_kst"],
            "features_rows": m["rows"], "features_columns": m["columns"]}


def save(df, model, meta=None):
    """Validate and write predictions/<model>.parquet (+ .meta.json); overwrites only that model's own files."""
    (PRED / f"{model}.meta.json").write_text(json.dumps(
        {"model": model, "written_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
         **data_version(), **(meta or {})}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    df = df.assign(model=model)[SCHEMA.names]
    assert set(df.horizon.unique()) <= set(HORIZONS) and set(df.split.unique()) <= set(SPLITS)
    assert not df.duplicated(["origin_time_kst", "horizon"]).any(), "duplicate origin/horizon"
    df = df.assign(origin_time_kst=pd.to_datetime(df.origin_time_kst).dt.tz_convert("Asia/Seoul"))
    pq.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False), PRED / f"{model}.parquet")
    return PRED / f"{model}.parquet"


def load(models):
    return pd.concat([pd.read_parquet(PRED / f"{m}.parquet") for m in models], ignore_index=True)


def wide_to_long(origins, splits, preds, model, meta=None):
    """preds: {h: array aligned with origins} -> long frame."""
    rows = [pd.DataFrame({"origin_time_kst": origins, "horizon": h, "split": splits, "y_hat": v})
            for h, v in preds.items()]
    return save(pd.concat(rows, ignore_index=True), model, meta)
