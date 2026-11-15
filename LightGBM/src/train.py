"""승인된 LightGBM 모델·Horizon 하나를 공통 절차로 학습한다."""
import hashlib
import importlib.metadata
import json
import logging
import platform
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data_loader import load_split
from .evaluate import metrics, plot_validation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def train_lightgbm(root, config, result_dir, model_dir):
    model = config["model"]
    if model not in ("M1", "M2", "M3", "M4", "M3T", "M4T") or config["horizon"] not in (3, 6, 12, 24):
        raise ValueError("현재 승인된 학습 모델은 M1·M2·M3·M4·M3T·M4T뿐입니다.")
    audit = json.loads((root/config["validation_report"]).read_text())
    if audit["status"] != "passed":
        raise ValueError("데이터 검증이 통과되지 않았습니다.")
    folder = root/config["data_folder"]
    source_hashes = {}
    for rel, expected in audit["input_sha256"].items():
        actual = sha(root/rel)
        # 승인된 공통 채점기 교체: 기존 원문은 별도 보존본으로 검증한다.
        if rel == "Models/score.py" and sha(root/"Models/score_legacy_20261108.py") == expected:
            source_hashes[rel] = actual
            continue
        if actual != expected:
            raise ValueError(f"검증 이후 입력/기존 코드가 변경되었습니다: {rel}")
        source_hashes[rel] = actual
    loaded = {}
    h = config["horizon"]
    for split in ("train", "validation"):
        loaded[split] = load_split(folder, config["feature_set"], h, split, config["expected_features"])
    tr, va = loaded["train"], loaded["validation"]
    if tr.eligible.sum() != config["expected_train_rows"] or va.eligible.sum() != config["expected_validation_rows"]:
        raise ValueError("검증된 적격 표본 수가 달라졌습니다.")
    if tr.meta[f"target_time_h{h}"].loc[tr.eligible].max() >= va.meta.origin_time_kst.min():
        raise ValueError("학습 정답이 검증 구간에 들어갑니다.")
    names = list(tr.X.columns)
    if names != list(va.X.columns):
        raise ValueError("분할별 Feature 순서가 다릅니다.")
    ytr = tr.meta.loc[tr.eligible, f"y_pm25_h{h}"].to_numpy(float)
    yva = va.meta.loc[va.eligible, f"y_pm25_h{h}"].to_numpy(float)
    xtr, xva = tr.X.loc[tr.eligible], va.X.loc[va.eligible]
    dtr = lgb.Dataset(xtr, label=np.log1p(ytr), free_raw_data=False)
    dva = lgb.Dataset(xva, label=np.log1p(yva), reference=dtr, free_raw_data=False)
    trials, best = [], None
    for i, params in enumerate(config["grid"]):
        logging.info("설정 %s/%s 시작: %s", i+1, len(config["grid"]), params)
        booster = lgb.train({**config["base_params"], **params, "seed": config["seeds"][0]},
                            dtr, config["rounds"], valid_sets=[dva], valid_names=["validation"],
                            callbacks=[lgb.early_stopping(config["early_stopping_rounds"], verbose=False)])
        prediction = np.expm1(booster.predict(xva))
        score = metrics(yva, prediction)
        trial = {"params": params, "best_iteration": booster.best_iteration, "validation": score,
                 "early_stopping_log_l2": float(booster.best_score["validation"]["l2"])}
        trials.append(trial)
        logging.info("설정 완료: %s", trial)
        if best is None or score["RMSE"] < best[0]["validation"]["RMSE"]:
            best = trial, booster
    selected, first = best
    boosters = [first]
    for seed in config["seeds"][1:]:
        logging.info("선택 설정 추가 시드 %s 학습", seed)
        boosters.append(lgb.train({**config["base_params"], **selected["params"], "seed": seed},
                                  dtr, selected["best_iteration"]))
    predictions, seed_scores, roundtrip = [], [], []
    for seed, booster in zip(config["seeds"], boosters):
        raw = np.expm1(booster.predict(va.X))
        path = model_dir/f"{model}_h{h}_seed{seed}.txt"
        booster.save_model(str(path), num_iteration=selected["best_iteration"])
        restored = lgb.Booster(model_file=str(path))
        restored_raw = np.expm1(restored.predict(va.X, num_threads=2))
        difference = float(np.max(np.abs(raw-restored_raw)))
        if not np.allclose(raw, restored_raw, rtol=1e-12, atol=1e-10):
            raise ValueError("저장 후 재로드 예측이 다릅니다.")
        roundtrip.append({"seed": seed, "max_absolute_prediction_difference": difference})
        predictions.append(raw)
        seed_scores.append({"seed": seed, **metrics(yva, raw[va.eligible])})
    raw_mean = np.mean(predictions, axis=0)
    result = pd.DataFrame({"origin_time_kst": va.meta.origin_time_kst,
                           "target_time_kst": va.meta[f"target_time_h{h}"],
                           "horizon": h, "model": model, "split": "validation",
                           "eligible": va.eligible, "y": va.meta[f"y_pm25_h{h}"],
                           "y_hat_raw": raw_mean, "y_hat": np.maximum(raw_mean, 0)})
    for seed, raw in zip(config["seeds"], predictions):
        result[f"y_hat_seed{seed}"] = raw
    result.to_parquet(result_dir/f"predictions_{model}_h{h}_validation.parquet", index=False)
    result.to_csv(result_dir/f"predictions_{model}_h{h}_validation.csv", index=False)
    from .common_scoring import export_predictions
    export_predictions(result.assign(y_hat=raw_mean), result_dir/f"{model}.parquet", {
        "seeds": config["seeds"], "settings": config, "selection": selected, "n_features": len(names),
        "training_years": [2021, 2022], "selection_year": 2023, "no_refit_on_2023": True,
        "trained_models_sha256": {str(p.relative_to(root)): sha(p) for p in model_dir.glob(f"{model}_h{h}_seed*.txt")},
        "validation_only": True})
    overall = metrics(yva, raw_mean[va.eligible])
    # Persistence comparison uses the exact same rows for both methods if y(t) is observed.
    now = va.X.seoul_pm25_now.to_numpy(float)
    common = va.eligible & np.isfinite(now)
    baselines = {"n_common": int(common.sum()),
                 f"{model}_common": metrics(result.y[common], raw_mean[common]),
                 "persistence_common": metrics(result.y[common], now[common])}
    information = {"status": "completed", "model": model, "horizon": h, "n_features": len(names),
                   "eligible_train_rows": int(tr.eligible.sum()), "eligible_validation_rows": int(va.eligible.sum()),
                   "validation_origins_predicted": len(result), "metrics": overall, "seed_metrics": seed_scores,
                   "selected": selected, "trials": trials, "persistence_comparison": baselines,
                   "model_reload_checks": roundtrip, "test_2024_loaded_or_predicted": False,
                   "input_missing_cells": {s: int(d.X.isna().sum().sum()) for s,d in loaded.items()},
                   "input_X_array_bytes": {s: int(d.X.to_numpy().nbytes) for s,d in loaded.items()},
                   "source_sha256": source_hashes, "target_transform": "log1p",
                   "selection_note": "2023년으로 설정을 선택했으므로 검증 성능에는 선택 편향이 있습니다. 최종 테스트 성능이 아닙니다.",
                   "known_limits": ["Historical-data research; publication latency not recreated", "2024 final evaluation deferred"],
                   "python": platform.python_version(),
                   "versions": {p: importlib.metadata.version(p) for p in ["numpy","pandas","pyarrow","lightgbm","scikit-learn","matplotlib"]}}
    plot_validation(result, result_dir/f"validation_{model}_h{h}.png")
    write_json(result_dir/"feature_names.json", names)
    write_json(result_dir/"config.json", config)
    write_json(result_dir/"metrics_and_selection.json", information)
    write_json(model_dir/"ensemble_manifest.json", {"model":model, "horizon":h, "aggregation":"mean of expm1(seed prediction), then clip at zero for scoring", "seeds": config["seeds"], "feature_names":names, "selected":selected})
    for rel, before in source_hashes.items():
        if sha(root/rel) != before:
            raise ValueError(f"입력 파일 변경 발견: {rel}")
    logging.info("완료: %s", overall)
    return information


# 이전 M1 실행기와 호환하며 학습 구현은 하나만 유지한다.
train_m1 = train_lightgbm
