"""원 단위 RMSE/MAE와 2023년 검증 그림. 정답 결측은 채우지 않는다."""
import numpy as np


def metrics(y, raw_prediction):
    y, raw = np.asarray(y, float), np.asarray(raw_prediction, float)
    if not np.isfinite(y).all() or not np.isfinite(raw).all() or len(y) != len(raw):
        raise ValueError("평가 배열의 길이 또는 유한성 오류입니다.")
    pred = np.maximum(raw, 0)
    return {"n": len(y), "RMSE": float(np.sqrt(np.mean((pred-y)**2))),
            "MAE": float(np.mean(np.abs(pred-y))),
            "negative_predictions_clipped": int(np.sum(raw < 0)), "unit": "ug/m3"}


def plot_validation(predictions, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from pathlib import Path
    font = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = [font_manager.FontProperties(fname=str(font)).get_name(), "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    d = predictions[predictions.eligible].copy()
    horizon = int(d.horizon.iloc[0])
    model = str(d.model.iloc[0])
    daily = d.set_index("target_time_kst")[["y", "y_hat"]].resample("D").mean()
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    ax[0].plot(daily.index, daily.y, label="실제값", linewidth=1.2)
    ax[0].plot(daily.index, daily.y_hat, label=f"{model} 예측값", linewidth=1.1, alpha=.85)
    ax[0].set(title=f"2023년 검증: {horizon}시간 후 PM2.5 (적격 표본의 일별 평균)", ylabel="PM2.5 (µg/m³)")
    ax[0].legend()
    ax[1].scatter(d.y, d.y_hat, s=8, alpha=.25)
    lim = max(d.y.max(), d.y_hat.max())*1.02
    ax[1].plot([0, lim], [0, lim], color="gray", linestyle="--")
    ax[1].set(xlabel="실제 PM2.5 (µg/m³)", ylabel="예측 PM2.5 (µg/m³)",
              title="적격 3시간 격자 표본: 실제값과 예측값")
    fig.savefig(path, dpi=160)
    plt.close(fig)
