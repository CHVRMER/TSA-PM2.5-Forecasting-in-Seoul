"""2023 validation DCRNN curve; preserved wide style and 0.1/0.5 ticks."""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "Evidence"
SOURCE = TABLES / "validation_condition_metrics.csv"
COMPARISONS = TABLES / "validation_predeclared_condition_comparisons.csv"
MATRIX = TABLES / "validation_rmse_matrix.csv"
H = [3, 6, 12, 24]
ORDER = ["M1", "M2", "M3", "M4", "M3T", "M4T"]
COLORS = {"M1": "#C73535", "M2": "#D48B2C", "M3": "#2E8B57",
          "M4": "#2866B2", "M3T": "#168C91", "M4T": "#7E57A2"}
MARKERS = {"M1": "o", "M2": "^", "M3": "s", "M4": "D", "M3T": "P", "M4T": "X"}
LABELS = {"M1": "M1: Domestic", "M2": "M2: +Nearby China", "M3": "M3: +All China",
          "M4": "M4: +Japan / Mongolia", "M3T": "M3T: +Trajectory summaries", "M4T": "M4T: +Trajectory summaries"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for folder in ["Results", "Evidence", "Audit"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    source_hash = sha(SOURCE)
    source_records = {str(path): sha(path) for path in [SOURCE, COMPARISONS, MATRIX]}
    all_data = pd.read_csv(SOURCE)
    assert len(all_data) == 36 and set(all_data.split) == {"validation"}
    assert set(all_data.seed) == {11} and set(all_data.prediction_kind) == {"seed"}
    data = all_data.loc[all_data.horizon.isin(H)].copy()
    assert len(data) == 24 and not data.duplicated(["horizon", "input_condition"]).any()
    values = data.pivot(index="horizon", columns="input_condition", values="RMSE").loc[H, ORDER]
    archived_matrix = pd.read_csv(MATRIX).set_index("input_condition").loc[ORDER, [str(h) for h in H]]
    assert np.allclose(values.to_numpy(), archived_matrix.to_numpy().T, rtol=0, atol=1e-12)
    assert data.groupby("horizon").n.nunique().eq(1).all()
    all_comparisons = pd.read_csv(COMPARISONS)
    assert len(all_comparisons) == 42 and set(all_comparisons.split) == {"validation"}
    comparisons = all_comparisons.loc[all_comparisons.horizon.isin(H)].copy()
    assert len(comparisons) == 28 and not comparisons.improved.any()
    assert not comparisons.significant_worsening.any()
    assert values.idxmin(axis=1).to_dict() == {3: "M2", 6: "M3", 12: "M1", 24: "M1"}
    assert values.M3T.gt(values.M3).all() and values.M4T.gt(values.M4).all()
    assert np.isfinite(values).all().all()
    assert values.gt(5.5).all().all() and values.lt(13.5).all().all()
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.0, 6.75), dpi=180, facecolor="white")
    ax = fig.add_axes([0.09, 0.28, 0.875, 0.67])
    for name in ORDER:
        artist, = ax.plot(H, values[name], color=COLORS[name], marker=MARKERS[name],
                          markerfacecolor="white", markeredgewidth=1.15, markersize=6.3,
                          linewidth=1.35, linestyle="--" if name.endswith("T") else "-", label=LABELS[name])
        assert np.array_equal(artist.get_xdata(), H)
        assert np.array_equal(artist.get_ydata(), values[name].to_numpy())
    ticks = np.arange(55, 136) / 10
    labelled_ticks = np.arange(55, 136, 5) / 10
    ax.set_xlim(2, 25)
    ax.set_xticks(H)
    ax.set_ylim(5.5, 13.5)
    ax.set_yticks(labelled_ticks)
    ax.set_yticks(ticks, minor=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="major", labelsize=11.5, length=3.5, width=0.7, pad=5)
    ax.tick_params(axis="y", which="minor", length=1.8, width=0.4, color="#AAAAAA", labelleft=False)
    ax.tick_params(axis="x", labelsize=12, width=0.7, pad=8)
    ax.set_xlabel("Forecast horizon (hours)", fontsize=13, labelpad=14)
    ax.set_ylabel("RMSE (µg/m³)", fontsize=13, labelpad=14)
    ax.grid(axis="y", which="major", color="#DEDEDE", linewidth=0.55)
    ax.grid(axis="y", which="minor", color="#F2F2F2", linewidth=0.25)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#999999")
        ax.spines[side].set_linewidth(0.7)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.54, 0.065),
               ncol=3, frameon=False, fontsize=10.3, columnspacing=1.5)
    caption = "D DCRNN, 2023 validation. RMSE (µg/m³) across M1–M4T input conditions."
    fig.text(0.09, 0.020, caption, fontsize=10.3, color="#555555")
    assert len(fig.axes) == 1 and not ax.patches and ax.get_yscale() == "linear"
    all_tick_positions = np.sort(np.concatenate([ax.get_yticks(), ax.get_yticks(minor=True)]))
    assert len(all_tick_positions) == 81 and np.allclose(all_tick_positions, ticks)
    assert np.allclose(np.diff(all_tick_positions), 0.1)
    assert len(ax.get_yticks()) == 17 and np.allclose(np.diff(ax.get_yticks()), 0.5)
    assert all(not label.get_text() for label in ax.get_yticklabels(minor=True))
    assert np.isclose(fig.get_figwidth() / fig.get_figheight(), 16 / 9)
    assert all("seed" not in item.get_text().lower()
               for item in fig.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text)))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [item.get_window_extent(renderer) for item in ax.get_yticklabels()]
    gaps = [above.y0 - below.y1 for below, above in zip(boxes[:-1], boxes[1:])]
    assert min(gaps) > 0, "The 0.5 numeric tick labels overlap."
    assert [item.get_text() for item in ax.get_yticklabels()] == [f"{v:.1f}" for v in labelled_ticks]
    for item in fig.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text)):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        assert box.x0 >= -2 and box.y0 >= -2, item.get_text()
        assert box.x1 <= fig.bbox.width + 2 and box.y1 <= fig.bbox.height + 2, item.get_text()
    outputs = []
    for ext in ["png", "svg", "pdf"]:
        path = ROOT / "Results" / f"dcrnn_validation2023_rmse_wide_3_24h.{ext}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=260, facecolor="white")
        outputs.append({"file": path.name, "sha256": sha(path)})
    plt.close(fig)
    assert sha(SOURCE) == source_hash
    for path in [SOURCE, COMPARISONS, MATRIX]:
        assert sha(path) == source_records[str(path)]
        evidence_copy = ROOT / "Evidence" / path.name
        if path.resolve() != evidence_copy.resolve():
            shutil.copy2(path, evidence_copy)
    data.to_csv(ROOT / "Results/validation_metrics_3_24h.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(ROOT / "Results/validation_comparisons_3_24h.csv", index=False, encoding="utf-8-sig")
    values.to_csv(ROOT / "Results/validation_rmse_matrix_3_24h.csv", encoding="utf-8-sig")
    slide_copy = (
        "DCRNN: Limited Gains from Added Inputs\n\n"
        "- Domestic-only PM2.5 inputs perform best at 12 and 24 h.\n"
        "- Foreign inputs offer no significant gains over M1.\n"
        "- Trajectory summaries show higher RMSE at all four horizons.\n\n"
        "2023 validation | 3–24 h forecasts\n"
    )
    (ROOT / "Results/slide_copy_en.txt").write_text(slide_copy, encoding="utf-8")
    audit = {"source": str(SOURCE), "source_sha256": source_hash, "source_unchanged": True,
             "all_sources_sha256": source_records, "split": "validation", "evaluation_year": 2023,
             "seed": 11, "metric_matrix_crosscheck": True,
             "visible_caption": caption, "seed_displayed_in_figure": False,
             "raw_RMSE": values.to_dict(), "horizons": H, "colors": COLORS, "markers": MARKERS,
             "y_limits": [5.5, 13.5], "y_tick_spacing": 0.1, "numeric_y_label_spacing": 0.5,
             "numeric_y_label_count": 17, "total_tick_count": 81, "aspect_ratio": "16:9",
             "figure_size_inches": [12.0, 6.75], "minor_y_labels_visible": False,
             "minimum_tick_label_gap_pixels_at_180dpi": float(min(gaps)),
             "axis_count": 1, "detail_axes": False, "axis_breaks": False, "masking": False,
             "best_condition_by_horizon": values.idxmin(axis=1).to_dict(),
             "planned_comparisons_3_24h": 28, "significant_input_improvements_3_24h": 0,
             "significant_input_worsenings_3_24h": 0,
             "original_holm_family_size_per_split": 84, "adjusted_p_values_recomputed": False,
             "summary_feature_rmse_increases_are_descriptive_not_significant": True,
             "outputs": outputs}
    (ROOT / "Audit/validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"exports": 3, "single_curve_panel": True, "y_tick_spacing": 0.1,
                      "split": "2023 validation", "numeric_label_spacing": 0.5, "label_count": 17, "aspect_ratio": "16:9",
                      "labels_do_not_overlap": True, "source_unchanged": True}))


if __name__ == "__main__":
    main()
