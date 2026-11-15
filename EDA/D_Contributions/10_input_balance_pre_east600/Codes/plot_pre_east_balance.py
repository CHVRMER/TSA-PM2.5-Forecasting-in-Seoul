"""Plain chart-only input inventories with an explicitly broken linear x axis.

Both visible axis segments use the same pixels per unit. Only Mongolia uses
an annotated minimum display width; its data value remains 1 in every export.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


PROJECT = Path(r"C:\Users\kimjy\Desktop\Time Series Analysis\Project")
CONFIG_SOURCE = Path(__file__).resolve().parents[1] / "Evidence/neighbor_experiments.json"
LEGACY_CONFIG_SOURCE = Path(__file__).resolve().parents[1] / "Evidence/legacy_regional_experiments.json"
COUNTRIES = ["KR", "CN", "JP", "MN"]
COUNTRY_NAMES = ["Domestic", "China", "Japan", "Mongolia"]
RED = "#C73535"
GRAY = "#B8B8B8"
TEXT = "#282828"
LOW_RANGE = (0.0, 150.0)
HIGH_RANGE = (490.0, 535.0)
MINIMUM_MONGOLIA_WIDTH = 3.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wave_gap(fig, left_edge, right_edge, y_bottom, y_top):
    """Make the two break edges wavy, without changing either linear scale."""
    y = np.linspace(y_bottom, y_top, 100)
    phase = np.linspace(0, 2 * np.pi, len(y))
    left_wave = left_edge - 0.0027 + 0.0025 * np.sin(phase)
    right_wave = right_edge + 0.0027 + 0.0025 * np.sin(phase)
    polygon = np.vstack([np.column_stack([left_wave, y]),
                         np.column_stack([right_wave[::-1], y[::-1]])])
    fig.add_artist(Polygon(polygon, closed=True, transform=fig.transFigure,
                           facecolor="white", edgecolor="none", zorder=10))
    for x in [left_wave, right_wave]:
        fig.add_artist(Line2D(x, y, transform=fig.transFigure,
                              color=TEXT, linewidth=0.9, zorder=11))


def verify_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text)):
        if not item.get_visible() or not item.get_text().strip():
            continue
        bounds = item.get_window_extent(renderer)
        assert bounds.x0 >= -2 and bounds.y0 >= -2, item.get_text()
        assert bounds.x1 <= fig.bbox.width + 2, item.get_text()
        assert bounds.y1 <= fig.bbox.height + 2, item.get_text()


def draw(output: Path, scope: str, counts: dict) -> dict:
    assert counts["MN"] == 1 and counts["KR"] == 514
    assert all(v < LOW_RANGE[1] for k, v in counts.items() if k != "KR")
    assert HIGH_RANGE[0] < counts["KR"] < HIGH_RANGE[1]
    foreign = sum(counts[k] for k in ["CN", "JP", "MN"])
    assert foreign < LOW_RANGE[1]

    fig = plt.figure(figsize=(10.2, 5.9), dpi=180, facecolor="white")
    left, bottom, total_width, height, gap = 0.178, 0.235, 0.759, 0.669, 0.026
    total_units = (LOW_RANGE[1] - LOW_RANGE[0]) + (HIGH_RANGE[1] - HIGH_RANGE[0])
    unit_width = (total_width - gap) / total_units
    left_width = unit_width * (LOW_RANGE[1] - LOW_RANGE[0])
    right_width = unit_width * (HIGH_RANGE[1] - HIGH_RANGE[0])
    right_left = left + left_width + gap
    ax1 = fig.add_axes([left, bottom, left_width, height])
    ax2 = fig.add_axes([right_left, bottom, right_width, height], sharey=ax1)
    values = [counts[k] for k in COUNTRIES]
    display_values = values.copy()
    display_values[3] = max(values[3], MINIMUM_MONGOLIA_WIDTH)
    colors = [RED, GRAY, GRAY, GRAY]
    for axis in [ax1, ax2]:
        axis.barh(range(4), display_values, height=0.48, color=colors, edgecolor="none", zorder=3)
        axis.set_ylim(3.55, -0.75)
        axis.tick_params(axis="x", colors="#555555", labelsize=12, length=4, width=0.7, pad=7)
        axis.tick_params(axis="y", length=0, pad=15)
        for name in ["top", "right", "left"]:
            axis.spines[name].set_visible(False)
        axis.spines["bottom"].set_color("#888888")
        axis.spines["bottom"].set_linewidth(0.7)
        axis.set_facecolor("white")
    ax1.set_xlim(*LOW_RANGE)
    ax2.set_xlim(*HIGH_RANGE)
    ax1.set_xticks([0, 50, 100, 150])
    ax2.set_xticks([500, 525])
    ax1.set_yticks(range(4), COUNTRY_NAMES, fontsize=17, color=TEXT)
    ax2.tick_params(axis="y", labelleft=False)
    ax1.axvline(0, color="#C4C4C4", linewidth=0.8, zorder=2)

    # Direct labels retain the original values, including Mongolia's true count.
    for i, value in enumerate(values):
        axis = ax2 if value > LOW_RANGE[1] else ax1
        x = display_values[i] + 3.5
        axis.text(x, i, "1*" if i == 3 else str(value),
                  va="center", ha="left", fontsize=16,
                  fontweight="bold" if i == 0 else "normal",
                  color=RED if i == 0 else TEXT)

    # Combined foreign count is a reference, not a fifth country bar.
    ax1.vlines(foreign, -0.43, 3.4, color="#767676", linewidth=1.0,
               linestyle=(0, (4, 4)), zorder=4)
    ax1.text(foreign, -0.59, f"All foreign = {foreign}", ha="right", va="center",
              fontsize=12, color="#5D5D5D")

    fig.canvas.draw()
    # Equal physical unit spacing in the two panels is essential for this break.
    pixels1 = ax1.transData.transform((1, 0))[0] - ax1.transData.transform((0, 0))[0]
    pixels2 = ax2.transData.transform((491, 0))[0] - ax2.transData.transform((490, 0))[0]
    assert np.isclose(pixels1, pixels2, rtol=1e-12)
    to_fig = fig.transFigure.inverted()
    y0 = to_fig.transform(ax1.transData.transform((0, -0.24)))[1]
    y1 = to_fig.transform(ax1.transData.transform((0, 0.24)))[1]
    wave_gap(fig, left + left_width, right_left, min(y0, y1), max(y0, y1))
    wave_gap(fig, left + left_width, right_left, bottom - 0.009, bottom + 0.009)

    fig.text(0.558, 0.128, "Number of PM2.5 input series", ha="center", va="center",
              fontsize=13, color=TEXT)
    fig.text(0.178, 0.042, "* Mongolia bar enlarged for visibility (actual = 1).",
              ha="left", fontsize=10.5, color="#606060")
    verify_text(fig)

    stem = f"input_series_{scope}_red_gray_broken_axis"
    exports = []
    for ext in ["png", "svg", "pdf"]:
        target = output / f"{stem}.{ext}"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite an existing figure: {target}")
        fig.savefig(target, dpi=260, facecolor="white")
        exports.append({"file": target.name, "bytes": target.stat().st_size, "sha256": digest(target)})
    plt.close(fig)
    return {
        "scope": scope, "actual_counts": counts,
        "foreign_total": foreign,
        "display_bar_widths_in_axis_units": dict(zip(COUNTRIES, display_values)),
        "visible_x_ranges": [LOW_RANGE, HIGH_RANGE],
        "omitted_x_range": [LOW_RANGE[1], HIGH_RANGE[0]],
        "equal_pixels_per_visible_unit": bool(np.isclose(pixels1, pixels2)),
        "mongolia_minimum_display_width_disclosed": True,
        "actual_data_unchanged": True,
        "countries_use_red_then_identical_gray": True,
        "exports": exports,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    parent = Path(__file__).resolve().parents[1]
    output = args.output or parent / ("Run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output = output.resolve()
    result_dir = output / "Results"
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output / "Audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42})
    source_hashes = {str(path): digest(path) for path in [CONFIG_SOURCE, LEGACY_CONFIG_SOURCE]}
    config = json.loads(CONFIG_SOURCE.read_text(encoding="utf-8-sig"))
    legacy = json.loads(LEGACY_CONFIG_SOURCE.read_text(encoding="utf-8-sig"))
    condition = config["conditions"]["F600_W24"]
    keys = condition["station_keys"]
    assert len(keys) == 600 and len(set(keys)) == 600
    assert "KR:111121" not in keys
    assert keys == legacy["conditions"]["F"]["station_keys"]
    assert condition["factors"] == {"china_east": False, "japan_east": False, "west_selection": "old24"}
    counts = pd.Series([key.split(":", 1)[0] for key in keys]).value_counts().to_dict()
    assert counts == {"KR": 514, "CN": 61, "JP": 24, "MN": 1}
    assert sum(counts[c] for c in ["CN", "JP", "MN"]) == 86
    scope = "pre_east_F600"
    counts_table = pd.DataFrame([{"scope": scope, "country": c, "label": label,
                                  "external_series": counts[c]}
                                 for c, label in zip(COUNTRIES, COUNTRY_NAMES)])
    reports = [draw(result_dir, scope, counts)]
    counts_table.to_csv(result_dir / "actual_country_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"station_key": keys, "country": [key.split(":", 1)[0] for key in keys]}).to_csv(
        result_dir / "F600_station_keys.csv", index=False, encoding="utf-8-sig")
    evidence_dir = output / "Evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for path, filename in [(CONFIG_SOURCE, "neighbor_experiments.json"),
                           (LEGACY_CONFIG_SOURCE, "legacy_regional_experiments.json")]:
        assert digest(path) == source_hashes[str(path)]
        evidence_copy = evidence_dir / filename
        if path.resolve() != evidence_copy.resolve():
            shutil.copy2(path, evidence_copy)
    audit = {"source_sha256": source_hashes, "figures": reports,
             "target_seoul_junggu_excluded": True,
             "scope_note": "Original F600_W24 before China East16 and Japan East18 additions; old Japan West24 retained.",
             "source_files_unchanged": True,
             "factors": condition["factors"], "external_total": 600, "foreign_total": 86,
             "nodes_including_seoul_target": 601, "station_order_matches_legacy_F": True,
             "country_display_names": COUNTRY_NAMES}
    (audit_dir / "chart_validation.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "figures": len(reports),
                      "exports": 3, "counts_and_equal_visible_unit_scale_verified": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()
