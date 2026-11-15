"""Uncluttered PM2.5 ranges: 1st–99th percentiles, IQR, and one maximum.

All statistics use the full selected hourly observations, in original units.
Only the display axis uses log1p so zero and very large values remain visible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RED = "#C73535"
REGION_ORDER = ["KR_CAPITAL", "KR_GANGWON", "KR_CHUNGCHEONG", "KR_HONAM", "KR_DAEGYEONG", "KR_DONGNAM", "KR_JEJU",
                "CN_SHANDONG", "CN_BTH", "CN_LIAONING", "CN_JILIN", "CN_HEILONGJIANG", "CN_INNER_MONGOLIA", "CN_OTHER",
                "JP_WEST", "JP_EAST", "MN"]
REGION_NAMES = {
    "KR_CAPITAL": "KR: Capital", "KR_GANGWON": "KR: Gangwon", "KR_CHUNGCHEONG": "KR: Chungcheong",
    "KR_HONAM": "KR: Honam", "KR_DAEGYEONG": "KR: Daegu / Gyeongbuk", "KR_DONGNAM": "KR: Southeast", "KR_JEJU": "KR: Jeju",
    "CN_SHANDONG": "CN: Shandong", "CN_BTH": "CN: Beijing / Tianjin / Hebei", "CN_LIAONING": "CN: Liaoning",
    "CN_JILIN": "CN: Jilin", "CN_HEILONGJIANG": "CN: Heilongjiang", "CN_INNER_MONGOLIA": "CN: Inner Mongolia",
    "CN_OTHER": "CN: Other / East", "JP_WEST": "JP: West", "JP_EAST": "JP: East", "MN": "MN: Ulaanbaatar",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_text(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -2 and box.y0 >= -2, artist.get_text()
        assert box.x1 <= fig.bbox.width + 2 and box.y1 <= fig.bbox.height + 2, artist.get_text()


def draw(stats, keys, labels, folder, name, *, all_regions=False):
    n = len(keys)
    height = max(5.0, n * 0.46 + 2.0)
    fig = plt.figure(figsize=(11.6, height), dpi=180, facecolor="white")
    ax = fig.add_axes([0.275, 0.18, 0.685, 0.69])
    shown = []
    for row, key in enumerate(keys):
        item = stats[key]
        p01, q1, q3, p99, maximum = [item[field] for field in ["p01", "q1", "q3", "p99", "max"]]
        assert 0 <= p01 <= q1 <= q3 <= p99 <= maximum
        ax.hlines(row, p01, p99, color="black", linewidth=1.0, zorder=3)
        ax.vlines([p01, p99], row - 0.105, row + 0.105, color="black", linewidth=1.0, zorder=3)
        ax.hlines(row, q1, q3, color="black", linewidth=5.5, zorder=4)
        ax.scatter([maximum], [row], s=52, facecolors="white", edgecolors=RED, linewidths=1.35, zorder=5)
        ax.annotate(f"{maximum:,.0f}", (maximum, row), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=12, color="#202020")
        shown.append({"id": key, "p01": p01, "q1": q1, "q3": q3, "p99": p99, "maximum": maximum})
    ax.set_yticks(range(n), labels)
    ax.set_ylim(n - 0.45, -0.65)
    ax.set_xscale("function", functions=(np.log1p, np.expm1))
    ax.set_xlim(0, 19000)
    ticks = [0, 10, 30, 100, 300, 1000, 3000, 10000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("PM2.5 (µg/m³, log1p scale)", fontsize=12, labelpad=13)
    ax.tick_params(axis="y", length=0, labelsize=11.5 if all_regions else 12.5, pad=15)
    ax.tick_params(axis="x", labelsize=11, width=0.65, length=4, pad=7)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.65)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#777777")
    ax.spines["bottom"].set_linewidth(0.7)
    legend = [
        Line2D([], [], color="black", linewidth=1, marker="|", markersize=7, label="1st–99th percentiles"),
        Line2D([], [], color="black", linewidth=5.5, label="Middle 50%"),
        Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=RED,
               markersize=7, markeredgewidth=1.25, label="Maximum"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.275, 0.982), ncol=3,
               frameon=False, fontsize=11, handlelength=1.7, columnspacing=1.9)
    fig.text(0.275, 0.038, "2021–2024 hourly observations", fontsize=10.5, color="#555555")
    check_text(fig)
    files = []
    for extension in ["png", "svg", "pdf"]:
        path = folder / (name + "." + extension)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=240, facecolor="white")
        files.append({"file": path.name, "sha256": sha(path)})
    plt.close(fig)
    return {"file_stem": name, "groups": shown, "red_points": n, "red_point_means": "maximum, not confirmed error",
            "full_data_statistics": True, "display_scale": "log1p", "footer": "2021–2024 hourly observations", "files": files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    region_file = ROOT / "Results/region_box_stats.json"
    station_file = ROOT / "Results/station_box_stats.json"
    regions = json.loads(region_file.read_text(encoding="utf-8"))
    stations = json.loads(station_file.read_text(encoding="utf-8"))
    assert set(regions) == set(REGION_ORDER)
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42})
    extreme_keys = [key for key in REGION_ORDER if regions[key]["max"] > 500]
    reports = [draw(regions, extreme_keys, [REGION_NAMES[k] for k in extreme_keys], output, "pm25_percentile_range_regions")]
    selected = ["CN:3578A", "CN:1751A", "KR:437421", "MN:UB1010001"]
    reports.append(draw(stations, selected, ["Tangshan (3578A)", "Anshan (1751A)", "Uljin (437421)", "Ulaanbaatar"],
                        output, "pm25_percentile_range_representative_stations"))
    reports.append(draw(regions, REGION_ORDER, [REGION_NAMES[k] for k in REGION_ORDER], output,
                        "pm25_percentile_range_all_17_regions", all_regions=True))
    report = {"source_profile_sha256": {str(path): sha(path) for path in [region_file, station_file]},
              "figures": reports, "source_data_unchanged": True}
    (output / "range_figure_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"figures": len(reports), "main_figure_maximum_points": len(extreme_keys), "output": str(output)}))


if __name__ == "__main__":
    main()
