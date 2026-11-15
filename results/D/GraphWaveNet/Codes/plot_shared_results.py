"""Plot copied Graph WaveNet legacy24 metrics without datasets or raw ZIPs.

Default: create a new figures/Run_<KST timestamp>_<UUID> directory.
--output-dir: export to an explicitly chosen empty directory (never overwrite).
The committed figures/ images are a published snapshot; regenerated runs stay
separate. Nothing is fitted and original significance tests are not recomputed.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "legacy24_review" / "Results" / "Tables"
HORIZONS = (3, 6, 12, 24)
CONDITIONS = ("M1", "M2", "M3", "M4", "M3T", "M4T")
MODEL = "GraphWaveNet_distance_fixed_plus_adaptive"
COLORS = {"M1": "#C73535", "M2": "#D48B2C", "M3": "#2E8B57",
          "M4": "#2866B2", "M3T": "#168C91", "M4T": "#7E57A2"}
MARKERS = {"M1": "o", "M2": "^", "M3": "s", "M4": "D", "M3T": "P", "M4T": "X"}
LABELS = {"M1": "M1: Domestic", "M2": "M2: +Nearby China", "M3": "M3: +All China",
          "M4": "M4: +Japan / Mongolia", "M3T": "M3T: +Trajectory summaries", "M4T": "M4T: +Trajectory summaries"}
FILES = {"validation": "validation_GWN_metrics.csv", "test": "test_GWN_metrics.csv"}
STEMS = {"validation": "validation2023_rmse", "test": "test2024preliminary_rmse"}
CAPTIONS = {"validation": "2023 validation", "test": "2024 test (preliminary)"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_metrics(split: str) -> tuple[pd.DataFrame, dict]:
    path = TABLES / FILES[split]
    if not path.is_file():
        raise FileNotFoundError(f"Missing copied results table: {path}")
    before = digest(path)
    data = pd.read_csv(path, encoding="utf-8-sig")
    required = {"split", "horizon", "model", "input_condition", "prediction_kind", "seed", "n", "RMSE"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing result columns in {path.name}: {sorted(required-set(data.columns))}")
    table_rows = len(data)
    if set(data["split"]) != {split} or not set(data["model"]).issubset({MODEL, "persistence", "seasonal_naive"}):
        raise ValueError("Unexpected result split or model in the shared metric table.")
    # The saved table also contains the two four-horizon common baselines.
    # The requested curves compare the six Graph WaveNet input conditions only.
    data = data.loc[data["model"].eq(MODEL)].copy()
    if len(data) != 24 or data.duplicated(["horizon", "input_condition"]).any():
        raise ValueError("Expected one result per condition/horizon: six conditions × four horizons.")
    if set(data["split"]) != {split} or set(data["model"]) != {MODEL}:
        raise ValueError("Result split/model does not match the archived Graph WaveNet experiment.")
    if set(data["prediction_kind"]) != {"seed"} or set(pd.to_numeric(data["seed"])) != {11}:
        raise ValueError("These published figures describe seed11, not seed averages or ensembles.")
    if set(data["horizon"]) != set(HORIZONS) or set(data["input_condition"]) != set(CONDITIONS):
        raise ValueError("Unexpected horizons/input conditions in the legacy24 experiment.")
    if not data.groupby("horizon")["n"].nunique().eq(1).all():
        raise ValueError("Models have different evaluation row counts at the same horizon.")
    values = data.pivot(index="horizon", columns="input_condition", values="RMSE").loc[list(HORIZONS), list(CONDITIONS)]
    if not np.isfinite(values.to_numpy()).all() or not values.gt(0).all().all():
        raise ValueError("RMSE values must be positive and finite.")
    if digest(path) != before:
        raise RuntimeError("The source result table changed while it was read.")
    return values, {"path": str(path.relative_to(ROOT)), "sha256": before,
                    "split": split, "seed": 11, "kind": "seed", "table_rows": table_rows,
                    "plotted_rows": len(data), "baseline_rows_excluded_from_curves": table_rows-len(data),
                    "evaluation_rows_by_horizon": data.groupby("horizon")["n"].first().to_dict(),
                    "raw_RMSE": values.to_dict()}


def choose_output(explicit: Path | None) -> Path:
    if explicit is None:
        stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
        output = ROOT / "figures" / f"Run_{stamp}_{uuid.uuid4().hex[:8]}"
    else:
        output = explicit.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output must be absent or empty; no existing file is overwritten: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def verify_text(fig) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text)):
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.width+2 or box.y1 > fig.bbox.height+2:
            raise AssertionError(f"Text is clipped: {item.get_text()}")


def plot(values: pd.DataFrame, split: str, limits: tuple[float, float], output: Path) -> list[dict]:
    fig = plt.figure(figsize=(12, 6.75), dpi=180, facecolor="white")
    ax = fig.add_axes([0.09, 0.28, 0.875, 0.67])
    for name in CONDITIONS:
        raw = values[name].to_numpy()
        line, = ax.plot(HORIZONS, raw, color=COLORS[name], marker=MARKERS[name],
                        markerfacecolor="white", markeredgewidth=1.15, markersize=6.3,
                        linewidth=1.35, linestyle="--" if name.endswith("T") else "-", label=LABELS[name])
        assert np.array_equal(line.get_xdata(), HORIZONS) and np.array_equal(line.get_ydata(), raw)
    lo, hi = limits
    all_ticks = np.arange(round(lo*10), round(hi*10)+1)/10
    number_ticks = np.arange(round(lo*10), round(hi*10)+1, 5)/10
    ax.set_xlim(2, 25)
    ax.set_xticks(HORIZONS)
    ax.set_ylim(lo, hi)
    ax.set_yticks(number_ticks)
    ax.set_yticks(all_ticks, minor=True)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", which="major", labelsize=11.5, length=3.5, width=.7, pad=5)
    ax.tick_params(axis="y", which="minor", length=1.8, width=.4, color="#AAAAAA", labelleft=False)
    ax.tick_params(axis="x", labelsize=12, width=.7, pad=8)
    ax.set_xlabel("Forecast horizon (hours)", fontsize=13, labelpad=14)
    ax.set_ylabel("RMSE (µg/m³)", fontsize=13, labelpad=14)
    ax.grid(axis="y", which="major", color="#DEDEDE", linewidth=.55)
    ax.grid(axis="y", which="minor", color="#F2F2F2", linewidth=.25)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#999999")
        ax.spines[side].set_linewidth(.7)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.54, .065),
               ncol=3, frameon=False, fontsize=10.3, columnspacing=1.5)
    fig.text(.09, .020, f"Graph WaveNet (legacy RF13), {CAPTIONS[split]}. RMSE across M1–M4T input conditions.",
             fontsize=10.2, color="#555555")
    assert len(fig.axes) == 1 and not ax.patches and ax.get_yscale() == "linear"
    assert np.allclose(np.diff(ax.get_yticks()), .5)
    assert np.allclose(np.diff(np.sort(np.r_[ax.get_yticks(), ax.get_yticks(minor=True)])), .1)
    verify_text(fig)
    records = []
    for ext in ("png", "svg", "pdf"):
        path = output / f"{STEMS[split]}.{ext}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=260, facecolor="white")
        records.append({"file": path.name, "sha256": digest(path)})
    plt.close(fig)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optional absent/empty export directory; existing files are never overwritten.")
    parser.add_argument("--validate-only", action="store_true", help="Check copied metric tables without generating figures.")
    args = parser.parse_args()
    inputs = {split: read_metrics(split) for split in FILES}
    low = np.floor(min(frame.to_numpy().min() for frame, _ in inputs.values())*2)/2
    high = np.ceil(max(frame.to_numpy().max() for frame, _ in inputs.values())*2)/2
    if args.validate_only:
        print(json.dumps({"status": "passed", "tables": 2, "rows": 48, "source": "legacy24", "seed": 11,
                          "shared_y_limits": [low, high]}, ensure_ascii=False))
        return 0
    output = choose_output(args.output_dir)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    # Arial matches prior DCRNN graphics when installed; DejaVu is a portable fallback.
    if any(item.name == "Arial" for item in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Arial"
    figures = []
    for split, (frame, source) in inputs.items():
        figures.append({"split": split, "files": plot(frame, split, (low, high), output)})
        if digest(ROOT/source["path"]) != source["sha256"]:
            raise RuntimeError("A source table was modified during plotting.")
    manifest = {"source_version": "completed legacy24 RF13", "prepared_H48_results": False,
                "seed": 11, "horizons": list(HORIZONS), "input_conditions": list(CONDITIONS),
                "sources": [item[1] for item in inputs.values()], "y_limits": [low, high],
                "numeric_tick_spacing": .5, "minor_tick_spacing": .1, "aspect_ratio": "16:9",
                "colors": COLORS, "markers": MARKERS, "masking": False, "axis_breaks": False,
                "statistical_tests_recomputed": False, "exports": figures}
    (output/"figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(output), "figures": 2, "exports": 6}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
