"""A light future-work concept, without fitted results or country assignments."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.4, 3.0), dpi=180, facecolor="white")
    ink = "#242424"
    fig.text(0.115, 0.70, "Season", ha="center", va="center", fontsize=22, color=ink)
    fig.text(0.475, 0.70, "Select foreign regions", ha="center", va="center", fontsize=22, color="#B82F34")
    fig.text(0.865, 0.70, "PM2.5 forecast", ha="center", va="center", fontsize=22, color=ink)
    fig.text(0.865, 0.20, "Domestic data", ha="center", va="center", fontsize=20, color=ink)
    arrows = [((0.202, 0.70), (0.295, 0.70)),
              ((0.660, 0.70), (0.755, 0.70)),
              ((0.865, 0.30), (0.865, 0.56))]
    for start, end in arrows:
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=14,
                                       linewidth=1.1, color="#777777"))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for item in fig.texts:
        bounds = item.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    exports = []
    for ext in ["png", "svg", "pdf"]:
        path = output / ("seasonal_input_selection_concept." + ext)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=260, facecolor="white")
        exports.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    (output / "concept_manifest.json").write_text(json.dumps({
        "purpose": "Futurework direction, not experimental evidence",
        "no_season_to_country_assignments": True, "no_performance_claims": True,
        "domestic_inputs_remain_available": True, "exports": exports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "exports": 3}))


if __name__ == "__main__":
    main()
