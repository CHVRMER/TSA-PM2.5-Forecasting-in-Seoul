"""Rebuild selected shared figures in a fresh directory without overwriting exports."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import shutil
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
BASE = "EDA/D_Contributions/"
JOBS = {
    "validation_curve": ("results/D/DCRNN/presentation2023", "Codes/plot_validation_caption.py", "self"),
    "pre_east_counts": (BASE + "10_input_balance_pre_east600", "Codes/plot_pre_east_balance.py", "root"),
    "current_counts": (BASE + "11_input_balance_dcrnn634", "Codes/plot_input_balance.py", "root"),
    "seasonal_concept": (BASE + "12_seasonal_selection_concept", "Codes/make_concept_figure.py", "results"),
    "outlier_ranges": (BASE + "03_extreme_values", "Codes/plot_percentile_ranges.py", "results"),
    "actual_network": (BASE + "07_dcrnn_actual_network", "Codes/plot_dcrnn_final_network.py", "map"),
    "expansion_stages": (BASE + "06_station_expansion_maps", "Codes/plot_station_stages.py", "map"),
}


def ignore(directory, names):
    return [name for name in names if name == "__pycache__" or Path(name).suffix.lower() in {".png", ".svg", ".pdf", ".pyc"}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=list(JOBS) + ["all"])
    parser.add_argument("--output-root", type=Path, default=REPO / "results/D/reproduced")
    args = parser.parse_args()
    label = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]
    run = args.output_root.resolve() / ("Run_" + label)
    run.mkdir(parents=True, exist_ok=False)
    selected = JOBS if args.job == "all" else {args.job: JOBS[args.job]}
    reports = []
    for job, (relative, script, mode) in selected.items():
        work = run / job
        shutil.copytree(REPO / relative, work, ignore=ignore)
        command = [sys.executable, str(work / script)]
        if mode == "root":
            command += ["--output", str(work)]
        elif mode == "results":
            command += ["--output", str(work / "Results")]
        elif mode == "map":
            command += ["--input-root", str(work), "--output-dir", str(work / "Results")]
        print(f"[{job}] rebuilding in {work}", flush=True)
        subprocess.run(command, check=True, cwd=work)
        exports = sorted(path.relative_to(work).as_posix() for path in work.rglob("*.png"))
        if not exports:
            raise RuntimeError(f"No PNG output: {job}")
        reports.append({"job": job, "png": exports})
    (run / "reproduction_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(run), "jobs": len(reports)}))


if __name__ == "__main__":
    main()
