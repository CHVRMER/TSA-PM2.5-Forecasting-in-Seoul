#!/usr/bin/env bash
# Foreground pilot; stdout is visible and copied to a new persistent log folder.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
task_bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_project_root="${TSA_PROJECT_ROOT:-/workspace/TSA}"
task_output_root="${TSA_OUTPUT_ROOT:-$task_project_root/Result/GraphWaveNet/VastAI}"
task_system_python="${TSA_SYSTEM_PYTHON:-python3}"
task_python_tag="$("$task_system_python" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
task_env_root="${TSA_GWN_ENV_ROOT:-$task_project_root/Environment/GraphWaveNet_VastAI}"
task_python="${TSA_GWN_PYTHON:-$task_env_root/venv_$task_python_tag/bin/python}"
if [[ ! -x "$task_python" ]]; then
  printf 'Run setup_vast.sh first, or set TSA_GWN_PYTHON to the prepared Python executable.\n' >&2
  exit 2
fi
if [[ ! -f "$task_bundle_dir/configs/default.json" || ! -f "$task_bundle_dir/Codes/run_experiments.py" ]]; then
  printf 'Incomplete experiment bundle: expected configs/default.json and Codes/run_experiments.py.\n' >&2
  exit 2
fi
task_stamp="$("$task_python" -c 'from datetime import datetime;from zoneinfo import ZoneInfo;import uuid;print(datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")+"_"+uuid.uuid4().hex[:8])')"
task_launch_dir="$task_output_root/Launch_pilot_$task_stamp"
mkdir -p -- "$task_output_root"
mkdir -- "$task_launch_dir"
printf 'Persistent launch log: %s\n' "$task_launch_dir/console.log"
"$task_python" -u "$task_bundle_dir/Codes/run_experiments.py" \
  --project-root "$task_project_root" \
  --output-root "$task_output_root" \
  --mode pilot \
  --config "$task_bundle_dir/configs/default.json" \
  "$@" 2>&1 | tee "$task_launch_dir/console.log"
printf 'Pilot ended. This script does not stop or delete the paid Pod.\n' | tee -a "$task_launch_dir/console.log"
