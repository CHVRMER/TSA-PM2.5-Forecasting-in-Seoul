#!/usr/bin/env bash
set -Eeuo pipefail
task_bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_project_root="${TSA_PROJECT_ROOT:-/workspace/TSA}"
task_system_python="${TSA_SYSTEM_PYTHON:-python3}"
task_python_tag="$("$task_system_python" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
task_env_root="${TSA_GWN48_ENV_ROOT:-$task_project_root/Environment/GraphWaveNet_VastAI_H48}"
task_python="${TSA_GWN48_PYTHON:-$task_env_root/venv_$task_python_tag/bin/python}"
if [[ ! -x "$task_python" ]]; then
  printf 'Run setup_vast.sh first or set TSA_GWN48_PYTHON.\n' >&2
  exit 2
fi
exec "$task_python" "$task_bundle_dir/Codes/run_checks.py" --project-root "$task_project_root" "$@"
