#!/usr/bin/env bash
# One command: create/attach tmux and resume the newest full experiment.
set -Eeuo pipefail
task_bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_system_python="${TSA_SYSTEM_PYTHON:-python3}"
exec "$task_system_python" "$task_bundle_dir/Codes/full_run_control.py" start "$@"
