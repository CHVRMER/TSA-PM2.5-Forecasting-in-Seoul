#!/usr/bin/env bash
# Download ONLY the two frozen project ZIPs; Drive result uploads still use rclone.
set -Eeuo pipefail
task_here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_mount="${TSA_VOLUME_MOUNT:-/workspace}"
task_project="${TSA_PROJECT_ROOT:-$task_mount/TSA}"
task_system_python="${TSA_SYSTEM_PYTHON:-python3}"

# Mandatory read-only check: a bare /workspace directory is not enough.
"$task_system_python" "$task_here/Codes/inspect_vast_storage.py" \
  --mount-path "$task_mount" --project-root "$task_project"

task_transfer_env="$task_mount/.tsa_gdown_env"
if [[ ! -x "$task_transfer_env/bin/python" ]]; then
  "$task_system_python" -m venv "$task_transfer_env"
fi
task_python="$task_transfer_env/bin/python"
if ! "$task_python" -c 'import gdown' >/dev/null 2>&1; then
  "$task_python" -m pip install --disable-pip-version-check gdown
fi
"$task_python" "$task_here/Codes/drive_download_gdown.py" --project-root "$task_project" "$@"
printf '\nDownload/extraction completed. The training and Drive upload code is unchanged.\n'
printf 'Next project location: %s\n' "$task_project"
