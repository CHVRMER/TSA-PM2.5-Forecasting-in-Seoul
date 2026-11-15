#!/usr/bin/env bash
# Run this INSIDE an existing Vast.ai PyTorch instance; it never rents or stops a Pod.
set -Eeuo pipefail
export PYTHONUNBUFFERED=1

task_bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task_project_root="${TSA_PROJECT_ROOT:-/workspace/TSA}"
task_system_python="${TSA_SYSTEM_PYTHON:-python3}"

if [[ ! -d "$task_project_root" ]]; then
  printf 'Project folder is missing: %s\nUpload the TSA project first, or set TSA_PROJECT_ROOT.\n' "$task_project_root" >&2
  exit 2
fi
if [[ ! -f "$task_bundle_dir/requirements.txt" ]]; then
  printf 'Missing requirements.txt beside setup_vast.sh. Upload the complete experiment bundle.\n' >&2
  exit 2
fi

# Read-only file-integrity check catches a partial ZIP upload before installing packages.
"$task_system_python" "$task_bundle_dir/Codes/verify_upload.py" --project-root "$task_project_root"

# Do not install or replace the CUDA PyTorch supplied by the official template.
"$task_system_python" - "$task_bundle_dir/requirements.txt" <<'PY'
import pathlib,re,sys
import torch
for raw in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8-sig').splitlines():
    line=raw.split('#',1)[0].strip()
    if re.match(r'(?i)^(torch|torchvision|torchaudio|triton)(?:\s|[<=>!~\[]|$)',line):
        raise SystemExit('requirements.txt must not install/replace the template PyTorch/CUDA stack: '+line)
print('Using template PyTorch:',torch.__version__)
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU is unavailable. Choose/check a compatible Vast.ai PyTorch GPU instance.')
# CUDA availability alone does not prove an RTX5090/sm_120 kernel is supported.
x=torch.randn(2,2,8,8,device='cuda');w=torch.randn(2,2,3,3,device='cuda')
y=torch.nn.functional.conv2d(x,w);z=x.flatten(1)@x.flatten(1).T
torch.cuda.synchronize()
assert torch.isfinite(y).all() and torch.isfinite(z).all()
print('GPU kernel smoke test passed:',torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))
PY

task_python_tag="$("$task_system_python" -c 'import sys;print(f"py{sys.version_info.major}{sys.version_info.minor}")')"
task_env_root="${TSA_DCRNN48_ENV_ROOT:-$task_project_root/Environment/DCRNN_VastAI_H48}"
task_venv_dir="$task_env_root/venv_$task_python_tag"
task_setup_stamp="$("$task_system_python" -c 'from datetime import datetime;from zoneinfo import ZoneInfo;import uuid;print(datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")+"_"+uuid.uuid4().hex[:8])')"
task_setup_dir="$task_env_root/Setup_$task_setup_stamp"
mkdir -p -- "$task_env_root"
mkdir -- "$task_setup_dir"
if [[ ! -x "$task_venv_dir/bin/python" ]]; then
  "$task_system_python" -m venv --system-site-packages "$task_venv_dir"
fi
task_python="$task_venv_dir/bin/python"
task_torch_before="$("$task_python" -c 'import torch;print(torch.__version__)')"
printf 'Experiment bundle: %s\nPersistent project: %s\nPython environment: %s\nSetup logs: %s\n' \
  "$task_bundle_dir" "$task_project_root" "$task_python" "$task_setup_dir"
"$task_python" -m pip install --disable-pip-version-check -r "$task_bundle_dir/requirements.txt" 2>&1 | tee "$task_setup_dir/pip_install.log"
task_torch_after="$("$task_python" -c 'import torch;print(torch.__version__)')"
if [[ "$task_torch_before" != "$task_torch_after" ]]; then
  printf 'PyTorch version changed unexpectedly. Stop and inspect %s before training.\n' "$task_setup_dir/pip_install.log" >&2
  exit 3
fi
"$task_python" - "$task_setup_dir/runtime.json" <<'PY'
import importlib.metadata as md,json,pathlib,platform,sys
import torch
record={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
        'cuda_available':torch.cuda.is_available(),'torch_cuda':torch.version.cuda,
        'packages':{n:md.version(n) for n in ['numpy','pandas','pyarrow','matplotlib']}}
if torch.cuda.is_available():
    record['gpu_name']=torch.cuda.get_device_name(0)
    record['gpu_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
pathlib.Path(sys.argv[1]).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False,indent=2))
PY
printf '\nSetup completed. Next: bash "%s/run_pilot.sh"\n' "$task_bundle_dir"
printf 'No GPU instance was created, stopped, or deleted by this script.\n'
