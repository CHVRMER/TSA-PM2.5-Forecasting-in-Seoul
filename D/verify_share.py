"""Verify pinned runtime files; optionally verify the separately supplied Dataset."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
MANIFESTS = [
    "Model/DCRNN/VastAI_M1_M4T_H48_20261111/References/upload_manifest.json",
    "Model/DCRNN/Graph_WaveNet/VastAI_M1_M4T_H48_20261112/References/upload_manifest.json",
]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--dataset", type=Path, help="Directory containing the supplied Dataset subfolders")
    args = parser.parse_args()
    if args.code_only and args.dataset:
        parser.error("Choose --code-only or --dataset.")
    code, data, errors = {}, {}, []
    for relative in MANIFESTS:
        document = json.loads((RUNTIME / relative).read_text(encoding="utf-8-sig"))
        for item in document["files"]:
            selected = data if item["path"].startswith("Dataset/") else code
            old = selected.setdefault(item["path"], item)
            if old["sha256"] != item["sha256"]:
                errors.append({"path": item["path"], "reason": "incompatible manifests"})
    for relative, item in code.items():
        path = RUNTIME / relative
        if not path.is_file() or sha(path) != item["sha256"]:
            errors.append({"path": relative, "reason": "missing or changed code"})
    check_data = bool(args.dataset) or not args.code_only
    if check_data:
        dataset = args.dataset.resolve() if args.dataset else RUNTIME / "Dataset"
        for relative, item in data.items():
            path = dataset / Path(relative).relative_to("Dataset")
            if not path.is_file() or sha(path) != item["sha256"]:
                errors.append({"path": relative, "reason": "missing or changed external Dataset file"})
    report = {"code_files": len(code), "external_dataset_files": len(data),
              "dataset_checked": check_data, "errors": errors, "passed": not errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
