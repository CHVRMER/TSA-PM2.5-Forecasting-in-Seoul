"""승인된 Stage 1 실행기. 실행마다 새 출력 폴더를 만들고 기존 파일을 보존한다."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"results/.matplotlib_cache"))
from src.train import train_m1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1_6h.json")
    args = parser.parse_args()
    config = json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    if config["model"] != "M1" or config["horizon"] not in (3, 6, 12, 24):
        raise ValueError("M1 공식 Horizon만 실행할 수 있습니다.")
    run_id = f"M1_h{config['horizon']}_"+datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S_%f")
    result_dir, model_dir = ROOT/"results"/run_id, ROOT/"models"/run_id
    # macOS may use a case-insensitive filesystem: existing Models/ and models/
    # resolve to the same folder. A unique run subfolder avoids any replacement.
    result_dir.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(result_dir/"run.log", encoding="utf-8"), logging.StreamHandler()])
    try:
        result = train_m1(ROOT, config, result_dir, model_dir)
        paths = {"run_id": run_id, "results": str(result_dir.resolve()), "models": str(model_dir.resolve())}
        from src.train import write_json
        write_json(result_dir/"run_paths.json", paths)
        source_files = [ROOT/args.config, *ROOT.joinpath("src").glob("*.py"), Path(__file__)]
        from src.train import sha
        write_json(result_dir/"code_sha256.json", {str(p.relative_to(ROOT)):sha(p) for p in source_files})
        print(json.dumps({**paths, "metrics":result["metrics"]}, ensure_ascii=False, indent=2), flush=True)
    except Exception:
        logging.exception("오류 발생: 추가 작업 없이 중단합니다.")
        raise


if __name__ == "__main__":
    main()
