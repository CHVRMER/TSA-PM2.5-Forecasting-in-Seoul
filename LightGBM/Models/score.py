"""공통 채점 실행기. 이전 구현은 score_legacy_20261108.py에 보존했다."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib_cache'))
from src.common_scoring import main
if __name__=='__main__':main()
