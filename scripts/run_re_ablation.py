"""
run_re_ablation.py
──────────────────
Wrapper nhỏ để chạy RE ablation từ command line.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation_re import main


if __name__ == "__main__":
    main()
