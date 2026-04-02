"""
Build toàn bộ pipeline offline và lưu ra data/index/.

Chạy:
    python scripts/build_index.py
    python scripts/build_index.py --data data/vnexpress_articles.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import INDEX_DIR, NewsSearchSystem


def parse_args():
    parser = argparse.ArgumentParser(description="Offline index builder")
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        default=str(ROOT / "data" / "vnexpress_articles.csv"),
        help="Đường dẫn dataset JSON/CSV",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=str(INDEX_DIR),
        help="Thư mục output index",
    )
    parser.add_argument(
        "--use-model",
        action="store_true",
        help="Dùng model NER/embedding thay vì fallback nhẹ",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    system = NewsSearchSystem(
        data_path=args.data,
        use_model=args.use_model,
        use_faiss=True,
        use_llm=False,
    )
    system.build()
    system.save_index(args.index_dir)


if __name__ == "__main__":
    main()
