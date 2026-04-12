"""
Build toàn bộ pipeline offline và lưu index ra disk.
Chạy một lần, sau đó dùng --load-index để search nhanh.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import INDEX_DIR, NewsSearchSystem


def parse_args():
    p = argparse.ArgumentParser(description="Build offline FAISS + KG index")
    p.add_argument(
        "--data",
        "-d",
        type=str,
        default=str(ROOT / "data" / "vnexpress_articles.csv"),
        help="Đường dẫn dataset CSV/JSON",
    )
    p.add_argument(
        "--index-dir", type=str, default=str(INDEX_DIR), help="Thư mục output index"
    )
    p.add_argument(
        "--ner-model-dir",
        type=str,
        default=None,
        help="Thư mục PhoBERT NER checkpoint (mặc định: data/ner_model)",
    )
    p.add_argument(
        "--reranker-dir",
        type=str,
        default=None,
        help="Thư mục cross-encoder checkpoint (mặc định: data/reranker_model)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    system = NewsSearchSystem(
        data_path=args.data,
        index_dir=args.index_dir,
        ner_model_dir=args.ner_model_dir,
        reranker_model_dir=args.reranker_dir,
        use_faiss=True,
    )
    system.build()
    system.save_index(args.index_dir)
    print(f"\n✅ Index build xong tại: {args.index_dir}")
    print("   Chạy tìm kiếm: python main.py --load-index")


if __name__ == "__main__":
    main()
