"""
Module: data_loader.py
Chức năng: Đọc và chuẩn hóa dữ liệu tin tức tiếng Việt từ file JSON.

Pipeline:
  JSON file → load → validate → normalize text → trả về danh sách document
"""

import json
import re
import os
from typing import List, Dict, Optional
from pathlib import Path


# ── Chuẩn hóa text tiếng Việt ──────────────────────────────────────────────


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt:
    - Loại bỏ ký tự thừa, khoảng trắng dư
    - Giữ nguyên dấu câu tiếng Việt
    """
    if not text:
        return ""
    # Xóa ký tự điều khiển và khoảng trắng dư thừa
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def split_sentences(text: str) -> List[str]:
    """
    Tách văn bản thành các câu đơn giản,
    dựa trên dấu chấm câu phổ biến tiếng Việt.
    """
    # Tách theo dấu chấm, chấm hỏi, chấm than
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# ── Document schema ─────────────────────────────────────────────────────────


def create_document(raw: Dict) -> Optional[Dict]:
    """
    Tạo document chuẩn từ raw JSON.
    Trả về None nếu document thiếu field bắt buộc.
    """
    required_fields = ["title", "content"]
    for field in required_fields:
        if field not in raw or not raw[field]:
            print(
                f"[WARN] Bỏ qua document thiếu field: {field} — {raw.get('url', 'unknown')}"
            )
            return None

    doc = {
        "id": raw.get("id", ""),
        "title": normalize_text(raw["title"]),
        "content": normalize_text(raw["content"]),
        "date": raw.get("date", ""),
        "source": raw.get("source", ""),
        "url": raw.get("url", ""),
        "category": raw.get("category", ""),
        # Nối title + content để embedding và NER trên toàn bộ văn bản
        "full_text": normalize_text(raw["title"] + ". " + raw["content"]),
    }

    # Tạo ID tự động nếu không có
    if not doc["id"]:
        doc["id"] = f"doc_{abs(hash(doc['url'] or doc['title'])) % 10**8}"

    return doc


# ── Loader chính ─────────────────────────────────────────────────────────────


class NewsDataLoader:
    """
    Class đọc và quản lý dataset tin tức tiếng Việt.

    Ví dụ sử dụng:
        loader = NewsDataLoader("data/news_dataset.json")
        docs = loader.load()
        print(f"Đã load {len(docs)} bài báo")
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.documents: List[Dict] = []

    def load(self) -> List[Dict]:
        """Đọc file JSON và trả về danh sách document đã chuẩn hóa."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")

        print(f"[DataLoader] Đang đọc dữ liệu từ: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("Dataset phải là một JSON array (danh sách bài báo).")

        loaded = []
        for item in raw_data:
            doc = create_document(item)
            if doc:
                loaded.append(doc)

        self.documents = loaded
        print(f"[DataLoader] Đã load thành công {len(loaded)}/{len(raw_data)} bài báo.")
        return self.documents

    def get_by_category(self, category: str) -> List[Dict]:
        """Lọc bài báo theo chủ đề (chính trị, y tế, kinh tế, ...)."""
        return [d for d in self.documents if d.get("category") == category]

    def get_by_source(self, source: str) -> List[Dict]:
        """Lọc bài báo theo nguồn (VnExpress, Tuổi Trẻ, ...)."""
        return [d for d in self.documents if d.get("source") == source]

    def summary(self) -> Dict:
        """Thống kê tổng quan dataset."""
        categories = {}
        sources = {}
        for doc in self.documents:
            cat = doc.get("category", "unknown")
            src = doc.get("source", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            sources[src] = sources.get(src, 0) + 1

        return {
            "total": len(self.documents),
            "by_category": categories,
            "by_source": sources,
        }


# ── Demo standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = NewsDataLoader("../data/news_dataset.json")
    docs = loader.load()

    print("\n=== THỐNG KÊ DATASET ===")
    stats = loader.summary()
    print(f"Tổng số bài báo: {stats['total']}")
    print("Theo chủ đề:", stats["by_category"])
    print("Theo nguồn:", stats["by_source"])

    print("\n=== VÍ DỤ BÀI BÁO ĐẦU TIÊN ===")
    doc = docs[0]
    print(f"  ID     : {doc['id']}")
    print(f"  Tiêu đề: {doc['title']}")
    print(f"  Nội dung: {doc['content'][:120]}...")
    print(f"  Ngày   : {doc['date']}")
    print(f"  Nguồn  : {doc['source']}")
