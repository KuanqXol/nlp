"""
Module: data_loader.py
Chức năng: Đọc và chuẩn hóa dữ liệu tin tức tiếng Việt từ file JSON/CSV.

Pipeline:
  JSON/CSV → load → validate → normalize text → dedup/filter → trả về document
"""

import csv
import json
import re
from typing import Dict, List, Optional, Tuple
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


def parse_vn_date(text: str) -> str:
    """
    Parse date kiểu VnExpress:
      "Thứ sáu, 31/7/2020, 18:15 (GMT+7)" -> "2020-07-31"
    """
    text = normalize_text(text)
    if not text:
        return ""

    iso_match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if iso_match:
        year, month, day = iso_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    vn_match = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text)
    if vn_match:
        day, month, year = vn_match.groups()
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"

    return text


def strip_author(text: str) -> str:
    """
    Cắt dòng tên tác giả/byline ở cuối bài nếu có.
    """
    if not text:
        return ""

    lines = [
        re.sub(r"\s+", " ", line).strip(" \t-*")
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    if len(lines) < 2:
        return text.strip()

    tail = lines[-1]
    tokens = tail.split()
    looks_like_author = (
        len(tail) <= 60
        and 1 <= len(tokens) <= 6
        and not re.search(r"[.!?]$", tail)
        and all(
            re.fullmatch(r"[A-ZÀ-ỸĐ][A-Za-zÀ-ỹĐđ.-]*", token) or token.isupper()
            for token in tokens
        )
    )
    looks_like_byline = bool(
        re.fullmatch(r"(?:Theo|theo)\s+[A-ZÀ-ỸĐ].{1,50}", tail)
    )

    if looks_like_author or looks_like_byline:
        lines = lines[:-1]

    return "\n".join(lines).strip()


def viet_ratio(text: str) -> float:
    """
    Tỉ lệ ký tự tiếng Việt có dấu / tổng số ký tự chữ.
    """
    if not text:
        return 0.0

    letters = re.findall(r"[A-Za-zÀ-ỹĐđ]", text)
    if not letters:
        return 0.0

    viet_letters = re.findall(
        r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
        r"óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        r"ÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
        r"ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ]",
        text,
    )
    return len(viet_letters) / len(letters)


def split_sentences(text: str) -> List[str]:
    """
    Tách văn bản thành các câu đơn giản,
    dựa trên dấu chấm câu phổ biến tiếng Việt.
    """
    # Tách theo dấu chấm, chấm hỏi, chấm than
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _guess_source(url: str) -> str:
    if "vnexpress.net" in (url or "").lower():
        return "VnExpress"
    return ""


# ── Document schema ─────────────────────────────────────────────────────────


def _build_document(raw: Dict) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Tạo document chuẩn từ raw JSON.
    Trả về (document, reason). reason != None nếu bị loại.
    """
    title = normalize_text(raw.get("title", ""))
    content_raw = raw.get("content") or raw.get("text") or ""
    content = normalize_text(strip_author(content_raw))

    if not title:
        return None, "missing_title"
    if not content:
        return None, "missing_content"

    doc = {
        "id": raw.get("id", ""),
        "title": title,
        "content": content,
        "date": parse_vn_date(str(raw.get("date", ""))),
        "source": normalize_text(raw.get("source", "") or _guess_source(raw.get("url", ""))),
        "url": normalize_text(raw.get("url", "")),
        "category": normalize_text(raw.get("category", "")).lower(),
        # Nối title + content để embedding và NER trên toàn bộ văn bản
        "full_text": normalize_text(title + ". " + content),
    }

    if viet_ratio(doc["full_text"]) < 0.05:
        return None, "lang_filter"

    # Tạo ID tự động nếu không có
    if not doc["id"]:
        doc["id"] = f"doc_{abs(hash(doc['url'] or doc['title'])) % 10**8}"

    return doc, None


def create_document(raw: Dict) -> Optional[Dict]:
    """
    Backward-compatible helper.
    """
    doc, _ = _build_document(raw)
    return doc


# ── Loader chính ─────────────────────────────────────────────────────────────


class NewsDataLoader:
    """
    Class đọc và quản lý dataset tin tức tiếng Việt.

    Ví dụ sử dụng:
        loader = NewsDataLoader("data/vnexpress_articles.csv")
        docs = loader.load()
        print(f"Đã load {len(docs)} bài báo")
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.documents: List[Dict] = []
        self.last_load_stats: Dict[str, int] = {}

    def load(self) -> List[Dict]:
        """Đọc file JSON/CSV và trả về danh sách document đã chuẩn hóa."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")

        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            return self.load_csv()
        if suffix == ".json":
            return self.load_json()
        raise ValueError(f"Định dạng không hỗ trợ: {self.data_path.suffix}")

    def load_json(self) -> List[Dict]:
        """Đọc file JSON array và chuẩn hóa tài liệu."""
        print(f"[DataLoader] Đang đọc dữ liệu từ: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError("Dataset phải là một JSON array (danh sách bài báo).")

        loaded = []
        seen_urls = set()
        skipped_invalid = 0
        skipped_lang = 0
        skipped_dedup = 0

        for item in raw_data:
            doc, reason = _build_document(item)
            if not doc:
                if reason == "lang_filter":
                    skipped_lang += 1
                else:
                    skipped_invalid += 1
                continue

            if doc["url"] and doc["url"] in seen_urls:
                skipped_dedup += 1
                continue
            if doc["url"]:
                seen_urls.add(doc["url"])
            loaded.append(doc)

        self.documents = loaded
        self.last_load_stats = {
            "total_rows": len(raw_data),
            "loaded": len(loaded),
            "skipped_invalid": skipped_invalid,
            "skipped_lang": skipped_lang,
            "skipped_dedup": skipped_dedup,
        }
        print(f"[DataLoader] Đã load thành công {len(loaded)}/{len(raw_data)} bài báo.")
        return self.documents

    def load_csv(self, chunk_size: int = 5000) -> List[Dict]:
        """
        Đọc CSV theo streaming chunk để tránh giữ toàn bộ raw rows trong RAM.
        """
        print(
            f"[DataLoader] Đang stream CSV từ: {self.data_path} (chunk_size={chunk_size})"
        )
        loaded: List[Dict] = []
        seen_urls = set()
        total_rows = 0
        skipped_invalid = 0
        skipped_lang = 0
        skipped_dedup = 0

        def _process_chunk(rows: List[Dict]):
            nonlocal skipped_invalid, skipped_lang, skipped_dedup
            for row in rows:
                doc, reason = _build_document(row)
                if not doc:
                    if reason == "lang_filter":
                        skipped_lang += 1
                    else:
                        skipped_invalid += 1
                    continue
                if doc["url"] and doc["url"] in seen_urls:
                    skipped_dedup += 1
                    continue
                if doc["url"]:
                    seen_urls.add(doc["url"])
                loaded.append(doc)

        with open(self.data_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            batch: List[Dict] = []
            for row in reader:
                total_rows += 1
                batch.append(row)
                if len(batch) >= chunk_size:
                    _process_chunk(batch)
                    print(
                        f"  [CSV] {total_rows:,} rows -> {len(loaded):,} docs "
                        f"(dedup={skipped_dedup:,}, lang={skipped_lang:,}, invalid={skipped_invalid:,})"
                    )
                    batch = []

            if batch:
                _process_chunk(batch)

        self.documents = loaded
        self.last_load_stats = {
            "total_rows": total_rows,
            "loaded": len(loaded),
            "skipped_invalid": skipped_invalid,
            "skipped_lang": skipped_lang,
            "skipped_dedup": skipped_dedup,
        }
        print(
            f"[DataLoader] CSV load xong: {len(loaded):,}/{total_rows:,} docs | "
            f"dedup={skipped_dedup:,}, lang={skipped_lang:,}, invalid={skipped_invalid:,}"
        )
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
            "load_stats": dict(self.last_load_stats),
        }


# ── Demo standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = NewsDataLoader("../data/vnexpress_articles.csv")
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
