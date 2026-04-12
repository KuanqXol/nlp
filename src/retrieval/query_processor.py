"""

Xử lý query tiếng Việt: NFC normalize → NER → entity linking → keyword extract.

Input:  "chiến tranh nga ukraine 2024"
Output: {
    'original':    'chiến tranh nga ukraine 2024',
    'normalized':  'chiến tranh nga ukraine 2024',
    'entities':    [{'canonical': 'Nga', 'type': 'LOC'}, ...],
    'keywords':    ['chiến tranh', 'nga', 'ukraine', '2024'],
    'topic':       'thế giới',
    'year_filter': '2024',
    'intent':      'news_search',
}
"""

import re
import unicodedata
from typing import Dict, List, Optional


# ── Stopwords ─────────────────────────────────────────────────────────────────

VIETNAMESE_STOPWORDS = {
    "là",
    "và",
    "của",
    "trong",
    "có",
    "các",
    "được",
    "với",
    "để",
    "đã",
    "tại",
    "từ",
    "về",
    "cho",
    "khi",
    "như",
    "thì",
    "mà",
    "vào",
    "ra",
    "đến",
    "theo",
    "này",
    "đó",
    "một",
    "những",
    "cũng",
    "hay",
    "hoặc",
    "bởi",
    "vì",
    "nên",
    "nếu",
    "thế",
    "sẽ",
    "đang",
    "đây",
    "còn",
    "qua",
    "lại",
    "sau",
    "trước",
    "trên",
    "dưới",
    "nhiều",
    "ít",
    "tìm",
    "kiếm",
    "thông tin",
    "bài báo",
    "tin tức",
    "cho tôi",
    "hãy",
    "xem",
}

TOPIC_KEYWORDS = {
    "chính trị": [
        "chính trị",
        "quốc hội",
        "chính phủ",
        "bầu cử",
        "tổng thống",
        "thủ tướng",
        "luật",
    ],
    "thế giới": [
        "thế giới",
        "quốc tế",
        "chiến tranh",
        "xung đột",
        "ngoại giao",
        "NATO",
        "LHQ",
    ],
    "y tế": [
        "y tế",
        "dịch bệnh",
        "covid",
        "vaccine",
        "bệnh viện",
        "sức khỏe",
        "WHO",
        "dịch",
    ],
    "giáo dục": [
        "giáo dục",
        "trường học",
        "đại học",
        "học sinh",
        "sinh viên",
        "đào tạo",
    ],
    "kinh tế": [
        "kinh tế",
        "gdp",
        "tăng trưởng",
        "ngân hàng",
        "đầu tư",
        "doanh nghiệp",
        "thị trường",
    ],
    "công nghệ": [
        "công nghệ",
        "ai",
        "trí tuệ nhân tạo",
        "5g",
        "phần mềm",
        "startup",
        "digital",
    ],
}


def _normalize(text: str) -> str:
    """NFC normalize + chuẩn hóa khoảng trắng."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _extract_year(text: str) -> Optional[str]:
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return m.group() if m else None


# Từ khóa chỉ ý định tìm tin tức theo thời gian (không cần expand entity)
_TEMPORAL_PATTERNS = re.compile(
    r"\b(mới nhất|gần đây|hôm nay|tuần này|tháng này|năm nay"
    r"|latest|recent|today|breaking|vừa|vừa qua|mới đây"
    r"|trong \d+ (ngày|tuần|tháng|năm) (qua|gần đây|vừa rồi))\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def _detect_intent(text: str, year_filter: Optional[str]) -> str:
    """Phân loại intent của query.

    Returns:
        'temporal_query' : query tập trung vào thời gian → không nên expand entity
        'news_search'    : tìm kiếm tin tức thông thường
    """
    if _TEMPORAL_PATTERNS.search(text):
        return "temporal_query"
    # Có năm cụ thể trong query nhưng không có entity → temporal
    if year_filter and len(text.split()) <= 3:
        return "temporal_query"
    return "news_search"


def _detect_topic(text: str) -> Optional[str]:
    tl = text.lower()
    scores = {t: sum(1 for kw in kws if kw in tl) for t, kws in TOPIC_KEYWORDS.items()}
    best = {t: s for t, s in scores.items() if s > 0}
    return max(best, key=best.get) if best else None


def _extract_keywords(text: str) -> List[str]:
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]


# ── QueryProcessor ────────────────────────────────────────────────────────────


class QueryProcessor:
    """
    Xử lý query tiếng Việt: normalize → NER → entity linking → intent detection.

    Ví dụ:
        proc = QueryProcessor(ner_engine, entity_linker)
        result = proc.process("WHO cảnh báo dịch cúm H5N1 tại Việt Nam")
    """

    def __init__(self, ner_engine, entity_linker):
        self.ner = ner_engine
        self.linker = entity_linker

    def process(self, query: str) -> Dict:
        if not query or not query.strip():
            return self._empty(query)

        normalized = _normalize(query)
        raw_entities = self.ner.extract(normalized)
        linked = self.linker.link_entities(raw_entities)
        keywords = _extract_keywords(normalized)
        topic = _detect_topic(normalized)
        year = _extract_year(normalized)
        intent = _detect_intent(normalized, year)

        return {
            "original": query,
            "normalized": normalized,
            "entities": linked,
            "keywords": keywords,
            "topic": topic,
            "year_filter": year,
            "intent": intent,
        }

    def _empty(self, query: str) -> Dict:
        return {
            "original": query,
            "normalized": "",
            "entities": [],
            "keywords": [],
            "topic": None,
            "year_filter": None,
            "intent": "news_search",
        }

    def get_query_entity_names(self, processed: Dict) -> List[str]:
        return [
            e["canonical"] for e in processed.get("entities", []) if e.get("canonical")
        ]

    def build_search_text(self, processed: Dict) -> str:
        parts = list(
            set(processed.get("keywords", []) + self.get_query_entity_names(processed))
        )
        return " ".join(parts)

    def format_for_display(self, processed: Dict) -> str:
        lines = [
            f"Query        : {processed['original']}",
            f"Keywords     : {', '.join(processed['keywords'])}",
            f"Chủ đề       : {processed['topic'] or 'không xác định'}",
        ]
        if processed["year_filter"]:
            lines.append(f"Năm          : {processed['year_filter']}")
        if processed["entities"]:
            lines.append("Entity:")
            for e in processed["entities"]:
                lines.append(f"  - {e['canonical']:20s} ({e['type']})")
        return "\n".join(lines)
