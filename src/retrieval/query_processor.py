"""
Module: query_processor.py
Chức năng: Xử lý query của người dùng — NER + chuẩn hóa + phân loại.

Input:  "chiến tranh nga ukraine 2024"
Output: {
    'original': 'chiến tranh nga ukraine 2024',
    'normalized': 'chiến tranh nga ukraine 2024',
    'entities': [
        {'text': 'Nga',     'canonical': 'Nga',     'type': 'LOC'},
        {'text': 'Ukraine', 'canonical': 'Ukraine', 'type': 'LOC'},
    ],
    'keywords': ['chiến tranh', 'nga', 'ukraine', '2024'],
    'intent': 'news_search',
    'time_filter': '2024',
}
"""

import re
from typing import Dict, List, Optional


# ── Stopword tiếng Việt ──────────────────────────────────────────────────────

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

# Keyword chỉ định chủ đề
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


def _extract_year(text: str) -> Optional[str]:
    """Trích xuất năm nếu có trong query."""
    match = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return match.group() if match else None


def _detect_topic(text: str) -> Optional[str]:
    """Phát hiện chủ đề chính của query."""
    text_lower = text.lower()
    topic_scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            topic_scores[topic] = score
    if not topic_scores:
        return None
    return max(topic_scores, key=topic_scores.get)


def _extract_keywords(text: str) -> List[str]:
    """
    Trích xuất từ khóa từ query:
    - Tách từ
    - Loại bỏ stopword
    - Giữ lại token có nghĩa
    """
    text_lower = text.lower().strip()
    # Tách từ đơn giản
    tokens = re.sub(r"[^\w\s]", " ", text_lower).split()
    keywords = [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]
    return keywords


# ── Query Processor ──────────────────────────────────────────────────────────


class QueryProcessor:
    """
    Xử lý query tiếng Việt: NER + chuẩn hóa + intent detection.

    Ví dụ:
        processor = QueryProcessor(ner_engine, entity_linker)
        result = processor.process("WHO cảnh báo dịch cúm H5N1 tại Việt Nam")
        print(result['entities'])   # [{'canonical': 'WHO', 'type': 'ORG'}, ...]
    """

    def __init__(self, ner_engine, entity_linker):
        """
        Args:
            ner_engine: VietnameseNER instance
            entity_linker: EntityLinker instance
        """
        self.ner = ner_engine
        self.linker = entity_linker

    def process(self, query: str) -> Dict:
        """
        Xử lý đầy đủ một query.

        Returns:
            {
                'original':    str,
                'normalized':  str,
                'entities':    list of linked entity dicts,
                'keywords':    list of keyword strings,
                'topic':       str hoặc None,
                'year_filter': str hoặc None,
                'intent':      'news_search',
            }
        """
        if not query or not query.strip():
            return self._empty_result(query)

        # 1. Normalize
        normalized = re.sub(r"\s+", " ", query.strip())

        # 2. NER
        raw_entities = self.ner.extract(normalized)

        # 3. Entity Linking
        linked_entities = self.linker.link_entities(raw_entities)

        # 4. Keywords
        keywords = _extract_keywords(normalized)

        # 5. Topic & time detection
        topic = _detect_topic(normalized)
        year_filter = _extract_year(normalized)

        return {
            "original": query,
            "normalized": normalized,
            "entities": linked_entities,
            "keywords": keywords,
            "topic": topic,
            "year_filter": year_filter,
            "intent": "news_search",
        }

    def _empty_result(self, query: str) -> Dict:
        return {
            "original": query,
            "normalized": "",
            "entities": [],
            "keywords": [],
            "topic": None,
            "year_filter": None,
            "intent": "news_search",
        }

    def get_query_entity_names(self, processed_query: Dict) -> List[str]:
        """Lấy danh sách canonical entity names từ processed query."""
        return [
            e["canonical"]
            for e in processed_query.get("entities", [])
            if e.get("canonical")
        ]

    def build_search_text(self, processed_query: Dict) -> str:
        """
        Tạo chuỗi text tìm kiếm từ query đã xử lý.
        Kết hợp keywords + entity canonicals.
        """
        parts = processed_query.get("keywords", [])
        entity_names = self.get_query_entity_names(processed_query)
        combined = list(set(parts + entity_names))
        return " ".join(combined)

    def format_for_display(self, processed_query: Dict) -> str:
        """Định dạng kết quả xử lý query để hiển thị cho người dùng."""
        lines = [
            f"Query gốc    : {processed_query['original']}",
            f"Keywords     : {', '.join(processed_query['keywords'])}",
            f"Chủ đề       : {processed_query['topic'] or 'không xác định'}",
        ]
        if processed_query["year_filter"]:
            lines.append(f"Năm          : {processed_query['year_filter']}")

        if processed_query["entities"]:
            lines.append("Entity phát hiện:")
            for e in processed_query["entities"]:
                lines.append(f"  - {e['canonical']:20s} ({e['type']})")

        return "\n".join(lines)


# ── Demo standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.preprocessing.ner import VietnameseNER
    from src.preprocessing.entity_linking import EntityLinker

    ner = VietnameseNER(use_model=False)
    linker = EntityLinker()
    proc = QueryProcessor(ner, linker)

    test_queries = [
        "chiến tranh nga ukraine 2024",
        "WHO cảnh báo dịch COVID-19 tại Việt Nam",
        "Samsung đầu tư vào Hà Nội",
        "bầu cử tổng thống Mỹ",
        "kinh tế Việt Nam tăng trưởng",
    ]

    for q in test_queries:
        result = proc.process(q)
        print("─" * 50)
        print(proc.format_for_display(result))
    print("─" * 50)
