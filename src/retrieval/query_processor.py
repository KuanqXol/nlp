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

# Các cụm thường xuyên bị tách rời nếu chỉ split theo whitespace
QUERY_PHRASES = [
    "tổng thống",
    "phó tổng thống",
    "thủ tướng",
    "chủ tịch nước",
    "bộ trưởng",
    "ngoại trưởng",
    "đầu tư",
    "việt nam",
    "thành phố hồ chí minh",
    "hà nội",
    "mỹ",
    "hoa kỳ",
    "liên minh châu âu",
    "trí tuệ nhân tạo",
    "thị trường",
    "doanh nghiệp",
]

QUERY_ROLE_GUARDS = {
    "tổng thống",
    "phó tổng thống",
    "thủ tướng",
    "chủ tịch nước",
    "bộ trưởng",
    "ngoại trưởng",
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


def _merge_common_phrases(tokens: List[str]) -> List[str]:
    """Ghép các cụm phổ biến bị tách rời do split theo khoảng trắng.

    Mục tiêu là tránh các keyword rác kiểu: "đầu", "tư".
    Ví dụ: ["samsung", "đầu", "tư", "vào", "việt", "nam"]
    -> ["samsung", "đầu tư", "việt nam"]
    """
    if not tokens:
        return []

    out: List[str] = []
    i = 0
    lower_tokens = [t.lower() for t in tokens]

    while i < len(lower_tokens):
        matched = False
        # ưu tiên ghép cụm dài trước
        for phrase in sorted(QUERY_PHRASES, key=lambda x: len(x.split()), reverse=True):
            parts = phrase.split()
            n = len(parts)
            if i + n <= len(lower_tokens) and lower_tokens[i : i + n] == parts:
                out.append(phrase)
                i += n
                matched = True
                break
        if matched:
            continue

        tok = lower_tokens[i].replace("_", " ").strip()
        if tok and tok not in VIETNAMESE_STOPWORDS and len(tok.replace(" ", "")) > 1:
            out.append(tok)
        i += 1

    return out


def _extract_keywords(text: str) -> List[str]:
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return _merge_common_phrases(tokens)


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

    @staticmethod
    def _is_role_guard(text: str) -> bool:
        return _normalize(text).lower() in QUERY_ROLE_GUARDS

    @staticmethod
    def _is_noisy_single_token_link(ent: Dict) -> bool:
        mention = _normalize(
            ent.get("text")
            or ent.get("surface_form")
            or ent.get("mention_text")
            or ""
        )
        canonical = _normalize(ent.get("canonical") or "")
        if not mention or not canonical:
            return False
        if len(mention.split()) != 1:
            return False
        if ent.get("match_type") != "exact":
            return False
        if mention.lower() == canonical.lower():
            return False
        return True

    def _segment_query_tokens(self, text: str) -> List[str]:
        segmented = ""
        if hasattr(self.ner, "segment_text"):
            try:
                segmented = self.ner.segment_text(text)
            except Exception:
                segmented = ""
        if segmented:
            tokens = [
                tok.replace("_", " ").strip()
                for tok in segmented.split()
                if tok.strip()
            ]
            if tokens:
                return tokens
        return [tok for tok in re.sub(r"[^\w\s]", " ", text).split() if tok]

    def _extract_keywords_for_query(self, text: str) -> List[str]:
        tokens = [tok.lower() for tok in self._segment_query_tokens(text)]
        return _merge_common_phrases(tokens)

    def _recover_query_entities(
        self, text: str, raw_entities: List[Dict]
    ) -> List[Dict]:
        existing_mentions = {
            _normalize((ent.get("text") or ent.get("entity_text") or "")).lower()
            for ent in raw_entities
            if (ent.get("text") or ent.get("entity_text"))
        }
        recovered: List[Dict] = []
        tokens = self._segment_query_tokens(text)
        if not tokens:
            return recovered

        max_ngram = min(5, len(tokens))
        i = 0
        while i < len(tokens):
            matched = None
            matched_span = 1
            for n in range(max_ngram, 0, -1):
                if i + n > len(tokens):
                    continue
                phrase = " ".join(tokens[i : i + n]).strip()
                norm_phrase = _normalize(phrase).lower()
                if (
                    not phrase
                    or norm_phrase in existing_mentions
                    or self._is_role_guard(phrase)
                ):
                    continue
                alias_hit = self.linker.lookup_alias(phrase)
                if alias_hit:
                    matched = {
                        "text": phrase,
                        "entity_text": phrase,
                        "mention_text": phrase,
                        "resolved_text": phrase,
                        "type": alias_hit.get("type", "MISC"),
                        "entity_type": alias_hit.get("type", "MISC"),
                        "score": float(alias_hit.get("similarity", 1.0)),
                    }
                    matched_span = n
                    existing_mentions.add(norm_phrase)
                    break
            if matched:
                recovered.append(matched)
                i += matched_span
            else:
                i += 1
        return recovered

    def process(self, query: str) -> Dict:
        if not query or not query.strip():
            return self._empty(query)

        normalized = _normalize(query)
        raw_entities = self.ner.extract(normalized)
        recovered_entities = self._recover_query_entities(normalized, raw_entities)
        all_entities = [
            ent
            for ent in (raw_entities + recovered_entities)
            if not self._is_role_guard(
                ent.get("text") or ent.get("entity_text") or ent.get("mention_text") or ""
            )
        ]
        linked = [
            ent
            for ent in self.linker.link_entities(all_entities)
            if not self._is_role_guard(ent.get("text") or ent.get("canonical") or "")
            and not self._is_noisy_single_token_link(ent)
        ]
        keywords = self._extract_keywords_for_query(normalized)
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
        parts = []
        seen = set()
        for item in processed.get("keywords", []) + self.get_query_entity_names(processed):
            item = item.strip().lower().replace("_", " ")
            if not item or item in seen:
                continue
            seen.add(item)
            parts.append(item)
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
