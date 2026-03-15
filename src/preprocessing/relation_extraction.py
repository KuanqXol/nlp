"""
Module: relation_extraction.py
Chức năng: Trích xuất triple (Subject, Relation, Object) từ văn bản tiếng Việt.

Cải tiến so với phiên bản cũ:
  1. Span-based: keyword phải nằm trong đoạn GIỮA 2 entity (≤50 ký tự)
  2. POS filter: giữa 2 entity cần có ít nhất 1 động từ (từ VnCoreNLP)
  3. Type constraint: mỗi relation có ràng buộc entity type
  4. Coreference: map đại từ → entity gần nhất cùng type
  5. Confidence score: base_conf × entity_link_score, chỉ giữ ≥ MIN_CONFIDENCE
  6. Temporal tagging: lấy từ metadata date + regex fallback
  7. Category routing: bài kinh tế → relation set kinh tế, bài chính trị → chính trị
"""

import re
import unicodedata
from collections import deque
from typing import Dict, List, Optional, Tuple


# ── Relation taxonomy ─────────────────────────────────────────────────────────


class R:
    """Tên nhãn quan hệ chuẩn hóa."""

    LEADS = "leads"
    MEMBER_OF = "member_of"
    APPOINTED = "appointed"
    COOPERATES = "cooperates_with"
    SIGNS_DEAL = "signs_deal_with"
    LOCATED_IN = "located_in"
    INVESTS_IN = "invests_in"
    ACQUIRES = "acquires"
    FOUNDED = "founded"
    PRODUCES = "produces"
    ATTACKS = "attacks"
    SUPPORTS = "supports"
    SANCTIONS = "sanctions"
    MEETS = "meets"
    WARNS_ABOUT = "warns_about"
    FOUND_IN = "found_in"
    RELATED_TO = "related_to"

    SYMMETRIC = {COOPERATES, MEETS, RELATED_TO, SIGNS_DEAL}


# ── Keyword → (relation, base_confidence, subj_types, obj_types) ─────────────
# subj_types / obj_types: set of allowed entity types, None = any

KEYWORD_RULES: List[Tuple[List[str], str, float, Optional[set], Optional[set]]] = [
    # Lãnh đạo / chính trị
    (["lãnh đạo", "đứng đầu"], R.LEADS, 0.90, {"PER"}, {"ORG", "LOC"}),
    (["bổ nhiệm", "đề cử", "chỉ định"], R.APPOINTED, 0.88, {"PER", "ORG"}, {"PER"}),
    (["thành viên", "gia nhập"], R.MEMBER_OF, 0.82, {"ORG", "LOC"}, {"ORG"}),
    (["hợp tác", "liên minh", "bắt tay"], R.COOPERATES, 0.80, None, None),
    (["ký kết", "ký hiệp định", "ký thỏa thuận", "ký"], R.SIGNS_DEAL, 0.88, None, None),
    # Địa lý
    (["tại", "ở", "đặt tại", "trụ sở tại"], R.LOCATED_IN, 0.78, None, {"LOC"}),
    # Kinh tế
    (
        ["đầu tư vào", "đầu tư", "rót vốn"],
        R.INVESTS_IN,
        0.88,
        {"ORG", "PER"},
        {"ORG", "LOC"},
    ),
    (["mua lại", "thâu tóm", "sáp nhập"], R.ACQUIRES, 0.90, {"ORG"}, {"ORG"}),
    (["thành lập", "sáng lập"], R.FOUNDED, 0.88, {"PER", "ORG"}, {"ORG"}),
    (["ra mắt", "khai trương", "sản xuất"], R.PRODUCES, 0.80, {"ORG"}, None),
    # Quân sự / ngoại giao
    (["tấn công", "không kích", "xâm chiếm"], R.ATTACKS, 0.92, {"LOC", "ORG"}, {"LOC"}),
    (
        ["hỗ trợ", "viện trợ", "ủng hộ"],
        R.SUPPORTS,
        0.85,
        {"ORG", "PER"},
        {"LOC", "ORG"},
    ),
    (["trừng phạt", "cấm vận"], R.SANCTIONS, 0.90, {"LOC", "ORG"}, {"LOC", "ORG"}),
    (["gặp", "hội đàm", "tiếp"], R.MEETS, 0.82, {"PER"}, {"PER"}),
    # Y tế
    (["cảnh báo", "khuyến cáo"], R.WARNS_ABOUT, 0.88, {"ORG"}, None),
    (
        ["phát hiện", "ghi nhận", "bùng phát"],
        R.FOUND_IN,
        0.83,
        {"MISC", "ORG"},
        {"LOC"},
    ),
]

# Mapping category bài báo → relation set được phép dùng
CATEGORY_RELATION_MAP: Dict[str, set] = {
    "kinh tế": {
        R.INVESTS_IN,
        R.ACQUIRES,
        R.FOUNDED,
        R.PRODUCES,
        R.SIGNS_DEAL,
        R.COOPERATES,
        R.LOCATED_IN,
        R.MEETS,
    },
    "chính trị": {
        R.LEADS,
        R.APPOINTED,
        R.MEMBER_OF,
        R.COOPERATES,
        R.SIGNS_DEAL,
        R.MEETS,
        R.SANCTIONS,
        R.ATTACKS,
        R.SUPPORTS,
        R.LOCATED_IN,
    },
    "thế giới": {
        R.ATTACKS,
        R.SUPPORTS,
        R.SANCTIONS,
        R.MEETS,
        R.SIGNS_DEAL,
        R.COOPERATES,
        R.LEADS,
        R.LOCATED_IN,
        R.WARNS_ABOUT,
    },
    "y tế": {R.WARNS_ABOUT, R.FOUND_IN, R.COOPERATES, R.LOCATED_IN, R.PRODUCES},
    "công nghệ": {
        R.PRODUCES,
        R.INVESTS_IN,
        R.ACQUIRES,
        R.FOUNDED,
        R.COOPERATES,
        R.LOCATED_IN,
        R.MEETS,
    },
    "giáo dục": {R.LOCATED_IN, R.COOPERATES, R.FOUNDED, R.MEETS},
}
_DEFAULT_RELATIONS = {r for rules in CATEGORY_RELATION_MAP.values() for r in rules}

MIN_CONFIDENCE = 0.60
MAX_SPAN_CHARS = 50  # Khoảng cách tối đa giữa 2 entity (ký tự)
CROSS_SENT_DECAY = 0.88  # Giảm confidence khi triple trải qua nhiều câu

# POS tag được coi là động từ (VnCoreNLP + underthesea)
VERB_POS = {"V", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VP"}

# Đại từ tiếng Việt → loại entity
PRONOUN_MAP: Dict[str, str] = {
    "ông": "PER",
    "ông ấy": "PER",
    "ông ta": "PER",
    "vị này": "PER",
    "bà": "PER",
    "bà ấy": "PER",
    "cô": "PER",
    "cô ấy": "PER",
    "anh ấy": "PER",
    "họ": None,
    "công ty này": "ORG",
    "tập đoàn này": "ORG",
    "tổ chức này": "ORG",
    "bộ này": "ORG",
    "cơ quan này": "ORG",
    "nước này": "LOC",
    "quốc gia này": "LOC",
    "thành phố này": "LOC",
    "dịch bệnh này": "MISC",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _nk(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_temporal(text: str, doc_date: str = "") -> Optional[str]:
    """Trích xuất temporal marker. Ưu tiên doc_date."""
    if doc_date:
        return doc_date
    for pat in [
        r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
        r"\b(tháng\s+\d{1,2}\s+năm\s+\d{4})\b",
        r"\b(năm\s+20\d{2}|năm\s+19\d{2})\b",
        r"\b(20\d{2}|19\d{2})\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group().strip()
    return None


def _has_verb_between(
    text: str, pos1: int, pos2: int, entities_with_pos: List[Dict]
) -> bool:
    """
    Kiểm tra có token động từ nằm trong khoảng (pos1, pos2) không.
    Dùng POS từ VnCoreNLP nếu có, fallback regex.
    """
    # Nếu entities có POS info từ VnCoreNLP
    for ent in entities_with_pos:
        if ent.get("pos") in VERB_POS:
            idx = text.lower().find(ent["text"].lower())
            if idx != -1 and min(pos1, pos2) < idx < max(pos1, pos2):
                return True

    # Fallback: regex tìm động từ phổ biến tiếng Việt
    between = text[min(pos1, pos2) : max(pos1, pos2)].lower()
    verb_patterns = [
        r"\b(tấn công|hỗ trợ|ký kết|gặp|cảnh báo|đầu tư|khai trương|"
        r"thành lập|mua lại|bổ nhiệm|lãnh đạo|hội đàm|viện trợ|"
        r"phát hiện|ghi nhận|tuyên bố|khẳng định|ra mắt)\b"
    ]
    for pat in verb_patterns:
        if re.search(pat, between):
            return True
    return False


# ── Coreference resolver ──────────────────────────────────────────────────────


class _CoreferenceResolver:
    """
    Giải đại từ → entity gần nhất cùng type trong sliding window 3 câu.
    """

    def __init__(self, window: int = 20):
        self._history: deque = deque(maxlen=window)

    def update(self, entities: List[Dict]):
        for e in entities:
            if e.get("canonical"):
                self._history.append(e)

    def resolve(self, pronoun: str) -> Optional[Dict]:
        expected_type = PRONOUN_MAP.get(_nk(pronoun))
        if expected_type is None and _nk(pronoun) not in PRONOUN_MAP:
            return None
        for ent in reversed(list(self._history)):
            if expected_type is None or ent.get("type") == expected_type:
                return ent
        return None

    def expand_entities(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Tìm đại từ trong text, resolve, trả về entity bổ sung."""
        extras = []
        for pronoun, exp_type in PRONOUN_MAP.items():
            if pronoun in text.lower():
                resolved = self.resolve(pronoun)
                if resolved:
                    extras.append(
                        {
                            "text": pronoun,
                            "canonical": resolved["canonical"],
                            "type": resolved.get("type", exp_type or "MISC"),
                            "link_score": 0.75,
                            "coref": True,
                        }
                    )
        return extras


# ── Core extraction ───────────────────────────────────────────────────────────


def _extract_from_span(
    span_text: str,
    entities: List[Dict],
    allowed_relations: set,
    full_text_entities: List[Dict],
) -> List[Tuple[str, str, str, float]]:
    """
    Trích xuất triple từ một đoạn text (câu hoặc window nhiều câu).
    Returns: [(subj, relation, obj, confidence)]
    """
    triples = []
    span_lower = span_text.lower()

    # Entity có trong đoạn này
    present = [
        e
        for e in entities
        if e.get("canonical", "").lower() in span_lower
        or e.get("text", "").lower() in span_lower
    ]
    if len(present) < 2:
        return triples

    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            e1, e2 = present[i], present[j]
            c1 = e1.get("canonical", e1.get("text", ""))
            c2 = e2.get("canonical", e2.get("text", ""))
            if not c1 or not c2 or c1 == c2:
                continue

            # Vị trí trong span
            pos1 = span_lower.find(c1.lower())
            pos2 = span_lower.find(c2.lower())
            if pos1 == -1 or pos2 == -1:
                continue

            # Đoạn giữa phải đủ ngắn
            gap_start = min(pos1, pos2) + (len(c1) if pos1 < pos2 else len(c2))
            gap_end = max(pos1, pos2)
            between = span_text[gap_start:gap_end].strip()
            if len(between) > MAX_SPAN_CHARS:
                continue

            # POS check: cần ít nhất 1 động từ trong khoảng giữa
            if not _has_verb_between(span_text, gap_start, gap_end, full_text_entities):
                continue

            # Subject là entity xuất hiện trước
            if pos1 < pos2:
                subj, obj = c1, c2
                s_ent, o_ent = e1, e2
            else:
                subj, obj = c2, c1
                s_ent, o_ent = e2, e1

            between_lower = between.lower()
            t1 = s_ent.get("type", "")
            t2 = o_ent.get("type", "")

            # Match keyword rules
            matched = False
            for keywords, relation, base_conf, subj_types, obj_types in KEYWORD_RULES:
                if relation not in allowed_relations:
                    continue
                if matched:
                    break
                for kw in keywords:
                    if kw not in between_lower:
                        continue
                    # Type constraint
                    if subj_types and t1 not in subj_types:
                        continue
                    if obj_types and t2 not in obj_types:
                        continue
                    # Confidence = base × link_score trung bình
                    ls = (
                        s_ent.get("link_score", 1.0) + o_ent.get("link_score", 1.0)
                    ) / 2
                    conf = round(base_conf * ls, 3)
                    if conf >= MIN_CONFIDENCE:
                        triples.append((subj, relation, obj, conf))
                    matched = True
                    break

    return triples


# ── Relation Extractor ────────────────────────────────────────────────────────


class RelationExtractor:
    """
    Trích xuất triple từ document tiếng Việt.

    Cách dùng:
        extractor = RelationExtractor()
        doc = {..., 'linked_entities': [...], 'date': '2024-01-15', 'category': 'kinh tế'}
        processed = extractor.process_document(doc)
        for t in processed['triples']:
            print(t)
        # {'subject','relation','object','confidence','temporal'}
    """

    def __init__(
        self,
        cross_window: int = 3,
        use_coreference: bool = True,
    ):
        self.cross_window = cross_window
        self.use_coreference = use_coreference

    def _get_allowed_relations(self, category: str) -> set:
        cat = _nk(category) if category else ""
        for key, rel_set in CATEGORY_RELATION_MAP.items():
            if key in cat:
                return rel_set
        return _DEFAULT_RELATIONS

    def extract(self, doc: Dict) -> List[Dict]:
        text = doc.get("full_text", doc.get("content", ""))
        entities = doc.get("linked_entities", doc.get("entities", []))
        doc_date = doc.get("date", "")
        category = doc.get("category", "")

        if not text or not entities:
            return []

        allowed_rels = self._get_allowed_relations(category)
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 8
        ]
        coref = _CoreferenceResolver() if self.use_coreference else None

        seen: Dict[Tuple, float] = {}  # (subj, rel, obj) → max confidence

        def _register(triples, decay=1.0):
            for s, r, o, c in triples:
                key = (s, r, o)
                conf = round(c * decay, 3)
                if conf >= MIN_CONFIDENCE:
                    seen[key] = max(seen.get(key, 0.0), conf)

        # A. Per-sentence
        for sent in sentences:
            sent_ents = list(entities)
            if coref:
                extras = coref.expand_entities(sent, sent_ents)
                sent_ents = sent_ents + extras
                coref.update(sent_ents)
            triples = _extract_from_span(sent, sent_ents, allowed_rels, entities)
            _register(triples)

        # B. Cross-sentence sliding window
        for i in range(len(sentences) - 1):
            window_text = " ".join(sentences[i : i + self.cross_window])
            triples = _extract_from_span(window_text, entities, allowed_rels, entities)
            _register(triples, decay=CROSS_SENT_DECAY)

        # C. Build output
        temporal = _extract_temporal(text, doc_date)
        result = []
        for (s, r, o), conf in seen.items():
            triple = {"subject": s, "relation": r, "object": o, "confidence": conf}
            if temporal:
                triple["temporal"] = temporal
            if r in R.SYMMETRIC:
                triple["symmetric"] = True
            result.append(triple)

        result.sort(key=lambda x: -x["confidence"])
        return result

    def process_document(self, doc: Dict) -> Dict:
        triples = self.extract(doc)
        out = doc.copy()
        out["triples"] = triples
        return out

    def batch_process(self, documents: List[Dict], log_every: int = 500) -> List[Dict]:
        print(f"[RelationExtractor] Xử lý {len(documents)} bài...")
        result, total = [], 0
        for i, doc in enumerate(documents):
            p = self.process_document(doc)
            result.append(p)
            total += len(p.get("triples", []))
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                print(f"  [{i+1}/{len(documents)}] triples so far: {total}")
        print(
            f"[RelationExtractor] Hoàn thành. Tổng triple (≥{MIN_CONFIDENCE}): {total}"
        )
        return result


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = RelationExtractor()

    sample_docs = [
        {
            "id": "d1",
            "date": "2024-01-15",
            "category": "thế giới",
            "full_text": (
                "Putin tuyên bố tiếp tục chiến dịch tại Ukraine năm 2024. "
                "Ông khẳng định sẽ không dừng lại. "
                "Zelensky kêu gọi NATO hỗ trợ thêm vũ khí cho Ukraine. "
                "WHO cảnh báo khủng hoảng nhân đạo tại Donetsk."
            ),
            "linked_entities": [
                {
                    "text": "Putin",
                    "canonical": "Putin",
                    "type": "PER",
                    "link_score": 1.0,
                },
                {
                    "text": "Ukraine",
                    "canonical": "Ukraine",
                    "type": "LOC",
                    "link_score": 1.0,
                },
                {
                    "text": "Zelensky",
                    "canonical": "Zelensky",
                    "type": "PER",
                    "link_score": 1.0,
                },
                {"text": "NATO", "canonical": "NATO", "type": "ORG", "link_score": 1.0},
                {"text": "WHO", "canonical": "WHO", "type": "ORG", "link_score": 1.0},
                {
                    "text": "Donetsk",
                    "canonical": "Donetsk",
                    "type": "LOC",
                    "link_score": 0.9,
                },
            ],
        },
        {
            "id": "d2",
            "date": "2024-02-01",
            "category": "kinh tế",
            "full_text": (
                "Google đầu tư vào VinAI tại Hà Nội năm 2024. "
                "Phạm Minh Chính gặp Sundar Pichai. "
                "Hai bên ký kết thỏa thuận hợp tác công nghệ."
            ),
            "linked_entities": [
                {
                    "text": "Google",
                    "canonical": "Google",
                    "type": "ORG",
                    "link_score": 1.0,
                },
                {
                    "text": "VinAI",
                    "canonical": "VinAI",
                    "type": "ORG",
                    "link_score": 1.0,
                },
                {
                    "text": "Hà Nội",
                    "canonical": "Hà Nội",
                    "type": "LOC",
                    "link_score": 1.0,
                },
                {
                    "text": "Phạm Minh Chính",
                    "canonical": "Phạm Minh Chính",
                    "type": "PER",
                    "link_score": 1.0,
                },
                {
                    "text": "Sundar Pichai",
                    "canonical": "Sundar Pichai",
                    "type": "PER",
                    "link_score": 1.0,
                },
            ],
        },
    ]

    for doc in sample_docs:
        processed = extractor.process_document(doc)
        print(f"\n📰 [{doc['category']}] {doc['id']}: {doc['full_text'][:55]}...")
        for t in processed["triples"]:
            temp = f" [{t.get('temporal','')}]" if t.get("temporal") else ""
            print(
                f"  ({t['subject']}) -[{t['relation']}]-> ({t['object']})  "
                f"conf={t['confidence']:.2f}{temp}"
            )
