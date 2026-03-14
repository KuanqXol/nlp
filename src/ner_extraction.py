"""
Module: ner_extraction.py
Chức năng: NER tiếng Việt với VnCoreNLP pipeline (tokenize + POS + NER).

Thứ tự ưu tiên backend:
  1. VnCoreNLP  — model chuẩn, POS đi kèm, nhẹ, chạy CPU tốt
  2. underthesea — fallback, không có POS chi tiết
  3. HuggingFace transformer — fallback nếu có model
  4. Rule-based dict — fallback cuối, không cần thư viện

POS tag được lưu lại để relation extraction dùng lọc verb-between-entities.
"""

import re
from typing import Dict, List, Optional

# ── Try import backends ───────────────────────────────────────────────────────

try:
    from vncorenlp import VnCoreNLP as _VnCoreNLP

    _VNCORENLP_AVAILABLE = True
except ImportError:
    _VNCORENLP_AVAILABLE = False

try:
    from underthesea import ner as _underthesea_ner

    _UNDERTHESEA_AVAILABLE = True
except ImportError:
    _UNDERTHESEA_AVAILABLE = False

try:
    from transformers import pipeline as _hf_pipeline

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


# ── Rule-based fallback ───────────────────────────────────────────────────────

_PERSON_DICT = [
    "Putin",
    "Zelensky",
    "Biden",
    "Trump",
    "Tập Cận Bình",
    "Phạm Minh Chính",
    "Nguyễn Phú Trọng",
    "Vương Đình Huệ",
    "Trương Gia Bình",
    "Phạm Nhật Vượng",
    "Elon Musk",
    "Sundar Pichai",
]
_LOCATION_DICT = [
    "Hà Nội",
    "TP.HCM",
    "TP Hồ Chí Minh",
    "Sài Gòn",
    "Việt Nam",
    "Nga",
    "Ukraine",
    "Trung Quốc",
    "Mỹ",
    "Hoa Kỳ",
    "Anh",
    "Pháp",
    "Đức",
    "Nhật Bản",
    "Hàn Quốc",
    "Philippines",
    "Thái Lan",
    "Indonesia",
    "châu Á",
    "châu Âu",
    "Đông Nam Á",
    "Biển Đông",
    "Gaza",
    "Israel",
    "Donetsk",
    "Đà Nẵng",
]
_ORG_DICT = [
    "WHO",
    "Liên Hợp Quốc",
    "LHQ",
    "ASEAN",
    "NATO",
    "EU",
    "VinAI",
    "Vingroup",
    "FPT",
    "Samsung",
    "Google",
    "Tesla",
    "Apple",
    "Microsoft",
    "Ngân hàng Nhà nước",
    "Ngân hàng Thế giới",
    "Bộ Y tế",
    "Bộ Giáo dục",
    "Quốc hội",
]
_MISC_DICT = [
    "COVID-19",
    "SARS-CoV-2",
    "H5N1",
    "Dengue",
    "vaccine",
    "AI",
    "trí tuệ nhân tạo",
    "5G",
    "IoT",
]

_VNCORENLP_TAG_MAP = {
    "B-PER": "PER",
    "I-PER": "PER",
    "B-LOC": "LOC",
    "I-LOC": "LOC",
    "B-ORG": "ORG",
    "I-ORG": "ORG",
    "B-MISC": "MISC",
    "I-MISC": "MISC",
}


def _rule_based_ner(text: str) -> List[Dict]:
    entities, found_spans = [], set()

    def _scan(word_list, ent_type):
        for word in word_list:
            for m in re.finditer(re.escape(word), text):
                span = (m.start(), m.end())
                if not any(span[0] < fe and span[1] > fs for fs, fe in found_spans):
                    found_spans.add(span)
                    entities.append(
                        {"text": m.group(), "type": ent_type, "pos": None, "score": 0.7}
                    )

    _scan(_PERSON_DICT, "PER")
    _scan(_LOCATION_DICT, "LOC")
    _scan(_ORG_DICT, "ORG")
    _scan(_MISC_DICT, "MISC")
    return entities


# ── VnCoreNLP backend ─────────────────────────────────────────────────────────


class _VnCoreNLPBackend:
    """
    Wrap VnCoreNLP server.
    Cài: pip install vncorenlp
    Tải jar + models: https://github.com/vncorenlp/VnCoreNLP
    """

    def __init__(self, jar_path: str, port: int = 9000):
        self.jar_path = jar_path
        self.port = port
        self._client = None

    def _load(self):
        if self._client is None:
            print(f"[NER/VnCoreNLP] Khởi động (jar={self.jar_path}, port={self.port})")
            self._client = _VnCoreNLP(
                self.jar_path,
                annotators="wseg,pos,ner",
                port=self.port,
                quiet=True,
            )
            print("[NER/VnCoreNLP] Sẵn sàng.")

    def annotate(self, text: str) -> List[Dict]:
        self._load()
        try:
            result = self._client.annotate(text)
        except Exception as e:
            print(f"[NER/VnCoreNLP] Lỗi: {e} — fallback rule-based")
            return _rule_based_ner(text)

        entities = []
        for sent in result.get("sentences", []):
            i = 0
            while i < len(sent):
                tok = sent[i]
                ner_tag = tok.get("nerLabel", "O")
                if not ner_tag.startswith("B-"):
                    i += 1
                    continue
                ent_type = _VNCORENLP_TAG_MAP.get(ner_tag)
                if not ent_type:
                    i += 1
                    continue
                span_toks = [tok]
                j = i + 1
                while j < len(sent):
                    nt = sent[j].get("nerLabel", "O")
                    if nt.startswith("I-") and _VNCORENLP_TAG_MAP.get(nt) == ent_type:
                        span_toks.append(sent[j])
                        j += 1
                    else:
                        break
                entities.append(
                    {
                        "text": " ".join(t["form"] for t in span_toks),
                        "type": ent_type,
                        "pos": span_toks[0].get("posTag"),
                        "score": 0.92,
                    }
                )
                i = j
        return entities

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


# ── underthesea backend ───────────────────────────────────────────────────────


class _UndertheseaBackend:
    _MAP = {
        "B-PER": "PER",
        "I-PER": "PER",
        "B-LOC": "LOC",
        "I-LOC": "LOC",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
        "B-MISC": "MISC",
        "I-MISC": "MISC",
    }

    def annotate(self, text: str) -> List[Dict]:
        try:
            raw = _underthesea_ner(text)
        except Exception:
            return _rule_based_ner(text)
        entities, i = [], 0
        while i < len(raw):
            word, pos, _, ner = raw[i]
            ent_type = self._MAP.get(ner)
            if ent_type and ner.startswith("B-"):
                span = [word]
                j = i + 1
                while j < len(raw):
                    _, _, _, nn = raw[j]
                    if self._MAP.get(nn) == ent_type and nn.startswith("I-"):
                        span.append(raw[j][0])
                        j += 1
                    else:
                        break
                entities.append(
                    {
                        "text": " ".join(span),
                        "type": ent_type,
                        "pos": pos,
                        "score": 0.85,
                    }
                )
                i = j
            else:
                i += 1
        return entities


# ── Main NER engine ───────────────────────────────────────────────────────────


class VietnameseNER:
    """
    NER tiếng Việt — tự chọn backend tốt nhất có sẵn.

    Khuyến nghị cho production (100k bài):
        ner = VietnameseNER(vncorenlp_jar="VnCoreNLP-1.1.1.jar")

    Chạy không cần cài gì thêm (fallback tự động):
        ner = VietnameseNER()
    """

    def __init__(
        self,
        vncorenlp_jar: Optional[str] = None,
        vncorenlp_port: int = 9000,
        use_transformer: bool = False,
        transformer_model: str = "NlpHUST/ner-vietnamese-electra-base",
        use_model: bool = None,  # backward-compat alias cho use_transformer
    ):
        if use_model is not None:
            use_transformer = use_model
        self._backend = None
        self._hf_pipe = None
        self._backend_name = "rule-based"

        # 1. VnCoreNLP
        if vncorenlp_jar and _VNCORENLP_AVAILABLE:
            try:
                self._backend = _VnCoreNLPBackend(vncorenlp_jar, vncorenlp_port)
                self._backend_name = "VnCoreNLP"
                print("[NER] Backend: VnCoreNLP")
                return
            except Exception as e:
                print(f"[NER] VnCoreNLP lỗi: {e}")

        # 2. underthesea
        if _UNDERTHESEA_AVAILABLE:
            self._backend = _UndertheseaBackend()
            self._backend_name = "underthesea"
            print("[NER] Backend: underthesea")
            return

        # 3. HuggingFace transformer
        if use_transformer and _TRANSFORMERS_AVAILABLE:
            try:
                self._hf_pipe = _hf_pipeline(
                    "ner",
                    model=transformer_model,
                    aggregation_strategy="simple",
                )
                self._backend_name = "transformer"
                print(f"[NER] Backend: HuggingFace ({transformer_model})")
                return
            except Exception as e:
                print(f"[NER] Transformer lỗi: {e}")

        # 4. Rule-based
        print("[NER] Backend: rule-based (fallback)")

    def extract(self, text: str) -> List[Dict]:
        """Trích xuất entity, trả về list {text, type, pos, score}."""
        if not text or not text.strip():
            return []
        if self._backend is not None:
            return self._backend.annotate(text)
        if self._hf_pipe:
            try:
                raw = self._hf_pipe(text)
                return [
                    {
                        "text": r["word"],
                        "type": r["entity_group"],
                        "pos": None,
                        "score": round(float(r["score"]), 3),
                    }
                    for r in raw
                ]
            except Exception:
                pass
        return _rule_based_ner(text)

    def extract_from_document(self, doc: Dict) -> Dict:
        """Thêm field 'entities' vào document. Dedup theo (text.lower, type)."""
        text = doc.get("full_text", doc.get("content", ""))
        raw = self.extract(text)
        seen, unique = set(), []
        for e in raw:
            key = (e["text"].lower(), e["type"])
            if key not in seen:
                seen.add(key)
                unique.append(
                    {
                        "text": e["text"],
                        "type": e["type"],
                        "pos": e.get("pos"),
                        "score": e.get("score", 0.8),
                    }
                )
        out = doc.copy()
        out["entities"] = unique
        return out

    def batch_extract(self, documents: List[Dict], log_every: int = 500) -> List[Dict]:
        """Xử lý hàng loạt, log progress."""
        print(f"[NER] Xử lý {len(documents)} bài (backend={self._backend_name})...")
        result = []
        for i, doc in enumerate(documents):
            result.append(self.extract_from_document(doc))
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                print(f"  [{i+1}/{len(documents)}]")
        print("[NER] Hoàn thành.")
        return result

    def close(self):
        if isinstance(self._backend, _VnCoreNLPBackend):
            self._backend.close()

    @property
    def backend_name(self) -> str:
        return self._backend_name


# ── Utilities ─────────────────────────────────────────────────────────────────


def get_entities_by_type(entities: List[Dict], entity_type: str) -> List[str]:
    return [e["text"] for e in entities if e["type"] == entity_type]


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ner = VietnameseNER()
    for text in [
        "Putin gặp Zelensky tại Ukraine để đàm phán hòa bình",
        "WHO cảnh báo dịch COVID-19 bùng phát tại Hà Nội và TP.HCM",
        "Samsung khai trương R&D tại Hà Nội với sự tham dự của Phạm Minh Chính",
    ]:
        print(f"\n{text}")
        for e in ner.extract(text):
            print(
                f"  {e['text']:20s} [{e['type']}] pos={e.get('pos')} score={e.get('score')}"
            )
