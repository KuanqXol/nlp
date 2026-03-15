"""
Module: ner_extraction.py
NER tiếng Việt với output có cấu trúc, có span ký tự và sentence index.

Output entity chuẩn:
{
    "text": "Vladimir Putin",
    "type": "PER",
    "start": 15,
    "end": 30,
    "sentence_id": 0,
    "pos": "Np",
    "score": 0.92,
    "entity_text": "Vladimir Putin",  # alias rõ nghĩa
    "entity_type": "PER"              # alias rõ nghĩa
}
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

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


def _split_sentences_with_offsets(text: str) -> List[Dict]:
    spans: List[Dict] = []
    for i, m in enumerate(re.finditer(r"[^.!?]+[.!?]?", text, flags=re.UNICODE)):
        sent = m.group().strip()
        if not sent:
            continue
        start = m.start()
        while start < len(text) and text[start].isspace():
            start += 1
        end = start + len(sent)
        spans.append({"sentence_id": i, "text": sent, "start": start, "end": end})
    if not spans and text.strip():
        spans.append(
            {
                "sentence_id": 0,
                "text": text.strip(),
                "start": 0,
                "end": len(text.strip()),
            }
        )
    return spans


def _make_entity(
    text: str,
    ent_type: str,
    start: int,
    end: int,
    sentence_id: int,
    pos: Optional[str],
    score: float,
) -> Dict:
    return {
        "text": text,
        "type": ent_type,
        "start": start,
        "end": end,
        "sentence_id": sentence_id,
        "pos": pos,
        "score": score,
        "entity_text": text,
        "entity_type": ent_type,
    }


def _attach_spans(raw_entities: List[Dict], text: str) -> List[Dict]:
    sentence_spans = _split_sentences_with_offsets(text)
    used_spans = set()
    out: List[Dict] = []

    for e in raw_entities:
        ent_text = e.get("text", "").strip()
        ent_type = e.get("type", "MISC")
        pos = e.get("pos")
        score = float(e.get("score", 0.8))
        if not ent_text:
            continue

        found = False
        for sent in sentence_spans:
            local_start = sent["text"].lower().find(ent_text.lower())
            if local_start < 0:
                continue
            global_start = sent["start"] + local_start
            global_end = global_start + len(ent_text)
            key = (global_start, global_end, ent_type)
            if key in used_spans:
                continue
            used_spans.add(key)
            out.append(
                _make_entity(
                    ent_text,
                    ent_type,
                    global_start,
                    global_end,
                    sent["sentence_id"],
                    pos,
                    score,
                )
            )
            found = True
            break

        if not found:
            # Fallback: nếu không gắn được span thì vẫn trả về entity với span -1.
            out.append(_make_entity(ent_text, ent_type, -1, -1, -1, pos, score))

    return out


def _rule_based_ner(text: str) -> List[Dict]:
    raw: List[Dict] = []

    def _scan(word_list: List[str], ent_type: str):
        for word in word_list:
            for m in re.finditer(re.escape(word), text, flags=re.IGNORECASE):
                raw.append(
                    {
                        "text": m.group(),
                        "type": ent_type,
                        "start": m.start(),
                        "end": m.end(),
                        "sentence_id": -1,
                        "pos": None,
                        "score": 0.7,
                    }
                )

    _scan(_PERSON_DICT, "PER")
    _scan(_LOCATION_DICT, "LOC")
    _scan(_ORG_DICT, "ORG")
    _scan(_MISC_DICT, "MISC")
    return _attach_spans(raw, text)


class _VnCoreNLPBackend:
    def __init__(self, jar_path: str, port: int = 9000):
        self.jar_path = jar_path
        self.port = port
        self._client = None

    def _load(self):
        if self._client is None:
            print(f"[NER/VnCoreNLP] Khởi động (jar={self.jar_path}, port={self.port})")
            self._client = _VnCoreNLP(
                self.jar_path, annotators="wseg,pos,ner", port=self.port, quiet=True
            )
            print("[NER/VnCoreNLP] Sẵn sàng.")

    def annotate(self, text: str) -> List[Dict]:
        self._load()
        try:
            result = self._client.annotate(text)
        except Exception as e:
            print(f"[NER/VnCoreNLP] Lỗi: {e} — fallback rule-based")
            return _rule_based_ner(text)

        raw_entities: List[Dict] = []
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

                raw_entities.append(
                    {
                        "text": " ".join(t.get("form", "") for t in span_toks).strip(),
                        "type": ent_type,
                        "pos": span_toks[0].get("posTag"),
                        "score": 0.92,
                    }
                )
                i = j

        return _attach_spans(raw_entities, text)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None


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

        entities: List[Dict] = []
        i = 0
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

        return _attach_spans(entities, text)


class VietnameseNER:
    def __init__(
        self,
        vncorenlp_jar: Optional[str] = None,
        vncorenlp_port: int = 9000,
        use_transformer: bool = False,
        transformer_model: str = "NlpHUST/ner-vietnamese-electra-base",
        use_model: bool = None,
    ):
        if use_model is not None:
            use_transformer = use_model

        self._backend = None
        self._hf_pipe = None
        self._backend_name = "rule-based"
        self._extract_cache: Dict[str, List[Dict]] = {}

        if vncorenlp_jar and _VNCORENLP_AVAILABLE:
            try:
                self._backend = _VnCoreNLPBackend(vncorenlp_jar, vncorenlp_port)
                self._backend_name = "VnCoreNLP"
                print("[NER] Backend: VnCoreNLP")
                return
            except Exception as e:
                print(f"[NER] VnCoreNLP lỗi: {e}")

        if _UNDERTHESEA_AVAILABLE:
            self._backend = _UndertheseaBackend()
            self._backend_name = "underthesea"
            print("[NER] Backend: underthesea")
            return

        if use_transformer and _TRANSFORMERS_AVAILABLE:
            try:
                self._hf_pipe = _hf_pipeline(
                    "ner", model=transformer_model, aggregation_strategy="simple"
                )
                self._backend_name = "transformer"
                print(f"[NER] Backend: HuggingFace ({transformer_model})")
                return
            except Exception as e:
                print(f"[NER] Transformer lỗi: {e}")

        print("[NER] Backend: rule-based (fallback)")

    def _cache_key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def extract(self, text: str) -> List[Dict]:
        """Trích xuất entity có span + sentence_id."""
        if not text or not text.strip():
            return []

        key = self._cache_key(text)
        if key in self._extract_cache:
            return [dict(e) for e in self._extract_cache[key]]

        if self._backend is not None:
            entities = self._backend.annotate(text)
        elif self._hf_pipe:
            try:
                raw = self._hf_pipe(text)
                entities = _attach_spans(
                    [
                        {
                            "text": r.get("word", ""),
                            "type": r.get("entity_group", "MISC"),
                            "pos": None,
                            "score": round(float(r.get("score", 0.8)), 3),
                        }
                        for r in raw
                    ],
                    text,
                )
            except Exception:
                entities = _rule_based_ner(text)
        else:
            entities = _rule_based_ner(text)

        self._extract_cache[key] = [dict(e) for e in entities]
        return entities

    def extract_from_document(self, doc: Dict) -> Dict:
        text = doc.get("full_text", doc.get("content", ""))
        raw = self.extract(text)

        seen = set()
        unique: List[Dict] = []
        for e in raw:
            key = (
                e.get("start"),
                e.get("end"),
                e.get("type"),
                e.get("text", "").lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)

        out = dict(doc)
        out["entities"] = unique
        return out

    def batch_extract(self, documents: List[Dict], log_every: int = 500) -> List[Dict]:
        print(f"[NER] Xử lý {len(documents)} bài (backend={self._backend_name})...")
        result: List[Dict] = []
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


def get_entities_by_type(entities: List[Dict], entity_type: str) -> List[str]:
    return [e.get("text", "") for e in entities if e.get("type") == entity_type]


if __name__ == "__main__":
    ner = VietnameseNER()
    text = "Putin gặp Zelensky tại Ukraine để đàm phán hòa bình. WHO cảnh báo dịch COVID-19 tại Hà Nội."
    for e in ner.extract(text):
        print(e)
