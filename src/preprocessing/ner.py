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
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from vncorenlp import VnCoreNLP as _VnCoreNLP

    _VNCORENLP_AVAILABLE = True
except ImportError:
    _VNCORENLP_AVAILABLE = False

try:
    from underthesea import (
        ner as _underthesea_ner,
        pos_tag as _underthesea_pos_tag,
        word_tokenize as _underthesea_word_tokenize,
    )

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

_COREFERENCE_PRONOUNS = {
    "ông": "PER",
    "ông ấy": "PER",
    "bà": "PER",
    "bà ấy": "PER",
    "anh ấy": "PER",
    "cô ấy": "PER",
    "họ": None,
    "tổ chức này": "ORG",
    "công ty này": "ORG",
    "tập đoàn này": "ORG",
    "thành phố này": "LOC",
    "quốc gia này": "LOC",
    "nước này": "LOC",
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


def _guess_entity_type(surface: str, entities: List[Dict]) -> str:
    lowered = surface.lower()

    for ent_type, lexicon in (
        ("PER", _PERSON_DICT),
        ("LOC", _LOCATION_DICT),
        ("ORG", _ORG_DICT),
        ("MISC", _MISC_DICT),
    ):
        if any(lowered == item.lower() for item in lexicon):
            return ent_type

    votes: Dict[str, int] = {}
    for entity in entities:
        entity_text = entity.get("text", "").lower()
        if entity_text and (entity_text in lowered or lowered in entity_text):
            ent_type = entity.get("type", "MISC")
            votes[ent_type] = votes.get(ent_type, 0) + 1
    if votes:
        return max(votes.items(), key=lambda item: item[1])[0]

    if surface.isupper() and len(surface) <= 8:
        return "ORG"
    return "MISC"


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

        try:
            tokenized = _underthesea_word_tokenize(text)
            tagged = _underthesea_pos_tag(
                " ".join(tokenized) if isinstance(tokenized, list) else tokenized
            )
        except Exception:
            tagged = []

        existing = {entity["text"].lower() for entity in entities}
        buffer: List[str] = []

        def _flush_buffer():
            if not buffer:
                return
            surface = " ".join(buffer).replace("_", " ").strip()
            buffer.clear()
            if len(surface.split()) < 2:
                return
            if surface.lower() in existing:
                return
            entities.append(
                {
                    "text": surface,
                    "type": _guess_entity_type(surface, entities),
                    "pos": "Np",
                    "score": 0.78,
                }
            )
            existing.add(surface.lower())

        for token, pos in tagged:
            if pos == "Np":
                buffer.append(token)
            else:
                _flush_buffer()
        _flush_buffer()

        for match in re.finditer(
            r"\b(?:[A-ZĐ][\wÀ-ỹ.-]*\s+){1,}[A-ZĐ][\wÀ-ỹ.-]*\b",
            text,
            flags=re.UNICODE,
        ):
            surface = match.group().strip()
            if surface.lower() in existing:
                continue
            entities.append(
                {
                    "text": surface,
                    "type": _guess_entity_type(surface, entities),
                    "pos": "Np",
                    "score": 0.76,
                }
            )
            existing.add(surface.lower())

        return _attach_spans(entities, text)


class VietnameseNER:
    def __init__(
        self,
        vncorenlp_jar: Optional[str] = None,
        vncorenlp_port: int = 9000,
        use_transformer: bool = False,
        transformer_model: str = "NlpHUST/ner-vietnamese-electra-base",
        use_model: bool = None,
        cache_path: Optional[str] = None,
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
                if cache_path:
                    self.load_cache(cache_path)
                return
            except Exception as e:
                print(f"[NER] VnCoreNLP lỗi: {e}")

        if use_transformer and _TRANSFORMERS_AVAILABLE:
            try:
                self._hf_pipe = _hf_pipeline(
                    "ner", model=transformer_model, aggregation_strategy="simple"
                )
                self._backend_name = "transformer"
                print(f"[NER] Backend: HuggingFace ({transformer_model})")
                if cache_path:
                    self.load_cache(cache_path)
                return
            except Exception as e:
                print(f"[NER] Transformer lỗi: {e}")

        if _UNDERTHESEA_AVAILABLE:
            self._backend = _UndertheseaBackend()
            self._backend_name = "underthesea"
            print("[NER] Backend: underthesea")
            if cache_path:
                self.load_cache(cache_path)
            return

        print("[NER] Backend: rule-based (fallback)")
        if cache_path:
            self.load_cache(cache_path)

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

    def load_cache(self, cache_path: str):
        path = Path(cache_path)
        if not path.exists():
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._extract_cache = {
                    str(key): list(value) for key, value in data.items()
                }
                print(f"[NER] Đã load cache: {path} ({len(self._extract_cache)} keys)")
        except Exception as e:
            print(f"[NER] Không load được cache {path}: {e}")

    def save_cache(self, cache_path: str):
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._extract_cache, f, ensure_ascii=False)

    def close(self):
        if isinstance(self._backend, _VnCoreNLPBackend):
            self._backend.close()

    @property
    def backend_name(self) -> str:
        return self._backend_name


def get_entities_by_type(entities: List[Dict], entity_type: str) -> List[str]:
    return [e.get("text", "") for e in entities if e.get("type") == entity_type]


def _documents_fingerprint(documents: List[Dict]) -> str:
    sha = hashlib.sha1()
    for doc in documents:
        sha.update(str(doc.get("id", "")).encode("utf-8"))
        sha.update(b"|")
        sha.update(str(doc.get("url", "")).encode("utf-8"))
        sha.update(b"|")
        sha.update(str(doc.get("date", "")).encode("utf-8"))
        sha.update(b"\n")
    return sha.hexdigest()


def ner_with_checkpoint(
    documents: List[Dict],
    ner: VietnameseNER,
    checkpoint_path: str,
    cache_path: Optional[str] = None,
    results_path: Optional[str] = None,
    log_every: int = 500,
) -> List[Dict]:
    checkpoint_file = Path(checkpoint_path)
    results_file = Path(results_path or checkpoint_file.with_suffix(".jsonl"))
    dataset_fingerprint = _documents_fingerprint(documents)
    start_idx = 0
    result: List[Dict] = []

    if cache_path:
        ner.load_cache(cache_path)

    if checkpoint_file.exists() and results_file.exists():
        try:
            with open(checkpoint_file, encoding="utf-8") as f:
                checkpoint = json.load(f)
            if checkpoint.get("fingerprint") == dataset_fingerprint:
                start_idx = int(checkpoint.get("next_index", 0))
                with open(results_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            result.append(json.loads(line))
                if len(result) < start_idx:
                    start_idx = len(result)
                else:
                    result = result[:start_idx]
                if start_idx:
                    print(f"[NER] Resume từ checkpoint: {start_idx}/{len(documents)}")
        except Exception as e:
            print(f"[NER] Bỏ qua checkpoint lỗi: {e}")
            start_idx = 0
            result = []

    file_mode = "a" if start_idx > 0 else "w"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, file_mode, encoding="utf-8") as sink:
        for i in range(start_idx, len(documents)):
            processed = ner.extract_from_document(documents[i])
            result.append(processed)
            sink.write(json.dumps(processed, ensure_ascii=False) + "\n")

            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "fingerprint": dataset_fingerprint,
                            "next_index": i + 1,
                            "total": len(documents),
                            "completed": (i + 1) == len(documents),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                if cache_path:
                    ner.save_cache(cache_path)
                print(f"  [{i+1}/{len(documents)}]")

    if cache_path:
        ner.save_cache(cache_path)
    return result


def resolve_coreference(documents: List[Dict]) -> List[Dict]:
    resolved_docs: List[Dict] = []

    for doc in documents:
        entities = sorted(
            [dict(entity) for entity in doc.get("entities", [])],
            key=lambda entity: (entity.get("start", -1), entity.get("end", -1)),
        )
        text = doc.get("full_text", doc.get("content", ""))
        if not text or not entities:
            resolved_docs.append(doc)
            continue

        history: List[Dict] = []
        coref_entities: List[Dict] = []
        sentence_spans = _split_sentences_with_offsets(text)

        for sent in sentence_spans:
            sent_entities = [
                entity
                for entity in entities
                if sent["start"] <= entity.get("start", -1) < sent["end"]
            ]
            sent_lower = sent["text"].lower()

            for pronoun, expected_type in _COREFERENCE_PRONOUNS.items():
                for match in re.finditer(rf"\b{re.escape(pronoun)}\b", sent_lower):
                    antecedent = None
                    for candidate in reversed(history):
                        if expected_type is None or candidate.get("type") == expected_type:
                            antecedent = candidate
                            break
                    if not antecedent:
                        continue

                    mention_text = sent["text"][match.start() : match.end()]
                    coref_entities.append(
                        {
                            "text": antecedent.get("text", mention_text),
                            "resolved_text": antecedent.get("text", mention_text),
                            "mention_text": mention_text,
                            "type": antecedent.get("type", "MISC"),
                            "start": sent["start"] + match.start(),
                            "end": sent["start"] + match.end(),
                            "sentence_id": sent["sentence_id"],
                            "pos": "PRO",
                            "score": round(
                                float(antecedent.get("score", 0.8)) * 0.85, 3
                            ),
                            "entity_text": antecedent.get("text", mention_text),
                            "entity_type": antecedent.get("type", "MISC"),
                            "coref": True,
                        }
                    )

            history.extend(sent_entities)

        merged = entities + coref_entities
        seen = set()
        deduped = []
        for entity in merged:
            key = (
                entity.get("start"),
                entity.get("end"),
                entity.get("type"),
                entity.get("text", "").lower(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entity)

        out = dict(doc)
        out["entities"] = deduped
        resolved_docs.append(out)

    return resolved_docs


if __name__ == "__main__":
    ner = VietnameseNER()
    text = "Putin gặp Zelensky tại Ukraine để đàm phán hòa bình. WHO cảnh báo dịch COVID-19 tại Hà Nội."
    for e in ner.extract(text):
        print(e)
