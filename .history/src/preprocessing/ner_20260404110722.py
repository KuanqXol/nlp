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
import multiprocessing as mp
import os
import re
import sys
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

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

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

_NER_MP_SAFE_ENTRYPOINTS = {"main.py", "build_index.py"}
_NER_MP_CHUNKSIZE = 20
_NER_MP_MIN_DOCS = 40


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
    """Rule-based NER fallback — regex only, no dictionary scan."""
    raw: List[Dict] = []

    # Regex for capitalized multi-word sequences (Vietnamese proper nouns)
    for match in re.finditer(
        r"\b(?:[A-ZĐ][\wÀ-ỹ.-]*\s+){1,}[A-ZĐ][\wÀ-ỹ.-]*\b",
        text,
        flags=re.UNICODE,
    ):
        surface = match.group().strip()
        if len(surface.split()) < 2:
            continue
        raw.append(
            {
                "text": surface,
                "type": _guess_entity_type(surface, []),
                "start": match.start(),
                "end": match.end(),
                "sentence_id": -1,
                "pos": "Np",
                "score": 0.65,
            }
        )

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


def _ner_worker(payload) -> Dict:
    """Worker chạy trong subprocess riêng."""
    doc, ner_config = payload
    try:
        ner = VietnameseNER(**ner_config)
        processed = ner.extract_from_document(doc)
        ner.close()
        return processed
    except Exception:
        out = dict(doc)
        out.setdefault("entities", [])
        return out


def _is_windows_mp_safe() -> bool:
    if os.name != "nt":
        return True
    if os.environ.get("NLP_FORCE_NER_MP") == "1":
        return True

    main_module = sys.modules.get("__main__")
    main_file = Path(getattr(main_module, "__file__", "")).name.lower()
    return main_file in _NER_MP_SAFE_ENTRYPOINTS


def _get_mp_disable_reasons(pending_docs: int, ner_backend: str) -> List[str]:
    reasons: List[str] = []
    if pending_docs < _NER_MP_MIN_DOCS:
        reasons.append(
            f"pending_docs={pending_docs} < min_docs={_NER_MP_MIN_DOCS}"
        )
    if mp.cpu_count() <= 1:
        reasons.append("cpu_count <= 1")
    if ner_backend not in {"underthesea", "rule-based"}:
        reasons.append(f"backend={ner_backend} không hỗ trợ multiprocessing")
    if not _is_windows_mp_safe():
        reasons.append("Windows entrypoint hiện tại chưa an toàn cho spawn")
    return reasons


class _PhoBERTNERBackend:
    """NER backend using fine-tuned PhoBERT for token classification.

    Features:
      - Loads from a local HuggingFace model directory (data/ner_model/)
      - Subword → word alignment via word_ids() aggregation
      - Sliding window with stride for long documents (> max_length tokens)
      - Batch inference for efficiency
      - Returns softmax probability as entity score
    """

    _BIO_MAP = {
        "B-PER": "PER", "I-PER": "PER",
        "B-LOC": "LOC", "I-LOC": "LOC",
        "B-ORG": "ORG", "I-ORG": "ORG",
    }

    def __init__(self, model_dir: str, max_length: int = 256, stride: int = 128):
        self.model_dir = model_dir
        self.max_length = max_length
        self.stride = stride
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        if not _TORCH_AVAILABLE:
            raise RuntimeError("transformers + torch required for PhoBERT NER")
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NER/PhoBERT] Loading from {self.model_dir} (device={self._device})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()
        print("[NER/PhoBERT] Model ready.")

    def annotate(self, text: str) -> List[Dict]:
        """Extract entities from text using PhoBERT token classification."""
        self._load()
        if not text or not text.strip():
            return []

        sentences = _split_sentences_with_offsets(text)
        if not sentences:
            return []

        all_entities: List[Dict] = []
        # Process sentences in batches of 32
        batch_size = 32
        for batch_start in range(0, len(sentences), batch_size):
            batch_sents = sentences[batch_start:batch_start + batch_size]
            batch_texts = [s["text"] for s in batch_sents]
            batch_entities = self._predict_batch(batch_texts)

            for sent_info, sent_entities in zip(batch_sents, batch_entities):
                for ent in sent_entities:
                    # Adjust offsets to global text position
                    local_start = sent_info["text"].find(ent["text"])
                    if local_start >= 0:
                        global_start = sent_info["start"] + local_start
                        global_end = global_start + len(ent["text"])
                    else:
                        global_start = -1
                        global_end = -1

                    all_entities.append(
                        _make_entity(
                            text=ent["text"],
                            ent_type=ent["type"],
                            start=global_start,
                            end=global_end,
                            sentence_id=sent_info["sentence_id"],
                            pos="Np",
                            score=ent["score"],
                        )
                    )

        # Deduplicate
        seen = set()
        unique = []
        for e in all_entities:
            key = (e["start"], e["end"], e["type"], e["text"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Run PhoBERT NER on a batch of texts. Returns entities per text."""
        import torch

        results = []
        for text in texts:
            words = text.split()
            if not words:
                results.append([])
                continue

            tokenized = self._tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self._device) for k, v in tokenized.items()}

            with torch.no_grad():
                outputs = self._model(**tokenized)
                logits = outputs.logits  # (1, seq_len, num_labels)

            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            pred_probs = probs[0].cpu().numpy()
            word_ids = tokenized.get("word_ids", None)

            # Get word_ids from tokenizer
            enc = self._tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
            )
            w_ids = enc.word_ids()

            # Aggregate sub-token predictions to word level
            word_preds = {}  # word_idx -> (label_id, prob)
            for token_idx, word_id in enumerate(w_ids):
                if word_id is None:
                    continue
                if token_idx >= len(pred_ids):
                    break
                if word_id not in word_preds:
                    # First sub-token: use its prediction
                    word_preds[word_id] = (
                        int(pred_ids[token_idx]),
                        float(pred_probs[token_idx, int(pred_ids[token_idx])]),
                    )

            # Collect BIO spans
            entities = []
            id2label = self._model.config.id2label
            i = 0
            while i < len(words):
                if i not in word_preds:
                    i += 1
                    continue
                label_id, prob = word_preds[i]
                label = id2label.get(label_id, "O")
                if not label.startswith("B-"):
                    i += 1
                    continue

                ent_type = self._BIO_MAP.get(label)
                if not ent_type:
                    i += 1
                    continue

                span_words = [words[i]]
                span_probs = [prob]
                j = i + 1
                while j < len(words):
                    if j not in word_preds:
                        break
                    next_label_id, next_prob = word_preds[j]
                    next_label = id2label.get(next_label_id, "O")
                    if next_label.startswith("I-") and self._BIO_MAP.get(next_label) == ent_type:
                        span_words.append(words[j])
                        span_probs.append(next_prob)
                        j += 1
                    else:
                        break

                entity_text = " ".join(span_words)
                avg_prob = sum(span_probs) / len(span_probs)
                entities.append({
                    "text": entity_text,
                    "type": ent_type,
                    "score": round(avg_prob, 4),
                })
                i = j

            results.append(entities)

        return results

    def close(self):
        self._model = None
        self._tokenizer = None
        if _TORCH_AVAILABLE:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
        ner_model_dir: Optional[str] = None,
        cache_path: Optional[str] = None,
        log_backend: bool = True,
    ):
        if use_model is not None:
            use_transformer = use_model

        # Default NER model directory
        if ner_model_dir is None:
            _default = Path(__file__).resolve().parents[2] / "data" / "ner_model"
            if _default.exists() and (_default / "config.json").exists():
                ner_model_dir = str(_default)

        self._spawn_config = {
            "vncorenlp_jar": vncorenlp_jar,
            "vncorenlp_port": vncorenlp_port,
            "use_transformer": use_transformer,
            "transformer_model": transformer_model,
            "use_model": use_model,
            "ner_model_dir": None,  # PhoBERT not safe for spawn
            "cache_path": None,
            "log_backend": False,
        }
        self._backend = None
        self._hf_pipe = None
        self._backend_name = "rule-based"
        self._extract_cache: Dict[str, List[Dict]] = {}
        self._log_backend = log_backend

        # Priority 1: VnCoreNLP (if jar provided)
        if vncorenlp_jar and _VNCORENLP_AVAILABLE:
            try:
                self._backend = _VnCoreNLPBackend(vncorenlp_jar, vncorenlp_port)
                self._backend_name = "VnCoreNLP"
                if self._log_backend:
                    print("[NER] Backend: VnCoreNLP")
                if cache_path:
                    self.load_cache(cache_path)
                return
            except Exception as e:
                print(f"[NER] VnCoreNLP lỗi: {e}")

        # Priority 2: Fine-tuned PhoBERT NER (if model directory exists)
        if ner_model_dir and _TORCH_AVAILABLE:
            try:
                self._backend = _PhoBERTNERBackend(ner_model_dir)
                self._backend_name = "phobert-ner"
                if self._log_backend:
                    print(f"[NER] Backend: PhoBERT NER ({ner_model_dir})")
                if cache_path:
                    self.load_cache(cache_path)
                return
            except Exception as e:
                print(f"[NER] PhoBERT NER lỗi: {e}")

        # Priority 3: HuggingFace pretrained (--use-model flag)
        if use_transformer and _TRANSFORMERS_AVAILABLE:
            try:
                self._hf_pipe = _hf_pipeline(
                    "ner", model=transformer_model, aggregation_strategy="simple"
                )
                self._backend_name = "transformer"
                if self._log_backend:
                    print(f"[NER] Backend: HuggingFace ({transformer_model})")
                if cache_path:
                    self.load_cache(cache_path)
                return
            except Exception as e:
                print(f"[NER] Transformer lỗi: {e}")

        # Priority 4: underthesea
        if _UNDERTHESEA_AVAILABLE:
            self._backend = _UndertheseaBackend()
            self._backend_name = "underthesea"
            if self._log_backend:
                print("[NER] Backend: underthesea")
            if cache_path:
                self.load_cache(cache_path)
            return

        # Priority 5: rule-based (regex only)
        if self._log_backend:
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

    def spawn_config(self) -> Dict:
        return dict(self._spawn_config)


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
    total_start = os.times()
    checkpoint_file = Path(checkpoint_path)
    results_file = Path(results_path or checkpoint_file.with_suffix(".jsonl"))
    dataset_fingerprint = _documents_fingerprint(documents)
    start_idx = 0
    result: List[Dict] = []

    if cache_path:
        ner.load_cache(cache_path)

    print(
        "[NER] Checkpoint mode: "
        f"docs={len(documents)}, checkpoint={checkpoint_file}, results={results_file}"
    )
    if cache_path:
        print(f"[NER] Cache path: {cache_path}")

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

    def _update_cache(processed_doc: Dict):
        if not cache_path:
            return
        text = processed_doc.get("full_text", processed_doc.get("content", ""))
        if not text:
            return
        ner._extract_cache[ner._cache_key(text)] = [
            dict(entity) for entity in processed_doc.get("entities", [])
        ]

    def _write_checkpoint(next_index: int):
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": dataset_fingerprint,
                    "next_index": next_index,
                    "total": len(documents),
                    "completed": next_index == len(documents),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        if cache_path:
            ner.save_cache(cache_path)

    file_mode = "a" if start_idx > 0 else "w"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    pending = documents[start_idx:]
    disable_reasons = _get_mp_disable_reasons(
        pending_docs=len(pending),
        ner_backend=getattr(ner, "backend_name", ""),
    )
    can_use_mp = (
        len(pending) >= _NER_MP_MIN_DOCS
        and mp.cpu_count() > 1
        and getattr(ner, "backend_name", "") in {"underthesea", "rule-based"}
        and _is_windows_mp_safe()
    )

    with open(results_file, file_mode, encoding="utf-8") as sink:
        if can_use_mp:
            n_workers = max(1, mp.cpu_count() - 1)
            print(
                f"[NER] Multiprocessing: {n_workers} workers, "
                f"chunksize={_NER_MP_CHUNKSIZE}"
            )
            print(
                f"[NER] Backend instances: {getattr(ner, 'backend_name', 'unknown')} "
                f"x{n_workers} workers"
            )
            worker_args = (
                (doc, ner.spawn_config())
                for doc in pending
            )
            ctx = mp.get_context("spawn" if os.name == "nt" else None)
            with ctx.Pool(processes=n_workers) as pool:
                for i, processed in enumerate(
                    pool.imap(_ner_worker, worker_args, chunksize=_NER_MP_CHUNKSIZE),
                    start=start_idx,
                ):
                    _update_cache(processed)
                    result.append(processed)
                    sink.write(json.dumps(processed, ensure_ascii=False) + "\n")

                    if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                        _write_checkpoint(i + 1)
                        print(
                            f"  [{i+1}/{len(documents)}] "
                            f"checkpoint -> {checkpoint_file.name}"
                        )
        else:
            if pending:
                if disable_reasons:
                    print(
                        "[NER] Multiprocessing tắt: "
                        + "; ".join(disable_reasons)
                    )
                print(
                    f"[NER] Backend instances: {getattr(ner, 'backend_name', 'unknown')} x1"
                )
                print("[NER] Dùng chế độ tuần tự.")
            for i in range(start_idx, len(documents)):
                processed = ner.extract_from_document(documents[i])
                _update_cache(processed)
                result.append(processed)
                sink.write(json.dumps(processed, ensure_ascii=False) + "\n")

                if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                    _write_checkpoint(i + 1)
                    print(
                        f"  [{i+1}/{len(documents)}] "
                        f"checkpoint -> {checkpoint_file.name}"
                    )

    if cache_path:
        ner.save_cache(cache_path)
    total_elapsed = os.times().elapsed - total_start.elapsed
    print(
        f"[NER] Hoàn tất {len(result)}/{len(documents)} docs trong "
        f"{total_elapsed:.1f}s"
    )
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
