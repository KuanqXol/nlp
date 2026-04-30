"""
ner.py
──────
NER tieng Viet dung PhoBERT fine-tuned (vinai/phobert-base-v2).

Yeu cau:
  - Checkpoint da fine-tune tai data/ner_model/
  - transformers, torch, py_vncorenlp, java 8+

Output entity chuan:
{
    "text":        "Vladimir Putin",
    "type":        "PER",          # PER | LOC | ORG
    "start":       15,
    "end":         30,
    "sentence_id": 0,
    "score":       0.92,
    "entity_text": "Vladimir Putin",
    "entity_type": "PER",
}
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> List[Dict]:
    spans = []
    for i, m in enumerate(re.finditer(r"[^.!?]+[.!?]?", text, flags=re.UNICODE)):
        sent = m.group().strip()
        if not sent:
            continue
        start = m.start()
        while start < len(text) and text[start].isspace():
            start += 1
        spans.append(
            {"sentence_id": i, "text": sent, "start": start, "end": start + len(sent)}
        )
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


def _make_entity(text, ent_type, start, end, sentence_id, score) -> Dict:
    return {
        "text": text,
        "type": ent_type,
        "start": start,
        "end": end,
        "sentence_id": sentence_id,
        "score": score,
        "entity_text": text,
        "entity_type": ent_type,
    }


# ── PhoBERT NER Backend ───────────────────────────────────────────────────────


class _PhoBERTNERBackend:
    """
    Token classification voi PhoBERT fine-tuned tren VLSP2016 NER.
    Labels: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG

    - Word segmentation bang VnCoreNLP (bat buoc de dong nhat voi pretraining)
    - Batch tokenize that: nhanh hon 3-5x tren GPU
    - word_ids bang SentencePiece prefix (U+2581): khong dung word_ids() cua fast tokenizer
    """

    _BIO_MAP = {
        "B-PER": "PER",
        "I-PER": "PER",
        "B-LOC": "LOC",
        "I-LOC": "LOC",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
    }

    def __init__(self, model_dir: str, max_length: int = 256, batch_size: int = None):
        self.model_dir = model_dir
        self.max_length = max_length
        self._batch_size_override = batch_size
        self._model = None
        self._tokenizer = None
        self._device = None
        self._segmenter = None

    def _load(self):
        if self._model is not None:
            return
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Can cai transformers va torch de dung PhoBERT NER.")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self._batch_size_override or (
            64 if self._device.type == "cuda" else 32
        )
        print(
            f"[NER/PhoBERT] Dang load tu {self.model_dir} (device={self._device}, batch_size={self.batch_size})"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()

        # VnCoreNLP word segmenter -- bat buoc de PhoBERT tokenize dung
        # "Ho Chi Minh" -> "Ho_Chi_Minh" -> PhoBERT nhan ra 1 entity
        self._segmenter = None
        try:
            import py_vncorenlp
            from pathlib import Path as _Path

            save_dir = str(_Path(self.model_dir).resolve().parent / "vncorenlp")
            _Path(save_dir).mkdir(parents=True, exist_ok=True)

            # Chi download neu jar chua co -- tranh dung wget tren Windows
            _jar = _Path(save_dir) / "VnCoreNLP-1.2.jar"
            if not _jar.exists():
                try:
                    py_vncorenlp.download_model(save_dir=save_dir)
                except Exception:
                    pass

            self._segmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=save_dir
            )
            print("[NER/PhoBERT] VnCoreNLP word segmenter ready.")
        except Exception as e:
            print(f"[NER/PhoBERT] WARNING: VnCoreNLP unavailable ({e}).")
            print(
                "[NER/PhoBERT] Falling back to whitespace split -- NER quality will be lower."
            )
            print(
                "[NER/PhoBERT] Fix: dam bao data/vncorenlp/VnCoreNLP-1.2.jar ton tai va Java da cai."
            )

        print("[NER/PhoBERT] San sang.")

    def _segment(self, text: str) -> str:
        """Word-segment: 'Ha Noi' -> 'Ha_Noi'. Fallback: tra nguyen text."""
        if self._segmenter is None:
            return text
        try:
            result = self._segmenter.word_segment(text)
            return " ".join(result) if isinstance(result, list) else str(result)
        except Exception:
            return text

    def _desegment_entity(self, seg_text: str) -> str:
        """Khoi phuc entity text ve dang goc: 'Ha_Noi' -> 'Ha Noi'."""
        return seg_text.replace("_", " ")

    def annotate(self, text: str) -> List[Dict]:
        self._load()
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        all_entities: List[Dict] = []

        for batch_start in range(0, len(sentences), self.batch_size):
            batch = sentences[batch_start : batch_start + self.batch_size]
            segmented_texts = [self._segment(s["text"]) for s in batch]
            for sent_info, sent_ents in zip(
                batch, self._predict_batch(segmented_texts)
            ):
                for ent in sent_ents:
                    orig_ent_text = self._desegment_entity(ent["text"])
                    # Dung re.search thay str.find -- chinh xac hon voi ky tu dac biet
                    _m = re.search(re.escape(orig_ent_text), sent_info["text"])
                    if _m:
                        gs = sent_info["start"] + _m.start()
                        ge = sent_info["start"] + _m.end()
                    else:
                        gs = ge = -1
                    all_entities.append(
                        _make_entity(
                            orig_ent_text,
                            ent["type"],
                            gs,
                            ge,
                            sent_info["sentence_id"],
                            ent["score"],
                        )
                    )

        # Dedup
        seen, unique = set(), []
        for e in all_entities:
            key = (e["start"], e["end"], e["type"], e["text"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        texts: list cau da segment ("Ha_Noi la thu_do Viet_Nam")

        - Batch tokenize that -> nhanh hon 3-5x tren GPU
        - Dung SentencePiece prefix U+2581 (▁) de map token -> word index
          PhobertTokenizer la slow BPE tokenizer, khong co word_ids()
        """
        if not texts:
            return []

        # Tokenize ca batch 1 lan
        enc = self._tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            logits = self._model(**enc).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        pred_probs = probs.cpu().numpy()

        id2label = self._model.config.id2label
        special_tokens = set(self._tokenizer.all_special_tokens)
        results = []

        for b_idx, text in enumerate(texts):
            words = text.split()
            input_ids = enc["input_ids"][b_idx].cpu().tolist()
            tokens = self._tokenizer.convert_ids_to_tokens(input_ids)

            # Map token -> word index dung SentencePiece prefix ▁
            # PhoBERT BPE: token bat dau word moi co prefix ▁ (U+2581)
            w_ids = []
            word_idx = -1
            for tok in tokens:
                if tok in special_tokens or tok is None:
                    w_ids.append(None)
                    continue
                if tok.startswith("\u2581"):  # bat dau word moi
                    word_idx += 1
                elif word_idx == -1:  # token dau tien khong co prefix
                    word_idx = 0
                # Guard tranh overflow khi truncate
                w_ids.append(word_idx if word_idx < len(words) else None)

            # Aggregate: lay token dau tien cua moi word
            word_preds: Dict[int, tuple] = {}
            for t_idx, wid in enumerate(w_ids):
                if wid is None:
                    continue
                if wid not in word_preds:
                    label_id = int(pred_ids[b_idx][t_idx])
                    prob = float(pred_probs[b_idx][t_idx][label_id])
                    word_preds[wid] = (label_id, prob)

            # Decode BIO -> entity spans
            entities = []
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
                span_words, span_probs = [words[i]], [prob]
                j = i + 1
                while j < len(words) and j in word_preds:
                    nl_id, np_ = word_preds[j]
                    nl = id2label.get(nl_id, "O")
                    if nl.startswith("I-") and self._BIO_MAP.get(nl) == ent_type:
                        span_words.append(words[j])
                        span_probs.append(np_)
                        j += 1
                    else:
                        break
                entities.append(
                    {
                        "text": " ".join(span_words),
                        "type": ent_type,
                        "score": round(sum(span_probs) / len(span_probs), 4),
                    }
                )
                i = j
            results.append(entities)

        return results

    def close(self):
        self._model = None
        self._tokenizer = None
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Underthesea Fallback NER Backend ─────────────────────────────────────────


class _UndertheseaNERBackend:
    """
    Fallback NER dung underthesea khi khong co PhoBERT model.
    Chat luong thap hon PhoBERT fine-tuned, nhung khong can training.
    Cai dat: pip install underthesea
    """

    _TYPE_MAP = {
        "B-PER": "PER",
        "I-PER": "PER",
        "B-LOC": "LOC",
        "I-LOC": "LOC",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
        "B-MISC": "MISC",
        "I-MISC": "MISC",
    }

    def __init__(self):
        self._ner_fn = None
        self._load()

    def _load(self):
        try:
            from underthesea import ner as _ner

            self._ner_fn = _ner
            print("[NER/underthesea] Backend san sang.")
        except ImportError:
            print("[NER/underthesea] WARNING: chua cai underthesea.")
            print("[NER/underthesea] Cai: pip install underthesea")
            self._ner_fn = None

    def annotate(self, text: str) -> List[Dict]:
        if not self._ner_fn or not text.strip():
            return []
        try:
            raw = self._ner_fn(text)
        except Exception as e:
            print(f"[NER/underthesea] Loi: {e}")
            return []

        entities: List[Dict] = []
        i = 0
        while i < len(raw):
            word, _, _, tag = raw[i][0], raw[i][1], raw[i][2], raw[i][3]
            etype = self._TYPE_MAP.get(tag)
            if etype is None:
                i += 1
                continue
            span_words = [word]
            j = i + 1
            while j < len(raw):
                nw, _, _, nt = raw[j][0], raw[j][1], raw[j][2], raw[j][3]
                if nt.startswith("I-") and self._TYPE_MAP.get(nt) == etype:
                    span_words.append(nw)
                    j += 1
                else:
                    break
            ent_text = " ".join(span_words)
            start = text.find(ent_text)
            entities.append(
                _make_entity(
                    ent_text,
                    etype,
                    start,
                    start + len(ent_text) if start >= 0 else -1,
                    0,
                    0.70,
                )
            )
            i = j
        return entities

    def close(self):
        self._ner_fn = None


# ── VietnameseNER ─────────────────────────────────────────────────────────────


class VietnameseNER:
    """
    NER tieng Viet dung PhoBERT fine-tuned.

    Cach dung:
        ner = VietnameseNER()                        # load tu data/ner_model/
        ner = VietnameseNER(model_dir="path/to/dir") # custom path
        entities = ner.extract("Ha Noi la thu do Viet Nam.")
    """

    DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "ner_model"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = None,
        cache_path: Optional[str] = None,
        **kwargs,
    ):
        resolved_dir = model_dir or str(self.DEFAULT_MODEL_DIR)
        self._cache: Dict[str, List[Dict]] = {}

        if Path(resolved_dir).exists():
            self._backend = _PhoBERTNERBackend(
                resolved_dir, max_length=max_length, batch_size=batch_size
            )
            self.backend_name = "phobert-ner"
            print(f"[NER] Backend: PhoBERT ({resolved_dir})")
        else:
            print(f"[NER] WARNING: Khong tim thay PhoBERT model tai {resolved_dir}.")
            print("[NER] Dung underthesea lam fallback NER. Chat luong se thap hon.")
            print("[NER] De dung PhoBERT: giai nen model vao data/ner_model/")
            self._backend = _UndertheseaNERBackend()
            self.backend_name = "underthesea-fallback"

        if cache_path:
            self.load_cache(cache_path)

    def _cache_key(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def load_cache(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            self._cache = {str(k): list(v) for k, v in data.items()}
            print(f"[NER] Cache loaded: {p} ({len(self._cache)} entries)")
        except Exception as e:
            print(f"[NER] Khong load duoc cache {p}: {e}")

    def save_cache(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)

    def extract(self, text: str) -> List[Dict]:
        text = _normalize_text(text)
        if not text:
            return []
        key = self._cache_key(text)
        if key in self._cache:
            return [dict(e) for e in self._cache[key]]
        entities = self._backend.annotate(text)
        self._cache[key] = [dict(e) for e in entities]
        return entities

    def extract_from_document(self, doc: Dict) -> Dict:
        text = doc.get("full_text", doc.get("content", ""))
        entities = self.extract(text)
        seen, unique = set(), []
        for e in entities:
            key = (
                e.get("start"),
                e.get("end"),
                e.get("type"),
                e.get("text", "").lower(),
            )
            if key not in seen:
                seen.add(key)
                unique.append(e)
        out = dict(doc)
        out["entities"] = unique
        return out

    def batch_extract(self, documents: List[Dict], log_every: int = 500) -> List[Dict]:
        print(f"[NER] Xu ly {len(documents)} bai...")
        result = []
        for i, doc in enumerate(documents):
            result.append(self.extract_from_document(doc))
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                print(f"  [{i+1}/{len(documents)}]")
        print("[NER] Hoan thanh.")
        return result

    def close(self):
        self._backend.close()


# ── Utilities ─────────────────────────────────────────────────────────────────


def get_entities_by_type(entities: List[Dict], entity_type: str) -> List[str]:
    return [e.get("text", "") for e in entities if e.get("type") == entity_type]


def ner_with_checkpoint(
    documents: List[Dict],
    ner: VietnameseNER,
    checkpoint_path: str,
    cache_path: Optional[str] = None,
    results_path: Optional[str] = None,
    log_every: int = 500,
) -> List[Dict]:
    """
    Batch NER voi checkpoint de resume neu bi ngat giua chung.
    Checkpoint luu tien do vao JSON, ket qua append vao JSONL.
    """
    import os

    checkpoint_file = Path(checkpoint_path)
    results_file = Path(results_path or checkpoint_file.with_suffix(".jsonl"))

    sha = hashlib.sha1()
    for doc in documents:
        sha.update(str(doc.get("id", "")).encode())
        sha.update(b"|")
    fingerprint = sha.hexdigest()

    start_idx = 0
    result: List[Dict] = []

    if cache_path:
        ner.load_cache(cache_path)

    if checkpoint_file.exists() and results_file.exists():
        try:
            with open(checkpoint_file, encoding="utf-8") as f:
                ckpt = json.load(f)
            if ckpt.get("fingerprint") == fingerprint:
                start_idx = int(ckpt.get("next_index", 0))
                with open(results_file, encoding="utf-8") as f:
                    result = [json.loads(l) for l in f if l.strip()]
                result = result[:start_idx]
                if start_idx:
                    print(f"[NER] Resume tu checkpoint: {start_idx}/{len(documents)}")
        except Exception as e:
            print(f"[NER] Bo qua checkpoint loi: {e}")
            start_idx, result = 0, []

    def _save_checkpoint(idx: int):
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "fingerprint": fingerprint,
                    "next_index": idx,
                    "total": len(documents),
                },
                f,
            )
        if cache_path:
            ner.save_cache(cache_path)

    file_mode = "a" if start_idx > 0 else "w"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, file_mode, encoding="utf-8") as sink:
        for i in range(start_idx, len(documents)):
            processed = ner.extract_from_document(documents[i])
            result.append(processed)
            sink.write(json.dumps(processed, ensure_ascii=False) + "\n")
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                _save_checkpoint(i + 1)
                print(f"  [{i+1}/{len(documents)}] checkpoint saved")

    if cache_path:
        ner.save_cache(cache_path)

    print(f"[NER] Hoan tat {len(result)}/{len(documents)} docs")
    return result
