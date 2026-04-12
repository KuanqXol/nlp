"""
NER tiếng Việt dùng PhoBERT fine-tuned (vinai/phobert-base-v2).

Yêu cầu:
  - Checkpoint đã fine-tune tại data/ner_model/  (chạy scripts/train_ner.py trước)
  - transformers, torch

Output entity chuẩn:
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

# Dùng shared sentence splitter — nhất quán với chunking.py
from src.utils.text import split_sentences_spans as _split_sentences

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Chuẩn hóa unicode NFC, bỏ khoảng trắng thừa."""
    text = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", text).strip()


# _split_sentences được import từ src.utils.text (split_sentences_spans)
# Xem: from src.utils.text import split_sentences_spans as _split_sentences


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
    Token classification với PhoBERT fine-tuned trên VLSP2016 NER.
    Labels: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
    """

    _BIO_MAP = {
        "B-PER": "PER",
        "I-PER": "PER",
        "B-LOC": "LOC",
        "I-LOC": "LOC",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
    }

    def __init__(self, model_dir: str, max_length: int = 256, batch_size: int = 32):
        self.model_dir = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        if not _TORCH_AVAILABLE:
            raise RuntimeError("Cần cài transformers và torch để dùng PhoBERT NER.")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[NER/PhoBERT] Đang load từ {self.model_dir} (device={self._device})")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self._model.to(self._device)
        self._model.eval()
        print("[NER/PhoBERT] Sẵn sàng.")

    def annotate(self, text: str) -> List[Dict]:
        self._load()
        if not text or not text.strip():
            return []

        sentences = _split_sentences(text)
        all_entities: List[Dict] = []

        for batch_start in range(0, len(sentences), self.batch_size):
            batch = sentences[batch_start : batch_start + self.batch_size]
            for sent_info, sent_ents in zip(
                batch, self._predict_batch([s["text"] for s in batch])
            ):
                for ent in sent_ents:
                    local_start = sent_info["text"].find(ent["text"])
                    if local_start >= 0:
                        gs = sent_info["start"] + local_start
                        ge = gs + len(ent["text"])
                    else:
                        gs = ge = -1
                    all_entities.append(
                        _make_entity(
                            ent["text"],
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
        results = []
        for text in texts:
            words = text.split()
            if not words:
                results.append([])
                continue

            enc = self._tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}

            with torch.no_grad():
                logits = self._model(**enc).logits

            probs = torch.nn.functional.softmax(logits, dim=-1)
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            pred_probs = probs[0].cpu().numpy()

            # word_ids từ tokenizer (không có return_tensors)
            plain_enc = self._tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
            )
            w_ids = plain_enc.word_ids()

            # Aggregate: first sub-token per word
            word_preds: Dict[int, tuple] = {}
            for tok_idx, wid in enumerate(w_ids):
                if wid is None or tok_idx >= len(pred_ids):
                    continue
                if wid not in word_preds:
                    word_preds[wid] = (
                        int(pred_ids[tok_idx]),
                        float(pred_probs[tok_idx, int(pred_ids[tok_idx])]),
                    )

            id2label = self._model.config.id2label
            entities, i = [], 0
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


# ── VietnameseNER ─────────────────────────────────────────────────────────────


class VietnameseNER:
    """
    NER tiếng Việt dùng PhoBERT fine-tuned.

    Cách dùng:
        ner = VietnameseNER()                        # load từ data/ner_model/
        ner = VietnameseNER(model_dir="path/to/dir") # custom path
        entities = ner.extract("Hà Nội là thủ đô Việt Nam.")
    """

    DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "ner_model"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        max_length: int = 256,
        batch_size: int = 32,
        cache_path: Optional[str] = None,
        # backward-compat kwargs (bỏ qua)
        **kwargs,
    ):
        resolved_dir = model_dir or str(self.DEFAULT_MODEL_DIR)
        if not Path(resolved_dir).exists():
            raise FileNotFoundError(
                f"[NER] Không tìm thấy model tại: {resolved_dir}\n"
                "Chạy: python scripts/train_ner.py --output-dir data/ner_model"
            )

        self._backend = _PhoBERTNERBackend(
            resolved_dir, max_length=max_length, batch_size=batch_size
        )
        self._cache: Dict[str, List[Dict]] = {}
        self.backend_name = "phobert-ner"
        print(f"[NER] Backend: PhoBERT ({resolved_dir})")

        if cache_path:
            self.load_cache(cache_path)

    # ── Cache ─────────────────────────────────────────────────────────────

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
            print(f"[NER] Không load được cache {p}: {e}")

    def save_cache(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False)

    # ── Extraction ────────────────────────────────────────────────────────

    def extract(self, text: str) -> List[Dict]:
        """Trích xuất entity từ text. Dùng cache nếu đã xử lý."""
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
        # Dedup
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
        print(f"[NER] Xử lý {len(documents)} bài...")
        result = []
        for i, doc in enumerate(documents):
            result.append(self.extract_from_document(doc))
            if (i + 1) % log_every == 0 or (i + 1) == len(documents):
                print(f"  [{i+1}/{len(documents)}]")
        print("[NER] Hoàn thành.")
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
    Batch NER với checkpoint để resume nếu bị ngắt giữa chừng.
    Checkpoint lưu tiến độ vào JSON, kết quả append vào JSONL.
    """
    import os

    checkpoint_file = Path(checkpoint_path)
    results_file = Path(results_path or checkpoint_file.with_suffix(".jsonl"))

    # Fingerprint để xác minh dataset không đổi
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
                    print(f"[NER] Resume từ checkpoint: {start_idx}/{len(documents)}")
        except Exception as e:
            print(f"[NER] Bỏ qua checkpoint lỗi: {e}")
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

    print(f"[NER] Hoàn tất {len(result)}/{len(documents)} docs")
    return result
