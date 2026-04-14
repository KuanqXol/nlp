"""
train_ner.py
────────────
Fine-tune vinai/phobert-base-v2 for Vietnamese NER on news domain.

Data strategy:
  1. Gold: data/ner_ground_truth.json (hand-labeled sentences)
  2. Silver: underthesea predictions on N news articles (filtered by confidence)

Subword alignment:
  PhoBERT uses BPE → a Vietnamese word may split into 2+ sub-tokens.
  First sub-token gets the word's BIO label; continuation sub-tokens get I- label
  (or -100 to ignore in loss, configurable).

Usage:
    python scripts/train_ner.py --help
    python scripts/train_ner.py --output-dir data/ner_model --epochs 5
    python scripts/train_ner.py --no-silver --output-dir data/ner_model_gold_only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# ── Word segmentation (VnCoreNLP RDRSegmenter) ───────────────────────────────
# PhoBERT được pretrain trên text đã word-segment: "Hà_Nội", "Việt_Nam"
# Không segment → model nhận "Hà Nội" thành 2 token riêng → NER sai hoàn toàn.

_SEGMENTER = None


def get_segmenter():
    """Load VnCoreNLP RDRSegmenter một lần, cache lại."""
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER
    try:
        import py_vncorenlp

        # Download model nếu chưa có
        save_dir = str(ROOT_DIR / "data" / "vncorenlp")
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        try:
            import os as _os

            _os.makedirs(save_dir, exist_ok=True)
            py_vncorenlp.download_model(save_dir=save_dir)
        except Exception:
            pass  # đã có rồi
        _SEGMENTER = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
        print("[Segmenter] VnCoreNLP RDRSegmenter ready.")
    except Exception as e:
        print(
            f"[Segmenter] VnCoreNLP unavailable ({e}), falling back to whitespace split."
        )
        _SEGMENTER = None
    return _SEGMENTER


def word_segment(text: str) -> str:
    """Tách từ tiếng Việt: 'Hà Nội' → 'Hà_Nội'.

    Nếu không có VnCoreNLP, fallback về whitespace (chất lượng thấp hơn).
    """
    seg = get_segmenter()
    if seg is None:
        return text
    try:
        results = seg.word_segment(text)
        # VnCoreNLP trả về list[str], mỗi string là 1 câu đã segment
        return " ".join(results) if isinstance(results, list) else str(results)
    except Exception:
        return text


# ── Label schema ─────────────────────────────────────────────────────────────

LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# ── Data loading utilities ───────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def load_vlsp2016(max_samples: int = 999999) -> List[Dict]:
    """Load VLSP2016 NER trực tiếp từ HuggingFace.
    Dataset: datnth1709/VLSP2016-NER-data
    Format: mỗi sample có 'tokens' và 'ner_tags' (BIO string list)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[train_ner] datasets not installed: pip install datasets")
        return []

    print("[train_ner] Loading VLSP2016 from HuggingFace...")
    try:
        ds = load_dataset("datnth1709/VLSP2016-NER-data")
    except Exception as e:
        print(f"[train_ner] Cannot load VLSP2016: {e}")
        return []

    # Map VLSP label strings về PER/LOC/ORG (bỏ MISC)
    VLSP_MAP = {
        "B-PER": "PER",
        "I-PER": "PER",
        "B-LOC": "LOC",
        "I-LOC": "LOC",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
        "B-MISC": None,
        "I-MISC": None,
        "O": None,
    }

    samples = []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for item in ds[split]:
            tokens = item.get("tokens") or item.get("words") or []
            ner_tags = item.get("ner_tags") or []

            # ner_tags có thể là:
            # 1. List[str]: ["O", "B-PER", ...] → dùng thẳng
            # 2. List[int]: [0, 1, 2, ...] → map qua label_names
            if ner_tags and isinstance(ner_tags[0], int):
                try:
                    feat = ds[split].features["ner_tags"]
                    # Sequence(ClassLabel) → .feature.names
                    label_names = feat.feature.names
                except AttributeError:
                    try:
                        label_names = feat.names
                    except AttributeError:
                        # Fallback mapping cứng VLSP2016
                        label_names = [
                            "O",
                            "B-PER",
                            "I-PER",
                            "B-ORG",
                            "I-ORG",
                            "B-LOC",
                            "I-LOC",
                            "B-MISC",
                            "I-MISC",
                        ]
                ner_tags = [
                    label_names[t] if t < len(label_names) else "O" for t in ner_tags
                ]

            if not tokens or not ner_tags:
                continue

            # Tái tạo entities từ BIO tags
            sentence = " ".join(tokens)
            entities = []
            i = 0
            while i < len(tokens):
                tag = ner_tags[i] if i < len(ner_tags) else "O"
                ent_type = VLSP_MAP.get(tag)
                if ent_type and tag.startswith("B-"):
                    span = [tokens[i]]
                    j = i + 1
                    while j < len(tokens):
                        next_tag = ner_tags[j] if j < len(ner_tags) else "O"
                        if (
                            next_tag.startswith("I-")
                            and VLSP_MAP.get(next_tag) == ent_type
                        ):
                            span.append(tokens[j])
                            j += 1
                        else:
                            break
                    entities.append({"text": " ".join(span), "type": ent_type})
                    i = j
                else:
                    i += 1

            if sentence and entities:
                # Word-segment để đồng nhất với PhoBERT pretraining format
                segmented = word_segment(sentence)
                seg_ents = [
                    {"text": word_segment(e["text"]), "type": e["type"]}
                    for e in entities
                ]
                samples.append(
                    {"sentence": segmented, "entities": seg_ents, "source": "vlsp2016"}
                )

        if len(samples) >= max_samples:
            break

    print(f"[train_ner] VLSP2016 loaded: {len(samples)} samples")
    return samples[:max_samples]


def load_gold_data(path: str | Path) -> List[Dict]:
    """Load ner_ground_truth.json → list of {sentence, entities}.
    Fallback: nếu file không tồn tại, trả về list rỗng (sẽ dùng VLSP2016 thay thế).
    """
    p = Path(path)
    if not p.exists():
        print(f"[train_ner] Gold file not found: {p} — skipping local gold data")
        return []
    with open(p, encoding="utf-8") as f:
        payload = json.load(f)
    samples = payload.get("samples", payload if isinstance(payload, list) else [])
    out = []
    for sample in samples:
        sentence = _normalize_text(sample.get("sentence", ""))
        entities = sample.get("entities", [])
        if sentence and entities:
            out.append({"sentence": sentence, "entities": entities})
    return out


def load_news_articles(csv_path: str | Path, max_articles: int = 10000) -> List[str]:
    """Load raw text từ CSV, stratified theo category để đảm bảo đa dạng chủ đề.

    Thay vì lấy tuần tự (dễ bị lệch nếu CSV sắp xếp theo category),
    hàm này nhóm bài theo category rồi lấy 10% mỗi category.
    Category không xác định được gom vào "unknown".
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[train_ner] CSV not found: {path}")
        return []

    # Gom tất cả bài theo category
    by_category: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if len(text) <= 50:
                continue
            cat = (row.get("category") or "unknown").strip().lower() or "unknown"
            by_category.setdefault(cat, []).append(text)

    # Thống kê phân phối
    total_available = sum(len(v) for v in by_category.values())
    print(f"[train_ner] CSV: {total_available} bai, {len(by_category)} category:")
    for cat, articles in sorted(by_category.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {cat}: {len(articles)}")
    if len(by_category) > 10:
        print(f"  ... ({len(by_category)-10} category khac)")

    # Stratified sampling: lấy tỉ lệ đều từ mỗi category
    sampled: List[str] = []
    for cat, articles in by_category.items():
        # Số bài lấy từ category này = tỉ lệ category × max_articles
        n = max(1, round(len(articles) / total_available * max_articles))
        random.shuffle(articles)
        sampled.extend(articles[:n])

    # Shuffle lại sau khi gom để tránh cluster theo category
    random.shuffle(sampled)

    # Trim hoặc bổ sung nếu lệch so với max_articles
    if len(sampled) > max_articles:
        sampled = sampled[:max_articles]
    elif len(sampled) < max_articles and total_available > len(sampled):
        # Lấy thêm từ category lớn nhất nếu thiếu
        extra = []
        for articles in sorted(by_category.values(), key=len, reverse=True):
            for a in articles:
                if a not in set(sampled):
                    extra.append(a)
                    if len(sampled) + len(extra) >= max_articles:
                        break
            if len(sampled) + len(extra) >= max_articles:
                break
        sampled.extend(extra)
        sampled = sampled[:max_articles]
        random.shuffle(sampled)

    print(
        f"[train_ner] Stratified sample: {len(sampled)} bai tu {len(by_category)} category"
    )
    return sampled


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex-based)."""
    import re

    sentences = []
    for m in re.finditer(r"[^.!?]+[.!?]?", text, flags=re.UNICODE):
        sent = m.group().strip()
        if sent and len(sent) > 10:
            sentences.append(sent)
    return sentences


# ── Silver label generation ──────────────────────────────────────────────────


def generate_silver_labels(
    articles: List[str],
    min_confidence: float = 0.85,
    max_sentences: int = 50000,
) -> List[Dict]:
    """Run underthesea NER on articles, filter by confidence."""
    try:
        from underthesea import ner as _ner
    except ImportError:
        print("[train_ner] underthesea not available, skipping silver labels")
        return []

    NER_MAP = {
        "B-PER": "PER",
        "I-PER": "PER",
        "B-LOC": "LOC",
        "I-LOC": "LOC",
        "B-ORG": "ORG",
        "I-ORG": "ORG",
    }

    silver = []
    total_sentences = 0

    for article_idx, article in enumerate(articles):
        if total_sentences >= max_sentences:
            break
        sentences = _split_sentences(article)
        for sent in sentences:
            if total_sentences >= max_sentences:
                break
            try:
                raw = _ner(sent)
            except Exception:
                continue

            entities = []
            i = 0
            all_high_conf = True

            while i < len(raw):
                word, pos, _, ner_tag = raw[i]
                ent_type = NER_MAP.get(ner_tag)
                if ent_type and ner_tag.startswith("B-"):
                    span = [word]
                    j = i + 1
                    while j < len(raw):
                        _, _, _, nn = raw[j]
                        if NER_MAP.get(nn) == ent_type and nn.startswith("I-"):
                            span.append(raw[j][0])
                            j += 1
                        else:
                            break
                    entity_text = " ".join(span)
                    # underthesea doesn't expose per-token confidence,
                    # so we use heuristic: multi-word proper nouns are higher conf
                    conf = 0.88 if len(span) >= 2 else 0.82
                    if conf < min_confidence:
                        all_high_conf = False
                    entities.append(
                        {
                            "text": entity_text,
                            "type": ent_type,
                            "confidence": conf,
                        }
                    )
                    i = j
                else:
                    i += 1

            # Only keep sentences where ALL entities meet confidence threshold
            if entities and all_high_conf:
                # Word-segment câu để đồng nhất với PhoBERT pretraining
                segmented_sent = word_segment(sent)
                # Entity text cũng cần segment để match được trong segmented sentence
                seg_entities = []
                for e in entities:
                    seg_text = word_segment(e["text"])
                    seg_entities.append({"text": seg_text, "type": e["type"]})
                silver.append(
                    {
                        "sentence": segmented_sent,
                        "entities": seg_entities,
                        "source": "silver",
                    }
                )
                total_sentences += 1

        if (article_idx + 1) % 1000 == 0:
            print(
                f"  Silver labeling: {article_idx+1} articles processed, {len(silver)} sentences kept"
            )

    print(
        f"[train_ner] Silver labels: {len(silver)} sentences from {len(articles)} articles"
    )
    return silver


# ── BIO conversion ───────────────────────────────────────────────────────────


def sentence_to_bio(sentence: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
    """Convert a sentence + entity annotations to (tokens, BIO_labels).

    Note: this does character-level alignment then maps to whitespace tokens.
    """
    words = sentence.split()
    labels = ["O"] * len(words)

    # Build character → word index mapping
    char_to_word = {}
    pos = 0
    for word_idx, word in enumerate(words):
        start = sentence.find(word, pos)
        if start < 0:
            start = pos
        for ch_idx in range(start, start + len(word)):
            char_to_word[ch_idx] = word_idx
        pos = start + len(word)

    for ent in entities:
        ent_text = ent.get("text", "").strip()
        ent_type = ent.get("type", "")
        if not ent_text or ent_type not in ("PER", "LOC", "ORG"):
            continue

        # Find entity in sentence (case-sensitive first, then insensitive)
        idx = sentence.find(ent_text)
        if idx < 0:
            idx = sentence.lower().find(ent_text.lower())
        if idx < 0:
            continue

        # Map character span to word indices
        ent_word_indices = sorted(
            set(
                char_to_word.get(ch, -1)
                for ch in range(idx, idx + len(ent_text))
                if ch in char_to_word
            )
        )
        ent_word_indices = [i for i in ent_word_indices if i >= 0]

        if not ent_word_indices:
            continue

        # Check for overlap with already-labeled words
        if any(labels[wi] != "O" for wi in ent_word_indices):
            continue

        for rank, wi in enumerate(ent_word_indices):
            if rank == 0:
                labels[wi] = f"B-{ent_type}"
            else:
                labels[wi] = f"I-{ent_type}"

    return words, labels


# ── Dataset class for HuggingFace Trainer ────────────────────────────────────


def _get_word_ids_manual(tokenizer, words: List[str], input_ids: List[int]) -> List:
    """Tự tính word_ids khi tokenizer không có .word_ids() (slow tokenizer như PhoBERT).

    Cách: encode từng word riêng → biết mỗi word tạo ra bao nhiêu sub-token
    → gán word_id cho từng position trong input_ids.
    Special tokens (CLS, SEP, PAD) → None.
    """
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    } - {None}

    # Encode từng word riêng, đếm số sub-token
    word_token_counts = []
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        word_token_counts.append(max(len(ids), 1))

    # Build word_ids list song song với input_ids
    word_ids = []
    word_idx = 0
    tokens_in_current_word = 0

    for tok_id in input_ids:
        if tok_id in special_ids:
            word_ids.append(None)
        else:
            if word_idx < len(word_token_counts):
                word_ids.append(word_idx)
                tokens_in_current_word += 1
                if tokens_in_current_word >= word_token_counts[word_idx]:
                    word_idx += 1
                    tokens_in_current_word = 0
            else:
                word_ids.append(None)

    return word_ids


def tokenize_and_align_labels(
    words: List[str],
    labels: List[str],
    tokenizer,
    max_length: int = 256,
    label_all_tokens: bool = False,
) -> Dict:
    """Tokenize words và align BIO labels về sub-token level.

    Hỗ trợ cả fast tokenizer (có .word_ids()) lẫn slow tokenizer (PhoBERT).
    PhoBERT dùng slow tokenizer nên không có .word_ids() → tự tính thủ công.

    Strategy:
      - First sub-token của mỗi word → label của word đó
      - Sub-token tiếp theo → I- label (label_all_tokens=True) hoặc -100
      - Special tokens (CLS, SEP, PAD) → -100
    """
    tokenized = tokenizer(
        words,
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding="max_length",
        return_tensors=None,
    )

    # Thử .word_ids() (fast tokenizer), fallback về manual (slow tokenizer / PhoBERT)
    try:
        word_ids = tokenized.word_ids()
    except (ValueError, AttributeError):
        word_ids = _get_word_ids_manual(tokenizer, words, tokenized["input_ids"])

    aligned_labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            label_str = labels[word_id] if word_id < len(labels) else "O"
            aligned_labels.append(LABEL2ID.get(label_str, 0))
        else:
            if label_all_tokens:
                label_str = labels[word_id] if word_id < len(labels) else "O"
                if label_str.startswith("B-"):
                    label_str = "I-" + label_str[2:]
                aligned_labels.append(LABEL2ID.get(label_str, 0))
            else:
                aligned_labels.append(-100)
        prev_word_id = word_id

    tokenized["labels"] = aligned_labels
    return tokenized


def prepare_dataset(
    samples: List[Dict],
    tokenizer,
    max_length: int = 256,
    label_all_tokens: bool = True,
) -> List[Dict]:
    """Convert all samples to tokenized + label-aligned format."""
    dataset = []
    for sample in samples:
        words, labels = sentence_to_bio(sample["sentence"], sample["entities"])
        if not words:
            continue
        tokenized = tokenize_and_align_labels(
            words,
            labels,
            tokenizer,
            max_length=max_length,
            label_all_tokens=label_all_tokens,
        )
        dataset.append(tokenized)
    return dataset


# ── Metrics ──────────────────────────────────────────────────────────────────


def compute_metrics_factory(id2label: Dict[int, str]):
    """Create a compute_metrics function for HuggingFace Trainer."""
    try:
        from seqeval.metrics import (
            f1_score,
            precision_score,
            recall_score,
            classification_report,
        )
    except ImportError:
        raise ImportError("seqeval is required: pip install seqeval")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_labels = []
        true_preds = []

        for pred_seq, label_seq in zip(predictions, labels):
            true_label_seq = []
            true_pred_seq = []
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                true_label_seq.append(id2label.get(int(label_id), "O"))
                true_pred_seq.append(id2label.get(int(pred_id), "O"))
            true_labels.append(true_label_seq)
            true_preds.append(true_pred_seq)

        precision = precision_score(true_labels, true_preds)
        recall = recall_score(true_labels, true_preds)
        f1 = f1_score(true_labels, true_preds)

        report = classification_report(true_labels, true_preds, output_dict=True)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "PER_f1": report.get("PER", {}).get("f1-score", 0.0),
            "LOC_f1": report.get("LOC", {}).get("f1-score", 0.0),
            "ORG_f1": report.get("ORG", {}).get("f1-score", 0.0),
        }

    return compute_metrics


# ── Torch Dataset wrapper ────────────────────────────────────────────────────


class NERDataset:
    """Dataset wrapper: tokenize lazily tại __getitem__ để tiết kiệm RAM.
    Thay vì pre-tokenize toàn bộ 69k samples vào RAM một lúc,
    mỗi sample chỉ được tokenize khi Trainer cần.
    """

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch

        sample = self.samples[idx]
        words, labels = sentence_to_bio(sample["sentence"], sample["entities"])
        if not words:
            words, labels = ["[UNK]"], ["O"]
        item = tokenize_and_align_labels(
            words,
            labels,
            self.tokenizer,
            max_length=self.max_length,
            label_all_tokens=True,
        )
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


# ── Training ─────────────────────────────────────────────────────────────────


def train(args):
    """Main training function."""
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_ner] Device: {device}")
    if device == "cpu":
        print("[train_ner] WARNING: Training on CPU will be very slow (8-12 hours).")
        print("[train_ner] Consider using --batch-size 4 --epochs 3 for CPU.")

    # ── Load tokenizer + model ───────────────────────────────────────────
    model_name = args.model_name
    print(f"[train_ner] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    print(
        f"[train_ner] Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # ── Load data ────────────────────────────────────────────────────────
    # 1. VLSP2016 từ HuggingFace (gold, ~16k)
    vlsp_samples = load_vlsp2016()

    # 2. Local gold (nếu có file tự label thêm)
    print(f"\n[train_ner] Loading local gold data from: {args.ground_truth}")
    local_gold = load_gold_data(args.ground_truth)
    print(f"[train_ner] Local gold samples: {len(local_gold)}")

    # 3. Silver từ 150k bài báo
    silver_samples = []
    if not args.no_silver:
        print(f"\n[train_ner] Generating silver labels from: {args.data_csv}")
        print(f"[train_ner] Max articles for silver: {args.silver_count}")
        articles = load_news_articles(args.data_csv, max_articles=args.silver_count)
        if articles:
            silver_samples = generate_silver_labels(
                articles,
                min_confidence=args.silver_confidence,
            )
    else:
        print("[train_ner] Silver labels disabled (--no-silver)")

    # ── Curriculum training ──────────────────────────────────────────────
    # Phase 1 (gold_epochs epoch đầu): chỉ VLSP2016 + local_gold
    #   → model học pattern NER sạch trước khi thấy nhiễu từ silver
    # Phase 2 (epochs còn lại): gold × 3 + silver (50% silver, confidence cao hơn)
    #   → mở rộng vocabulary entity domain báo chí, LR thấp hơn để không overwrite
    gold_samples = vlsp_samples + local_gold

    if not gold_samples:
        print("[train_ner] ERROR: Không có gold data. Cần ít nhất VLSP2016.")
        return

    random.seed(args.seed)

    # Dev set lấy từ gold để evaluation không bị ảnh hưởng bởi silver noise
    dev_size = max(1, int(len(gold_samples) * args.dev_split))
    random.shuffle(gold_samples)
    dev_samples = gold_samples[:dev_size]
    gold_train = gold_samples[dev_size:]

    # Phase 1: gold × 3
    phase1_samples = gold_train * 3
    random.shuffle(phase1_samples)

    # Phase 2: gold × 3 + silver (chỉ lấy 50% silver để giảm noise thêm)
    silver_trimmed = (
        random.sample(silver_samples, len(silver_samples) // 2)
        if silver_samples
        else []
    )
    phase2_samples = gold_train * 3 + silver_trimmed
    random.shuffle(phase2_samples)

    mix_epochs = args.epochs - args.gold_epochs

    print(f"\n[train_ner] Curriculum setup:")
    print(
        f"  Phase 1 — gold only:    {args.gold_epochs} epochs × {len(phase1_samples)} samples"
    )
    print(
        f"  Phase 2 — gold+silver:  {mix_epochs} epochs × {len(phase2_samples)} samples"
    )
    print(f"  Dev (gold only):        {len(dev_samples)} samples")
    print(f"  Silver dùng: {len(silver_trimmed)}/{len(silver_samples)} (50%)")

    if len(phase1_samples) < 5:
        print("[train_ner] ERROR: Not enough training data.")
        return

    effective_batch = args.batch_size
    grad_accum = 1
    if device == "cpu" and args.batch_size > 4:
        grad_accum = args.batch_size // 4
        effective_batch = 4
        print(f"[train_ner] CPU mode: batch={effective_batch}, grad_accum={grad_accum}")

    compute_metrics = compute_metrics_factory(ID2LABEL)
    dev_dataset = NERDataset(dev_samples, tokenizer, max_length=args.max_seq_length)

    t0 = time.time()

    # ── Phase 1: gold only ────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(
        f"  Phase 1: Gold only — {args.gold_epochs} epochs, LR={args.learning_rate:.0e}"
    )
    print(f"{'='*56}")

    warmup_p1 = int(len(phase1_samples) / effective_batch * args.gold_epochs * 0.1)
    p1_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints_p1"),
        num_train_epochs=args.gold_epochs,
        per_device_train_batch_size=effective_batch,
        per_device_eval_batch_size=effective_batch * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_p1,
        label_smoothing_factor=args.label_smoothing,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=(device == "cuda"),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=p1_args,
        train_dataset=NERDataset(
            phase1_samples, tokenizer, max_length=args.max_seq_length
        ),
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=(
            [EarlyStoppingCallback(args.early_stopping)]
            if args.early_stopping > 0
            else []
        ),
    )
    trainer.train()
    p1_result = trainer.evaluate()
    print(f"  Phase 1 done — F1: {p1_result.get('eval_f1', 0):.4f}")

    # ── Phase 2: gold + silver (nếu có) ──────────────────────────────────
    if mix_epochs > 0 and phase2_samples:
        phase2_lr = (
            args.learning_rate * 0.3
        )  # LR thấp hơn để không overwrite gold knowledge
        warmup_p2 = int(len(phase2_samples) / effective_batch * mix_epochs * 0.05)

        print(f"\n{'='*56}")
        print(f"  Phase 2: Gold+Silver — {mix_epochs} epochs, LR={phase2_lr:.0e}")
        print(f"{'='*56}")

        p2_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints_p2"),
            num_train_epochs=mix_epochs,
            per_device_train_batch_size=effective_batch,
            per_device_eval_batch_size=effective_batch * 2,
            gradient_accumulation_steps=grad_accum,
            learning_rate=phase2_lr,
            weight_decay=0.01,
            warmup_steps=warmup_p2,
            label_smoothing_factor=args.label_smoothing,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=(device == "cuda"),
            report_to="none",
            seed=args.seed,
        )
        trainer = Trainer(
            model=model,
            args=p2_args,
            train_dataset=NERDataset(
                phase2_samples, tokenizer, max_length=args.max_seq_length
            ),
            eval_dataset=dev_dataset,
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    elapsed = time.time() - t0
    print(f"\n[train_ner] Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n[train_ner] Final evaluation:")
    eval_result = trainer.evaluate()
    for key, value in sorted(eval_result.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    final_dir = output_dir
    print(f"\n[train_ner] Saving to: {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metadata = {
        "model_name": model_name,
        "labels": LABEL_LIST,
        "num_labels": NUM_LABELS,
        "gold_samples": len(gold_samples),
        "silver_samples": len(silver_samples),
        "silver_used": len(silver_trimmed),
        "phase1_samples": len(phase1_samples),
        "phase2_samples": len(phase2_samples),
        "dev_size": len(dev_samples),
        "gold_epochs": args.gold_epochs,
        "total_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "label_smoothing": args.label_smoothing,
        "training_time_seconds": round(elapsed, 1),
        "eval_metrics": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in eval_result.items()
        },
    }
    with open(final_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[train_ner] Done! Model saved to: {final_dir}")
    print(f"[train_ner] To use: pass --ner-model-dir {final_dir} to main.py")
    return metadata


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Fine-tune PhoBERT for Vietnamese NER on news domain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="vinai/phobert-base-v2",
        help="HuggingFace model ID for base model",
    )
    parser.add_argument(
        "--ground-truth",
        default=str(ROOT_DIR / "data" / "ner_ground_truth.json"),
        help="Path to NER ground truth JSON",
    )
    parser.add_argument(
        "--data-csv",
        default=str(ROOT_DIR / "data" / "vnexpress_articles.csv"),
        help="Path to news articles CSV for silver labels",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "ner_model"),
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Tổng số epoch (phase1 + phase2)"
    )
    parser.add_argument(
        "--gold-epochs",
        type=int,
        default=2,
        help="Số epoch Phase 1 chỉ train gold (VLSP + local). "
        "Phần còn lại train gold+silver với LR × 0.3",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dev-split", type=float, default=0.1)
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=2,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--silver-count",
        type=int,
        default=10000,
        help="Number of articles for silver labeling",
    )
    parser.add_argument(
        "--silver-confidence",
        type=float,
        default=0.90,
        help="Minimum confidence cho silver labels (tăng lên 0.90 để giảm nhiễu)",
    )
    parser.add_argument(
        "--no-silver", action="store_true", help="Disable silver label generation"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
