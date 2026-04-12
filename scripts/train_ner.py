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
                samples.append(
                    {"sentence": sentence, "entities": entities, "source": "vlsp2016"}
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
    """Load raw text from vnexpress_articles.csv for silver labeling."""
    texts = []
    path = Path(csv_path)
    if not path.exists():
        print(f"[train_ner] CSV not found: {path}")
        return texts

    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_articles * 3:  # read more, filter later
                break
            text = (row.get("text") or "").strip()
            if len(text) > 50:
                texts.append(text)
    random.shuffle(texts)
    return texts[:max_articles]


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
                silver.append(
                    {
                        "sentence": sent,
                        "entities": [
                            {"text": e["text"], "type": e["type"]} for e in entities
                        ],
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


def tokenize_and_align_labels(
    words: List[str],
    labels: List[str],
    tokenizer,
    max_length: int = 256,
    label_all_tokens: bool = False,
) -> Dict:
    """Tokenize words and align BIO labels to sub-tokens.

    Strategy:
      - First sub-token of each word → word's label
      - Subsequent sub-tokens → I- label (if label_all_tokens) or -100 (ignore)
      - Special tokens ([CLS], [SEP], padding) → -100
    """
    tokenized = tokenizer(
        words,
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
        padding="max_length",
        return_tensors=None,
    )

    word_ids = tokenized.word_ids()
    aligned_labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            # Special token
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            # First sub-token of a word
            label_str = labels[word_id] if word_id < len(labels) else "O"
            aligned_labels.append(LABEL2ID.get(label_str, 0))
        else:
            # Continuation sub-token
            if label_all_tokens:
                label_str = labels[word_id] if word_id < len(labels) else "O"
                # Convert B- to I- for continuation
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
    """Simple dataset wrapper for HuggingFace Trainer."""

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        import torch

        item = self.data[idx]
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

    # Mix: VLSP2016 × 3 + local_gold × 3 + silver × 1
    # Oversample gold/VLSP vì quality cao hơn silver
    gold_samples = vlsp_samples + local_gold
    all_samples = gold_samples * 3 + silver_samples
    random.seed(args.seed)
    random.shuffle(all_samples)
    print(
        f"\n[train_ner] Mix: VLSP2016={len(vlsp_samples)}, local_gold={len(local_gold)}, "
        f"silver={len(silver_samples)}"
    )
    print(f"[train_ner] Total after 3× oversample gold: {len(all_samples)} samples")

    if len(all_samples) < 5:
        print("[train_ner] ERROR: Not enough training data. Need at least 5 samples.")
        return

    # ── Tokenize + align ─────────────────────────────────────────────────
    print("[train_ner] Tokenizing and aligning labels...")
    tokenized_data = prepare_dataset(
        all_samples,
        tokenizer,
        max_length=args.max_seq_length,
        label_all_tokens=True,
    )
    print(f"[train_ner] Tokenized samples: {len(tokenized_data)}")

    # ── Split train/dev ──────────────────────────────────────────────────
    dev_size = max(1, int(len(tokenized_data) * args.dev_split))
    dev_data = tokenized_data[:dev_size]
    train_data = tokenized_data[dev_size:]
    print(f"[train_ner] Train: {len(train_data)}, Dev: {len(dev_data)}")

    train_dataset = NERDataset(train_data)
    dev_dataset = NERDataset(dev_data)

    # ── Training arguments ───────────────────────────────────────────────
    effective_batch = args.batch_size
    grad_accum = 1
    if device == "cpu" and args.batch_size > 4:
        grad_accum = args.batch_size // 4
        effective_batch = 4
        print(f"[train_ner] CPU mode: batch={effective_batch}, grad_accum={grad_accum}")

    warmup_steps = int(len(train_data) / effective_batch * args.epochs * 0.1)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=effective_batch,
        per_device_eval_batch_size=effective_batch * 2,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
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

    compute_metrics = compute_metrics_factory(ID2LABEL)

    callbacks = []
    if args.early_stopping > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping)
        )

    # ── Train ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print(f"\n[train_ner] Starting training...")
    print(f"  Model:          {model_name}")
    print(f"  Labels:         {LABEL_LIST}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {effective_batch} (grad_accum={grad_accum})")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Label smooth:   {args.label_smoothing}")
    print(f"  Max seq length: {args.max_seq_length}")
    print(f"  Warmup steps:   {warmup_steps}")
    print(f"  Device:         {device}")
    print()

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\n[train_ner] Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n[train_ner] Final evaluation on dev set:")
    eval_result = trainer.evaluate()
    for key, value in sorted(eval_result.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # ── Save model ───────────────────────────────────────────────────────
    final_dir = output_dir
    print(f"\n[train_ner] Saving model to: {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training metadata
    metadata = {
        "model_name": model_name,
        "labels": LABEL_LIST,
        "num_labels": NUM_LABELS,
        "gold_samples": len(gold_samples),
        "silver_samples": len(silver_samples),
        "total_samples": len(all_samples),
        "train_size": len(train_data),
        "dev_size": len(dev_data),
        "epochs": args.epochs,
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
    parser.add_argument("--epochs", type=int, default=5)
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
        default=0.85,
        help="Minimum confidence for silver labels",
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
