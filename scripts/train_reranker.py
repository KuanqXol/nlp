"""
train_reranker.py
─────────────────
Fine-tune ViDeBERTa làm cross-encoder reranker cho tiếng Việt.

Base model: Fsoft-AIC/videberta-base  (DeBERTaV3, tốt hơn PhoBERT cho pair ranking)
Training data (theo thứ tự ưu tiên):
  1. MMARCO-Vi (unicamp-dl/mmarco, "vietnamese") — 532k pair, base ranking knowledge
  2. UIT-ViQuAD2.0 — câu hỏi tự nhiên tiếng Việt
  3. 150k bài báo VnExpress — pseudo query từ title + BM25 hard negative

Curriculum:
  Epoch 1 = MMARCO-Vi random sample + ViQuAD random negatives
  Epoch 2-3 = BM25 hard negatives từ corpus tin tức

Usage:
    python scripts/train_reranker.py --help
    python scripts/train_reranker.py --output-dir data/reranker_model
    python scripts/train_reranker.py --no-viquad --epochs 3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# ── BM25 for hard negative mining ───────────────────────────────────────────


def _tokenize(text: str) -> List[str]:
    import re

    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


class SimpleBM25:
    """Lightweight BM25 for hard negative mining during training."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._tfs: List[Counter] = []
        self._doc_len: List[int] = []
        self._avgdl = 0.0
        self._idf: Dict[str, float] = {}

    def build(self, texts: List[str], ids: List[str] = None):
        self._texts = list(texts)
        self._ids = list(ids or range(len(texts)))
        df_counter: Counter = Counter()
        self._tfs = []
        self._doc_len = []

        for text in texts:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._doc_len.append(len(tokens))
            df_counter.update(tf.keys())

        n = max(len(self._ids), 1)
        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._idf = {
            t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in df_counter.items()
        }

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        qtf = Counter(_tokenize(query))
        if not qtf:
            return []

        scores = []
        avgdl = self._avgdl or 1.0
        for idx, tf in enumerate(self._tfs):
            score = 0.0
            dl = self._doc_len[idx] or 1
            for term, qfreq in qtf.items():
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += idf * (f * (self.k1 + 1)) / max(denom, 1e-8) * qfreq
            if score > 0:
                scores.append((self._ids[idx], score))

        scores.sort(key=lambda x: -x[1])
        return scores[:k]


# ── Pseudo-query generation ─────────────────────────────────────────────────


def generate_pseudo_queries(article: Dict) -> List[str]:
    """Generate multiple pseudo-queries from a news article.

    Type 1: title (current)
    Type 2: first sentence (news lead)
    Type 3: "{entity} {category_keyword}" combinations
    Type 4: interrogative pattern from title
    """
    queries = []
    title = (article.get("title") or "").strip()
    text = (article.get("text") or article.get("full_text") or "").strip()
    category = (article.get("category") or "").strip()
    entities = article.get("entities", [])

    # Type 1: title
    if title and len(title) > 5:
        queries.append(title)

    # Type 2: first sentence
    if text:
        first_sent = text.split(".")[0].strip()
        if first_sent and len(first_sent) > 10 and first_sent != title:
            queries.append(first_sent)

    # Type 3: entity + category
    if entities and category:
        for ent in entities[:3]:
            ent_text = ent.get("text") or ent.get("canonical", "")
            if ent_text:
                queries.append(f"{ent_text} {category}")

    # Type 4: interrogative from title
    if title:
        import re

        # "X ký kết Y" → "X ký kết gì?"
        if re.search(r"ký kết|hợp tác|thỏa thuận", title, re.IGNORECASE):
            queries.append(
                re.sub(r"(ký kết|hợp tác|thỏa thuận)\s+\S+", r"\1 gì?", title)
            )
        # "X tăng Y%" → "X tăng bao nhiêu?"
        if re.search(r"tăng\s+[\d,]+%?", title):
            queries.append(re.sub(r"tăng\s+[\d,]+%?", "tăng bao nhiêu?", title))
        # "X bổ nhiệm Y" → "X bổ nhiệm ai?"
        if re.search(r"bổ nhiệm|bầu|phong|chọn", title, re.IGNORECASE):
            queries.append(re.sub(r"(bổ nhiệm|bầu|phong|chọn)\s+\S+", r"\1 ai?", title))

    # Deduplicate
    seen = set()
    unique = []
    for q in queries:
        q = q.strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
    return unique


# ── Data preparation ─────────────────────────────────────────────────────────


def load_mmarco_vi(max_pairs: int = 100000) -> List[Tuple[str, str, str]]:
    """Load MMARCO-Vi training triples (query, positive, negative).

    Dataset: unicamp-dl/mmarco, config "vietnamese"
    Mỗi sample đã có sẵn positive + negative — không cần mine thêm.
    Trả về list of (query, positive_passage, negative_passage).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[train_reranker] datasets not installed")
        return []

    print("[train_reranker] Loading MMARCO-Vi from HuggingFace...")
    try:
        ds = load_dataset("unicamp-dl/mmarco", "vietnamese", split="train")
    except Exception as e:
        print(f"[train_reranker] Cannot load MMARCO-Vi: {e}")
        return []

    triples = []
    for item in ds:
        q = (item.get("query") or "").strip()
        pos = (item.get("positive") or "").strip()
        neg = (item.get("negative") or "").strip()
        if q and pos and neg:
            triples.append((q, pos[:512], neg[:512]))
        if len(triples) >= max_pairs:
            break

    print(f"[train_reranker] MMARCO-Vi triples loaded: {len(triples)}")
    return triples

    """Load (question, positive_passage) pairs from UIT-ViQuAD2.0."""
    try:
        from datasets import load_dataset

        ds = load_dataset("uitnlp/vi_quad", split="train")
    except Exception as e:
        print(f"[train_reranker] Cannot load ViQuAD: {e}")
        print("[train_reranker] Using pseudo-labeled data only.")
        return []

    pairs = []
    for item in ds:
        question = (item.get("question") or "").strip()
        context = (item.get("context") or "").strip()
        if question and context and len(context) > 50:
            pairs.append((question, context[:512]))
            if len(pairs) >= max_pairs:
                break

    print(f"[train_reranker] ViQuAD pairs loaded: {len(pairs)}")
    return pairs


def load_news_corpus(csv_path: str, max_articles: int = 20000) -> List[Dict]:
    """Load news articles for pseudo-label and hard negative generation."""
    articles = []
    path = Path(csv_path)
    if not path.exists():
        return articles

    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_articles:
                break
            text = (row.get("text") or "").strip()
            title = (row.get("title") or "").strip()
            if text and title and len(text) > 50:
                articles.append(
                    {
                        "id": str(i),
                        "title": title,
                        "text": text[:1000],
                        "category": (row.get("category") or "").strip(),
                    }
                )

    return articles


def build_training_pairs(
    mmarco_triples: List[Tuple[str, str, str]],
    viquad_pairs: List[Tuple[str, str]],
    articles: List[Dict],
    bm25: SimpleBM25,
    n_hard_negatives: int = 3,
    n_pseudo_articles: int = 5000,
    curriculum_epoch: int = 1,
    mmarco_sample: int = 50000,
    seed: int = 42,
) -> List[Dict]:
    """Build training pairs cho cross-encoder.

    Nguồn dữ liệu theo thứ tự ưu tiên:
    1. MMARCO-Vi: đã có positive+negative sẵn, chất lượng cao nhất
    2. ViQuAD: positive từ QA, negative từ BM25 hoặc random
    3. Pseudo từ báo: title→body positive, BM25 hard negative

    Returns list of {query, passage, label}.
    """
    rng = random.Random(seed)
    pairs = []

    # ── 1. MMARCO-Vi triples (sẵn positive + negative) ───────────────────
    sampled_mmarco = rng.sample(mmarco_triples, min(mmarco_sample, len(mmarco_triples)))
    for q, pos, neg in sampled_mmarco:
        pairs.append({"query": q, "passage": pos, "label": 1})
        pairs.append({"query": q, "passage": neg, "label": 0})
    print(f"[train_reranker] MMARCO-Vi pairs added: {len(sampled_mmarco) * 2}")

    # ── 2. ViQuAD positive pairs ─────────────────────────────────────────
    article_texts = [a["text"] for a in articles]
    for question, context in viquad_pairs:
        pairs.append({"query": question, "passage": context, "label": 1})

        if curriculum_epoch == 1:
            neg_idx = rng.randint(0, max(len(article_texts) - 1, 0))
            neg_text = article_texts[neg_idx][:512] if article_texts else ""
            if neg_text:
                pairs.append({"query": question, "passage": neg_text, "label": 0})
        else:
            bm25_results = bm25.search(question, k=20)
            negatives = [
                (doc_id, score)
                for doc_id, score in bm25_results
                if articles[int(doc_id)]["text"][:512] != context[:512]
            ][:n_hard_negatives]
            for doc_id, _ in negatives:
                neg_text = articles[int(doc_id)]["text"][:512]
                pairs.append({"query": question, "passage": neg_text, "label": 0})

    # ── 3. Pseudo-labeled từ tin tức ─────────────────────────────────────
    pseudo_articles = rng.sample(articles, min(n_pseudo_articles, len(articles)))
    for article in pseudo_articles:
        pseudo_queries = generate_pseudo_queries(article)
        for pq in pseudo_queries[:2]:
            pairs.append({"query": pq, "passage": article["text"][:512], "label": 1})

            if curriculum_epoch == 1:
                neg_idx = rng.randint(0, max(len(articles) - 1, 0))
                pairs.append(
                    {
                        "query": pq,
                        "passage": articles[neg_idx]["text"][:512],
                        "label": 0,
                    }
                )
            else:
                bm25_results = bm25.search(pq, k=10)
                negatives = [
                    (doc_id, score)
                    for doc_id, score in bm25_results
                    if doc_id != article["id"]
                ][:n_hard_negatives]
                for doc_id, _ in negatives:
                    pairs.append(
                        {
                            "query": pq,
                            "passage": articles[int(doc_id)]["text"][:512],
                            "label": 0,
                        }
                    )

    rng.shuffle(pairs)
    n_pos = sum(1 for p in pairs if p["label"] == 1)
    n_neg = sum(1 for p in pairs if p["label"] == 0)
    print(
        f"[train_reranker] Total pairs (epoch={curriculum_epoch}): {len(pairs)} "
        f"(pos={n_pos}, neg={n_neg})"
    )
    return pairs


# ── Torch Dataset ────────────────────────────────────────────────────────────


class RerankerDataset:
    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import torch

        pair = self.pairs[idx]
        encoded = self.tokenizer(
            pair["query"],
            pair["passage"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(pair["label"], dtype=torch.float),
        }


# ── Training ─────────────────────────────────────────────────────────────────


def train(args):
    """Main training function with curriculum learning."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_reranker] Device: {device}")

    model_name = args.model_name
    print(f"[train_reranker] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.to(device)

    # ── Load data ────────────────────────────────────────────────────────
    # 1. MMARCO-Vi — base ranking knowledge, 532k triples sẵn positive+negative
    mmarco_triples = []
    if not args.no_mmarco:
        mmarco_triples = load_mmarco_vi(max_pairs=args.mmarco_max)

    # 2. ViQuAD — câu hỏi tự nhiên tiếng Việt
    viquad_pairs = []
    if not args.no_viquad:
        viquad_pairs = load_viquad_pairs(max_pairs=args.viquad_max)

    # 3. Tin tức — pseudo query + hard negative domain-specific
    articles = load_news_corpus(args.data_csv, max_articles=args.corpus_size)
    if not articles and not mmarco_triples and not viquad_pairs:
        print("[train_reranker] ERROR: No data loaded at all.")
        return

    # Build BM25 index for hard negative mining
    print("[train_reranker] Building BM25 index for hard negative mining...")
    bm25 = SimpleBM25()
    bm25.build(
        [a["text"] for a in articles],
        [a["id"] for a in articles],
    )

    # ── Curriculum training ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Curriculum: epoch 1 = random negatives, epoch 2+ = BM25 hard negatives
        curriculum_epoch = 1 if epoch == 1 else 2
        pairs = build_training_pairs(
            mmarco_triples,
            viquad_pairs,
            articles,
            bm25,
            n_hard_negatives=args.hard_negatives,
            n_pseudo_articles=args.pseudo_count,
            curriculum_epoch=curriculum_epoch,
            mmarco_sample=args.mmarco_sample_per_epoch,
            seed=args.seed + epoch,
        )

        dataset = RerankerDataset(pairs, tokenizer, max_length=args.max_length)
        batch_size = args.batch_size
        if device == "cpu" and batch_size > 8:
            batch_size = 8

        model.train()
        total_loss = 0.0
        n_batches = 0

        # Manual batching (simpler than DataLoader for this use case)
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            batch_items = [dataset[i] for i in batch_indices]

            input_ids = torch.stack([item["input_ids"] for item in batch_items]).to(
                device
            )
            attention_mask = torch.stack(
                [item["attention_mask"] for item in batch_items]
            ).to(device)
            labels = torch.stack([item["labels"] for item in batch_items]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches % 100 == 0:
                avg_loss = total_loss / n_batches
                print(f"  Batch {n_batches}: avg_loss={avg_loss:.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        print(f"\n  Epoch {epoch} done. Avg loss: {avg_loss:.4f}")

    # ── Save model ───────────────────────────────────────────────────────
    print(f"\n[train_reranker] Saving model to: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save metadata
    import json

    metadata = {
        "base_model": model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "viquad_pairs": len(viquad_pairs),
        "corpus_articles": len(articles),
        "hard_negatives": args.hard_negatives,
        "pseudo_count": args.pseudo_count,
    }
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[train_reranker] Done! Model saved to: {output_dir}")
    print(f"[train_reranker] To use: pass --reranker-model-dir {output_dir} to main.py")


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Fine-tune cross-encoder reranker on Vietnamese data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="Fsoft-AIC/videberta-base",
        help="HuggingFace model ID cho base cross-encoder (mặc định: ViDeBERTa)",
    )
    parser.add_argument(
        "--data-csv",
        default=str(ROOT_DIR / "data" / "vnexpress_articles.csv"),
        help="Path to news corpus CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "reranker_model"),
        help="Output directory for fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--hard-negatives", type=int, default=3)
    parser.add_argument("--corpus-size", type=int, default=20000)
    parser.add_argument("--pseudo-count", type=int, default=5000)
    parser.add_argument("--viquad-max", type=int, default=5000)
    parser.add_argument("--no-viquad", action="store_true")
    parser.add_argument(
        "--mmarco-max",
        type=int,
        default=200000,
        help="Tổng số triple MMARCO-Vi load (default 200k)",
    )
    parser.add_argument(
        "--mmarco-sample-per-epoch",
        type=int,
        default=50000,
        help="Sample bao nhiêu triple MMARCO mỗi epoch",
    )
    parser.add_argument("--no-mmarco", action="store_true", help="Bỏ qua MMARCO-Vi")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
