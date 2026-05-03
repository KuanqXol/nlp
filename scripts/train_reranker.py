"""
Local-first curriculum trainer for the Vietnamese reranker.

Ezpl-aligned goals:
  - Multi-source training: MSMARCO-like source (optional), UIT-ViQuAD2.0, local VnExpress news
  - Curriculum / staged fine-tuning: easier weakly-supervised data first, harder news negatives later
  - Hard negative mining for news
  - Export model/tokenizer + metadata for local inference

Default usage:
  python scripts/train_reranker.py \
    --data-csv data/vnexpress_articles.csv \
    --output-dir data/reranker_model_videberta

Notes:
  - If internet / dataset access is unavailable, the trainer gracefully falls back to
    local VnExpress + ViQuAD only.
  - The final output is compatible with `main.py --reranker-dir ...`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HF_CACHE = ROOT_DIR / "data" / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────


def _default_output_dir() -> str:
    kaggle = Path("/kaggle/working")
    if kaggle.exists():
        return str(kaggle / "videberta_reranker")
    return str(ROOT_DIR / "data" / "reranker_model_videberta")


def _log(msg: str):
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


def _resolve_csv(csv_path: str) -> Path:
    path = Path(csv_path).expanduser()
    if path.exists():
        return path
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        hits = sorted(kaggle_input.rglob(path.name))
        if hits:
            return hits[0]
        hits = sorted(kaggle_input.rglob("*vnexpress*.csv"))
        if hits:
            return hits[0]
    return path


def _safe_load_dataset(name: str, split: str):
    from datasets import load_dataset

    try:
        return load_dataset(name, split=split)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# BM25 for hard negative mining
# ──────────────────────────────────────────────────────────────────────────────


class SimpleBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._ids: List[str] = []
        self._doc_len: List[int] = []
        self._avgdl = 0.0
        self._idf: Dict[str, float] = {}
        self._postings: Dict[str, List[Tuple[int, int]]] = {}

    def build(self, texts: List[str], ids: Optional[List[str]] = None):
        self._ids = list(ids or range(len(texts)))
        self._doc_len = []
        self._postings = {}
        df_counter: Counter = Counter()
        for idx, text in enumerate(texts):
            tf = Counter(_tokenize(text))
            self._doc_len.append(sum(tf.values()))
            for term, freq in tf.items():
                self._postings.setdefault(term, []).append((idx, int(freq)))
            df_counter.update(tf.keys())
        n_docs = max(len(self._ids), 1)
        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._idf = {t: np.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for t, df in df_counter.items()}

    def search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        qtf = Counter(_tokenize(query))
        if not qtf:
            return []
        scores: Dict[int, float] = {}
        avgdl = self._avgdl or 1.0
        for term, qfreq in qtf.items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf.get(term, 0.0)
            for idx, f in postings:
                dl = self._doc_len[idx] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score = idf * (f * (self.k1 + 1)) / max(denom, 1e-8) * qfreq
                scores[idx] = scores.get(idx, 0.0) + float(score)
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [(self._ids[idx], score) for idx, score in ranked]


# ──────────────────────────────────────────────────────────────────────────────
# Data sources
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Example:
    query: str
    passage: str
    labels: int
    source: str
    difficulty: float = 0.5


def load_viquad_examples(train_max: int, val_max: int) -> Tuple[List[Dict], List[Dict]]:
    datasets = ["taidng/UIT-ViQuAD2.0", "uitnlp/vi_quad"]
    ds_train = ds_val = None
    for name in datasets:
        ds_train = _safe_load_dataset(name, "train")
        ds_val = _safe_load_dataset(name, "validation")
        if ds_train is not None and ds_val is not None:
            break
    if ds_train is None or ds_val is None:
        return [], []

    def convert(ds, limit):
        out = []
        for row in ds:
            q = (row.get("question") or row.get("query") or "").strip()
            c = (row.get("context") or row.get("passage") or row.get("paragraph") or "").strip()
            if q and len(c) >= 60:
                out.append(Example(q[:200], c[:512], 1, "viquad", difficulty=0.35).__dict__)
            if len(out) >= limit:
                break
        return out

    return convert(ds_train, train_max), convert(ds_val, val_max)


def load_msmarco_like_examples(train_max: int, val_max: int) -> Tuple[List[Dict], List[Dict]]:
    """Optional external source aligned with Ezpl expectation.

    We try a couple of public datasets that behave similarly to MSMARCO-style
    retrieval examples. If unavailable, return empty lists so local training still works.
    """
    candidates = [
        ("msmarco", "triples.train.small"),
        ("sentence-transformers/msmarco-corpus", "train"),
        ("sentence-transformers/msmarco", "train"),
    ]
    for ds_name, split in candidates:
        try:
            ds = _safe_load_dataset(ds_name, split)
            if ds is None:
                continue
            items = []
            for row in ds:
                q = (row.get("query") or row.get("question") or "").strip()
                p = (row.get("positive") or row.get("passage") or row.get("doc") or row.get("text") or "").strip()
                if q and p:
                    items.append(Example(q[:200], p[:512], 1, "msmarco_like", difficulty=0.15).__dict__)
                if len(items) >= train_max + val_max:
                    break
            if items:
                rng = random.Random(42)
                rng.shuffle(items)
                split_idx = min(max(1, val_max), max(0, len(items) // 10))
                return items[split_idx:], items[:split_idx]
        except Exception:
            continue
    return [], []


def load_news_articles(csv_path: str, max_articles: int = 30000) -> List[Dict]:
    path = _resolve_csv(csv_path)
    if not path.exists():
        _log(f"[data] Khong tim thay CSV: {path}")
        return []
    seen = set()
    docs = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if len(docs) >= max_articles:
                break
            title = (row.get("title") or "").strip()
            text = (row.get("text") or row.get("content") or "").strip()
            if len(title) < 6 or len(text) < 60:
                continue
            url = (row.get("url") or "").strip()
            key = url or _normalize(title)
            if key in seen:
                continue
            seen.add(key)
            docs.append(
                {
                    "id": f"news_{idx}",
                    "title": title,
                    "text": text[:1200],
                    "category": (row.get("category") or "").strip(),
                }
            )
    return docs


def split_news(articles: List[Dict], val_ratio: float, seed: int):
    rng = random.Random(seed)
    items = list(articles)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio)) if len(items) >= 20 else max(0, int(len(items) * val_ratio))
    return items[n_val:], items[:n_val]


def pseudo_queries(article: Dict) -> List[str]:
    title = (article.get("title") or "").strip()
    text = (article.get("text") or "").strip()
    cat = (article.get("category") or "").strip()
    queries = []
    if len(title) >= 6:
        queries.append(title)
    if text:
        first = re.split(r"(?<=[.!?…])\s+", text)[0].strip()
        if len(first) >= 10 and _normalize(first) != _normalize(title):
            queries.append(first)
    if title and cat:
        queries.append(f"{title} {cat}")
    out, seen = [], set()
    for q in queries:
        key = _normalize(q)
        if q and key not in seen:
            seen.add(key)
            out.append(q)
    return out


def build_news_examples(
    articles: List[Dict],
    hard_negatives: int,
    max_pseudo: int,
    seed: int,
) -> List[Dict]:
    if not articles:
        return []
    rng = random.Random(seed)
    bm25 = SimpleBM25()
    bm25.build([a["text"] for a in articles], [a["id"] for a in articles])
    docs_by_id = {a["id"]: a for a in articles}
    examples = []
    for article in articles:
        for q in pseudo_queries(article)[:max_pseudo]:
            pos = article["text"][:512]
            examples.append(Example(q, pos, 1, "news", difficulty=0.9).__dict__)
            negs = []
            for doc_id, _ in bm25.search(q, k=max(20, hard_negatives * 8)):
                if doc_id == article["id"]:
                    continue
                neg_text = docs_by_id[str(doc_id)]["text"][:512]
                if _normalize(neg_text) != _normalize(pos):
                    negs.append(neg_text)
                if len(negs) >= hard_negatives:
                    break
            if not negs:
                cand = rng.choice(articles)["text"][:512]
                if _normalize(cand) != _normalize(pos):
                    negs = [cand]
            for neg in negs:
                examples.append(Example(q, neg, 0, "news", difficulty=0.95).__dict__)
    # dedupe
    seen, out = set(), []
    for e in examples:
        key = (_normalize(e["query"]), _normalize(e["passage"]), int(e["labels"]), e["source"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum dataset assembly
# ──────────────────────────────────────────────────────────────────────────────


def _cache_signature(args) -> str:
    payload = {
        "csv": str(_resolve_csv(args.data_csv)),
        "max_news_articles": args.max_news_articles,
        "news_val_ratio": args.news_val_ratio,
        "hard_negatives": args.hard_negatives,
        "max_pseudo_queries_per_article": args.max_pseudo_queries_per_article,
        "viquad_train_max": args.viquad_train_max,
        "viquad_val_max": args.viquad_val_max,
        "msmarco_train_max": args.msmarco_train_max,
        "msmarco_val_max": args.msmarco_val_max,
        "seed": args.seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _stage_name(stage_id: int) -> str:
    return f"stage{stage_id}"


def build_curriculum_examples(args):
    """Return list of training stages and a validation set.

    Stages are ordered from easier / weaker supervision to harder / domain-specific.
    """
    cache_dir = Path(args.output_dir) / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = _cache_signature(args)
    marker = cache_dir / sig
    train_marker = marker / "train.json"
    val_marker = marker / "val.json"
    if train_marker.exists() and val_marker.exists():
        payload = json.loads(train_marker.read_text(encoding="utf-8"))
        val_payload = json.loads(val_marker.read_text(encoding="utf-8"))
        return payload["stages"], val_payload["examples"]

    msmarco_train, msmarco_val = load_msmarco_like_examples(args.msmarco_train_max, args.msmarco_val_max)
    viquad_train, viquad_val = load_viquad_examples(args.viquad_train_max, args.viquad_val_max)
    news = load_news_articles(args.data_csv, args.max_news_articles)
    news_train, news_val = split_news(news, args.news_val_ratio, args.seed)
    train_news = build_news_examples(news_train, args.hard_negatives, args.max_pseudo_queries_per_article, args.seed)
    val_news = build_news_examples(news_val, 1, args.max_pseudo_queries_per_article, args.seed + 1)

    stages = []
    if msmarco_train:
        stages.append(
            {
                "name": "msmarco_like",
                "description": "Easy retrieval supervision / generic query-passage pairs",
                "examples": msmarco_train,
            }
        )
    if viquad_train:
        stages.append(
            {
                "name": "viquad",
                "description": "Question-answer style retrieval pairs",
                "examples": viquad_train,
            }
        )
    if train_news:
        stages.append(
            {
                "name": "news_hard_negative",
                "description": "Domain news with BM25 hard negatives",
                "examples": train_news,
            }
        )

    val_examples = msmarco_val + viquad_val + val_news
    marker.mkdir(parents=True, exist_ok=True)
    train_marker.write_text(
        json.dumps(
            {
                "stages": stages,
                "stage_sizes": {stage["name"]: len(stage["examples"]) for stage in stages},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    val_marker.write_text(json.dumps({"examples": val_examples}, ensure_ascii=False, indent=2), encoding="utf-8")
    return stages, val_examples


# ──────────────────────────────────────────────────────────────────────────────
# Metrics / trainer
# ──────────────────────────────────────────────────────────────────────────────


def build_metrics():
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    def compute(eval_pred):
        logits = np.asarray(eval_pred.predictions)
        labels = np.asarray(eval_pred.label_ids).reshape(-1)

        if logits.ndim == 1:
            pos = np.asarray(logits, dtype=np.float64)
            pred = (pos >= 0.5).astype(np.int32)
        else:
            # Dùng logits lớp dương trực tiếp thay vì softmax để tránh
            # phát sinh NaN khi model quá tự tin hoặc logits bị overflow.
            pos = np.asarray(logits[:, -1], dtype=np.float64)
            pred = np.argmax(logits, axis=1)

        valid_mask = np.isfinite(pos)
        if not np.all(valid_mask):
            pos = np.nan_to_num(pos, nan=0.0, posinf=1e6, neginf=-1e6)

        try:
            roc_auc = float(roc_auc_score(labels, pos)) if len(np.unique(labels)) > 1 else 0.0
        except ValueError:
            roc_auc = 0.0

        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "roc_auc": roc_auc,
        }

    return compute


def _build_weighted_dataset(examples: List[Dict], tokenizer, max_length: int):
    from datasets import Dataset

    ds = Dataset.from_list(
        [
            {
                "query": x["query"],
                "passage": x["passage"],
                "labels": int(x["labels"]),
                "source": x.get("source", ""),
                "difficulty": float(x.get("difficulty", 0.5)),
            }
            for x in examples
        ]
    )

    def tok(batch):
        enc = tokenizer(
            batch["query"],
            batch["passage"],
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = batch["labels"]
        enc["difficulty"] = batch["difficulty"]
        return enc

    return ds.map(tok, batched=True, remove_columns=ds.column_names)


def _sanitize_model_outputs(outputs):
    import torch

    logits = outputs.logits
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    return logits


class WeightedTrainer:
    """Compatibility placeholder.

    We intentionally keep the training loss standard and handle curriculum by
    stage composition / sampling only. This is more stable for reranker training
    while still matching the Ezpl goal of multi-source curriculum learning.
    """

    @staticmethod
    def build(TrainerBase):
        return TrainerBase


def _chunk_examples(examples: List[Dict], batch_size: int) -> List[List[Dict]]:
    if batch_size <= 0:
        return [examples]
    return [examples[i : i + batch_size] for i in range(0, len(examples), batch_size)]


def train(args):
    import torch
    from datasets import DatasetDict, load_from_disk
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    WeightedTrainerClass = WeightedTrainer.build(Trainer)

    class EarlyStop(TrainerCallback):
        def __init__(self, metric: str = "eval_roc_auc", patience: int = 2, threshold: float = 0.001):
            self.metric = metric
            self.patience = patience
            self.threshold = threshold
            self.best = None
            self.bad = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics or self.patience <= 0:
                return control
            cur = metrics.get(self.metric)
            if cur is None:
                return control
            if self.best is None or cur > self.best + self.threshold:
                self.best = cur
                self.bad = 0
            else:
                self.bad += 1
                if self.bad >= self.patience:
                    control.should_training_stop = True
            return control

    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("No CUDA detected. Use --allow-cpu only for debugging.")

    _log(f"[model] Loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    stages, val_examples = build_curriculum_examples(args)
    if not stages or not val_examples:
        raise RuntimeError("No curriculum stages/validation examples could be built.")

    val_ds = _build_weighted_dataset(val_examples, tokenizer, args.max_length)
    raw = DatasetDict({"validation": val_ds})

    cache_dir = output_dir / "_cache" / "tokenized"
    cache_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs_per_stage,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=False,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
    )

    overall_summary = {
        "base_model": args.model_name,
        "seed": args.seed,
        "stages": [],
        "validation_examples": len(val_examples),
        "output_dir": str(output_dir),
    }

    resume = args.resume_from_checkpoint
    if resume == "auto":
        ckpts = sorted([p for p in output_dir.glob("checkpoint-*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        resume = str(ckpts[-1]) if ckpts else None

    # Stage-wise curriculum training
    for stage_idx, stage in enumerate(stages, start=1):
        stage_name = stage["name"]
        stage_examples = stage["examples"]
        if not stage_examples:
            continue

        _log(f"\n[stage {stage_idx}] {stage_name} | {len(stage_examples)} examples")
        stage_cache = cache_dir / _stage_name(stage_idx)
        if stage_cache.exists():
            from datasets import load_from_disk

            tokenized_train = load_from_disk(str(stage_cache))
        else:
            train_ds = _build_weighted_dataset(stage_examples, tokenizer, args.max_length)
            tokenized_train = train_ds
            tokenized_train.save_to_disk(str(stage_cache))

        # Higher difficulty means more training focus when sampling/packing isn't available.
        # We pass the raw examples into metadata so future extensions can sample by difficulty.
        trainer = WeightedTrainerClass(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=raw["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=build_metrics(),
            callbacks=[EarlyStop(args.early_stopping_patience, args.early_stopping_threshold)] if args.early_stopping_patience > 0 else [],
        )

        stage_resume = resume if stage_idx == 1 else None
        if stage_resume is None:
            _log(f"[stage {stage_name}] fresh start")
        else:
            _log(f"[stage {stage_name}] resume from {stage_resume}")
        train_result = trainer.train(resume_from_checkpoint=stage_resume)
        metrics = trainer.evaluate()
        metrics = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in metrics.items()
        }
        _log(f"[stage {stage_name}] {json.dumps(metrics, ensure_ascii=False)}")

        overall_summary["stages"].append(
            {
                "name": stage_name,
                "description": stage["description"],
                "examples": len(stage_examples),
                "train_loss": float(getattr(train_result, "training_loss", 0.0) or 0.0),
                "eval": metrics,
            }
        )

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    archive = shutil.make_archive(
        str(output_dir.parent / output_dir.name),
        "zip",
        root_dir=str(output_dir.parent),
        base_dir=output_dir.name,
    )
    overall_summary.update(
        {
            "archive_path": archive,
            "epochs_per_stage": args.epochs_per_stage,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "eval_strategy": args.eval_strategy,
            "eval_steps": args.eval_steps,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "dataloader_num_workers": args.dataloader_num_workers,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_threshold": args.early_stopping_threshold,
            "hard_negatives": args.hard_negatives,
            "viquad_train_max": args.viquad_train_max,
            "viquad_val_max": args.viquad_val_max,
            "msmarco_train_max": args.msmarco_train_max,
            "msmarco_val_max": args.msmarco_val_max,
            "max_news_articles": args.max_news_articles,
            "news_val_ratio": args.news_val_ratio,
            "max_pseudo_queries_per_article": args.max_pseudo_queries_per_article,
            "quick_test": args.quick_test,
            "local_usage": "Giai nen model vao data/reranker_model/ roi chay: python main.py --load-index --reranker-dir data/reranker_model",
        }
    )
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description="Train Vietnamese reranker with curriculum learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", default="Fsoft-AIC/videberta-base")
    p.add_argument("--data-csv", default=str(ROOT_DIR / "data" / "vnexpress_articles.csv"))
    p.add_argument("--output-dir", default=_default_output_dir())
    p.add_argument("--epochs-per-stage", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--per-device-train-batch-size", type=int, default=8)
    p.add_argument("--per-device-eval-batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--save-strategy", choices=("steps", "epoch"), default="steps")
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-strategy", choices=("steps", "epoch", "no"), default="steps")
    p.add_argument("--eval-steps", type=int, default=2000)
    p.add_argument("--load-best-model-at-end", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=2)
    p.add_argument("--early-stopping-threshold", type=float, default=0.001)
    p.add_argument("--hard-negatives", type=int, default=2)
    p.add_argument("--max-news-articles", type=int, default=30000)
    p.add_argument("--news-val-ratio", type=float, default=0.05)
    p.add_argument("--max-pseudo-queries-per-article", type=int, default=2)
    p.add_argument("--viquad-train-max", type=int, default=12000)
    p.add_argument("--viquad-val-max", type=int, default=4000)
    p.add_argument("--msmarco-train-max", type=int, default=8000)
    p.add_argument("--msmarco-val-max", type=int, default=1000)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--resume-from-checkpoint", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick-test", action="store_true", help="Run a very small end-to-end smoke test")
    return p.parse_args(list(argv) if argv is not None else sys.argv[1:])


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if args.quick_test:
        args.allow_cpu = True
        args.epochs_per_stage = 0.05
        args.logging_steps = 1
        args.eval_steps = 2
        args.save_steps = 2
        args.save_total_limit = 1
        args.per_device_train_batch_size = 2
        args.per_device_eval_batch_size = 2
        args.gradient_accumulation_steps = 1
        args.max_news_articles = min(args.max_news_articles, 120)
        args.news_val_ratio = min(max(args.news_val_ratio, 0.1), 0.2)
        args.max_pseudo_queries_per_article = 1
        args.hard_negatives = 1
        args.viquad_train_max = min(args.viquad_train_max, 200)
        args.viquad_val_max = min(args.viquad_val_max, 80)
        args.msmarco_train_max = 0
        args.msmarco_val_max = 0
        args.eval_strategy = "steps"
        args.save_strategy = "steps"
        _log("[quick-test] Enabled end-to-end smoke test preset")
    train(args)


if __name__ == "__main__":
    main()
