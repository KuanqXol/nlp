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
    --output-dir data/reranker_model

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
    return str(ROOT_DIR / "data" / "reranker_model")


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


def _cuda_device_names() -> List[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _is_t4x2() -> bool:
    names = _cuda_device_names()
    return len(names) >= 2 and all("T4" in name.upper() for name in names[:2])


def _provided_cli_options(argv: Sequence[str]) -> set:
    out = set()
    for item in argv:
        if item.startswith("--"):
            out.add(item.split("=", 1)[0])
    return out


def _set_if_not_provided(args, provided: set, attr: str, option: str, value):
    if option not in provided:
        setattr(args, attr, value)


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
        _log(f"[data] Loading dataset {name} ({split})...")
        return load_dataset(name, split=split)
    except Exception as e:
        _log(f"[data] Skip dataset {name} ({split}): {type(e).__name__}: {e}")
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
    if train_max <= 0 and val_max <= 0:
        _log("[data] ViQuAD disabled (train/val max = 0)")
        return [], []
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

    train = convert(ds_train, train_max)
    val = convert(ds_val, val_max)
    _log(f"[data] ViQuAD examples: train={len(train):,}, val={len(val):,}")
    return train, val


def load_msmarco_like_examples(train_max: int, val_max: int) -> Tuple[List[Dict], List[Dict]]:
    """Optional external source aligned with Ezpl expectation.

    We try a couple of public datasets that behave similarly to MSMARCO-style
    retrieval examples. If unavailable, return empty lists so local training still works.
    """
    if train_max <= 0 and val_max <= 0:
        _log("[data] MSMARCO-like disabled (train/val max = 0)")
        return [], []
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
                train = items[split_idx:]
                val = items[:split_idx]
                _log(
                    f"[data] MSMARCO-like examples from {ds_name}: "
                    f"train={len(train):,}, val={len(val):,}"
                )
                return train, val
        except Exception:
            continue
    _log("[data] MSMARCO-like examples: train=0, val=0")
    return [], []


def load_news_articles(csv_path: str, max_articles: int = 30000) -> List[Dict]:
    path = _resolve_csv(csv_path)
    if not path.exists():
        _log(f"[data] Khong tim thay CSV: {path}")
        return []
    if max_articles <= 0:
        _log("[data] News disabled (max_news_articles <= 0)")
        return []
    _log(f"[data] Loading up to {max_articles:,} news articles from {path}...")
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
            if len(docs) % 1000 == 0:
                _log(f"[data] Loaded news articles: {len(docs):,}/{max_articles:,}")
    _log(f"[data] News articles loaded: {len(docs):,}")
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
    _log(
        f"[data] Building news examples: articles={len(articles):,}, "
        f"hard_negatives={hard_negatives}, max_pseudo={max_pseudo}"
    )
    rng = random.Random(seed)
    bm25 = SimpleBM25()
    _log("[data] Building BM25 for news hard negatives...")
    bm25.build([a["text"] for a in articles], [a["id"] for a in articles])
    docs_by_id = {a["id"]: a for a in articles}
    examples = []
    for idx, article in enumerate(articles, start=1):
        for q in pseudo_queries(article)[:max_pseudo]:
            pos = article["text"][:512]
            examples.append(Example(q, pos, 1, "news", difficulty=0.9).__dict__)
            negs = []
            if hard_negatives > 0:
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
        if idx % 1000 == 0 or idx == len(articles):
            _log(f"[data] News examples progress: {idx:,}/{len(articles):,} articles")
    # dedupe
    seen, out = set(), []
    for e in examples:
        key = (_normalize(e["query"]), _normalize(e["passage"]), int(e["labels"]), e["source"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    _log(f"[data] News examples built: {len(out):,}")
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


def _tokenized_cache_signature(args) -> str:
    payload = {
        "curriculum": _cache_signature(args),
        "model_name": args.model_name,
        "max_length": args.max_length,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _training_state_path(output_dir: Path) -> Path:
    return output_dir / "training_state.json"


def _load_training_state(output_dir: Path) -> Dict:
    path = _training_state_path(output_dir)
    if not path.exists():
        return {"completed_stage_names": [], "stage_checkpoints": {}, "latest_stage_idx": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        completed = data.get("completed_stage_names", [])
        if not isinstance(completed, list):
            completed = []
        stage_checkpoints = data.get("stage_checkpoints", {})
        if not isinstance(stage_checkpoints, dict):
            stage_checkpoints = {}
        latest_stage_idx = int(data.get("latest_stage_idx", len(completed)))
        return {
            "completed_stage_names": [str(x) for x in completed],
            "stage_checkpoints": {str(k): str(v) for k, v in stage_checkpoints.items()},
            "latest_stage_idx": max(0, latest_stage_idx),
        }
    except Exception:
        return {"completed_stage_names": [], "stage_checkpoints": {}, "latest_stage_idx": 0}


def _save_training_state(output_dir: Path, state: Dict):
    path = _training_state_path(output_dir)
    tmp = path.with_suffix(".json.tmp")
    payload = {
        "completed_stage_names": list(dict.fromkeys(state.get("completed_stage_names", []))),
        "stage_checkpoints": dict(state.get("stage_checkpoints", {})),
        "latest_stage_idx": int(state.get("latest_stage_idx", 0)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
    }
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _checkpoint_epoch(checkpoint_dir: Optional[Path]) -> Optional[float]:
    if checkpoint_dir is None:
        return None
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        epoch = data.get("epoch")
        return float(epoch) if epoch is not None else None
    except Exception:
        return None


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
        _log(f"[data] Loading cached curriculum: {marker}")
        payload = json.loads(train_marker.read_text(encoding="utf-8"))
        val_payload = json.loads(val_marker.read_text(encoding="utf-8"))
        _log(
            "[data] Cached stage sizes: "
            + ", ".join(f"{s['name']}={len(s['examples']):,}" for s in payload["stages"])
            + f" | val={len(val_payload['examples']):,}"
        )
        return payload["stages"], val_payload["examples"]

    _log("[data] Building curriculum examples...")
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
    _log(
        "[data] Curriculum stage sizes: "
        + ", ".join(f"{stage['name']}={len(stage['examples']):,}" for stage in stages)
        + f" | val={len(val_examples):,}"
    )
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
        return enc

    return ds.map(tok, batched=True, remove_columns=ds.column_names)


def _strip_unused_training_columns(ds):
    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    drop = [name for name in ds.column_names if name not in keep]
    return ds.remove_columns(drop) if drop else ds


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

    def _stage_output_dir(stage_name: str) -> Path:
        return output_dir / "stage_checkpoints" / stage_name

    def _stage_done_marker(stage_name: str) -> Path:
        return _stage_output_dir(stage_name) / "_DONE"

    def _stage_is_complete(stage_name: str) -> bool:
        stage_dir = _stage_output_dir(stage_name)
        has_weights = (stage_dir / "model.safetensors").exists() or (stage_dir / "pytorch_model.bin").exists()
        return (
            stage_dir.exists()
            and _stage_done_marker(stage_name).exists()
            and (stage_dir / "config.json").exists()
            and has_weights
        )

    def _latest_checkpoint_in_dir(stage_dir: Path) -> Optional[Path]:
        if not stage_dir.exists():
            return None
        def checkpoint_key(path: Path):
            match = re.match(r"checkpoint-(\d+)$", path.name)
            step = int(match.group(1)) if match else -1
            return (step, path.stat().st_mtime)

        ckpts = sorted([p for p in stage_dir.glob("checkpoint-*") if p.is_dir()], key=checkpoint_key)
        return ckpts[-1] if ckpts else None

    def _output_dir_has_usable_training_artifact(path: Path) -> bool:
        if (path / "training_state.json").exists():
            return True
        if (path / "config.json").exists() and (
            (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
        ):
            return True
        stage_root = path / "stage_checkpoints"
        if not stage_root.exists():
            return False
        for stage_dir in stage_root.iterdir():
            if not stage_dir.is_dir():
                continue
            if _stage_done_marker(stage_dir.name).exists() and (
                (stage_dir / "model.safetensors").exists() or (stage_dir / "pytorch_model.bin").exists()
            ):
                return True
            if _latest_checkpoint_in_dir(stage_dir) is not None:
                return True
        return False

    def _validate_output_dir_safety(path: Path):
        if not path.exists() or not any(path.iterdir()):
            return
        if _output_dir_has_usable_training_artifact(path):
            return
        if args.allow_partial_output_dir:
            _log(
                f"[safety] Continuing with partial output dir without usable checkpoints: {path}"
            )
            return
        raise RuntimeError(
            f"Output dir exists but has no usable reranker checkpoint/model: {path}\n"
            "This usually means an earlier run was interrupted after creating cache/stage folders. "
            "To avoid silently overwriting training state, choose a new --output-dir, remove this partial "
            "folder intentionally, or pass --allow-partial-output-dir if you really want to continue here."
        )

    def _force_float32_model(model):
        # Fsoft-AIC/videberta-base can carry dtype=float16 in config. Training a
        # sequence classifier from that state may silently save NaN weights, so
        # keep the reranker in fp32 unless mixed precision is explicitly added.
        model = model.float()
        for attr in ("torch_dtype", "dtype"):
            if hasattr(model.config, attr):
                try:
                    setattr(model.config, attr, "float32")
                except Exception:
                    pass
        return model

    def _assert_finite_model(model, context: str):
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                raise RuntimeError(
                    f"Non-finite reranker weights detected {context}: {name}. "
                    "Discard this checkpoint and rerun training from a clean output dir."
                )

    def _load_completed_stage_model(stage_name: str):
        stage_dir = _stage_output_dir(stage_name)
        _log(f"[stage {stage_name}] loading completed stage model from {stage_dir}")
        loaded = AutoModelForSequenceClassification.from_pretrained(
            str(stage_dir),
            num_labels=2,
            torch_dtype=torch.float32,
        )
        loaded = _force_float32_model(loaded)
        _assert_finite_model(loaded, f"after loading completed stage {stage_name}")
        return loaded

    def _resolve_fp16() -> bool:
        if args.no_fp16 or args.mixed_precision == "fp32":
            return False
        if args.fp16 or args.mixed_precision == "fp16":
            return torch.cuda.is_available()
        if args.mixed_precision == "auto":
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            # T4 benefits heavily from AMP tensor cores. The model is still
            # loaded in fp32, so AMP avoids the old half-config NaN issue.
            return torch.cuda.is_available() and any("T4" in name.upper() for name in names)
        return False

    def _configure_cuda_runtime():
        if not torch.cuda.is_available():
            return
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision(args.float32_matmul_precision)
            except Exception:
                pass

    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    _validate_output_dir_safety(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(args.export_dir) if args.export_dir else output_dir

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("No CUDA detected. Use --allow-cpu only for debugging.")

    _configure_cuda_runtime()
    fp16_enabled = _resolve_fp16()
    bf16_enabled = False
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    device_names = [torch.cuda.get_device_name(i) for i in range(n_gpu)] if n_gpu else ["CPU"]
    gpu_mem = (
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        if torch.cuda.is_available()
        else 0
    )
    effective_batch = args.per_device_train_batch_size * max(1, n_gpu) * args.gradient_accumulation_steps
    _log(
        "[config] "
        f"device={device_name}, visible_gpus={n_gpu}, gpu_names={device_names}, gpu_mem_gb={gpu_mem}, "
        f"epochs_per_stage={args.epochs_per_stage}, "
        f"output_dir={output_dir}, export_dir={export_dir}, "
        f"max_length={args.max_length}, train_batch={args.per_device_train_batch_size}, "
        f"eval_batch={args.per_device_eval_batch_size}, grad_accum={args.gradient_accumulation_steps}, "
        f"effective_global_batch={effective_batch}, "
        f"news={args.max_news_articles}, viquad={args.viquad_train_max}/{args.viquad_val_max}, "
        f"msmarco={args.msmarco_train_max}/{args.msmarco_val_max}, hard_neg={args.hard_negatives}, "
        f"precision={'fp16_amp' if fp16_enabled else 'fp32'}, "
        f"gradient_checkpointing={args.gradient_checkpointing}, "
        f"t4x2_preset={getattr(args, '_t4x2_preset_active', False)}, "
        f"early_stopping_patience={args.early_stopping_patience}"
    )

    _log(f"[model] Loading {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        torch_dtype=torch.float32,
    )
    model = _force_float32_model(model)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    _assert_finite_model(model, "after base model load")

    stages, val_examples = build_curriculum_examples(args)
    if not stages or not val_examples:
        raise RuntimeError("No curriculum stages/validation examples could be built.")

    val_ds = _build_weighted_dataset(val_examples, tokenizer, args.max_length)
    raw = DatasetDict({"validation": val_ds})

    cache_dir = output_dir / "_cache" / "tokenized" / _tokenized_cache_signature(args)
    cache_dir.mkdir(parents=True, exist_ok=True)

    overall_summary = {
        "base_model": args.model_name,
        "seed": args.seed,
        "stages": [],
        "validation_examples": len(val_examples),
        "output_dir": str(output_dir),
        "export_dir": str(export_dir),
        "visible_gpus": n_gpu,
        "gpu_names": device_names,
        "fp16_amp": fp16_enabled,
        "gradient_checkpointing": args.gradient_checkpointing,
        "t4x2_preset": getattr(args, "_t4x2_preset_active", False),
        "stage_complete": {},
    }

    training_state = _load_training_state(output_dir)
    completed_stage_names = set(training_state.get("completed_stage_names", []))
    stage_checkpoints = dict(training_state.get("stage_checkpoints", {}))
    latest_stage_idx = int(training_state.get("latest_stage_idx", 0))

    resume = args.resume_from_checkpoint
    if resume == "auto":
        ckpts = sorted([p for p in output_dir.glob("checkpoint-*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
        resume = str(ckpts[-1]) if ckpts else None

    last_trainer = None
    last_completed_stage_name = None

    for stage_idx, stage in enumerate(stages, start=1):
        stage_name = stage["name"]
        stage_examples = stage["examples"]
        if not stage_examples:
            continue

        stage_dir = _stage_output_dir(stage_name)
        stage_dir.mkdir(parents=True, exist_ok=True)

        stage_files_complete = _stage_is_complete(stage_name)
        stage_resume = _latest_checkpoint_in_dir(stage_dir)
        stage_resume_epoch = _checkpoint_epoch(stage_resume)
        stage_state_complete = stage_name in completed_stage_names
        if stage_state_complete and not stage_files_complete:
            _log(f"\n[stage {stage_idx}] {stage_name} is listed complete but checkpoint files are missing; retraining")
        stage_marked_complete = stage_files_complete
        stage_needs_more_epochs = (
            stage_marked_complete
            and stage_resume is not None
            and stage_resume_epoch is not None
            and stage_resume_epoch + 1e-6 < args.epochs_per_stage
        )
        retrain_completed_stage = False
        if stage_needs_more_epochs and not args.retrain_completed_stages:
            _log(
                f"\n[stage {stage_idx}] {stage_name} was marked complete at epoch "
                f"{stage_resume_epoch:.4g}; resuming to target epoch {args.epochs_per_stage}"
            )
            done_marker = _stage_done_marker(stage_name)
            if done_marker.exists():
                done_marker.unlink()
            completed_stage_names.discard(stage_name)
            stage_marked_complete = False
        if stage_marked_complete and not args.retrain_completed_stages:
            _log(f"\n[stage {stage_idx}] {stage_name} already complete, skip")
            model = _load_completed_stage_model(stage_name)
            overall_summary["stage_complete"][stage_name] = True
            last_completed_stage_name = stage_name
            continue
        if stage_marked_complete and args.retrain_completed_stages:
            _log(f"\n[stage {stage_idx}] {stage_name} marked complete; retraining because --retrain-completed-stages is set")
            done_marker = _stage_done_marker(stage_name)
            if done_marker.exists():
                done_marker.unlink()
            completed_stage_names.discard(stage_name)
            retrain_completed_stage = True

        _log(f"\n[stage {stage_idx}] {stage_name} | {len(stage_examples)} examples")
        stage_cache = cache_dir / stage_name
        if stage_cache.exists():
            tokenized_train = _strip_unused_training_columns(load_from_disk(str(stage_cache)))
        else:
            tokenized_train = _strip_unused_training_columns(
                _build_weighted_dataset(stage_examples, tokenizer, args.max_length)
            )
            tokenized_train.save_to_disk(str(stage_cache))

        stage_save_steps = args.save_steps
        if stage_idx == 2 and args.stage2_save_steps > 0:
            stage_save_steps = args.stage2_save_steps

        stage_eval_steps = args.eval_steps
        if (
            args.load_best_model_at_end
            and args.eval_strategy == "steps"
            and args.save_strategy == "steps"
            and stage_eval_steps > 0
            and stage_save_steps % stage_eval_steps != 0
        ):
            stage_eval_steps = stage_save_steps
            _log(
                f"[stage {stage_name}] eval_steps adjusted to {stage_eval_steps} "
                "so load_best_model_at_end can use the stage checkpoint cadence"
            )

        _log(f"[stage {stage_name}] save_steps={stage_save_steps}, eval_steps={stage_eval_steps}")

        stage_training_args = TrainingArguments(
            output_dir=str(stage_dir),
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
            save_steps=stage_save_steps,
            eval_steps=stage_eval_steps,
            logging_steps=args.logging_steps,
            load_best_model_at_end=args.load_best_model_at_end and args.eval_strategy != "no",
            metric_for_best_model="eval_roc_auc",
            greater_is_better=True,
            fp16=fp16_enabled,
            bf16=bf16_enabled,
            report_to=[],
            remove_unused_columns=True,
            save_total_limit=None if args.save_total_limit <= 0 else args.save_total_limit,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        callbacks = []
        if args.early_stopping_patience > 0:
            if args.eval_strategy == "no":
                _log("[early-stop] Disabled because eval_strategy=no")
            else:
                callbacks.append(
                    EarlyStop(
                        patience=args.early_stopping_patience,
                        threshold=args.early_stopping_threshold,
                    )
                )
        trainer = WeightedTrainerClass(
            model=model,
            args=stage_training_args,
            train_dataset=tokenized_train,
            eval_dataset=raw["validation"],
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=build_metrics(),
            callbacks=callbacks,
        )
        last_trainer = trainer

        stage_resume = None if retrain_completed_stage else stage_resume
        if stage_resume is None and not retrain_completed_stage and stage_idx == 1 and resume:
            stage_resume = Path(resume)
        if stage_resume is None:
            _log(f"[stage {stage_name}] fresh start")
        else:
            _log(f"[stage {stage_name}] resume from {stage_resume}")

        train_result = trainer.train(resume_from_checkpoint=str(stage_resume) if stage_resume else None)
        metrics = trainer.evaluate()
        metrics = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()}
        _log(f"[stage {stage_name}] {json.dumps(metrics, ensure_ascii=False)}")

        _force_float32_model(trainer.model)
        _assert_finite_model(trainer.model, f"after stage {stage_name}")
        trainer.save_model(str(stage_dir))
        tokenizer.save_pretrained(str(stage_dir))
        _stage_done_marker(stage_name).write_text("done\n", encoding="utf-8")

        completed_stage_names.add(stage_name)
        latest_stage_idx = stage_idx
        stage_checkpoints[stage_name] = str(stage_dir)
        _save_training_state(output_dir, {
            "completed_stage_names": sorted(completed_stage_names, key=lambda n: [s["name"] for s in stages].index(n) if n in [s["name"] for s in stages] else 10**9),
            "stage_checkpoints": stage_checkpoints,
            "latest_stage_idx": latest_stage_idx,
        })
        overall_summary["stage_complete"][stage_name] = True
        last_completed_stage_name = stage_name

        overall_summary["stages"].append(
            {
                "name": stage_name,
                "description": stage["description"],
                "examples": len(stage_examples),
                "train_loss": float(getattr(train_result, "training_loss", 0.0) or 0.0),
                "eval": metrics,
                "checkpoint_dir": str(stage_dir),
                "save_steps": stage_save_steps,
                "eval_steps": stage_eval_steps,
            }
        )

    if last_completed_stage_name:
        final_source = _stage_output_dir(last_completed_stage_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(final_source),
            num_labels=2,
            torch_dtype=torch.float32,
        )
        model = _force_float32_model(model)
        _assert_finite_model(model, f"before final export from {final_source}")
        export_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(export_dir))
        tokenizer.save_pretrained(str(export_dir))
    elif last_trainer is not None:
        _force_float32_model(last_trainer.model)
        _assert_finite_model(last_trainer.model, "before final export")
        export_dir.mkdir(parents=True, exist_ok=True)
        last_trainer.save_model(str(export_dir))
        tokenizer.save_pretrained(str(export_dir))

    _save_training_state(output_dir, {
        "completed_stage_names": sorted(completed_stage_names, key=lambda n: [s["name"] for s in stages].index(n) if n in [s["name"] for s in stages] else 10**9),
        "stage_checkpoints": stage_checkpoints,
        "latest_stage_idx": latest_stage_idx,
    })

    archive = shutil.make_archive(
        str(output_dir.parent / output_dir.name),
        "zip",
        root_dir=str(output_dir.parent),
        base_dir=output_dir.name,
    )
    overall_summary.update(
        {
            "archive_path": archive,
            "training_state_path": str(_training_state_path(output_dir)),
            "export_dir": str(export_dir),
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
            "stage2_save_steps": args.stage2_save_steps,
            "save_total_limit": args.save_total_limit,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_global_batch": effective_batch,
            "dataloader_num_workers": args.dataloader_num_workers,
            "fp16_amp": fp16_enabled,
            "mixed_precision": args.mixed_precision,
            "gradient_checkpointing": args.gradient_checkpointing,
            "visible_gpus": n_gpu,
            "gpu_names": device_names,
            "t4x2_preset": getattr(args, "_t4x2_preset_active", False),
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
            "retrain_completed_stages": args.retrain_completed_stages,
            "resume_from_checkpoint": args.resume_from_checkpoint,
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
    p.add_argument(
        "--export-dir",
        default=None,
        help="Optional final runtime model directory; defaults to --output-dir",
    )
    # Defaults target a personal laptop with RTX 3050 Ti 4GB + 16GB RAM.
    # They favor unattended stability over maximum throughput.
    p.add_argument("--epochs-per-stage", type=float, default=5.0)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument(
        "--mixed-precision",
        choices=("auto", "fp16", "fp32"),
        default="auto",
        help="auto enables fp16 AMP on CUDA T4; fp32 disables mixed precision",
    )
    p.add_argument("--fp16", action="store_true", help="Alias for --mixed-precision fp16")
    p.add_argument("--no-fp16", action="store_true", help="Alias for --mixed-precision fp32")
    p.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduce memory use at the cost of speed; useful if larger batches OOM",
    )
    p.add_argument(
        "--float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
        help="PyTorch matmul precision hint; mainly affects newer GPUs, harmless on T4",
    )
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=16)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-total-limit", type=int, default=5, help="Max checkpoints kept per stage; <=0 keeps all")
    p.add_argument("--save-strategy", choices=("steps", "epoch"), default="steps")
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument(
        "--stage2-save-steps",
        type=int,
        default=20,
        help="Checkpoint interval for stage 2 when save_strategy=steps; <=0 uses --save-steps",
    )
    p.add_argument("--eval-strategy", choices=("steps", "epoch", "no"), default="steps")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--load-best-model-at-end", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=0, help="0 disables early stopping")
    p.add_argument("--early-stopping-threshold", type=float, default=0.001)
    p.add_argument("--hard-negatives", type=int, default=1)
    p.add_argument("--max-news-articles", type=int, default=6000)
    p.add_argument("--news-val-ratio", type=float, default=0.05)
    p.add_argument("--max-pseudo-queries-per-article", type=int, default=1)
    p.add_argument("--viquad-train-max", type=int, default=3000)
    p.add_argument("--viquad-val-max", type=int, default=500)
    p.add_argument("--msmarco-train-max", type=int, default=0)
    p.add_argument("--msmarco-val-max", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument(
        "--allow-partial-output-dir",
        action="store_true",
        help="Allow training in an existing output dir that has cache folders but no usable checkpoint/model",
    )
    p.add_argument("--resume-from-checkpoint", type=str, default="auto")
    p.add_argument(
        "--retrain-completed-stages",
        action="store_true",
        help="Train stages even if _DONE markers exist; completed stages start again from the current model",
    )
    p.add_argument(
        "--t4x2-preset",
        choices=("auto", "on", "off"),
        default="auto",
        help="Auto-tune conservative throughput defaults when Kaggle exposes 2 Tesla T4 GPUs",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick-test", action="store_true", help="Run a very small end-to-end smoke test")
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = p.parse_args(raw_argv)
    args._provided_options = _provided_cli_options(raw_argv)
    return args


def _apply_t4x2_preset(args):
    provided = getattr(args, "_provided_options", set())
    active = args.t4x2_preset == "on" or (args.t4x2_preset == "auto" and _is_t4x2())
    args._t4x2_preset_active = bool(active)
    if not active:
        return

    # Conservative T4 x2 defaults: enough work to keep both GPUs busy while
    # staying below 16GB/GPU for ViDeBERTa at max_length=192.
    _set_if_not_provided(args, provided, "per_device_train_batch_size", "--per-device-train-batch-size", 4)
    _set_if_not_provided(args, provided, "per_device_eval_batch_size", "--per-device-eval-batch-size", 8)
    _set_if_not_provided(args, provided, "gradient_accumulation_steps", "--gradient-accumulation-steps", 4)
    _set_if_not_provided(args, provided, "dataloader_num_workers", "--dataloader-num-workers", 2)
    _set_if_not_provided(args, provided, "save_total_limit", "--save-total-limit", 2)
    _set_if_not_provided(args, provided, "stage2_save_steps", "--stage2-save-steps", args.save_steps)
    _set_if_not_provided(args, provided, "max_news_articles", "--max-news-articles", 10000)
    _set_if_not_provided(args, provided, "viquad_train_max", "--viquad-train-max", 5000)
    _set_if_not_provided(args, provided, "viquad_val_max", "--viquad-val-max", 800)

    if "--mixed-precision" not in provided and "--fp16" not in provided and "--no-fp16" not in provided:
        args.mixed_precision = "auto"

    _log(
        "[preset] T4 x2 preset active: "
        f"train_batch={args.per_device_train_batch_size}, "
        f"eval_batch={args.per_device_eval_batch_size}, "
        f"grad_accum={args.gradient_accumulation_steps}, "
        f"workers={args.dataloader_num_workers}, "
        f"max_news={args.max_news_articles}, "
        f"mixed_precision={args.mixed_precision}"
    )


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    if args.quick_test:
        args.allow_cpu = True
        args.epochs_per_stage = 0.05
        args.logging_steps = 1
        args.eval_steps = 2
        args.save_steps = 2
        args.stage2_save_steps = min(args.stage2_save_steps, args.save_steps)
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
    else:
        _apply_t4x2_preset(args)
    train(args)


if __name__ == "__main__":
    main()
