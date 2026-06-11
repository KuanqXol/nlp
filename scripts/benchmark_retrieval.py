"""Benchmark retrieval methods on a 1,000-document subset.

Compares:
- Current pipeline (vector retrieval + optional rerank)
- Vector only (FAISS / bi-encoder)
- BM25
- TF-IDF cosine similarity

The default evaluation strategy is self-retrieval:
- sample up to 1,000 documents from the corpus
- use each document title as a query
- treat the originating document as the relevant item

Outputs:
- JSON summary
- Markdown report
- Optional CSV with per-query results

Example:
    python scripts/benchmark_retrieval.py --data data/vnexpress_articles.csv --sample-size 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import NewsDataLoader, create_document
from src.retrieval import EmbeddingManager, Retriever, chunk_documents
from src.retrieval.retriever import DEFAULT_CROSS_ENCODER


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


DEFAULT_OUTPUT_DIR = Path("data") / "benchmarks"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_TOP_K = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_QUERY_FIELD = "title"
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "data" / "bi_encoder_model"
DEFAULT_MODEL_NAME = (
    str(DEFAULT_LOCAL_MODEL)
    if (DEFAULT_LOCAL_MODEL / "modules.json").exists()
    else "bkai-foundation-models/vietnamese-bi-encoder"
)


@dataclass
class QueryResult:
    query_id: str
    query_text: str
    relevant_doc_id: str
    method: str
    top1_hit: bool
    top5_hit: bool
    top10_hit: bool
    rank: Optional[int]
    latency_ms: float
    returned_doc_ids: str


class SimpleBM25:
    """Minimal BM25 implementation without external dependencies."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [self._tokenize(doc) for doc in documents]
        self.doc_len = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.df = Counter()
        for doc in self.documents:
            for term in set(doc):
                self.df[term] += 1
        self.n_docs = len(self.documents)
        self.idf = {
            term: math.log(1 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }
        self.tf = [Counter(doc) for doc in self.documents]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in (text or "").lower().split() if t]

    def get_scores(self, query: str) -> List[float]:
        q_tokens = self._tokenize(query)
        scores = [0.0] * self.n_docs
        for i, doc_tf in enumerate(self.tf):
            score = 0.0
            dl = self.doc_len[i] or 1
            for term in q_tokens:
                if term not in doc_tf:
                    continue
                idf = self.idf.get(term, 0.0)
                tf = doc_tf[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (tf * (self.k1 + 1)) / denom
            scores[i] = score
        return scores

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        scores = self.get_scores(query)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:top_k]
        return ranked


class TfidfBackend:
    def __init__(self, documents: Sequence[str]):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for TF-IDF benchmark")
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        ranked = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in ranked]


def _tokenize_title(text: str) -> str:
    return " ".join((text or "").split())


def _select_documents(docs: List[Dict], sample_size: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    eligible = [d for d in docs if d.get("title") and d.get("content")]
    if len(eligible) <= sample_size:
        return eligible
    return rng.sample(eligible, sample_size)


def _load_csv_subset(
    data_path: Path,
    sample_size: int,
    seed: int,
    max_docs: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    """Reservoir-sample a CSV without loading the full corpus into memory.

    This mirrors scripts/evaluate_test_model.py so benchmark runs use the same
    self-retrieval setup while still allowing BM25 / TF-IDF / reranker compare.
    """
    rng = random.Random(seed)
    selected: List[Dict] = []
    seen_urls = set()
    total_rows = 0
    eligible_seen = 0
    skipped_invalid = 0
    skipped_dedup = 0

    with open(data_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            doc = create_document(row)
            if not doc:
                skipped_invalid += 1
                continue
            if doc.get("url") and doc["url"] in seen_urls:
                skipped_dedup += 1
                continue
            if doc.get("url"):
                seen_urls.add(doc["url"])

            eligible_seen += 1
            if max_docs is not None and max_docs > 0:
                selected.append(doc)
                if len(selected) >= max_docs:
                    break
                continue

            if len(selected) < sample_size:
                selected.append(doc)
                continue

            replace_at = rng.randint(0, eligible_seen - 1)
            if replace_at < sample_size:
                selected[replace_at] = doc

    return selected, {
        "loader": "stream_csv_reservoir",
        "total_rows_seen": total_rows,
        "eligible_seen": eligible_seen,
        "loaded_docs": len(selected),
        "skipped_invalid": skipped_invalid,
        "skipped_dedup": skipped_dedup,
        "selection_mode": "head" if max_docs is not None and max_docs > 0 else "sample",
    }


def _load_documents(
    data_path: str,
    sample_size: int,
    seed: int,
    max_docs: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    if path.suffix.lower() == ".csv":
        return _load_csv_subset(path, sample_size=sample_size, seed=seed, max_docs=max_docs)
    loader = NewsDataLoader(str(path))
    docs = loader.load()
    selected = (
        [d for d in docs if d.get("title") and d.get("content")][:max_docs]
        if max_docs is not None and max_docs > 0
        else _select_documents(docs, sample_size=sample_size, seed=seed)
    )
    return selected, {
        "loader": "NewsDataLoader",
        "load_stats": dict(loader.last_load_stats),
        "selection_mode": "head" if max_docs is not None and max_docs > 0 else "sample",
    }


def _build_query_items(documents: List[Dict], query_field: str) -> List[Dict]:
    items = []
    for idx, doc in enumerate(documents):
        query_text = _tokenize_title(doc.get(query_field, ""))
        if not query_text:
            continue
        items.append(
            {
                "query_id": f"q{idx}",
                "query_text": query_text,
                "relevant_doc_id": doc["id"],
                "relevant_title": doc.get("title", ""),
            }
        )
    return items


def _load_query_items_from_file(query_file: str, valid_doc_ids: set[str]) -> Tuple[List[Dict], Dict]:
    path = Path(query_file)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy query file: {path}")

    rows: List[Dict] = []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Query JSONL lỗi ở dòng {line_no}: {exc}") from exc
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("queries", [])
        if not isinstance(payload, list):
            raise ValueError("Query JSON phải là list hoặc object có field 'queries'.")
        rows = payload
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError("Query file chỉ hỗ trợ .jsonl, .json, hoặc .csv")

    items = []
    skipped_missing_text = 0
    skipped_missing_relevant = 0
    skipped_not_in_corpus = 0
    for idx, row in enumerate(rows):
        query_text = _tokenize_title(str(row.get("query_text") or row.get("query") or ""))
        relevant_doc_id = str(row.get("relevant_doc_id") or row.get("doc_id") or "")
        if not query_text:
            skipped_missing_text += 1
            continue
        if not relevant_doc_id:
            skipped_missing_relevant += 1
            continue
        if relevant_doc_id not in valid_doc_ids:
            skipped_not_in_corpus += 1
            continue
        items.append(
            {
                "query_id": str(row.get("query_id") or f"q{idx}"),
                "query_text": query_text,
                "relevant_doc_id": relevant_doc_id,
                "relevant_title": str(row.get("relevant_title") or ""),
            }
        )

    return items, {
        "path": str(path),
        "loaded_rows": len(rows),
        "usable_queries": len(items),
        "skipped_missing_text": skipped_missing_text,
        "skipped_missing_relevant": skipped_missing_relevant,
        "skipped_relevant_not_in_corpus": skipped_not_in_corpus,
    }


def _rank_of_target(results: List[Tuple[int, float]], target_idx: int) -> Optional[int]:
    for rank, (idx, _) in enumerate(results, start=1):
        if idx == target_idx:
            return rank
    return None


def _metrics_from_ranks(ranks: List[Optional[int]], top_k: int) -> Dict[str, float]:
    total = len(ranks)
    valid = [r for r in ranks if r is not None]
    denom = total or 1
    hit1 = sum(1 for r in ranks if r == 1)
    hit5 = sum(1 for r in ranks if r is not None and r <= 5)
    hit10 = sum(1 for r in ranks if r is not None and r <= 10)
    hitk = sum(1 for r in ranks if r is not None and r <= top_k)
    precision_at_10 = hit10 / (denom * 10)
    recall_at_10 = hit10 / denom
    f1_at_10 = (
        2 * precision_at_10 * recall_at_10 / (precision_at_10 + recall_at_10)
        if (precision_at_10 + recall_at_10)
        else 0.0
    )
    return {
        "queries": total,
        "query_count": total,
        "coverage": len(valid) / denom,
        "miss_count": total - len(valid),
        # Backward-compatible names from older reports.
        "top1": hit1 / denom,
        "top5": hit5 / denom,
        "top10": hit10 / denom,
        "mrr": sum((1 / r) for r in valid) / denom if valid else 0.0,
        # Explicit IR metrics. Single-positive self-retrieval: recall@K equals hit@K.
        "hit@1": hit1 / denom,
        "hit@5": hit5 / denom,
        "hit@10": hit10 / denom,
        "recall@1": hit1 / denom,
        "recall@5": hit5 / denom,
        "recall@10": recall_at_10,
        "recall@top_k": hitk / denom,
        "precision@1": hit1 / denom,
        "precision@5": hit5 / (denom * 5),
        "precision@10": precision_at_10,
        "f1@10": f1_at_10,
        "mrr@10": sum((1 / r) for r in ranks if r is not None and r <= 10) / denom,
        "map@10": sum((1 / r) for r in ranks if r is not None and r <= 10) / denom,
        "ndcg@10": sum(
            (1 / math.log2(r + 1)) for r in ranks if r is not None and r <= 10
        )
        / denom,
        "mean_rank": statistics.mean(valid) if valid else float("nan"),
        "median_rank": statistics.median(valid) if valid else float("nan"),
    }


def _latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = sorted(latencies)
    def _pct(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        k = (len(arr) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return arr[int(k)]
        return arr[f] * (c - k) + arr[c] * (k - f)
    return {
        "avg_ms": statistics.mean(arr),
        "p50_ms": _pct(0.50),
        "p95_ms": _pct(0.95),
        "p99_ms": _pct(0.99),
    }


def _evaluate_method(
    method_name: str,
    search_fn,
    query_items: List[Dict],
    doc_id_to_idx: Dict[str, int],
    top_k: int,
) -> Tuple[List[QueryResult], Dict[str, float], Dict[str, float]]:
    rows: List[QueryResult] = []
    ranks: List[Optional[int]] = []
    latencies: List[float] = []

    for item in query_items:
        t0 = time.perf_counter()
        results = search_fn(item["query_text"], top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        ranked_ids = [doc_id for doc_id, _ in results]
        rank = _rank_of_target(results, doc_id_to_idx[item["relevant_doc_id"]])
        ranks.append(rank)
        rows.append(
            QueryResult(
                query_id=item["query_id"],
                query_text=item["query_text"],
                relevant_doc_id=item["relevant_doc_id"],
                method=method_name,
                top1_hit=rank == 1,
                top5_hit=rank is not None and rank <= 5,
                top10_hit=rank is not None and rank <= 10,
                rank=rank,
                latency_ms=latency_ms,
                returned_doc_ids="|".join(str(idx) for idx, _ in results[:top_k]),
            )
        )

    metrics = _metrics_from_ranks(ranks, top_k=top_k)
    latency = _latency_stats(latencies)
    return rows, metrics, latency


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_csv(path: Path, rows: List[QueryResult]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(QueryResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _format_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _markdown_report(summary: Dict) -> str:
    lines = []
    lines.append("# Retrieval benchmark report")
    lines.append("")
    lines.append(f"- Corpus subset: **{summary['setup']['sample_size']} documents**")
    lines.append(f"- Queries: **{summary['setup']['query_count']}**")
    lines.append(f"- Top-K evaluated: **{summary['setup']['top_k']}**")
    lines.append(f"- Query source: **{summary['setup'].get('query_source', 'self_retrieval')}**")
    if summary["setup"].get("query_source") == "query_file":
        lines.append(f"- Query file: **{summary['setup'].get('query_file')}**")
    else:
        lines.append(f"- Query field: **{summary['setup']['query_field']}**")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    for key, value in summary["setup"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Method | R@1 | R@5 | R@10 | P@10 | MRR@10 | nDCG@10 | Avg latency (ms) | P95 (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method, data in summary["methods"].items():
        lines.append(
            f"| {method} | {_format_pct(data['metrics']['recall@1'])} | "
            f"{_format_pct(data['metrics']['recall@5'])} | "
            f"{_format_pct(data['metrics']['recall@10'])} | "
            f"{data['metrics']['precision@10']:.4f} | "
            f"{data['metrics']['mrr@10']:.3f} | "
            f"{data['metrics']['ndcg@10']:.3f} | "
            f"{data['latency']['avg_ms']:.2f} | {data['latency']['p95_ms']:.2f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if summary["setup"].get("query_source") == "query_file":
        lines.append("- Đây là benchmark dùng **query file**: query tách khỏi title gốc, nhưng nhãn vẫn có thể là weak-label nếu được sinh tự động.")
        lines.append("- Nên rà thủ công các query sai/top miss trước khi dùng làm kết luận cuối cùng về reranker.")
    else:
        lines.append("- Đây là benchmark **self-retrieval**: dùng title làm query và coi bài gốc là ground truth.")
        lines.append("- Chỉ số này phản ánh độ nhanh và khả năng kéo đúng bài tương ứng, không thay thế bộ đánh giá relevance gán nhãn thủ công.")
        lines.append("- Nếu muốn đánh giá sát thực tế hơn, nên tạo thêm 30-100 query thủ công có nhãn relevance.")
    return "\n".join(lines)


def build_backends(
    documents: List[Dict],
    use_current_pipeline: bool = True,
    use_reranker: bool = True,
    reranker_model_dir: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
):
    texts = []
    for doc in documents:
        texts.append(doc.get("full_text") or f"{doc.get('title', '')}. {doc.get('content', '')}")

    bm25 = SimpleBM25(texts)
    tfidf = TfidfBackend(texts) if _SKLEARN_AVAILABLE else None

    em = EmbeddingManager(model_name=model_name)
    chunk_docs, doc_to_chunks = chunk_documents(
        documents,
        strategy="sentence_window",
        max_chars=400,
        overlap=1,
        prepend_title=True,
    )
    chunk_dicts = [{"id": c["chunk_id"], "full_text": c["chunk_text"]} for c in chunk_docs]
    em.build_document_index(chunk_dicts)

    retriever = None
    if use_current_pipeline:
        retriever = Retriever(
            use_cross_encoder=use_reranker,
            reranker_model_dir=reranker_model_dir,
        )
        retriever.build(chunk_docs, em, doc_to_chunks, documents)

    return bm25, tfidf, em, retriever


def run_benchmark(
    data_path: str,
    sample_size: int,
    top_k: int,
    query_field: str,
    output_dir: str,
    seed: int,
    max_docs: Optional[int] = None,
    use_current_pipeline: bool = True,
    use_reranker: bool = True,
    reranker_model_dir: Optional[str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    query_file: Optional[str] = None,
    write_details: bool = True,
):
    sample_docs, load_info = _load_documents(
        data_path,
        sample_size=sample_size,
        seed=seed,
        max_docs=max_docs,
    )
    query_file_info = None
    if query_file:
        query_items, query_file_info = _load_query_items_from_file(
            query_file,
            valid_doc_ids={d.get("id", "") for d in sample_docs},
        )
    else:
        query_items = _build_query_items(sample_docs, query_field=query_field)
        query_items = query_items[:sample_size]

    if not query_items:
        if query_file_info:
            raise RuntimeError(
                "Không tạo được query items từ query file. "
                f"Chi tiết: {query_file_info}"
            )
        raise RuntimeError("Không tạo được query items từ dữ liệu đã chọn.")

    if not query_file:
        sample_docs = [
            d for d in sample_docs if d.get("id") in {q["relevant_doc_id"] for q in query_items}
        ]
    doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(sample_docs)}

    bm25, tfidf, em, retriever = build_backends(
        sample_docs,
        use_current_pipeline=use_current_pipeline,
        use_reranker=use_reranker,
        reranker_model_dir=reranker_model_dir,
        model_name=model_name,
    )

    methods = {}
    all_rows: List[QueryResult] = []

    def bm25_search(query: str, k: int):
        return bm25.search(query, top_k=k)

    def tfidf_search(query: str, k: int):
        if tfidf is None:
            return []
        return tfidf.search(query, top_k=k)

    def vector_search(query: str, k: int):
        qvec = em.encode_query(query)
        chunk_ids, scores = retriever._backend.search(qvec, k=min(50, len(em.doc_ids)))
        results = []
        for cid, score in zip(chunk_ids, scores):
            chunk = retriever._chunks.get(cid, {})
            doc_id = chunk.get("doc_id")
            if doc_id in doc_id_to_idx:
                results.append((doc_id_to_idx[doc_id], float(score)))
        # dedupe by doc index, keep best score
        best = {}
        for idx, score in results:
            best[idx] = max(best.get(idx, float("-inf")), score)
        return sorted(best.items(), key=lambda x: -x[1])[:k]

    def current_search(query: str, k: int):
        if not retriever:
            return []
        results = retriever.retrieve(query, top_k=k, rerank=False, apply_decay=False)
        ranked = []
        for item in results:
            doc_id = item.get("id", "")
            if doc_id in doc_id_to_idx:
                ranked.append((doc_id_to_idx[doc_id], float(item.get("retrieval_score", 0.0))))
        best = {}
        for idx, score in ranked:
            best[idx] = max(best.get(idx, float("-inf")), score)
        return sorted(best.items(), key=lambda x: -x[1])[:k]

    def rerank_search(query: str, k: int):
        if not retriever:
            return []
        results = retriever.retrieve(
            query,
            top_k=k,
            rerank=True,
            apply_decay=False,
            use_graph_boost=False,
        )
        ranked = []
        for item in results:
            doc_id = item.get("id", "")
            if doc_id in doc_id_to_idx:
                ranked.append((doc_id_to_idx[doc_id], float(item.get("retrieval_score", 0.0))))
        best = {}
        for idx, score in ranked:
            best[idx] = max(best.get(idx, float("-inf")), score)
        return sorted(best.items(), key=lambda x: -x[1])[:k]

    benches = [
        ("BM25", bm25_search),
        ("TF-IDF", tfidf_search),
        ("Vector only", vector_search),
    ]
    if use_current_pipeline and retriever is not None:
        benches.insert(0, ("Current pipeline", current_search))
        if use_reranker and getattr(retriever, "_reranker", None) is not None:
            benches.insert(1, ("Vector + reranker", rerank_search))

    for method_name, search_fn in benches:
        rows, metrics, latency = _evaluate_method(
            method_name, search_fn, query_items, doc_id_to_idx, top_k=top_k
        )
        methods[method_name] = {"metrics": metrics, "latency": latency}
        all_rows.extend(rows)

    summary = {
        "setup": {
            "data_path": data_path,
            "sample_size": len(sample_docs),
            "requested_sample_size": sample_size,
            "max_docs": max_docs,
            "query_count": len(query_items),
            "top_k": top_k,
            "query_field": query_field,
            "query_source": "query_file" if query_file else "self_retrieval",
            "query_file": query_file,
            "query_file_info": query_file_info,
            "seed": seed,
            "load_info": load_info,
            "current_pipeline_enabled": use_current_pipeline,
            "reranker_enabled": use_reranker,
            "reranker_loaded": bool(retriever and getattr(retriever, "_reranker", None)),
            "reranker_model_dir": reranker_model_dir,
            "model_name": model_name,
            "sklearn_available": _SKLEARN_AVAILABLE,
        },
        "methods": methods,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = out / f"retrieval_benchmark_{stamp}.json"
    md_path = out / f"retrieval_benchmark_{stamp}.md"
    csv_path = out / f"retrieval_benchmark_{stamp}.csv"
    _write_json(json_path, summary)
    md_path.write_text(_markdown_report(summary), encoding="utf-8")
    if write_details:
        _write_csv(csv_path, all_rows)

    return summary, json_path, md_path, csv_path


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Benchmark retrieval methods on a 1,000-document subset")
    parser.add_argument("--data", type=str, required=True, help="Path to the CSV/JSON corpus")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of documents to sample")
    parser.add_argument("--max-docs", type=int, default=None, help="Take the first N eligible documents instead of reservoir sampling")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K cutoff for evaluation")
    parser.add_argument("--query-field", type=str, default=DEFAULT_QUERY_FIELD, choices=["title", "content", "full_text"], help="Field used as the query text")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory where report files are saved")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for sampling")
    parser.add_argument("--no-current-pipeline", action="store_true", help="Skip the current KG+FAISS pipeline and benchmark only traditional methods")
    parser.add_argument(
        "--reranker-dir",
        type=str,
        default=None,
        help=(
            "Path hoặc Hugging Face model id của cross-encoder reranker. "
            f"Mặc định: local data/reranker_model nếu hợp lệ, ngược lại {DEFAULT_CROSS_ENCODER}"
        ),
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Tên hoặc path của sentence-transformer bi-encoder")
    parser.add_argument("--query-file", type=str, default=None, help="Optional JSONL/JSON/CSV file with query_text and relevant_doc_id")
    parser.add_argument("--no-reranker", action="store_true", help="Do not evaluate the reranker method")
    parser.add_argument("--no-details", action="store_true", help="Do not write per-query CSV details")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary, json_path, md_path, csv_path = run_benchmark(
        data_path=args.data,
        sample_size=args.sample_size,
        top_k=args.top_k,
        query_field=args.query_field,
        output_dir=args.output_dir,
        seed=args.seed,
        max_docs=args.max_docs,
        use_current_pipeline=not args.no_current_pipeline,
        use_reranker=not args.no_reranker,
        reranker_model_dir=args.reranker_dir,
        model_name=args.model_name,
        query_file=args.query_file,
        write_details=not args.no_details,
    )

    print("\nBenchmark completed.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if not args.no_details:
        print(f"Per-query CSV: {csv_path}")
    for method, data in summary["methods"].items():
        print(
            f"- {method}: "
            f"R@1={data['metrics']['recall@1']:.3f}, "
            f"R@10={data['metrics']['recall@10']:.3f}, "
            f"P@10={data['metrics']['precision@10']:.4f}, "
            f"MRR@10={data['metrics']['mrr@10']:.3f}, "
            f"avg_latency={data['latency']['avg_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
