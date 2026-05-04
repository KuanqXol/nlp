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

from src.data_loader import NewsDataLoader
from src.retrieval import EmbeddingManager, Retriever, chunk_documents


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


def _rank_of_target(results: List[Tuple[int, float]], target_idx: int) -> Optional[int]:
    for rank, (idx, _) in enumerate(results, start=1):
        if idx == target_idx:
            return rank
    return None


def _metrics_from_ranks(ranks: List[Optional[int]]) -> Dict[str, float]:
    total = len(ranks)
    valid = [r for r in ranks if r is not None]
    return {
        "queries": total,
        "coverage": len(valid) / total if total else 0.0,
        "top1": sum(1 for r in ranks if r == 1) / total if total else 0.0,
        "top5": sum(1 for r in ranks if r is not None and r <= 5) / total if total else 0.0,
        "top10": sum(1 for r in ranks if r is not None and r <= 10) / total if total else 0.0,
        "mrr": sum((1 / r) for r in valid) / total if total and valid else 0.0,
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
            )
        )

    metrics = _metrics_from_ranks(ranks)
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
    lines.append(f"- Query field: **{summary['setup']['query_field']}**")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    for key, value in summary["setup"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Method | Top-1 | Top-5 | Top-10 | MRR | Avg latency (ms) | P95 (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for method, data in summary["methods"].items():
        lines.append(
            f"| {method} | {_format_pct(data['metrics']['top1'])} | {_format_pct(data['metrics']['top5'])} | {_format_pct(data['metrics']['top10'])} | {data['metrics']['mrr']:.3f} | {data['latency']['avg_ms']:.2f} | {data['latency']['p95_ms']:.2f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Đây là benchmark **self-retrieval**: dùng title làm query và coi bài gốc là ground truth.")
    lines.append("- Chỉ số này phản ánh độ nhanh và khả năng kéo đúng bài tương ứng, không thay thế bộ đánh giá relevance gán nhãn thủ công.")
    lines.append("- Nếu muốn đánh giá sát thực tế hơn, nên tạo thêm 30-100 query thủ công có nhãn relevance.")
    return "\n".join(lines)


def build_backends(documents: List[Dict], use_current_pipeline: bool = True):
    texts = []
    for doc in documents:
        texts.append(doc.get("full_text") or f"{doc.get('title', '')}. {doc.get('content', '')}")

    bm25 = SimpleBM25(texts)
    tfidf = TfidfBackend(texts) if _SKLEARN_AVAILABLE else None

    em = EmbeddingManager()
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
        retriever = Retriever(use_cross_encoder=False)
        retriever.build(chunk_docs, em, doc_to_chunks, documents)

    return bm25, tfidf, em, retriever


def run_benchmark(
    data_path: str,
    sample_size: int,
    top_k: int,
    query_field: str,
    output_dir: str,
    seed: int,
    use_current_pipeline: bool = True,
    write_details: bool = True,
):
    loader = NewsDataLoader(data_path)
    docs = loader.load()
    sample_docs = _select_documents(docs, sample_size=sample_size, seed=seed)
    query_items = _build_query_items(sample_docs, query_field=query_field)
    query_items = query_items[:sample_size]

    if not query_items:
        raise RuntimeError("Không tạo được query items từ dữ liệu đã chọn.")

    sample_docs = [d for d in sample_docs if d.get("id") in {q["relevant_doc_id"] for q in query_items}]
    doc_id_to_idx = {doc["id"]: idx for idx, doc in enumerate(sample_docs)}

    bm25, tfidf, em, retriever = build_backends(sample_docs, use_current_pipeline=use_current_pipeline)

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

    benches = [
        ("BM25", bm25_search),
        ("TF-IDF", tfidf_search),
        ("Vector only", vector_search),
    ]
    if use_current_pipeline and retriever is not None:
        benches.insert(0, ("Current pipeline", current_search))

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
            "query_count": len(query_items),
            "top_k": top_k,
            "query_field": query_field,
            "seed": seed,
            "current_pipeline_enabled": use_current_pipeline,
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
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K cutoff for evaluation")
    parser.add_argument("--query-field", type=str, default=DEFAULT_QUERY_FIELD, choices=["title", "content", "full_text"], help="Field used as the query text")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory where report files are saved")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for sampling")
    parser.add_argument("--no-current-pipeline", action="store_true", help="Skip the current KG+FAISS pipeline and benchmark only traditional methods")
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
        use_current_pipeline=not args.no_current_pipeline,
        write_details=not args.no_details,
    )

    print("\nBenchmark completed.")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    if not args.no_details:
        print(f"Per-query CSV: {csv_path}")
    for method, data in summary["methods"].items():
        print(
            f"- {method}: top1={data['metrics']['top1']:.3f}, top10={data['metrics']['top10']:.3f}, avg_latency={data['latency']['avg_ms']:.2f} ms"
        )


if __name__ == "__main__":
    main()
