"""Self-retrieval benchmark for dense Vietnamese bi-encoder retrieval.

Evaluation logic:
- each document title is one query
- the only positive document is the source article itself
- retrieval is scored at document level, even when chunk indexing is enabled

Designed to stay lightweight:
- no KG / NER / query expansion / reranking
- chunk-level indexing by default to match the current retriever
- ``--no-chunks`` for dense document-level comparison
- defaults to 1000 documents for fast BM25-vs-dense comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import NewsDataLoader, create_document
from src.retrieval import EmbeddingManager, Retriever, chunk_documents


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks"
DEFAULT_TOP_K = 10
DEFAULT_SEED = 42
DEFAULT_MAX_DOCS = 1000
DEFAULT_USE_CHUNKS = True
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "data" / "bi_encoder_model"
DEFAULT_MODEL_NAME = (
    str(DEFAULT_LOCAL_MODEL)
    if (DEFAULT_LOCAL_MODEL / "modules.json").exists()
    else "bkai-foundation-models/vietnamese-bi-encoder"
)


@dataclass
class QueryRow:
    query_id: str
    query_text: str
    relevant_doc_id: str
    relevant_title: str
    rank: Optional[int]
    top1_hit: bool
    top5_hit: bool
    top10_hit: bool
    latency_ms: float
    returned_doc_ids: str


def _normalize_query(text: str) -> str:
    return " ".join((text or "").split())


def _maybe_limit_documents(
    docs: List[Dict], sample_size: Optional[int], max_docs: Optional[int], seed: int
) -> Tuple[List[Dict], str]:
    eligible = [d for d in docs if d.get("id") and d.get("title") and d.get("content")]
    if max_docs is not None and max_docs > 0:
        return eligible[:max_docs], "head"
    if sample_size is not None and sample_size > 0 and len(eligible) > sample_size:
        rng = random.Random(seed)
        return rng.sample(eligible, sample_size), "sample"
    return eligible, "all"


def _load_csv_subset(
    data_path: Path,
    sample_size: Optional[int],
    max_docs: Optional[int],
    seed: int,
) -> Tuple[List[Dict], Dict[str, int | str]]:
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

            if sample_size is None or sample_size <= 0:
                selected.append(doc)
                continue

            if len(selected) < sample_size:
                selected.append(doc)
                continue

            replace_at = rng.randint(0, eligible_seen - 1)
            if replace_at < sample_size:
                selected[replace_at] = doc

    return selected, {
        "total_rows_seen": total_rows,
        "eligible_seen": eligible_seen,
        "loaded_docs": len(selected),
        "skipped_invalid": skipped_invalid,
        "skipped_dedup": skipped_dedup,
        "selection_mode": "head" if max_docs is not None and max_docs > 0 else "sample",
    }


def load_documents(
    data_path: str,
    sample_size: Optional[int],
    max_docs: Optional[int],
    seed: int,
) -> Tuple[List[Dict], Dict[str, int | str | Dict]]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")

    if path.suffix.lower() == ".csv" and ((sample_size and sample_size > 0) or (max_docs and max_docs > 0)):
        docs, stats = _load_csv_subset(path, sample_size=sample_size, max_docs=max_docs, seed=seed)
        return docs, {"loader": "stream_csv_subset", "load_stats": stats}

    loader = NewsDataLoader(str(path))
    docs = loader.load()
    limited, mode = _maybe_limit_documents(
        docs, sample_size=sample_size, max_docs=max_docs, seed=seed
    )
    return limited, {
        "loader": "NewsDataLoader",
        "selection_mode": mode,
        "load_stats": dict(loader.last_load_stats),
    }


def build_queries(documents: List[Dict]) -> List[Dict]:
    items = []
    for idx, doc in enumerate(documents):
        query_text = _normalize_query(doc.get("title", ""))
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


def latency_stats(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    arr = sorted(latencies)

    def _pct(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        idx = (len(arr) - 1) * p
        low = int(idx)
        high = min(low + 1, len(arr) - 1)
        weight = idx - low
        return arr[low] * (1 - weight) + arr[high] * weight

    return {
        "avg_ms": round(statistics.mean(arr), 4),
        "p50_ms": round(_pct(0.50), 4),
        "p95_ms": round(_pct(0.95), 4),
        "p99_ms": round(_pct(0.99), 4),
    }


def build_metrics(ranks: List[Optional[int]], top_k: int) -> Dict[str, float | int | None]:
    total = len(ranks)
    valid = [r for r in ranks if r is not None]
    denom = total or 1
    hits_at_10 = sum(1 for r in ranks if r is not None and r <= 10)

    return {
        "query_count": total,
        "coverage": round(len(valid) / denom, 6),
        "recall@1": round(sum(1 for r in ranks if r == 1) / denom, 6),
        "recall@5": round(sum(1 for r in ranks if r is not None and r <= 5) / denom, 6),
        "recall@10": round(hits_at_10 / denom, 6),
        # Single-positive self-retrieval: mỗi query đóng góp tối đa 1 relevant doc vào top-10.
        "precision@10": round(hits_at_10 / (denom * 10), 6),
        "recall@top_k": round(
            sum(1 for r in ranks if r is not None and r <= top_k) / denom, 6
        ),
        "mrr@10": round(
            sum((1.0 / r) for r in ranks if r is not None and r <= 10) / denom, 6
        ),
        "mean_rank": round(statistics.mean(valid), 4) if valid else None,
        "median_rank": round(statistics.median(valid), 4) if valid else None,
    }


def write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[QueryRow]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(QueryRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_markdown(path: Path, summary: Dict):
    lines = [
        "# Dense self-retrieval report",
        "",
        f"- Method: **{summary['method']}**",
        f"- Documents: **{summary['corpus']['document_count']}**",
        f"- Queries: **{summary['metrics']['query_count']}**",
        f"- Use chunks: **{summary['setup']['use_chunks']}**",
        f"- Index field: **{summary['setup']['index_field']}**",
        "",
        "## Metrics",
        "",
        f"- Recall@1: **{summary['metrics']['recall@1']:.4f}**",
        f"- Recall@5: **{summary['metrics']['recall@5']:.4f}**",
        f"- Recall@10: **{summary['metrics']['recall@10']:.4f}**",
        f"- Precision@10: **{summary['metrics']['precision@10']:.4f}**",
        f"- MRR@10: **{summary['metrics']['mrr@10']:.4f}**",
        "",
        "## Cost",
        "",
        f"- Build index (s): **{summary['timing']['build_index_s']:.3f}**",
        f"- Avg latency/query (ms): **{summary['latency']['avg_ms']:.3f}**",
        f"- Embedding matrix (MB): **{summary['corpus']['embedding_matrix_mb']:.3f}**",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_index_texts(documents: List[Dict], index_field: str) -> List[Dict]:
    items = []
    for doc in documents:
        if index_field == "content":
            text = doc.get("content", "")
        else:
            text = doc.get("full_text") or f"{doc.get('title', '')}. {doc.get('content', '')}"
        if not text:
            continue
        items.append({"id": doc["id"], "full_text": text})
    return items


def build_dense_backend(
    documents: List[Dict],
    model_name: str,
    batch_size: Optional[int],
    use_chunks: bool,
    index_field: str,
) -> Tuple[EmbeddingManager, Retriever, Dict[str, int | float | str]]:
    em = EmbeddingManager(model_name=model_name)
    retriever = Retriever(use_cross_encoder=False)

    build_t0 = time.perf_counter()
    if use_chunks:
        prepend_title = index_field == "full_text"
        chunks, doc_to_chunks = chunk_documents(
            documents,
            strategy="sentence_window",
            max_chars=400,
            overlap=1,
            prepend_title=prepend_title,
            log_every=2000,
        )
        chunk_dicts = [{"id": c["chunk_id"], "full_text": c["chunk_text"]} for c in chunks]
        em.build_document_index(chunk_dicts, batch_size=batch_size)
        retriever.build(chunks, em, doc_to_chunks, documents)
        chunk_count = len(chunks)
    else:
        docs_for_index = build_index_texts(documents, index_field=index_field)
        em.build_document_index(docs_for_index, batch_size=batch_size)
        retriever.build_simple(documents, em)
        chunk_count = 0
    build_index_s = time.perf_counter() - build_t0

    embeddings = em.doc_embeddings
    embedding_bytes = int(embeddings.nbytes) if embeddings is not None else 0

    return em, retriever, {
        "document_count": len(documents),
        "chunk_count": chunk_count,
        "index_unit_count": len(em.doc_ids),
        "vector_count": len(em.doc_ids),
        "embedding_dim": int(embeddings.shape[1]) if embeddings is not None else 0,
        "embedding_matrix_bytes": embedding_bytes,
        "embedding_matrix_mb": round(embedding_bytes / (1024 * 1024), 6),
        "memory_estimate_bytes": embedding_bytes,
        "memory_estimate_mb": round(embedding_bytes / (1024 * 1024), 6),
        "build_index_s": round(build_index_s, 6),
        "model_name": model_name,
    }


def evaluate_dense(
    retriever: Retriever,
    queries: List[Dict],
    top_k: int,
) -> Tuple[List[QueryRow], Dict[str, float | int | None], Dict[str, float]]:
    rows: List[QueryRow] = []
    ranks: List[Optional[int]] = []
    latencies: List[float] = []
    search_k = max(top_k, 10)

    for item in queries:
        t0 = time.perf_counter()
        results = retriever.retrieve(
            item["query_text"],
            top_k=search_k,
            rerank=False,
            apply_decay=False,
            use_graph_boost=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        returned_ids = [doc.get("id", "") for doc in results]
        rank = None
        for idx, doc_id in enumerate(returned_ids, start=1):
            if doc_id == item["relevant_doc_id"]:
                rank = idx
                break
        ranks.append(rank)

        rows.append(
            QueryRow(
                query_id=item["query_id"],
                query_text=item["query_text"],
                relevant_doc_id=item["relevant_doc_id"],
                relevant_title=item["relevant_title"],
                rank=rank,
                top1_hit=rank == 1,
                top5_hit=rank is not None and rank <= 5,
                top10_hit=rank is not None and rank <= 10,
                latency_ms=round(latency_ms, 4),
                returned_doc_ids="|".join(returned_ids[:top_k]),
            )
        )

    return rows, build_metrics(ranks, top_k=top_k), latency_stats(latencies)


def run(
    data_path: str,
    sample_size: Optional[int],
    max_docs: Optional[int],
    top_k: int,
    seed: int,
    output_dir: str,
    model_name: str,
    batch_size: Optional[int],
    use_chunks: bool,
    index_field: str,
) -> Tuple[Dict, Path, Path, Path]:
    total_t0 = time.perf_counter()

    load_t0 = time.perf_counter()
    documents, load_info = load_documents(
        data_path=data_path,
        sample_size=sample_size,
        max_docs=max_docs,
        seed=seed,
    )
    load_data_s = time.perf_counter() - load_t0

    queries = build_queries(documents)
    if not documents or not queries:
        raise RuntimeError("Không tạo được documents/queries hợp lệ để benchmark.")

    em, retriever, corpus_stats = build_dense_backend(
        documents=documents,
        model_name=model_name,
        batch_size=batch_size,
        use_chunks=use_chunks,
        index_field=index_field,
    )

    eval_t0 = time.perf_counter()
    rows, metrics, latency = evaluate_dense(retriever=retriever, queries=queries, top_k=top_k)
    evaluation_s = time.perf_counter() - eval_t0
    total_s = time.perf_counter() - total_t0

    summary = {
        "method": "dense_bi_encoder",
        "setup": {
            "data_path": data_path,
            "sample_size": sample_size,
            "max_docs": max_docs,
            "actual_docs": len(documents),
            "query_count": len(queries),
            "top_k": top_k,
            "seed": seed,
            "use_chunks": use_chunks,
            "index_field": index_field,
            "batch_size": batch_size,
            "model_name": model_name,
            "load_info": load_info,
        },
        "corpus": corpus_stats,
        "metrics": metrics,
        "latency": latency,
        "timing": {
            "load_data_s": round(load_data_s, 6),
            "build_index_s": corpus_stats["build_index_s"],
            "evaluation_s": round(evaluation_s, 6),
            "total_s": round(total_s, 6),
        },
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    json_path = out_dir / f"dense_self_retrieval_{stamp}.json"
    csv_path = out_dir / f"dense_self_retrieval_{stamp}.csv"
    md_path = out_dir / f"dense_self_retrieval_{stamp}.md"
    write_json(json_path, summary)
    write_csv(csv_path, rows)
    write_markdown(md_path, summary)

    _ = em  # giữ biến để code path rõ ràng và tránh bị tối ưu nhầm trong refactor
    return summary, json_path, csv_path, md_path


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Dense self-retrieval benchmark: each title is one query and its article is the only positive."
    )
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn CSV/JSON corpus")
    parser.add_argument("--sample-size", type=int, default=None, help="Reservoir sample size for CSV or random sample size after load")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="Chỉ lấy N document đầu để chạy nhanh và giảm RAM")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K cutoff để đánh giá")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Thư mục lưu JSON/CSV/MD report")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Tên hoặc path của sentence-transformer bi-encoder")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size khi encode documents/chunks")
    parser.add_argument("--use-chunks", dest="use_chunks", action="store_true", help="Bật chunk indexing và dedupe về document level")
    parser.add_argument("--no-chunks", dest="use_chunks", action="store_false", help="Tắt chunking, chỉ index theo document")
    parser.add_argument("--index-field", type=str, default="full_text", choices=["content", "full_text"], help="Field dùng để index ở document mode; chunk mode sẽ prepend title nếu chọn full_text")
    parser.set_defaults(use_chunks=DEFAULT_USE_CHUNKS)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary, json_path, csv_path, md_path = run(
        data_path=args.data,
        sample_size=args.sample_size,
        max_docs=args.max_docs,
        top_k=args.top_k,
        seed=args.seed,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        use_chunks=args.use_chunks,
        index_field=args.index_field,
    )

    print("\nDense self-retrieval benchmark completed.")
    print(f"JSON summary : {json_path}")
    print(f"Per-query CSV: {csv_path}")
    print(f"Markdown     : {md_path}")
    print(
        "Metrics      : "
        f"R@1={summary['metrics']['recall@1']:.4f}, "
        f"R@5={summary['metrics']['recall@5']:.4f}, "
        f"R@10={summary['metrics']['recall@10']:.4f}, "
        f"P@10={summary['metrics']['precision@10']:.4f}, "
        f"MRR@10={summary['metrics']['mrr@10']:.4f}"
    )
    print(
        "Cost         : "
        f"build={summary['timing']['build_index_s']:.3f}s, "
        f"avg_query={summary['latency']['avg_ms']:.3f}ms, "
        f"embed_mem={summary['corpus']['embedding_matrix_mb']:.3f}MB"
    )


if __name__ == "__main__":
    main()
