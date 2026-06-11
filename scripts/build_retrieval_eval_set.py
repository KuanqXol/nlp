"""Build a small, harder retrieval eval set from the local VnExpress corpus.

The generated benchmark is still weak-labeled: each query points to the source
article it was derived from. It is meant to be harder than exact title
self-retrieval, not a replacement for manually judged qrels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import create_document


DEFAULT_SOURCE = PROJECT_ROOT / "data" / "vnexpress_articles.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmarks"
DEFAULT_CORPUS_SIZE = 2000
DEFAULT_QUERY_COUNT = 200
DEFAULT_SEED = 42

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-/][0-9A-Za-zÀ-ỹĐđ]+)*")
ENTITY_RE = re.compile(
    r"\b(?:[A-ZÀ-ỸĐ][0-9A-Za-zÀ-ỹĐđ.-]*|[A-ZĐ]{2,}|Covid-19|COVID-19)"
    r"(?:\s+(?:[A-ZÀ-ỸĐ][0-9A-Za-zÀ-ỹĐđ.-]*|[A-ZĐ]{2,}|Covid-19|COVID-19)){0,3}"
)

STOPWORDS = {
    "a",
    "ai",
    "anh",
    "bị",
    "bởi",
    "các",
    "cái",
    "cần",
    "càng",
    "cho",
    "chưa",
    "chị",
    "có",
    "con",
    "còn",
    "của",
    "cùng",
    "đã",
    "đang",
    "đây",
    "để",
    "đến",
    "đều",
    "do",
    "đó",
    "được",
    "em",
    "gì",
    "hai",
    "hay",
    "hơn",
    "khi",
    "không",
    "là",
    "lại",
    "làm",
    "lên",
    "mà",
    "một",
    "mới",
    "này",
    "năm",
    "nên",
    "nếu",
    "ngày",
    "người",
    "nhất",
    "những",
    "nhiều",
    "ở",
    "phải",
    "qua",
    "ra",
    "rằng",
    "trả",
    "lời",
    "sau",
    "sao",
    "sẽ",
    "sự",
    "tại",
    "theo",
    "thì",
    "trên",
    "trước",
    "trong",
    "từ",
    "tuổi",
    "và",
    "vào",
    "về",
    "vì",
    "với",
}

QUERY_TYPES = (
    "keyword",
    "lead_keyword",
    "entity_topic",
    "category_year",
    "natural",
    "short_title",
)


def _stable_id(doc: Dict, ordinal: int) -> str:
    key = doc.get("url") or f"{doc.get('title', '')}|{doc.get('date', '')}|{ordinal}"
    return "vnx_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _tokens(text: str) -> List[str]:
    return TOKEN_RE.findall(text or "")


def _clean_query(text: str, max_tokens: int = 12) -> str:
    tokens = _tokens(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return " ".join(tokens).strip()


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    for part in parts:
        if len(part.strip()) >= 45:
            return part.strip()
    return (text or "").strip()[:260]


def _informative_terms(text: str, max_terms: int = 8) -> List[str]:
    terms: List[str] = []
    seen = set()
    for token in _tokens(text):
        lower = token.lower()
        if lower in STOPWORDS:
            continue
        if len(lower) < 3 and not lower.isdigit():
            continue
        if lower in seen:
            continue
        seen.add(lower)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _entities(text: str, max_items: int = 3) -> List[str]:
    items: List[str] = []
    seen = set()
    for match in ENTITY_RE.finditer(text or ""):
        ent = _clean_query(match.group(0), max_tokens=5)
        low = ent.lower()
        if not ent or low in STOPWORDS or low in seen:
            continue
        ent_tokens = ent.split()
        if len(ent_tokens) == 1 and not (ent.isupper() or re.search(r"\d|covid", ent, re.I)):
            continue
        if ent_tokens and ent_tokens[0].lower() in STOPWORDS:
            continue
        if low in {"theo", "ngày", "ông", "bà"}:
            continue
        seen.add(low)
        items.append(ent)
        if len(items) >= max_items:
            break
    return items


def _make_query(doc: Dict, query_type: str) -> Tuple[str, str]:
    title = doc.get("title", "")
    content = doc.get("content", "")
    lead = _first_sentence(content)
    category = (doc.get("category") or "").replace(".html", "").strip()
    year = (doc.get("date") or "")[:4] if (doc.get("date") or "")[:4].isdigit() else ""

    if query_type == "keyword":
        query = " ".join((_informative_terms(title, 5) + _informative_terms(lead, 4))[:8])
    elif query_type == "lead_keyword":
        query = " ".join((_informative_terms(title, 3) + _informative_terms(lead, 5))[:8])
    elif query_type == "entity_topic":
        ents = _entities(title, 2)
        if len(ents) < 3:
            ents.extend(e for e in _entities(lead, 3) if e.lower() not in {x.lower() for x in ents})
        query = " ".join((ents + _informative_terms(title, 4))[:8])
    elif query_type == "category_year":
        parts = [category, year, " ".join(_informative_terms(title, 5))]
        query = " ".join(p for p in parts if p)
    elif query_type == "natural":
        terms = " ".join((_entities(title, 2) + _informative_terms(title + " " + lead, 5))[:7])
        query = f"tin về {terms}" if terms else title
    elif query_type == "short_title":
        query = " ".join(_informative_terms(title, 7))
    else:
        query = title

    query = _clean_query(query)
    if len(query.split()) < 3 or _norm(query) == _norm(title):
        fallback = " ".join((_entities(title, 2) + _informative_terms(title + " " + lead, 6))[:8])
        query = _clean_query(fallback)
    if len(query.split()) < 3:
        query = _clean_query(title)
    return query, query_type


def _iter_documents(path: Path, max_source_rows: Optional[int] = None) -> Iterable[Dict]:
    seen_urls = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for ordinal, row in enumerate(reader, start=1):
            if max_source_rows is not None and max_source_rows > 0 and ordinal > max_source_rows:
                break
            doc = create_document(row)
            if not doc:
                continue
            url = doc.get("url") or ""
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            doc["id"] = _stable_id(doc, ordinal)
            yield doc


def _reservoir_sample(
    path: Path,
    sample_size: int,
    seed: int,
    max_source_rows: Optional[int] = None,
) -> Tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    selected: List[Dict] = []
    eligible_seen = 0
    for doc in _iter_documents(path, max_source_rows=max_source_rows):
        eligible_seen += 1
        if len(selected) < sample_size:
            selected.append(doc)
            continue
        replace_at = rng.randint(0, eligible_seen - 1)
        if replace_at < sample_size:
            selected[replace_at] = doc
    return selected, {
        "eligible_seen": eligible_seen,
        "selected": len(selected),
        "max_source_rows": max_source_rows,
    }


def _select_query_docs(docs: List[Dict], query_count: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_category: Dict[str, List[Dict]] = defaultdict(list)
    for doc in docs:
        by_category[doc.get("category") or "unknown"].append(doc)
    for group in by_category.values():
        rng.shuffle(group)

    categories = sorted(by_category, key=lambda c: len(by_category[c]), reverse=True)
    selected: List[Dict] = []
    used = set()
    while len(selected) < query_count and categories:
        progressed = False
        for category in list(categories):
            group = by_category[category]
            while group and group[-1]["id"] in used:
                group.pop()
            if not group:
                categories.remove(category)
                continue
            doc = group.pop()
            selected.append(doc)
            used.add(doc["id"])
            progressed = True
            if len(selected) >= query_count:
                break
        if not progressed:
            break
    return selected


def _write_corpus(path: Path, docs: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "url", "date", "category", "title", "text"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc in docs:
            writer.writerow(
                {
                    "id": doc.get("id", ""),
                    "url": doc.get("url", ""),
                    "date": doc.get("date", ""),
                    "category": doc.get("category", ""),
                    "title": doc.get("title", ""),
                    "text": doc.get("content", ""),
                }
            )


def _write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_review_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "query_text",
        "query_type",
        "relevant_doc_id",
        "relevant_title",
        "category",
        "date",
        "source_url",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_eval_set(
    source: Path,
    corpus_out: Path,
    queries_out: Path,
    query_count: int,
    corpus_size: int,
    seed: int,
    max_source_rows: Optional[int] = None,
) -> Dict:
    if query_count > corpus_size:
        raise ValueError("--query-count không được lớn hơn --corpus-size")

    corpus_docs, load_stats = _reservoir_sample(
        source,
        sample_size=corpus_size,
        seed=seed,
        max_source_rows=max_source_rows,
    )
    if len(corpus_docs) < query_count:
        raise RuntimeError(f"Chỉ lấy được {len(corpus_docs)} docs, không đủ {query_count} query")

    query_docs = _select_query_docs(corpus_docs, query_count=query_count, seed=seed + 17)
    rng = random.Random(seed + 29)
    rng.shuffle(query_docs)

    rows = []
    used_queries = set()
    for idx, doc in enumerate(query_docs):
        preferred_type = QUERY_TYPES[idx % len(QUERY_TYPES)]
        query, query_type = _make_query(doc, preferred_type)
        if _norm(query) in used_queries:
            for alt in QUERY_TYPES:
                query, query_type = _make_query(doc, alt)
                if _norm(query) not in used_queries:
                    break
        used_queries.add(_norm(query))
        rows.append(
            {
                "query_id": f"eval_{idx + 1:04d}",
                "query_text": query,
                "query_type": query_type,
                "relevant_doc_id": doc["id"],
                "relevant_title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "date": doc.get("date", ""),
                "source_url": doc.get("url", ""),
            }
        )

    _write_corpus(corpus_out, corpus_docs)
    _write_jsonl(queries_out, rows)
    review_csv = queries_out.with_suffix(".review.csv")
    _write_review_csv(review_csv, rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(source),
        "corpus_out": str(corpus_out),
        "queries_out": str(queries_out),
        "review_csv": str(review_csv),
        "seed": seed,
        "corpus_size": len(corpus_docs),
        "query_count": len(rows),
        "max_source_rows": max_source_rows,
        "load_stats": load_stats,
        "query_type_counts": dict(Counter(row["query_type"] for row in rows)),
        "category_counts": dict(Counter(row["category"] for row in rows).most_common()),
    }
    summary_path = queries_out.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Create a 200-query weak-labeled retrieval eval set from VnExpress data."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source VnExpress CSV")
    parser.add_argument(
        "--corpus-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "retrieval_eval_200_corpus.csv",
        help="Output CSV corpus used by benchmark_retrieval.py",
    )
    parser.add_argument(
        "--queries-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "retrieval_eval_200_queries.jsonl",
        help="Output JSONL query/qrels file",
    )
    parser.add_argument("--query-count", type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument("--corpus-size", type=int, default=DEFAULT_CORPUS_SIZE)
    parser.add_argument(
        "--max-source-rows",
        type=int,
        default=None,
        help="Optional cap for source CSV rows. Use 30000-50000 for a faster local refresh.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    summary = build_eval_set(
        source=args.source,
        corpus_out=args.corpus_out,
        queries_out=args.queries_out,
        query_count=args.query_count,
        corpus_size=args.corpus_size,
        seed=args.seed,
        max_source_rows=args.max_source_rows,
    )
    print("Eval set created.")
    print(f"Corpus: {summary['corpus_out']}")
    print(f"Queries: {summary['queries_out']}")
    print(f"Review CSV: {summary['review_csv']}")
    print(f"Summary: {summary['summary_path']}")
    print(f"Queries: {summary['query_count']} | Corpus docs: {summary['corpus_size']}")
    print(f"Query types: {summary['query_type_counts']}")


if __name__ == "__main__":
    main()
    "ông",
