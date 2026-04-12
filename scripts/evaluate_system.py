"""
Đánh giá end-to-end toàn bộ pipeline Vietnamese KG-Enhanced News Search.

Tasks:
  ner        — Precision/Recall/F1 per-type (PER/LOC/ORG) + micro_avg
               Yêu cầu: data/ner_model/ đã train, data/ner_ground_truth.json
  retrieval  — Recall@K, MRR@K, NDCG@K, MAP, latency
               Dùng synthetic queries từ KG hoặc qrels file tự cung cấp
  graph      — PageRank distribution, Gini, PPR sanity check, top entities
  embedding  — Coverage, norm stats, intra/inter-query cosine similarity
  smoke      — 8 query cố định, kiểm tra field, score order, không crash
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / k if k else 0.0


def mrr_at_k(retrieved, relevant, k):
    rel = set(relevant)
    for rank, d in enumerate(retrieved[:k], 1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(r + 1) for r, d in enumerate(retrieved[:k], 1) if d in rel
    )
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def average_precision(retrieved, relevant):
    rel = set(relevant)
    hits = ap = 0.0
    for rank, d in enumerate(retrieved, 1):
        if d in rel:
            hits += 1
            ap += hits / rank
    return ap / len(relevant) if relevant else 0.0


def _avg(lst):
    return round(sum(lst) / len(lst), 4) if lst else 0.0


def _hdr(title: str):
    pad = 58 - len(title)
    print(f"\n╔{'═'*60}╗")
    print(f"║  {title}{' '*pad}║")
    print(f"╚{'═'*60}╝")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NER EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def _norm(text: str) -> str:
    import re, unicodedata

    t = unicodedata.normalize("NFKC", text or "").lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def _soft_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    return short in long and len(short.split()) >= 2


def _dedup(entities):
    seen, out = set(), []
    for e in entities:
        t = e.get("type", "")
        tx = e.get("text") or e.get("entity_text") or ""
        k = (t, _norm(tx))
        if t and tx and k not in seen:
            seen.add(k)
            out.append({"text": tx, "type": t})
    return out


def evaluate_ner(ner_engine, ground_truth_path: str, verbose: bool = False) -> Dict:
    gt_path = Path(ground_truth_path)
    if not gt_path.exists():
        return {"error": f"Không tìm thấy: {gt_path}"}

    with open(gt_path, encoding="utf-8") as f:
        raw = json.load(f)
    samples = raw.get("samples", raw) if isinstance(raw, dict) else raw

    TYPES = ("PER", "LOC", "ORG")
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}
    n = 0

    for sample in samples:
        sent = sample.get("sentence", "")
        if not sent:
            continue
        n += 1
        gold = _dedup([e for e in sample.get("entities", []) if e.get("type") in TYPES])
        pred = _dedup([e for e in ner_engine.extract(sent) if e.get("type") in TYPES])

        for etype in TYPES:
            g_e = [e for e in gold if e["type"] == etype]
            p_e = [e for e in pred if e["type"] == etype]
            matched, tp, fp = set(), 0, 0
            for pe in p_e:
                found = next(
                    (
                        i
                        for i, ge in enumerate(g_e)
                        if i not in matched and _soft_match(pe["text"], ge["text"])
                    ),
                    None,
                )
                if found is not None:
                    matched.add(found)
                    tp += 1
                else:
                    fp += 1
            counts[etype]["tp"] += tp
            counts[etype]["fp"] += fp
            counts[etype]["fn"] += len(g_e) - len(matched)

        if verbose:
            print(f"  [{n}] {sent[:60]}")
            print(
                f"    gold={[e['text'] for e in gold]}  pred={[e['text'] for e in pred]}"
            )

    results: Dict = {
        "n_samples": n,
        "ner_backend": getattr(ner_engine, "backend_name", "?"),
    }
    mtp = mfp = mfn = 0
    for etype in TYPES:
        tp, fp, fn = counts[etype]["tp"], counts[etype]["fp"], counts[etype]["fn"]
        p, r, f = _prf(tp, fp, fn)
        results[etype] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": tp + fn,
        }
        mtp += tp
        mfp += fp
        mfn += fn
    p, r, f = _prf(mtp, mfp, mfn)
    results["micro_avg"] = {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "tp": mtp,
        "fp": mfp,
        "fn": mfn,
        "support": mtp + mfn,
    }
    return results


def print_ner_results(res: Dict):
    _hdr("NER EVALUATION")
    if "error" in res:
        print(f"  ❌ {res['error']}")
        return
    print(f"  Backend: {res.get('ner_backend')}   Samples: {res.get('n_samples')}")
    h = f"\n  {'Type':<10}{'TP':>6}{'FP':>6}{'FN':>6}{'Prec':>9}{'Rec':>9}{'F1':>9}{'Support':>9}"
    print(h)
    print("  " + "─" * (len(h) - 3))
    for key in ("PER", "LOC", "ORG", "micro_avg"):
        if key not in res:
            continue
        r = res[key]
        print(
            f"  {key:<10}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}"
            f"{r['precision']:>9.3f}{r['recall']:>9.3f}{r['f1']:>9.3f}{r['support']:>9}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. RETRIEVAL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def _synthetic_qrels(system, n: int = 30) -> List[Dict]:
    scores = system.ranker.compute_importance_scores(system.kg)
    top = sorted(scores.items(), key=lambda x: -x[1])[: n * 2]
    qrels, seen = [], set()
    for entity, _ in top:
        if len(qrels) >= n or entity.lower() in seen:
            continue
        seen.add(entity.lower())
        relevant = [
            doc["id"]
            for doc in system._documents
            if any(e.get("canonical") == entity for e in doc.get("linked_entities", []))
        ]
        if relevant:
            qrels.append({"query": entity, "relevant_doc_ids": relevant})
    return qrels


def evaluate_retrieval(
    system, qrels: List[Dict], ks=(5, 10, 20), verbose: bool = False
) -> Dict:
    bucket = {
        k: {"recall": [], "precision": [], "mrr": [], "ndcg": [], "ap": []} for k in ks
    }
    latencies, errors = [], []

    for item in qrels:
        query = item.get("query", "")
        relevant = item.get("relevant_doc_ids", [])
        if not query or not relevant:
            continue
        try:
            t0 = time.time()
            docs, _ = system.search(query, top_k=max(ks))
            latencies.append((time.time() - t0) * 1000)
            ids = [d.get("id", "") for d in docs]
            for k in ks:
                bucket[k]["recall"].append(recall_at_k(ids, relevant, k))
                bucket[k]["precision"].append(precision_at_k(ids, relevant, k))
                bucket[k]["mrr"].append(mrr_at_k(ids, relevant, k))
                bucket[k]["ndcg"].append(ndcg_at_k(ids, relevant, k))
                bucket[k]["ap"].append(average_precision(ids, relevant))
            if verbose:
                print(
                    f"  {query!r}  ids={ids[:4]}  mrr@10={mrr_at_k(ids, relevant, 10):.3f}"
                )
        except Exception as e:
            errors.append({"query": query, "error": str(e)})

    out: Dict = {"n_queries": len(bucket[ks[0]]["recall"])}
    for k in ks:
        out[f"@{k}"] = {
            "Recall": _avg(bucket[k]["recall"]),
            "Precision": _avg(bucket[k]["precision"]),
            "MRR": _avg(bucket[k]["mrr"]),
            "NDCG": _avg(bucket[k]["ndcg"]),
            "MAP": _avg(bucket[k]["ap"]),
        }
    if latencies:
        sl = sorted(latencies)
        out["latency_ms"] = {
            "mean": round(sum(sl) / len(sl), 1),
            "median": round(sl[len(sl) // 2], 1),
            "p95": round(sl[int(len(sl) * 0.95)], 1),
            "max": round(sl[-1], 1),
        }
    if errors:
        out["errors"] = errors
    return out


def print_retrieval_results(res: Dict):
    _hdr("RETRIEVAL EVALUATION")
    if "error" in res:
        print(f"  ❌ {res['error']}")
        return
    print(f"  Queries: {res.get('n_queries', 0)}")
    ks = sorted(int(k[1:]) for k in res if k.startswith("@"))
    if ks:
        h = f"  {'Metric':<12}" + "".join(f"{'@'+str(k):>10}" for k in ks)
        print()
        print(h)
        print("  " + "─" * (len(h) - 2))
        for m in ("Recall", "Precision", "MRR", "NDCG", "MAP"):
            row = f"  {m:<12}" + "".join(
                f"{res.get(f'@{k}', {}).get(m, 0):>10.4f}" for k in ks
            )
            print(row)
    lat = res.get("latency_ms", {})
    if lat:
        print(
            f"\n  Latency  mean={lat['mean']}ms  median={lat['median']}ms  "
            f"p95={lat['p95']}ms  max={lat['max']}ms"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GRAPH EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def _gini(vals):
    if len(vals) <= 1:
        return 0.0
    s = sorted(vals)
    n = len(s)
    gini = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(s))
    total = sum(s)
    return gini / (n * total) if total else 0.0


def evaluate_graph(system) -> Dict:
    kg = system.kg
    ranker = system.ranker
    stats = kg.stats()
    n_ent = stats.get("n_entities", 0)
    n_rel = stats.get("n_relations", 0)
    if n_ent == 0:
        return {"error": "KG rỗng"}

    global_scores = ranker.compute_importance_scores(kg)
    vals = sorted(global_scores.values(), reverse=True)
    pr_dist = (
        {
            "top1": round(vals[0], 6),
            "top10_mean": round(sum(vals[:10]) / min(10, len(vals)), 6),
            "gini": round(_gini(vals), 4),
        }
        if vals
        else {}
    )

    # PPR sanity: 2 seed khác nhau → top-3 phải khác nhau
    ppr_check = {}
    top5 = sorted(global_scores.items(), key=lambda x: -x[1])[:5]
    if len(top5) >= 2:
        s1, s2 = top5[0][0], top5[1][0]
        ppr1 = ranker.personalized_pagerank(kg, seeds=[s1])
        ppr2 = ranker.personalized_pagerank(kg, seeds=[s2])
        t1 = [e for e, _ in sorted(ppr1.items(), key=lambda x: -x[1])[:3]]
        t2 = [e for e, _ in sorted(ppr2.items(), key=lambda x: -x[1])[:3]]
        ppr_check = {
            "seed_1": s1,
            "top3_ppr1": t1,
            "seed_2": s2,
            "top3_ppr2": t2,
            "diverges": t1 != t2,
        }

    type_counts: Dict[str, int] = defaultdict(int)
    for _, d in kg.graph.nodes(data=True):
        type_counts[d.get("type", "?")] += 1

    return {
        "n_entities": n_ent,
        "n_relations": n_rel,
        "entity_types": dict(type_counts),
        "pagerank_distribution": pr_dist,
        "ppr_sanity_check": ppr_check,
        "top_10": [{"entity": e, "score": round(s, 6)} for e, s in top5[:10]],
    }


def print_graph_results(res: Dict):
    _hdr("GRAPH & PAGERANK EVALUATION")
    if "error" in res:
        print(f"  ❌ {res['error']}")
        return
    print(f"  Entities : {res['n_entities']}")
    print(f"  Relations: {res['n_relations']}")
    ec = res.get("entity_types", {})
    if ec:
        print("  Types    : " + "  ".join(f"{t}={c}" for t, c in sorted(ec.items())))
    pr = res.get("pagerank_distribution", {})
    if pr:
        print(
            f"\n  PageRank  top1={pr['top1']}  top10_mean={pr['top10_mean']}  gini={pr['gini']}"
        )
        print(f"    gini=0 → flat  gini=1 → tập trung 1 node")
    ppr = res.get("ppr_sanity_check", {})
    if ppr:
        print(f"\n  PPR sanity:")
        print(f"    seed='{ppr['seed_1']}' → {ppr['top3_ppr1']}")
        print(f"    seed='{ppr['seed_2']}' → {ppr['top3_ppr2']}")
        print(f"    Cá nhân hóa đúng: {'✓' if ppr.get('diverges') else '✗ DEGENERATE'}")
    top10 = res.get("top_10", [])
    if top10:
        print(f"\n  Top entities:")
        for i, item in enumerate(top10, 1):
            print(f"    {i:2d}. {item['entity']:<30} {item['score']:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EMBEDDING EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_embedding(system) -> Dict:
    import numpy as np

    em = system.em
    if em.doc_embeddings is None:
        return {"error": "Chưa có embedding"}

    embs = em.doc_embeddings
    n, dim = embs.shape
    norms = np.linalg.norm(embs, axis=1)
    zero = norms < 1e-6
    coverage = round(1.0 - zero.mean(), 4)
    vn = norms[~zero]
    norm_stats = (
        {
            "mean": round(float(vn.mean()), 4),
            "std": round(float(vn.std()), 4),
            "min": round(float(vn.min()), 4),
            "max": round(float(vn.max()), 4),
        }
        if len(vn)
        else {}
    )

    test_queries = [
        "chiến tranh nga ukraine",
        "kinh tế việt nam tăng trưởng",
        "covid dịch bệnh sức khỏe",
        "samsung đầu tư hà nội",
        "bầu cử tổng thống mỹ",
    ]
    intra, inter = [], []
    try:

        def nv(v):
            nn = np.linalg.norm(v)
            return v / nn if nn > 1e-8 else v

        qvecs = [nv(em.encode_query(q)) for q in test_queries]
        embs_n = embs / np.where(norms[:, None] < 1e-8, 1.0, norms[:, None])
        for qv in qvecs:
            intra.append(float((embs_n @ qv).max()))
        for i in range(len(qvecs)):
            for j in range(i + 1, len(qvecs)):
                inter.append(float(np.dot(qvecs[i], qvecs[j])))
    except Exception as e:
        return {"error": str(e), "n_docs": n}

    return {
        "n_docs": n,
        "dim": dim,
        "coverage": coverage,
        "norm_stats": norm_stats,
        "intra_top1_sim": {
            "mean": _avg(intra),
            "min": round(min(intra), 4),
            "max": round(max(intra), 4),
        },
        "inter_query_sim": {"mean": _avg(inter)},
    }


def print_embedding_results(res: Dict):
    _hdr("EMBEDDING EVALUATION")
    if "error" in res:
        print(f"  ❌ {res['error']}")
        return
    print(
        f"  Docs: {res['n_docs']}   dim: {res['dim']}   coverage: {res['coverage']:.2%}"
    )
    ns = res.get("norm_stats", {})
    if ns:
        print(
            f"  Norms  mean={ns['mean']}  std={ns['std']}  min={ns['min']}  max={ns['max']}"
        )
    iq = res.get("intra_top1_sim", {})
    if iq:
        print(
            f"\n  Intra-query top-1 sim  mean={iq['mean']}  min={iq['min']}  max={iq['max']}"
        )
        print(f"    ↑ cao → query khớp đúng doc liên quan")
    iq2 = res.get("inter_query_sim", {})
    if iq2:
        print(f"  Inter-query sim  mean={iq2['mean']}")
        print(f"    ↑ thấp → các query đủ khác biệt trong embedding space")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════


def run_smoke_test(system, top_k: int = 5) -> Dict:
    QUERIES = [
        "chiến tranh nga ukraine",
        "kinh tế việt nam 2023",
        "samsung đầu tư hà nội",
        "WHO covid dịch bệnh",
        "bầu cử mỹ tổng thống",
        "giáo dục đại học việt nam",
        "ngân hàng nhà nước lãi suất",
        "biển đông trung quốc",
    ]
    REQUIRED = {"id", "title", "retrieval_score"}
    rows, latencies = [], []

    for query in QUERIES:
        row: Dict = {"query": query}
        try:
            t0 = time.time()
            docs, _ = system.search(query, top_k=top_k)
            ms = (time.time() - t0) * 1000
            latencies.append(ms)
            row["status"] = "ok"
            row["n"] = len(docs)
            row["ms"] = round(ms, 1)
            if docs:
                row["missing_fields"] = list(REQUIRED - set(docs[0].keys()))
                scores = [d.get("retrieval_score", 0) for d in docs]
                row["scores_valid"] = all(
                    isinstance(s, (int, float)) and not math.isnan(s) and s >= 0
                    for s in scores
                )
                row["scores_sorted"] = all(
                    scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
                )
                row["top1_score"] = round(scores[0], 4)
                row["top1_title"] = docs[0].get("title", "")[:55]
            else:
                row["missing_fields"] = []
                row["scores_valid"] = row["scores_sorted"] = True
        except Exception as e:
            row["status"] = "error"
            row["error"] = str(e)
        rows.append(row)

    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    return {
        "n_queries": len(QUERIES),
        "n_ok": n_ok,
        "pass_rate": round(n_ok / len(QUERIES), 4),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0,
            "max": round(max(latencies), 1) if latencies else 0,
        },
        "per_query": rows,
    }


def print_smoke_results(res: Dict):
    _hdr("PIPELINE SMOKE TEST")
    n_ok, n_total = res["n_ok"], res["n_queries"]
    icon = "✅" if n_ok == n_total else ("⚠️" if n_ok else "❌")
    lat = res.get("latency_ms", {})
    print(
        f"  {icon} {n_ok}/{n_total} passed   mean={lat.get('mean')}ms  max={lat.get('max')}ms\n"
    )
    for r in res.get("per_query", []):
        q = r["query"][:42]
        if r.get("status") == "ok":
            v = "✓" if r.get("scores_valid") else "✗SCORE"
            s = "✓" if r.get("scores_sorted") else "✗ORDER"
            miss = f" MISS:{r['missing_fields']}" if r.get("missing_fields") else ""
            print(
                f"  ✓ {q:<44} n={r['n']} {r['ms']:6.0f}ms top1={r.get('top1_score',0):.3f} {v} {s}{miss}"
            )
        else:
            print(f"  ✗ {q:<44} ERROR: {r.get('error','')[:45]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════


def print_summary(all_results: Dict):
    _hdr("SUMMARY")
    rows = []

    ner = all_results.get("ner", {})
    if ner and "error" not in ner:
        f1 = ner.get("micro_avg", {}).get("f1", 0)
        rows.append(("NER micro-F1", f"{f1:.4f}", "✓" if f1 >= 0.7 else "⚠"))

    ret = all_results.get("retrieval", {})
    if ret and "error" not in ret:
        at10 = ret.get("@10", {})
        for m, thr in [("Recall", 0.5), ("MRR", 0.4), ("NDCG", 0.45)]:
            v = at10.get(m, 0)
            rows.append((f"Retrieval {m}@10", f"{v:.4f}", "✓" if v >= thr else "⚠"))

    graph = all_results.get("graph", {})
    if graph and "error" not in graph:
        rows.append(
            (
                "KG entities",
                str(graph["n_entities"]),
                "✓" if graph["n_entities"] > 0 else "✗",
            )
        )
        rows.append(
            (
                "KG relations",
                str(graph["n_relations"]),
                "✓" if graph["n_relations"] > 0 else "✗",
            )
        )
        ok = graph.get("ppr_sanity_check", {}).get("diverges", False)
        rows.append(("PPR personalizes", "yes" if ok else "no", "✓" if ok else "⚠"))

    emb = all_results.get("embedding", {})
    if emb and "error" not in emb:
        cov = emb.get("coverage", 0)
        rows.append(("Embedding coverage", f"{cov:.2%}", "✓" if cov >= 0.99 else "⚠"))

    smoke = all_results.get("smoke", {})
    if smoke and "error" not in smoke:
        pr = smoke.get("pass_rate", 0)
        rows.append(
            (
                "Smoke pass rate",
                f"{pr:.0%}",
                "✓" if pr == 1.0 else ("⚠" if pr >= 0.8 else "✗"),
            )
        )

    if rows:
        w = max(len(r[0]) for r in rows) + 2
        for name, val, icon in rows:
            print(f"  {icon}  {name:<{w}} {val}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end evaluation — Vietnamese KG News Search"
    )
    p.add_argument("--load-index", action="store_true")
    p.add_argument("--data", "-d", type=str, default=None)
    p.add_argument("--index-dir", type=str, default=None)
    p.add_argument(
        "--ner-ground-truth",
        type=str,
        default=str(ROOT / "data" / "ner_ground_truth.json"),
    )
    p.add_argument(
        "--retrieval-qrels",
        type=str,
        default=None,
        help='JSON: [{"query":"...", "relevant_doc_ids":[...]}]',
    )
    p.add_argument("--tasks", type=str, default="ner,retrieval,graph,embedding,smoke")
    p.add_argument("--top-k", "-k", type=int, default=10)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tasks = {t.strip().lower() for t in args.tasks.split(",")}

    print("\n" + "═" * 62)
    print("   Vietnamese KG News Search — End-to-End Evaluation")
    print("═" * 62)

    need_system = bool(tasks & {"retrieval", "graph", "embedding", "smoke"})
    system = None

    if need_system:
        from main import NewsSearchSystem

        system = NewsSearchSystem(data_path=args.data, index_dir=args.index_dir)
        if args.load_index:
            print("\n📂 Loading index...")
            try:
                system.load_index(args.index_dir)
            except Exception as e:
                print(f"❌ Không load được index: {e}")
                sys.exit(1)
        else:
            data = args.data or str(ROOT / "data" / "vnexpress_articles.csv")
            if not Path(data).exists():
                print(f"❌ Không tìm thấy data: {data}")
                sys.exit(1)
            system.data_path = data
            print("\n🔧 Building pipeline...")
            system.build()
            system.save_index(args.index_dir)

    all_results: Dict = {}

    if "ner" in tasks:
        print("\n⏳ NER evaluation...")
        # Cần data/ner_model/ đã train (python scripts/train_ner.py)
        ner_engine = (
            system.ner
            if system is not None
            else __import__(
                "src.preprocessing.ner", fromlist=["VietnameseNER"]
            ).VietnameseNER()
        )
        res = evaluate_ner(ner_engine, args.ner_ground_truth, verbose=args.verbose)
        all_results["ner"] = res
        print_ner_results(res)

    if "retrieval" in tasks and system is not None:
        print("\n⏳ Retrieval evaluation...")
        qrels = []
        if args.retrieval_qrels and Path(args.retrieval_qrels).exists():
            with open(args.retrieval_qrels, encoding="utf-8") as f:
                qrels = json.load(f)
            print(f"   {len(qrels)} qrels từ {args.retrieval_qrels}")
        else:
            print("   Generating synthetic queries từ KG top entities...")
            qrels = _synthetic_qrels(system, n=30)
            print(f"   {len(qrels)} queries")
        res = evaluate_retrieval(system, qrels, ks=(5, 10, 20), verbose=args.verbose)
        all_results["retrieval"] = res
        print_retrieval_results(res)

    if "graph" in tasks and system is not None:
        print("\n⏳ Graph evaluation...")
        res = evaluate_graph(system)
        all_results["graph"] = res
        print_graph_results(res)

    if "embedding" in tasks and system is not None:
        print("\n⏳ Embedding evaluation...")
        res = evaluate_embedding(system)
        all_results["embedding"] = res
        print_embedding_results(res)

    if "smoke" in tasks and system is not None:
        print("\n⏳ Smoke test...")
        res = run_smoke_test(system, top_k=args.top_k)
        all_results["smoke"] = res
        print_smoke_results(res)

    if all_results:
        print_summary(all_results)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Lưu: {out}")


if __name__ == "__main__":
    main()
