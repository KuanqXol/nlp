"""
query_expansion.py
──────────────────────
Query expansion nâng cấp — thay string concatenation bằng multi-query strategy.

Vấn đề v1:
  Expansion = nối entity names vào 1 chuỗi:
  "nga ukraine putin zelensky nato biden eu liên hợp quốc"
  → SBERT encode 1 lần → vector bị dilute bởi nhiều entity không liên quan
  → Precision giảm, recall tăng không có kiểm soát

Cải tiến :
  1. Multi-query: tạo nhiều query variant từ entity groups
     Query gốc + mỗi nhóm entity → encode riêng → lấy max similarity
  2. PPR-guided expansion: dùng Personalized PageRank thay vì
     global importance để chọn neighbors (entity gần seeds theo context query)
  3. Relation-aware expansion: ưu tiên neighbor qua relation có nghĩa
     (leads, attacks, supports) hơn co-occurrence thuần túy
  4. Expansion trace với confidence score để debug
"""

from typing import Dict, List, Optional, Set, Tuple


# ── Cấu hình ─────────────────────────────────────────────────────────────────

# Relation types được ưu tiên trong expansion (thứ tự giảm dần)
HIGH_VALUE_RELATIONS = {
    "leads",
    "attacks",
    "supports",
    "sanctions",
    "signs_deal_with",
    "invests_in",
    "acquires",
    "warns_about",
    "found_in",
    "member_of",
}
# Relation types bị loại khỏi expansion path
EXCLUDE_RELATIONS = {"co_occurrence", "similar_to"}

MAX_SEED_ENTITIES = 5
MAX_HOP1_PER_SEED = 4
MAX_HOP2_PER_SEED = 3
MAX_TOTAL_ENTITIES = 12


class QueryExpander:
    """
    Query expander nâng cấp.

    Usage:
        expander = QueryExpander(kg, graph_ranker)

        # Tại query time (sau khi có seed entities từ NER)
        result = expander.expand(processed_query, use_ppr=True)

        # Lấy multi-query list để search nhiều lần
        queries = expander.get_multi_queries(result)
        for q in queries:
            hits = retriever.retrieve(q, top_k=5)
    """

    def __init__(
        self,
        kg,
        graph_ranker=None,
        importance_scores: Dict[str, float] = None,
        max_hop1: int = MAX_HOP1_PER_SEED,
        max_hop2: int = MAX_HOP2_PER_SEED,
    ):
        self.kg = kg
        # Backward compatibility: code cũ truyền importance_scores ở tham số thứ 2.
        if isinstance(graph_ranker, dict) and importance_scores is None:
            self.graph_ranker = None
            self.importance_scores = graph_ranker
        else:
            self.graph_ranker = graph_ranker
            self.importance_scores = importance_scores or {}
        self.max_hop1 = max_hop1
        self.max_hop2 = max_hop2

    # ─────────────────────────────────────────────────────────────────────
    # Main expand
    # ─────────────────────────────────────────────────────────────────────

    def expand(
        self,
        processed_query: Dict,
        hops: int = 2,
        use_ppr: bool = True,
        max_entities: int = MAX_TOTAL_ENTITIES,
    ) -> Dict:
        """
        Mở rộng query với multi-query strategy.

        Returns:
            {
                'seed_entities':     list[str],
                'hop1_entities':     list[str],
                'hop2_entities':     list[str],
                'all_entities':      list[str],
                'entity_scores':     {entity: score},  # PPR hoặc global
                'relation_paths':    {entity: relation_used},
                'expanded_query':    str,   # Backward compat: 1 query string
                'multi_queries':     list[str],  # NEW: nhiều query variants
                'expansion_trace':   dict,
            }
        """
        # Seed entities từ query
        seeds = [
            e.get("canonical", e.get("text", ""))
            for e in processed_query.get("entities", [])
        ]
        seeds = [s for s in seeds if s][:MAX_SEED_ENTITIES]

        if not seeds:
            seeds = processed_query.get("keywords", [])[:3]

        original_query = processed_query.get("normalized", "")
        keywords = processed_query.get("keywords", [])

        # Tính PPR score tại query time
        entity_scores: Dict[str, float] = {}
        if use_ppr and seeds and self.graph_ranker:
            try:
                entity_scores = self.graph_ranker.query_time_scores(
                    self.kg, seeds=seeds
                )
            except Exception:
                entity_scores = dict(self.importance_scores)
        else:
            entity_scores = dict(self.importance_scores)

        # Expand qua KG
        hop1_entities: Set[str] = set()
        hop2_entities: Set[str] = set()
        relation_paths: Dict[str, str] = {}  # entity → relation dùng để đến đó
        expansion_trace: Dict[str, Dict] = {}

        for seed in seeds:
            neighbors = self.kg.get_neighbors(
                seed,
                hops=hops,
                exclude_cooccur=True,
            )

            # Hop 1: rank theo PPR + relation quality
            h1 = self._rank_with_relation(
                seed,
                neighbors.get("hop1", []),
                entity_scores,
                n=self.max_hop1,
                relation_paths=relation_paths,
            )
            hop1_entities.update(h1)
            expansion_trace[seed] = {"hop1": h1, "hop2": []}

            # Hop 2
            if hops >= 2:
                h2 = self._rank_with_relation(
                    seed,
                    neighbors.get("hop2", []),
                    entity_scores,
                    n=self.max_hop2,
                    relation_paths=relation_paths,
                )
                h2 = [e for e in h2 if e not in hop1_entities and e not in seeds]
                hop2_entities.update(h2)
                expansion_trace[seed]["hop2"] = h2

        # Gộp và rank tổng
        all_entities = _merge_ranked(
            seeds, hop1_entities, hop2_entities, entity_scores, max_total=max_entities
        )

        # Build outputs
        expanded_query = self._build_expanded_query(
            original_query, keywords, all_entities
        )
        multi_queries = self._build_multi_queries(
            original_query, keywords, seeds, hop1_entities, hop2_entities
        )

        return {
            "seed_entities": seeds,
            "hop1_entities": list(hop1_entities),
            "hop2_entities": list(hop2_entities),
            "all_entities": all_entities,
            "entity_scores": {
                e: round(entity_scores.get(e, 0), 4) for e in all_entities
            },
            "relation_paths": relation_paths,
            "expanded_query": expanded_query,
            "multi_queries": multi_queries,
            "expansion_trace": expansion_trace,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Multi-query builder
    # ─────────────────────────────────────────────────────────────────────

    def _build_multi_queries(
        self,
        original: str,
        keywords: List[str],
        seeds: List[str],
        hop1: Set[str],
        hop2: Set[str],
    ) -> List[str]:
        """
        Tạo danh sách query variants:
        1. Query gốc (không mở rộng)
        2. Query + hop1 entities (related, high-confidence)
        3. Query + high-PPR hop2 (nếu có thêm context)

        Mỗi variant được encode độc lập → max similarity khi retrieve.
        Điều này giữ precision của query gốc trong khi tăng recall qua variants.
        """
        queries = []

        # Q1: Original
        if original:
            queries.append(original)

        # Q2: original + top hop1 (tối đa 3)
        top_h1 = list(hop1)[:3]
        if top_h1:
            q2 = f"{original} {' '.join(top_h1)}"
            queries.append(q2.strip())

        # Q3: original + top hop2 (tối đa 3)
        top_h2 = list(hop2)[:3]
        if top_h2:
            q3 = f"{original} {' '.join(top_h2)}"
            queries.append(q3.strip())

        # Dedupe (giữ thứ tự)
        seen = set()
        unique = []
        for q in queries:
            if q not in seen and q:
                seen.add(q)
                unique.append(q)

        return unique

    def _build_expanded_query(
        self,
        original: str,
        keywords: List[str],
        entities: List[str],
    ) -> str:
        """Backward compat: 1 chuỗi expanded."""
        existing = set(original.lower().split())
        parts = [original]
        for ent in entities:
            el = ent.lower()
            if el not in existing:
                parts.append(el)
                existing.add(el)
        return " ".join(parts)

    # ─────────────────────────────────────────────────────────────────────
    # Rank neighbors với relation quality
    # ─────────────────────────────────────────────────────────────────────

    def _rank_with_relation(
        self,
        seed: str,
        neighbors: List[str],
        entity_scores: Dict[str, float],
        n: int,
        relation_paths: Dict[str, str],
    ) -> List[str]:
        """
        Rank neighbors theo:
          score = PPR_score × relation_multiplier

        relation_multiplier:
          - Semantic relation (leads, attacks...) → 1.0
          - Unknown relation → 0.7
          - co_occurrence → 0.3 (nếu lọt qua, penalize)
        """
        scored = []
        for nb in neighbors:
            base_score = entity_scores.get(nb, 0.0)
            rel_mult = self._relation_multiplier(seed, nb)
            final = base_score * rel_mult
            scored.append((final, nb))
            # Lưu relation path
            if nb not in relation_paths:
                best_rel = self._get_best_relation(seed, nb)
                if best_rel:
                    relation_paths[nb] = best_rel

        scored.sort(reverse=True)
        return [nb for _, nb in scored[:n]]

    def _relation_multiplier(self, e1: str, e2: str) -> float:
        """Hệ số nhân dựa trên loại relation giữa 2 entity."""
        rels = self.kg.get_relations_between(e1, e2)
        if not rels:
            return 0.5
        for rel_info in rels:
            rel = (
                rel_info.get("relation", "")
                if isinstance(rel_info, dict)
                else str(rel_info)
            )
            if rel in HIGH_VALUE_RELATIONS:
                return 1.0
            if rel in EXCLUDE_RELATIONS:
                return 0.3
        return 0.7

    def _get_best_relation(self, e1: str, e2: str) -> Optional[str]:
        """Lấy relation có nghĩa nhất giữa 2 entity."""
        rels = self.kg.get_relations_between(e1, e2)
        for rel_info in rels:
            rel = (
                rel_info.get("relation", "")
                if isinstance(rel_info, dict)
                else str(rel_info)
            )
            if rel in HIGH_VALUE_RELATIONS:
                return rel
        return rels[0].get("relation") if rels and isinstance(rels[0], dict) else None

    # ─────────────────────────────────────────────────────────────────────
    # Explain
    # ─────────────────────────────────────────────────────────────────────

    def explain(self, result: Dict) -> str:
        lines = ["=== QUERY EXPANSION  ===\n"]
        lines.append(f"Seed entities: {', '.join(result['seed_entities'])}")
        lines.append("")

        for seed, trace in result["expansion_trace"].items():
            lines.append(f"[{seed}]")
            if trace["hop1"]:
                h1_with_scores = [
                    f"{e}({result['entity_scores'].get(e,0):.2f})"
                    for e in trace["hop1"]
                ]
                lines.append(f"  Hop-1 → {', '.join(h1_with_scores)}")
            if trace.get("hop2"):
                h2_with_scores = [
                    f"{e}({result['entity_scores'].get(e,0):.2f})"
                    for e in trace["hop2"]
                ]
                lines.append(f"  Hop-2 → {', '.join(h2_with_scores)}")

        lines.append("")
        lines.append(f"Multi-queries ({len(result['multi_queries'])}):")
        for i, q in enumerate(result["multi_queries"], 1):
            lines.append(f'  Q{i}: "{q}"')

        return "\n".join(lines)

    def get_multi_queries(self, result: Dict) -> List[str]:
        """Trả về danh sách query để retriever search nhiều lần."""
        return result.get("multi_queries", [result.get("expanded_query", "")])


# ── Multi-query retrieval helper ──────────────────────────────────────────────


def multi_query_retrieve(
    queries: List[str],
    retriever,
    top_k: int = 10,
    seed_entities: List[str] = None,
    dedup_by: str = "id",
) -> List[Dict]:
    """
    Search với nhiều query variant, merge và dedup kết quả.
    Score cuối = max score qua các query.

    Args:
        queries:  List query strings từ QueryExpander.get_multi_queries()
        retriever: Retriever instance
        top_k:    Số kết quả cuối
        seed_entities: Cho PPR boost
        dedup_by: Field để dedup ("id" cho doc-level)

    Returns:
        Merged + sorted results
    """
    seen: Dict[str, Dict] = {}
    fetch_k = max(top_k * 2, 20)

    for q in queries:
        hits = retriever.retrieve(
            q,
            top_k=fetch_k,
            seed_entities=seed_entities,
            rerank=False,  # Rerank chỉ chạy một lần sau khi merge
        )
        for hit in hits:
            key = hit.get(dedup_by, hit.get("chunk_id", ""))
            if key not in seen:
                seen[key] = hit
            else:
                # Giữ score cao nhất
                if hit["retrieval_score"] > seen[key]["retrieval_score"]:
                    seen[key] = hit

    merged = sorted(seen.values(), key=lambda x: -x["retrieval_score"])
    return merged[:top_k]


# ── Helper ────────────────────────────────────────────────────────────────────


def _merge_ranked(
    seeds: List[str],
    hop1: Set[str],
    hop2: Set[str],
    scores: Dict[str, float],
    max_total: int,
) -> List[str]:
    """Gộp seed + hop1 + hop2, sort theo score, giữ max_total."""
    seen: Set[str] = set()
    result: List[str] = []

    for group in [seeds, hop1, hop2]:
        ranked = sorted(group, key=lambda e: -scores.get(e, 0))
        for e in ranked:
            if e not in seen and len(result) < max_total:
                result.append(e)
                seen.add(e)

    return result


# ── Backward compat ───────────────────────────────────────────────────────────
QueryExpander = QueryExpander


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.graph.knowledge_graph import KnowledgeGraph
    from src.graph.ranking import GraphRanker

    kg = KnowledgeGraph()
    docs = [
        {
            "id": "d1",
            "date": "2024-01-15",
            "linked_entities": [
                {"canonical": c, "type": t, "link_score": 1.0}
                for c, t in [
                    ("Putin", "PER"),
                    ("Nga", "LOC"),
                    ("Ukraine", "LOC"),
                    ("Zelensky", "PER"),
                    ("NATO", "ORG"),
                    ("Biden", "PER"),
                    ("EU", "ORG"),
                    ("Donetsk", "LOC"),
                ]
            ],
            "triples": [
                {
                    "subject": "Putin",
                    "relation": "leads",
                    "object": "Nga",
                    "confidence": 0.90,
                },
                {
                    "subject": "Nga",
                    "relation": "attacks",
                    "object": "Ukraine",
                    "confidence": 0.92,
                },
                {
                    "subject": "Zelensky",
                    "relation": "leads",
                    "object": "Ukraine",
                    "confidence": 0.90,
                },
                {
                    "subject": "NATO",
                    "relation": "supports",
                    "object": "Ukraine",
                    "confidence": 0.85,
                },
                {
                    "subject": "Biden",
                    "relation": "supports",
                    "object": "Ukraine",
                    "confidence": 0.82,
                },
                {
                    "subject": "Nga",
                    "relation": "attacks",
                    "object": "Donetsk",
                    "confidence": 0.88,
                },
            ],
        }
    ]
    kg.build_from_documents(docs)

    ranker = GraphRanker()
    ranker.compute_global_scores(kg)

    expander = QueryExpander(kg, ranker)
    pq = {
        "normalized": "chiến tranh nga ukraine",
        "keywords": ["chiến tranh", "nga", "ukraine"],
        "entities": [
            {"canonical": "Nga", "type": "LOC"},
            {"canonical": "Ukraine", "type": "LOC"},
        ],
    }

    result = expander.expand(pq, hops=2, use_ppr=True)
    print(expander.explain(result))
