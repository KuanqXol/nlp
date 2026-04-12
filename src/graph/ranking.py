from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx

    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False


# ── Cấu hình ─────────────────────────────────────────────────────────────────

DEFAULT_DAMPING = 0.85
PPR_ALPHA = 0.85  # Damping cho PPR
PPR_MAX_ITER = 50  # Ít iteration hơn global (query time cần nhanh)
SCORE_W_PAGERANK = 0.50
SCORE_W_PPR = 0.30  # Chỉ có ý nghĩa khi có seed entities
SCORE_W_FREQUENCY = 0.15
SCORE_W_QUALITY = 0.05  # avg link_score của entity


class GraphRanker:
    """
    Graph ranker nâng cấp với Personalized PageRank.

    Usage:
        ranker = GraphRanker()

        # Offline: tính global scores
        global_scores = ranker.compute_global_scores(kg)

        # Query time: tính PPR với seed entities
        ppr_scores = ranker.personalized_pagerank(kg, seeds=["Ukraine", "NATO"])

        # Combine
        final_scores = ranker.combine_scores(global_scores, ppr_scores)
        top_entities = ranker.get_top_k(final_scores, k=10)
    """

    def __init__(
        self,
        damping: float = DEFAULT_DAMPING,
        max_iter: int = 100,
        tol: float = 1e-6,
        exclude_cooccur: bool = True,
    ):
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.exclude_cooccur = exclude_cooccur

        # Cache global scores để không tính lại
        self._global_pagerank: Dict[str, float] = {}
        self._importance_scores: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Global PageRank (offline)
    # ─────────────────────────────────────────────────────────────────────

    def _get_semantic_graph(self, kg):
        """
        Lấy subgraph chỉ chứa semantic edges (không có co-occurrence).
        Co-occurrence edges nhiễu và làm PageRank không phản ánh đúng
        tầm quan trọng ngữ nghĩa.
        """
        if not self.exclude_cooccur:
            return kg.graph

        # Tạo view chỉ các edge không phải co_occurrence
        # MultiDiGraph: dùng subgraph_view
        def edge_filter(u, v, key):
            data = kg.graph.get_edge_data(u, v, key) or {}
            return data.get("relation") != "co_occurrence"

        try:
            return nx.subgraph_view(
                kg.graph,
                filter_edge=edge_filter,
            )
        except Exception:
            return kg.graph

    def compute_pagerank(self, kg) -> Dict[str, float]:
        """Global PageRank trên semantic edges."""
        G = self._get_semantic_graph(kg)
        if G.number_of_nodes() == 0:
            return {}

        # Dùng edge weight = occurrences × max_confidence để weighted PR
        try:
            # nx.pagerank trên MultiDiGraph dùng weight mặc định
            raw = nx.pagerank(
                G,
                alpha=self.damping,
                max_iter=self.max_iter,
                tol=self.tol,
                weight="weight",
            )
        except (nx.PowerIterationFailedConvergence, Exception):
            print("[GraphRanker] PageRank không hội tụ → degree fallback")
            raw = {n: G.in_degree(n, weight="weight") for n in G.nodes()}

        # Normalize [0, 1]
        scores = _normalize(raw)
        self._global_pagerank = scores
        kg.set_pagerank_scores(scores)
        print(f"[GraphRanker] Global PageRank: {len(scores)} entities")
        return scores

    def compute_global_scores(self, kg) -> Dict[str, float]:
        """
        Tính combined global importance score.
        Dùng để rank entity khi không có query context.
        """
        pr = self.compute_pagerank(kg)
        freq = _get_frequency_scores(kg)
        qual = _get_quality_scores(kg)

        scores = {}
        for entity in kg.graph.nodes():
            scores[entity] = (
                SCORE_W_PAGERANK * pr.get(entity, 0.0)
                + SCORE_W_FREQUENCY * freq.get(entity, 0.0)
                + SCORE_W_QUALITY * qual.get(entity, 0.0)
            )

        scores = _normalize(scores)
        self._importance_scores = scores
        return scores

    def compute_importance_scores(self, kg) -> Dict[str, float]:
        """
        Backward-compatible alias cho code cũ.
        """
        return self.compute_global_scores(kg)

    # ─────────────────────────────────────────────────────────────────────
    # Personalized PageRank (query time)
    # ─────────────────────────────────────────────────────────────────────

    def personalized_pagerank(
        self,
        kg,
        seeds: List[str],
        alpha: float = PPR_ALPHA,
        max_iter: int = PPR_MAX_ITER,
        uniform_weight: float = 0.1,
    ) -> Dict[str, float]:
        """
        Personalized PageRank từ seed entities.

        PPR bắt đầu với personalization vector tập trung vào seeds.
        Kết quả: entity nào "gần" seeds trong đồ thị sẽ có score cao hơn
        so với global PageRank.

        Args:
            seeds: Danh sách entity canonical names từ query.
            alpha: Damping. Nhỏ hơn → PPR "lan tỏa" ra xa hơn.
            uniform_weight: Trọng số uniform trong personalization
                            (tránh rank = 0 cho node không liên kết với seed).

        Returns:
            {entity: ppr_score} normalized [0,1]
        """
        G = self._get_semantic_graph(kg)
        if G.number_of_nodes() == 0 or not seeds:
            return {}

        # Lọc seeds thực sự tồn tại trong graph
        valid_seeds = [s for s in seeds if s in G]
        if not valid_seeds:
            # Nếu không seed nào trong graph → PPR = global PR
            return dict(self._global_pagerank)

        # Personalization vector:
        # - valid_seeds nhận trọng số cao
        # - tất cả node nhận một lượng nhỏ (uniform_weight) để tránh sink nodes
        n_nodes = G.number_of_nodes()
        all_nodes = list(G.nodes())

        personalization = {}
        seed_weight = (1.0 - uniform_weight) / len(valid_seeds)
        uniform_each = uniform_weight / n_nodes

        for node in all_nodes:
            personalization[node] = uniform_each
        for seed in valid_seeds:
            personalization[seed] += seed_weight

        # Normalize personalization
        total = sum(personalization.values())
        personalization = {k: v / total for k, v in personalization.items()}

        try:
            ppr = nx.pagerank(
                G,
                alpha=alpha,
                personalization=personalization,
                max_iter=max_iter,
                tol=1e-4,  # Looser tolerance → nhanh hơn
                weight="weight",
            )
        except (nx.PowerIterationFailedConvergence, Exception):
            ppr = personalization  # Fallback: dùng personalization vector

        return _normalize(ppr)

    def query_time_scores(
        self,
        kg,
        seeds: List[str],
        ppr_weight: float = 0.40,
        global_weight: float = 0.60,
    ) -> Dict[str, float]:
        """
        Combine global scores + PPR cho query cụ thể.

        Args:
            ppr_weight:    Trọng số PPR score (cao → kết quả gần seeds hơn)
            global_weight: Trọng số global importance

        Khi seeds rỗng: trả về global scores (không tính PPR).
        """
        if not seeds or not self._global_pagerank:
            return dict(self._importance_scores)

        ppr = self.personalized_pagerank(kg, seeds)
        global_s = self._importance_scores

        combined = {}
        for entity in kg.graph.nodes():
            combined[entity] = global_weight * global_s.get(
                entity, 0.0
            ) + ppr_weight * ppr.get(entity, 0.0)

        return _normalize(combined)

    # ─────────────────────────────────────────────────────────────────────
    # Top-K
    # ─────────────────────────────────────────────────────────────────────

    def get_top_k(
        self,
        scores: Dict[str, float],
        k: int = 20,
        entity_type: Optional[str] = None,
        kg=None,
        exclude_types: Set[str] = None,
    ) -> List[Tuple[str, float]]:
        filtered = dict(scores)

        if entity_type and kg:
            filtered = {
                e: s
                for e, s in filtered.items()
                if kg.graph.nodes.get(e, {}).get("type") == entity_type
            }
        if exclude_types and kg:
            filtered = {
                e: s
                for e, s in filtered.items()
                if kg.graph.nodes.get(e, {}).get("type") not in exclude_types
            }

        return sorted(filtered.items(), key=lambda x: -x[1])[:k]

    def report(self, scores: Dict[str, float], kg, top_k: int = 10) -> str:
        lines = ["=== ENTITY RANKING REPORT ===\n"]
        for etype in ["PER", "LOC", "ORG", "MISC"]:
            top = self.get_top_k(scores, k=top_k, entity_type=etype, kg=kg)
            lines.append(f"--- Top {etype} ---")
            for rank, (ent, s) in enumerate(top, 1):
                lines.append(f"  {rank:2d}. {ent:25s} {s:.4f}")
            lines.append("")
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    m = max(scores.values())
    if m == 0:
        return scores
    return {k: round(v / m, 6) for k, v in scores.items()}


def _get_frequency_scores(kg) -> Dict[str, float]:
    freq = {n: d.get("frequency", 0) for n, d in kg.graph.nodes(data=True)}
    return _normalize(freq)


def _get_quality_scores(kg) -> Dict[str, float]:
    """avg_link_score phản ánh độ tin cậy của entity linking."""
    qual = {}
    for n, d in kg.graph.nodes(data=True):
        freq = max(d.get("frequency", 1), 1)
        qual[n] = d.get("link_score_sum", freq) / freq
    return _normalize(qual)


# ── Backward compat ───────────────────────────────────────────────────────────
GraphRanker = GraphRanker


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.graph.knowledge_graph import KnowledgeGraph

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
                    ("WHO", "ORG"),
                    ("Việt Nam", "LOC"),
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
                    "subject": "NATO",
                    "relation": "supports",
                    "object": "Ukraine",
                    "confidence": 0.85,
                },
                {
                    "subject": "Zelensky",
                    "relation": "leads",
                    "object": "Ukraine",
                    "confidence": 0.90,
                },
            ],
        }
    ]
    kg.build_from_documents(docs)

    ranker = GraphRanker()
    global_s = ranker.compute_global_scores(kg)

    print("=== GLOBAL SCORES ===")
    for ent, s in sorted(global_s.items(), key=lambda x: -x[1]):
        print(f"  {ent:20s} {s:.4f}")

    print("\n=== PPR từ seed=['Ukraine', 'NATO'] ===")
    query_s = ranker.query_time_scores(kg, seeds=["Ukraine", "NATO"])
    for ent, s in sorted(query_s.items(), key=lambda x: -x[1]):
        diff = s - global_s.get(ent, 0)
        print(f"  {ent:20s} {s:.4f}  (delta={diff:+.4f})")
