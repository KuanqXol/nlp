import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

try:
    import networkx as nx

    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False

# ── Cấu hình ─────────────────────────────────────────────────────────────────

MIN_TRIPLE_CONFIDENCE = 0.60  # Triple confidence thấp hơn → bỏ qua
MAX_COOCCUR_PER_DOC = 5  # Tối đa N cặp co-occurrence per document
COOCCUR_MIN_ENTITY_SCORE = 0.70  # Chỉ co-occur entity có link_score cao


class TemporalEdge:
    """Lưu một occurrence của relation tại thời điểm cụ thể."""

    __slots__ = ("doc_id", "date", "confidence", "relation", "sentence")

    def __init__(
        self,
        doc_id: str,
        date: str,
        confidence: float,
        relation: str,
        sentence: str = "",
    ):
        self.doc_id = doc_id
        self.date = date
        self.confidence = confidence
        self.relation = relation
        self.sentence = sentence


def _new_type_dict():
    return defaultdict(int)


class KnowledgeGraph:

    def __init__(self, min_confidence: float = MIN_TRIPLE_CONFIDENCE):
        if not _NX_AVAILABLE:
            raise ImportError("pip install networkx")
        self.graph = nx.MultiDiGraph()  # MultiDiGraph: nhiều edge giữa 2 node
        self.min_confidence = min_confidence
        self._pagerank: Dict[str, float] = {}
        self._type_votes: Dict[str, Dict[str, int]] = defaultdict(_new_type_dict)

    # ─────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────

    def build_from_documents(self, documents: List[Dict]):
        print(f"[KG] Build từ {len(documents)} bài (min_conf={self.min_confidence})...")

        total_triples = 0
        skipped_conf = 0
        cooccur_added = 0

        for doc in documents:
            doc_id = doc.get("id", "")
            doc_date = doc.get("date", "")

            # 1. Add entity nodes
            entities = doc.get("linked_entities", [])
            for ent in entities:
                canonical = ent.get("canonical", "")
                etype = ent.get("type", "MISC")
                ls = ent.get("link_score", 1.0)
                if canonical:
                    self._add_entity(canonical, etype, doc_id, ls)

            # 2. Add relation edges (confidence-filtered)
            triple_entities: Set[Tuple[str, str]] = set()
            for triple in doc.get("triples", []):
                subj = triple.get("subject", "")
                rel = triple.get("relation", "")
                obj = triple.get("object", "")
                conf = triple.get("confidence", 1.0)
                temporal = triple.get("temporal", doc_date)
                source_sentence = triple.get("sentence", "")

                if not subj or not obj or subj == obj:
                    continue

                if conf < self.min_confidence:
                    skipped_conf += 1
                    continue

                # Đảm bảo node tồn tại
                if subj not in self.graph:
                    self._add_entity(subj, "MISC", doc_id)
                if obj not in self.graph:
                    self._add_entity(obj, "MISC", doc_id)

                self._add_relation(
                    subj,
                    rel,
                    obj,
                    doc_id,
                    temporal,
                    conf,
                    source_sentence=source_sentence,
                )
                triple_entities.add((subj, obj))
                triple_entities.add((obj, subj))
                total_triples += 1

            # 3. Co-occurrence edges — chỉ những cặp CHƯA có relation
            #    và entity có link_score đủ cao, giới hạn MAX_COOCCUR_PER_DOC
            high_conf_entities = [
                e
                for e in entities
                if e.get("link_score", 0) >= COOCCUR_MIN_ENTITY_SCORE
                and e.get("canonical")
            ]
            cooccur_candidates = []
            for i in range(len(high_conf_entities)):
                for j in range(i + 1, len(high_conf_entities)):
                    e1 = high_conf_entities[i]["canonical"]
                    e2 = high_conf_entities[j]["canonical"]
                    if (e1, e2) not in triple_entities and e1 != e2:
                        # Score co-occurrence bằng tích link_score
                        pair_score = high_conf_entities[i].get(
                            "link_score", 1.0
                        ) * high_conf_entities[j].get("link_score", 1.0)
                        cooccur_candidates.append((pair_score, e1, e2))

            # Chỉ lấy top MAX_COOCCUR_PER_DOC cặp
            cooccur_candidates.sort(reverse=True)
            for _, e1, e2 in cooccur_candidates[:MAX_COOCCUR_PER_DOC]:
                self._add_relation(
                    e1,
                    "co_occurrence",
                    e2,
                    doc_id,
                    doc_date,
                    0.50,
                    source_sentence="",
                )
                cooccur_added += 1

        # Finalize type majority vote
        self._finalize_types()

        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        print(
            f"[KG] Done: {n_nodes} nodes, {n_edges} edges | "
            f"triples={total_triples}, skipped_low_conf={skipped_conf}, "
            f"cooccur_edges={cooccur_added}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Node / Edge helpers
    # ─────────────────────────────────────────────────────────────────────

    def _add_entity(
        self,
        entity: str,
        etype: str,
        doc_id: str,
        link_score: float = 1.0,
    ):
        if entity not in self.graph:
            self.graph.add_node(
                entity, type=etype, frequency=0, doc_ids=[], link_score_sum=0.0
            )
        node = self.graph.nodes[entity]
        node["frequency"] += 1
        node["link_score_sum"] = node.get("link_score_sum", 0) + link_score
        if doc_id and doc_id not in node["doc_ids"]:
            node["doc_ids"].append(doc_id)
        # Vote for type
        self._type_votes[entity][etype] += 1

    def _add_relation(
        self,
        subj: str,
        relation: str,
        obj: str,
        doc_id: str,
        date: str,
        confidence: float,
        source_sentence: str = "",
    ):
        """MultiDiGraph: một cặp (subj, obj) có thể có nhiều edge khác relation."""
        # Tìm edge cùng relation nếu đã tồn tại → cập nhật weight
        for key, data in self.graph.get_edge_data(subj, obj, default={}).items():
            if data.get("relation") == relation:
                data["weight"] += confidence
                data["occurrences"] += 1
                data["temporal_edges"].append(
                    TemporalEdge(doc_id, date, confidence, relation, source_sentence)
                )
                if source_sentence and source_sentence not in data.get(
                    "source_sentences", []
                ):
                    data.setdefault("source_sentences", []).append(source_sentence)
                data["max_confidence"] = max(data["max_confidence"], confidence)
                return

        # Edge mới
        self.graph.add_edge(
            subj,
            obj,
            relation=relation,
            weight=confidence,
            occurrences=1,
            max_confidence=confidence,
            temporal_edges=[
                TemporalEdge(doc_id, date, confidence, relation, source_sentence)
            ],
            source_sentences=[source_sentence] if source_sentence else [],
        )

    def add_relation(
        self,
        subj: str,
        relation: str,
        obj: str,
        weight: float = 1.0,
        doc_id: str = "",
        date: str = "",
        confidence: Optional[float] = None,
        source_sentence: str = "",
    ):
        if not subj or not obj or subj == obj:
            return

        if subj not in self.graph:
            self._add_entity(subj, "MISC", doc_id)
        if obj not in self.graph:
            self._add_entity(obj, "MISC", doc_id)

        conf = confidence if confidence is not None else weight
        self._add_relation(
            subj,
            relation,
            obj,
            doc_id,
            date,
            float(conf),
            source_sentence=source_sentence,
        )

    def _finalize_types(self):
        """Gán type theo majority vote cho mỗi node."""
        for entity, votes in self._type_votes.items():
            if entity in self.graph and votes:
                majority_type = max(votes, key=votes.get)
                self.graph.nodes[entity]["type"] = majority_type

    # ─────────────────────────────────────────────────────────────────────
    # Query API
    # ─────────────────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        entity: str,
        hops: int = 1,
        relation_filter: Optional[str] = None,
        exclude_cooccur: bool = True,
        min_edge_confidence: float = 0.0,
    ) -> Dict[str, List]:
        result = {"hop1": [], "hop2": []}
        if entity not in self.graph:
            return result

        def _valid_edge(u, v) -> bool:
            for _, data in self.graph.get_edge_data(u, v, default={}).items():
                if exclude_cooccur and data.get("relation") == "co_occurrence":
                    continue
                if data.get("max_confidence", 1.0) < min_edge_confidence:
                    continue
                if relation_filter and data.get("relation") != relation_filter:
                    continue
                return True
            return False

        hop1: Set[str] = set()
        for nb in list(self.graph.successors(entity)) + list(
            self.graph.predecessors(entity)
        ):
            if _valid_edge(entity, nb) or _valid_edge(nb, entity):
                hop1.add(nb)
        result["hop1"] = list(hop1)

        if hops >= 2:
            hop2: Set[str] = set()
            for h1 in hop1:
                for nb in list(self.graph.successors(h1)) + list(
                    self.graph.predecessors(h1)
                ):
                    if nb != entity and nb not in hop1:
                        if _valid_edge(h1, nb) or _valid_edge(nb, h1):
                            hop2.add(nb)
            result["hop2"] = list(hop2)

        return result

    def get_temporal_relations(
        self,
        entity1: str,
        entity2: str,
        after_date: str = None,
        before_date: str = None,
    ) -> List[Dict]:
        """
        Lấy relations giữa 2 entity lọc theo thời gian.

        Returns:
            [{"relation", "date", "confidence", "doc_id"}, ...]
        """
        results = []
        for u, v in [(entity1, entity2), (entity2, entity1)]:
            edges = self.graph.get_edge_data(u, v)
            if not edges:
                continue
            for _, data in edges.items():
                for te in data.get("temporal_edges", []):
                    if after_date and te.date and te.date < after_date:
                        continue
                    if before_date and te.date and te.date > before_date:
                        continue
                    results.append(
                        {
                            "relation": te.relation,
                            "date": te.date,
                            "confidence": te.confidence,
                            "sentence": te.sentence,
                            "doc_id": te.doc_id,
                            "direction": f"{u} → {v}",
                        }
                    )
        results.sort(key=lambda x: x.get("date", ""), reverse=True)
        return results

    def get_entity_info(self, entity: str) -> Optional[Dict]:
        if entity not in self.graph:
            return None
        data = dict(self.graph.nodes[entity])
        data["name"] = entity
        data["pagerank"] = self._pagerank.get(entity, 0.0)
        data["degree"] = self.graph.degree(entity)
        data["in_degree"] = self.graph.in_degree(entity)
        data["out_degree"] = self.graph.out_degree(entity)
        # avg link_score
        freq = data.get("frequency", 1)
        data["avg_link_score"] = data.get("link_score_sum", freq) / max(freq, 1)
        return data

    def get_relations_between(self, e1: str, e2: str) -> List[Dict]:
        """Tất cả relation giữa 2 entity (cả 2 chiều)."""
        rels = []
        for u, v in [(e1, e2), (e2, e1)]:
            edges = self.graph.get_edge_data(u, v)
            if not edges:
                continue
            for _, data in edges.items():
                rels.append(
                    {
                        "relation": data.get("relation"),
                        "direction": f"{u} → {v}",
                        "occurrences": data.get("occurrences", 1),
                        "max_confidence": data.get("max_confidence", 1.0),
                    }
                )
        return rels

    def filter_low_confidence_edges(self, min_confidence: float = 0.65):
        """
        Xóa các edge có max_confidence thấp sau khi build.
        Gọi trước khi chạy PageRank để KG sạch hơn.
        """
        to_remove = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get("max_confidence", 1.0) < min_confidence:
                if data.get("relation") != "co_occurrence":  # Giữ co-occur
                    to_remove.append((u, v, key))
        self.graph.remove_edges_from(to_remove)
        print(
            f"[KG] Removed {len(to_remove)} low-confidence edges (< {min_confidence})"
        )

    def get_top_entities(self, top_k: int = 20, entity_type: Optional[str] = None):
        entities = []
        for node, data in self.graph.nodes(data=True):
            if entity_type and data.get("type") != entity_type:
                continue
            entities.append((node, self._pagerank.get(node, 0.0)))
        return sorted(entities, key=lambda x: -x[1])[:top_k]

    def search_entities(self, query: str, top_k: int = 10):
        q = query.lower()
        return sorted(
            [
                (n, self._pagerank.get(n, 0.0))
                for n in self.graph.nodes()
                if q in n.lower()
            ],
            key=lambda x: -x[1],
        )[:top_k]

    def set_pagerank_scores(self, scores: Dict[str, float]):
        self._pagerank = scores

    def stats(self) -> Dict:
        type_counts: Dict[str, int] = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            type_counts[data.get("type", "?")] += 1

        rel_counts: Dict[str, int] = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel_counts[data.get("relation", "?")] += 1

        # Số unique relation types (không tính co-occurrence)
        semantic_rels = {k: v for k, v in rel_counts.items() if k != "co_occurrence"}

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "entity_types": dict(type_counts),
            "top_relations": dict(
                sorted(semantic_rels.items(), key=lambda x: -x[1])[:10]
            ),
            "cooccur_edges": rel_counts.get("co_occurrence", 0),
            "semantic_edges": sum(semantic_rels.values()),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"graph": self.graph, "pagerank": self._pagerank}, f)
        print(f"[KG] Saved → {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data["graph"]
        self._pagerank = data.get("pagerank", {})
        print(
            f"[KG] Loaded ← {path} "
            f"({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)"
        )


# ── Backward-compat alias ─────────────────────────────────────────────────────
KnowledgeGraph = KnowledgeGraph


if __name__ == "__main__":
    kg = KnowledgeGraph(min_confidence=0.60)
    docs = [
        {
            "id": "d1",
            "date": "2024-01-15",
            "linked_entities": [
                {"canonical": "Putin", "type": "PER", "link_score": 1.0},
                {"canonical": "Nga", "type": "LOC", "link_score": 1.0},
                {"canonical": "Ukraine", "type": "LOC", "link_score": 1.0},
                {"canonical": "NATO", "type": "ORG", "link_score": 1.0},
                {"canonical": "Zelensky", "type": "PER", "link_score": 1.0},
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
                    "subject": "Nga",
                    "relation": "sanctions",
                    "object": "EU",
                    "confidence": 0.40,
                },  # Low conf → bỏ
            ],
        },
    ]
    kg.build_from_documents(docs)
    print("\nStats:", kg.stats())
    print("\nNeighbors Ukraine (exclude cooccur):", kg.get_neighbors("Ukraine", hops=2))
    print("\nRelations Putin↔Nga:", kg.get_relations_between("Putin", "Nga"))
    print("\nTemporal Nga→Ukraine:", kg.get_temporal_relations("Nga", "Ukraine"))
