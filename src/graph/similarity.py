"""
Module: similarity_graph.py
Chức năng: Tạo edge similarity giữa các entity có embedding gần nhau.

Logic:
  - Tính cosine similarity giữa tất cả cặp entity
  - Nếu cosine_similarity > threshold → thêm edge "similar_to"

Ví dụ:
  "COVID-19" ←→ "SARS-CoV-2" (similarity > 0.8)
  "Nga"      ←→ "Liên bang Nga" (similarity > 0.85)
"""

from typing import Dict, List, Tuple
import numpy as np


class SimilarityGraphBuilder:
    """
    Thêm edge 'similar_to' vào Knowledge Graph
    dựa trên embedding similarity giữa các entity.

    Ví dụ:
        builder = SimilarityGraphBuilder(threshold=0.75)
        sim_edges = builder.build(kg, embedding_manager)
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def build(self, kg, embedding_manager) -> List[Tuple[str, str, float]]:
        """
        Tính similarity giữa tất cả cặp entity trong KG
        và thêm edge vào graph.

        Args:
            kg: KnowledgeGraph instance
            embedding_manager: EmbeddingManager instance

        Returns:
            Danh sách (entity1, entity2, similarity_score)
        """
        entity_names = list(kg.graph.nodes())
        if len(entity_names) < 2:
            return []

        print(f"[SimilarityGraph] Tính similarity cho {len(entity_names)} entities...")

        # Encode tất cả entity
        entity_embeddings = embedding_manager.encode_entities(entity_names)

        # Chuyển về matrix để tính batch cosine similarity
        valid_names = [n for n in entity_names if n in entity_embeddings]
        if len(valid_names) < 2:
            return []

        matrix = np.array([entity_embeddings[n] for n in valid_names])

        # Normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        matrix_norm = matrix / norms

        # Cosine similarity matrix (N × N)
        sim_matrix = matrix_norm @ matrix_norm.T

        # Tìm cặp có similarity > threshold
        sim_edges = []
        n = len(valid_names)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= self.threshold:
                    e1, e2 = valid_names[i], valid_names[j]
                    # Không thêm edge nếu đã có quan hệ rõ ràng hơn
                    if not kg.graph.has_edge(e1, e2):
                        kg.add_relation(e1, "similar_to", e2, confidence=sim)
                        sim_edges.append((e1, e2, round(sim, 4)))

        print(
            f"[SimilarityGraph] Thêm {len(sim_edges)} similarity edges (threshold={self.threshold})"
        )
        return sim_edges
