"""
retriever.py
───────────────
Retriever nâng cấp — 3 cải tiến chính:

  1. Chunk-aware retrieval: tìm chunk liên quan nhất → dedupe về doc level
     (thay vì encode toàn bộ document thành 1 vector)

  2. Graph-aware hybrid scoring:
     Thay vì: final = 0.8*vec + 0.2*max(entity_importance)
     Dùng:    final = vec_score * (1 + graph_boost)
     graph_boost = PPR score của document dựa trên query seed entities
     → Document liên quan đến entity quan trọng với QUERY CỤ THỂ được boost.

  3. Cross-encoder reranking (optional):
     Sau khi retrieve top-K candidates bằng bi-encoder,
     cross-encoder re-score top-K theo pair (query, chunk_text).
     Chính xác hơn nhiều nhưng chậm hơn → chỉ chạy trên top candidates.
     Model: cross-encoder/ms-marco-MiniLM-L-6- (multilingual fallback)
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder

    _CROSS_ENCODER_AVAILABLE = True
except ImportError:
    _CROSS_ENCODER_AVAILABLE = False


# ── Cấu hình ─────────────────────────────────────────────────────────────────

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
GRAPH_BOOST_ALPHA = 0.30  # graph_boost weight trong hybrid score
VECTOR_ALPHA = 0.70  # vector score weight
MAX_CHUNKS_PER_DOC = 2  # Tối đa N chunk/doc trước khi dedupe


# ── Bi-encoder backends (giữ nguyên từ v1) ───────────────────────────────────


class _NumpyBackend:
    def __init__(self):
        self._matrix = None
        self._ids: List[str] = []

    def build(self, embeddings: np.ndarray, ids: List[str]):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._matrix = (embeddings / np.where(norms == 0, 1e-8, norms)).astype(
            np.float32
        )
        self._ids = ids

    def search(self, qvec: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        if self._matrix is None:
            return [], []
        q = qvec.astype(np.float32)
        q = q / (np.linalg.norm(q) or 1e-8)
        scores = self._matrix @ q
        k = min(k, len(self._ids))
        idx = np.argsort(-scores)[:k]
        return [self._ids[i] for i in idx], [float(scores[i]) for i in idx]


class _FaissBackend:
    def __init__(self):
        self._index = None
        self._ids: List[str] = []
        self._dim = 0

    def build(self, embeddings: np.ndarray, ids: List[str]):
        n, d = embeddings.shape
        self._dim = d
        self._ids = ids
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = (embeddings / np.where(norms == 0, 1e-8, norms)).astype(np.float32)
        self._index = faiss.IndexFlatIP(d)
        self._index.add(normed)

    def search(self, qvec: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        if self._index is None:
            return [], []
        q = qvec.astype(np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) or 1e-8)
        k = min(k, len(self._ids))
        dists, idxs = self._index.search(q, k)
        ids = [self._ids[i] for i in idxs[0] if i >= 0]
        scores = [float(d) for d in dists[0][: len(ids)]]
        return ids, scores


# ── Cross-encoder reranker ────────────────────────────────────────────────────


class CrossEncoderReranker:
    """
    Reranker dùng cross-encoder model.

    Cross-encoder encode cặp (query, passage) → relevance score.
    Chính xác hơn bi-encoder nhưng O(n) inference → chỉ dùng trên top candidates.

    Nếu không có model → trả về candidates không đổi thứ tự.
    """

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL):
        self._model = None
        self._available = False
        if _CROSS_ENCODER_AVAILABLE:
            try:
                print(f"[CrossEncoder] Tải model: {model_name}")
                self._model = CrossEncoder(model_name)
                self._available = True
                print("[CrossEncoder] Sẵn sàng.")
            except Exception as e:
                print(f"[CrossEncoder] Lỗi tải model: {e} → reranking bị tắt")
        else:
            print("[CrossEncoder] sentence-transformers chưa cài → reranking tắt")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        text_field: str = "chunk_text",
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Re-score và sắp xếp lại candidates.

        Args:
            query:      Query string
            candidates: List document/chunk dicts đã có retrieval_score
            text_field: Field chứa text để score với cross-encoder
            top_k:      Trả về top K sau rerank (None = trả về tất cả)

        Returns:
            Candidates được sắp xếp lại theo cross-encoder score.
        """
        if not self._available or not candidates:
            return candidates[:top_k] if top_k else candidates

        pairs = [
            (query, c.get(text_field, c.get("full_text", ""))[:512]) for c in candidates
        ]

        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as e:
            print(f"[CrossEncoder] Lỗi predict: {e}")
            return candidates[:top_k] if top_k else candidates

        for doc, score in zip(candidates, scores):
            doc["cross_encoder_score"] = round(float(score), 4)
            # Blend: giữ vector score nhưng boost theo cross-encoder
            vec = doc.get("retrieval_score", 0.5)
            # Normalize cross-encoder score (logistic)
            import math

            ce_norm = 1 / (1 + math.exp(-float(score) / 3))
            doc["retrieval_score"] = round(0.4 * vec + 0.6 * ce_norm, 4)

        candidates.sort(key=lambda x: -x["retrieval_score"])
        return candidates[:top_k] if top_k else candidates

    @property
    def available(self):
        return self._available


# ── Main Retriever  ─────────────────────────────────────────────────────────


class Retriever:
    """
    Retriever nâng cấp với:
    - Chunk-aware indexing
    - Graph-aware hybrid scoring (PPR-based)
    - Optional cross-encoder reranking

    Usage (production):
        ret = Retriever(use_cross_encoder=True)
        ret.build(chunks, embedding_manager, doc_to_chunks,
                  graph_ranker, kg, documents)
        results = ret.retrieve(query, seed_entities=["Ukraine", "NATO"], top_k=10)

    Usage (backward compat — giống v1):
        ret = Retriever()
        ret.build_simple(documents, embedding_manager, importance_scores)
        results = ret.retrieve(query, top_k=10)
    """

    def __init__(
        self,
        use_faiss: bool = True,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
    ):
        self._backend = (
            _FaissBackend() if (use_faiss and _FAISS_AVAILABLE) else _NumpyBackend()
        )
        print(
            f"[Retriever] Backend: {'FAISS' if isinstance(self._backend, _FaissBackend) else 'NumPy'}"
        )

        self._reranker = (
            CrossEncoderReranker(cross_encoder_model) if use_cross_encoder else None
        )
        self._em = None
        self._documents: Dict[str, Dict] = {}  # doc_id → full document
        self._chunks: Dict[str, Dict] = {}  # chunk_id → chunk
        self._doc_to_chunks: Dict[str, List[str]] = {}
        self._chunk_mode = False

        # Graph scoring
        self._graph_ranker = None
        self._kg = None
        self._global_scores: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────────────────────
    # Build (chunk mode — recommended)
    # ─────────────────────────────────────────────────────────────────────

    def build(
        self,
        chunks: List[Dict],
        embedding_manager,
        doc_to_chunks: Dict[str, List[str]],
        documents: List[Dict],
        graph_ranker=None,
        kg=None,
        importance_scores: Dict[str, float] = None,
    ):
        """
        Build chunk-aware index.

        Args:
            chunks:            Output của chunk_documents()
            embedding_manager: EmbeddingManager đã encode chunks
            doc_to_chunks:     {doc_id: [chunk_id, ...]}
            documents:         Danh sách document gốc (để trả về kết quả đầy đủ)
            graph_ranker:      GraphRanker instance (cho PPR)
            kg:                KnowledgeGraph instance (cho PPR)
            importance_scores: Global importance (fallback khi không có PPR)
        """
        self._em = embedding_manager
        self._doc_to_chunks = doc_to_chunks
        self._documents = {d["id"]: d for d in documents}
        self._chunks = {c["chunk_id"]: c for c in chunks}
        self._graph_ranker = graph_ranker
        self._kg = kg
        self._global_scores = importance_scores or {}
        self._chunk_mode = True

        embeddings = embedding_manager.doc_embeddings
        chunk_ids = embedding_manager.doc_ids  # doc_ids thực ra là chunk_ids

        if embeddings is None or not chunk_ids:
            print("[Retriever] WARNING: embeddings chưa build!")
            return

        self._backend.build(embeddings, chunk_ids)
        print(
            f"[Retriever] Chunk index: {len(chunk_ids)} chunks, "
            f"{len(self._documents)} docs"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Build (backward compat — document mode)
    # ─────────────────────────────────────────────────────────────────────

    def build_simple(
        self,
        documents: List[Dict],
        embedding_manager,
        importance_scores: Dict[str, float] = None,
    ):
        """Backward-compat: không chunking, index theo document."""
        self._em = embedding_manager
        self._documents = {d["id"]: d for d in documents}
        self._global_scores = importance_scores or {}
        self._chunk_mode = False

        embeddings = embedding_manager.doc_embeddings
        doc_ids = embedding_manager.doc_ids

        if embeddings is None:
            return

        self._backend.build(embeddings, doc_ids)
        print(f"[Retriever] Document index: {len(doc_ids)} docs")

    # ─────────────────────────────────────────────────────────────────────
    # Retrieve
    # ─────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        seed_entities: List[str] = None,
        rerank: bool = True,
        fetch_multiplier: int = 3,
    ) -> List[Dict]:
        """
        Retrieve top-k documents.

        Args:
            query:           Query string (expanded hoặc original)
            top_k:           Số document trả về
            seed_entities:   Entity từ query để tính PPR boost
            rerank:          Có dùng cross-encoder rerank không
            fetch_multiplier: Lấy K×multiplier candidates trước khi rerank/dedupe

        Returns:
            List document dicts với các fields:
            retrieval_score, vector_score, graph_boost, cross_encoder_score (nếu có)
        """
        if self._em is None:
            return []

        query_vec = self._em.encode_query(query)
        fetch_k = min(top_k * fetch_multiplier, max(len(self._documents), 1))
        ids, vec_scores = self._backend.search(query_vec, k=fetch_k)

        if not ids:
            return []

        # Tính PPR scores một lần nếu có seed entities
        ppr_scores: Dict[str, float] = {}
        if seed_entities and self._graph_ranker and self._kg:
            ppr_scores = self._graph_ranker.query_time_scores(
                self._kg, seeds=seed_entities
            )

        # Build candidate list
        candidates = []
        if self._chunk_mode:
            candidates = self._build_candidates_from_chunks(
                ids, vec_scores, ppr_scores, top_k
            )
        else:
            candidates = self._build_candidates_from_docs(ids, vec_scores, ppr_scores)

        candidates = self._rerank(query, candidates, top_k=top_k, rerank=rerank)

        candidates.sort(key=lambda x: -x["retrieval_score"])
        return candidates[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        seed_entities: List[str] = None,
        rerank: bool = True,
        fetch_multiplier: int = 3,
    ) -> List[Dict]:
        return self.search(
            query,
            top_k=top_k,
            seed_entities=seed_entities,
            rerank=rerank,
            fetch_multiplier=fetch_multiplier,
        )

    def _rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: int,
        rerank: bool = True,
    ) -> List[Dict]:
        if not rerank or not candidates:
            return candidates[:top_k]
        if not self._reranker:
            return candidates[:top_k]

        text_field = "chunk_text" if self._chunk_mode else "full_text"
        return self._reranker.rerank(
            query,
            candidates,
            text_field=text_field,
            top_k=top_k,
        )

    def _build_candidates_from_chunks(
        self,
        chunk_ids: List[str],
        vec_scores: List[float],
        ppr_scores: Dict[str, float],
        top_k: int,
    ) -> List[Dict]:
        """
        Chunk-level → dedupe về document level.
        Giữ chunk có score cao nhất per document.
        """
        # chunk_id → best (vec_score, chunk)
        best_per_doc: Dict[str, Tuple[float, Dict]] = {}

        for cid, vscore in zip(chunk_ids, vec_scores):
            chunk = self._chunks.get(cid)
            if not chunk:
                continue
            doc_id = chunk.get("doc_id", "")
            if doc_id not in best_per_doc or vscore > best_per_doc[doc_id][0]:
                best_per_doc[doc_id] = (vscore, chunk)

        candidates = []
        for doc_id, (vscore, chunk) in best_per_doc.items():
            doc = self._documents.get(doc_id, {})
            graph_boost = self._compute_graph_boost(doc, ppr_scores)
            final_score = VECTOR_ALPHA * vscore + GRAPH_BOOST_ALPHA * graph_boost

            result = dict(doc)
            result.update(
                {
                    "retrieval_score": round(final_score, 4),
                    "vector_score": round(vscore, 4),
                    "graph_boost": round(graph_boost, 4),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                }
            )
            candidates.append(result)

        return candidates

    def _build_candidates_from_docs(
        self,
        doc_ids: List[str],
        vec_scores: List[float],
        ppr_scores: Dict[str, float],
    ) -> List[Dict]:
        candidates = []
        for doc_id, vscore in zip(doc_ids, vec_scores):
            doc = self._documents.get(doc_id)
            if not doc:
                continue
            graph_boost = self._compute_graph_boost(doc, ppr_scores)
            final_score = VECTOR_ALPHA * vscore + GRAPH_BOOST_ALPHA * graph_boost

            result = dict(doc)
            result.update(
                {
                    "retrieval_score": round(final_score, 4),
                    "vector_score": round(vscore, 4),
                    "graph_boost": round(graph_boost, 4),
                }
            )
            candidates.append(result)
        return candidates

    def _compute_graph_boost(
        self,
        doc: Dict,
        ppr_scores: Dict[str, float],
    ) -> float:
        """
        Graph boost = trung bình PPR scores của entity trong document.

        Thay vì max() (v1), dùng mean của top-3 entity scores.
        → Tránh boost quá cao cho doc chỉ có 1 entity quan trọng
          không liên quan đến query.
        """
        if not ppr_scores:
            # Fallback: global importance
            entities = doc.get("linked_entities", [])
            scores = [
                self._global_scores.get(e.get("canonical", ""), 0.0) for e in entities
            ]
            if not scores:
                return 0.0
            top3 = sorted(scores, reverse=True)[:3]
            return sum(top3) / len(top3)

        entities = doc.get("linked_entities", [])
        scores = [ppr_scores.get(e.get("canonical", ""), 0.0) for e in entities]
        if not scores:
            return 0.0
        top3 = sorted(scores, reverse=True)[:3]
        return sum(top3) / len(top3)

    # ─────────────────────────────────────────────────────────────────────
    # Retrieve with expansion result (API compatibility)
    # ─────────────────────────────────────────────────────────────────────

    def retrieve_with_expansion(
        self,
        expansion_result: Dict,
        top_k: int = 10,
        rerank: bool = True,
    ) -> List[Dict]:
        expanded_query = expansion_result.get("expanded_query", "")
        seed_entities = expansion_result.get("seed_entities", [])
        return self.retrieve(
            expanded_query,
            top_k=top_k,
            seed_entities=seed_entities,
            rerank=rerank,
        )

    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self._documents.get(doc_id)


# ── Backward compat ───────────────────────────────────────────────────────────
Retriever = Retriever


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from src.retrieval.embedding import EmbeddingManager
    from src.retrieval.chunking import chunk_documents

    docs = [
        {
            "id": "1",
            "title": "Nga Ukraine xung đột",
            "content": "Putin tuyên bố tiếp tục chiến dịch. Zelensky kêu gọi NATO hỗ trợ vũ khí.",
            "full_text": "Putin tuyên bố. Zelensky kêu gọi NATO.",
            "source": "VnExpress",
            "date": "2024-01-15",
            "url": "https://vne.vn/1",
            "category": "thế giới",
            "linked_entities": [
                {"canonical": "Putin", "type": "PER", "link_score": 1.0},
                {"canonical": "Ukraine", "type": "LOC", "link_score": 1.0},
                {"canonical": "NATO", "type": "ORG", "link_score": 1.0},
            ],
        },
        {
            "id": "2",
            "title": "WHO cảnh báo COVID",
            "content": "WHO phát cảnh báo COVID-19 tại châu Á.",
            "full_text": "WHO cảnh báo COVID-19.",
            "source": "VNN",
            "date": "2024-01-16",
            "url": "https://vnn.vn/2",
            "category": "y tế",
            "linked_entities": [
                {"canonical": "WHO", "type": "ORG", "link_score": 1.0},
                {"canonical": "COVID-19", "type": "MISC", "link_score": 1.0},
            ],
        },
        {
            "id": "3",
            "title": "Samsung khai trương Hà Nội",
            "content": "Samsung mở R&D tại Hà Nội với 3000 kỹ sư.",
            "full_text": "Samsung khai trương R&D tại Hà Nội.",
            "source": "Thanh Niên",
            "date": "2024-01-17",
            "url": "https://tn.vn/3",
            "category": "công nghệ",
            "linked_entities": [
                {"canonical": "Samsung", "type": "ORG", "link_score": 1.0},
                {"canonical": "Hà Nội", "type": "LOC", "link_score": 1.0},
            ],
        },
    ]

    # Chunk + embed
    chunks, doc_to_chunks = chunk_documents(
        docs, strategy="sentence_window", max_chars=200
    )
    em = EmbeddingManager(use_sbert=False)
    em.build_document_index(
        [{"id": c["chunk_id"], "full_text": c["chunk_text"]} for c in chunks]
    )

    ret = Retriever(use_faiss=False, use_cross_encoder=False)
    ret.build(
        chunks,
        em,
        doc_to_chunks,
        docs,
        importance_scores={"Ukraine": 0.9, "NATO": 0.8, "Putin": 0.85},
    )

    for q in ["chiến tranh nga ukraine", "dịch bệnh WHO", "công nghệ samsung"]:
        results = ret.retrieve(q, top_k=2, seed_entities=["Ukraine"])
        print(f"\nQuery: '{q}'")
        for r in results:
            print(
                f"  [{r['retrieval_score']:.3f}] {r['title']} "
                f"(vec={r['vector_score']:.3f}, boost={r['graph_boost']:.3f})"
            )
