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

import math
import pickle
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
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
GRAPH_BOOST_ALPHA = 0.15
VECTOR_ALPHA = 0.55
BM25_ALPHA = 0.20
RRF_ALPHA = 0.10
MAX_CHUNKS_PER_DOC = 2  # Tối đa N chunk/doc trước khi dedupe
TOP_RERANK_CANDIDATES = 20
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
# Date decay: exp(-days_ago / halflife). Bài cũ 365 ngày ≈ 0.37× score
DATE_DECAY_HALFLIFE = 365.0
# Set None để tắt date decay hoàn toàn
DATE_DECAY_ENABLED = True


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower(), flags=re.UNICODE)


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

    def save(self, path: str):
        if self._index is None or not _FAISS_AVAILABLE:
            return
        faiss.write_index(self._index, path)

    def load(self, path: str, ids: List[str]):
        if not _FAISS_AVAILABLE:
            raise RuntimeError("FAISS không khả dụng trong môi trường hiện tại.")
        self._index = faiss.read_index(path)
        self._ids = list(ids)
        self._dim = self._index.d


class BM25Backend:
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self._ids: List[str] = []
        self._doc_len: List[int] = []
        self._avgdl = 0.0
        self._idf: Dict[str, float] = {}
        self._tfs: List[Counter] = []

    def build(self, texts: List[str], ids: List[str]):
        self._ids = list(ids)
        self._tfs = []
        self._doc_len = []

        df_counter: Counter = Counter()
        for text in texts:
            tokens = _tokenize(text)
            tf = Counter(tokens)
            self._tfs.append(tf)
            self._doc_len.append(len(tokens))
            df_counter.update(tf.keys())

        n_docs = max(len(self._ids), 1)
        self._avgdl = sum(self._doc_len) / max(len(self._doc_len), 1)
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for term, df in df_counter.items()
        }

    def search(self, query: str, k: int) -> Tuple[List[str], List[float]]:
        if not self._ids:
            return [], []

        q_terms = _tokenize(query)
        if not q_terms:
            return [], []

        qtf = Counter(q_terms)
        scores: List[Tuple[int, float]] = []
        avgdl = self._avgdl or 1.0

        for idx, tf in enumerate(self._tfs):
            score = 0.0
            dl = self._doc_len[idx] or 1
            for term, qfreq in qtf.items():
                f = tf.get(term, 0)
                if f == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += idf * ((f * (self.k1 + 1)) / max(denom, 1e-8)) * qfreq
            if score > 0:
                scores.append((idx, float(score)))

        scores.sort(key=lambda item: -item[1])
        top = scores[: min(k, len(scores))]
        return [self._ids[i] for i, _ in top], [score for _, score in top]

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "ids": self._ids,
                    "doc_len": self._doc_len,
                    "avgdl": self._avgdl,
                    "idf": self._idf,
                    "tfs": self._tfs,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.k1 = float(state.get("k1", BM25_K1))
        self.b = float(state.get("b", BM25_B))
        self._ids = list(state.get("ids", []))
        self._doc_len = list(state.get("doc_len", []))
        self._avgdl = float(state.get("avgdl", 0.0))
        self._idf = dict(state.get("idf", {}))
        self._tfs = list(state.get("tfs", []))


def rrf_merge(
    vector_ids: List[str],
    bm25_ids: List[str],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    fused: Dict[str, float] = defaultdict(float)
    for rank, doc_id in enumerate(vector_ids, start=1):
        fused[doc_id] += 1.0 / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


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
        use_bm25: bool = True,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
    ):
        self._backend = (
            _FaissBackend() if (use_faiss and _FAISS_AVAILABLE) else _NumpyBackend()
        )
        print(
            f"[Retriever] Backend: {'FAISS' if isinstance(self._backend, _FaissBackend) else 'NumPy'}"
        )

        self._bm25 = BM25Backend() if use_bm25 else None
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

    def _attach_state(
        self,
        embedding_manager,
        documents: List[Dict],
        chunks: Optional[List[Dict]] = None,
        doc_to_chunks: Optional[Dict[str, List[str]]] = None,
        graph_ranker=None,
        kg=None,
        importance_scores: Dict[str, float] = None,
        chunk_mode: bool = False,
    ):
        self._em = embedding_manager
        self._documents = {d["id"]: d for d in documents}
        self._chunks = {c["chunk_id"]: c for c in (chunks or [])}
        self._doc_to_chunks = doc_to_chunks or {}
        self._graph_ranker = graph_ranker
        self._kg = kg
        self._global_scores = importance_scores or {}
        self._chunk_mode = chunk_mode

    def attach_state(
        self,
        embedding_manager,
        documents: List[Dict],
        chunks: Optional[List[Dict]] = None,
        doc_to_chunks: Optional[Dict[str, List[str]]] = None,
        graph_ranker=None,
        kg=None,
        importance_scores: Dict[str, float] = None,
        chunk_mode: bool = False,
    ):
        self._attach_state(
            embedding_manager=embedding_manager,
            documents=documents,
            chunks=chunks,
            doc_to_chunks=doc_to_chunks,
            graph_ranker=graph_ranker,
            kg=kg,
            importance_scores=importance_scores,
            chunk_mode=chunk_mode,
        )

    def _build_bm25(self):
        if not self._bm25 or self._em is None:
            return
        texts: List[str] = []
        ids = list(self._em.doc_ids)
        if self._chunk_mode:
            for chunk_id in ids:
                chunk = self._chunks.get(chunk_id, {})
                texts.append(chunk.get("chunk_text", ""))
        else:
            for doc_id in ids:
                doc = self._documents.get(doc_id, {})
                texts.append(doc.get("full_text", doc.get("content", "")))
        self._bm25.build(texts, ids)

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
        self._attach_state(
            embedding_manager=embedding_manager,
            documents=documents,
            chunks=chunks,
            doc_to_chunks=doc_to_chunks,
            graph_ranker=graph_ranker,
            kg=kg,
            importance_scores=importance_scores,
            chunk_mode=True,
        )

        embeddings = embedding_manager.doc_embeddings
        chunk_ids = embedding_manager.doc_ids  # doc_ids thực ra là chunk_ids

        if embeddings is None or not chunk_ids:
            print("[Retriever] WARNING: embeddings chưa build!")
            return

        self._backend.build(embeddings, chunk_ids)
        self._build_bm25()
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
        self._attach_state(
            embedding_manager=embedding_manager,
            documents=documents,
            importance_scores=importance_scores,
            chunk_mode=False,
        )

        embeddings = embedding_manager.doc_embeddings
        doc_ids = embedding_manager.doc_ids

        if embeddings is None:
            return

        self._backend.build(embeddings, doc_ids)
        self._build_bm25()
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
        corpus_size = max(len(self._em.doc_ids), 1)
        fetch_k = min(
            max(top_k * fetch_multiplier, TOP_RERANK_CANDIDATES),
            corpus_size,
        )
        vec_ids, vec_scores = self._backend.search(query_vec, k=fetch_k)
        bm25_ids, bm25_scores = (
            self._bm25.search(query, k=fetch_k) if self._bm25 else ([], [])
        )

        fused = rrf_merge(vec_ids, bm25_ids)
        ids = [doc_id for doc_id, _ in fused[:fetch_k]]

        if not ids and vec_ids:
            ids = vec_ids
        if not ids and bm25_ids:
            ids = bm25_ids
        if not ids:
            return []

        vec_score_map = dict(zip(vec_ids, vec_scores))
        bm25_score_map = dict(zip(bm25_ids, bm25_scores))
        rrf_score_map = dict(fused)
        bm25_max = max(bm25_score_map.values(), default=0.0)
        # Normalize RRF: tính max để chuẩn hóa về [0,1]
        rrf_max = max(rrf_score_map.values(), default=0.0)

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
                ids,
                vec_score_map,
                bm25_score_map,
                rrf_score_map,
                bm25_max,
                ppr_scores,
                rrf_max=rrf_max,
            )
        else:
            candidates = self._build_candidates_from_docs(
                ids,
                vec_score_map,
                bm25_score_map,
                rrf_score_map,
                bm25_max,
                ppr_scores,
                rrf_max=rrf_max,
            )

        candidates = self._rerank(query, candidates, top_k=top_k, rerank=rerank)
        # Date decay: áp dụng sau rerank để cross-encoder score không bị ảnh hưởng
        candidates = self._apply_date_decay(candidates)

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
        if not candidates:
            return candidates[:top_k]
        if not rerank:
            return candidates[:top_k]

        text_field = "chunk_text" if self._chunk_mode else "full_text"
        head = list(candidates[:TOP_RERANK_CANDIDATES])
        tail = list(candidates[TOP_RERANK_CANDIDATES:])
        if self._reranker:
            head = self._reranker.rerank(
                query,
                head,
                text_field=text_field,
                top_k=None,
            )
        combined = head + tail
        combined.sort(key=lambda x: -x["retrieval_score"])
        return combined[:top_k]

    def _build_candidates_from_chunks(
        self,
        chunk_ids: List[str],
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        rrf_scores: Dict[str, float],
        bm25_max: float,
        ppr_scores: Dict[str, float],
        rrf_max: float = 0.0,
    ) -> List[Dict]:
        """
        Chunk-level → dedupe về document level.
        Giữ chunk có score cao nhất per document.
        """
        # chunk_id → best (hybrid_score, chunk, raw scores)
        best_per_doc: Dict[str, Tuple[float, Dict, float, float, float]] = {}

        for cid in chunk_ids:
            chunk = self._chunks.get(cid)
            if not chunk:
                continue
            doc_id = chunk.get("doc_id", "")
            vscore = vec_scores.get(cid, 0.0)
            bm25_score = bm25_scores.get(cid, 0.0)
            rrf_score = rrf_scores.get(cid, 0.0)
            hybrid = self._base_rank_score(vscore, bm25_score, rrf_score, bm25_max, rrf_max)
            if doc_id not in best_per_doc or hybrid > best_per_doc[doc_id][0]:
                best_per_doc[doc_id] = (hybrid, chunk, vscore, bm25_score, rrf_score)

        candidates = []
        for doc_id, (_, chunk, vscore, bm25_score, rrf_score) in best_per_doc.items():
            doc = self._documents.get(doc_id, {})
            graph_boost = self._compute_graph_boost(doc, ppr_scores)
            final_score = self._base_rank_score(
                vscore, bm25_score, rrf_score, bm25_max, rrf_max
            ) + GRAPH_BOOST_ALPHA * graph_boost

            result = dict(doc)
            result.update(
                {
                    "retrieval_score": round(final_score, 4),
                    "vector_score": round(vscore, 4),
                    "bm25_score": round(bm25_score, 4),
                    "rrf_score": round(rrf_score, 4),
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
        vec_scores: Dict[str, float],
        bm25_scores: Dict[str, float],
        rrf_scores: Dict[str, float],
        bm25_max: float,
        ppr_scores: Dict[str, float],
        rrf_max: float = 0.0,
    ) -> List[Dict]:
        candidates = []
        for doc_id in doc_ids:
            doc = self._documents.get(doc_id)
            if not doc:
                continue
            vscore = vec_scores.get(doc_id, 0.0)
            bm25_score = bm25_scores.get(doc_id, 0.0)
            rrf_score = rrf_scores.get(doc_id, 0.0)
            graph_boost = self._compute_graph_boost(doc, ppr_scores)
            final_score = self._base_rank_score(
                vscore, bm25_score, rrf_score, bm25_max, rrf_max
            ) + GRAPH_BOOST_ALPHA * graph_boost

            result = dict(doc)
            result.update(
                {
                    "retrieval_score": round(final_score, 4),
                    "vector_score": round(vscore, 4),
                    "bm25_score": round(bm25_score, 4),
                    "rrf_score": round(rrf_score, 4),
                    "graph_boost": round(graph_boost, 4),
                }
            )
            candidates.append(result)
        return candidates

    def _base_rank_score(
        self,
        vector_score: float,
        bm25_score: float,
        rrf_score: float,
        bm25_max: float,
        rrf_max: float = 0.0,
    ) -> float:
        bm25_norm = (bm25_score / bm25_max) if bm25_max > 0 else 0.0
        # Normalize RRF về [0,1] thay vì dùng raw score (~0.01–0.03)
        rrf_norm = (rrf_score / rrf_max) if rrf_max > 0 else 0.0
        return (
            VECTOR_ALPHA * vector_score
            + BM25_ALPHA * bm25_norm
            + RRF_ALPHA * rrf_norm
        )

    def _apply_date_decay(
        self,
        candidates: List[Dict],
        reference_date: date = None,
    ) -> List[Dict]:
        """
        Nhân retrieval_score với date decay weight = exp(-days_ago / halflife).

        Bài mới nhất (days_ago=0)  → weight=1.0  (không đổi)
        Bài cũ 1 năm (days_ago=365) → weight≈0.37
        Bài cũ 2 năm (days_ago=730) → weight≈0.14

        Nếu không parse được ngày (thiếu trường date / sai format)
        → weight=0.5 (tránh penalty không công bằng).
        """
        if not DATE_DECAY_ENABLED or not candidates:
            return candidates

        today = reference_date or date.today()
        for doc in candidates:
            raw_date = doc.get("date", "")
            weight = 0.5  # default khi thiếu/lỗi ngày
            if raw_date:
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
                    try:
                        doc_date = datetime.strptime(str(raw_date)[:10], fmt).date()
                        days_ago = max((today - doc_date).days, 0)
                        weight = math.exp(-days_ago / DATE_DECAY_HALFLIFE)
                        break
                    except ValueError:
                        continue
            doc["date_decay_weight"] = round(weight, 4)
            doc["retrieval_score"] = round(doc["retrieval_score"] * weight, 4)
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

    def save_artifacts(self, index_dir: str):
        target_dir = Path(index_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self._backend, _FaissBackend) and self._em is not None:
            self._backend.save(str(target_dir / "vector.index"))
        if self._bm25:
            self._bm25.save(str(target_dir / "bm25.pkl"))

    def load_artifacts(self, index_dir: str) -> bool:
        target_dir = Path(index_dir)
        loaded_any = False

        if self._em is not None:
            vector_path = target_dir / "vector.index"
            if vector_path.exists() and isinstance(self._backend, _FaissBackend):
                self._backend.load(str(vector_path), self._em.doc_ids)
                loaded_any = True
            elif self._em.doc_embeddings is not None and self._em.doc_ids:
                self._backend.build(self._em.doc_embeddings, self._em.doc_ids)

        bm25_path = target_dir / "bm25.pkl"
        if self._bm25:
            if bm25_path.exists():
                self._bm25.load(str(bm25_path))
                loaded_any = True
            else:
                self._build_bm25()

        return loaded_any


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
