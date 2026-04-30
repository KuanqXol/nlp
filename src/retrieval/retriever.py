"""
Retriever dùng FAISS vector search + cross-encoder reranking.

Pipeline:
  1. Encode query với vietnamese-bi-encoder
  2. FAISS tìm FAISS_FETCH_K=50 chunks gần nhất
  3. Deduplicate về document level (giữ chunk tốt nhất mỗi doc)
  4. Graph boost theo PPR score từ seed entities
  5. Cross-encoder rerank top-50 → trả về top_k (mặc định 10)
  6. Date decay (bài cũ bị giảm điểm nhẹ)

Yêu cầu: faiss-cpu, sentence-transformers
"""

import math
import pickle
from collections import defaultdict

from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ── Cấu hình ──────────────────────────────────────────────────────────────────

FAISS_FETCH_K = 50  # Số chunk lấy từ FAISS
GRAPH_BOOST_ALPHA = 0.15  # Trọng số graph boost
DEFAULT_TOP_K = 10  # Số doc trả về mặc định
DATE_DECAY_HALFLIFE = 365.0  # Ngày → decay = exp(-days/365)
DATE_DECAY_ENABLED = True
DATE_DECAY_WEIGHT = 0.08  # Contribution của date decay vào final score (0→tắt, 1→full)
MAX_CHUNKS_PER_DOC = 2  # Tối đa giữ N chunk/doc khi dedupe

# Cross-encoder: dùng model fine-tuned nếu có, fallback ms-marco
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L6-v2"


# ── FAISS Backend ─────────────────────────────────────────────────────────────


class _FaissBackend:
    def __init__(self):
        self._index = None
        self._ids: List[str] = []

    def build(self, embeddings: np.ndarray, ids: List[str]):
        n, d = embeddings.shape
        self._ids = list(ids)
        normed = self._normalize(embeddings)

        if n > 50_000:
            nlist = min(256, int(n**0.5))
            q = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(normed)
            self._index.add(normed)
            self._index.nprobe = 32
            print(f"[FAISS] IVFFlat index: {n} vectors, nlist={nlist}, nprobe=32")
        else:
            self._index = faiss.IndexFlatIP(d)
            self._index.add(normed)
            print(f"[FAISS] Flat index: {n} vectors")

    def search(self, qvec: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        if self._index is None:
            return [], []
        q = self._normalize(qvec.reshape(1, -1))
        k = min(k, len(self._ids))
        dists, idxs = self._index.search(q, k)
        ids = [self._ids[i] for i in idxs[0] if i >= 0]
        scores = [float(d) for d in dists[0][: len(ids)]]
        return ids, scores

    def save(self, path: str):
        if self._index is not None:
            faiss.write_index(self._index, path)

    def load(self, path: str, ids: List[str]):
        self._index = faiss.read_index(path)
        self._ids = list(ids)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.where(norms == 0, 1e-8, norms)


# ── Cross-encoder Reranker ────────────────────────────────────────────────────


class _CrossEncoderReranker:
    def __init__(self, model_dir: str, device: str = None):
        if not _CROSS_ENCODER_AVAILABLE:
            raise ImportError("Cần cài sentence-transformers để dùng cross-encoder.")
        # Auto-detect GPU
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device
        # Batch size tối ưu: GPU 128, CPU 32
        self._batch_size = 128 if device.startswith("cuda") else 32
        print(f"[Reranker] Đang load: {model_dir} (device={device})")
        self._model = CrossEncoder(model_dir, device=device)
        print("[Reranker] Sẵn sàng.")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        text_field: str = "chunk_text",
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        if not candidates:
            return candidates

        pairs = [(query, c.get(text_field, c.get("full_text", ""))) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        for cand, score in zip(candidates, scores):
            cand["cross_encoder_score"] = round(float(score), 4)
            cand["retrieval_score"] = round(float(score), 4)

        candidates.sort(key=lambda x: -x["retrieval_score"])
        return candidates[:top_k] if top_k else candidates


# ── Retriever ─────────────────────────────────────────────────────────────────


class Retriever:
    """
    FAISS vector retriever với cross-encoder reranking và graph boost.

    Cách dùng:
        ret = Retriever()
        ret.build(chunks, em, doc_to_chunks, docs, graph_ranker=ranker, kg=kg)
        results = ret.retrieve("chiến tranh nga ukraine", top_k=10)
    """

    def __init__(
        self,
        use_faiss: bool = True,
        reranker_model_dir: Optional[str] = None,
        use_cross_encoder: bool = True,
        **kwargs,  # backward-compat
    ):
        if not _FAISS_AVAILABLE:
            raise ImportError("Cần cài faiss-cpu: pip install faiss-cpu")

        # Auto-detect GPU device (dùng cho cross-encoder)
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            self._device = "cpu"

        self._backend = _FaissBackend()

        # Cross-encoder: ưu tiên fine-tuned model nếu có
        self._reranker: Optional[_CrossEncoderReranker] = None
        if use_cross_encoder and _CROSS_ENCODER_AVAILABLE:
            model_path = reranker_model_dir
            if model_path is None:
                default = (
                    Path(__file__).resolve().parents[2] / "data" / "reranker_model"
                )
                model_path = (
                    str(default)
                    if (default / "config.json").exists()
                    else DEFAULT_CROSS_ENCODER
                )
            try:
                self._reranker = _CrossEncoderReranker(model_path, device=self._device)
            except Exception as e:
                print(f"[Reranker] Không load được ({e}), tắt rerank.")

        # State
        self._em = None
        self._documents: Dict[str, Dict] = {}
        self._chunks: Dict[str, Dict] = {}
        self._doc_to_chunks: Dict[str, List[str]] = {}
        self._graph_ranker = None
        self._kg = None
        self._global_scores: Dict[str, float] = {}
        self._chunk_mode = False

    # ── Build ─────────────────────────────────────────────────────────────

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
        """Build chunk-aware FAISS index."""
        self._em = embedding_manager
        self._documents = {d["id"]: d for d in documents}
        self._chunks = {c["chunk_id"]: c for c in chunks}
        self._doc_to_chunks = doc_to_chunks
        self._graph_ranker = graph_ranker
        self._kg = kg
        self._global_scores = importance_scores or {}
        self._chunk_mode = True

        embs = embedding_manager.doc_embeddings
        chunk_ids = embedding_manager.doc_ids
        if embs is None or not chunk_ids:
            print("[Retriever] WARNING: embeddings chưa build!")
            return
        self._backend.build(embs, chunk_ids)
        print(
            f"[Retriever] Index: {len(chunk_ids)} chunks, {len(self._documents)} docs"
        )

    def build_simple(
        self,
        documents: List[Dict],
        embedding_manager,
        importance_scores: Dict[str, float] = None,
    ):
        """Backward-compat: index theo document (không chunking)."""
        self._em = embedding_manager
        self._documents = {d["id"]: d for d in documents}
        self._global_scores = importance_scores or {}
        self._chunk_mode = False

        embs = embedding_manager.doc_embeddings
        doc_ids = embedding_manager.doc_ids
        if embs is None:
            return
        self._backend.build(embs, doc_ids)
        print(f"[Retriever] Document index: {len(doc_ids)} docs")

    # ── Search ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        seed_entities: List[str] = None,
        rerank: bool = True,
        apply_decay: bool = True,
    ) -> List[Dict]:
        return self.search(
            query,
            top_k=top_k,
            seed_entities=seed_entities,
            rerank=rerank,
            apply_decay=apply_decay,
        )

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        seed_entities: List[str] = None,
        rerank: bool = True,
        apply_decay: bool = True,
    ) -> List[Dict]:

        if self._em is None:
            return []

        query_vec = self._em.encode_query(query)
        fetch_k = min(FAISS_FETCH_K, max(len(self._em.doc_ids), 1))
        chunk_ids, vec_scores = self._backend.search(query_vec, k=fetch_k)

        if not chunk_ids:
            return []

        vec_score_map = dict(zip(chunk_ids, vec_scores))

        # PPR scores nếu có seed entities
        ppr_scores: Dict[str, float] = {}
        if seed_entities and self._graph_ranker and self._kg:
            ppr_scores = self._graph_ranker.query_time_scores(
                self._kg, seeds=seed_entities
            )

        # Build candidates (chunk → doc dedup)
        if self._chunk_mode:
            candidates = self._candidates_from_chunks(
                chunk_ids, vec_score_map, ppr_scores
            )
        else:
            candidates = self._candidates_from_docs(
                chunk_ids, vec_score_map, ppr_scores
            )

        # Cross-encoder rerank
        if rerank and self._reranker and candidates:
            text_field = "chunk_text" if self._chunk_mode else "full_text"
            candidates = self._reranker.rerank(query, candidates, text_field=text_field)
        else:
            candidates.sort(key=lambda x: -x["retrieval_score"])

        # Date decay sau rerank để không ảnh hưởng cross-encoder score
        if apply_decay:
            candidates = self._apply_date_decay(candidates)
            candidates.sort(key=lambda x: -x["retrieval_score"])

        return candidates[:top_k]

    def retrieve_with_expansion(
        self,
        expansion_result: Dict,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = True,
    ) -> List[Dict]:
        expanded_query = expansion_result.get("expanded_query", "")
        seed_entities = expansion_result.get("seed_entities", [])
        return self.retrieve(
            expanded_query, top_k=top_k, seed_entities=seed_entities, rerank=rerank
        )

    # ── Candidates ────────────────────────────────────────────────────────

    def _candidates_from_chunks(
        self,
        chunk_ids: List[str],
        vec_scores: Dict[str, float],
        ppr_scores: Dict[str, float],
    ) -> List[Dict]:
        """Chunk → dedupe theo doc, giữ tối đa MAX_CHUNKS_PER_DOC chunk/doc.

        Trước đây chỉ giữ 1 chunk duy nhất (best dict không dùng MAX_CHUNKS_PER_DOC).
        Giờ gom top-N chunk/doc, chọn chunk_text từ chunk có vscore cao nhất
        nhưng tính retrieval_score từ mean top-N để ổn định hơn.
        """
        # Gom tất cả chunk theo doc_id
        per_doc: Dict[str, List[Tuple[float, Dict]]] = defaultdict(list)
        for cid in chunk_ids:
            chunk = self._chunks.get(cid)
            if not chunk:
                continue
            doc_id = chunk.get("doc_id", "")
            vscore = vec_scores.get(cid, 0.0)
            per_doc[doc_id].append((vscore, chunk))

        candidates = []
        for doc_id, scored_chunks in per_doc.items():
            # Sắp xếp giảm dần theo vscore, giữ tối đa MAX_CHUNKS_PER_DOC
            scored_chunks.sort(key=lambda x: -x[0])
            top_chunks = scored_chunks[:MAX_CHUNKS_PER_DOC]

            # vscore đại diện = chunk tốt nhất (top-1)
            best_vscore, best_chunk = top_chunks[0]

            doc = self._documents.get(doc_id, {})
            graph_boost = self._graph_boost(doc, ppr_scores)
            final = best_vscore * (1 + GRAPH_BOOST_ALPHA * graph_boost)

            result = dict(doc)
            result.update(
                {
                    "retrieval_score": round(final, 4),
                    "vector_score": round(best_vscore, 4),
                    "graph_boost": round(graph_boost, 4),
                    "chunk_text": best_chunk.get("chunk_text", ""),
                    "chunk_id": best_chunk.get("chunk_id", ""),
                    # Metadata: tất cả chunk_id được dùng (hữu ích khi debug)
                    "matched_chunk_ids": [c.get("chunk_id", "") for _, c in top_chunks],
                }
            )
            candidates.append(result)
        return candidates

    def _candidates_from_docs(
        self,
        doc_ids: List[str],
        vec_scores: Dict[str, float],
        ppr_scores: Dict[str, float],
    ) -> List[Dict]:
        candidates = []
        for doc_id in doc_ids:
            doc = self._documents.get(doc_id)
            if not doc:
                continue
            vscore = vec_scores.get(doc_id, 0.0)
            graph_boost = self._graph_boost(doc, ppr_scores)
            final = vscore * (1 + GRAPH_BOOST_ALPHA * graph_boost)
            result = dict(doc)
            result.update(
                {
                    "retrieval_score": round(final, 4),
                    "vector_score": round(vscore, 4),
                    "graph_boost": round(graph_boost, 4),
                }
            )
            candidates.append(result)
        return candidates

    def _graph_boost(self, doc: Dict, ppr_scores: Dict[str, float]) -> float:
        """Tính graph boost = mean top-3 PPR scores của entity trong doc."""
        entities = doc.get("linked_entities", [])
        score_src = ppr_scores if ppr_scores else self._global_scores
        scores = [score_src.get(e.get("canonical", ""), 0.0) for e in entities]
        if not scores:
            return 0.0
        top3 = sorted(scores, reverse=True)[:3]
        return sum(top3) / len(top3)

    # ── Date decay ────────────────────────────────────────────────────────

    def _apply_date_decay(self, candidates: List[Dict]) -> List[Dict]:
        """Áp dụng date decay theo công thức additive blend:
            final = score * (1 - w) + score * decay * w
                  = score * (1 - w * (1 - decay))

        Với w = DATE_DECAY_WEIGHT = 0.08, bài 1 năm trước chỉ giảm tối đa
        8% thay vì 63% như khi nhân trực tiếp (exp(-1) ≈ 0.37).
        Bài không có ngày được gán decay = 0.85 (tương đương ~60 ngày tuổi)
        thay vì 0.5 như trước.
        """
        if not DATE_DECAY_ENABLED or DATE_DECAY_WEIGHT <= 0:
            return candidates
        today = date.today()
        for doc in candidates:
            raw = doc.get("date", "")
            decay = 0.85  # fallback khi không parse được ngày
            if raw:
                for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
                    try:
                        d = datetime.strptime(str(raw)[:10], fmt).date()
                        days_ago = max((today - d).days, 0)
                        decay = math.exp(-days_ago / DATE_DECAY_HALFLIFE)
                        break
                    except ValueError:
                        continue
            doc["date_decay_weight"] = round(decay, 4)
            # Additive blend: chỉ DATE_DECAY_WEIGHT phần trăm score bị ảnh hưởng bởi decay
            doc["retrieval_score"] = round(
                doc["retrieval_score"] * (1.0 - DATE_DECAY_WEIGHT * (1.0 - decay)), 4
            )
        return candidates

    # ── Multi-query retrieve ───────────────────────────────────────────────

    def multi_query_retrieve(
        self,
        queries: List[str],
        top_k: int = DEFAULT_TOP_K,
        seed_entities: List[str] = None,
        rerank: bool = True,
    ) -> List[Dict]:
        """
        Retrieve từ nhiều query variant (từ query expansion).

        Luồng:
          1. Mỗi query variant → FAISS riêng → lấy top_k*2 candidates
          2. Merge theo max vector score (mỗi doc giữ lần xuất hiện tốt nhất)
          3. Rerank toàn bộ merged pool 1 lần duy nhất với queries[0]
          4. Date decay → sort → top_k

        Tại sao rerank chỉ dùng queries[0] (query gốc):
          Cross-encoder đọc [query | doc] như 1 chuỗi — attention cross chạy
          giữa từng token của query với từng token của doc. Query gốc ngắn, sạch,
          rõ intent → cross-encoder score chính xác nhất.
          Concat thêm expansion vào query ("nga ukraine Putin NATO Zelensky") không
          giúp cross-encoder vì nó đã có đủ token để attend; ngược lại làm dài
          chuỗi input, tăng latency, có thể nhiễu attention với entity không liên quan.
          Expansion chỉ có tác dụng ở bước FAISS (bi-encoder) — nơi mỗi query
          variant được encode thành vector riêng, bắt được candidate khác nhau.
        """
        if not queries:
            return []
        if len(queries) == 1:
            return self.retrieve(
                queries[0], top_k=top_k, seed_entities=seed_entities, rerank=rerank
            )

        # Bước 1: FAISS retrieval riêng cho từng query variant, không decay
        merged: Dict[str, Dict] = {}
        for q in queries:
            results = self.retrieve(
                q,
                top_k=top_k * 2,
                seed_entities=seed_entities,
                rerank=False,
                apply_decay=False,
            )
            for r in results:
                doc_id = r.get("id", "")
                if (
                    doc_id not in merged
                    or r["retrieval_score"] > merged[doc_id]["retrieval_score"]
                ):
                    merged[doc_id] = r

        candidates = sorted(merged.values(), key=lambda x: -x["retrieval_score"])

        # Bước 2: Rerank 1 lần với query gốc (queries[0]) — KHÔNG concat expansion
        if rerank and self._reranker and candidates:
            text_field = "chunk_text" if self._chunk_mode else "full_text"
            candidates = self._reranker.rerank(
                queries[0], candidates[:FAISS_FETCH_K], text_field=text_field
            )

        # Bước 3: Date decay sau rerank, sort, trả về top_k
        candidates = self._apply_date_decay(candidates)
        candidates.sort(key=lambda x: -x["retrieval_score"])
        return candidates[:top_k]

    # ── Attach State (for load_index path) ───────────────────────────────

    def attach_state(
        self,
        embedding_manager,
        documents: List[Dict],
        chunks,
        doc_to_chunks: Dict[str, List[str]],
        graph_ranker=None,
        kg=None,
        importance_scores: Dict[str, float] = None,
        chunk_mode: bool = True,
    ):
        """
        Gắn lại state đã load từ disk vào Retriever mà không rebuild index.
        Được gọi bởi NewsSearchSystem.load_index().
        """
        self._em = embedding_manager
        self._documents = {d["id"]: d for d in documents}
        self._graph_ranker = graph_ranker
        self._kg = kg
        self._global_scores = importance_scores or {}
        self._chunk_mode = chunk_mode

        # chunks có thể là list hoặc dict
        if isinstance(chunks, dict):
            self._chunks = chunks
        else:
            self._chunks = {c["chunk_id"]: c for c in chunks if "chunk_id" in c}

        self._doc_to_chunks = doc_to_chunks

    # ── Persistence ──────────────────────────────────────────────────────

    def save_artifacts(self, index_dir: str):
        target = Path(index_dir)
        target.mkdir(parents=True, exist_ok=True)
        if self._em is not None:
            self._backend.save(str(target / "vector.index"))
            print(f"[Retriever] FAISS index lưu tại {target / 'vector.index'}")

    def load_artifacts(self, index_dir: str) -> bool:
        target = Path(index_dir)
        vector_path = target / "vector.index"
        if vector_path.exists() and self._em is not None:
            self._backend.load(str(vector_path), self._em.doc_ids)
            print(f"[Retriever] Loaded FAISS index từ {vector_path}")
            return True
        elif self._em is not None and self._em.doc_embeddings is not None:
            self._backend.build(self._em.doc_embeddings, self._em.doc_ids)
            return True
        return False

    def get_document(self, doc_id: str) -> Optional[Dict]:
        return self._documents.get(doc_id)
