"""
Embedding tiếng Việt dùng bkai-foundation-models/vietnamese-bi-encoder.

Model xuất ra vector 768 chiều, đã được pretrain trên corpus tiếng Việt
domain báo chí và hội thoại — phù hợp với tập VnExpress 150k bài.
Cần cài:
    pip install sentence-transformers
"""

import hashlib
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


# ── Model mặc định ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"


# ── Embedder ───────────────────────────────────────────────────────────────────


class VietnameseBiEncoder:
    """
    Wrapper quanh SentenceTransformer cho vietnamese-bi-encoder.

    Ví dụ:
        enc = VietnameseBiEncoder()
        vec = enc.encode("Samsung đầu tư tại Việt Nam")   # shape (768,)
        vecs = enc.encode(["query 1", "query 2"])          # shape (2, 768)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None):
        if not _SBERT_AVAILABLE:
            raise ImportError(
                "Cần cài sentence-transformers: pip install sentence-transformers"
            )
        self.model_name = model_name
        # Auto-detect GPU nếu không truyền device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        self._model: Optional[SentenceTransformer] = None
        # Batch size tối ưu: GPU dùng 256, CPU dùng 64
        self._default_batch_size = 64 if device.startswith("cuda") else 32

    def _load(self):
        if self._model is None:
            print(
                f"[Embedding] Đang tải model: {self.model_name} (device={self.device}) ..."
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            print(
                f"[Embedding] Model sẵn sàng. Dim={self._model.get_sentence_embedding_dimension()}"
            )

    def encode(
        self,
        text: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Trả về:
          - shape (dim,)    nếu input là string
          - shape (N, dim)  nếu input là list
        Vectors đã được L2-normalize (cosine = dot product).
        """
        self._load()
        single = isinstance(text, str)
        texts = [text] if single else text
        if batch_size is None:
            batch_size = self._default_batch_size
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vecs[0] if single else vecs

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()


# ── EmbeddingManager ──────────────────────────────────────────────────────────


class EmbeddingManager:
    """
    Quản lý embedding cho chunk và query.

    Ví dụ:
        em = EmbeddingManager()
        em.build_document_index(chunks)   # chunks là list dict có field "full_text"
        vec = em.encode_query("chiến tranh nga ukraine")
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = None, **kwargs):
        # kwargs cho backward-compat (use_sbert, v.v.) — bỏ qua
        self._enc = VietnameseBiEncoder(model_name, device=device)
        self._doc_embeddings: Optional[np.ndarray] = None
        self._doc_ids: List[str] = []
        # Query cache: dict đơn giản.
        # Tác dụng thực tế hạn chế trong CLI (user hiếm khi repeat query y chang),
        # nhưng zero cost khi hit và hữu ích trong batch eval / web service.
        # Không dùng LRU vì: (1) encode query ~2ms, không phải bottleneck thật;
        # (2) cache sẽ bị clear khi process restart — không tích lũy lâu dài.
        self._query_cache: Dict[str, np.ndarray] = {}

    # ── Build index ────────────────────────────────────────────────────────

    def build_document_index(self, documents: List[Dict], batch_size: int = None):
        """
        Encode tất cả document (hoặc chunk) và lưu vào index.
        Mỗi dict cần có field 'full_text' hoặc 'content'.
        """
        texts, ids = [], []
        for doc in documents:
            text = doc.get("full_text", doc.get("content", ""))
            if text:
                texts.append(text)
                ids.append(doc.get("id", ""))

        print(f"[Embedding] Đang encode {len(texts)} texts...")
        self._doc_embeddings = self._enc.encode(
            texts, batch_size=batch_size, show_progress=True
        )
        self._doc_ids = ids
        print(f"[Embedding] Xong. Shape: {self._doc_embeddings.shape}")

    # ── Query ──────────────────────────────────────────────────────────────

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query với cache đơn giản.

        Cache hữu ích trong batch eval (cùng query set chạy nhiều lần)
        và web service (nhiều user search cùng từ khóa phổ biến).
        Với CLI 1 session, ít khi hit nhưng cũng không có overhead đáng kể.
        """
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        if key not in self._query_cache:
            self._query_cache[key] = self._enc.encode(query)
        return self._query_cache[key]

    def clear_query_cache(self):
        """Xóa query cache — hữu ích khi test hoặc đổi model."""
        self._query_cache.clear()

    def encode_entities(self, entity_names: List[str]) -> Dict[str, np.ndarray]:
        """Encode entity names để dùng trong similarity graph."""
        if not entity_names:
            return {}
        vecs = self._enc.encode(entity_names)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        return dict(zip(entity_names, vecs))

    # ── Similarity ─────────────────────────────────────────────────────────

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def get_similar_entities(
        self,
        query_entity: str,
        all_entities: List[str],
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[tuple]:
        embeddings = self.encode_entities([query_entity] + all_entities)
        if query_entity not in embeddings:
            return []
        qvec = embeddings[query_entity]
        results = [
            (e, round(self.cosine_similarity(qvec, embeddings[e]), 4))
            for e in all_entities
            if e != query_entity and e in embeddings
        ]
        return sorted(
            [(e, s) for e, s in results if s >= threshold], key=lambda x: -x[1]
        )[:top_k]

    # ── Persistence ─────────────────────────────────────────────────────────

    def export_state(self) -> Dict:
        return {
            "model_name": self._enc.model_name,
            "doc_embeddings": self._doc_embeddings,
            "doc_ids": list(self._doc_ids),
        }

    @classmethod
    def from_state(cls, state: Dict) -> "EmbeddingManager":
        em = cls(model_name=state.get("model_name", DEFAULT_MODEL))
        if state.get("doc_embeddings") is not None:
            em._doc_embeddings = np.asarray(state["doc_embeddings"], dtype=np.float32)
        em._doc_ids = list(state.get("doc_ids", []))
        return em

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def doc_embeddings(self) -> Optional[np.ndarray]:
        return self._doc_embeddings

    @property
    def doc_ids(self) -> List[str]:
        return self._doc_ids

    @property
    def embedding_dim(self) -> int:
        if self._doc_embeddings is not None:
            return self._doc_embeddings.shape[1]
        return self._enc.dim
