"""
Module: embedding.py
Chức năng: Tạo vector embedding cho entity và document tiếng Việt.

Thư viện:
  - sentence-transformers (primary)
  - transformers (fallback)

Model ưu tiên:
  - keepitreal/vietnamese-sbert
  - VoVanPhuc/sup-SimCSE-VietNamese-phobert-base

Nếu không có GPU / model → dùng TF-IDF fallback để hệ thống vẫn hoạt động.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Dict, List, Optional, Union

import numpy as np

# ── Cố gắng import SentenceTransformers ─────────────────────────────────────

try:
    from sentence_transformers import SentenceTransformer

    _SBERT_AVAILABLE = True
except ImportError:
    _SBERT_AVAILABLE = False


# ── TF-IDF Fallback Embedder ─────────────────────────────────────────────────


class TFIDFEmbedder:
    """
    Embedder đơn giản dựa trên TF-IDF + cosine similarity.
    Dùng khi không có GPU hoặc model SentenceTransformer.
    """

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab: List[str] = []
        self.idf: np.ndarray = np.array([])
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Tách từ đơn giản cho tiếng Việt."""
        text = text.lower()
        # Giữ lại chữ cái tiếng Việt, số và khoảng trắng
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]

    def fit(self, texts: List[str]):
        """Xây dựng vocab và IDF từ corpus."""
        # Đếm DF (document frequency) cho mỗi từ
        df_counter: Counter = Counter()
        tokenized = []
        for text in texts:
            tokens = set(self._tokenize(text))
            tokenized.append(tokens)
            df_counter.update(tokens)

        # Chọn vocab: top N từ theo DF
        common_words = [w for w, _ in df_counter.most_common(self.vocab_size)]
        self.vocab = common_words
        vocab_set = {w: i for i, w in enumerate(self.vocab)}

        n_docs = len(texts)
        self.idf = np.zeros(len(self.vocab))
        for i, word in enumerate(self.vocab):
            df = df_counter.get(word, 0)
            self.idf[i] = math.log((n_docs + 1) / (df + 1)) + 1  # smooth IDF

        self._fitted = True
        print(f"[TFIDFEmbedder] Đã xây dựng vocab với {len(self.vocab)} từ.")

    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Tạo embedding TF-IDF cho text hoặc list of texts."""
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = text
            single = False

        if not self._fitted:
            # Auto-fit với texts đang encode nếu chưa fit
            self.fit(texts)

        vocab_idx = {w: i for i, w in enumerate(self.vocab)}
        embeddings = []

        for t in texts:
            tokens = self._tokenize(t)
            tf_counter: Counter = Counter(tokens)
            n_tokens = max(len(tokens), 1)

            vec = np.zeros(len(self.vocab))
            for word, count in tf_counter.items():
                if word in vocab_idx:
                    idx = vocab_idx[word]
                    tf = count / n_tokens
                    vec[idx] = tf * self.idf[idx]

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        result = np.array(embeddings)
        return result[0] if single else result


# ── SentenceTransformer Embedder ─────────────────────────────────────────────


class SBERTEmbedder:
    """
    Wrapper quanh SentenceTransformer cho tiếng Việt.

    Model ưu tiên:
        keepitreal/vietnamese-sbert → phù hợp nhất cho tiếng Việt
    """

    DEFAULT_MODEL = "keepitreal/vietnamese-sbert"
    FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: Optional[SentenceTransformer] = None

    def _load(self):
        if self._model is None:
            print(f"[SBERTEmbedder] Đang tải model: {self.model_name} ...")
            try:
                self._model = SentenceTransformer(self.model_name)
                print(f"[SBERTEmbedder] Model sẵn sàng: {self.model_name}")
            except Exception as e:
                print(f"[SBERTEmbedder] Lỗi tải {self.model_name}: {e}")
                print(f"[SBERTEmbedder] Thử fallback: {self.FALLBACK_MODEL}")
                self._model = SentenceTransformer(self.FALLBACK_MODEL)

    def encode(
        self,
        text: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Tạo embedding.

        Args:
            text: Một chuỗi hoặc list chuỗi
            batch_size: Kích thước batch
            show_progress: Hiển thị progress bar

        Returns:
            np.ndarray shape (dim,) nếu input là string,
                       shape (N, dim) nếu input là list
        """
        self._load()
        single = isinstance(text, str)
        texts = [text] if single else text

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize → cosine = dot product
        )
        return embeddings[0] if single else embeddings


# ── Unified Embedding Manager ────────────────────────────────────────────────


class EmbeddingManager:
    """
    Quản lý embedding cho cả entity và document.
    Tự động chọn backend phù hợp (SBERT → TFIDF).

    Ví dụ:
        em = EmbeddingManager()
        em.build_document_index(documents)
        vec = em.encode_query("chiến tranh nga ukraine")
        similar_docs = em.get_similar_documents(vec, top_k=5)
    """

    def __init__(self, use_sbert: bool = True, model_name: str = None):
        """
        Args:
            use_sbert: Nếu True và thư viện sẵn có → dùng SBERT
            model_name: Model tùy chọn
        """
        self.use_sbert = use_sbert and _SBERT_AVAILABLE

        if self.use_sbert:
            self._embedder = SBERTEmbedder(model_name)
            print("[EmbeddingManager] Backend: SentenceTransformer (Vietnamese SBERT)")
        else:
            self._embedder = TFIDFEmbedder()
            print("[EmbeddingManager] Backend: TF-IDF (fallback)")

        # Storage
        self._doc_embeddings: Optional[np.ndarray] = None
        self._doc_ids: List[str] = []
        self._entity_embeddings: Dict[str, np.ndarray] = {}
        self._query_embedding_cache: Dict[str, np.ndarray] = {}

    # ── Document index ─────────────────────────────────────────────────────

    def build_document_index(self, documents: List[Dict]):
        """
        Tạo embedding cho tất cả document và lưu vào index.

        Mỗi document cần có ít nhất field 'full_text' hoặc 'content'.
        """
        print(f"[EmbeddingManager] Tạo embedding cho {len(documents)} bài báo...")

        texts = []
        doc_ids = []
        for doc in documents:
            text = doc.get("full_text", doc.get("content", ""))
            if text:
                texts.append(text)
                doc_ids.append(doc.get("id", ""))

        # Nếu dùng TFIDF, cần fit trước
        if not self.use_sbert and hasattr(self._embedder, "fit"):
            self._embedder.fit(texts)

        # Encode theo batch
        if self.use_sbert:
            self._doc_embeddings = self._embedder.encode(texts, show_progress=True)
        else:
            self._doc_embeddings = self._embedder.encode(texts)
            if self._doc_embeddings.ndim == 1:
                self._doc_embeddings = self._doc_embeddings.reshape(1, -1)

        self._doc_ids = doc_ids
        print(
            f"[EmbeddingManager] Index xây dựng xong. Shape: {self._doc_embeddings.shape}"
        )

    def encode_document(self, doc: Dict) -> np.ndarray:
        """Tạo embedding cho một document đơn lẻ."""
        text = doc.get("full_text", doc.get("content", ""))
        return self._embedder.encode(text)

    # ── Query ──────────────────────────────────────────────────────────────

    def encode_query(self, query: str) -> np.ndarray:
        """Tạo embedding cho query string."""
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        if key in self._query_embedding_cache:
            return self._query_embedding_cache[key]
        vec = self._embedder.encode(query)
        self._query_embedding_cache[key] = vec
        return vec

    # ── Entity embeddings ──────────────────────────────────────────────────

    def encode_entities(self, entity_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Tạo embedding cho danh sách entity.
        Cache lại để tránh tính toán lại.
        """
        new_entities = [e for e in entity_names if e not in self._entity_embeddings]

        if new_entities:
            embeddings = self._embedder.encode(new_entities)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            for i, name in enumerate(new_entities):
                self._entity_embeddings[name] = embeddings[i]

        return {
            name: self._entity_embeddings[name]
            for name in entity_names
            if name in self._entity_embeddings
        }

    # ── Similarity helpers ─────────────────────────────────────────────────

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Tính cosine similarity giữa 2 vector."""
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (n1 * n2))

    def get_similar_entities(
        self,
        query_entity: str,
        all_entities: List[str],
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[tuple]:
        """
        Tìm entity tương tự theo embedding similarity.
        Trả về [(entity_name, similarity_score), ...]
        """
        to_encode = [query_entity] + [e for e in all_entities if e != query_entity]
        embeddings = self.encode_entities(to_encode)

        if query_entity not in embeddings:
            return []

        query_vec = embeddings[query_entity]
        results = []
        for ent in all_entities:
            if ent == query_entity or ent not in embeddings:
                continue
            sim = self.cosine_similarity(query_vec, embeddings[ent])
            if sim >= threshold:
                results.append((ent, round(sim, 4)))

        return sorted(results, key=lambda x: -x[1])[:top_k]

    # ── Persistence ───────────────────────────────────────────────────────

    def export_state(self) -> Dict:
        state = {
            "use_sbert": self.use_sbert,
            "model_name": getattr(self._embedder, "model_name", None),
            "doc_embeddings": self._doc_embeddings,
            "doc_ids": list(self._doc_ids),
        }

        if not self.use_sbert and isinstance(self._embedder, TFIDFEmbedder):
            state["tfidf"] = {
                "vocab_size": self._embedder.vocab_size,
                "vocab": list(self._embedder.vocab),
                "idf": np.asarray(self._embedder.idf, dtype=np.float32),
            }

        return state

    @classmethod
    def from_state(cls, state: Dict) -> "EmbeddingManager":
        requested_use_sbert = bool(state.get("use_sbert", False))
        em = cls(
            use_sbert=requested_use_sbert,
            model_name=state.get("model_name"),
        )

        if requested_use_sbert and not em.use_sbert:
            raise RuntimeError(
                "Index này được build bằng SBERT nhưng môi trường hiện tại "
                "không có sentence-transformers."
            )

        if not em.use_sbert and isinstance(em._embedder, TFIDFEmbedder):
            tfidf_state = state.get("tfidf", {})
            em._embedder.vocab_size = int(
                tfidf_state.get("vocab_size", em._embedder.vocab_size)
            )
            em._embedder.vocab = list(tfidf_state.get("vocab", []))
            em._embedder.idf = np.asarray(
                tfidf_state.get("idf", []), dtype=np.float32
            )
            em._embedder._fitted = len(em._embedder.vocab) > 0

        if state.get("doc_embeddings") is not None:
            em._doc_embeddings = np.asarray(
                state["doc_embeddings"], dtype=np.float32
            )
        em._doc_ids = list(state.get("doc_ids", []))
        em._entity_embeddings = {}
        em._query_embedding_cache = {}
        return em

    # ── Properties ─────────────────────────────────────────────────────────

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
        return 0


# ── Demo standalone ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dùng TFIDF để demo nhanh, không cần download model
    em = EmbeddingManager(use_sbert=False)

    sample_docs = [
        {
            "id": "1",
            "full_text": "Putin tuyên bố tiếp tục chiến dịch quân sự tại Ukraine",
        },
        {"id": "2", "full_text": "WHO cảnh báo dịch COVID-19 bùng phát tại châu Á"},
        {"id": "3", "full_text": "VinAI ra mắt mô hình AI tiếng Việt mới"},
        {"id": "4", "full_text": "Kinh tế Việt Nam tăng trưởng mạnh trong quý đầu năm"},
    ]

    em.build_document_index(sample_docs)

    query = "chiến tranh nga ukraine"
    query_vec = em.encode_query(query)

    print(f"\nQuery embedding shape: {query_vec.shape}")
    print(f"Document index shape: {em.doc_embeddings.shape}")

    # So sánh cosine similarity với từng doc
    print("\n=== COSINE SIMILARITY ===")
    for i, doc in enumerate(sample_docs):
        sim = em.cosine_similarity(query_vec, em.doc_embeddings[i])
        print(f"  Doc {doc['id']}: {sim:.4f} | {doc['full_text'][:50]}")
