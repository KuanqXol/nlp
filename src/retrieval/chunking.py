"""
chunking.py
───────────
Document chunking để cải thiện embedding quality.

Vấn đề v1: mỗi document được encode thành 1 vector duy nhất cho toàn bộ
full_text. Bài báo 800 từ về Nga-Ukraine có đoạn ngắn về WHO →
vector bị "diluted", recall kém với query cụ thể.

Giải pháp: chia document thành chunks nhỏ, embed từng chunk,
khi retrieve tìm chunk liên quan nhất → lấy document chứa chunk đó.

Chiến lược chunking:
  - Paragraph-based: tách theo dấu xuống dòng kép
  - Sentence-window: mỗi chunk = N câu liên tiếp (sliding window)
  - Title-prefixed: prepend title vào mỗi chunk (quan trọng cho news)
"""

import re
from typing import Dict, List, Optional, Tuple

# Dùng shared sentence splitter — nhất quán với ner.py
from src.utils.text import split_sentences as _split_sentences


# ── Cấu hình ─────────────────────────────────────────────────────────────────

DEFAULT_MAX_CHUNK_CHARS = 400  # ~100 tokens SBERT
DEFAULT_MIN_CHUNK_CHARS = 80
DEFAULT_OVERLAP_SENTENCES = 1  # Số câu overlap giữa 2 chunk liên tiếp


# ── Chunking functions ────────────────────────────────────────────────────────


# _split_sentences được import từ src.utils.text (split_sentences)
# Xem: from src.utils.text import split_sentences as _split_sentences


def chunk_by_sentences(
    text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_SENTENCES,
    min_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> List[str]:
    """
    Sliding window qua sentences.
    Mỗi chunk tối đa max_chars ký tự, overlap N câu với chunk trước.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text[:max_chars]] if len(text) >= min_chars else []

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sents = []
        chunk_len = 0
        j = i
        while j < len(sentences):
            s_len = len(sentences[j])
            if chunk_len + s_len > max_chars and chunk_sents:
                break
            chunk_sents.append(sentences[j])
            chunk_len += s_len + 1
            j += 1

        chunk_text = " ".join(chunk_sents)
        if len(chunk_text) >= min_chars:
            chunks.append(chunk_text)

        # Overlap: tiến i về sau overlap câu từ đầu chunk hiện tại
        advance = max(1, len(chunk_sents) - overlap)
        i += advance

    return chunks


def chunk_document(
    doc: Dict,
    strategy: str = "sentence_window",
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_SENTENCES,
    prepend_title: bool = True,
) -> List[Dict]:
    """
    Chia một document thành danh sách chunk dicts.

    Mỗi chunk có:
        chunk_id:   "{doc_id}_chunk_{i}"
        doc_id:     ID của document gốc
        chunk_text: Nội dung chunk (có prepend title nếu prepend_title=True)
        chunk_index: Thứ tự chunk trong document
        title:       Tiêu đề document gốc
        date:        Ngày đăng
        source:      Nguồn
        category:    Thể loại
        url:         URL
        is_title_chunk: True nếu là chunk đầu tiên (quan trọng nhất)

    Args:
        strategy: "sentence_window" | "full" (no chunking, backward compat)
    """
    doc_id = doc.get("id", "")
    title = doc.get("title", "")
    content = doc.get("content", doc.get("full_text", ""))
    date = doc.get("date", "")
    source = doc.get("source", "")
    category = doc.get("category", "")
    url = doc.get("url", "")

    if strategy == "full" or not content:
        # Backward compat: không chunk
        text = f"{title}. {content}" if title else content
        return [
            {
                "chunk_id": f"{doc_id}_chunk_0",
                "doc_id": doc_id,
                "chunk_text": text,
                "chunk_index": 0,
                "title": title,
                "date": date,
                "source": source,
                "category": category,
                "url": url,
                "is_title_chunk": True,
            }
        ]

    # Sentence window chunking
    raw_chunks = chunk_by_sentences(content, max_chars=max_chars, overlap=overlap)

    if not raw_chunks:
        raw_chunks = [content[:max_chars]]

    chunks = []
    for idx, chunk_text in enumerate(raw_chunks):
        # Prepend title vào mỗi chunk → cải thiện retrieval
        if prepend_title and title:
            display_text = f"{title}. {chunk_text}"
        else:
            display_text = chunk_text

        chunks.append(
            {
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "doc_id": doc_id,
                "chunk_text": display_text,
                "chunk_index": idx,
                "title": title,
                "date": date,
                "source": source,
                "category": category,
                "url": url,
                "is_title_chunk": (idx == 0),
            }
        )

    return chunks


def chunk_documents(
    documents: List[Dict],
    strategy: str = "sentence_window",
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap: int = DEFAULT_OVERLAP_SENTENCES,
    prepend_title: bool = True,
    log_every: int = 1000,
) -> Tuple[List[Dict], Dict[str, List[str]]]:
    """
    Chunk toàn bộ corpus.

    Returns:
        (all_chunks, doc_to_chunks)
        doc_to_chunks: {doc_id: [chunk_id, ...]}
    """
    all_chunks: List[Dict] = []
    doc_to_chunks: Dict[str, List[str]] = {}

    for i, doc in enumerate(documents):
        chunks = chunk_document(
            doc,
            strategy=strategy,
            max_chars=max_chars,
            overlap=overlap,
            prepend_title=prepend_title,
        )
        doc_id = doc.get("id", "")
        doc_to_chunks[doc_id] = [c["chunk_id"] for c in chunks]
        all_chunks.extend(chunks)

        if (i + 1) % log_every == 0 or (i + 1) == len(documents):
            print(f"[Chunking] {i+1}/{len(documents)} docs → {len(all_chunks)} chunks")

    avg_chunks = len(all_chunks) / max(len(documents), 1)
    print(
        f"[Chunking] Done: {len(all_chunks)} chunks "
        f"(avg {avg_chunks:.1f}/doc, strategy={strategy})"
    )
    return all_chunks, doc_to_chunks


class ChunkAwareEmbeddingManager:
    """
    EmbeddingManager mở rộng hỗ trợ chunk-level indexing.

    Thay vì index document → index chunk.
    Khi retrieve: tìm chunk → trả về document gốc.

    Usage:
        cem = ChunkAwareEmbeddingManager()
        chunks, doc_to_chunks = chunk_documents(docs)
        cem.build_chunk_index(chunks)
        results = cem.search_chunks(query_vec, top_k=20)
        # results: list of chunk dicts, deduped về doc level
    """

    def __init__(self, embedding_manager):
        self._em = embedding_manager
        self._chunks: List[Dict] = []
        self._chunk_ids: List[str] = []
        self._doc_to_chunks: Dict[str, List[str]] = {}
        self._chunk_to_doc: Dict[str, str] = {}

    def build_chunk_index(self, chunks: List[Dict]):
        """Encode tất cả chunks và build FAISS / numpy index."""
        import numpy as np

        self._chunks = chunks
        self._chunk_ids = [c["chunk_id"] for c in chunks]
        self._chunk_to_doc = {c["chunk_id"]: c["doc_id"] for c in chunks}

        texts = [c["chunk_text"] for c in chunks]

        print(f"[ChunkIndex] Encoding {len(texts)} chunks...")

        if not self._em.use_sbert and hasattr(self._em._embedder, "fit"):
            self._em._embedder.fit(texts)

        if self._em.use_sbert:
            embeddings = self._em._embedder.encode(texts, show_progress=True)
        else:
            embeddings = self._em._embedder.encode(texts)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self._em._doc_embeddings = embeddings
        self._em._doc_ids = self._chunk_ids

        print(f"[ChunkIndex] Index shape: {embeddings.shape}")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict]:
        for c in self._chunks:
            if c["chunk_id"] == chunk_id:
                return c
        return None

    def get_doc_id_for_chunk(self, chunk_id: str) -> Optional[str]:
        return self._chunk_to_doc.get(chunk_id)


if __name__ == "__main__":
    doc = {
        "id": "test_001",
        "title": "Xung đột Nga-Ukraine leo thang",
        "content": (
            "Putin tuyên bố tiếp tục chiến dịch quân sự tại Ukraine. "
            "Ông khẳng định Nga sẽ không dừng lại cho đến khi đạt mục tiêu chiến lược. "
            "Zelensky kêu gọi NATO tăng cường viện trợ vũ khí. "
            "Liên Hợp Quốc bày tỏ lo ngại về tình hình nhân đạo tại Donetsk. "
            "WHO cảnh báo về nguy cơ dịch bệnh do xung đột vũ trang. "
            "Kinh tế Nga chịu ảnh hưởng nặng từ các lệnh trừng phạt của EU và Mỹ. "
            "Biden tuyên bố Mỹ sẽ tiếp tục ủng hộ Ukraine về quân sự và tài chính."
        ),
        "date": "2024-01-15",
        "source": "VnExpress",
        "category": "thế giới",
        "url": "https://vnexpress.net/001",
    }

    chunks = chunk_document(doc, strategy="sentence_window", max_chars=200, overlap=1)
    print(f"Document chia thành {len(chunks)} chunks:\n")
    for c in chunks:
        print(f"  [{c['chunk_index']}] ({len(c['chunk_text'])} chars)")
        print(f"       {c['chunk_text'][:100]}...")
        print()
